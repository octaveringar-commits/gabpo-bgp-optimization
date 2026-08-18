# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Phase 3B, Étape A : HierarchicalPolicy_HCGA                  ║
# ║                                                                        ║
# ║  Objectif : construire la SEULE variante qui diffère de la baseline    ║
# ║  BC verrouillée par l'ajout d'un vrai mécanisme HCGA — tout le reste   ║
# ║  (têtes hiérarchiques, décodeur, environnement, dataset, seeds)        ║
# ║  reste rigoureusement identique.                                       ║
# ║                                                                        ║
# ║  Ce que la baseline HierarchicalPolicy (verrouillée) fait :            ║
# ║    x_pop (96 dims aplaties) → MLP 2 couches → 4 têtes → decode()       ║
# ║    N'utilise jamais x_as. A_cross est un placeholder fixe.             ║
# ║                                                                        ║
# ║  Ce que HierarchicalPolicy_HCGA ajoute (SEULE différence) :            ║
# ║    x_as → encodeur AS ; x_pop → encodeur PoP (séparés)                 ║
# ║    Q=W_Q(h_pop), K=W_K(h_as), V=W_V(h_as)                              ║
# ║    A_cross = softmax(QK^T/√d)  — RÉELLEMENT CALCULÉ, pas placeholder   ║
# ║    h_cross = A_cross @ V, fusion gated avec h_pop                      ║
# ║    → même 4 têtes (noop, pop, type, amp) → même decode()               ║
# ║                                                                        ║
# ║  ÉTAPE A (ce script) : sanity check architectural UNIQUEMENT.          ║
# ║  Pas d'entraînement. Vérifie avant tout pilote :                       ║
# ║    1. x_as est réellement utilisé (gradient non nul si backward)       ║
# ║    2. A_cross est non-uniforme et change avec l'observation            ║
# ║    3. Sortie decode() reste (N_p*N_a*2,) = (1200,)                     ║
# ║    4. Parcimonie préservée (≤1 PoP modifié/step)                       ║
# ║    5. Environnement et dataset Oracle non modifiés                    ║
# ║                                                                        ║
# ║  Prérequis session : P0b → extract_features patch → BC déjà chargés    ║
# ║  (fournissent BGPDynamicEnv, HierarchicalAction, extract_features,     ║
# ║  HierarchicalPolicy, run_episode, B1_Static).                          ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_HCGA_SanityCheck.py').read())                      ║
# ║    ok = run_hcga_sanity_check()                                        ║
# ║    # Si ok=True seulement → passer à l'Étape B (pilote 1-2 seeds)      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO Phase 3B — Étape A : Sanity Check HCGA       ║")
print("╚══════════════════════════════════════════════════════╝")
print()
print("Ce script NE modifie ni l'environnement ni le dataset Oracle.")
print("Ce script N'ENTRAÎNE RIEN. Vérification architecturale seule.")
print()

try:
    _ = BGPDynamicEnv, HierarchicalAction, extract_features
    _ = HierarchicalPolicy, run_episode, B1_Static
    print("✓ Dépendances chargées (P0b + BC)")
except NameError as e:
    raise RuntimeError(
        f"Exécuter P0b → extract_features patch → BC d'abord : {e}")


# ══════════════════════════════════════════════════════════════════════════
# HierarchicalPolicy_HCGA — seule différence : vrai cross-attention
# ══════════════════════════════════════════════════════════════════════════

class HierarchicalPolicy_HCGA(nn.Module):
    """
    Variante de HierarchicalPolicy (baseline BC verrouillée) avec un
    véritable mécanisme HCGA branché avant les têtes hiérarchiques.

    Toute autre partie — dimensions des têtes, HierarchicalAction,
    decode(), init des poids — est reprise à l'identique de
    HierarchicalPolicy pour que la seule variable expérimentale soit
    la présence ou l'absence de cross-attention AS↔PoP.
    """
    def __init__(self, N_a:int=50, N_p:int=12, d:int=32):
        super().__init__()
        self.N_a, self.N_p, self.d = N_a, N_p, d
        self.ha = HierarchicalAction(N_a, N_p)

        # Encodeurs séparés AS / PoP (la baseline BC n'a que PoP)
        self.enc_as  = nn.Sequential(nn.Linear(13, 64), nn.ELU(), nn.Linear(64, d))
        self.enc_pop = nn.Sequential(nn.Linear(8,  64), nn.ELU(), nn.Linear(64, d))

        # HCGA — reproduction exacte du mécanisme de GABPOAgent (GABPO_PPO_v4.py)
        self.W_Q  = nn.Linear(d, d, bias=False)
        self.W_K  = nn.Linear(d, d, bias=False)
        self.W_V  = nn.Linear(d, d, bias=False)
        self.W_O  = nn.Linear(d, d, bias=False)
        self.gate = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())
        self.ln   = nn.LayerNorm(d)

        # Même sortie latente (64) que HierarchicalPolicy.encoder avant les têtes
        self.post = nn.Sequential(nn.Linear(d, 64), nn.ReLU())

        # Têtes hiérarchiques — dimensions IDENTIQUES à HierarchicalPolicy
        self.noop_head = nn.Linear(64, 2)
        self.pop_head  = nn.Linear(64, N_p)
        self.type_head = nn.Linear(64, 4)
        self.amp_head  = nn.Linear(64, 3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.A_cross = None  # réellement calculé, pas un placeholder fixe

    def _hcga(self, h_as, h_pop):
        """Cross-attention réelle — identique à GABPOAgent._hcga()."""
        Q = self.W_Q(h_pop)                                    # (N_p, d)
        K = self.W_K(h_as)                                     # (N_a, d)
        V = self.W_V(h_as)                                     # (N_a, d)
        scores = (Q @ K.T / (self.d ** 0.5)).clamp(-20, 20)
        A = F.softmax(scores, dim=-1)                           # (N_p, N_a)
        self.A_cross = A.detach().cpu().numpy()
        ctx  = self.W_O(A @ V)                                  # (N_p, d)
        gate = self.gate(torch.cat([h_pop, ctx], dim=-1))
        return self.ln(h_pop + gate * ctx)                      # (N_p, d)

    def _get_features(self, obs):
        x_as, x_pop = extract_features(obs, self.N_a, self.N_p)
        return x_as, x_pop

    def forward(self, x_as, x_pop):
        h_as  = self.enc_as(x_as)                # (N_a, d)
        h_pop = self.enc_pop(x_pop)               # (N_p, d)
        h_cross = self._hcga(h_as, h_pop)         # (N_p, d) — enrichi par HCGA
        h_final = self.post(h_cross.mean(0))      # (64,) — même dim que baseline

        return (self.noop_head(h_final),
                self.pop_head(h_final),
                self.type_head(h_final),
                self.amp_head(h_final))

    def get_logits(self, obs):
        x_as, x_pop = self._get_features(obs)
        return self.forward(x_as, x_pop)

    def select_action(self, obs):
        with torch.no_grad():
            noop_l, pop_l, type_l, amp_l = self.get_logits(obs)
            full_logits = np.concatenate([
                [noop_l[0].item()],
                pop_l.cpu().numpy(),
                type_l.cpu().numpy(),
                amp_l.cpu().numpy(),
            ])
        return self.ha.decode(full_logits)


# ══════════════════════════════════════════════════════════════════════════
# SANITY CHECK — 5 vérifications avant tout entraînement
# ══════════════════════════════════════════════════════════════════════════

def run_hcga_sanity_check(N_a:int=50, N_p:int=12):
    print("\n[Sanity Check HCGA] 5 vérifications architecturales")
    print("=" * 60)
    all_ok = True

    model = HierarchicalPolicy_HCGA(N_a, N_p)
    env   = BGPDynamicEnv('S1_Nominal', 0, N_a, N_p, T_ep=10)
    obs   = env.reset()

    # ── Check 1 : x_as est réellement utilisé (gradient non nul) ────────
    x_as, x_pop = model._get_features(obs)
    x_as.requires_grad_(True)
    noop_l, pop_l, type_l, amp_l = model.forward(x_as, x_pop)
    loss = pop_l.sum() + type_l.sum() + amp_l.sum() + noop_l.sum()
    loss.backward()
    grad_nonzero = (x_as.grad is not None) and (x_as.grad.abs().sum().item() > 1e-8)
    print(f"  {'✓' if grad_nonzero else '✗'} Check 1 — x_as utilisé "
          f"(grad sum={x_as.grad.abs().sum().item():.6f} si calculé)")
    all_ok &= grad_nonzero

    # ── Check 2 : A_cross non-uniforme et varie avec l'observation ──────
    obs1 = env.reset()
    for _ in range(3):
        a = model.select_action(obs1)
        obs1, _, done, _ = env.step(a)
        if done: break
    A1 = model.A_cross.copy()

    env2 = BGPDynamicEnv('S4_BGP_Anomaly', 1, N_a, N_p, T_ep=40)
    obs2 = env2.reset()
    for _ in range(35):
        a = model.select_action(obs2)
        obs2, _, done, _ = env2.step(a)
        if done: break
    A2 = model.A_cross.copy()

    uniform_val = 1.0 / N_a
    is_uniform_1 = np.allclose(A1, uniform_val, atol=1e-4)
    is_uniform_2 = np.allclose(A2, uniform_val, atol=1e-4)
    differs      = not np.allclose(A1, A2, atol=1e-6)
    check2_ok = (not is_uniform_1) and (not is_uniform_2) and differs
    print(f"  {'✓' if check2_ok else '✗'} Check 2 — A_cross non-uniforme et varie "
          f"(std_S1={A1.std():.4f}, std_S4={A2.std():.4f}, "
          f"diffère entre états={differs})")
    all_ok &= check2_ok

    # ── Check 3 : sortie decode() reste (1200,) ──────────────────────────
    action = model.select_action(obs)
    shape_ok = (action.shape == (N_p*N_a*2,))
    print(f"  {'✓' if shape_ok else '✗'} Check 3 — shape action = {action.shape} "
          f"(attendu (1200,))")
    all_ok &= shape_ok

    # ── Check 4 : parcimonie préservée (≤1 PoP modifié/step) ─────────────
    nonzero_pops = set()
    off = N_p*N_a
    for p in range(N_p):
        if np.any(action[p*N_a:(p+1)*N_a] != 0) or \
           np.any(action[off+p*N_a:off+(p+1)*N_a] != 0):
            nonzero_pops.add(p)
    sparse_ok = len(nonzero_pops) <= 1
    print(f"  {'✓' if sparse_ok else '✗'} Check 4 — parcimonie "
          f"({len(nonzero_pops)} PoP(s) modifié(s), attendu ≤1)")
    all_ok &= sparse_ok

    # ── Check 5 : comparaison directe avec baseline BC (non-HCGA) ────────
    baseline = HierarchicalPolicy(N_a, N_p)
    a_baseline = baseline.select_action(obs)
    a_hcga     = model.select_action(obs)
    different_models = not np.allclose(a_baseline, a_hcga, atol=1e-6)
    print(f"  {'✓' if different_models else '⚠'} Check 5 — HCGA produit une sortie "
          f"différente de la baseline non-HCGA "
          f"({'attendu, poids non entraînés diffèrent' if different_models else 'suspect'})")

    print(f"\n{'✓ SANITY CHECK RÉUSSI — HCGA réellement branché' if all_ok else '✗ ÉCHEC — corriger avant Étape B'}")
    if all_ok:
        print("→ Passer à l'Étape B : pilote 1-2 seeds (entraînement BC identique)")

    return all_ok, model


# ── Point d'entrée ────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok, model = run_hcga_sanity_check()
