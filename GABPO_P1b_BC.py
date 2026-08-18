# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO Phase P1.2 — Behavioral Cloning → PPO fine-tuning              ║
# ║                                                                        ║
# ║  Stratégie (avis expert) :                                             ║
# ║    1. Générer 5 000–20 000 (état, action_Oracle) pairs                 ║
# ║    2. Entraîner le policy network en supervised learning               ║
# ║    3. Mesurer si L_BC ≈ L_Oracle (capacité architecturale validée)     ║
# ║    4. PPO fine-tuning si BC réussit                                    ║
# ║                                                                        ║
# ║  Action hiérarchique (1200 → ~15 dims) :                               ║
# ║    π(a|s) = π(PoP|s) · π(Type|PoP,s) · π(Amplitude|PoP,Type,s)       ║
# ║    PoP : 12 choix                                                      ║
# ║    Type : {LOCAL_PREF, PREPEND}                                        ║
# ║    Amplitude : {0.25, 0.5, 1.0}                                        ║
# ║    + NO-OP                                                             ║
# ║                                                                        ║
# ║  Curriculum :                                                          ║
# ║    Phase A : S3 uniquement                                             ║
# ║    Phase B : S4 uniquement                                             ║
# ║    Phase C : S3 + S4                                                   ║
# ║    Phase D : S1 + S2 + S3 + S4                                        ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_P1b_BC.py').read())                                ║
# ║    results = run_bc(quick=True)   # ~5-10 min                          ║
# ║    results = run_bc(quick=False)  # ~30-60 min GPU                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scipy import stats
import json, time, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('results_bc', exist_ok=True)
os.makedirs('checkpoints_bc', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"╔══════════════════════════════════════════════════════╗")
print(f"║  GABPO Phase P1.2 — Behavioral Cloning → PPO       ║")
print(f"╚══════════════════════════════════════════════════════╝")
print(f"Device: {DEVICE}")
print()

# ── Vérification P0b chargé ───────────────────────────────────────────────
try:
    _ = BGPDynamicEnv
    _ = OracleAgent
    _ = B1_Static
    _ = run_episode
    _ = extract_features
    print("✓ P0b chargé (BGPDynamicEnv, OracleAgent, B1_Static)")
except NameError as e:
    raise RuntimeError(
        f"Exécuter GABPO_P0b_DynamicEnv.py d'abord : {e}")


# ══════════════════════════════════════════════════════════════════════════
# 1. ACTION HIÉRARCHIQUE — 1200 → ~15 dimensions
# ══════════════════════════════════════════════════════════════════════════

class HierarchicalAction:
    """
    Décomposition de l'espace d'action BGP :
      Niveau 0 : NO-OP (ne rien faire)
      Niveau 1 : PoP à modifier (0..N_p-1)
      Niveau 2 : Type {0=LP+, 1=LP-, 2=Prep+, 3=Prep-}
      Niveau 3 : Amplitude {0=0.25, 1=0.5, 2=1.0}

    Taille totale : 1 (noop) + N_p + 4 + 3 = N_p + 8
    Pour N_p=12 : 20 dimensions (vs 1200 avant)
    """
    AMPLITUDES = [0.25, 0.5, 1.0]
    TYPES      = ['LP+', 'LP-', 'Prep+', 'Prep-']

    def __init__(self, N_a:int=50, N_p:int=12):
        self.N_a, self.N_p = N_a, N_p
        # Dimensions : [noop(1), pop_logits(N_p), type_logits(4), amp_logits(3)]
        self.dim = 1 + N_p + 4 + 3

    def decode(self, logits:np.ndarray) -> np.ndarray:
        """
        Convertit logits hiérarchiques en action BGP (N_p*N_a*2,).
        logits[0]       = score NO-OP
        logits[1:N_p+1] = scores PoP
        logits[N_p+1:N_p+5] = scores type {LP+,LP-,Prep+,Prep-}
        logits[N_p+5:N_p+8] = scores amplitude
        """
        N_p, N_a = self.N_p, self.N_a
        action = np.zeros(N_p * N_a * 2, dtype=np.float32)

        # NO-OP decision
        if logits[0] > logits[1:N_p+1].max():
            return action   # NO-OP

        # Choisir PoP
        pop_scores = logits[1:N_p+1]
        pop_idx    = int(np.argmax(pop_scores))

        # Choisir type
        type_scores = logits[N_p+1:N_p+5]
        type_idx    = int(np.argmax(type_scores))
        action_type = self.TYPES[type_idx]

        # Choisir amplitude
        amp_scores = logits[N_p+5:N_p+8]
        amp_idx    = int(np.argmax(amp_scores))
        amplitude  = self.AMPLITUDES[amp_idx]

        # Construire l'action BGP
        off = N_p * N_a
        for i in range(N_a):
            if action_type == 'LP+':
                action[off + pop_idx*N_a + i] = amplitude
            elif action_type == 'LP-':
                action[off + pop_idx*N_a + i] = -amplitude
            elif action_type == 'Prep+':
                action[pop_idx*N_a + i] = amplitude
            elif action_type == 'Prep-':
                action[pop_idx*N_a + i] = -amplitude

        return action

    def encode_oracle(self, oracle_action:np.ndarray) -> dict:
        """
        Inverse : convertit une action Oracle (1200,) en labels hiérarchiques.
        Retourne {noop, pop, type, amp} pour supervised learning.
        """
        N_p, N_a = self.N_p, self.N_a
        off = N_p * N_a

        prepend   = oracle_action[:N_p*N_a].reshape(N_p, N_a)
        localpref = oracle_action[N_p*N_a:].reshape(N_p, N_a)

        # Détecter NO-OP
        if np.abs(oracle_action).max() < 0.05:
            return {'noop':1, 'pop':0, 'type':0, 'amp':0}

        # Trouver le PoP dominant
        lp_magnitude   = np.abs(localpref).mean(axis=1)
        prep_magnitude = np.abs(prepend).mean(axis=1)
        combined       = lp_magnitude + prep_magnitude

        if combined.max() < 0.01:
            return {'noop':1, 'pop':0, 'type':0, 'amp':0}

        pop_idx = int(np.argmax(combined))

        # Déterminer type et amplitude
        lp_val   = localpref[pop_idx].mean()
        prep_val = prepend[pop_idx].mean()

        if abs(lp_val) >= abs(prep_val):
            type_idx = 0 if lp_val > 0 else 1
            amplitude = abs(lp_val)
        else:
            type_idx = 2 if prep_val > 0 else 3
            amplitude = abs(prep_val)

        # Quantifier l'amplitude
        diffs = [abs(amplitude - a) for a in self.AMPLITUDES]
        amp_idx = int(np.argmin(diffs))

        return {'noop':0, 'pop':pop_idx, 'type':type_idx, 'amp':amp_idx}


# ══════════════════════════════════════════════════════════════════════════
# 2. RÉSEAU DE POLITIQUE HIÉRARCHIQUE
# ══════════════════════════════════════════════════════════════════════════

class HierarchicalPolicy(nn.Module):
    """
    Politique hiérarchique sur espace action réduit (N_p+8 dims).
    Entrée : features PoP (N_p×8) aplaties = N_p*8 = 96 dims.
    Sortie : logits hiérarchiques pour BC et PPO.
    """
    def __init__(self, N_a:int=50, N_p:int=12):
        super().__init__()
        self.N_a, self.N_p = N_a, N_p
        self.ha = HierarchicalAction(N_a, N_p)
        in_dim  = N_p * 8   # features PoP aplaties

        # Encodeur partagé
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),    nn.ReLU())

        # Têtes hiérarchiques
        self.noop_head = nn.Linear(64, 2)       # NO-OP vs Act
        self.pop_head  = nn.Linear(64, N_p)     # quel PoP
        self.type_head = nn.Linear(64, 4)       # quel type
        self.amp_head  = nn.Linear(64, 3)       # quelle amplitude

        # Init stable
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.5)
                nn.init.zeros_(m.bias)

        self.A_cross = np.ones((N_p, N_a)) / N_a

    def _get_features(self, obs):
        _, x_pop = extract_features(obs, self.N_a, self.N_p)
        return x_pop.flatten()

    def forward(self, feat):
        """feat : (N_p*8,) tensor"""
        h = self.encoder(feat)
        return (self.noop_head(h),
                self.pop_head(h),
                self.type_head(h),
                self.amp_head(h))

    def select_action(self, obs):
        feat = self._get_features(obs)
        with torch.no_grad():
            noop_l, pop_l, type_l, amp_l = self.forward(feat.to(DEVICE))
            logits = torch.cat([
                noop_l[1:2],   # score "act" vs score "noop"
                pop_l, type_l, amp_l
            ]).cpu().numpy()
            # Corriger index : [noop_score, act_score, pop×12, type×4, amp×3]
            full_logits = np.concatenate([
                [noop_l[0].item()],
                pop_l.cpu().numpy(),
                type_l.cpu().numpy(),
                amp_l.cpu().numpy(),
            ])
        return self.ha.decode(full_logits)


# ══════════════════════════════════════════════════════════════════════════
# 3. GÉNÉRATION DU DATASET ORACLE
# ══════════════════════════════════════════════════════════════════════════

def generate_oracle_dataset(scenarios:list, n_episodes:int=50,
                             N_a:int=50, N_p:int=12,
                             T_ep:int=72) -> list:
    """
    Génère (state_features, oracle_action, labels_hierarchiques).
    Exécute l'Oracle sur n_episodes épisodes par scénario.
    """
    print(f"\n[Dataset Oracle] {len(scenarios)} scénarios × {n_episodes} épisodes × {T_ep} steps")
    ha      = HierarchicalAction(N_a, N_p)
    dataset = []
    total   = len(scenarios) * n_episodes

    for si, scen in enumerate(scenarios):
        for ep in range(n_episodes):
            seed    = si * 1000 + ep
            env     = BGPDynamicEnv(scen, seed, N_a, N_p, T_ep)
            oracle  = OracleAgent(N_a, N_p, env, n_candidates=50)
            obs     = env.reset()

            for _ in range(T_ep):
                # Features
                _, x_pop  = extract_features(obs, N_a, N_p)
                feat      = x_pop.flatten().numpy()

                # Action Oracle
                a_oracle  = oracle.select_action(obs)

                # Labels hiérarchiques
                labels    = ha.encode_oracle(a_oracle)

                dataset.append({
                    'feat':   feat,
                    'action': a_oracle,
                    'noop':   labels['noop'],
                    'pop':    labels['pop'],
                    'type':   labels['type'],
                    'amp':    labels['amp'],
                    'scenario': scen,
                })

                obs, _, done, _ = env.step(a_oracle)
                if done: break

        print(f"  {(si+1)*n_episodes}/{total} épisodes ({scen})")

    print(f"  ✓ Dataset : {len(dataset)} samples")

    # Statistiques
    noops = sum(1 for d in dataset if d['noop']==1)
    print(f"  NO-OP rate : {noops/len(dataset):.1%}")
    pops  = [d['pop'] for d in dataset if d['noop']==0]
    if pops:
        unique, counts = np.unique(pops, return_counts=True)
        top3 = unique[np.argsort(-counts)[:3]]
        print(f"  Top-3 PoPs modifiés : {top3.tolist()}")

    return dataset


# ══════════════════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT BEHAVIORAL CLONING
# ══════════════════════════════════════════════════════════════════════════

def train_bc(policy:HierarchicalPolicy, dataset:list,
             n_epochs:int=50, batch_size:int=128,
             lr:float=1e-3) -> dict:
    """
    Supervised learning sur les actions Oracle.
    Minimise la cross-entropy sur chaque tête hiérarchique.
    """
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # Préparer les tensors
    feats    = torch.FloatTensor(np.array([d['feat'] for d in dataset]))
    y_noop   = torch.LongTensor([d['noop'] for d in dataset])
    y_pop    = torch.LongTensor([d['pop']  for d in dataset])
    y_type   = torch.LongTensor([d['type'] for d in dataset])
    y_amp    = torch.LongTensor([d['amp']  for d in dataset])
    n        = len(dataset)

    logs = []
    policy.train()

    for epoch in range(n_epochs):
        idx     = torch.randperm(n)
        ep_loss = {'total':[], 'noop':[], 'pop':[], 'type':[], 'amp':[]}

        for start in range(0, n, batch_size):
            b = idx[start:start+batch_size]
            f = feats[b].to(DEVICE)

            noop_l, pop_l, type_l, amp_l = policy.forward(f)

            # Losses séparées pour chaque niveau de la hiérarchie
            l_noop = F.cross_entropy(noop_l, y_noop[b].to(DEVICE))
            l_pop  = F.cross_entropy(pop_l,  y_pop[b].to(DEVICE))
            l_type = F.cross_entropy(type_l, y_type[b].to(DEVICE))
            l_amp  = F.cross_entropy(amp_l,  y_amp[b].to(DEVICE))

            # Pondérer : noop est le plus important
            loss = 2.0*l_noop + 1.5*l_pop + 1.0*l_type + 0.5*l_amp

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            ep_loss['total'].append(loss.item())
            ep_loss['noop'].append(l_noop.item())
            ep_loss['pop'].append(l_pop.item())
            ep_loss['type'].append(l_type.item())
            ep_loss['amp'].append(l_amp.item())

        # Accuracy sur tout le dataset
        policy.eval()
        with torch.no_grad():
            noop_l, pop_l, type_l, amp_l = policy.forward(feats.to(DEVICE))
            acc_noop = (noop_l.argmax(1).cpu()==y_noop).float().mean().item()
            acc_pop  = (pop_l.argmax(1).cpu()==y_pop).float().mean().item()
            acc_type = (type_l.argmax(1).cpu()==y_type).float().mean().item()
            acc_amp  = (amp_l.argmax(1).cpu()==y_amp).float().mean().item()
        policy.train()

        log = {
            'epoch':    epoch+1,
            'loss':     float(np.mean(ep_loss['total'])),
            'loss_noop':float(np.mean(ep_loss['noop'])),
            'loss_pop': float(np.mean(ep_loss['pop'])),
            'acc_noop': acc_noop,
            'acc_pop':  acc_pop,
            'acc_type': acc_type,
            'acc_amp':  acc_amp,
        }
        logs.append(log)

        if (epoch+1) % 10 == 0:
            print(f"  Ep {epoch+1:3d} | "
                  f"loss={log['loss']:.4f} | "
                  f"acc_noop={acc_noop:.2%} "
                  f"acc_pop={acc_pop:.2%} "
                  f"acc_type={acc_type:.2%} "
                  f"acc_amp={acc_amp:.2%}")

    return logs


# ══════════════════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def run_bc(quick:bool=True, N_a:int=50, N_p:int=12):
    """
    Phase P1.2 — Behavioral Cloning → PPO.

    quick=True  : 20 épisodes × 2 scénarios, 30 epochs BC (~5-10 min)
    quick=False : 100 épisodes × 4 scénarios, 100 epochs BC (~30-60 min)
    """
    n_episodes = 20  if quick else 100
    bc_epochs  = 30  if quick else 100

    # Curriculum : Phase A (S3) → Phase B (S4) → Phase C (S3+S4)
    curriculum = {
        'A': ['S3_PoP_Failure'],
        'B': ['S4_BGP_Anomaly'],
        'C': ['S3_PoP_Failure','S4_BGP_Anomaly'],
    }
    if not quick:
        curriculum['D'] = ['S1_Nominal','S2_FlashCrowd',
                           'S3_PoP_Failure','S4_BGP_Anomaly']

    print(f"\n[{'PILOTE' if quick else 'COMPLET'}] "
          f"{n_episodes} épisodes/scénario × {bc_epochs} epochs BC")
    print(f"Curriculum : {list(curriculum.keys())}")

    policy = HierarchicalPolicy(N_a, N_p).to(DEVICE)
    b1     = B1_Static(N_a, N_p)
    all_results = {}

    for phase, scenarios in curriculum.items():
        print(f"\n{'═'*60}")
        print(f"Phase {phase} — Scénarios : {scenarios}")
        print(f"{'═'*60}")

        # 1. Générer dataset Oracle
        dataset = generate_oracle_dataset(
            scenarios, n_episodes, N_a, N_p, T_ep=72)

        if len(dataset) < 100:
            print(f"  ⚠ Dataset trop petit ({len(dataset)}) — skip")
            continue

        # 2. Behavioral Cloning
        print(f"\n[BC] Entraînement {bc_epochs} epochs sur {len(dataset)} samples...")
        bc_logs = train_bc(policy, dataset, bc_epochs, batch_size=128)

        # 3. Évaluation BC vs Oracle vs B1
        print(f"\n[Évaluation Phase {phase}]")
        phase_results = {}
        for scen in scenarios:
            oracle_Ls, bc_Ls, b1_Ls = [], [], []
            for seed in range(5):
                env_ref = BGPDynamicEnv(scen, seed, N_a, N_p, 288)
                oracle  = OracleAgent(N_a, N_p, env_ref, n_candidates=50)
                r_or = run_episode(oracle,  scen, seed, N_a, N_p, 288)
                r_bc = run_episode(policy,  scen, seed, N_a, N_p, 288)
                r_b1 = run_episode(b1,      scen, seed, N_a, N_p, 288)
                oracle_Ls.append(r_or['L_med'])
                bc_Ls.append(r_bc['L_med'])
                b1_Ls.append(r_b1['L_med'])

            or_m = float(np.mean(oracle_Ls))
            bc_m = float(np.mean(bc_Ls))
            b1_m = float(np.mean(b1_Ls))
            gap  = bc_m - or_m   # gap BC-Oracle (voulons <10ms)
            vs_b1= (b1_m - bc_m) / b1_m * 100

            sym = '✅' if gap < 10 else ('⚠' if gap < 30 else '❌')
            print(f"  {sym} {scen}:")
            print(f"     Oracle={or_m:.1f}ms  BC={bc_m:.1f}ms  "
                  f"B1={b1_m:.1f}ms")
            print(f"     Gap BC-Oracle={gap:+.1f}ms  "
                  f"BC vs B1={vs_b1:+.1f}%")

            phase_results[scen] = {
                'oracle_L': or_m, 'bc_L': bc_m, 'b1_L': b1_m,
                'gap_bc_oracle': gap, 'bc_vs_b1_pct': vs_b1,
            }

        all_results[f'Phase_{phase}'] = phase_results

        # Sauvegarder checkpoint de phase
        torch.save(policy.state_dict(),
                   f'checkpoints_bc/bc_phase_{phase}.pt')

    # ── Verdict final ─────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"VERDICT BC — Capacité architecturale")
    print(f"{'═'*60}")

    final_s3 = all_results.get('Phase_C',{}).get('S3_PoP_Failure', {})
    final_s4 = all_results.get('Phase_C',{}).get('S4_BGP_Anomaly', {})

    if final_s3 and final_s4:
        gap_s3 = final_s3.get('gap_bc_oracle', 999)
        gap_s4 = final_s4.get('gap_bc_oracle', 999)

        print(f"  S3 — Gap BC-Oracle : {gap_s3:+.1f}ms")
        print(f"  S4 — Gap BC-Oracle : {gap_s4:+.1f}ms")

        if gap_s3 < 10 and gap_s4 < 10:
            verdict = 'ARCHITECTURE_CAPABLE'
            next_action = '→ Passer à PPO fine-tuning avec warm-start BC'
        elif gap_s3 < 30 or gap_s4 < 30:
            verdict = 'PARTIAL_CAPACITY'
            next_action = '→ Augmenter dataset (20k samples) + epochs (200)'
        else:
            verdict = 'ARCHITECTURE_INSUFFICIENT'
            next_action = '→ Revoir features ou architecture encodeur'

        print(f"\n  Verdict : {verdict}")
        print(f"  {next_action}")

    else:
        print("  Phase C pas encore exécutée (quick=True)")
        print("  → Lancer run_bc(quick=False) pour le verdict complet")
        verdict = 'INCOMPLETE'

    # Export
    with open('results_bc/GABPO_BC_results.json','w') as f:
        json.dump({
            'timestamp': time.strftime('%Y%m%d_%H%M'),
            'quick': quick, 'n_episodes': n_episodes,
            'bc_epochs': bc_epochs,
            'results': all_results,
            'verdict': verdict,
        }, f, indent=2, default=str)

    print(f"\n✓ Sauvegardé : results_bc/GABPO_BC_results.json")
    return all_results


# ── Point d'entrée ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Lancement BC pilote (quick=True)...")
    print("Exécuter GABPO_P0b_DynamicEnv.py d'abord si nécessaire.")
    results = run_bc(quick=True)
