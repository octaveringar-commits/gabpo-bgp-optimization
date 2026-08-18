# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Ablation Baselines B3-B6, réécrites pour comparabilité stricte ║
# ║                                                                        ║
# ║  Contrat commun (identique pour B3, B4, B5, B6, GABPO) :               ║
# ║    Observation → Encodeur spécifique → représentation latente          ║
# ║      → 4 têtes (noop, pop, type, amp) → HierarchicalAction.decode()   ║
# ║      → action BGP (N_p*N_a*2,) → même env, même reward, même métriques║
# ║                                                                        ║
# ║  Ce qui NE change JAMAIS entre les 5 modèles :                         ║
# ║    - HierarchicalAction.decode() (import direct depuis GABPO_P1b_BC)   ║
# ║    - dimensions des têtes : noop(2), pop(N_p), type(4), amp(3)         ║
# ║    - environnement BGPDynamicEnv, protocole d'entraînement BC          ║
# ║    - protocole d'évaluation (seeds, T_ep, métriques, tests stat.)     ║
# ║                                                                        ║
# ║  Ce qui change (UNIQUEMENT l'encodeur) :                               ║
# ║    B3 BiLSTM    : LSTM bidirectionnel — aucune structure de graphe     ║
# ║    B4 GCN       : vraie convolution graphique (matrice d'adjacence)    ║
# ║    B5 GraphSAGE : vraie agrégation inductive (sample & aggregate)      ║
# ║    B6 GAT       : vraie attention intra-graphe sur G_AS (self-attn)   ║
# ║    GABPO        : ST-GNN + HCGA (cross-graph attention) — référence   ║
# ║                                                                        ║
# ║  Prérequis session : GABPO_P0b_DynamicEnv.py puis GABPO_P1b_BC.py      ║
# ║  déjà exécutés (fournissent extract_features, HierarchicalAction,      ║
# ║  BGPDynamicEnv, run_episode, train_bc, generate_oracle_dataset).       ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_Ablation_B3B6.py').read())                         ║
# ║    df = run_sanity_check()          # vérifie que decode() reste       ║
# ║                                       identique pour les 5 modèles     ║
# ║    df = run_ablation(quick=True)    # entraînement pilote 1 seed       ║
# ║    df = run_ablation(quick=False)   # 10 seeds complets                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('results_ablation',     exist_ok=True)
os.makedirs('checkpoints_ablation', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Ablation B3-B6 (encodeurs réels, décodeur   ║")
print("║  hiérarchique partagé et inchangé)                   ║")
print("╚══════════════════════════════════════════════════════╝")

# ── Vérification des dépendances ──────────────────────────────────────────
try:
    _ = BGPDynamicEnv, run_episode, extract_features
    _ = HierarchicalAction, generate_oracle_dataset, train_bc
    print("✓ P0b + BC chargés (BGPDynamicEnv, HierarchicalAction, train_bc)")
except NameError as e:
    raise RuntimeError(
        f"Exécuter GABPO_P0b_DynamicEnv.py puis GABPO_P1b_BC.py d'abord : {e}")


# ══════════════════════════════════════════════════════════════════════════
# 1. CLASSE DE BASE — contrat commun à B3, B4, B5, B6
# ══════════════════════════════════════════════════════════════════════════

class AblationPolicyBase(nn.Module):
    """
    Squelette commun : toute sous-classe implémente uniquement `encode()`,
    qui doit retourner un vecteur latent (d,). Les 4 têtes et le décodeur
    hiérarchique sont fixés ici, identiques pour tous les modèles —
    c'est ce qui rend l'ablation défendable : seul l'encodeur varie.
    """
    def __init__(self, N_a:int=50, N_p:int=12, d_latent:int=64):
        super().__init__()
        self.N_a, self.N_p, self.d_latent = N_a, N_p, d_latent
        self.ha = HierarchicalAction(N_a, N_p)

        # Têtes hiérarchiques — MÊMES dimensions que HierarchicalPolicy
        self.noop_head = nn.Linear(d_latent, 2)
        self.pop_head  = nn.Linear(d_latent, N_p)
        self.type_head = nn.Linear(d_latent, 4)
        self.amp_head  = nn.Linear(d_latent, 3)

        for head in [self.noop_head, self.pop_head, self.type_head, self.amp_head]:
            nn.init.orthogonal_(head.weight, 0.5)
            nn.init.zeros_(head.bias)

        self.A_cross = np.ones((N_p, N_a)) / N_a  # placeholder XAI (pas de cross-attn ici)

    def encode(self, obs) -> torch.Tensor:
        """À implémenter par chaque sous-classe. Retourne (d_latent,)."""
        raise NotImplementedError

    def forward_from_latent(self, h):
        return (self.noop_head(h), self.pop_head(h),
                self.type_head(h), self.amp_head(h))

    def get_logits(self, obs):
        """Encode puis produit les 4 têtes — utilisé pour BC training."""
        h = self.encode(obs)
        return self.forward_from_latent(h)

    def select_action(self, obs):
        """
        Chemin identique pour B3-B6 et GABPO :
        encode → 4 têtes → concatène → HierarchicalAction.decode()
        """
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
# 2. B3 — BiLSTM (aucune structure de graphe)
# ══════════════════════════════════════════════════════════════════════════

class B3_BiLSTM_Ablation(AblationPolicyBase):
    """
    Objectif d'ablation : supprimer toute modélisation graphique.
    Traite les features PoP comme une séquence temporelle plate —
    aucune notion d'adjacence AS ni de structure de graphe.
    """
    def __init__(self, N_a=50, N_p=12, hidden=64):
        super().__init__(N_a, N_p, d_latent=hidden*2)
        in_dim = N_p * 8
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)

    def encode(self, obs):
        _, x_pop = extract_features(obs, self.N_a, self.N_p)
        feat = x_pop.flatten().unsqueeze(0).unsqueeze(0)  # (1,1,in_dim)
        out, _ = self.lstm(feat)
        return out[0, -1, :]  # (hidden*2,)


# ══════════════════════════════════════════════════════════════════════════
# 3. B4 — GCN véritable (convolution graphique avec adjacence)
# ══════════════════════════════════════════════════════════════════════════

def build_as_adjacency(N_a:int=50, seed:int=0) -> torch.Tensor:
    """
    Construit une matrice d'adjacence AS fixe et déterministe.
    Approxime la topologie CAIDA 2-hop réelle : quelques hubs fortement
    connectés (proxy Tier-1 / IXP), le reste peu connecté.
    Fixée une fois — ne change jamais pendant l'ablation, comme la vraie
    topologie G_AS ne change pas entre B3-B6 et GABPO.
    """
    rng = np.random.RandomState(seed)
    A = np.eye(N_a, dtype=np.float32)
    # Quelques hubs (proxy Tier-1) très connectés
    n_hubs = max(2, N_a // 10)
    hubs = rng.choice(N_a, n_hubs, replace=False)
    for h in hubs:
        deg = rng.randint(N_a//3, N_a//2)
        neighbors = rng.choice(N_a, deg, replace=False)
        A[h, neighbors] = 1.0
        A[neighbors, h] = 1.0
    # Connexions locales creuses pour le reste (proxy peering régional)
    for i in range(N_a):
        deg = rng.randint(2, 5)
        neighbors = rng.choice(N_a, deg, replace=False)
        A[i, neighbors] = 1.0
        A[neighbors, i] = 1.0
    # Normalisation symétrique (Kipf & Welling, GCN standard)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-8)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.FloatTensor(A_norm)


class B4_GCN_Ablation(AblationPolicyBase):
    """
    Objectif d'ablation : remplacer HCGA par une vraie convolution
    graphique. Message passing standard : H' = ReLU(A_norm @ H @ W).
    Utilise une adjacence AS fixe (build_as_adjacency) — ce que
    l'ancienne B4 (mean-pool+MLP) n'avait pas.
    """
    def __init__(self, N_a=50, N_p=12, d=64):
        super().__init__(N_a, N_p, d_latent=d)
        self.register_buffer('A_norm', build_as_adjacency(N_a, seed=0))
        self.gc1 = nn.Linear(13, 32, bias=False)
        self.gc2 = nn.Linear(32, d,  bias=False)
        self.pop_proj = nn.Linear(8, d)

    def encode(self, obs):
        x_as, x_pop = extract_features(obs, self.N_a, self.N_p)
        # Convolution graphique : agrégation par voisinage (2 couches)
        h1 = F.relu(self.A_norm @ self.gc1(x_as))      # (N_a, 32)
        h2 = F.relu(self.A_norm @ self.gc2(h1))         # (N_a, d)
        as_repr  = h2.mean(0)                            # pooling global AS
        pop_repr = F.relu(self.pop_proj(x_pop)).mean(0)  # pooling global PoP
        return as_repr + pop_repr


# ══════════════════════════════════════════════════════════════════════════
# 4. B5 — GraphSAGE véritable (agrégation inductive sample-and-aggregate)
# ══════════════════════════════════════════════════════════════════════════

class B5_GraphSAGE_Ablation(AblationPolicyBase):
    """
    Objectif d'ablation : remplacer HCGA par une vraie agrégation
    inductive (Hamilton et al., 2017). Chaque nœud agrège les features
    de ses voisins échantillonnés puis concatène avec sa propre feature,
    contrairement au simple mean-pool de l'ancienne B5.
    """
    def __init__(self, N_a=50, N_p=12, d=64, n_samples=8):
        super().__init__(N_a, N_p, d_latent=d)
        self.register_buffer('A_norm', build_as_adjacency(N_a, seed=0))
        self.n_samples = n_samples
        self.self_proj = nn.Linear(13, d//2)
        self.agg_proj  = nn.Linear(13, d//2)
        self.pop_proj  = nn.Linear(8, d)
        self.rng = np.random.RandomState(1)

    def _sample_neighbors(self, adj_row, k):
        idx = np.where(adj_row.numpy() > 0)[0]
        if len(idx) == 0:
            return np.array([], dtype=int)
        if len(idx) <= k:
            return idx
        return self.rng.choice(idx, k, replace=False)

    def encode(self, obs):
        x_as, x_pop = extract_features(obs, self.N_a, self.N_p)
        self_h = self.self_proj(x_as)  # (N_a, d/2)

        # Agrégation : pour chaque nœud, moyenne des voisins échantillonnés
        agg_h = torch.zeros_like(self_h)
        for i in range(self.N_a):
            neigh_idx = self._sample_neighbors(self.A_norm[i], self.n_samples)
            if len(neigh_idx) > 0:
                agg_h[i] = self.agg_proj(x_as[neigh_idx]).mean(0)
            else:
                agg_h[i] = self.agg_proj(x_as[i:i+1]).mean(0)

        h = torch.cat([self_h, agg_h], dim=-1)  # (N_a, d) — SAGE concat
        as_repr  = F.relu(h).mean(0)
        pop_repr = F.relu(self.pop_proj(x_pop)).mean(0)
        return as_repr + pop_repr


# ══════════════════════════════════════════════════════════════════════════
# 5. B6 — GAT véritable (attention intra-graphe, sans couplage cross-graph)
# ══════════════════════════════════════════════════════════════════════════

class B6_GAT_Ablation(AblationPolicyBase):
    """
    Objectif d'ablation : isoler l'attention intra-graphe de HCGA.
    B6 a un vrai mécanisme d'attention (comme GABPO) MAIS uniquement
    sur G_AS — jamais de couplage cross-graph avec G_PoP. C'est
    précisément ce que HCGA ajoute : la comparaison B6 vs GABPO
    répond à "l'attention seule suffit-elle, ou le couplage
    cross-graph AS-PoP apporte-t-il un gain distinct ?"
    """
    def __init__(self, N_a=50, N_p=12, d=64, n_heads=4):
        super().__init__(N_a, N_p, d_latent=d)
        self.d, self.n_heads = d, n_heads
        self.proj_in = nn.Linear(13, d)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.pop_proj = nn.Linear(8, d)

    def encode(self, obs):
        x_as, x_pop = extract_features(obs, self.N_a, self.N_p)
        h0 = self.proj_in(x_as)                          # (N_a, d)
        Q, K, V = self.W_Q(h0), self.W_K(h0), self.W_V(h0)
        scores = (Q @ K.T / (self.d ** 0.5)).clamp(-20, 20)
        A_self = F.softmax(scores, dim=-1)                # (N_a, N_a) — intra-graphe
        h_attn = A_self @ V                                # (N_a, d)
        as_repr  = h_attn.mean(0)
        pop_repr = F.relu(self.pop_proj(x_pop)).mean(0)
        # Pas de couplage cross-graph — simple somme, contrairement à HCGA
        return as_repr + pop_repr


# ══════════════════════════════════════════════════════════════════════════
# 6. SANITY CHECK — vérifier que decode() reste identique pour les 5 modèles
# ══════════════════════════════════════════════════════════════════════════

def run_sanity_check(N_a:int=50, N_p:int=12):
    """
    Vérifie que les 5 modèles produisent une action de même forme
    ET utilisent le même décodeur — condition nécessaire pour que
    l'ablation soit valide. N'entraîne rien, juste un test structurel.
    """
    print("\n[Sanity Check] Vérification du contrat commun (decode() partagé)")
    print("=" * 60)

    models = {
        'B3_BiLSTM': B3_BiLSTM_Ablation(N_a, N_p),
        'B4_GCN':    B4_GCN_Ablation(N_a, N_p),
        'B5_SAGE':   B5_GraphSAGE_Ablation(N_a, N_p),
        'B6_GAT':    B6_GAT_Ablation(N_a, N_p),
    }
    if 'HierarchicalPolicy' in globals():
        gabpo_ref = HierarchicalPolicy(N_a, N_p)
        models['GABPO_ref'] = gabpo_ref

    env = BGPDynamicEnv('S1_Nominal', 0, N_a, N_p, T_ep=5)
    obs = env.reset()

    all_ok = True
    for name, model in models.items():
        try:
            if name == 'GABPO_ref':
                action = model.select_action(obs)
            else:
                action = model.select_action(obs)
            shape_ok = (action.shape == (N_p*N_a*2,))
            # Vérifier la parcimonie (un seul PoP modifié, ou NO-OP)
            nonzero_pops = set()
            off = N_p*N_a
            for p in range(N_p):
                if np.any(action[p*N_a:(p+1)*N_a] != 0) or \
                   np.any(action[off+p*N_a:off+(p+1)*N_a] != 0):
                    nonzero_pops.add(p)
            sparse_ok = len(nonzero_pops) <= 1
            status = '✓' if (shape_ok and sparse_ok) else '✗'
            if not (shape_ok and sparse_ok):
                all_ok = False
            print(f"  {status} {name:<12} shape={action.shape} "
                  f"PoPs modifiés={len(nonzero_pops)} "
                  f"{'(NO-OP ou 1 PoP — OK)' if sparse_ok else '(ANOMALIE)'}")
        except Exception as e:
            all_ok = False
            print(f"  ✗ {name:<12} ERREUR: {type(e).__name__}: {e}")

    print(f"\n{'✓ Contrat commun respecté — ablation valide' if all_ok else '✗ PROBLÈME — ne pas lancer entraînement'}")
    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# 7. ENTRAÎNEMENT — même protocole BC pour toutes les baselines
# ══════════════════════════════════════════════════════════════════════════

def train_ablation_model(model, dataset, n_epochs=100, batch_size=128, lr=1e-3):
    """
    Identique à train_bc() mais générique sur AblationPolicyBase.
    Même cross-entropy pondérée (2.0/1.5/1.0/0.5), même curriculum.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(dataset)

    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        ep_loss = []
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start+batch_size]
            losses = []
            for i in batch_idx:
                d = dataset[i]
                noop_l, pop_l, type_l, amp_l = model.get_logits(d['obs_raw'])
                l_noop = F.cross_entropy(noop_l.unsqueeze(0),
                                          torch.LongTensor([d['noop']]))
                l_pop  = F.cross_entropy(pop_l.unsqueeze(0),
                                          torch.LongTensor([d['pop']]))
                l_type = F.cross_entropy(type_l.unsqueeze(0),
                                          torch.LongTensor([d['type']]))
                l_amp  = F.cross_entropy(amp_l.unsqueeze(0),
                                          torch.LongTensor([d['amp']]))
                loss = 2.0*l_noop + 1.5*l_pop + 1.0*l_type + 0.5*l_amp
                losses.append(loss)
            batch_loss = torch.stack(losses).mean()
            opt.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss.append(batch_loss.item())

        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | loss={np.mean(ep_loss):.4f}")

    return model


def generate_dataset_with_obs(scenarios, n_episodes=100, N_a=50, N_p=12, T_ep=72):
    """
    Variante de generate_oracle_dataset qui conserve obs_raw
    (nécessaire car chaque encodeur B3-B6 appelle extract_features
    lui-même depuis obs, pas depuis un vecteur déjà aplati).
    """
    ha = HierarchicalAction(N_a, N_p)
    dataset = []
    for si, scen in enumerate(scenarios):
        for ep in range(n_episodes):
            seed = si*1000 + ep
            env = BGPDynamicEnv(scen, seed, N_a, N_p, T_ep)
            oracle = OracleAgent(N_a, N_p, env, n_candidates=50)
            obs = env.reset()
            for _ in range(T_ep):
                a_oracle = oracle.select_action(obs)
                labels = ha.encode_oracle(a_oracle)
                dataset.append({
                    'obs_raw': dict(obs),  # copie — obs mutable sinon
                    'noop': labels['noop'], 'pop': labels['pop'],
                    'type': labels['type'], 'amp': labels['amp'],
                })
                obs, _, done, _ = env.step(a_oracle)
                if done: break
        print(f"  {(si+1)*n_episodes} épisodes ({scen})")
    return dataset


# ══════════════════════════════════════════════════════════════════════════
# 8. PIPELINE PRINCIPAL — entraînement + évaluation 10 seeds
# ══════════════════════════════════════════════════════════════════════════

def run_ablation(quick:bool=True, N_a:int=50, N_p:int=12):
    """
    quick=True  : 1 seed, dataset réduit (~10 min) — validation du pipeline
    quick=False : 10 seeds, protocole complet identique à BC (~2-3h)
    """
    print(f"\n[Ablation — {'PILOTE' if quick else 'COMPLET'}]")

    n_episodes = 20 if quick else 100
    n_epochs   = 30 if quick else 100
    seeds      = [0] if quick else list(range(10))
    scenarios  = ['S3_PoP_Failure', 'S4_BGP_Anomaly']  # même curriculum de base que BC

    model_classes = {
        'B3_BiLSTM': B3_BiLSTM_Ablation,
        'B4_GCN':    B4_GCN_Ablation,
        'B5_SAGE':   B5_GraphSAGE_Ablation,
        'B6_GAT':    B6_GAT_Ablation,
    }

    print(f"\n[Dataset Oracle partagé] {scenarios} × {n_episodes} épisodes")
    dataset = generate_dataset_with_obs(scenarios, n_episodes, N_a, N_p, T_ep=72)
    print(f"  ✓ {len(dataset)} samples (identique pour toutes les baselines)")

    b1 = B1_Static(N_a, N_p)
    all_evals = []

    for name, ModelClass in model_classes.items():
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        torch.manual_seed(0); np.random.seed(0)
        model = ModelClass(N_a, N_p)
        print(f"  Entraînement ({n_epochs} epochs)...")
        model = train_ablation_model(model, dataset, n_epochs=n_epochs)

        ck = f'checkpoints_ablation/{name}.pt'
        torch.save(model.state_dict(), ck)

        for seed in seeds:
            for scen in scenarios + ['S1_Nominal']:
                r_m  = run_episode(model, scen, seed+900, N_a, N_p, T_ep=288)
                r_b1 = run_episode(b1,    scen, seed+900, N_a, N_p, T_ep=288)
                imp  = (r_b1['L_med']-r_m['L_med'])/r_b1['L_med']*100
                all_evals.append({
                    'model': name, 'seed': seed, 'scenario': scen,
                    'L_med': r_m['L_med'], 'B1_L_med': r_b1['L_med'],
                    'improvement_pct': imp,
                })
        last = [r for r in all_evals if r['model']==name]
        for scen in scenarios + ['S1_Nominal']:
            scen_vals = [r['improvement_pct'] for r in last if r['scenario']==scen]
            print(f"  {scen:<16} gain vs B1 : {np.mean(scen_vals):+.1f}%")

    df = pd.DataFrame(all_evals)
    df.to_csv('results_ablation/GABPO_ablation_results.csv', index=False)

    print(f"\n{'='*70}\nRÉSUMÉ ABLATION — par scénario\n{'='*70}")

    # Référence BC verrouillée (mémoire) pour comparaison directe
    bc_reference = {'S1_Nominal': 1.5, 'S3_PoP_Failure': 3.6, 'S4_BGP_Anomaly': 2.2}

    all_scenarios = scenarios + ['S1_Nominal']
    print(f"\n  {'Model':<12} " + " ".join(f"{s[:14]:<16}" for s in all_scenarios))
    print("  " + "─"*(14 + 17*len(all_scenarios)))
    for name in model_classes:
        sub = df[df.model==name]
        row_str = f"  {name:<12} "
        for scen in all_scenarios:
            scen_sub = sub[sub.scenario==scen]
            m = scen_sub.improvement_pct.mean()
            row_str += f"{m:>+7.2f}%        "
        print(row_str)

    print(f"\n  {'BC (verrouillé)':<12} " +
          "".join(f"{bc_reference.get(s, float('nan')):>+7.2f}%        "
                   for s in all_scenarios))

    print(f"\n{'─'*70}")
    print("Test statistique par scénario (Wilcoxon vs B1, n=nb_seeds)")
    print(f"{'─'*70}")
    for name in model_classes:
        sub = df[df.model==name]
        for scen in all_scenarios:
            scen_sub = sub[sub.scenario==scen]
            if len(scen_sub) < 2:
                continue
            try:
                _, p = stats.wilcoxon(scen_sub.L_med.values, scen_sub.B1_L_med.values)
            except Exception:
                p = float('nan')
            sig = '★' if p < 0.05 else ' '
            print(f"  {name:<12} {scen:<16} p={p:.4f} {sig}")

    print(f"\n✓ Sauvegardé : results_ablation/GABPO_ablation_results.csv")
    if quick:
        print("\n→ Pilote terminé. Si les résultats sont cohérents, "
              "lancer run_ablation(quick=False) pour 10 seeds complets.")
    return df


# ── Point d'entrée ────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok = run_sanity_check()
    if ok:
        print("\nLancement pilote (quick=True)...")
        df = run_ablation(quick=True)
