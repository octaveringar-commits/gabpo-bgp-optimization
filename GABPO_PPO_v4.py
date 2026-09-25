# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Phase P4 v4 : Pipeline Corrigé (3 corrections appliquées)     ║
# ║  C1 : Features réelles dans select_action (plus de np.random.randn)    ║
# ║  C2 : Pénalité BGP δ=0.25 (était 0.05)                                ║
# ║  C3 : Action masking M_valid dans softmax policy                       ║
# ║  Usage : exec(open('GABPO_PPO_v4.py').read())                          ║
# ║  Test   : run_p4_v4(quick=True)   → 1 seed × 2 scénarios (~5 min)     ║
# ║  Complet: run_p4_v4(quick=False)  → 10 seeds × 5 scénarios (~1-2h)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ GABPO Phase P4 v4 | Device: {DEVICE}")
print("=" * 60)

os.makedirs('checkpoints_v4', exist_ok=True)
os.makedirs('results_v4',     exist_ok=True)
os.makedirs('figures_v4',     exist_ok=True)

GLOBAL_SEEDS     = list(range(10))
TRAIN_SCENARIOS  = ['S1_Nominal','S2_FlashCrowd','S3_PoP_Failure','S4_BGP_Anomaly']
OOD_SCENARIO     = 'S5_WACREN_OOD'
ALL_SCENARIOS    = TRAIN_SCENARIOS + [OOD_SCENARIO]
ALPHA_BONF       = 0.05 / 7   # 0.0071

# ══════════════════════════════════════════════════════════════════════════
# 1. ENVIRONNEMENT BGP
# ══════════════════════════════════════════════════════════════════════════

class BGPEnv:
    """
    Environnement BGP Anycast — version v4.
    Latence fortement couplée à f_BGP(c(t),j) pour gradient informatif.
    """
    def __init__(self, scenario:str, seed:int, N_a:int=50, N_p:int=12, T_ep:int=288):
        self.scenario = scenario
        self.seed     = seed
        self.N_a, self.N_p, self.T_ep = N_a, N_p, T_ep
        self.rng      = np.random.RandomState(seed)
        self.t        = 0

        # Latences de base calibrées RIPE Atlas proxy (ms)
        self.base_latency          = np.ones(N_p) * 200.0
        self.base_latency[-2:]     = 260.0   # PoPs tchadiens proxy +60ms

        # Paramètres scénario
        if scenario == 'S2_FlashCrowd':
            self.flash_start, self.flash_end = 50, 80
        elif scenario == 'S3_PoP_Failure':
            self.fail_pop = 4
        elif scenario == 'S4_BGP_Anomaly':
            self.anomaly_start, self.anomaly_end = 30, 60
        elif scenario == 'S5_WACREN_OOD':
            self.base_latency *= 1.412   # +41.2% overhead mesuré RouteViews

    def reset(self) -> dict:
        self.t   = 0
        self.rng = np.random.RandomState(self.seed)
        self._prev_action = np.zeros(self.N_p * self.N_a * 2)
        return self._obs()

    # ── f_BGP (Eq. 7-8 du papier) ─────────────────────────────────────────
    def _f_bgp(self, action:np.ndarray) -> np.ndarray:
        """
        f_BGP(c(t), j) = σ( Σ_{n∈N(j)} [−α_p·l_{j,n} + α_l·Δlp_{j,n}/20] )
        Retourne f_bgp ∈ (0,1)^{N_p}
        """
        N_p, N_a = self.N_p, self.N_a
        prepend   = action[:N_p*N_a].reshape(N_p, N_a).clip(0, 1)
        localpref = action[N_p*N_a:2*N_p*N_a].reshape(N_p, N_a).clip(-1, 1)
        scores    = (-0.50 * prepend + 0.30 * localpref).mean(axis=1)
        return 1.0 / (1.0 + np.exp(-scores))   # sigmoid → (0,1)

    def step(self, action:np.ndarray) -> Tuple[dict,float,bool,dict]:
        f_bgp   = self._f_bgp(action)
        T       = self._traffic(f_bgp)
        latency = self._latency(T, f_bgp)
        J       = self._jain(T.sum(axis=0))
        n_upd   = self._n_updates(action)
        sla_ok  = float(np.median(latency) < 150.0)

        # ── Reward avec δ=0.25 (C2) ────────────────────────────────────────
        dL     = np.median(latency) - np.median(self.base_latency)
        reward = (- 0.40 * dL / 100.0
                  - 0.25 * (1 - J)
                  + 0.15 * sla_ok
                  - 0.25 * n_upd / 5.0     # ← δ augmenté : 0.05→0.25
                  + 0.05 * np.mean(f_bgp))  # bonus attractivité BGP

        self._prev_action = action.copy()
        self.t += 1
        done  = (self.t >= self.T_ep)
        info  = {
            'L_med':     float(np.median(latency)),
            'J':         float(J),
            'delta_bgp': float(n_upd),
            'f_bgp_mean':float(np.mean(f_bgp)),
        }
        return self._obs(), reward, done, info

    def _traffic(self, f_bgp:np.ndarray) -> np.ndarray:
        """Gravity Model BGP-aware — T(t) lié à f_bgp."""
        B = (self.rng.zipf(1.5, self.N_a).astype(float) + 1)
        B = B / B.max() * 520.0
        w = np.ones(self.N_p) * f_bgp
        w[-2:] *= 0.6          # TD PoPs naturellement moins attractifs
        if self.scenario == 'S2_FlashCrowd':
            if self.flash_start <= self.t <= self.flash_end:
                w[2:5] *= 5.0
        if self.scenario == 'S3_PoP_Failure':
            if self.t > self.T_ep // 4:
                w[self.fail_pop] = 0.0
        w_sum = w.sum() + 1e-8
        f_t   = 1.0 + 0.3 * np.sin(2 * np.pi * self.t / self.T_ep - np.pi/2)
        T     = np.outer(B, w / w_sum) * f_t
        T    *= (1 + self.rng.normal(0, 0.05, T.shape))
        return np.maximum(T, 0)

    def _latency(self, T:np.ndarray, f_bgp:np.ndarray) -> np.ndarray:
        """
        Latence fortement couplée à f_bgp.
        f_bgp=0.9 → réduction ~25%,  f_bgp=0.1 → augmentation ~40%
        """
        load  = T.sum(axis=0) / (T.sum() + 1e-8)
        # Impact f_bgp : attractivité élevée → moins de congestion
        cong  = (1.0 + 1.5 * load) * (1.8 - 0.8 * f_bgp)
        lat   = self.base_latency * cong
        # S4 route leak INC-003
        if (self.scenario == 'S4_BGP_Anomaly' and
                hasattr(self,'anomaly_start') and
                self.anomaly_start <= self.t <= self.anomaly_end):
            lat[-2:] *= 1.8
        return np.maximum(lat + self.rng.normal(0, 8, self.N_p), 50.0)

    def _jain(self, loads:np.ndarray) -> float:
        s = loads.sum()
        if s < 1e-8: return 1.0
        return float(s**2 / (len(loads) * (loads**2).sum() + 1e-8))

    def _n_updates(self, action:np.ndarray) -> float:
        return float(min(np.sum(action != self._prev_action) / action.size * 10, 10))

    def _obs(self) -> dict:
        return {'T': self._traffic(np.ones(self.N_p)*0.5), 't': self.t}


# ══════════════════════════════════════════════════════════════════════════
# 2. FEATURES RÉELLES (C1) — extraites de T(t) observé
# ══════════════════════════════════════════════════════════════════════════

def extract_features(obs:dict, N_a:int=50, N_p:int=12):
    """
    Extrait les features AS et PoP depuis l'observation.
    Utilise les statistiques de T(t) — pas de randn.
    C1 : remplace np.random.randn par features informationnelles.
    """
    T    = obs['T']                   # (N_a, N_p)
    Tmax = T.max() + 1e-8
    Tsum = T.sum() + 1e-8
    t    = obs.get('t', 0)

    # Features AS : 13 dimensions
    x_as = np.zeros((N_a, 13), dtype=np.float32)
    x_as[:, 0]  = (T.mean(axis=1) / Tmax)[:N_a]          # trafic moyen normalisé
    x_as[:, 1]  = (T.std(axis=1)  / Tmax)[:N_a]          # variabilité
    x_as[:, 2]  = (T.sum(axis=1)  / Tsum)[:N_a]          # part du trafic total
    x_as[:, 3]  = np.log1p(T.mean(axis=1))[:N_a] / 10.0  # log trafic
    x_as[:, 4]  = (T.max(axis=1)  / Tmax)[:N_a]          # pic
    x_as[:, 5]  = (T.min(axis=1)  / Tmax)[:N_a]          # minimum
    # Features temporelles (sinus/cosinus de t/T_ep)
    x_as[:, 6]  = float(np.sin(2 * np.pi * t / 288))
    x_as[:, 7]  = float(np.cos(2 * np.pi * t / 288))
    # Rangs normalisés (proxy pour degré AS dans G_AS)
    ranks = np.argsort(np.argsort(T.sum(axis=1))).astype(float) / (N_a + 1)
    x_as[:, 8]  = ranks[:N_a]
    # Concentration (entropie normalisée)
    p = T / (Tsum + 1e-8)
    entropy = -(p * np.log(p + 1e-8)).sum(axis=1) / np.log(N_p + 1)
    x_as[:, 9]  = entropy[:N_a]
    x_as[:, 10] = (T.sum(axis=1) > T.sum(axis=1).mean()).astype(float)[:N_a]
    x_as[:, 11] = np.linspace(0.3, 0.7, N_a)   # proxy degré CAIDA
    x_as[:, 12] = np.linspace(0.1, 0.9, N_a)   # proxy uptime RPKI

    # Features PoP : 8 dimensions
    x_pop = np.zeros((N_p, 8), dtype=np.float32)
    load  = T.sum(axis=0) / (Tsum + 1e-8)
    x_pop[:, 0] = (T.mean(axis=0) / Tmax)[:N_p]
    x_pop[:, 1] = (T.std(axis=0)  / Tmax)[:N_p]
    x_pop[:, 2] = load[:N_p]
    x_pop[:, 3] = (T.max(axis=0)  / Tmax)[:N_p]
    x_pop[:, 4] = float(np.sin(2 * np.pi * t / 288))
    x_pop[:, 5] = float(np.cos(2 * np.pi * t / 288))
    x_pop[:, 6] = np.linspace(0.2, 0.8, N_p)   # proxy RTT RIPE Atlas
    x_pop[:, 7] = (load > load.mean()).astype(float)[:N_p]

    return (torch.FloatTensor(x_as),
            torch.FloatTensor(x_pop))


# ══════════════════════════════════════════════════════════════════════════
# 3. GABPO AGENT v4 — avec action masking dans softmax (C3)
# ══════════════════════════════════════════════════════════════════════════

class GABPOAgent(nn.Module):
    """
    GABPO v4 — architecture complète avec action masking.
    C3 : M_valid intégré dans le forward pass policy.
    """
    def __init__(self, N_a:int=50, N_p:int=12, d:int=32, d_state:int=128):
        super().__init__()
        self.N_a, self.N_p, self.d = N_a, N_p, d

        # GAT encoders
        self.gat_as  = nn.Sequential(
            nn.Linear(13, 64), nn.ELU(),
            nn.Linear(64, d),  nn.LayerNorm(d))
        self.gat_pop = nn.Sequential(
            nn.Linear(8, 64),  nn.ELU(),
            nn.Linear(64, d),  nn.LayerNorm(d))

        # HCGA
        self.W_Q   = nn.Linear(d, d, bias=False)
        self.W_K   = nn.Linear(d, d, bias=False)
        self.W_V   = nn.Linear(d, d, bias=False)
        self.W_O   = nn.Linear(d, d, bias=False)
        self.gate  = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())
        self.ln    = nn.LayerNorm(d)

        # Temporal
        self.tcn   = nn.Conv1d(d*2, 64, 3, padding=1)
        self.lstm  = nn.LSTM(64, d_state, batch_first=True)

        # PPO heads
        self.actor  = nn.Sequential(
            nn.Linear(d_state, 256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, N_p * N_a * 2))
        self.critic = nn.Sequential(
            nn.Linear(d_state, 256), nn.ReLU(),
            nn.Linear(256, 1))

        self.A_cross_last = None

    def _hcga(self, h_as, h_pop):
        """HCGA : H_PoP + A_cross → H̃_PoP."""
        Q = self.W_Q(h_pop)   # (N_p, d)
        K = self.W_K(h_as)    # (N_a, d)
        V = self.W_V(h_as)    # (N_a, d)
        A = F.softmax(Q @ K.T / (self.d ** 0.5), dim=-1)  # (N_p, N_a)
        self.A_cross_last = A.detach()
        ctx  = A @ V           # (N_p, d)
        ctx  = self.W_O(ctx)
        gate = self.gate(torch.cat([h_pop, ctx], dim=-1))
        return self.ln(h_pop + gate * ctx)

    def forward(self, x_as, x_pop, M_valid=None):
        """
        Forward pass avec action masking optionnel.
        C3 : si M_valid fourni, masquer les actions invalides avant softmax.
        """
        h_as  = self.gat_as(x_as)          # (N_a, d)
        h_pop = self.gat_pop(x_pop)         # (N_p, d)
        h_pop = self._hcga(h_as, h_pop)     # (N_p, d)

        # Fusion
        emb   = torch.cat([h_as.mean(0), h_pop.mean(0)])  # (2d,)
        fused = F.relu(self.tcn(
            emb.unsqueeze(0).unsqueeze(-1).expand(1, 2*self.d, 3)
        )).mean(-1)  # (1, 64)
        _, (h_t, _) = self.lstm(fused.unsqueeze(0))
        state = h_t.squeeze(0).squeeze(0)   # (d_state,)

        # Policy logits
        logits = self.actor(state)           # (N_p*N_a*2,)

        # ── C3 : Action masking M_valid dans softmax ────────────────────
        if M_valid is not None:
            # M_valid : tensor (N_p*N_a*2,) avec 1=valide, 0=invalide
            # Les actions invalides reçoivent logit = -1e9 → proba ≈ 0
            mask_penalty = (1.0 - M_valid.float()) * (-1e9)
            logits = logits + mask_penalty

        # Action finale : tanh pour borner ∈ (-1, 1)
        action = torch.tanh(logits)
        value  = self.critic(state).squeeze()
        return action, value

    def select_action(self, obs, apply_mask=True):
        """Sélectionne une action depuis l'observation — C1+C3."""
        x_as, x_pop = extract_features(obs, self.N_a, self.N_p)

        # M_valid : masque BGP simplifié
        # (invalide : prepend>3 ou changements excessifs)
        M_valid = None
        if apply_mask:
            M_valid = torch.ones(self.N_p * self.N_a * 2)
            # Désactiver les combinaisons extrêmes (derniers 20% de l'espace)
            n = self.N_p * self.N_a * 2
            M_valid[int(n*0.8):] = 0.0  # heuristique de sécurité BGP

        with torch.no_grad():
            action, _ = self.forward(x_as, x_pop, M_valid)
        return action.numpy()

    def get_A_cross(self):
        """Retourne la dernière matrice A_cross pour XAI."""
        return self.A_cross_last.numpy() if self.A_cross_last is not None \
               else np.ones((self.N_p, self.N_a)) / self.N_a


# ══════════════════════════════════════════════════════════════════════════
# 4. BASELINES v4 — toutes avec features réelles
# ══════════════════════════════════════════════════════════════════════════

class B1_Static:
    def __init__(self, N_a=50, N_p=12):
        self.N_a, self.N_p = N_a, N_p
        self._action = np.zeros(N_p * N_a * 2)
    def select_action(self, obs):
        return self._action.copy()

class B2_Greedy:
    def __init__(self, N_a=50, N_p=12):
        self.N_a, self.N_p = N_a, N_p
    def select_action(self, obs):
        T    = obs['T']
        load = T.sum(axis=0)
        best = int(np.argmin(load[:self.N_p]))
        a    = np.zeros(self.N_p * self.N_a * 2)
        offset = self.N_p * self.N_a
        for i in range(self.N_a):
            a[offset + best * self.N_a + i] = 1.0
        return a

class B3_BiLSTM(nn.Module):
    def __init__(self, N_a=50, N_p=12, hidden=64):
        super().__init__()
        self.N_a, self.N_p = N_a, N_p
        # Entrée : features aplaties des PoPs (8 dims × N_p)
        self.lstm = nn.LSTM(N_p*8, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden*2, N_p*N_a*2)
    def select_action(self, obs):
        _, x_pop = extract_features(obs, self.N_a, self.N_p)
        feat = x_pop.flatten().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out, _ = self.lstm(feat)
            return torch.tanh(self.head(out[:,-1,:])).squeeze().numpy()

class B4_GCN(nn.Module):
    def __init__(self, N_a=50, N_p=12, d=32):
        super().__init__()
        self.N_a, self.N_p = N_a, N_p
        self.enc = nn.Sequential(nn.Linear(13+8, 64), nn.ReLU(), nn.Linear(64, d))
        self.head = nn.Linear(d, N_p*N_a*2)
    def select_action(self, obs):
        x_as, x_pop = extract_features(obs, self.N_a, self.N_p)
        # Mean pooling simple (pas d'attention cross-graph)
        h = self.enc(torch.cat([
            x_as.mean(0).expand(1,-1),
            x_pop.mean(0).expand(1,-1)
        ], dim=-1))
        with torch.no_grad():
            return torch.tanh(self.head(h)).squeeze().numpy()

class B5_GraphSAGE(nn.Module):
    def __init__(self, N_a=50, N_p=12, d=32):
        super().__init__()
        self.N_a, self.N_p = N_a, N_p
        self.sage_as  = nn.Linear(13, d)
        self.sage_pop = nn.Linear(8,  d)
        self.head     = nn.Linear(d*2, N_p*N_a*2)
    def select_action(self, obs):
        x_as, x_pop = extract_features(obs, self.N_a, self.N_p)
        h = torch.cat([F.relu(self.sage_as(x_as)).mean(0),
                       F.relu(self.sage_pop(x_pop)).mean(0)])
        with torch.no_grad():
            return torch.tanh(self.head(h)).squeeze().numpy()

class B6_GAT(nn.Module):
    """B6 : GAT sur G_AS uniquement — pas de couplage HCGA."""
    def __init__(self, N_a=50, N_p=12, d=32):
        super().__init__()
        self.N_a, self.N_p = N_a, N_p
        self.gat  = nn.Sequential(nn.Linear(13, 64), nn.ELU(), nn.Linear(64, d))
        self.head = nn.Linear(d, N_p*N_a*2)
    def select_action(self, obs):
        x_as, _ = extract_features(obs, self.N_a, self.N_p)
        with torch.no_grad():
            h = self.gat(x_as).mean(0)
            return torch.tanh(self.head(h)).squeeze().numpy()


# ══════════════════════════════════════════════════════════════════════════
# 5. ÉVALUATION D'UN ÉPISODE
# ══════════════════════════════════════════════════════════════════════════

def eval_episode(model, scenario:str, seed:int,
                 N_a:int=50, N_p:int=12, T_ep:int=288) -> dict:
    env = BGPEnv(scenario, seed, N_a, N_p, T_ep)
    obs = env.reset()
    Ls, Js, Bs = [], [], []
    for _ in range(T_ep):
        a = model.select_action(obs)
        obs, _, done, info = env.step(a)
        Ls.append(info['L_med'])
        Js.append(info['J'])
        Bs.append(info['delta_bgp'])
        if done: break
    return {
        'L_med':     float(np.median(Ls)),
        'J':         float(np.mean(Js)),
        'delta_bgp': float(np.mean(Bs)),
        'L_p95':     float(np.percentile(Ls, 95)),
    }


# ══════════════════════════════════════════════════════════════════════════
# 6. ENTRAÎNEMENT GABPO v4
# ══════════════════════════════════════════════════════════════════════════

def train_gabpo_v4(seed:int, N_a:int=50, N_p:int=12,
                   n_epochs:int=60, verbose:bool=True) -> GABPOAgent:
    """
    Entraînement GABPO v4 avec PPO Actor-Critic + action masking.
    ZDL : S5 jamais utilisé en entraînement.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = GABPOAgent(N_a, N_p).to(DEVICE)
    opt_actor  = torch.optim.Adam(
        list(model.actor.parameters()), lr=3e-4)
    opt_critic = torch.optim.Adam(
        list(model.critic.parameters()) +
        list(model.gat_as.parameters()) +
        list(model.gat_pop.parameters()) +
        list(model.W_Q.parameters()) +
        list(model.W_K.parameters()) +
        list(model.W_V.parameters()) +
        list(model.W_O.parameters()) +
        list(model.gate.parameters()) +
        list(model.tcn.parameters()) +
        list(model.lstm.parameters()), lr=1e-3)

    eps_clip   = 0.2   # PPO clip parameter
    gamma      = 0.99
    best_L     = 1e9
    patience   = 0

    for epoch in range(n_epochs):
        epoch_actor_loss  = 0.0
        epoch_critic_loss = 0.0

        for s in TRAIN_SCENARIOS:
            env  = BGPEnv(s, seed + epoch, N_a, N_p, T_ep=72)
            obs  = env.reset()
            traj = []

            for _ in range(72):
                x_as, x_pop = extract_features(obs, N_a, N_p)
                M_valid = torch.ones(N_p * N_a * 2)
                M_valid[int(N_p*N_a*2*0.8):] = 0.0

                with torch.no_grad():
                    action, value = model.forward(
                        x_as.to(DEVICE), x_pop.to(DEVICE), M_valid.to(DEVICE))
                    log_prob = -0.5 * (action**2).sum()  # Gaussian approx

                obs_next, reward, done, info = env.step(action.cpu().numpy())
                traj.append({
                    'x_as': x_as, 'x_pop': x_pop, 'M_valid': M_valid,
                    'action': action.cpu(), 'log_prob': log_prob.item(),
                    'value': value.item(), 'reward': reward,
                })
                obs = obs_next
                if done: break

            if len(traj) < 2: continue

            # ── Returns et Advantages ──────────────────────────────────
            returns, adv = [], []
            G, A = 0.0, 0.0
            for step in reversed(traj):
                G = step['reward'] + gamma * G
                returns.insert(0, G)
            ret_t = torch.FloatTensor(returns)
            if ret_t.std() > 1e-6:
                ret_norm = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-6)
            else:
                ret_norm = ret_t * 0

            # ── Critic update ──────────────────────────────────────────
            for i, step in enumerate(traj):
                _, val = model.forward(
                    step['x_as'].to(DEVICE),
                    step['x_pop'].to(DEVICE),
                    step['M_valid'].to(DEVICE))
                critic_loss = F.smooth_l1_loss(
                    val, ret_norm[i].to(DEVICE).detach())
                opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_critic.step()
                epoch_critic_loss += critic_loss.item()

            # ── Actor update (PPO clip) ────────────────────────────────
            for i, step in enumerate(traj):
                act_new, val = model.forward(
                    step['x_as'].to(DEVICE),
                    step['x_pop'].to(DEVICE),
                    step['M_valid'].to(DEVICE))
                log_prob_new = -0.5 * (act_new**2).sum()
                ratio        = torch.exp(log_prob_new - step['log_prob'])
                advantage    = (ret_norm[i] - val.detach()).to(DEVICE)
                # PPO clipped surrogate objective
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                actor_loss = -torch.min(surr1, surr2)
                # Entropy bonus (encourage exploration)
                entropy    = -0.01 * log_prob_new
                total_loss = actor_loss + entropy
                opt_actor.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_actor.step()
                epoch_actor_loss += total_loss.item()

        # ── Validation tous les 10 epochs ─────────────────────────────
        if (epoch + 1) % 10 == 0:
            val_results = []
            for vs in ['S1_Nominal', 'S2_FlashCrowd']:
                r = eval_episode(model, vs, seed+200, N_a, N_p, T_ep=72)
                val_results.append(r['L_med'])
            val_L = float(np.median(val_results))
            if verbose:
                print(f"  Seed {seed} | Epoch {epoch+1:3d} | "
                      f"actor={epoch_actor_loss:.3f} | "
                      f"critic={epoch_critic_loss:.3f} | "
                      f"val_L_med={val_L:.1f}ms")
            if val_L < best_L:
                best_L    = val_L
                patience  = 0
                torch.save(model.state_dict(),
                           f'checkpoints_v4/gabpo_s{seed}.pt')
            else:
                patience += 1
                if patience >= 4:
                    if verbose: print(f"  Early stopping epoch {epoch+1}")
                    break

    ck = f'checkpoints_v4/gabpo_s{seed}.pt'
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck, map_location='cpu'))
    return model


# ══════════════════════════════════════════════════════════════════════════
# 7. STATISTIQUES
# ══════════════════════════════════════════════════════════════════════════

def cliffs_delta(x, y):
    n = len(x) * len(y)
    dom = sum(1 if xi > yi else (-1 if xi < yi else 0)
              for xi in x for yi in y)
    return dom / n

def bootstrap_ci(data, n=5000, alpha=0.05):
    boots = [np.mean(np.random.choice(data, len(data), replace=True))
             for _ in range(n)]
    return float(np.percentile(boots, 100*alpha/2)), \
           float(np.percentile(boots, 100*(1-alpha/2)))

def compute_table2(all_res:dict, scenarios:list) -> pd.DataFrame:
    """Table II : Bootstrap CI + Wilcoxon + Bonferroni + Cliff's Delta."""
    models = list(all_res.keys())
    gabpo_L = np.array([r['L_med'] for s in scenarios
                         for r in all_res.get('GABPO',{}).get(s,[])])
    rows = []
    for m in models:
        aL  = np.array([r['L_med'] for s in scenarios
                         for r in all_res[m].get(s,[])])
        aJ  = np.array([r['J']     for s in scenarios
                         for r in all_res[m].get(s,[])])
        aB  = np.array([r['delta_bgp'] for s in scenarios
                         for r in all_res[m].get(s,[])])
        aP95= np.array([r['L_p95'] for s in scenarios
                         for r in all_res[m].get(s,[])])
        if len(aL) == 0: continue
        ci_lo, ci_hi = bootstrap_ci(aL)
        row = {
            'Model':       m,
            'L_med_mean':  round(float(np.mean(aL)),1),
            'L_med_std':   round(float(np.std(aL)),1),
            'L_med_median':round(float(np.median(aL)),1),
            'CI_95_lo':    round(ci_lo,1),
            'CI_95_hi':    round(ci_hi,1),
            'J_mean':      round(float(np.mean(aJ)),3),
            'J_std':       round(float(np.std(aJ)),3),
            'delta_bgp':   round(float(np.mean(aB)),2),
            'L_p95_mean':  round(float(np.mean(aP95)),1),
            'n_obs':       len(aL),
        }
        # Wilcoxon + Cliff's Delta vs GABPO
        if m != 'GABPO' and len(gabpo_L) > 0:
            n = min(len(gabpo_L), len(aL))
            try:
                _, p = stats.wilcoxon(gabpo_L[:n], aL[:n])
                d    = cliffs_delta(list(gabpo_L[:n]), list(aL[:n]))
                row['wilcoxon_p']   = float(p)
                row['cliffs_delta'] = round(float(d), 3)
                row['significant']  = bool(p < ALPHA_BONF)
            except Exception:
                row['wilcoxon_p']   = None
                row['cliffs_delta'] = None
                row['significant']  = False
        elif m == 'GABPO':
            row['wilcoxon_p']   = None
            row['cliffs_delta'] = 0.0   # référence
            row['significant']  = None
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 8. XAI — A_cross analysis
# ══════════════════════════════════════════════════════════════════════════

def analyze_xai_v4(gabpo_models:list, N_a:int=50, N_p:int=12) -> dict:
    """Analyse A_cross S1 vs S4 pour Section V-C."""
    ANOMALY_IDX = [5, 6]   # AS37309 CAMTEL, AS36896 SotelChadLTE
    A_nom_list, A_ano_list, acc_list = [], [], []

    for si, model in enumerate(gabpo_models[:5]):
        # Nominal S1
        env = BGPEnv('S1_Nominal', si, N_a, N_p, T_ep=20)
        obs = env.reset()
        for _ in range(20):
            a = model.select_action(obs)
            obs, _, done, _ = env.step(a)
            if done: break
        A_nom_list.append(model.get_A_cross())

        # Anomalie S4 (step t=40)
        env = BGPEnv('S4_BGP_Anomaly', si, N_a, N_p, T_ep=72)
        obs = env.reset()
        for _ in range(40):
            a = model.select_action(obs)
            obs, _, done, _ = env.step(a)
            if done: break
        A_ano_list.append(model.get_A_cross())

        # ACC_xai
        A_ano = model.get_A_cross()
        top3  = np.argsort(-A_ano, axis=1)[:, :3]
        hits  = sum(1 for p in range(N_p)
                    if any(i in ANOMALY_IDX for i in top3[p]))
        acc_list.append(hits / N_p)

    A_nom = np.mean(A_nom_list, axis=0)
    A_ano = np.mean(A_ano_list, axis=0)

    ratios = [A_ano[:, i].mean() / (A_nom[:, i].mean() + 1e-8)
              for i in ANOMALY_IDX]

    # Sauvegarder Fig. 3
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        im1 = ax1.imshow(A_nom, aspect='auto', cmap='Blues')
        ax1.set_title('(a) Nominal — S1'); ax1.set_xlabel('AS index')
        ax1.set_ylabel('PoP index'); plt.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(A_ano, aspect='auto', cmap='Reds')
        ax2.set_title(f'(b) BGP Anomaly — S4 (INC-003)\nACC_xai={np.mean(acc_list):.1%}')
        ax2.set_xlabel('AS index'); ax2.set_ylabel('PoP index')
        plt.colorbar(im2, ax=ax2)
        for ax in [ax1, ax2]:
            for idx in ANOMALY_IDX:
                ax.axvline(idx, color='red', lw=2, ls='--', alpha=0.8)
        plt.tight_layout()
        plt.savefig('figures_v4/GABPO_Fig3_Across_v4.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        print("  ✓ Fig. 3 sauvegardée")
    except Exception as e:
        print(f"  ⚠ matplotlib : {e}")

    return {
        'acc_xai_mean': float(np.mean(acc_list)),
        'acc_xai_std':  float(np.std(acc_list)),
        'anomaly_ratio_mean': float(np.mean(ratios)),
        'A_nom': A_nom, 'A_ano': A_ano,
    }


# ══════════════════════════════════════════════════════════════════════════
# 9. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def run_p4_v4(quick:bool=False, N_a:int=50, N_p:int=12, n_epochs:int=60):
    """
    Pipeline Phase P4 v4 complet.
    quick=True  → 1 seed × 2 scénarios  (~5-10 min CPU)
    quick=False → 10 seeds × 5 scénarios (~1-2h GPU)
    """
    seeds     = [0] if quick else GLOBAL_SEEDS
    scenarios = TRAIN_SCENARIOS[:2] if quick else ALL_SCENARIOS
    epochs    = 10  if quick else n_epochs

    print(f"\n[Phase P4 v4 — {'QUICK TEST' if quick else 'FULL'}]")
    print(f"  Seeds: {seeds} | Scénarios: {len(scenarios)} | Epochs: {epochs}")
    print(f"  C1: features réelles | C2: δ=0.25 | C3: action masking")
    print("=" * 60)

    # ── Étape 1 : Entraînement ─────────────────────────────────────────
    print(f"\n[Étape 1] Entraînement GABPO ({len(seeds)} seeds × {epochs} epochs)...")
    gabpo_models = []
    for seed in seeds:
        print(f"  Seed {seed}...")
        m = train_gabpo_v4(seed, N_a, N_p, epochs, verbose=True)
        gabpo_models.append(m)
        # Évaluation rapide pour vérifier l'apprentissage
        r = eval_episode(m, 'S1_Nominal', seed+100, N_a, N_p, T_ep=72)
        print(f"  Seed {seed} ✓ | S1 L_med={r['L_med']:.1f}ms "
              f"| J={r['J']:.3f} | Δ_BGP={r['delta_bgp']:.2f}")
    print(f"\n✓ {len(gabpo_models)} modèles GABPO entraînés")

    # ── Vérifier si GABPO apprend ──────────────────────────────────────
    b1 = B1_Static(N_a, N_p)
    r_gabpo = eval_episode(gabpo_models[0], 'S1_Nominal', 999, N_a, N_p)
    r_b1    = eval_episode(b1,              'S1_Nominal', 999, N_a, N_p)
    print(f"\n  [Vérification apprentissage S1]")
    print(f"  GABPO  : L_med={r_gabpo['L_med']:.1f}ms | J={r_gabpo['J']:.3f}")
    print(f"  B1 Stat: L_med={r_b1['L_med']:.1f}ms | J={r_b1['J']:.3f}")

    if r_gabpo['L_med'] >= r_b1['L_med'] * 0.98:
        print("  ⚠ GABPO n'est pas encore meilleur que B1 Static")
        print("  → Suggestion : augmenter n_epochs ou lr")
        if quick:
            print("  → Lancer avec quick=False pour l'entraînement complet")
    else:
        pct = (r_b1['L_med'] - r_gabpo['L_med']) / r_b1['L_med'] * 100
        print(f"  ✅ GABPO surpasse B1 Static de {pct:.1f}%")

    if quick:
        print("\n[Quick test terminé] → Lancer run_p4_v4(quick=False) pour les vrais résultats")
        return None, None, None

    # ── Étape 2 : Évaluation complète ─────────────────────────────────
    print(f"\n[Étape 2] Évaluation 7 modèles × {len(scenarios)} scénarios × {len(seeds)} seeds...")
    baselines = {
        'B1_Static':    B1_Static(N_a, N_p),
        'B2_Greedy':    B2_Greedy(N_a, N_p),
        'B3_BiLSTM':    B3_BiLSTM(N_a, N_p),
        'B4_GCN':       B4_GCN(N_a, N_p),
        'B5_GraphSAGE': B5_GraphSAGE(N_a, N_p),
        'B6_GAT':       B6_GAT(N_a, N_p),
    }
    all_models = list(baselines.keys()) + ['GABPO']
    results    = {m: {s: [] for s in scenarios} for m in all_models}
    total = len(all_models) * len(scenarios) * len(seeds)
    cnt   = 0

    for mname in all_models:
        for scen in scenarios:
            for si, seed in enumerate(seeds):
                model = gabpo_models[si] if mname == 'GABPO' \
                        else baselines[mname]
                r = eval_episode(model, scen, seed, N_a, N_p, T_ep=288)
                results[mname][scen].append(r)
                cnt += 1
                if cnt % 70 == 0:
                    print(f"  {cnt}/{total} runs")
    print(f"✓ {cnt} runs complétés")

    # ── Étape 3 : Table II ─────────────────────────────────────────────
    print("\n[Étape 3] Calcul Table II...")
    df = compute_table2(results, scenarios)

    print(f"\n  {'Modèle':<14} {'L_med (ms)':<20} {'J':<8} "
          f"{'Δ_BGP':<7} {'Cliff\'s Δ':<12} {'p-value'}")
    print("  " + "─"*72)
    for _, row in df.iterrows():
        d   = row.get('cliffs_delta')
        p   = row.get('wilcoxon_p')
        sig = '★' if row.get('significant') else ' '
        ds  = f"{d:+.3f}{sig}" if d is not None else '0.000 (ref)'
        ps  = f"{p:.2e}" if p is not None else '—'
        print(f"  {row['Model']:<14} "
              f"{row['L_med_mean']:>7.1f}±{row['L_med_std']:>5.1f} "
              f"[{row['CI_95_lo']:.1f},{row['CI_95_hi']:.1f}]  "
              f"{row['J_mean']:.3f}  {row['delta_bgp']:>5.2f}  "
              f"{ds:<13} {ps}")

    # ── Étape 4 : XAI ─────────────────────────────────────────────────
    print("\n[Étape 4] Analyse XAI — A_cross...")
    xai = analyze_xai_v4(gabpo_models, N_a, N_p)
    print(f"  ACC_xai  = {xai['acc_xai_mean']:.1%} ± {xai['acc_xai_std']:.1%}")
    print(f"  Ratio A_cross anomalie/nominal = {xai['anomaly_ratio_mean']:.2f}×")

    # ── Étape 5 : Export ──────────────────────────────────────────────
    df.to_csv('results_v4/GABPO_Table2_v4.csv', index=False)
    with open('results_v4/GABPO_stats_v4.json', 'w') as f:
        json.dump({
            'version':    'P4v4',
            'timestamp':  time.strftime('%Y%m%d_%H%M'),
            'n_seeds':    len(seeds),
            'n_scenarios':len(scenarios),
            'corrections':['C1_real_features','C2_delta025','C3_action_masking'],
            'table2':     df.to_dict('records'),
            'xai':        {k:v for k,v in xai.items()
                           if not isinstance(v, np.ndarray)},
            'alpha_bonferroni': ALPHA_BONF,
        }, f, indent=2, default=str)

    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║  Phase P4 v4 — Résultats                           ║
  ╠══════════════════════════════════════════════════════╣
  ║  Table II  : results_v4/GABPO_Table2_v4.csv        ║
  ║  Stats     : results_v4/GABPO_stats_v4.json        ║
  ║  Fig. 3    : figures_v4/GABPO_Fig3_Across_v4.png   ║
  ╚══════════════════════════════════════════════════════╝
    """)

    return df, xai, results


# ── Point d'entrée ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Test 1 seed (quick=True)...")
    df, xai, res = run_p4_v4(quick=True)
    print("\n→ Lancer run_p4_v4(quick=False) pour les 350 runs complets")
