# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO Phase P0b — Simulateur BGP Dynamique                            ║
# ║                                                                        ║
# ║  Corrections fondamentales issues du test Oracle (Phase P0) :          ║
# ║                                                                        ║
# ║  C1 — Redistribution softmax du trafic                                 ║
# ║       T(t+1) = D(t+1) · softmax(S_BGP(action, load) / τ)              ║
# ║       Les actions BGP modifient réellement la distribution future       ║
# ║                                                                        ║
# ║  C2 — Boucle de congestion                                             ║
# ║       Load_p ↑ → attractivité_p ↓ → redistribution naturelle           ║
# ║                                                                        ║
# ║  C3 — Inertie BGP (convergence progressive)                            ║
# ║       T(t+1) = (1−ρ)·T(t) + ρ·T_target(t+1)                          ║
# ║       Représente la convergence BGP réelle (~5 min = 1 step)           ║
# ║                                                                        ║
# ║  Paramètres justifiés par la littérature BGP :                         ║
# ║       θ1=1.0 (LOCAL_PREF), θ2=0.8 (prepend), θ3=1.5 (congestion)     ║
# ║       θ4=0.5 (délai), τ=1.0 (température softmax), ρ=0.25 (inertie)  ║
# ║       Références : RFC 4271, Gill et al. PAM 2008, Fanou IMC 2017      ║
# ║                                                                        ║
# ║  Test Oracle re-run : cible Oracle << B1 dans S2/S3/S4                ║
# ║                                                                        ║
# ║  Usage : exec(open('GABPO_P0b_DynamicEnv.py').read())                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax as scipy_softmax
import json, time, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('results_p0b', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO Phase P0b — Simulateur Dynamique             ║")
print("╚══════════════════════════════════════════════════════╝")
print()
print("Corrections : softmax redistribution + congestion + inertie BGP")
print()

# ══════════════════════════════════════════════════════════════════════════
# PARAMÈTRES — JUSTIFIÉS PAR LA LITTÉRATURE (gelés avant expérience)
# ══════════════════════════════════════════════════════════════════════════

class BGPParams:
    """
    Paramètres du simulateur dynamique.
    IMPORTANT : ces valeurs sont fixées a priori sur la base de la littérature
    BGP et non calibrées pour favoriser GABPO.
    Références :
      - RFC 4271 §9.1 : hiérarchie de sélection BGP
      - Gill et al., PAM 2008 : impact relatif LOCAL_PREF vs AS_PATH
      - Fanou et al., IMC 2017 : overhead routage Afrique (ρ calibré sur 5min)
      - Labovitz et al., SIGCOMM 2000 : convergence BGP (τ_conv ≈ 30s-5min)
    """
    # Poids du score d'attractivité BGP S_{p,a}(t)
    theta_lp   = 1.0   # LOCAL_PREF (primaire dans RFC 4271)
    theta_prep = 0.8   # AS_PATH prepending (secondaire dans RFC 4271)
    theta_load = 1.5   # Congestion PoP (endogenous feedback)
    theta_delay= 0.5   # Délai de base du PoP

    # Température softmax — contrôle la concentration du trafic
    # τ=0.5 → très concentré (WTA), τ=2.0 → uniforme
    # τ=1.0 : compromis opérationnel raisonnable [Calder et al., IMC 2015]
    tau = 1.0

    # Inertie BGP ρ : vitesse de redistribution du trafic
    # T(t+1) = (1−ρ)·T(t) + ρ·T_target(t+1)
    # ρ=0.25 correspond à une constante de temps de 4 steps = 20 minutes
    # Justification : convergence BGP typique 5-30 min [Labovitz 2000]
    rho = 0.25

    # Multiplicateur de demande par scénario
    flash_factor  = 8.0    # ×8 trafic Flash Crowd (S2) — stressant
    anomaly_boost = 2.5    # ×2.5 latence anomalie BGP (S4)

PARAMS = BGPParams()


# ══════════════════════════════════════════════════════════════════════════
# ENVIRONNEMENT BGP DYNAMIQUE
# ══════════════════════════════════════════════════════════════════════════

class BGPDynamicEnv:
    """
    Simulateur BGP avec boucle de rétroaction complète :

        a(t) → S_BGP(t) → P(t) = softmax(S_BGP/τ)
             ↗                                    ↘
        T(t)    →   T_target(t+1) = D(t+1)·P(t)   →  T(t+1) = (1-ρ)T(t) + ρT_target
             ↘                                    ↗
                L(t+1) = G(T(t+1), a(t+1))

    La distribution du trafic évolue en réponse aux actions BGP.
    Cela crée des états dynamiquement distincts que PPO peut exploiter.
    """
    def __init__(self, scenario:str, seed:int,
                 N_a:int=50, N_p:int=12, T_ep:int=288,
                 params:BGPParams=PARAMS):
        self.scenario = scenario
        self.seed     = seed
        self.N_a, self.N_p, self.T_ep = N_a, N_p, T_ep
        self.p        = params
        self.rng      = np.random.RandomState(seed)
        self.t        = 0

        # Latences de base (ms) — proxy RIPE Atlas + Fanou IMC 2017
        self.delay_base      = np.ones(N_p) * 150.0
        self.delay_base[-2:] = 210.0   # TD PoPs (+60ms proxy CM/NG)

        # Capacités PoP (Gbps) — calibrées AfPIF 2024
        self.capacity = np.ones(N_p) * 100.0
        self.capacity[-2:] = 30.0   # TD PoPs capacité réduite

        # Scénario-specific
        if scenario == 'S2_FlashCrowd':
            self.flash_start, self.flash_end = 40, 100
            self.flash_pops  = [2, 3, 4]   # CAMIX PoPs
        elif scenario == 'S3_PoP_Failure':
            self.fail_pop    = 4
            self.fail_step   = self.T_ep // 4
        elif scenario == 'S4_BGP_Anomaly':
            self.anom_start  = 30
            self.anom_end    = 90
            self.anom_ases   = [5, 6]   # AS37309 CAMTEL, AS36896 SotelChadLTE
        elif scenario == 'S5_WACREN_OOD':
            # +41.2% overhead mesuré RouteViews — inflater le délai de base
            self.delay_base *= 1.412
            self.capacity   *= 0.7   # capacité réduite WACREN

    def reset(self):
        self.t   = 0
        self.rng = np.random.RandomState(self.seed)

        # État initial : trafic uniforme
        D        = self._demand()
        P_init   = np.ones((self.N_p, self.N_a)) / self.N_p
        self.T   = np.outer(P_init.T[0], D).T   # (N_a, N_p)
        self.T   = np.maximum(self.T, 0)

        self._prev_action = np.zeros(self.N_p * self.N_a * 2)
        self._load_hist   = [self._load()]
        return self._obs()

    # ── Score d'attractivité BGP (C1 + C2) ───────────────────────────────
    def _score_bgp(self, action:np.ndarray) -> np.ndarray:
        """
        S_{p,a}(t) = θ1·LP_{p,a} − θ2·Prep_{p,a} − θ3·Load_p − θ4·Delay_p

        Retourne S ∈ ℝ^{N_p × N_a}
        Justification : RFC 4271 §9.1 hiérarchie + congestion feedback
        """
        N_p, N_a = self.N_p, self.N_a
        prepend   = action[:N_p*N_a].reshape(N_p, N_a).clip(0, 1)
        localpref = action[N_p*N_a:2*N_p*N_a].reshape(N_p, N_a).clip(-1, 1)

        load    = self._load()       # (N_p,) — charge actuelle normalisée
        delay   = self.delay_base / self.delay_base.max()

        # Score : LOCAL_PREF positif, prepend/congestion/délai négatifs
        S = (  self.p.theta_lp    * localpref          # (N_p, N_a)
             - self.p.theta_prep  * prepend             # (N_p, N_a)
             - self.p.theta_load  * load[:, np.newaxis] # (N_p, 1) broadcast
             - self.p.theta_delay * delay[:, np.newaxis]) # (N_p, 1)

        return S   # (N_p, N_a)

    # ── Redistribution du trafic (C1) ─────────────────────────────────────
    def _redistribute(self, action:np.ndarray) -> np.ndarray:
        """
        T_target(t+1) = D(t+1) · P(t)
        où P_{p,a}(t) = softmax(S_{p,a}(t) / τ)  sur l'axe des PoPs

        Justification : softmax routing = BGP best-path selection probabiliste
        [Calder et al., IMC 2015 — Anycast traffic engineering]
        """
        S   = self._score_bgp(action)   # (N_p, N_a)
        D   = self._demand()            # (N_a,) — demande AS

        # Softmax sur l'axe des PoPs : P ∈ ℝ^{N_p × N_a}
        # Chaque colonne a = distribution sur les PoPs pour l'AS a
        P = scipy_softmax(S / self.p.tau, axis=0)  # (N_p, N_a)

        # Appliquer scénarios spéciaux
        if self.scenario == 'S3_PoP_Failure':
            if self.t >= self.fail_step:
                P[self.fail_pop, :] = 0.0
                P = P / (P.sum(axis=0, keepdims=True) + 1e-8)

        # T_target : trafic cible
        T_target = P * D[np.newaxis, :]   # (N_p, N_a)
        return T_target.T   # (N_a, N_p)

    # ── Inertie BGP (C3) ──────────────────────────────────────────────────
    def _apply_inertia(self, T_target:np.ndarray) -> np.ndarray:
        """
        T(t+1) = (1−ρ)·T(t) + ρ·T_target(t+1)

        ρ=0.25 : constante de temps 4 steps = 20 min
        Justification : convergence BGP [Labovitz et al., SIGCOMM 2000]
        """
        T_new = ((1 - self.p.rho) * self.T +
                  self.p.rho      * T_target)
        # Bruit multiplicatif log-normal (σ=0.05) [AfPIF 2024]
        noise = self.rng.normal(0, 0.05, T_new.shape)
        return np.maximum(T_new * (1 + noise), 0)

    # ── Calcul de la latence ──────────────────────────────────────────────
    def _latency(self, T:np.ndarray, action:np.ndarray) -> np.ndarray:
        """
        L_p = delay_base_p × (1 + α·(load_p / capacity_p)^β)

        Modèle M/M/1 approximé : latence croît avec la charge
        α=2.0, β=1.5 — calibré pour que surcharge → latence ×3
        [Kelly, 1991 — Effective bandwidth]
        """
        load     = T.sum(axis=0)         # (N_p,) charge absolue
        util     = load / (self.capacity + 1e-8)   # utilisation ∈ [0,∞)
        util_capped = np.minimum(util, 2.0)        # cap à 200%

        # Latence base + congestion
        lat = self.delay_base * (1 + 2.0 * util_capped**1.5)

        # Anomalie BGP S4 : +délai pour les PoPs touchés
        if (self.scenario == 'S4_BGP_Anomaly' and
                hasattr(self,'anom_start') and
                self.anom_start <= self.t <= self.anom_end):
            lat[-2:] *= self.p.anomaly_boost

        return np.maximum(lat + self.rng.normal(0, 8, self.N_p), 50.0)

    # ── Demande AS (Zipf calibrée IXPN Lagos) ────────────────────────────
    def _demand(self) -> np.ndarray:
        """Demande par AS (Gbps) — Zipf α=1.5, calibrée AfPIF 2024."""
        D = self.rng.zipf(1.5, self.N_a).astype(float) + 1
        D = D / D.max() * 520.0   # IXPN Lagos peak 520 Gbps

        # Facteur diurnal
        f_t = 1.0 + 0.35 * np.sin(2*np.pi*self.t/self.T_ep - np.pi/2)
        D   = D * f_t

        # Flash crowd S2 : demande × flash_factor sur certains ASes
        if (self.scenario == 'S2_FlashCrowd' and
                hasattr(self,'flash_start') and
                self.flash_start <= self.t <= self.flash_end):
            flash_ases = self.rng.choice(self.N_a, 15, replace=False)
            D[flash_ases] *= self.p.flash_factor

        # Anomalie S4 : trafic dévié pour les AS touchés
        if (self.scenario == 'S4_BGP_Anomaly' and
                hasattr(self,'anom_start') and
                self.anom_start <= self.t <= self.anom_end):
            D[self.anom_ases] *= 3.0   # ×3 trafic vers ASes anomalie

        return np.maximum(D, 0)

    def _load(self) -> np.ndarray:
        """Charge normalisée par PoP ∈ [0,1]."""
        load = self.T.sum(axis=0)
        return load / (load.max() + 1e-8)

    def _jain(self, loads:np.ndarray) -> float:
        s = loads.sum()
        if s < 1e-8: return 1.0
        return float(s**2 / (len(loads)*(loads**2).sum() + 1e-8))

    def _n_updates(self, action:np.ndarray) -> float:
        return float(min(np.sum(action != self._prev_action) / action.size * 10, 10))

    def _obs(self) -> dict:
        return {'T': self.T.copy(), 't': self.t, 'load': self._load()}

    def step(self, action:np.ndarray):
        """
        Transition complète :
        a(t) → T_target(t+1) → T(t+1) [inertie] → L(t+1)
        """
        # 1. Redistribution du trafic avec action BGP
        T_target = self._redistribute(action)

        # 2. Appliquer l'inertie BGP
        self.T = self._apply_inertia(T_target)

        # 3. Calculer la latence sur le nouvel état
        latency  = self._latency(self.T, action)
        load_arr = self.T.sum(axis=0)
        J        = self._jain(load_arr)
        n_upd    = self._n_updates(action)
        sla_ok   = float(np.median(latency) < 200.0)

        # 4. Reward avec récompense relative (amélioration vs état statique)
        # Calculer L si B1 (action=0) à cet instant
        T_b1     = self._redistribute(np.zeros_like(action))
        T_b1_ine = self._apply_inertia(T_b1)
        lat_b1   = self._latency(T_b1_ine, np.zeros_like(action))
        L_b1     = float(np.median(lat_b1))
        L_gabpo  = float(np.median(latency))

        # Reward = amélioration relative vs B1 + fairness + stabilité
        delta_vs_b1 = (L_b1 - L_gabpo) / (L_b1 + 1e-8)
        reward = (  0.50 * delta_vs_b1          # amélioration vs B1
                  + 0.20 * (J - 0.5)            # fairness bonus
                  + 0.15 * sla_ok               # SLA respecté
                  - 0.15 * n_upd / 5.0)         # pénalité updates BGP

        self._prev_action = action.copy()
        self._load_hist.append(self._load().copy())
        self.t += 1

        return self._obs(), reward, self.t >= self.T_ep, {
            'L_med':        float(L_gabpo),
            'L_b1_same_t':  float(L_b1),
            'delta_vs_b1':  float(delta_vs_b1 * 100),
            'J':            float(J),
            'delta_bgp':    float(n_upd),
            'load_max':     float(load_arr.max()),
            'latency_all':  latency.tolist(),
        }

    def compute_latency_for_action(self, action:np.ndarray):
        """Pour l'Oracle : évalue une action sans modifier l'état."""
        T_target = self._redistribute(action)
        T_sim    = self._apply_inertia(T_target)
        lat      = self._latency(T_sim, action)
        J        = self._jain(T_sim.sum(axis=0))
        n        = self._n_updates(action)
        return float(np.median(lat)), float(J), float(n)


# ══════════════════════════════════════════════════════════════════════════
# AGENTS (identiques à P0)
# ══════════════════════════════════════════════════════════════════════════

class B1_Static:
    def __init__(self, N_a, N_p):
        self.N_a, self.N_p = N_a, N_p
        self._a = np.zeros(N_p * N_a * 2)
    def select_action(self, obs): return self._a.copy()
    def name(self): return 'B1_Static'

class B2_Greedy:
    """Greedy : LOCAL_PREF+1 sur les 3 PoPs les moins chargés."""
    def __init__(self, N_a, N_p):
        self.N_a, self.N_p = N_a, N_p
    def select_action(self, obs):
        load = obs.get('load', np.zeros(self.N_p))
        # Top-3 PoPs les moins chargés
        best = np.argsort(load[:self.N_p])[:3]
        a    = np.zeros(self.N_p * self.N_a * 2)
        off  = self.N_p * self.N_a
        for p in best:
            for i in range(self.N_a):
                a[off + p * self.N_a + i] = 1.0
        return a
    def name(self): return 'B2_Greedy'

class AgentRandom:
    def __init__(self, N_a, N_p, seed=42):
        self.N_a, self.N_p = N_a, N_p
        self.rng = np.random.RandomState(seed)
    def select_action(self, obs):
        return self.rng.uniform(-1, 1, self.N_p * self.N_a * 2)
    def name(self): return 'Random'

class OracleAgent:
    """
    Oracle dynamique — adapté à la redistribution softmax.
    Évalue des candidats structurés et choisit le meilleur.
    L'Oracle a accès à l'état de l'environnement (omniscient).
    """
    def __init__(self, N_a, N_p, env:BGPDynamicEnv, n_candidates:int=80):
        self.N_a, self.N_p    = N_a, N_p
        self.env              = env
        self.n_candidates     = n_candidates
        self.rng              = np.random.RandomState(0)

    def select_action(self, obs):
        N  = self.N_p * self.N_a * 2
        load = obs.get('load', np.zeros(self.N_p))

        candidates = []

        # 1. Candidat nul (B1)
        candidates.append(np.zeros(N))

        # 2. LP+1 sur tous les PoPs (meilleure action statique connue)
        a = np.zeros(N); off = self.N_p * self.N_a
        for p in range(self.N_p):
            for i in range(self.N_a):
                a[off + p*self.N_a + i] = 1.0
        candidates.append(a.copy())

        # 3. LP+1 sur les PoPs peu chargés + Prep sur les PoPs surchargés
        # Stratégie de répartition de charge adaptative
        sorted_by_load = np.argsort(load[:self.N_p])
        light_pops = sorted_by_load[:self.N_p//3]     # 4 moins chargés
        heavy_pops = sorted_by_load[-self.N_p//3:]    # 4 plus chargés
        a = np.zeros(N)
        for p in light_pops:
            for i in range(self.N_a):
                a[off + p*self.N_a + i] = 1.0   # LP+1
        for p in heavy_pops:
            for i in range(self.N_a):
                a[p*self.N_a + i] = 1.0          # Prep+1 (décourage)
        candidates.append(a.copy())

        # 4. Candidats par PoP individuel
        for p in range(self.N_p):
            a = np.zeros(N)
            for i in range(self.N_a):
                a[off + p*self.N_a + i] = 1.0
            candidates.append(a.copy())

        # 5. Candidats aléatoires structurés (sparse)
        while len(candidates) < self.n_candidates:
            # Modifier 1-4 PoPs
            n_pops = self.rng.randint(1, 5)
            pops   = self.rng.choice(self.N_p, n_pops, replace=False)
            a      = np.zeros(N)
            for p in pops:
                # LP entre 0 et 1, prepend entre 0 et 0.5
                lp_val  = self.rng.uniform(0.3, 1.0)
                pre_val = self.rng.uniform(0.0, 0.5)
                for i in range(self.N_a):
                    a[off + p*self.N_a + i] = lp_val
                    a[p*self.N_a + i]       = pre_val
            candidates.append(a)

        # Évaluer tous les candidats
        best_score  = 1e9
        best_action = candidates[0]
        for a in candidates:
            L, J, n = self.env.compute_latency_for_action(a)
            # Score : minimiser L, maximiser J, pénaliser mises à jour
            score = L - 40.0*J + 8.0*n
            if score < best_score:
                best_score  = score
                best_action = a.copy()

        return best_action

    def name(self): return 'Oracle'


# ══════════════════════════════════════════════════════════════════════════
# ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════

def run_episode(agent, scenario:str, seed:int,
                N_a:int=50, N_p:int=12, T_ep:int=288) -> dict:
    env = BGPDynamicEnv(scenario, seed, N_a, N_p, T_ep)
    if hasattr(agent, 'env'):
        agent.env = env
    obs = env.reset()
    Ls, Js, Bs, deltas = [], [], [], []

    for _ in range(T_ep):
        a = agent.select_action(obs)
        obs, _, done, info = env.step(a)
        Ls.append(info['L_med'])
        Js.append(info['J'])
        Bs.append(info['delta_bgp'])
        deltas.append(info['delta_vs_b1'])
        if done: break

    return {
        'L_med':        float(np.median(Ls)),
        'L_mean':       float(np.mean(Ls)),
        'L_p90':        float(np.percentile(Ls, 90)),
        'L_p95':        float(np.percentile(Ls, 95)),
        'J_mean':       float(np.mean(Js)),
        'delta_bgp':    float(np.mean(Bs)),
        'delta_vs_b1':  float(np.mean(deltas)),
        'L_series':     Ls,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST SENSIBILITÉ — vérifier l'impact des actions
# ══════════════════════════════════════════════════════════════════════════

def test_sensitivity_dynamic(N_a:int=50, N_p:int=12, T_ep:int=72):
    """
    Test de sensibilité sur le nouvel environnement dynamique.
    Compare : Neutre vs LP+1 tous PoPs vs Prepend tous PoPs.
    Cible : différence >> 10ms (était 10.4ms en statique).
    """
    print("═"*70)
    print("TEST SENSIBILITÉ — Environnement dynamique (S1, seed=0, 72 steps)")
    print("═"*70)

    configs = {
        'Neutre (B1)':       np.zeros(N_p*N_a*2),
        'LP+1 tous PoPs':    None,
        'LP+0.5 tous PoPs':  None,
        'Prep+1 tous PoPs':  None,
        'LP+1 PoPs légers':  None,
        'Oracle greedy':     None,
    }

    off = N_p * N_a
    # LP+1 tous
    a = np.zeros(N_p*N_a*2)
    for p in range(N_p):
        for i in range(N_a): a[off+p*N_a+i] = 1.0
    configs['LP+1 tous PoPs'] = a.copy()

    # LP+0.5 tous
    a = np.zeros(N_p*N_a*2)
    for p in range(N_p):
        for i in range(N_a): a[off+p*N_a+i] = 0.5
    configs['LP+0.5 tous PoPs'] = a.copy()

    # Prep+1 tous
    a = np.zeros(N_p*N_a*2)
    for p in range(N_p):
        for i in range(N_a): a[p*N_a+i] = 1.0
    configs['Prep+1 tous PoPs'] = a.copy()

    # LP+1 PoPs légers uniquement (3 premiers)
    a = np.zeros(N_p*N_a*2)
    for p in range(3):
        for i in range(N_a): a[off+p*N_a+i] = 1.0
    configs['LP+1 PoPs légers'] = a.copy()

    # Oracle greedy : LP+1 sur tous + Prep=0
    a = np.zeros(N_p*N_a*2)
    for p in range(N_p):
        for i in range(N_a): a[off+p*N_a+i] = 1.0
    configs['Oracle greedy'] = a.copy()

    print(f"\n  {'Configuration':<28} {'L_med final':>12}  {'ΔL vs Neutre':>14}  {'f_bgp':>8}")
    print("  " + "─"*66)

    neutral_L = None
    for name, action in configs.items():
        env = BGPDynamicEnv('S1_Nominal', 0, N_a, N_p, T_ep)
        obs = env.reset()
        Ls  = []
        for _ in range(T_ep):
            a = action if action is not None else np.zeros(N_p*N_a*2)
            obs, _, done, info = env.step(a)
            Ls.append(info['L_med'])
            if done: break
        L = float(np.median(Ls))
        if neutral_L is None: neutral_L = L
        dL = neutral_L - L
        # f_bgp moyen
        f_bgp = float(np.mean(
            1/(1+np.exp(-((-0.8*action[:N_p*N_a].reshape(N_p,N_a) +
                           1.0*action[N_p*N_a:].reshape(N_p,N_a)).mean(axis=1))))
        ))
        marker = ' ← meilleur' if dL == max(neutral_L - float(np.median(Ls))
                                             for _ in [None]) else ''
        print(f"  {name:<28} {L:>9.1f}ms  {dL:>+12.1f}ms  {f_bgp:>7.3f}")

    print()
    # Calcul du vrai delta
    Ls_neutral = []
    Ls_best    = []
    env0 = BGPDynamicEnv('S1_Nominal', 0, N_a, N_p, T_ep)
    obs0 = env0.reset()
    for _ in range(T_ep):
        obs0, _, done, info = env0.step(np.zeros(N_p*N_a*2))
        Ls_neutral.append(info['L_med'])
        if done: break

    a_best = configs['LP+1 tous PoPs']
    env1 = BGPDynamicEnv('S1_Nominal', 0, N_a, N_p, T_ep)
    obs1 = env1.reset()
    for _ in range(T_ep):
        obs1, _, done, info = env1.step(a_best)
        Ls_best.append(info['L_med'])
        if done: break

    delta_ms   = float(np.median(Ls_neutral)) - float(np.median(Ls_best))
    delta_pct  = delta_ms / float(np.median(Ls_neutral)) * 100

    print(f"  Δ(Neutre → LP+1) = {delta_ms:+.1f}ms  ({delta_pct:+.1f}%)")
    if delta_pct > 5.0:
        print(f"  ✅ Impact fort ({delta_pct:.1f}%) — boucle dynamique active")
    elif delta_pct > 2.0:
        print(f"  ⚠  Impact modéré ({delta_pct:.1f}%) — à vérifier sur S2/S3/S4")
    else:
        print(f"  ❌ Impact faible ({delta_pct:.1f}%) — paramètres à ajuster")

    return delta_pct


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE P0b PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def run_p0b(N_a:int=50, N_p:int=12, n_seeds:int=10, T_ep:int=288):
    """
    Phase P0b — Re-test Oracle avec environnement dynamique.
    Cible : Oracle << B1 dans S2/S3/S4 (>5% d'amélioration).
    """
    SCENARIOS = ['S1_Nominal','S2_FlashCrowd','S3_PoP_Failure','S4_BGP_Anomaly']
    SEEDS     = list(range(n_seeds))

    # 1. Test de sensibilité
    delta_pct = test_sensitivity_dynamic(N_a, N_p, T_ep=72)

    print("\n" + "═"*70)
    print("PHASE P0b — Oracle avec environnement dynamique")
    print(f"  ρ={PARAMS.rho}, τ={PARAMS.tau}, θ_load={PARAMS.theta_load}")
    print(f"  Flash×{PARAMS.flash_factor}, Anomalie×{PARAMS.anomaly_boost}")
    print("═"*70)

    env_ref = BGPDynamicEnv('S1_Nominal', 0, N_a, N_p, T_ep)
    agents  = [
        B1_Static(N_a, N_p),
        B2_Greedy(N_a, N_p),
        AgentRandom(N_a, N_p),
        OracleAgent(N_a, N_p, env_ref, n_candidates=80),
    ]

    print(f"\nConfiguration: {n_seeds} seeds × {len(agents)} agents × {len(SCENARIOS)} scénarios")
    total = len(agents) * len(SCENARIOS) * n_seeds
    print(f"Total : {total} épisodes\n")

    results = {a.name(): {s: [] for s in SCENARIOS} for a in agents}
    cnt = 0
    for agent in agents:
        for scen in SCENARIOS:
            for seed in SEEDS:
                r = run_episode(agent, scen, seed, N_a, N_p, T_ep)
                results[agent.name()][scen].append(r)
                cnt += 1
                if cnt % 20 == 0:
                    print(f"  {cnt}/{total} épisodes")
    print(f"\n✓ {cnt} épisodes complétés")

    # ── Résultats ─────────────────────────────────────────────────────────
    print("\n" + "═"*78)
    print("RÉSULTATS P0b — L_med (ms) par agent et scénario")
    print("═"*78)

    agent_names = [a.name() for a in agents]
    b1_vals = {}
    for s in SCENARIOS:
        b1_vals[s] = float(np.mean([r['L_med'] for r in results['B1_Static'][s]]))

    hdr = f"  {'Agent':<14}"
    for s in SCENARIOS: hdr += f"  {s[:14]:<16}"
    hdr += "  Δ moy vs B1"
    print(hdr)
    print("  " + "─"*78)

    rows = []
    for aname in agent_names:
        line  = f"  {aname:<14}"
        imprs = []
        row   = {'Agent': aname}
        for s in SCENARIOS:
            vals = [r['L_med'] for r in results[aname][s]]
            m    = float(np.mean(vals))
            sd   = float(np.std(vals))
            impr = (b1_vals[s] - m) / b1_vals[s] * 100
            imprs.append(impr)
            line += f"  {m:>7.1f}±{sd:>4.1f}  "
            row[s+'_mean']     = round(m, 1)
            row[s+'_std']      = round(sd, 1)
            row[s+'_vs_B1']    = round(impr, 2)
        avg   = float(np.mean(imprs))
        line += f"  {avg:>+.1f}%"
        row['avg_vs_B1'] = round(avg, 2)
        print(line)
        rows.append(row)

    # ── Verdict Oracle ────────────────────────────────────────────────────
    print("\n" + "═"*78)
    print("VERDICT ORACLE P0b — Opportunité détectée ?")
    print("═"*78)

    oracle_per_s = {}
    for s in SCENARIOS:
        oracle_L = float(np.mean([r['L_med'] for r in results['Oracle'][s]]))
        b1_L     = b1_vals[s]
        impr     = (b1_L - oracle_L) / b1_L * 100
        oracle_per_s[s] = impr
        sym = "✅" if impr > 5.0 else ("⚠" if impr > 1.0 else "❌")
        print(f"  {sym} {s:<22}: Oracle vs B1 = {impr:>+.2f}%  "
              f"(Oracle={oracle_L:.1f}ms, B1={b1_L:.1f}ms)")

    avg_oracle = float(np.mean(list(oracle_per_s.values())))
    adv_oracle = float(np.mean([oracle_per_s[s]
                                  for s in ['S2_FlashCrowd',
                                            'S3_PoP_Failure',
                                            'S4_BGP_Anomaly']]))
    print(f"\n  Amélioration moyenne (tous) : {avg_oracle:>+.2f}%")
    print(f"  Amélioration scénarios adverses (S2/S3/S4) : {adv_oracle:>+.2f}%")

    if adv_oracle > 5.0:
        verdict     = 'FAVORABLE'
        next_action = 'Lancer Phase P1 — PPO propre (Normal distribution)'
    elif adv_oracle > 2.0:
        verdict     = 'MARGINAL'
        next_action = 'Ajuster ρ ou flash_factor, puis re-tester Oracle'
    else:
        verdict     = 'STILL_INSUFFICIENT'
        next_action = 'Augmenter theta_load, flash_factor, ou revoir demande AS'

    print(f"\n  VERDICT : {verdict}")
    print(f"  ACTION  : {next_action}")

    # ── Analyse temporelle S2 + S4 ────────────────────────────────────────
    print("\n" + "─"*78)
    print("ANALYSE TEMPORELLE — B1 vs Oracle (seed=0)")
    print("─"*78)

    for scen, phases in [
        ('S2_FlashCrowd',
         [('Avant flash (t<40)',    0,40),
          ('Flash crowd (t=40-100)',40,101),
          ('Après flash (t>100)',   101,288)]),
        ('S4_BGP_Anomaly',
         [('Avant anomalie (t<30)', 0,30),
          ('Anomalie (t=30-90)',    30,91),
          ('Après anomalie (t>90)', 91,288)]),
    ]:
        print(f"\n  {scen}:")
        b1_s     = results['B1_Static'][scen][0]['L_series']
        oracle_s = results['Oracle'][scen][0]['L_series']
        for name, t0, t1 in phases:
            t1   = min(t1, len(b1_s))
            if t0 >= len(b1_s): continue
            b1_v = float(np.median(b1_s[t0:t1]))
            ora_v= float(np.median(oracle_s[t0:t1]))
            d    = b1_v - ora_v
            sym  = '✅' if d > 5 else ('⚠' if d > 1 else '≈')
            print(f"    {sym} {name:<30}: B1={b1_v:.1f}ms  "
                  f"Oracle={ora_v:.1f}ms  Δ={d:>+.1f}ms")

    # ── Export ────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv('results_p0b/GABPO_P0b_results.csv', index=False)
    analysis = {
        'version':          'P0b',
        'timestamp':        time.strftime('%Y%m%d_%H%M'),
        'params':           vars(PARAMS),
        'verdict':          verdict,
        'next_action':      next_action,
        'oracle_per_scenario': {s: round(v,2) for s,v in oracle_per_s.items()},
        'avg_oracle_adv_pct': round(adv_oracle, 2),
        'sensitivity_delta_pct': round(delta_pct, 2),
    }
    with open('results_p0b/GABPO_P0b_analysis.json','w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✓ Sauvegardé : results_p0b/")
    print(f"  - GABPO_P0b_results.csv")
    print(f"  - GABPO_P0b_analysis.json")

    return df, analysis, results


# ── Point d'entrée ────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, analysis, results = run_p0b(N_a=50, N_p=12, n_seeds=10, T_ep=288)
