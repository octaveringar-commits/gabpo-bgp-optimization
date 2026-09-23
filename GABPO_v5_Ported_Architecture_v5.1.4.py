# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Architecture v5 "Ported" : HCGA + PPO discret + Temporel réel   ║
# ║                                                                          ║
# ║  Objectif de ce script                                                  ║
# ║  ------------------------                                               ║
# ║  Réconcilier ce que le manuscrit décrit (GAT + HCGA + représentation    ║
# ║  spatio-temporelle) avec ce qui a réellement produit les résultats      ║
# ║  verrouillés (HierarchicalPolicy = MLP 2 couches, cf. bc_phase_D.pt).   ║
# ║                                                                          ║
# ║  Décision utilisateur (verrouillée) : "Porter l'architecture v4         ║
# ║  corrigée dans l'environnement P0b" — c'est-à-dire :                    ║
# ║    - corriger les 4 bugs identifiés dans GABPO_PPO_v4.py                ║
# ║        (1) log-prob PPO invalide (continu, sans distribution)           ║
# ║        (2) masking d'action state-independent (M_valid[int(n*0.8):]=0) ║
# ║        (3) séquence temporelle fake (.expand() d'un seul embedding)     ║
# ║        (4) features proxy non qualifiées (np.linspace CAIDA/RPKI/RTT)   ║
# ║    - MAIS faire tourner le résultat DANS BGPDynamicEnv (P0b), avec      ║
# ║      Eq.1-3, la même reward, et le même protocole stats (Student-t      ║
# ║      95% CI, Wilcoxon, Bonferroni α'=0.0125) que Tables I/II/PPO,       ║
# ║      pour que le résultat reste directement comparable — rien de ce     ║
# ║      qui est déjà verrouillé (Tables I/II, TE1-3, audit NO-OP) n'est    ║
# ║      perdu ou remis en cause par ce script.                             ║
# ║                                                                          ║
# ║  Ce script NE remplace PAS bc_phase_D.pt / HierarchicalPolicy. Il       ║
# ║  ajoute une variante candidate (v5) à évaluer EN PLUS, avec le même     ║
# ║  protocole, avant toute décision sur ce qui sera décrit au §IV.         ║
# ║                                                                          ║
# ║  Prérequis session (comme les autres scripts du dépôt) :                ║
# ║    exec(open('GABPO_P0b_DynamicEnv.py').read())      # env + Eq.1-3     ║
# ║    exec(open('GABPO_P1b_BC.py').read())               # HierarchicalAction,
# ║                                                        # extract_features,
# ║                                                        # HierarchicalPolicy,
# ║                                                        # run_episode, B1_Static
# ║    exec(open('GABPO_v5_Ported_Architecture.py').read())
# ║    df = run_v5_pilot(quick=True)     # 1 seed, test rapide d'intégrité  ║
# ║    df = run_v5_full(quick=False)     # 10 seeds × 5 scénarios           ║
# ║                                                                          ║
# ║  ⚠ POINTS À VALIDER PAR L'UTILISATEUR AVANT LA CAMPAGNE COMPLÈTE        ║
# ║  (voir docstrings marquées "DÉCISION NOUVELLE — NON DÉRIVÉE DU CODE")   ║
# ║  ------------------------------------------------------------------     ║
# ║  D1. Module temporel : LSTMCell + troncature du gradient sur 1 pas      ║
# ║      (BPTT tronqué à profondeur 1). Choix standard en RL récurrent      ║
# ║      (cf. Kapturowski et al. 2019, "R2D2"), mais NE FIGURE PAS dans     ║
# ║      le dépôt existant — c'est une conception nouvelle, pas une         ║
# ║      extraction de code déjà validé.                                    ║
# ║  D2. Warm-start depuis bc_phase_D.pt : SEULES les 4 têtes hiérarchiques ║
# ║      (noop/pop/type/amp, Linear(64,·)) sont transférables — le tronc    ║
# ║      MLP (encoder.0/2) n'a pas d'équivalent dans l'encodeur HCGA et     ║
# ║      n'est donc PAS chargé. C'est un warm-start partiel, pas un         ║
# ║      équivalent du warm-start complet utilisé pour Table PPO.           ║
# ║  D3. Features AS (13 dims) et PoP (8 dims) : reprises telles quelles    ║
# ║      de extract_features (P0b patché) — AUCUNE feature proxy v4         ║
# ║      (np.linspace) n'est réutilisée ici.                                ║
# ║                                                                          ║
# ║  === v5.1 — corrections apportées après revue externe =================║
# ║  R1 (🔴 bloquant) : evaluate_logprob() omettait log P(NOOP=0|s) dans la ║
# ║      log-prob jointe dès que le NO-OP était actif pendant le rollout,   ║
# ║      rendant le ratio PPO r_t=π_θ/π_θold faux. Corrigé : la distribution║
# ║      jointe utilisée à la ré-évaluation reproduit EXACTEMENT celle du   ║
# ║      rollout, y compris quand NO-OP était désactivé pour ce pas (auquel ║
# ║      cas lp_noop=0 des deux côtés, cohérent).                           ║
# ║  R2 : self.noop_weight supprimé — ne pondérait rien (ni logits, ni      ║
# ║      entropie, ni loss) et le commentaire "NO-OP progressif" était donc ║
# ║      trompeur. Remplacé par un simple booléen noop_enabled=(epoch>=20)  ║
# ║      transmis explicitement à chaque appel.                             ║
# ║  R3 : Bonferroni α'=0.0125 ne s'applique qu'à la famille confirmatoire  ║
# ║      S1-S4 (4 comparaisons) ; S5 (OOD) est un test de transfert séparé, ║
# ║      rapporté sans correction de famille — cohérent avec le protocole   ║
# ║      déjà verrouillé pour Table I/II/PPO.                               ║
# ║  R4 : train_scenarios ne contient plus S2_FlashCrowd — seuls S3/S4      ║
# ║      (50/50) servent à l'entraînement, comme le protocole BC/PPO        ║
# ║      historique, pour que S1/S2 restent des scénarios de test de        ║
# ║      transfert non vus, et que la comparaison à B1/BC reste valide.     ║
# ║  R5 : run_v5_full() évalue maintenant trois agents sur les MÊMES        ║
# ║      graines/scénarios : B1_Static, BC (HierarchicalPolicy chargée      ║
# ║      depuis bc_phase_D.pt — les résultats déjà verrouillés), et v5.     ║
# ║      Sans ce contrôle, un gain de v5 sur B1 ne dit rien sur un gain     ║
# ║      par rapport à l'architecture qui a produit les résultats actuels. ║
# ║  R6 : n_epochs=200 est reformulé partout en "200 PPO epochs / rollout  ║
# ║      updates" (~72 pas chacun, ≈14 400 transitions au total) — CE       ║
# ║      N'EST PAS équivalent à 200 epochs d'entraînement BC supervisé sur ║
# ║      un jeu de paires Oracle fixe ; les deux ne sont pas comparables    ║
# ║      terme à terme.                                                     ║
# ║                                                                          ║
# ║  === v5.1.3 — Palier 3 (test empirique post-entraînement) ==============║
# ║  R7 : std(A_cross) mesuré après 20 epochs PPO (pilote 1 seed) = 4,6e-6, ║
# ║      PAS mesurablement supérieur à l'initialisation (déjà quasi-nul).   ║
# ║      Cause identifiée : W_Q/W_K/W_V/W_O/gate/enc_as/enc_pop partent     ║
# ║      d'une init ALÉATOIRE (aucun warm-start, cf. D2) mais étaient       ║
# ║      entraînés au même lr=1e-5 que les têtes déjà pré-entraînées — trop ║
# ║      lent de plusieurs ordres de grandeur pour apprendre un mécanisme   ║
# ║      d'attention depuis rien. Corrigé par policy.make_optimizer() :     ║
# ║      lr=1e-5 conservé UNIQUEMENT pour les 4 têtes warm-startées,        ║
# ║      lr=3e-4 (nouveau, à valider empiriquement — pas dérivé du dépôt)   ║
# ║      pour tout le reste. Pratique standard de fine-tuning différentiel, ║
# ║      ne change AUCUN calcul déjà audité (HCGA, PPO discret, Eq.1-3) —   ║
# ║      seule la configuration de l'optimiseur change.                     ║
# ║      → Re-vérifier le Palier 3 (std(A_cross) avant/après) avec ce       ║
# ║      nouvel optimiseur avant de rouvrir la porte de décision Palier 4.  ║
# ║                                                                          ║
# ║  === v5.1.4 — Palier 3 bis (R7 insuffisant à lui seul) ==================║
# ║  R8 : re-testé avec lr_new=3e-4 (R7) — std(A_cross) après 20 epochs =   ║
# ║      5,6e-6, quasi inchangé vs 4,6e-6 sans R7. Cause réelle identifiée : ║
# ║      ctx=W_O(A@V) a une échelle ~30x plus petite que h_pop (moyenne     ║
# ║      d'attention quasi-uniforme sur N_a=50 → réduction de variance      ║
# ║      ~1/√50, puis gate sigmoid≈0.5). LayerNorm(h_pop + gate·ctx) est    ║
# ║      alors dominé à >95% par h_pop, écrasant le gradient utile vers     ║
# ║      W_Q/W_K/W_V quel que soit le lr — ce n'était donc pas (seulement)  ║
# ║      un problème de vitesse d'apprentissage (R7) mais de déséquilibre   ║
# ║      d'échelle entre les deux branches à fusionner (R8). Corrigé par    ║
# ║      self.ln_ctx = nn.LayerNorm(d) appliqué à ctx SEUL avant le gating  ║
# ║      — pattern standard de fusion à deux branches (pré-normalisation    ║
# ║      par branche). À re-tester au Palier 3 avant de rouvrir Palier 4.   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pandas as pd
from scipy import stats
import json, time, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('results_v5', exist_ok=True)
os.makedirs('checkpoints_v5', exist_ok=True)

V5_SCRIPT_VERSION = "v5.1.4 (R8: LayerNorm(ctx) avant mélange avec h_pop — corrige l'écrasement du gradient HCGA)"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO v5 — HCGA + PPO discret + Temporel réel       ║")
print("╚══════════════════════════════════════════════════════╝")
print(f"Version du script : {V5_SCRIPT_VERSION}")
print(f"Device: {DEVICE}")

# ── Vérification dépendances (mêmes garde-fous que les autres scripts) ─────
try:
    _ = BGPDynamicEnv, HierarchicalAction, extract_features
    _ = HierarchicalPolicy, run_episode, B1_Static
    print("✓ P0b + BC chargés (env, extract_features, HierarchicalPolicy, B1_Static)")
except NameError as e:
    raise RuntimeError(
        f"Exécuter GABPO_P0b_DynamicEnv.py puis GABPO_P1b_BC.py d'abord : {e}")

BC_CHECKPOINT = 'checkpoints_bc/bc_phase_D.pt'
if not os.path.exists(BC_CHECKPOINT):
    print(f"⚠ Checkpoint BC introuvable ({BC_CHECKPOINT}) — le warm-start "
          f"partiel des têtes (D2) sera sauté ; v5 démarrera poids aléatoires.")


# ══════════════════════════════════════════════════════════════════════════
# 1. POLITIQUE v5 : enc_as/enc_pop + HCGA réel + mémoire temporelle réelle
#    + têtes hiérarchiques + critic PPO + log-prob discret Categorical
# ══════════════════════════════════════════════════════════════════════════

class HierarchicalPolicy_HCGA_Temporal(nn.Module):
    """
    Politique "portée" : réunit dans une seule classe les composantes déjà
    validées séparément ailleurs dans le dépôt, plus un module temporel réel
    (nouveau, cf. D1) :

      - enc_as / enc_pop / HCGA (_hcga)      : repris À L'IDENTIQUE de
        HierarchicalPolicy_HCGA (GABPO_HCGA_SanityCheck.py, déjà audité par
        run_hcga_sanity_check — x_as réellement utilisé, A_cross non-uniforme
        et dépendant de l'état).
      - têtes hiérarchiques (noop/pop/type/amp) : mêmes dimensions que
        HierarchicalPolicy ET HierarchicalPolicy_HCGA (Linear(64,·)), donc
        transférables depuis un checkpoint existant si les formes coïncident.
      - critic_head + get_action_and_logprob + evaluate_logprob : repris du
        schéma de PPOHierarchicalPolicy (GABPO_PPO_WarmStart.py) — dist.
        Categorical discrète sur chaque tête, PAS le log-prob continu
        invalide de GABPOAgent (v4).
      - mémoire temporelle (self.lstm) : NOUVEAU (D1). Remplace le bug v4
        (embedding unique répété 3× via .expand()) par une vraie récurrence
        LSTMCell sur l'état spatial (h_final) observé à chaque pas réel de
        l'épisode. Le gradient est tronqué à profondeur 1 (BPTT(1)) : l'état
        caché entrant (h_in, c_in) est traité comme une constante détachée
        pendant la ré-évaluation PPO, seul le pas courant est différencié —
        pratique standard en RL récurrent, mais à valider empiriquement ici.
    """
    def __init__(self, N_a: int = 50, N_p: int = 12, d: int = 32, d_state: int = 64):
        super().__init__()
        self.N_a, self.N_p, self.d, self.d_state = N_a, N_p, d, d_state
        self.ha = HierarchicalAction(N_a, N_p)

        # --- Encodeurs séparés AS / PoP (identique à HierarchicalPolicy_HCGA) ---
        self.enc_as  = nn.Sequential(nn.Linear(13, 64), nn.ELU(), nn.Linear(64, d))
        self.enc_pop = nn.Sequential(nn.Linear(8,  64), nn.ELU(), nn.Linear(64, d))

        # --- HCGA réel (identique à HierarchicalPolicy_HCGA / GABPOAgent._hcga) ---
        self.W_Q  = nn.Linear(d, d, bias=False)
        self.W_K  = nn.Linear(d, d, bias=False)
        self.W_V  = nn.Linear(d, d, bias=False)
        self.W_O  = nn.Linear(d, d, bias=False)
        self.gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.ln_ctx = nn.LayerNorm(d)   # R8 : normalise ctx AVANT le mélange
        self.ln   = nn.LayerNorm(d)
        self.post = nn.Sequential(nn.Linear(d, d_state), nn.ReLU())

        # --- Mémoire temporelle réelle (NOUVEAU — cf. D1) -----------------
        self.temporal = nn.LSTMCell(d_state, d_state)

        # --- Têtes hiérarchiques — mêmes dimensions que HierarchicalPolicy ---
        self.noop_head = nn.Linear(d_state, 2)
        self.pop_head  = nn.Linear(d_state, N_p)
        self.type_head = nn.Linear(d_state, 4)
        self.amp_head  = nn.Linear(d_state, 3)

        # --- Critic PPO (identique en esprit à PPOHierarchicalPolicy) ------
        self.critic_head = nn.Sequential(
            nn.Linear(d_state, 64), nn.Tanh(), nn.Linear(64, 1))
        for m in self.critic_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in self.critic_head.modules():
                nn.init.orthogonal_(m.weight, 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.A_cross = None      # dernière carte d'attention (diagnostic)
        # NB (R2) : pas de self.noop_weight. Le NO-OP est activé/désactivé
        # uniquement via le paramètre explicite noop_enabled, transmis à
        # chaque appel de get_action_and_logprob / evaluate_logprob /
        # select_action — jamais par un état interne implicite.

    # -- état caché initial (début d'épisode) --------------------------------
    def init_hidden(self):
        h0 = torch.zeros(1, self.d_state, device=DEVICE)
        c0 = torch.zeros(1, self.d_state, device=DEVICE)
        return h0, c0

    def _spatial_state(self, obs):
        """enc_as/enc_pop + HCGA -> h_final (1, d_state). Identique en calcul
        à HierarchicalPolicy_HCGA.forward, mais renvoie aussi les features
        brutes pour permettre la ré-évaluation PPO."""
        x_as, x_pop = extract_features(obs, self.N_a, self.N_p)
        x_as, x_pop = x_as.to(DEVICE), x_pop.to(DEVICE)
        h_as  = self.enc_as(x_as)
        h_pop = self.enc_pop(x_pop)
        h_cross = self._hcga(h_as, h_pop)
        h_final = self.post(h_cross.mean(0)).unsqueeze(0)   # (1, d_state)
        return h_final, x_as, x_pop

    def _hcga(self, h_as, h_pop):
        Q = self.W_Q(h_pop)
        K = self.W_K(h_as)
        V = self.W_V(h_as)
        raw_scores = Q @ K.T / (self.d ** 0.5)
        scores = raw_scores.clamp(-20, 20)
        A = F.softmax(scores, dim=-1)
        self.A_cross = A.detach().cpu().numpy()
        # Diagnostic (cf. Test 3 du pilote) : si raw_scores sature massivement
        # le clamp(-20,20), softmax(clamp(x)) devient artificiellement
        # uniforme même si le calcul HCGA est correct — à distinguer d'un vrai
        # bug (ex. x_as constant entre AS, ou script non rechargé).
        with torch.no_grad():
            rs = raw_scores.detach()
            self._hcga_diag = {
                'raw_min': rs.min().item(), 'raw_max': rs.max().item(),
                'frac_clamped': (rs.abs() > 20).float().mean().item(),
            }
        ctx  = self.W_O(A @ V)
        # R8 : ctx est normalisé AVANT le gating/mélange avec h_pop. Sans ça,
        # l'attention quasi-uniforme moyenne V sur N_a=50 AS (réduction de
        # variance ~1/√50) puis le sigmoid du gate (~0.5) donnent à ctx une
        # échelle ~30x plus petite que h_pop — LayerNorm(h_pop + gate·ctx) est
        # alors dominé à >95% par h_pop, et le gradient utile vers
        # W_Q/W_K/W_V reste écrasé quel que soit le lr (constaté empiriquement
        # au Palier 3 : std(A_cross) ne bougeait presque pas même à lr=3e-4).
        ctx = self.ln_ctx(ctx)
        gate = self.gate(torch.cat([h_pop, ctx], dim=-1))
        return self.ln(h_pop + gate * ctx)

    def get_state(self, obs, h_in, c_in):
        """Combine l'état spatial courant (HCGA) avec la mémoire temporelle
        réelle (LSTMCell), en résiduel — remplace HierarchicalPolicy.encoder
        et PPOHierarchicalPolicy.get_state."""
        h_final, x_as, x_pop = self._spatial_state(obs)
        h_out, c_out = self.temporal(h_final, (h_in, c_in))
        final_state = (h_final + h_out).squeeze(0)          # (d_state,)
        return final_state, x_as, x_pop, h_out, c_out

    # -- échantillonnage d'action + log-prob (discret, Categorical) ---------
    def get_action_and_logprob(self, obs, h_in, c_in, noop_enabled=False):
        state, x_as, x_pop, h_out, c_out = self.get_state(obs, h_in, c_in)

        noop_l, pop_l, type_l, amp_l = (
            self.noop_head(state), self.pop_head(state),
            self.type_head(state), self.amp_head(state))
        value = self.critic_head(state).squeeze()

        if noop_enabled:
            noop_dist = Categorical(logits=noop_l)
            noop_act  = noop_dist.sample()
            lp_noop   = noop_dist.log_prob(noop_act)
        else:
            noop_act  = torch.tensor(0)
            lp_noop   = torch.tensor(0.0, device=DEVICE)

        if noop_act.item() == 1:
            action_bgp = np.zeros(self.N_p * self.N_a * 2, dtype=np.float32)
            log_prob   = lp_noop
            decisions  = {'noop': 1, 'pop': 0, 'type': 0, 'amp': 0,
                          'noop_enabled': bool(noop_enabled)}
        else:
            pop_dist  = Categorical(logits=pop_l)
            type_dist = Categorical(logits=type_l)
            amp_dist  = Categorical(logits=amp_l)
            pop_act, type_act, amp_act = pop_dist.sample(), type_dist.sample(), amp_dist.sample()
            log_prob = (lp_noop + pop_dist.log_prob(pop_act)
                        + type_dist.log_prob(type_act) + amp_dist.log_prob(amp_act))
            decisions = {'noop': 0, 'pop': pop_act.item(),
                         'type': type_act.item(), 'amp': amp_act.item(),
                         'noop_enabled': bool(noop_enabled)}
            action_bgp = self.ha.decode(np.concatenate([
                [0.0],
                np.eye(self.N_p)[pop_act.item()],
                np.eye(4)[type_act.item()],
                np.eye(3)[amp_act.item()],
            ]).astype(np.float32))

        # h_out/c_out détachés : ils deviennent la mémoire d'entrée du pas
        # suivant (BPTT tronqué à profondeur 1, cf. D1).
        return action_bgp, log_prob, value, decisions, h_out.detach(), c_out.detach()

    # -- ré-évaluation pour l'update PPO (mêmes h_in/c_in stockés, détachés) -
    def evaluate_logprob(self, obs, h_in, c_in, decisions):
        """Reconstruit EXACTEMENT la distribution jointe utilisée lors du
        rollout (cf. get_action_and_logprob), y compris la contribution du
        NO-OP — corrigé suite revue externe (R1) : omettre log P(NOOP=0|s)
        quand noop_enabled=True fausse le ratio PPO r_t=π_θ/π_θold pour
        toute action non-NO-OP après activation du NO-OP."""
        state, _, _, _, _ = self.get_state(obs, h_in, c_in)
        noop_l, pop_l, type_l, amp_l = (
            self.noop_head(state), self.pop_head(state),
            self.type_head(state), self.amp_head(state))
        value = self.critic_head(state).squeeze()

        noop_was_enabled = decisions.get('noop_enabled', False)
        if noop_was_enabled:
            noop_dist = Categorical(logits=noop_l)
            lp_noop  = noop_dist.log_prob(torch.tensor(decisions['noop'], device=DEVICE))
            ent_noop = noop_dist.entropy()
        else:
            # Symétrique du rollout : NO-OP désactivé => lp_noop=0 des deux
            # côtés (pas une distribution dégénérée qu'on évalue quand même).
            lp_noop  = torch.tensor(0.0, device=DEVICE)
            ent_noop = torch.tensor(0.0, device=DEVICE)

        if decisions['noop'] == 1:
            lp  = lp_noop
            ent = ent_noop
        else:
            pop_dist  = Categorical(logits=pop_l)
            type_dist = Categorical(logits=type_l)
            amp_dist  = Categorical(logits=amp_l)
            lp = (lp_noop
                  + pop_dist.log_prob(torch.tensor(decisions['pop'], device=DEVICE))
                  + type_dist.log_prob(torch.tensor(decisions['type'], device=DEVICE))
                  + amp_dist.log_prob(torch.tensor(decisions['amp'], device=DEVICE)))
            ent = ent_noop + pop_dist.entropy() + type_dist.entropy() + amp_dist.entropy()
        return lp, value, ent

    def select_action(self, obs, h_in=None, c_in=None, noop_enabled=True):
        """Compatible avec run_episode (évaluation gloutonne, sans PPO)."""
        if h_in is None or c_in is None:
            h_in, c_in = self.init_hidden()
        with torch.no_grad():
            action, _, _, _, h_out, c_out = self.get_action_and_logprob(
                obs, h_in, c_in, noop_enabled=noop_enabled)
        # NB : run_episode() du dépôt ne transmet pas d'état récurrent entre
        # appels — pour une évaluation multi-pas correcte, utiliser
        # run_episode_v5() ci-dessous plutôt que run_episode() générique.
        return action

    def make_optimizer(self, lr_heads=1e-5, lr_new=3e-4, weight_decay=1e-5):
        """Optimiseur à taux d'apprentissage différenciés (correctif R7,
        suite au Palier 3 : std(A_cross) ne bougeait pas après 20 epochs à
        lr=1e-5 uniforme).

        Les 4 têtes hiérarchiques sont warm-startées depuis bc_phase_D.pt
        (poids déjà entraînés) : lr_heads=1e-5, du fine-tuning conservateur,
        comme PPOHierarchicalPolicy.

        Tout le reste (enc_as, enc_pop, W_Q/K/V/O, gate, ln, post, temporal,
        critic_head) part d'une initialisation ALÉATOIRE et doit apprendre
        depuis rien : lr_new plus élevé (3e-4 par défaut — pratique standard
        de fine-tuning différentiel, pas une valeur dérivée du dépôt, à
        surveiller comme toute nouvelle décision de ce script)."""
        head_params, new_params = [], []
        head_modules = {self.noop_head, self.pop_head, self.type_head, self.amp_head}
        for name, module in self.named_children():
            target = head_params if module in head_modules else new_params
            target.extend(p for p in module.parameters() if p.requires_grad)
        return torch.optim.Adam([
            {'params': head_params, 'lr': lr_heads},
            {'params': new_params,  'lr': lr_new},
        ], eps=1e-5, weight_decay=weight_decay)

    def load_bc_heads(self, checkpoint_path=BC_CHECKPOINT):
        """Warm-start PARTIEL (cf. D2) : charge uniquement les 4 têtes
        hiérarchiques depuis bc_phase_D.pt. Le tronc (encoder MLP) n'a pas
        d'équivalent ici et n'est jamais chargé."""
        if not os.path.exists(checkpoint_path):
            print("  ⚠ Pas de checkpoint BC — têtes initialisées aléatoirement.")
            return
        bc_state = torch.load(checkpoint_path, map_location=DEVICE)
        head_keys = ['noop_head.weight', 'noop_head.bias',
                     'pop_head.weight',  'pop_head.bias',
                     'type_head.weight', 'type_head.bias',
                     'amp_head.weight',  'amp_head.bias']
        own_state = self.state_dict()
        loaded = []
        for k in head_keys:
            if k in bc_state and bc_state[k].shape == own_state[k].shape:
                own_state[k] = bc_state[k].clone()
                loaded.append(k)
        self.load_state_dict(own_state)
        print(f"  ✓ Warm-start partiel : {len(loaded)}/{len(head_keys)} "
              f"tenseurs de têtes chargés depuis {checkpoint_path}")


# ══════════════════════════════════════════════════════════════════════════
# 2. ÉVALUATION (épisode complet, état récurrent correctement propagé)
# ══════════════════════════════════════════════════════════════════════════

def run_episode_v5(policy, scenario, seed, N_a=50, N_p=12, T_ep=288, noop_enabled=True):
    """Équivalent de run_episode() du dépôt (même clé info['L_med'], même
    agrégation par médiane — cf. GABPO_P0b_DynamicEnv.py:451-477), mais
    propage explicitement (h,c) entre les pas — nécessaire car
    policy.select_action() seul ne peut pas porter la mémoire d'un appel à
    l'autre.

    CORRECTIF (v5.1 bis) : la première version de cette fonction cherchait
    info['L_med_step'] ou info['latency'], des clés qui n'existent pas dans
    le dict retourné par BGPDynamicEnv.step() — la vraie clé, utilisée par
    run_episode() lui-même, est info['L_med']. Résultat : 'latencies'
    restait toujours vide et L_med valait NaN pour CHAQUE épisode v5 (voir
    le CSV de la première campagne : v5_Lmed vide sur les 50 lignes, alors
    que B1_Lmed et BC_Lmed — calculés via l'ancien run_episode() correct —
    étaient bien peuplés). Corrigé en reprenant exactement la clé et
    l'agrégation de run_episode()."""
    env = BGPDynamicEnv(scenario, seed, N_a, N_p, T_ep)
    obs = env.reset()
    h, c = policy.init_hidden()
    Ls, done = [], False
    policy.eval()
    with torch.no_grad():
        for _ in range(T_ep):
            action, _, _, _, h, c = policy.get_action_and_logprob(
                obs, h, c, noop_enabled=noop_enabled)
            obs, reward, done, info = env.step(action)
            Ls.append(info['L_med'])
            if done:
                break
    L_med = float(np.median(Ls)) if Ls else float('nan')
    return {'L_med': L_med, 'n_steps': len(Ls)}


# ══════════════════════════════════════════════════════════════════════════
# 3. COLLECTE ROLLOUT (état récurrent stocké et détaché à chaque pas)
# ══════════════════════════════════════════════════════════════════════════

def collect_rollout_v5(policy, scenario, seed, N_a=50, N_p=12, T_ep=72,
                        gamma=0.99, lam=0.95, noop_enabled=False):
    env = BGPDynamicEnv(scenario, seed, N_a, N_p, T_ep)
    obs = env.reset()
    h, c = policy.init_hidden()

    obss, h_ins, c_ins, decisions = [], [], [], []
    log_probs_old, rewards, values, dones = [], [], [], []

    policy.eval()
    with torch.no_grad():
        for _ in range(T_ep):
            h_in, c_in = h, c
            action, lp, val, dec, h, c = policy.get_action_and_logprob(
                obs, h_in, c_in, noop_enabled=noop_enabled)
            next_obs, reward, done, info = env.step(action)
            reward = float(np.clip(reward, -5.0, 5.0))

            obss.append(obs)
            h_ins.append(h_in); c_ins.append(c_in)
            decisions.append(dec)
            log_probs_old.append(lp.item())
            rewards.append(reward)
            values.append(val.item())
            dones.append(float(done))

            obs = next_obs
            if done:
                break

    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    with torch.no_grad():
        state_last, _, _, _, _ = policy.get_state(obs, h, c)
        v_last = policy.critic_head(state_last).squeeze().item()
    next_v = v_last if not dones[-1] else 0.0

    gae = 0.0
    for t in reversed(range(T)):
        nv    = values[t + 1] if t + 1 < T else next_v
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    ret = adv + np.array(values, dtype=np.float32)
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return {'obss': obss, 'h_ins': h_ins, 'c_ins': c_ins,
            'decisions': decisions,
            'lp_old': np.array(log_probs_old, dtype=np.float32),
            'returns': ret, 'advantages': adv,
            'rewards': np.array(rewards, dtype=np.float32)}


# ══════════════════════════════════════════════════════════════════════════
# 4. UPDATE PPO (même formule que ppo_update_hier — clip, GAE, entropie)
# ══════════════════════════════════════════════════════════════════════════

def ppo_update_v5(policy, optimizer, rollout, eps_clip=0.2, n_epochs=4,
                   batch_size=64, vf_coef=0.5, ent_coef=0.001):
    T = len(rollout['obss'])
    indices = np.arange(T)
    metrics = {'actor': [], 'critic': [], 'entropy': [], 'kl': [], 'clip_frac': []}

    policy.train()
    for _ in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, T, batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 2:
                continue

            lp_news, vals, ents, valid_idx = [], [], [], []
            for i in idx:
                try:
                    lp, v, e = policy.evaluate_logprob(
                        rollout['obss'][i], rollout['h_ins'][i],
                        rollout['c_ins'][i], rollout['decisions'][i])
                    if torch.isnan(lp).any() or torch.isinf(lp).any():
                        continue
                    lp_news.append(lp); vals.append(v); ents.append(e)
                    valid_idx.append(i)
                except Exception:
                    continue
            if len(valid_idx) < 2:
                continue

            lp_new = torch.stack(lp_news)
            val_t  = torch.stack(vals)
            ent_t  = torch.stack(ents).mean()
            lp_old = torch.FloatTensor(rollout['lp_old'][valid_idx]).to(DEVICE)
            ret_t  = torch.FloatTensor(rollout['returns'][valid_idx]).to(DEVICE)
            adv_t  = torch.FloatTensor(rollout['advantages'][valid_idx]).to(DEVICE)

            log_diff = (lp_new - lp_old).clamp(-5, 5)
            ratio = torch.exp(log_diff)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv_t
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = F.smooth_l1_loss(val_t, ret_t)
            loss = actor_loss + vf_coef * critic_loss - ent_coef * ent_t

            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            for p in policy.parameters():
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            optimizer.step()

            with torch.no_grad():
                kl = ((lp_old - lp_new) ** 2).mean().item() / 2
                clip_frac = ((ratio - 1).abs() > eps_clip).float().mean().item()
            metrics['actor'].append(actor_loss.item())
            metrics['critic'].append(critic_loss.item())
            metrics['entropy'].append(ent_t.item())
            metrics['kl'].append(kl)
            metrics['clip_frac'].append(clip_frac)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


# ══════════════════════════════════════════════════════════════════════════
# 5. PILOTE D'INTÉGRITÉ — À LANCER EN PREMIER (1 seed, quelques epochs)
#    Vérifie que la classe tourne de bout en bout avant toute campagne.
# ══════════════════════════════════════════════════════════════════════════

def run_v5_pilot(quick=True, N_a=50, N_p=12, warm_start=True):
    print("\n[GABPO v5 — pilote d'intégrité]")
    torch.manual_seed(0); np.random.seed(0)

    policy = HierarchicalPolicy_HCGA_Temporal(N_a, N_p).to(DEVICE)
    if warm_start:
        policy.load_bc_heads()

    optimizer = policy.make_optimizer()  # R7 : lr différenciés têtes/nouveaux modules

    print("  Test 1 — rollout + update sur S1_Nominal (8 pas)...")
    rollout = collect_rollout_v5(policy, 'S1_Nominal', 0, N_a, N_p, T_ep=8, noop_enabled=False)
    m = ppo_update_v5(policy, optimizer, rollout, batch_size=8)
    print(f"    actor={m['actor']:+.4f} critic={m['critic']:.4f} "
          f"entropy={m['entropy']:.4f} kl={m['kl']:.4f}")

    print("  Test 2 — évaluation épisode complet (mémoire temporelle propagée)...")
    r = run_episode_v5(policy, 'S1_Nominal', 1, N_a, N_p, T_ep=40)
    print(f"    L_med={r['L_med']:.1f}ms sur {r['n_steps']} pas")

    print("  Test 3 — A_cross non-uniforme (HCGA réellement branché)...")
    uniform_val = 1.0 / N_a
    is_uniform = np.allclose(policy.A_cross, uniform_val, atol=1e-4)
    diag = getattr(policy, '_hcga_diag', None)
    print(f"    {'✗ SUSPECT' if is_uniform else '✓'} std(A_cross)={policy.A_cross.std():.4f}")
    if is_uniform and diag is not None:
        print(f"    diag scores bruts (avant clamp) : min={diag['raw_min']:.2f} "
              f"max={diag['raw_max']:.2f} frac_saturée={diag['frac_clamped']:.2%}")
        if diag['frac_clamped'] > 0.5:
            print("    → cause probable : scores HCGA saturent le clamp(-20,20), "
                  "softmax(clamp) devient artificiellement uniforme malgré un "
                  "calcul HCGA correct. Vérifier l'échelle des poids "
                  "orthogonal_(gain=0.5) / la normalisation des features x_as.")
        else:
            print("    → scores NON saturés : l'uniformité n'est pas due au clamp. "
                  "Vérifier si x_as est bien non-constant entre AS (bug potentiel "
                  "dans extract_features ou dans le scénario testé).")

    print("\n  ✓ Pilote terminé sans exception. Passer à run_v5_full() pour la "
          "campagne complète (10 seeds × 5 scénarios) une fois D1/D2 validés.")
    return policy


# ══════════════════════════════════════════════════════════════════════════
# 6. CAMPAGNE COMPLÈTE — même protocole statistique que Table I/II/PPO
#
#    Famille confirmatoire S1-S4 : Student-t 95% CI, Wilcoxon signé,
#      Bonferroni α'=0.05/4=0.0125 (identique à Table I/II/PPO).
#    S5_WACREN_OOD : test de transfert séparé, rapporté sans correction de
#      famille (ce n'est pas une 5e comparaison de la même famille — voir R3
#      en en-tête de fichier).
#
#    Entraînement (R4) : uniquement S3_PoP_Failure/S4_BGP_Anomaly (50/50),
#    comme le protocole BC/PPO historique — S1/S2/S5 restent des scénarios
#    de test de transfert non vus à l'entraînement.
#
#    Contrôle (R5) : évalue B1_Static, BC (HierarchicalPolicy chargée depuis
#    bc_phase_D.pt — résultats déjà verrouillés) et v5 sur les MÊMES graines
#    et scénarios, pour que "v5 apporte-t-elle quelque chose ?" se réponde
#    par rapport à l'architecture existante, pas seulement par rapport à B1.
# ══════════════════════════════════════════════════════════════════════════

FAMILY_SCENARIOS = ['S1_Nominal', 'S2_FlashCrowd', 'S3_PoP_Failure', 'S4_BGP_Anomaly']
OOD_SCENARIO = 'S5_WACREN_OOD'
BONFERRONI_ALPHA = 0.05 / len(FAMILY_SCENARIOS)   # = 0.0125, 4 comparaisons


def run_v5_full(quick=False, N_a=50, N_p=12, n_seeds=10, n_epochs=200, warm_start=True):
    seeds  = [0] if quick else list(range(n_seeds))
    epochs = 20 if quick else n_epochs
    scenarios = FAMILY_SCENARIOS + [OOD_SCENARIO]
    # R4 : S2 retiré de l'entraînement — seuls S3/S4 (comme le protocole
    # BC/PPO historique) pour que S1/S2/S5 restent des tests de transfert.
    train_scenarios = {'S3_PoP_Failure': 0.50, 'S4_BGP_Anomaly': 0.50}

    print(f"\n[GABPO v5 — campagne {'PILOTE' if quick else 'COMPLÈTE'}]")
    print(f"  Seeds: {seeds} | {epochs} PPO epochs (rollout updates, ~72 pas "
          f"chacun, ≈{epochs*72} transitions au total — R6 : PAS comparable "
          f"terme à terme à {epochs} epochs BC supervisées)")
    print(f"  lr_heads=1e-5 (têtes warm-startées) | lr_new=3e-4 (HCGA/temporel/"
          f"critic, init aléatoire) | warm_start(têtes seulement)={warm_start} | "
          f"train={list(train_scenarios.keys())} | test-transfert=S1/S2/S5")

    b1 = B1_Static(N_a, N_p)
    bc = HierarchicalPolicy(N_a, N_p).to(DEVICE)
    bc_loaded = False
    if os.path.exists(BC_CHECKPOINT):
        bc.load_state_dict(torch.load(BC_CHECKPOINT, map_location=DEVICE))
        bc_loaded = True
        print(f"  ✓ BC (HierarchicalPolicy verrouillée) chargée depuis {BC_CHECKPOINT}")
    else:
        print(f"  ⚠ {BC_CHECKPOINT} introuvable — comparaison BC omise (R5 incomplet).")
    bc.eval()

    all_rows = []
    for seed in seeds:
        print(f"\n[Seed {seed}]")
        torch.manual_seed(seed); np.random.seed(seed)
        policy = HierarchicalPolicy_HCGA_Temporal(N_a, N_p).to(DEVICE)
        if warm_start:
            policy.load_bc_heads()
        optimizer = policy.make_optimizer()  # R7 : lr différenciés têtes/nouveaux modules

        for epoch in range(epochs):
            noop_enabled = (epoch >= 20)   # booléen explicite, cf. R2
            scen = np.random.choice(list(train_scenarios.keys()), p=list(train_scenarios.values()))
            ep_seed = (seed * 1000 + epoch) % 200
            rollout = collect_rollout_v5(policy, scen, ep_seed, N_a, N_p, T_ep=72, noop_enabled=noop_enabled)
            ppo_update_v5(policy, optimizer, rollout, batch_size=min(64, len(rollout['obss'])))

        # Sauvegarde du checkpoint entraîné AVANT l'évaluation : si un bug
        # d'évaluation survient (cf. correctif run_episode_v5 ci-dessus), les
        # poids entraînés restent récupérables sans refaire tourner les 200
        # epochs PPO sur GPU.
        ck_path = f'checkpoints_v5/v5_s{seed}.pt'
        torch.save(policy.state_dict(), ck_path)
        print(f"  ✓ Checkpoint sauvegardé : {ck_path}")

        policy.eval()
        for scen in scenarios:
            r_v5 = run_episode_v5(policy, scen, seed + 900, N_a, N_p, T_ep=288)
            r_b1 = run_episode(b1, scen, seed + 900, N_a, N_p, 288)
            row = {'seed': seed, 'scenario': scen,
                   'v5_Lmed': r_v5['L_med'], 'B1_Lmed': r_b1['L_med'],
                   'v5_vs_b1_pct': (r_b1['L_med'] - r_v5['L_med']) / r_b1['L_med'] * 100}
            line = (f"  {scen}: v5={r_v5['L_med']:.1f}ms B1={r_b1['L_med']:.1f}ms "
                    f"v5>B1={row['v5_vs_b1_pct']:+.1f}%")
            if bc_loaded:
                r_bc = run_episode(bc, scen, seed + 900, N_a, N_p, 288)
                row['BC_Lmed'] = r_bc['L_med']
                row['v5_vs_bc_pct'] = (r_bc['L_med'] - r_v5['L_med']) / r_bc['L_med'] * 100
                line += f" BC={r_bc['L_med']:.1f}ms v5>BC={row['v5_vs_bc_pct']:+.1f}%"
            print(line)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv('results_v5/v5_raw_results.csv', index=False)

    if not quick:
        print(f"\n{'═'*65}\nFamille confirmatoire S1-S4 — Student-t 95% CI, "
              f"Bonferroni α'={BONFERRONI_ALPHA} (4 comparaisons)\n{'═'*65}")
        for scen in FAMILY_SCENARIOS:
            sub = df[df.scenario == scen]
            v = sub.v5_vs_b1_pct
            ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
            _, p = stats.wilcoxon(sub.v5_Lmed.values, sub.B1_Lmed.values)
            extra = ""
            if bc_loaded:
                extra = f" | v5>BC={sub.v5_vs_bc_pct.mean():+.1f}%"
            print(f"  {scen:<18} v5>B1={v.mean():>+.1f}% [{ci[0]:>+.1f},{ci[1]:>+.1f}] "
                  f"p={p:.3e}{extra}")

        print(f"\n{'─'*65}\n{OOD_SCENARIO} — test de transfert séparé (hors famille Bonferroni)\n{'─'*65}")
        sub = df[df.scenario == OOD_SCENARIO]
        v = sub.v5_vs_b1_pct
        ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
        _, p = stats.wilcoxon(sub.v5_Lmed.values, sub.B1_Lmed.values)
        extra = f" | v5>BC={sub.v5_vs_bc_pct.mean():+.1f}%" if bc_loaded else ""
        print(f"  {OOD_SCENARIO:<18} v5>B1={v.mean():>+.1f}% [{ci[0]:>+.1f},{ci[1]:>+.1f}] "
              f"p={p:.3e}{extra}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 7. RÉ-ÉVALUATION SEULE depuis des checkpoints déjà entraînés
#    (utile si un bug d'évaluation est corrigé après coup — cf. run_episode_v5)
# ══════════════════════════════════════════════════════════════════════════

def reeval_v5_from_checkpoints(seeds=range(10), N_a=50, N_p=12, ckpt_dir='checkpoints_v5'):
    b1 = B1_Static(N_a, N_p)
    bc = HierarchicalPolicy(N_a, N_p).to(DEVICE)
    bc_loaded = os.path.exists(BC_CHECKPOINT)
    if bc_loaded:
        bc.load_state_dict(torch.load(BC_CHECKPOINT, map_location=DEVICE))
    bc.eval()

    scenarios = FAMILY_SCENARIOS + [OOD_SCENARIO]
    rows = []
    for seed in seeds:
        ck_path = f'{ckpt_dir}/v5_s{seed}.pt'
        if not os.path.exists(ck_path):
            print(f"  ⚠ {ck_path} introuvable — seed {seed} sauté.")
            continue
        policy = HierarchicalPolicy_HCGA_Temporal(N_a, N_p).to(DEVICE)
        policy.load_state_dict(torch.load(ck_path, map_location=DEVICE))
        policy.eval()
        for scen in scenarios:
            r_v5 = run_episode_v5(policy, scen, seed + 900, N_a, N_p, T_ep=288)
            r_b1 = run_episode(b1, scen, seed + 900, N_a, N_p, 288)
            row = {'seed': seed, 'scenario': scen,
                   'v5_Lmed': r_v5['L_med'], 'B1_Lmed': r_b1['L_med'],
                   'v5_vs_b1_pct': (r_b1['L_med'] - r_v5['L_med']) / r_b1['L_med'] * 100}
            if bc_loaded:
                r_bc = run_episode(bc, scen, seed + 900, N_a, N_p, 288)
                row['BC_Lmed'] = r_bc['L_med']
                row['v5_vs_bc_pct'] = (r_bc['L_med'] - r_v5['L_med']) / r_bc['L_med'] * 100
            rows.append(row)
            print(f"  seed {seed} {scen}: v5_Lmed={row['v5_Lmed']:.1f}ms")
    df = pd.DataFrame(rows)
    df.to_csv('results_v5/v5_raw_results_reeval.csv', index=False)
    return df


if __name__ == "__main__":
    print("Ce fichier est conçu pour être exécuté via exec() après P0b + BC, "
          "comme les autres scripts du dépôt (cf. en-tête).")
