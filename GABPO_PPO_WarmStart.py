# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO Phase P2 — PPO Warm-Start depuis BC                            ║
# ║                                                                        ║
# ║  Configuration (avis expert) :                                         ║
# ║    - Actor initialisé avec checkpoint BC                               ║
# ║    - lr = 1e-5 (fine-tuning, pas réapprentissage)                      ║
# ║    - clip ε = 0.2 · GAE λ = 0.95 · γ = 0.99                          ║
# ║    - entropy coef = 0.001 · value coef = 0.5                           ║
# ║    - gradient clipping 0.5                                             ║
# ║    - NO-OP réintroduit progressivement                                 ║
# ║    - arrêt NaN/Inf immédiat                                            ║
# ║    - checkpoint BC conservé comme fallback                             ║
# ║                                                                        ║
# ║  Critère de réussite : PPO > BC > B1                                  ║
# ║  Si PPO ≈ BC : conserver BC (ne pas dégrader)                          ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_PPO_WarmStart.py').read())                         ║
# ║    df = run_ppo_warmstart(quick=True)   # 1 seed × 50 epochs           ║
# ║    df = run_ppo_warmstart(quick=False)  # 10 seeds × 200 epochs        ║
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

os.makedirs('results_ppo', exist_ok=True)
os.makedirs('checkpoints_ppo', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO Phase P2 — PPO Warm-Start depuis BC         ║")
print("╚══════════════════════════════════════════════════════╝")
print(f"Device: {DEVICE}")

# ── Vérification dépendances ──────────────────────────────────────────────
try:
    _ = BGPDynamicEnv, OracleAgent, B1_Static, run_episode
    _ = extract_features, HierarchicalAction, HierarchicalPolicy
    print("✓ P0b + BC chargés")
except NameError as e:
    raise RuntimeError(f"Exécuter P0b puis BC d'abord : {e}")

BC_CHECKPOINT = 'checkpoints_bc/bc_phase_D.pt'
if not os.path.exists(BC_CHECKPOINT):
    raise RuntimeError(f"Checkpoint BC introuvable : {BC_CHECKPOINT}\n"
                       "Exécuter run_bc(quick=False) d'abord.")
print(f"✓ Checkpoint BC trouvé : {BC_CHECKPOINT}")


# ══════════════════════════════════════════════════════════════════════════
# 1. POLITIQUE PPO HYBRIDE (hérite de HierarchicalPolicy)
# ══════════════════════════════════════════════════════════════════════════

class PPOHierarchicalPolicy(HierarchicalPolicy):
    """
    Extension de HierarchicalPolicy pour PPO.
    Ajoute : tête critic, log_prob hiérarchique, évaluation d'action.
    Initialisée avec les poids BC.
    """
    def __init__(self, N_a=50, N_p=12):
        super().__init__(N_a, N_p)
        # Tête critic séparée (ne pas modifier les têtes actor BC)
        d_state = 64
        self.critic_head = nn.Sequential(
            nn.Linear(d_state, 64), nn.Tanh(),
            nn.Linear(64, 1))
        # Init critic à zéro pour ne pas perturber l'actor BC
        for m in self.critic_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # NO-OP progressif : ratio initial = 0 (désactivé au début)
        self.noop_weight = 0.0   # sera augmenté progressivement

    def get_state(self, obs):
        _, x_pop = extract_features(obs, self.N_a, self.N_p)
        feat  = x_pop.flatten().to(DEVICE)
        state = self.encoder(feat)
        return state, feat

    def get_action_and_logprob(self, obs, noop_enabled=False):
        """
        Échantillonne une action hiérarchique et calcule log_prob.
        Retourne (action_bgp, log_prob, value, decision_dict)
        """
        state, feat = self.get_state(obs)

        noop_l, pop_l, type_l, amp_l = (
            self.noop_head(state),
            self.pop_head(state),
            self.type_head(state),
            self.amp_head(state))
        value = self.critic_head(state).squeeze()

        # ── NO-OP decision ─────────────────────────────────────────────
        if noop_enabled and self.noop_weight > 0:
            noop_dist = Categorical(logits=noop_l)
            noop_act  = noop_dist.sample()
            lp_noop   = noop_dist.log_prob(noop_act)
        else:
            noop_act  = torch.tensor(0)   # toujours agir
            lp_noop   = torch.tensor(0.0)

        if noop_act.item() == 1:
            # NO-OP
            action_bgp = np.zeros(self.N_p * self.N_a * 2, dtype=np.float32)
            log_prob   = lp_noop
            decisions  = {'noop':1, 'pop':0, 'type':0, 'amp':0}
        else:
            # Choisir PoP, Type, Amplitude
            pop_dist  = Categorical(logits=pop_l)
            type_dist = Categorical(logits=type_l)
            amp_dist  = Categorical(logits=amp_l)

            pop_act  = pop_dist.sample()
            type_act = type_dist.sample()
            amp_act  = amp_dist.sample()

            lp_pop   = pop_dist.log_prob(pop_act)
            lp_type  = type_dist.log_prob(type_act)
            lp_amp   = amp_dist.log_prob(amp_act)

            log_prob = lp_noop + lp_pop + lp_type + lp_amp

            # Construire action BGP
            decisions = {
                'noop': 0,
                'pop':  pop_act.item(),
                'type': type_act.item(),
                'amp':  amp_act.item(),
            }
            action_bgp = self.ha.decode(
                np.array([0.0] +                              # noop score (low)
                         [1.0 if i==pop_act.item() else 0.0
                          for i in range(self.N_p)] +         # pop logits
                         [1.0 if i==type_act.item() else 0.0
                          for i in range(4)] +                # type logits
                         [1.0 if i==amp_act.item() else 0.0
                          for i in range(3)]))                # amp logits

        return action_bgp, log_prob, value, decisions

    def evaluate_logprob(self, feat, decisions_batch):
        """
        Recalcule log_prob pour l'update PPO.
        decisions_batch : liste de dicts {noop, pop, type, amp}
        """
        state = self.encoder(feat)
        noop_l, pop_l, type_l, amp_l = (
            self.noop_head(state),
            self.pop_head(state),
            self.type_head(state),
            self.amp_head(state))
        value = self.critic_head(state).squeeze()

        log_probs, entropies = [], []
        for d in decisions_batch:
            lp = torch.tensor(0.0)
            ent = torch.tensor(0.0)

            if d['noop'] == 1:
                dist = Categorical(logits=noop_l)
                lp   = dist.log_prob(torch.tensor(1))
                ent  = dist.entropy()
            else:
                pop_dist  = Categorical(logits=pop_l)
                type_dist = Categorical(logits=type_l)
                amp_dist  = Categorical(logits=amp_l)
                lp  = (pop_dist.log_prob(torch.tensor(d['pop'])) +
                       type_dist.log_prob(torch.tensor(d['type'])) +
                       amp_dist.log_prob(torch.tensor(d['amp'])))
                ent = (pop_dist.entropy() + type_dist.entropy() +
                       amp_dist.entropy())
            log_probs.append(lp)
            entropies.append(ent)

        return torch.stack(log_probs), value, torch.stack(entropies).mean()

    def select_action(self, obs):
        """Compatible avec run_episode."""
        with torch.no_grad():
            action, _, _, _ = self.get_action_and_logprob(obs)
        return action


# ══════════════════════════════════════════════════════════════════════════
# 2. COLLECTE ROLLOUT HIÉRARCHIQUE
# ══════════════════════════════════════════════════════════════════════════

def collect_rollout_hier(policy, scenario, seed,
                          N_a=50, N_p=12, T_ep=72,
                          gamma=0.99, lam=0.95,
                          noop_enabled=False):
    env = BGPDynamicEnv(scenario, seed, N_a, N_p, T_ep)
    obs = env.reset()

    feats, decisions = [], []
    log_probs_old, rewards, values, dones = [], [], [], []

    policy.eval()
    with torch.no_grad():
        for _ in range(T_ep):
            _, x_pop = extract_features(obs, N_a, N_p)
            feat     = x_pop.flatten().to(DEVICE)

            action, lp, val, dec = policy.get_action_and_logprob(
                obs, noop_enabled=noop_enabled)

            obs, reward, done, info = env.step(action)

            # Reward relatif vs B1 (même logique que P0b)
            reward = float(np.clip(reward, -5.0, 5.0))

            feats.append(feat.cpu())
            decisions.append(dec)
            log_probs_old.append(lp.item())
            rewards.append(reward)
            values.append(val.item())
            dones.append(float(done))
            if done: break

    # GAE
    T   = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    ret = np.zeros(T, dtype=np.float32)
    gae = 0.0

    _, x_pop = extract_features(obs, N_a, N_p)
    with torch.no_grad():
        state   = policy.encoder(x_pop.flatten().to(DEVICE))
        v_last  = policy.critic_head(state).squeeze().item()
    next_v = v_last if not dones[-1] else 0.0

    for t in reversed(range(T)):
        nv    = values[t+1] if t+1 < T else next_v
        delta = rewards[t] + gamma*nv*(1-dones[t]) - values[t]
        gae   = delta + gamma*lam*(1-dones[t])*gae
        adv[t]= gae

    ret = adv + np.array(values, dtype=np.float32)
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return {
        'feats':     feats,
        'decisions': decisions,
        'lp_old':    np.array(log_probs_old, dtype=np.float32),
        'returns':   ret,
        'advantages':adv,
        'rewards':   np.array(rewards, dtype=np.float32),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. UPDATE PPO HIÉRARCHIQUE
# ══════════════════════════════════════════════════════════════════════════

def ppo_update_hier(policy, optimizer, rollout,
                    eps_clip=0.2, n_epochs=4,
                    batch_size=64, vf_coef=0.5,
                    ent_coef=0.001):
    T       = len(rollout['feats'])
    indices = np.arange(T)
    metrics = {'actor':[], 'critic':[], 'entropy':[],
               'kl':[], 'clip_frac':[]}

    policy.train()
    for _ in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, T, batch_size):
            idx = indices[start:start+batch_size]
            if len(idx) < 2: continue

            lp_news, vals, ents = [], [], []
            valid_idx = []
            for i in idx:
                feat = rollout['feats'][i].to(DEVICE)
                dec  = [rollout['decisions'][i]]
                try:
                    lp, v, e = policy.evaluate_logprob(feat, dec)
                    if torch.isnan(lp).any() or torch.isinf(lp).any():
                        continue
                    lp_news.append(lp.squeeze())
                    vals.append(v)
                    ents.append(e)
                    valid_idx.append(i)
                except Exception:
                    continue

            if len(valid_idx) < 2: continue

            lp_new = torch.stack(lp_news)
            val_t  = torch.stack(vals)
            ent_t  = torch.stack(ents).mean()

            lp_old = torch.FloatTensor(
                rollout['lp_old'][valid_idx]).to(DEVICE)
            ret_t  = torch.FloatTensor(
                rollout['returns'][valid_idx]).to(DEVICE)
            adv_t  = torch.FloatTensor(
                rollout['advantages'][valid_idx]).to(DEVICE)

            # Ratio PPO avec clip sur log-diff
            log_diff = (lp_new - lp_old).clamp(-5, 5)
            ratio    = torch.exp(log_diff)

            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * adv_t
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = F.smooth_l1_loss(val_t, ret_t)
            loss = actor_loss + vf_coef*critic_loss - ent_coef*ent_t

            # ── Vérification NaN avant backward ──────────────────────
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            # Zeroing NaN gradients
            for p in policy.parameters():
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(
                        p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            optimizer.step()

            with torch.no_grad():
                kl        = ((lp_old-lp_new)**2).mean().item()/2
                clip_frac = ((ratio-1).abs()>eps_clip).float().mean().item()

            metrics['actor'].append(actor_loss.item())
            metrics['critic'].append(critic_loss.item())
            metrics['entropy'].append(ent_t.item())
            metrics['kl'].append(kl)
            metrics['clip_frac'].append(clip_frac)

    return {k: float(np.mean(v)) if v else 0.0
            for k, v in metrics.items()}


# ══════════════════════════════════════════════════════════════════════════
# 4. PIPELINE PPO WARM-START
# ══════════════════════════════════════════════════════════════════════════

def run_ppo_warmstart(quick=True, N_a=50, N_p=12,
                       n_epochs=200, n_seeds=10):
    """
    Phase P2 — PPO warm-start depuis BC.
    quick=True  : 1 seed × 50 epochs (~10-15 min CPU)
    quick=False : 10 seeds × 200 epochs (~2h GPU)
    """
    seeds   = [0] if quick else list(range(n_seeds))
    epochs  = 50  if quick else n_epochs

    # Scénarios entraînement (S1 = validation seulement)
    train_scenarios = {
        'S2_FlashCrowd': 0.20,
        'S3_PoP_Failure':0.40,
        'S4_BGP_Anomaly':0.40,
    }

    print(f"\n[PPO Warm-Start — {'PILOTE' if quick else 'COMPLET'}]")
    print(f"  Seeds: {seeds} | Epochs: {epochs} | lr=1e-5")
    print(f"  Train: {list(train_scenarios.keys())} | Val: S1_Nominal")
    print("=" * 60)

    b1      = B1_Static(N_a, N_p)
    all_evals = []

    for seed in seeds:
        print(f"\n[Seed {seed}]")
        torch.manual_seed(seed); np.random.seed(seed)

        # ── Charger checkpoint BC (warm-start) ────────────────────────
        policy = PPOHierarchicalPolicy(N_a, N_p).to(DEVICE)
        bc_state = torch.load(BC_CHECKPOINT, map_location=DEVICE)
        # Charger uniquement les poids partagés (pas critic)
        policy.load_state_dict(bc_state, strict=False)
        print(f"  ✓ Warm-start depuis BC ({BC_CHECKPOINT})")

        # Conserver une copie BC pour fallback
        bc_Ls = {}
        for scen in ['S1_Nominal','S2_FlashCrowd',
                     'S3_PoP_Failure','S4_BGP_Anomaly']:
            r = run_episode(policy, scen, seed+900, N_a, N_p, 288)
            bc_Ls[scen] = r['L_med']
        print(f"  BC baseline : "
              f"S1={bc_Ls['S1_Nominal']:.1f} "
              f"S2={bc_Ls['S2_FlashCrowd']:.1f} "
              f"S3={bc_Ls['S3_PoP_Failure']:.1f} "
              f"S4={bc_Ls['S4_BGP_Anomaly']:.1f}ms")

        # Optimizer avec lr très faible
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=1e-5, eps=1e-5,
            weight_decay=1e-5)

        best_ppo_score = sum(bc_Ls.values())  # référence BC
        best_ck_path   = f'checkpoints_ppo/ppo_s{seed}.pt'
        torch.save(policy.state_dict(), best_ck_path)  # sauver BC comme fallback

        noop_schedule = 0  # NO-OP désactivé au début

        for epoch in range(epochs):
            # Activer NO-OP progressivement après epoch 20
            noop_enabled = (epoch >= 20)
            if epoch == 20:
                policy.noop_weight = 0.3
                print(f"  Epoch {epoch+1} : NO-OP activé (weight=0.3)")

            scen    = np.random.choice(
                list(train_scenarios.keys()),
                p=list(train_scenarios.values()))
            ep_seed = (seed*1000 + epoch) % 200

            # Collecte + update
            rollout = collect_rollout_hier(
                policy, scen, ep_seed, N_a, N_p,
                T_ep=72, noop_enabled=noop_enabled)
            ppo_m   = ppo_update_hier(
                policy, optimizer, rollout,
                eps_clip=0.2, n_epochs=4,
                batch_size=min(64, len(rollout['feats'])))

            # Validation tous les 10 epochs
            if (epoch+1) % 10 == 0:
                policy.eval()
                ppo_Ls = {}
                for scen_val in ['S1_Nominal','S3_PoP_Failure',
                                  'S4_BGP_Anomaly']:
                    r = run_episode(policy, scen_val,
                                     seed+800, N_a, N_p, 72)
                    ppo_Ls[scen_val] = r['L_med']

                ppo_score = sum(ppo_Ls.values())

                # Comparer vs BC
                delta_s3 = bc_Ls['S3_PoP_Failure'] - ppo_Ls['S3_PoP_Failure']
                delta_s4 = bc_Ls['S4_BGP_Anomaly'] - ppo_Ls['S4_BGP_Anomaly']

                sym_s3 = '✅' if delta_s3>0 else '❌'
                sym_s4 = '✅' if delta_s4>0 else '❌'

                print(f"  Ep {epoch+1:3d} | "
                      f"actor={ppo_m['actor']:>+.4f} | "
                      f"KL={ppo_m['kl']:.4f} | "
                      f"S1={ppo_Ls['S1_Nominal']:.0f}ms | "
                      f"{sym_s3}S3={ppo_Ls['S3_PoP_Failure']:.0f}({delta_s3:+.1f}) | "
                      f"{sym_s4}S4={ppo_Ls['S4_BGP_Anomaly']:.0f}({delta_s4:+.1f})")

                # Sauvegarder si meilleur que BC
                if ppo_score < best_ppo_score:
                    best_ppo_score = ppo_score
                    torch.save(policy.state_dict(), best_ck_path)
                    print(f"  ✓ Nouveau meilleur ({ppo_score:.1f} < "
                          f"{sum(bc_Ls.values()):.1f})")

                # Arrêt si KL diverge
                if ppo_m['kl'] > 0.05:
                    print(f"  ⚠ KL={ppo_m['kl']:.4f} > 0.05 — "
                          f"lr réduit")
                    for pg in optimizer.param_groups:
                        pg['lr'] *= 0.5

        # ── Évaluation finale — charger meilleur checkpoint ───────────
        policy.load_state_dict(
            torch.load(best_ck_path, map_location=DEVICE))

        print(f"\n  Évaluation finale seed {seed} :")
        for scen in ['S1_Nominal','S2_FlashCrowd',
                     'S3_PoP_Failure','S4_BGP_Anomaly']:
            r_ppo = run_episode(policy, scen, seed+900, N_a, N_p, 288)
            r_b1  = run_episode(b1,     scen, seed+900, N_a, N_p, 288)
            ppo_vs_b1 = (r_b1['L_med']-r_ppo['L_med'])/r_b1['L_med']*100
            ppo_vs_bc = bc_Ls[scen] - r_ppo['L_med']
            sym = '✅' if ppo_vs_bc>0 else ('≈' if ppo_vs_bc>-2 else '❌')
            print(f"  {sym} {scen}: "
                  f"PPO={r_ppo['L_med']:.1f}ms "
                  f"BC={bc_Ls[scen]:.1f}ms "
                  f"B1={r_b1['L_med']:.1f}ms "
                  f"PPO>B1={ppo_vs_b1:+.1f}% "
                  f"PPO>BC={ppo_vs_bc:+.1f}ms")
            all_evals.append({
                'seed': seed, 'scenario': scen,
                'PPO_Lmed':   r_ppo['L_med'],
                'BC_Lmed':    bc_Ls[scen],
                'B1_Lmed':    r_b1['L_med'],
                'ppo_vs_b1':  ppo_vs_b1,
                'ppo_vs_bc':  ppo_vs_bc,
            })

    # ── Résultats finaux ──────────────────────────────────────────────
    df = pd.DataFrame(all_evals)

    if not quick:
        print(f"\n{'═'*65}")
        print(f"RÉSULTATS FINAUX — PPO vs BC vs B1 (10 seeds)")
        print(f"{'═'*65}")
        for scen in ['S1_Nominal','S2_FlashCrowd',
                     'S3_PoP_Failure','S4_BGP_Anomaly']:
            sub   = df[df.scenario==scen]
            vb1   = sub.ppo_vs_b1
            vbc   = sub.ppo_vs_bc
            ci_b1 = stats.t.interval(
                0.95, len(vb1)-1, loc=vb1.mean(), scale=stats.sem(vb1))
            _, p  = stats.wilcoxon(
                sub.PPO_Lmed.values, sub.B1_Lmed.values)
            sym = '✅' if ci_b1[0]>0 else '⚠'
            print(f"  {sym} {scen:<22} "
                  f"PPO>B1={vb1.mean():>+.1f}% [{ci_b1[0]:>+.1f},{ci_b1[1]:>+.1f}] "
                  f"PPO>BC={vbc.mean():>+.1f}ms "
                  f"p={p:.3e}")

        print(f"\n  Verdict PPO>BC>B1 :")
        for scen in ['S2_FlashCrowd','S3_PoP_Failure','S4_BGP_Anomaly']:
            sub = df[df.scenario==scen]
            ppo_beats_bc = (sub.ppo_vs_bc.mean() > 0)
            bc_beats_b1  = (sub.ppo_vs_b1.mean() > 0)
            sym = '✅' if (ppo_beats_bc and bc_beats_b1) else '⚠'
            print(f"  {sym} {scen}: "
                  f"PPO>BC={'Oui' if ppo_beats_bc else 'Non'} | "
                  f"BC>B1={'Oui' if bc_beats_b1 else 'Non'}")

    # Export
    df.to_csv('results_ppo/GABPO_PPO_WarmStart_eval.csv', index=False)
    with open('results_ppo/GABPO_PPO_WarmStart_summary.json','w') as f:
        json.dump({
            'version':   'P2_WarmStart',
            'timestamp': time.strftime('%Y%m%d_%H%M'),
            'quick':     quick, 'n_seeds': len(seeds),
            'bc_checkpoint': BC_CHECKPOINT,
            'lr': 1e-5,
        }, f, indent=2)

    if quick:
        print(f"\n[Pilote terminé]")
        print(f"→ Vérifier : KL<0.05, PPO≥BC sur S3/S4")
        print(f"→ Si sain : lancer run_ppo_warmstart(quick=False)")

    return df


# ── Point d'entrée ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Lancement PPO warm-start pilote (quick=True)...")
    df = run_ppo_warmstart(quick=True)
