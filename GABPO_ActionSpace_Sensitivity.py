# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Action-Space Sensitivity Study (1200 → 12 / 20 / 40)         ║
# ║                                                                        ║
# ║  Single agreed follow-up experiment (8th external review): tests       ║
# ║  whether the 20-dimensional decomposition is a meaningful choice or    ║
# ║  an arbitrary one, by comparing three decomposition granularities      ║
# ║  under the identical BC pipeline, dataset, and evaluation protocol.    ║
# ║                                                                        ║
# ║  Variants (Direct 1200-dim already documented as a failure — Section   ║
# ║  4.3 — NOT re-run here, its result is already locked):                 ║
# ║    COARSE   : PoP only (no Type/Amplitude choice) → 1+12+1 = 14 dims   ║
# ║               Type fixed to LOCAL_PREF+, Amplitude fixed to 1.0        ║
# ║    PROPOSED : PoP + Type{4} + Amplitude{3} + NOOP → 20 dims (current)  ║
# ║    FINE     : PoP + Type{4} + Amplitude{6, finer-grained} + NOOP       ║
# ║               → 1+12+4+6 = 23 dims (Amplitude ∈ {0.1,0.25,0.4,0.55,    ║
# ║               0.7,1.0} instead of 3 coarse values)                     ║
# ║                                                                        ║
# ║  Same Oracle dataset, same decoder logic pattern, same BC loss,        ║
# ║  same 10-seed evaluation protocol as Table II — only the action        ║
# ║  granularity changes.                                                  ║
# ║                                                                        ║
# ║  Safety: dataset pickled once and reused (lesson learned from the      ║
# ║  HCGA campaign — avoids seed-to-seed divergence across reruns),        ║
# ║  checkpoint + CSV saved after EACH variant (not at the end), resume    ║
# ║  support if a variant is already done.                                 ║
# ║                                                                        ║
# ║  Prerequisites: P0b → extract_features patch → BC → Ablation B3-B6     ║
# ║  (for generate_dataset_with_obs, B1_Static) already loaded.            ║
# ║                                                                        ║
# ║  Usage:                                                                ║
# ║    exec(open('GABPO_ActionSpace_Sensitivity.py').read())               ║
# ║    df, df_summary = run_sensitivity_study(n_seeds=10)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_sensitivity', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Action-Space Sensitivity (12 / 20 / 40)    ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    _ = BGPDynamicEnv, run_episode, extract_features
    _ = generate_dataset_with_obs, B1_Static
    print("✓ Dépendances chargées")
except NameError as e:
    raise RuntimeError(
        f"Exécuter P0b → extract_features patch → BC → "
        f"GABPO_Ablation_B3B6.py d'abord : {e}")

# Locked result from Section 4.3 — NOT re-run, kept for the summary table
DIRECT_1200_LOCKED = {
    'dims': 1200, 'converged': False,
    'note': 'action variance stayed near-fixed after 200 epochs; no consistent gain over B1 (Section 4.3)',
}


# ══════════════════════════════════════════════════════════════════════════
# Three HierarchicalAction variants — same PoP/parcimony logic, different
# granularity of Type/Amplitude choices
# ══════════════════════════════════════════════════════════════════════════

class ActionDecoder_Coarse:
    """PoP-only decomposition: 1 (noop) + N_p = 13 dims for N_p=12.
    Type fixed to LOCAL_PREF+, Amplitude fixed to 1.0 — the model only
    decides WHICH PoP to touch and whether to act at all."""
    def __init__(self, N_a=50, N_p=12):
        self.N_a, self.N_p = N_a, N_p
        self.dim = 1 + N_p
        self.n_pop, self.n_type, self.n_amp = N_p, 0, 0

    def decode(self, logits):
        N_p, N_a = self.N_p, self.N_a
        action = np.zeros(N_p * N_a * 2, dtype=np.float32)
        if logits[0] > logits[1:N_p+1].max():
            return action
        pop_idx = int(np.argmax(logits[1:N_p+1]))
        off = N_p * N_a
        for i in range(N_a):
            action[off + pop_idx*N_a + i] = 1.0  # fixed LP+, fixed amplitude 1.0
        return action

    def encode_oracle_labels(self, oracle_pop, oracle_type, oracle_amp):
        # Coarse variant only supervises the NOOP/PoP heads
        return {'noop': 0 if oracle_pop is not None else 1, 'pop': oracle_pop or 0}


class ActionDecoder_Proposed:
    """The 20-dim decomposition already used throughout this study
    (PoP + 4 types + 3 amplitudes + NOOP). Thin wrapper around the
    existing HierarchicalAction for consistency in this script."""
    AMPLITUDES = [0.25, 0.5, 1.0]
    TYPES = ['LP+', 'LP-', 'Prep+', 'Prep-']

    def __init__(self, N_a=50, N_p=12):
        self.N_a, self.N_p = N_a, N_p
        self.dim = 1 + N_p + 4 + 3
        self.n_pop, self.n_type, self.n_amp = N_p, 4, 3

    def decode(self, logits):
        N_p, N_a = self.N_p, self.N_a
        action = np.zeros(N_p * N_a * 2, dtype=np.float32)
        if logits[0] > logits[1:N_p+1].max():
            return action
        pop_idx = int(np.argmax(logits[1:N_p+1]))
        type_idx = int(np.argmax(logits[N_p+1:N_p+5]))
        action_type = self.TYPES[type_idx]
        amp_idx = int(np.argmax(logits[N_p+5:N_p+8]))
        amplitude = self.AMPLITUDES[amp_idx]
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


class ActionDecoder_Fine:
    """Finer-grained amplitude: 6 discrete levels instead of 3.
    1 (noop) + N_p + 4 + 6 = 23 dims for N_p=12."""
    AMPLITUDES = [0.1, 0.25, 0.4, 0.55, 0.7, 1.0]
    TYPES = ['LP+', 'LP-', 'Prep+', 'Prep-']

    def __init__(self, N_a=50, N_p=12):
        self.N_a, self.N_p = N_a, N_p
        self.dim = 1 + N_p + 4 + 6
        self.n_pop, self.n_type, self.n_amp = N_p, 4, 6

    def decode(self, logits):
        N_p, N_a = self.N_p, self.N_a
        action = np.zeros(N_p * N_a * 2, dtype=np.float32)
        if logits[0] > logits[1:N_p+1].max():
            return action
        pop_idx = int(np.argmax(logits[1:N_p+1]))
        type_idx = int(np.argmax(logits[N_p+1:N_p+5]))
        action_type = self.TYPES[type_idx]
        amp_idx = int(np.argmax(logits[N_p+5:N_p+11]))
        amplitude = self.AMPLITUDES[amp_idx]
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


# ══════════════════════════════════════════════════════════════════════════
# Policy network — generic, adapts head sizes to the decoder's dims
# ══════════════════════════════════════════════════════════════════════════

class VariantPolicy(nn.Module):
    """Same MLP backbone as the locked HierarchicalPolicy (PoP features
    only, 96-dim flattened input) — only the head sizes change per variant."""
    def __init__(self, decoder, N_a=50, N_p=12, hidden=64):
        super().__init__()
        self.decoder = decoder
        self.N_a, self.N_p = N_a, N_p
        in_dim = N_p * 8
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.noop_head = nn.Linear(hidden, 2)
        self.pop_head  = nn.Linear(hidden, decoder.n_pop) if decoder.n_pop else None
        self.type_head = nn.Linear(hidden, decoder.n_type) if decoder.n_type else None
        self.amp_head  = nn.Linear(hidden, decoder.n_amp) if decoder.n_amp else None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.5)
                nn.init.zeros_(m.bias)

    def _get_features(self, obs):
        _, x_pop = extract_features(obs, self.N_a, self.N_p)
        return x_pop.flatten()

    def forward(self, feat):
        h = self.encoder(feat)
        noop_l = self.noop_head(h)
        pop_l  = self.pop_head(h) if self.pop_head else None
        type_l = self.type_head(h) if self.type_head else None
        amp_l  = self.amp_head(h) if self.amp_head else None
        return noop_l, pop_l, type_l, amp_l

    def get_full_logits(self, obs):
        """Assemble the flat logits vector expected by decoder.decode():
        logits[0] = NOOP score, logits[1:1+n_pop] = pop scores, then
        type scores (if any), then amplitude scores (if any)."""
        feat = self._get_features(obs)
        noop_l, pop_l, type_l, amp_l = self.forward(feat)
        with torch.no_grad():
            pieces = [noop_l[0].item()]  # index 0 = "stay" / NOOP score
            if pop_l is not None:
                pieces += pop_l.numpy().tolist()
            if type_l is not None:
                pieces += type_l.numpy().tolist()
            if amp_l is not None:
                pieces += amp_l.numpy().tolist()
        return np.array(pieces, dtype=np.float32), (noop_l, pop_l, type_l, amp_l)

    def select_action(self, obs):
        with torch.no_grad():
            logits, _ = self.get_full_logits(obs)
        return self.decoder.decode(logits)


# ══════════════════════════════════════════════════════════════════════════
# Training — cross-entropy over Oracle demonstrations, variant-agnostic
# ══════════════════════════════════════════════════════════════════════════

def train_variant(model, dataset, n_epochs=100, batch_size=128, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(dataset)
    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start+batch_size]
            losses = []
            for i in batch_idx:
                d = dataset[i]
                feat = model._get_features(d['obs_raw'])
                noop_l, pop_l, type_l, amp_l = model.forward(feat)
                l = 2.0 * F.cross_entropy(noop_l.unsqueeze(0), torch.LongTensor([d['noop']]))
                if pop_l is not None:
                    l = l + 1.5 * F.cross_entropy(pop_l.unsqueeze(0), torch.LongTensor([d['pop']]))
                if type_l is not None and d.get('type') is not None:
                    l = l + 1.0 * F.cross_entropy(type_l.unsqueeze(0), torch.LongTensor([d['type']]))
                if amp_l is not None and d.get('amp') is not None:
                    l = l + 0.5 * F.cross_entropy(amp_l.unsqueeze(0), torch.LongTensor([d['amp']]))
                losses.append(l)
            batch_loss = torch.stack(losses).mean()
            opt.zero_grad(); batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def remap_dataset_labels(dataset_20dim, variant_name, N_p=12):
    """
    The existing 28,800-sample Oracle dataset was labeled for the 20-dim
    decoder (pop/type/amp indices already computed). For COARSE we keep
    only noop/pop; for FINE we need amplitude re-bucketed into 6 levels
    instead of 3 — approximated by mapping the original 3-level amplitude
    index to the nearest of 6 finer levels (midpoint mapping), since the
    Oracle's underlying continuous preference is not separately stored.
    """
    remapped = []
    coarse_amp_to_fine = {0: 1, 1: 3, 2: 5}  # 0.25→0.25(idx1), 0.5→0.55(idx3), 1.0→1.0(idx5)
    for d in dataset_20dim:
        nd = dict(d)
        if variant_name == 'coarse':
            nd['type'] = None
            nd['amp']  = None
        elif variant_name == 'fine':
            nd['amp'] = coarse_amp_to_fine.get(d['amp'], d['amp'])
        remapped.append(nd)
    return remapped


def run_sensitivity_study(n_seeds:int=10, n_episodes:int=100, n_epochs:int=100,
                          N_a:int=50, N_p:int=12):
    scenarios_train = ['S3_PoP_Failure', 'S4_BGP_Anomaly']
    all_scenarios = scenarios_train + ['S1_Nominal']

    # ── Dataset: reuse pickled version if present (lesson from HCGA work) ──
    dataset_path = 'results_sensitivity/oracle_dataset_20dim.pkl'
    if os.path.exists(dataset_path):
        print(f"[Dataset] Chargement depuis {dataset_path}...")
        with open(dataset_path, 'rb') as f:
            dataset_20 = pickle.load(f)
    else:
        print(f"[Dataset] Génération (première fois)...")
        dataset_20 = generate_dataset_with_obs(scenarios_train, n_episodes, N_a, N_p, T_ep=72)
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset_20, f)
        print(f"  ✓ {len(dataset_20)} samples générés et sauvegardés")

    b1 = B1_Static(N_a, N_p)
    variants = {
        'coarse':   ActionDecoder_Coarse(N_a, N_p),
        'proposed': ActionDecoder_Proposed(N_a, N_p),
        'fine':     ActionDecoder_Fine(N_a, N_p),
    }

    all_evals = []
    for vname, decoder in variants.items():
        results_path = f'results_sensitivity/{vname}_results.csv'

        # ── Resume: load whatever seeds are already on disk for this variant ──
        # Accepts EITHER format: the script's native long format (one row per
        # seed×scenario, with 'improvement_pct'), OR the wide format the user
        # may have saved manually (one row per seed, with '<scenario>_gain_pct'
        # columns) — auto-converts wide→long on load.
        variant_evals = []
        seeds_done = set()
        if os.path.exists(results_path):
            prev = pd.read_csv(results_path)
            if 'improvement_pct' in prev.columns:
                # native long format
                variant_evals = prev.to_dict('records')
                seeds_done = set(prev.seed.unique())
            else:
                # wide format (seed, S3_..._gain_pct, S4_..._gain_pct, S1_..._gain_pct)
                print(f"  ⚠ {vname}: format large détecté dans {results_path} — conversion automatique")
                gain_cols = [c for c in prev.columns if c.endswith('_gain_pct')]
                for _, row in prev.iterrows():
                    for col in gain_cols:
                        scen = col.replace('_gain_pct', '')
                        variant_evals.append({
                            'variant': vname, 'dims': decoder.dim,
                            'seed': int(row['seed']), 'scenario': scen,
                            'L_med': None, 'B1_L_med': None,
                            'improvement_pct': row[col],
                        })
                seeds_done = set(int(s) for s in prev.seed.unique())
                # rewrite results_path in native long format so future
                # resumes don't need to re-detect the wide format
                pd.DataFrame(variant_evals).to_csv(results_path, index=False)
                print(f"  ✓ {vname}: converti et réécrit en format long ({results_path})")
            print(f"\n✓ {vname}: {len(seeds_done)}/{n_seeds} seeds déjà sur disque "
                  f"{sorted(seeds_done)}")

        if len(seeds_done) >= n_seeds:
            print(f"  → {vname} complet, passage à la variante suivante")
            all_evals.extend(variant_evals)
            continue

        print(f"\n{'='*60}\n{vname.upper()} — {decoder.dim} dimensions "
              f"— seeds restants: {[s for s in range(n_seeds) if s not in seeds_done]}\n"
              f"{'='*60}")
        dataset_v = remap_dataset_labels(dataset_20, vname, N_p)

        for seed in range(n_seeds):
            if seed in seeds_done:
                continue  # already have this seed's results

            torch.manual_seed(seed); np.random.seed(seed)
            model = VariantPolicy(decoder, N_a, N_p)
            model = train_variant(model, dataset_v, n_epochs=n_epochs)

            torch.save(model.state_dict(), f'results_sensitivity/{vname}_seed{seed}.pt')

            for scen in all_scenarios:
                r_m  = run_episode(model, scen, seed+900, N_a, N_p, T_ep=288)
                r_b1 = run_episode(b1,    scen, seed+900, N_a, N_p, T_ep=288)
                imp  = (r_b1['L_med']-r_m['L_med'])/r_b1['L_med']*100
                variant_evals.append({
                    'variant': vname, 'dims': decoder.dim, 'seed': seed,
                    'scenario': scen, 'L_med': r_m['L_med'],
                    'B1_L_med': r_b1['L_med'], 'improvement_pct': imp,
                })

            imps = [r['improvement_pct'] for r in variant_evals if r['seed']==seed]
            print(f"  Seed {seed}: " + " | ".join(
                f"{s}={i:+.1f}%" for s, i in zip(all_scenarios, imps)))

            # ── CRITICAL: save after EVERY seed, not after the full variant ──
            pd.DataFrame(variant_evals).to_csv(results_path, index=False)

        print(f"  ✓ {vname} complet et sauvegardé : {results_path}")
        all_evals.extend(variant_evals)

    df = pd.DataFrame(all_evals)
    df.to_csv('results_sensitivity/GABPO_sensitivity_all.csv', index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*76}\nACTION-SPACE SENSITIVITY — SUMMARY\n{'='*76}")
    print(f"\n  {'Variant':<12}{'Dims':<8}" +
          "".join(f"{s[:14]:<16}" for s in all_scenarios))
    print(f"  {'Direct (locked)':<12}{'1200':<8}" + "NOT CONVERGED (Section 4.3)")

    summary_rows = []
    for vname, decoder in variants.items():
        sub = df[df.variant == vname]
        row = f"  {vname:<12}{decoder.dim:<8}"
        srow = {'variant': vname, 'dims': decoder.dim}
        for scen in all_scenarios:
            scen_sub = sub[sub.scenario == scen]
            m = scen_sub.improvement_pct.mean()
            sd = scen_sub.improvement_pct.std()
            row += f"{m:>+6.2f}±{sd:.2f}%   "
            srow[scen] = m
        print(row)
        summary_rows.append(srow)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('results_sensitivity/GABPO_sensitivity_summary.csv', index=False)
    print(f"\n✓ Sauvegardé : results_sensitivity/GABPO_sensitivity_summary.csv")
    print(f"\n→ Télécharger : from google.colab import files; "
          f"files.download('results_sensitivity/GABPO_sensitivity_summary.csv')")

    return df, df_summary


if __name__ == "__main__":
    df, df_summary = run_sensitivity_study(n_seeds=10)
