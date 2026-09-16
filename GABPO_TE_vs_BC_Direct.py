# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Comparaison DIRECTE TE1/TE2/TE3 vs BC (pas via B1)            ║
# ║                                                                        ║
# ║  Corrige le test précédent : TE vs B1 répond à "TE bat-il l'inaction   ║
# ║  totale ?" — pas à la vraie question "TE bat-il BC ?". B1_Static est   ║
# ║  un vecteur d'action ENTIÈREMENT NUL (vérifié), donc n'importe quelle  ║
# ║  action non-triviale le bat mécaniquement. La comparaison correcte     ║
# ║  est directe : TE_i vs BC, appariée sur les mêmes seeds/scénarios.     ║
# ║                                                                        ║
# ║  Réutilise les résultats déjà calculés (TE_L_med par seed/scénario     ║
# ║  dans results_te/*.csv) — PAS de recalcul de TE, seulement un nouveau  ║
# ║  calcul de BC sur les mêmes seeds/scénarios pour permettre la          ║
# ║  comparaison appariée correcte.                                        ║
# ║                                                                        ║
# ║  Prérequis : P0b, BC (checkpoint bc_phase_D.pt chargé), TE_Baselines   ║
# ║  déjà exécutés — et results_te/{TE1,TE2,TE3}_results.csv déjà          ║
# ║  présents (issus du run précédent).                                    ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_TE_vs_BC_Direct.py').read())                       ║
# ║    df, df_summary = run_te_vs_bc_comparison(n_seeds=10)                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import numpy as np
import torch
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_te_vs_bc', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — TE1/TE2/TE3 vs BC : comparaison DIRECTE    ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    _ = BGPDynamicEnv, run_episode, HierarchicalPolicy
except NameError as e:
    raise RuntimeError(f"Exécuter P0b puis BC d'abord : {e}")

BC_CHECKPOINT = 'checkpoints_bc/bc_phase_D.pt'
if not os.path.exists(BC_CHECKPOINT):
    raise RuntimeError(f"Checkpoint BC introuvable : {BC_CHECKPOINT} — "
                        f"le restaurer avant de continuer.")

ALL_SCENARIOS = ['S1_Nominal', 'S2_FlashCrowd', 'S3_PoP_Failure',
                  'S4_BGP_Anomaly', 'S5_WACREN_OOD']
TE_NAMES = ['TE1_OSPF', 'TE2_MATE', 'TE3_TeXCP']


def run_te_vs_bc_comparison(n_seeds:int=10, N_a:int=50, N_p:int=12, T_ep:int=288):
    """
    Recalcule BC_L_med sur les MÊMES (seed, scenario) déjà utilisés
    pour TE1/TE2/TE3 (results_te/*.csv), puis compare directement
    TE_i vs BC (paired), pas TE_i vs B1 vs BC séparément.
    """
    # ── Charger BC une fois ────────────────────────────────────────────
    policy = HierarchicalPolicy(N_a, N_p)
    policy.load_state_dict(torch.load(BC_CHECKPOINT, map_location='cpu'))
    policy.eval()
    print("✓ Checkpoint BC chargé")

    # ── Charger les résultats TE déjà calculés ─────────────────────────
    te_data = {}
    for te_name in TE_NAMES:
        path = f'results_te/{te_name}_results.csv'
        if not os.path.exists(path):
            raise RuntimeError(f"Fichier manquant : {path} — relancer "
                                f"d'abord GABPO_TE_Evaluation.py")
        te_data[te_name] = pd.read_csv(path)
        print(f"✓ {te_name} : {len(te_data[te_name])} lignes chargées")

    # ── Recalculer BC sur les mêmes (seed, scenario) — sauvegarde par seed ──
    bc_path = 'results_te_vs_bc/BC_matched_results.csv'
    bc_evals = []
    seeds_done = set()
    if os.path.exists(bc_path):
        prev = pd.read_csv(bc_path)
        bc_evals = prev.to_dict('records')
        seeds_done = set(prev.seed.unique())
        print(f"✓ BC déjà évalué pour seeds : {sorted(seeds_done)}")

    print(f"\n[Évaluation BC] sur les mêmes seeds/scénarios que TE1/2/3")
    for seed in range(n_seeds):
        if seed in seeds_done:
            continue
        for scen in ALL_SCENARIOS:
            r_bc = run_episode(policy, scen, seed+900, N_a, N_p, T_ep)
            bc_evals.append({
                'seed': seed, 'scenario': scen, 'BC_L_med': r_bc['L_med'],
            })
        imps_preview = [r['BC_L_med'] for r in bc_evals if r['seed']==seed]
        print(f"  Seed {seed}: BC evalué sur {len(imps_preview)} scénarios")
        pd.DataFrame(bc_evals).to_csv(bc_path, index=False)

    df_bc = pd.DataFrame(bc_evals)

    # ══════════════════════════════════════════════════════════════════
    # COMPARAISON DIRECTE : TE_i vs BC, appariée sur (seed, scenario)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\nCOMPARAISON DIRECTE — TE1/TE2/TE3 vs BC (pas via B1)\n{'='*80}")

    all_rows = []
    for te_name in TE_NAMES:
        print(f"\n{'-'*60}\n{te_name} vs BC\n{'-'*60}")
        te_df = te_data[te_name]

        for scen in ALL_SCENARIOS:
            te_sub = te_df[te_df.scenario == scen].sort_values('seed')
            bc_sub = df_bc[df_bc.scenario == scen].sort_values('seed')

            # Aligner sur les seeds communs
            common_seeds = sorted(set(te_sub.seed) & set(bc_sub.seed))
            if len(common_seeds) < 2:
                print(f"  ⚠ {scen}: pas assez de seeds communs, ignoré")
                continue

            te_vals = te_sub[te_sub.seed.isin(common_seeds)].sort_values('seed').TE_L_med.values
            bc_vals = bc_sub[bc_sub.seed.isin(common_seeds)].sort_values('seed').BC_L_med.values

            # Gain de TE par rapport à BC (positif = TE meilleur que BC)
            gains = (bc_vals - te_vals) / bc_vals * 100
            mean_g, sd_g = gains.mean(), gains.std(ddof=1) if len(gains) > 1 else 0
            try:
                _, p = stats.wilcoxon(te_vals, bc_vals)
            except Exception:
                p = float('nan')

            sig = '★ TE bat BC' if (p < 0.05 and mean_g > 0) else \
                  '★ BC bat TE' if (p < 0.05 and mean_g < 0) else \
                  '  (pas de différence significative)'

            print(f"  {scen:<18} Δ(TE-vs-BC)={mean_g:+6.2f}%  p={p:.4f}  {sig}")

            all_rows.append({
                'baseline': te_name, 'scenario': scen,
                'gain_te_vs_bc_pct': mean_g, 'sd': sd_g,
                'wilcoxon_p': p, 'n_seeds': len(common_seeds),
                'verdict': sig.strip(),
            })

    df_summary = pd.DataFrame(all_rows)
    df_summary.to_csv('results_te_vs_bc/GABPO_TE_vs_BC_summary.csv', index=False)

    print(f"\n{'='*80}\nRÉSUMÉ FINAL\n{'='*80}")
    print(df_summary.to_string(index=False))

    print(f"\n✓ Sauvegardé : results_te_vs_bc/GABPO_TE_vs_BC_summary.csv")
    print(f"\n→ Télécharger : from google.colab import files; "
          f"files.download('results_te_vs_bc/GABPO_TE_vs_BC_summary.csv')")

    return df_bc, df_summary


if __name__ == "__main__":
    df_bc, df_summary = run_te_vs_bc_comparison(n_seeds=10)
