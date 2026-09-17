# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Reconstruction indépendante de Table I                       ║
# ║  Oracle vs B1_Static sur S1-S4, 10 seeds                               ║
# ║                                                                        ║
# ║  Objectif : vérifier de façon indépendante et vérifiable les valeurs   ║
# ║  affichées dans Table I (+4.7%, -1.3%, +4.8%, +3.9%, p=0.0137/0.1520)  ║
# ║  du manuscrit ICCT, en utilisant EXACTEMENT le patron statistique      ║
# ║  identifié dans GABPO_S5_Evaluation.py :                              ║
# ║    1. gain par seed = (B1_L_med - Oracle_L_med) / B1_L_med × 100       ║
# ║    2. CI95% via stats.t.interval(0.95, n-1, loc=mean, scale=sem)      ║
# ║       (Student-t, PAS un bootstrap malgré ce que dit le commentaire   ║
# ║       du script S5 — divergence documentaire déjà identifiée)         ║
# ║    3. p-value via stats.wilcoxon(Oracle_L, B1_L) — apparié, NON        ║
# ║       corrigé à ce stade                                              ║
# ║    4. Correction Bonferroni appliquée EN AVAL, sur les 4 comparaisons  ║
# ║       S1-S4 ensemble (α'=0.05/4=0.0125, comme annoncé dans la légende ║
# ║       de Table I)                                                     ║
# ║                                                                        ║
# ║  Ce script ne prouve pas que Table I a été calculée EXACTEMENT ainsi, ║
# ║  mais permet de vérifier si ce patron reproduit — ou non — les        ║
# ║  valeurs publiées, ce qui départagera l'hypothèse de l'avis externe.  ║
# ║                                                                        ║
# ║  Sauvegarde par seed, reprise automatique — mêmes garanties que les    ║
# ║  autres scripts de ce projet.                                         ║
# ║                                                                        ║
# ║  Prérequis : P0b (BGPDynamicEnv, OracleAgent, run_episode, B1_Static)  ║
# ║  déjà chargé.                                                          ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_TableI_Reconstruction.py').read())                 ║
# ║    df, df_summary = run_table1_reconstruction(n_seeds=10)              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_table1_audit', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Reconstruction indépendante de Table I     ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    _ = BGPDynamicEnv, OracleAgent, run_episode, B1_Static
    print("✓ Dépendances P0b chargées")
except NameError as e:
    raise RuntimeError(f"Exécuter GABPO_P0b_DynamicEnv.py d'abord : {e}")

# Valeurs PUBLIÉES actuellement dans Table I (v11, après correction 0.0150→0.0137)
PUBLISHED_TABLE_I = {
    'S1_Nominal':     {'gain': 4.7,  'ci': (2.1, 7.3),  'p': 0.0137},
    'S2_FlashCrowd':  {'gain': -1.3, 'ci': (-4.2, 1.6), 'p': 0.1520},
    'S3_PoP_Failure': {'gain': 4.8,  'ci': (1.3, 8.3),  'p': 0.0137},
    'S4_BGP_Anomaly': {'gain': 3.9,  'ci': (0.7, 7.1),  'p': 0.0137},
}

ALL_SCENARIOS = ['S1_Nominal', 'S2_FlashCrowd', 'S3_PoP_Failure', 'S4_BGP_Anomaly']


def run_table1_reconstruction(n_seeds:int=10, N_a:int=50, N_p:int=12,
                              T_ep:int=288, n_candidates:int=80):
    """
    Reproduit Oracle vs B1 sur S1-S4, 10 seeds, avec le patron
    statistique identifié dans GABPO_S5_Evaluation.py.
    """
    results_path = 'results_table1_audit/oracle_vs_b1_results.csv'
    rows = []
    seeds_done = set()
    if os.path.exists(results_path):
        prev = pd.read_csv(results_path)
        rows = prev.to_dict('records')
        seeds_done = set(prev.seed.unique())
        print(f"\n✓ Seeds déjà évalués : {sorted(seeds_done)}")

    b1 = B1_Static(N_a, N_p)

    print(f"\n[Reconstruction] Oracle vs B1, {n_seeds} seeds, "
          f"{n_candidates} candidats Oracle/step")
    print("=" * 70)

    for seed in range(n_seeds):
        if seed in seeds_done:
            continue
        for scen in ALL_SCENARIOS:
            env = BGPDynamicEnv(scen, seed+900, N_a, N_p, T_ep)
            oracle = OracleAgent(N_a, N_p, env, n_candidates=n_candidates)

            r_oracle = run_episode(oracle, scen, seed+900, N_a, N_p, T_ep)
            r_b1     = run_episode(b1,     scen, seed+900, N_a, N_p, T_ep)

            imp = (r_b1['L_med'] - r_oracle['L_med']) / r_b1['L_med'] * 100
            rows.append({
                'seed': seed, 'scenario': scen,
                'Oracle_L_med': r_oracle['L_med'], 'B1_L_med': r_b1['L_med'],
                'gain_pct': imp,
            })

        seed_rows = [r for r in rows if r['seed']==seed]
        print(f"  Seed {seed}: " + " | ".join(
            f"{r['scenario'][:10]}={r['gain_pct']:+.2f}%" for r in seed_rows))

        pd.DataFrame(rows).to_csv(results_path, index=False)

    df = pd.DataFrame(rows)

    # ══════════════════════════════════════════════════════════════════
    # ANALYSE — patron exact identifié dans GABPO_S5_Evaluation.py
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("RECONSTRUCTION — patron S5 (t-interval + Wilcoxon, Bonferroni en aval)")
    print(f"{'='*90}")

    raw_pvalues = []
    summary_rows = []

    for scen in ALL_SCENARIOS:
        sub = df[df.scenario == scen].sort_values('seed')
        gains = sub.gain_pct.values
        oracle_L = sub.Oracle_L_med.values
        b1_L = sub.B1_L_med.values
        n = len(gains)

        mean_gain = gains.mean()
        ci = stats.t.interval(0.95, n-1, loc=mean_gain, scale=stats.sem(gains))
        try:
            _, p_raw = stats.wilcoxon(oracle_L, b1_L)
        except Exception:
            p_raw = float('nan')

        raw_pvalues.append(p_raw)
        summary_rows.append({
            'scenario': scen, 'mean_gain': mean_gain,
            'ci_lo': ci[0], 'ci_hi': ci[1], 'p_raw_wilcoxon': p_raw,
        })

    # Bonferroni EN AVAL sur les 4 comparaisons (comme suggéré par l'avis)
    n_tests = len(raw_pvalues)
    bonferroni_alpha = 0.05 / n_tests
    for i, row in enumerate(summary_rows):
        row['p_bonferroni_alpha'] = bonferroni_alpha
        row['significant_bonferroni'] = row['p_raw_wilcoxon'] < bonferroni_alpha

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('results_table1_audit/table1_reconstructed.csv', index=False)

    print(f"\nSeuil Bonferroni appliqué en aval : α' = 0.05/{n_tests} = {bonferroni_alpha:.4f}\n")
    print(f"{'Scenario':<18}{'Gain':<10}{'95% CI':<20}{'p (Wilcoxon)':<15}{'Sig. Bonf.'}")
    print("-" * 90)
    for row in summary_rows:
        sig = '★' if row['significant_bonferroni'] else ''
        print(f"{row['scenario']:<18}{row['mean_gain']:>+6.2f}%   "
              f"[{row['ci_lo']:>+.2f}%, {row['ci_hi']:>+.2f}%]   "
              f"{row['p_raw_wilcoxon']:<15.4f}{sig}")

    # ══════════════════════════════════════════════════════════════════
    # COMPARAISON avec les valeurs PUBLIÉES dans Table I
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("COMPARAISON — valeurs reconstruites vs valeurs PUBLIÉES (Table I, v11)")
    print(f"{'='*90}")

    comparison_rows = []
    for row in summary_rows:
        scen = row['scenario']
        pub = PUBLISHED_TABLE_I[scen]
        gain_match = abs(row['mean_gain'] - pub['gain']) < 0.5
        p_match = abs(row['p_raw_wilcoxon'] - pub['p']) < 0.01

        comparison_rows.append({
            'scenario': scen,
            'gain_reconstructed': row['mean_gain'], 'gain_published': pub['gain'],
            'gain_match': gain_match,
            'p_reconstructed': row['p_raw_wilcoxon'], 'p_published': pub['p'],
            'p_match': p_match,
        })
        status = '✓ COHÉRENT' if (gain_match and p_match) else '✗ DIVERGENT'
        print(f"\n  {scen}")
        print(f"    Gain  : reconstruit={row['mean_gain']:+.2f}%  "
              f"publié={pub['gain']:+.2f}%  {'✓' if gain_match else '✗'}")
        print(f"    p     : reconstruit={row['p_raw_wilcoxon']:.4f}  "
              f"publié={pub['p']:.4f}  {'✓' if p_match else '✗'}")
        print(f"    → {status}")

    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv('results_table1_audit/comparison_vs_published.csv', index=False)

    all_match = df_comparison[['gain_match', 'p_match']].all(axis=None)
    print(f"\n{'='*90}")
    if all_match:
        print("VERDICT : Les valeurs reconstruites sont COHÉRENTES avec Table I "
              "publiée — le patron S5 (t-interval + Wilcoxon + Bonferroni en aval) "
              "est probablement bien la méthode utilisée.")
    else:
        print("VERDICT : DIVERGENCE détectée entre les valeurs reconstruites et "
              "Table I publiée — Table I n'a probablement PAS été calculée avec "
              "exactement ce patron, ou les seeds/paramètres (n_candidates, T_ep) "
              "diffèrent de ceux utilisés à l'origine. Investiguer davantage "
              "avant de considérer Table I comme définitivement vérifiée.")
    print(f"{'='*90}")

    print(f"\n✓ Sauvegardé : results_table1_audit/table1_reconstructed.csv")
    print(f"✓ Sauvegardé : results_table1_audit/comparison_vs_published.csv")
    print(f"\n→ Télécharger : from google.colab import files; "
          f"files.download('results_table1_audit/comparison_vs_published.csv')")

    return df, df_summary


if __name__ == "__main__":
    df, df_summary = run_table1_reconstruction(n_seeds=10)
