# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Reconstruction Table I v2 : ENVIRONNEMENT ET PARAMÈTRES       ║
# ║  D'ORIGINE (GABPO_P0_Oracle.py), pas P0b                              ║
# ║                                                                        ║
# ║  Suite à l'investigation (a) : GABPO_P0_Oracle.py a été identifié      ║
# ║  comme le script probable d'origine de Table I, avec deux paramètres  ║
# ║  DIFFÉRENTS de ceux utilisés dans la première tentative de            ║
# ║  reconstruction (GABPO_TableI_Reconstruction.py, qui utilisait P0b) : ║
# ║                                                                        ║
# ║    - seeds = 0..9 DIRECTEMENT (pas seed+900, convention apparue       ║
# ║      seulement dans P0b, plus tard)                                   ║
# ║    - n_candidates = 50 pour l'Oracle (pas 80)                         ║
# ║    - environnement BGPEnvP0 et agents AgentOracle/AgentB1_Static      ║
# ║      d'origine (PAS BGPDynamicEnv/OracleAgent/B1_Static de P0b —      ║
# ║      le simulateur lui-même a pu évoluer entre les deux versions)     ║
# ║                                                                        ║
# ║  CADRE FALSIFIABLE (tel que convenu) : ce script teste OBJECTIVEMENT  ║
# ║  si ces paramètres et cet environnement reproduisent Table I. Il ne   ║
# ║  modifie RIEN pour "faire rentrer" le résultat — le verdict peut       ║
# ║  être COHÉRENT ou DIVERGENT, sans présupposer lequel.                 ║
# ║                                                                        ║
# ║  Ce script réutilise DIRECTEMENT les classes de GABPO_P0_Oracle.py    ║
# ║  (à charger avant celui-ci) — pas de redéfinition, pas de             ║
# ║  réinterprétation.                                                     ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_P0_Oracle.py').read())                            ║
# ║    exec(open('GABPO_TableI_Reconstruction_v2_original.py').read())    ║
# ║    df, df_summary = run_table1_reconstruction_v2()                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_table1_audit', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Reconstruction Table I v2 (env. ORIGINAL)  ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    _ = BGPEnvP0, AgentOracle, AgentB1_Static, run_episode
    print("✓ Classes ORIGINALES (GABPO_P0_Oracle.py) chargées")
except NameError as e:
    raise RuntimeError(
        f"Exécuter GABPO_P0_Oracle.py d'abord (PAS P0b) : {e}")

# Valeurs PUBLIÉES actuellement dans Table I (v11)
PUBLISHED_TABLE_I = {
    'S1_Nominal':     {'gain': 4.7,  'ci': (2.1, 7.3),  'p': 0.0137},
    'S2_FlashCrowd':  {'gain': -1.3, 'ci': (-4.2, 1.6), 'p': 0.1520},
    'S3_PoP_Failure': {'gain': 4.8,  'ci': (1.3, 8.3),  'p': 0.0137},
    'S4_BGP_Anomaly': {'gain': 3.9,  'ci': (0.7, 7.1),  'p': 0.0137},
}

ALL_SCENARIOS = ['S1_Nominal', 'S2_FlashCrowd', 'S3_PoP_Failure', 'S4_BGP_Anomaly']

# ── Paramètres ORIGINAUX identifiés dans GABPO_P0_Oracle.py::run_p0() ────
ORIGINAL_N_CANDIDATES = 50
ORIGINAL_SEEDS = list(range(10))  # 0..9 DIRECTEMENT, pas seed+900


def run_table1_reconstruction_v2(N_a:int=50, N_p:int=12, T_ep:int=288):
    """
    Reproduit Oracle vs B1 sur S1-S4 en utilisant EXACTEMENT
    l'environnement, les agents, et les paramètres trouvés dans
    GABPO_P0_Oracle.py::run_p0() — pas P0b, pas seed+900, pas
    n_candidates=80.
    """
    results_path = 'results_table1_audit/oracle_vs_b1_v2_original.csv'
    rows = []
    seeds_done = set()
    if os.path.exists(results_path):
        prev = pd.read_csv(results_path)
        rows = prev.to_dict('records')
        seeds_done = set(prev.seed.unique())
        print(f"\n✓ Seeds déjà évalués : {sorted(seeds_done)}")

    b1 = AgentB1_Static(N_a, N_p)
    env_ref = BGPEnvP0('S1_Nominal', 0, N_a, N_p, T_ep)
    oracle = AgentOracle(N_a, N_p, env_ref, n_candidates=ORIGINAL_N_CANDIDATES)

    print(f"\n[Reconstruction v2] Environnement ORIGINAL (BGPEnvP0), "
          f"{ORIGINAL_N_CANDIDATES} candidats Oracle, seeds={ORIGINAL_SEEDS}")
    print("=" * 70)

    for seed in ORIGINAL_SEEDS:
        if seed in seeds_done:
            continue
        for scen in ALL_SCENARIOS:
            r_oracle = run_episode(oracle, scen, seed, N_a, N_p, T_ep)
            r_b1     = run_episode(b1,     scen, seed, N_a, N_p, T_ep)

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
    # ANALYSE — même patron statistique que la v1 (t-interval + Wilcoxon)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("RECONSTRUCTION v2 — environnement/paramètres ORIGINAUX")
    print(f"{'='*90}")

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

        summary_rows.append({
            'scenario': scen, 'mean_gain': mean_gain,
            'ci_lo': ci[0], 'ci_hi': ci[1], 'p_raw_wilcoxon': p_raw,
        })

    bonferroni_alpha = 0.05 / len(ALL_SCENARIOS)
    for row in summary_rows:
        row['significant_bonferroni'] = row['p_raw_wilcoxon'] < bonferroni_alpha

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('results_table1_audit/table1_reconstructed_v2.csv', index=False)

    print(f"\n{'Scenario':<18}{'Gain':<10}{'95% CI':<22}{'p (Wilcoxon)':<15}{'Sig.'}")
    print("-" * 90)
    for row in summary_rows:
        sig = '★' if row['significant_bonferroni'] else ''
        print(f"{row['scenario']:<18}{row['mean_gain']:>+6.2f}%   "
              f"[{row['ci_lo']:>+.2f}%, {row['ci_hi']:>+.2f}%]   "
              f"{row['p_raw_wilcoxon']:<15.4f}{sig}")

    # ══════════════════════════════════════════════════════════════════
    # COMPARAISON — v2 (env. original) vs Table I publiée
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("COMPARAISON v2 — env./params ORIGINAUX vs Table I PUBLIÉE")
    print(f"{'='*90}")

    comparison_rows = []
    for row in summary_rows:
        scen = row['scenario']
        pub = PUBLISHED_TABLE_I[scen]
        gain_match = abs(row['mean_gain'] - pub['gain']) < 0.5
        p_match = abs(row['p_raw_wilcoxon'] - pub['p']) < 0.01

        comparison_rows.append({
            'scenario': scen,
            'gain_v2': row['mean_gain'], 'gain_published': pub['gain'],
            'gain_match': gain_match,
            'p_v2': row['p_raw_wilcoxon'], 'p_published': pub['p'],
            'p_match': p_match,
        })
        status = '✓ COHÉRENT' if (gain_match and p_match) else '✗ DIVERGENT'
        print(f"\n  {scen}")
        print(f"    Gain : v2={row['mean_gain']:+.2f}%  publié={pub['gain']:+.2f}%  "
              f"{'✓' if gain_match else '✗'}")
        print(f"    p    : v2={row['p_raw_wilcoxon']:.4f}  publié={pub['p']:.4f}  "
              f"{'✓' if p_match else '✗'}")
        print(f"    → {status}")

    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv('results_table1_audit/comparison_v2_vs_published.csv', index=False)

    all_match = df_comparison[['gain_match', 'p_match']].all(axis=None)
    print(f"\n{'='*90}")
    if all_match:
        print("VERDICT v2 : COHÉRENT SUR TOUS LES SCÉNARIOS — l'environnement et "
              "les paramètres d'origine (BGPEnvP0, n_candidates=50, seeds 0-9 "
              "directs) REPRODUISENT Table I. Chaîne de calcul CONFIRMÉE : "
              "option (a) de l'audit réussie.")
    else:
        n_matches = sum(1 for r in comparison_rows if r['gain_match'] and r['p_match'])
        print(f"VERDICT v2 : {n_matches}/4 scénarios cohérents — "
              f"{'AMÉLIORATION' if n_matches > 1 else 'PAS D AMÉLIORATION'} "
              f"par rapport à la v1 (1/4 cohérent). "
              f"{'Chaîne de calcul PARTIELLEMENT confirmée' if n_matches >= 3 else 'Divergence substantielle persistante'} "
              f"— {'considérer option (a) comme suffisamment établie' if n_matches >= 3 else 'basculer vers option (b), recalcul avec protocole documenté'}.")
    print(f"{'='*90}")

    print(f"\n✓ Sauvegardé : results_table1_audit/table1_reconstructed_v2.csv")
    print(f"✓ Sauvegardé : results_table1_audit/comparison_v2_vs_published.csv")
    print(f"\n→ Télécharger : from google.colab import files; "
          f"files.download('results_table1_audit/comparison_v2_vs_published.csv')")

    return df, df_summary


if __name__ == "__main__":
    df, df_summary = run_table1_reconstruction_v2()
