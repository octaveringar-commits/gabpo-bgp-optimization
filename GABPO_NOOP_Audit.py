# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Audit : taux de NO-OP de BC vs avantage relatif TE1/TE2       ║
# ║                                                                        ║
# ║  Teste l'hypothèse : le simulateur P0b récompense une intervention     ║
# ║  BGP FORTE ET CONSTANTE (TE1=amplitude 1.0 à 100% des steps,           ║
# ║  TE2=quasi pareil) plutôt qu'une intervention MODULÉE/SÉLECTIVE        ║
# ║  (BC apprend un NO-OP fréquent + amplitude variable {0.25,0.5,1.0}).   ║
# ║                                                                        ║
# ║  Si l'hypothèse est correcte : le taux de NO-OP de BC devrait être     ║
# ║  PLUS ÉLEVÉ précisément sur les scénarios où TE1/TE2 le battent le     ║
# ║  plus largement (S1/S3/S4/S5), et PLUS FAIBLE sur S2 (le seul          ║
# ║  scénario où TE1 ne bat pas significativement BC).                    ║
# ║                                                                        ║
# ║  Mesure directement, pour chaque scénario × seed :                     ║
# ║    - taux de NO-OP de BC (fraction des steps où action == 0)          ║
# ║    - amplitude moyenne des actions non-NO-OP de BC                    ║
# ║    - fraction des steps où TE1/TE2 divergent de l'action BC            ║
# ║  puis corrèle ces métriques au gain relatif TE_i vs BC déjà mesuré     ║
# ║  (results_te_vs_bc/GABPO_TE_vs_BC_summary.csv).                        ║
# ║                                                                        ║
# ║  Prérequis : P0b, BC (checkpoint chargé), TE1/TE2/TE3, et le fichier   ║
# ║  results_te_vs_bc/GABPO_TE_vs_BC_summary.csv déjà présent (résultat    ║
# ║  du script précédent GABPO_TE_vs_BC_Direct.py).                       ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_NOOP_Audit.py').read())                            ║
# ║    df_audit = run_noop_audit(n_seeds=10)                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import numpy as np
import torch
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_noop_audit', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Audit taux de NO-OP vs avantage TE1/TE2    ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    _ = BGPDynamicEnv, HierarchicalPolicy
except NameError as e:
    raise RuntimeError(f"Exécuter P0b puis BC d'abord : {e}")

BC_CHECKPOINT = 'checkpoints_bc/bc_phase_D.pt'
if not os.path.exists(BC_CHECKPOINT):
    raise RuntimeError(f"Checkpoint BC introuvable : {BC_CHECKPOINT}")

TE_SUMMARY_PATH = 'results_te_vs_bc/GABPO_TE_vs_BC_summary.csv'
if not os.path.exists(TE_SUMMARY_PATH):
    print(f"⚠ {TE_SUMMARY_PATH} introuvable — le script fonctionnera mais "
          f"sans la corrélation finale avec les résultats TE déjà obtenus")

ALL_SCENARIOS = ['S1_Nominal', 'S2_FlashCrowd', 'S3_PoP_Failure',
                  'S4_BGP_Anomaly', 'S5_WACREN_OOD']


def measure_bc_noop_stats(policy, scenario, seed, N_a=50, N_p=12, T_ep=288):
    """
    Fait tourner BC sur un épisode et mesure, step par step :
      - le taux de NO-OP (fraction des actions entièrement nulles)
      - l'amplitude moyenne des actions non-NO-OP (valeur absolue moyenne
        des entrées non-nulles)
      - le nombre de PoPs distincts modifiés au cours de l'épisode
        (BC modifie-t-il toujours le même PoP, ou change-t-il de cible ?)
    """
    env = BGPDynamicEnv(scenario, seed, N_a, N_p, T_ep)
    obs = env.reset()

    n_noop = 0
    n_total = 0
    amplitudes = []
    pops_touched = set()

    for _ in range(T_ep):
        action = policy.select_action(obs)
        n_total += 1
        nonzero = action[action != 0]
        if len(nonzero) == 0:
            n_noop += 1
        else:
            amplitudes.append(np.abs(nonzero).mean())
            # Identifier quel(s) PoP(s) ont été modifiés (indices dans la
            # moitié LOCAL_PREF du vecteur, la plus utilisée par BC)
            off = N_p * N_a
            for p in range(N_p):
                if np.any(action[off + p*N_a:off + (p+1)*N_a] != 0) or \
                   np.any(action[p*N_a:(p+1)*N_a] != 0):
                    pops_touched.add(p)

        obs, _, done, _ = env.step(action)
        if done:
            break

    noop_rate = n_noop / n_total
    mean_amplitude = float(np.mean(amplitudes)) if amplitudes else 0.0

    return {
        'noop_rate': noop_rate,
        'mean_amplitude_when_active': mean_amplitude,
        'n_distinct_pops_touched': len(pops_touched),
        'n_total_steps': n_total,
    }


def run_noop_audit(n_seeds:int=10, N_a:int=50, N_p:int=12, T_ep:int=288):
    """
    Mesure les statistiques NO-OP de BC pour chaque (scenario, seed),
    puis agrège par scénario et corrèle avec le gain relatif TE vs BC
    déjà obtenu (si le fichier existe).
    """
    policy = HierarchicalPolicy(N_a, N_p)
    policy.load_state_dict(torch.load(BC_CHECKPOINT, map_location='cpu'))
    policy.eval()
    print("✓ Checkpoint BC chargé")

    results_path = 'results_noop_audit/bc_noop_stats.csv'
    rows = []
    seeds_done = set()
    if os.path.exists(results_path):
        prev = pd.read_csv(results_path)
        rows = prev.to_dict('records')
        seeds_done = set(prev.seed.unique())
        print(f"✓ Seeds déjà mesurés : {sorted(seeds_done)}")

    print(f"\n[Mesure NO-OP] BC sur {len(ALL_SCENARIOS)} scénarios × {n_seeds} seeds")
    print("=" * 70)

    for seed in range(n_seeds):
        if seed in seeds_done:
            continue
        for scen in ALL_SCENARIOS:
            stats_bc = measure_bc_noop_stats(policy, scen, seed+900, N_a, N_p, T_ep)
            rows.append({'seed': seed, 'scenario': scen, **stats_bc})

        seed_rows = [r for r in rows if r['seed']==seed]
        print(f"  Seed {seed}: " + " | ".join(
            f"{r['scenario'].replace('_','')[:10]}=NOOP{r['noop_rate']:.0%}"
            for r in seed_rows))

        # Sauvegarde après chaque seed
        pd.DataFrame(rows).to_csv(results_path, index=False)

    df = pd.DataFrame(rows)

    # ── Agrégation par scénario ──────────────────────────────────────────
    print(f"\n{'='*80}\nTAUX DE NO-OP DE BC PAR SCÉNARIO (moyenne sur {n_seeds} seeds)\n{'='*80}")
    agg = df.groupby('scenario').agg(
        noop_rate_mean=('noop_rate', 'mean'),
        noop_rate_sd=('noop_rate', 'std'),
        amplitude_mean=('mean_amplitude_when_active', 'mean'),
        n_pops_mean=('n_distinct_pops_touched', 'mean'),
    ).reindex(ALL_SCENARIOS)

    print(agg.to_string())

    agg.to_csv('results_noop_audit/bc_noop_summary_by_scenario.csv')

    # ── Corrélation avec le gain relatif TE1/TE2 vs BC ──────────────────
    if os.path.exists(TE_SUMMARY_PATH):
        print(f"\n{'='*80}\nCORRÉLATION : taux de NO-OP de BC vs avantage TE1/TE2 sur BC\n{'='*80}")
        te_summary = pd.read_csv(TE_SUMMARY_PATH)

        combined_rows = []
        for scen in ALL_SCENARIOS:
            noop_rate = agg.loc[scen, 'noop_rate_mean']
            te1_gain = te_summary[(te_summary.baseline=='TE1_OSPF') &
                                   (te_summary.scenario==scen)].gain_te_vs_bc_pct.values
            te2_gain = te_summary[(te_summary.baseline=='TE2_MATE') &
                                   (te_summary.scenario==scen)].gain_te_vs_bc_pct.values
            te1_gain = te1_gain[0] if len(te1_gain) else float('nan')
            te2_gain = te2_gain[0] if len(te2_gain) else float('nan')
            combined_rows.append({
                'scenario': scen, 'bc_noop_rate': noop_rate,
                'te1_gain_over_bc': te1_gain, 'te2_gain_over_bc': te2_gain,
            })
            print(f"  {scen:<18} BC_NOOP_rate={noop_rate:.1%}   "
                  f"TE1_gain_vs_BC={te1_gain:+.2f}%   TE2_gain_vs_BC={te2_gain:+.2f}%")

        df_combined = pd.DataFrame(combined_rows)
        df_combined.to_csv('results_noop_audit/noop_vs_te_advantage_correlation.csv', index=False)

        # Corrélation de Spearman (rang) — plus robuste avec seulement 5 points
        valid = df_combined.dropna()
        if len(valid) >= 3:
            corr_te1, p_te1 = stats.spearmanr(valid.bc_noop_rate, valid.te1_gain_over_bc)
            corr_te2, p_te2 = stats.spearmanr(valid.bc_noop_rate, valid.te2_gain_over_bc)
            print(f"\n  Corrélation Spearman (NO-OP rate ↔ avantage TE1 sur BC) : "
                  f"ρ={corr_te1:+.3f}, p={p_te1:.3f}")
            print(f"  Corrélation Spearman (NO-OP rate ↔ avantage TE2 sur BC) : "
                  f"ρ={corr_te2:+.3f}, p={p_te2:.3f}")
            print(f"\n  Interprétation : ρ>0 signifierait que BC intervient MOINS "
                  f"(NO-OP élevé) précisément là où TE1/TE2 le battent le plus —"
                  f" ce qui SOUTIENDRAIT l'hypothèse d'intervention "
                  f"forte-et-constante vs modulée-et-sélective.")
        else:
            print("  ⚠ Pas assez de points valides pour la corrélation")

    print(f"\n✓ Sauvegardé : results_noop_audit/bc_noop_summary_by_scenario.csv")
    print(f"✓ Sauvegardé : results_noop_audit/noop_vs_te_advantage_correlation.csv")
    print(f"\n→ Télécharger : from google.colab import files; "
          f"files.download('results_noop_audit/noop_vs_te_advantage_correlation.csv')")

    return df


if __name__ == "__main__":
    df_audit = run_noop_audit(n_seeds=10)
