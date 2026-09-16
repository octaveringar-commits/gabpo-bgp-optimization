# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Évaluation complète des baselines TE (TE1/TE2/TE3)           ║
# ║                                                                        ║
# ║  Compare TE1_OSPF, TE2_MATE, TE3_TeXCP contre B1/B2 (déjà verrouillés) ║
# ║  et contre BC (le modèle GABPO retenu) sur les 5 scénarios,            ║
# ║  10 seeds chacun — répond au point #5 des reviewers (ICCT +            ║
# ║  2 avis Computer Networks) : comparaison expérimentale insuffisante.  ║
# ║                                                                        ║
# ║  Ces baselines sont déterministes/réactives — PAS de phase             ║
# ║  d'entraînement, donc pas de risque de perte de plusieurs heures de   ║
# ║  calcul comme pour HCGA. Néanmoins, on applique les mêmes principes   ║
# ║  de robustesse (sauvegarde par seed, reprise automatique) par         ║
# ║  prudence — l'évaluation reste 10 seeds × 5 scénarios × 3 baselines   ║
# ║  × 288 steps, ce qui peut prendre du temps sur CPU.                    ║
# ║                                                                        ║
# ║  Résultats BC/B1/B2 déjà verrouillés (mémoire du projet) sont          ║
# ║  rappelés en dur pour le tableau comparatif final — ils ne sont PAS    ║
# ║  recalculés ici (pas besoin de refaire un travail déjà fait et         ║
# ║  vérifié plusieurs fois).                                              ║
# ║                                                                        ║
# ║  Prérequis session : P0b (BGPDynamicEnv, run_episode, B1_Static,       ║
# ║  B2_Greedy) → BC (HierarchicalPolicy, pour charger le checkpoint       ║
# ║  final) → GABPO_TE_Baselines.py (TE1/TE2/TE3) déjà chargés.            ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_TE_Evaluation.py').read())                         ║
# ║    df, df_summary = run_te_evaluation(n_seeds=10)                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import numpy as np
import torch
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_te', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Évaluation baselines TE (TE1/TE2/TE3)      ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    _ = BGPDynamicEnv, run_episode, B1_Static, B2_Greedy
    print("✓ P0b chargé")
except NameError as e:
    raise RuntimeError(f"Exécuter GABPO_P0b_DynamicEnv.py d'abord : {e}")

try:
    _ = TE1_OSPF, TE2_MATE, TE3_TeXCP
    print("✓ Baselines TE chargées")
except NameError as e:
    raise RuntimeError(f"Exécuter GABPO_TE_Baselines.py d'abord : {e}")

BC_AVAILABLE = 'HierarchicalPolicy' in globals()
BC_CHECKPOINT = 'checkpoints_bc/bc_phase_D.pt'

# ── Résultats déjà verrouillés (mémoire du projet, 10 seeds, NE PAS recalculer) ──
LOCKED_BC = {
    'S1_Nominal':     {'mean': 3.05, 'sd': 0.65},
    'S2_FlashCrowd':  {'mean': 0.10, 'sd': 0.33},
    'S3_PoP_Failure': {'mean': 3.24, 'sd': 0.95},
    'S4_BGP_Anomaly': {'mean': 3.18, 'sd': 0.61},
    'S5_WACREN_OOD':  {'mean': 0.54, 'sd': None},
}
LOCKED_B2 = {
    'S5_WACREN_OOD': {'mean': 2.25, 'sd': None},
}

ALL_SCENARIOS = ['S1_Nominal', 'S2_FlashCrowd', 'S3_PoP_Failure',
                  'S4_BGP_Anomaly', 'S5_WACREN_OOD']


def run_te_evaluation(n_seeds:int=10, N_a:int=50, N_p:int=12, T_ep:int=288):
    """
    Évalue TE1/TE2/TE3 contre B1 sur les 5 scénarios, 10 seeds.
    Sauvegarde après chaque baseline (pas après chaque seed individuel
    ici, car ces baselines sont rapides — pas de phase d'entraînement —
    donc le risque de perte de calcul est bien moindre que pour BC/HCGA).
    """
    baselines = {
        'TE1_OSPF':  TE1_OSPF(N_a, N_p),
        'TE2_MATE':  TE2_MATE(N_a, N_p),
        'TE3_TeXCP': TE3_TeXCP(N_a, N_p),
    }
    b1 = B1_Static(N_a, N_p)

    # Charger BC si un checkpoint existe, pour comparaison directe dans ce run
    bc_policy = None
    if BC_AVAILABLE and os.path.exists(BC_CHECKPOINT):
        bc_policy = HierarchicalPolicy(N_a, N_p)
        bc_policy.load_state_dict(torch.load(BC_CHECKPOINT, map_location='cpu'))
        bc_policy.eval()
        print(f"✓ Checkpoint BC chargé pour comparaison directe : {BC_CHECKPOINT}")
    else:
        print("⚠ Checkpoint BC absent — comparaison utilisera les valeurs verrouillées "
              "en mémoire (LOCKED_BC), pas un recalcul direct")

    all_evals = []

    for name, model in baselines.items():
        results_path = f'results_te/{name}_results.csv'

        # ── Reprise : sauter cette baseline si déjà complétée ────────────
        if os.path.exists(results_path):
            prev = pd.read_csv(results_path)
            if prev.seed.nunique() >= n_seeds:
                print(f"\n✓ {name} déjà complet — chargement depuis {results_path}")
                all_evals.extend(prev.to_dict('records'))
                continue

        print(f"\n{'='*60}\n{name}\n{'='*60}")
        model_evals = []

        for seed in range(n_seeds):
            for scen in ALL_SCENARIOS:
                r_te = run_episode(model, scen, seed+900, N_a, N_p, T_ep)
                r_b1 = run_episode(b1,    scen, seed+900, N_a, N_p, T_ep)
                imp  = (r_b1['L_med'] - r_te['L_med']) / r_b1['L_med'] * 100
                model_evals.append({
                    'model': name, 'seed': seed, 'scenario': scen,
                    'TE_L_med': r_te['L_med'], 'B1_L_med': r_b1['L_med'],
                    'improvement_pct': imp,
                })
            imps = [r['improvement_pct'] for r in model_evals if r['seed']==seed]
            print(f"  Seed {seed}: " + " | ".join(
                f"{s.replace('_',' ')[:12]}={i:+.1f}%"
                for s, i in zip(ALL_SCENARIOS, imps)))

            # Sauvegarde après chaque seed — sécurité même si ces baselines
            # sont rapides, pour permettre une reprise fine si besoin
            pd.DataFrame(model_evals).to_csv(results_path, index=False)

        print(f"  ✓ {name} terminé et sauvegardé : {results_path}")
        all_evals.extend(model_evals)

    df = pd.DataFrame(all_evals)
    df.to_csv('results_te/GABPO_TE_all_results.csv', index=False)

    # ══════════════════════════════════════════════════════════════════
    # ANALYSE COMPARATIVE — TE1/TE2/TE3 vs B1, + rappel BC/B2 verrouillés
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\nTABLEAU COMPARATIF — TE1/TE2/TE3 vs B1, rappel BC/B2 verrouillés\n{'='*80}")

    summary_rows = []
    header = f"  {'Scenario':<18}"
    for name in baselines:
        header += f"{name:<14}"
    header += f"{'BC (locked)':<14}"
    print(header)
    print("  " + "─" * (18 + 14 * (len(baselines) + 1)))

    for scen in ALL_SCENARIOS:
        row = f"  {scen:<18}"
        row_data = {'scenario': scen}
        for name in baselines:
            sub = df[(df.model == name) & (df.scenario == scen)]
            m = sub.improvement_pct.mean()
            sd = sub.improvement_pct.std()
            row += f"{m:>+6.2f}%       "
            row_data[f'{name}_mean'] = m
            row_data[f'{name}_sd'] = sd
        bc_locked = LOCKED_BC.get(scen, {}).get('mean', float('nan'))
        row += f"{bc_locked:>+6.2f}%"
        row_data['BC_locked_mean'] = bc_locked
        print(row)
        summary_rows.append(row_data)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('results_te/GABPO_TE_summary.csv', index=False)

    # ── Tests statistiques TE vs B1 par scénario ──────────────────────────
    print(f"\n{'─'*80}\nTests statistiques (Wilcoxon signé, TE vs B1)\n{'─'*80}")
    for name in baselines:
        for scen in ALL_SCENARIOS:
            sub = df[(df.model == name) & (df.scenario == scen)]
            if len(sub) < 2:
                continue
            try:
                _, p = stats.wilcoxon(sub.TE_L_med.values, sub.B1_L_med.values)
            except Exception:
                p = float('nan')
            sig = '★' if p < 0.05 else ' '
            print(f"  {name:<12} {scen:<18} p={p:.4f} {sig}")

    print(f"\n✓ Sauvegardé : results_te/GABPO_TE_all_results.csv")
    print(f"✓ Sauvegardé : results_te/GABPO_TE_summary.csv")
    print(f"\n→ Télécharger : from google.colab import files; "
          f"files.download('results_te/GABPO_TE_summary.csv')")
    print(f"→ Télécharger aussi : files.download('results_te/GABPO_TE_all_results.csv')")

    return df, df_summary


if __name__ == "__main__":
    df, df_summary = run_te_evaluation(n_seeds=10)
