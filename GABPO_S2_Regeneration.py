# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Régénération S2 (Flash Crowd), 10 seeds individuels           ║
# ║                                                                        ║
# ║  Complète Table II : S1, S3, S4 ont déjà leurs 10 valeurs per-seed     ║
# ║  verrouillées ; S2 n'a qu'une moyenne agrégée (+3.10%, décision         ║
# ║  consciente Option C). Un avis de review externe identifie ce trou     ║
# ║  comme le seul gap concret et peu coûteux à corriger.                  ║
# ║                                                                        ║
# ║  Ce script :                                                           ║
# ║    - Entraîne UNIQUEMENT sur S2 (curriculum single-scenario, pas le    ║
# ║      curriculum complet S3→S4→S3+S4→S1-S4 du modèle final) — cohérent  ║
# ║      avec ce que S1/S3/S4 mesurent déjà : gain de BC (checkpoint       ║
# ║      final, curriculum complet) évalué SUR S2, pas un nouveau modèle   ║
# ║      entraîné spécifiquement sur S2 uniquement.                        ║
# ║    - RÉUTILISE le checkpoint BC déjà entraîné (bc_phase_D.pt, si       ║
# ║      présent) plutôt que de ré-entraîner — c'est la même politique     ║
# ║      finale qui a produit S1/S3/S4, on veut juste son évaluation sur   ║
# ║      S2 avec les 10 mêmes seeds, pas un nouveau modèle.                ║
# ║    - Si le checkpoint n'est pas disponible, propose de le régénérer    ║
# ║      via run_bc(quick=False) — mêmes prérequis que d'habitude.         ║
# ║                                                                        ║
# ║  Sauvegarde après CHAQUE seed (leçon HCGA/sensitivity), reprise        ║
# ║  automatique si interrompu.                                            ║
# ║                                                                        ║
# ║  Prérequis : P0b → extract_features patch → BC déjà chargés.           ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_S2_Regeneration.py').read())                       ║
# ║    df_s2, df_summary = run_s2_regeneration(n_seeds=10)                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import numpy as np
import torch
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_s2', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Régénération S2 Flash Crowd (10 seeds)     ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    _ = BGPDynamicEnv, run_episode, extract_features
    _ = HierarchicalPolicy, B1_Static
    print("✓ Dépendances chargées")
except NameError as e:
    raise RuntimeError(
        f"Exécuter P0b → extract_features patch → BC d'abord : {e}")

BC_CHECKPOINT = 'checkpoints_bc/bc_phase_D.pt'
BC_AVAILABLE = os.path.exists(BC_CHECKPOINT)
if not BC_AVAILABLE:
    print(f"⚠ Checkpoint BC introuvable : {BC_CHECKPOINT}")
    print("  Il faut le régénérer d'abord :")
    print("    results = run_bc(quick=False)  # ~30 min")
    print("  Puis relancer ce script.")
else:
    print(f"✓ Checkpoint BC trouvé : {BC_CHECKPOINT}")

# Référence verrouillée pour comparaison (S1/S3/S4 déjà connus)
LOCKED_REFERENCE = {
    'S1_Nominal':     {'mean': 3.05, 'ci': [2.59, 3.51], 'p': 0.00195},
    'S3_PoP_Failure': {'mean': 3.24, 'ci': [2.56, 3.92], 'p': 0.00195},
    'S4_BGP_Anomaly': {'mean': 3.18, 'ci': [2.74, 3.62], 'p': 0.00195},
    'S2_FlashCrowd':  {'mean': 3.10, 'ci': None, 'p': None},  # ce qu'on va compléter
}


def run_s2_regeneration(n_seeds:int=10, N_a:int=50, N_p:int=12, T_ep:int=288):
    """
    Évalue le checkpoint BC final (déjà entraîné, curriculum complet)
    sur S2_FlashCrowd, seed par seed — même protocole que S1/S3/S4
    (Table II), pas un nouvel entraînement spécifique à S2.
    """
    if not BC_AVAILABLE:
        raise RuntimeError(
            "Checkpoint BC absent — lancer run_bc(quick=False) d'abord.")

    SCEN = 'S2_FlashCrowd'
    results_path = f'results_s2/{SCEN}_results.csv'

    # ── Charger le modèle BC (une seule fois, poids gelés) ────────────────
    policy = HierarchicalPolicy(N_a, N_p)
    policy.load_state_dict(torch.load(BC_CHECKPOINT, map_location='cpu'))
    policy.eval()
    b1 = B1_Static(N_a, N_p)

    # ── Reprise : détecter les seeds déjà évalués ──────────────────────────
    evals = []
    seeds_done = set()
    if os.path.exists(results_path):
        prev = pd.read_csv(results_path)
        evals = prev.to_dict('records')
        seeds_done = set(prev.seed.unique())
        print(f"\n✓ Seeds déjà évalués : {sorted(seeds_done)}")

    seeds_to_run = [s for s in range(n_seeds) if s not in seeds_done]
    print(f"\n[Évaluation S2] Seeds restants : {seeds_to_run}")
    print("=" * 60)

    for seed in seeds_to_run:
        r_bc = run_episode(policy, SCEN, seed+900, N_a, N_p, T_ep)
        r_b1 = run_episode(b1,     SCEN, seed+900, N_a, N_p, T_ep)
        imp  = (r_b1['L_med'] - r_bc['L_med']) / r_b1['L_med'] * 100

        evals.append({
            'seed': seed, 'scenario': SCEN,
            'BC_L_med': r_bc['L_med'], 'B1_L_med': r_b1['L_med'],
            'improvement_pct': imp,
        })

        print(f"  Seed {seed}: BC={r_bc['L_med']:.1f}ms  B1={r_b1['L_med']:.1f}ms  "
              f"gain={imp:+.2f}%")

        # ── Sauvegarde après CHAQUE seed ────────────────────────────────
        pd.DataFrame(evals).to_csv(results_path, index=False)

    df = pd.DataFrame(evals)

    # ── Analyse statistique — même méthode que Table II (S1/S3/S4) ────────
    print(f"\n{'='*70}")
    print("ANALYSE S2 — méthode identique à Table II (t-interval + Wilcoxon)")
    print(f"{'='*70}")

    gains = df.improvement_pct.values
    mean_g = gains.mean()
    sd_g   = gains.std(ddof=1)
    ci     = stats.t.interval(0.95, len(gains)-1, loc=mean_g, scale=stats.sem(gains))
    try:
        _, p_wilcoxon = stats.wilcoxon(df.BC_L_med.values, df.B1_L_med.values)
    except Exception:
        p_wilcoxon = float('nan')
    dz = mean_g / sd_g
    # Cliff's delta (contre zéro, cohérent avec l'interprétation "10/10 seeds gagnants")
    cliff = sum(1 if g > 0 else (-1 if g < 0 else 0) for g in gains) / len(gains)

    print(f"\n  S2_FlashCrowd (n={len(gains)})")
    print(f"    Gains : {gains.tolist()}")
    print(f"    Moyenne ± SD : {mean_g:+.2f}% ± {sd_g:.2f}%")
    print(f"    95% CI       : [{ci[0]:+.2f}%, {ci[1]:+.2f}%]")
    print(f"    Wilcoxon p   : {p_wilcoxon:.5f}")
    print(f"    Cohen's dz   : {dz:+.2f}")
    print(f"    Cliff's Δ    : {cliff:+.2f}  "
          f"({'10/10 seeds gagnants' if cliff==1.0 else f'{sum(g>0 for g in gains)}/10 seeds gagnants'})")

    summary = {
        'scenario': SCEN, 'n': len(gains),
        'mean': mean_g, 'sd': sd_g,
        'ci_lo': ci[0], 'ci_hi': ci[1],
        'wilcoxon_p': p_wilcoxon, 'cohens_dz': dz, 'cliffs_delta': cliff,
    }
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv('results_s2/GABPO_S2_summary.csv', index=False)

    # ── Comparaison à la référence agrégée précédente ──────────────────────
    print(f"\n{'─'*70}")
    print(f"Comparaison à l'ancienne moyenne agrégée (non vérifiée) : +3.10%")
    print(f"Nouvelle moyenne (10 seeds individuels, vérifiés)        : {mean_g:+.2f}%")
    print(f"{'─'*70}")

    print(f"\n✓ Sauvegardé : {results_path}")
    print(f"✓ Sauvegardé : results_s2/GABPO_S2_summary.csv")
    print(f"\n→ Télécharger : from google.colab import files; "
          f"files.download('results_s2/GABPO_S2_summary.csv')")
    print(f"→ Télécharger aussi : files.download('{results_path}')")

    return df, df_summary


if __name__ == "__main__":
    df_s2, df_summary = run_s2_regeneration(n_seeds=10)
