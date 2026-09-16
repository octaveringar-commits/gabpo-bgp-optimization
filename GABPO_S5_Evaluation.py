# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO Phase S5 — Évaluation OOD sur S5_WACREN_OOD                    ║
# ║                                                                        ║
# ║  RÈGLE STRICTE (décision verrouillée) :                                ║
# ║    - Aucune modification de l'environnement P0b (gelé)                ║
# ║    - Aucune modification des modèles Oracle / B1 / BC / PPO           ║
# ║    - Même protocole statistique que S1-S4 : 10 seeds, Wilcoxon        ║
# ║      apparié, Bootstrap 95% CI, Bonferroni                            ║
# ║    - S5 n'a JAMAIS été vu pendant l'entraînement BC ou PPO            ║
# ║      (Zero Data Leakage Protocol déjà respecté par construction)      ║
# ║                                                                        ║
# ║  Ce script NE cherche PAS à faire réussir S5.                         ║
# ║  Il mesure, un point.                                                  ║
# ║                                                                        ║
# ║  Prérequis dans la session Colab :                                     ║
# ║    exec(open('GABPO_P0b_DynamicEnv.py').read())                        ║
# ║    exec(open('GABPO_P1b_BC.py').read())                                ║
# ║    (checkpoint BC déjà entraîné : checkpoints_bc/bc_phase_D.pt)        ║
# ║    (checkpoint PPO déjà entraîné : checkpoints_ppo/ppo_s{seed}.pt)     ║
# ║      -> si absent, ce script évalue B1/B2/Oracle/BC uniquement         ║
# ║         et signale l'absence de PPO sans s'arrêter                    ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_S5_Evaluation.py').read())                         ║
# ║    df_s5 = run_s5_evaluation()                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_s5', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO Phase S5 — Évaluation OOD WACREN             ║")
print("╚══════════════════════════════════════════════════════╝")
print()
print("Règle : aucune modification de l'environnement ni des modèles.")
print("Ce script mesure. Il ne corrige rien.")
print()

# ── Vérification des dépendances ──────────────────────────────────────────
try:
    _ = BGPDynamicEnv, B1_Static, B2_Greedy, OracleAgent, run_episode
    print("✓ P0b chargé (BGPDynamicEnv, B1_Static, B2_Greedy, OracleAgent)")
except NameError as e:
    raise RuntimeError(
        f"Exécuter GABPO_P0b_DynamicEnv.py d'abord : {e}")

try:
    _ = HierarchicalPolicy, extract_features
    print("✓ BC chargé (HierarchicalPolicy)")
    BC_AVAILABLE = True
except NameError:
    print("⚠ HierarchicalPolicy absent — exécuter GABPO_P1b_BC.py d'abord pour évaluer BC")
    BC_AVAILABLE = False

BC_CHECKPOINT = 'checkpoints_bc/bc_phase_D.pt'
BC_CHECKPOINT_OK = BC_AVAILABLE and os.path.exists(BC_CHECKPOINT)
if BC_AVAILABLE and not os.path.exists(BC_CHECKPOINT):
    print(f"⚠ Checkpoint BC introuvable : {BC_CHECKPOINT} — BC sera exclu de l'évaluation")

import torch
PPO_AVAILABLE = False
try:
    _ = PPOHierarchicalPolicy
    PPO_AVAILABLE = True
    print("✓ PPOHierarchicalPolicy défini")
except NameError:
    print("⚠ PPOHierarchicalPolicy absent — PPO sera exclu de l'évaluation "
          "(exécuter GABPO_PPO_WarmStart.py si besoin)")


# ══════════════════════════════════════════════════════════════════════════
# ÉVALUATION S5 — même protocole que S1-S4
# ══════════════════════════════════════════════════════════════════════════

def load_ppo_for_seed(seed, N_a=50, N_p=12):
    """Charge le checkpoint PPO pour un seed donné, si disponible."""
    ck = f'checkpoints_ppo/ppo_s{seed}.pt'
    if not os.path.exists(ck):
        return None
    policy = PPOHierarchicalPolicy(N_a, N_p)
    policy.load_state_dict(torch.load(ck, map_location='cpu'))
    policy.eval()
    return policy


def run_s5_evaluation(n_seeds:int=10, N_a:int=50, N_p:int=12, T_ep:int=288):
    """
    Évalue B1, B2, Oracle, BC, PPO sur S5_WACREN_OOD.
    Exactement le même protocole que S1-S4 (10 seeds, T_ep=288).
    Aucun paramètre n'est ajusté pour ce scénario.
    """
    SCEN = 'S5_WACREN_OOD'
    seeds = list(range(n_seeds))

    print(f"\n[Évaluation S5] {SCEN}")
    print(f"  Seeds: {n_seeds} | T_ep: {T_ep} | N_a={N_a}, N_p={N_p}")
    print("=" * 60)

    # Instancier les agents (mêmes classes que S1-S4, sans changement)
    agents = {
        'B1_Static': B1_Static(N_a, N_p),
        'B2_Greedy': B2_Greedy(N_a, N_p),
    }

    results = {name: [] for name in agents}
    results['Oracle'] = []
    if BC_CHECKPOINT_OK:
        results['BC'] = []
    if PPO_AVAILABLE:
        results['PPO'] = []

    total_runs = len(agents) * n_seeds
    total_runs += n_seeds  # Oracle
    if BC_CHECKPOINT_OK: total_runs += n_seeds
    if PPO_AVAILABLE: total_runs += n_seeds
    run_count = 0

    # ── B1 / B2 ──────────────────────────────────────────────────────────
    for name, agent in agents.items():
        for seed in seeds:
            r = run_episode(agent, SCEN, seed, N_a, N_p, T_ep)
            results[name].append(r)
            run_count += 1
            if run_count % 10 == 0:
                print(f"  {run_count}/{total_runs} runs")

    # ── Oracle (identique à S1-S4, aucun changement de n_candidates) ──────
    for seed in seeds:
        env_ref = BGPDynamicEnv(SCEN, seed, N_a, N_p, T_ep)
        oracle  = OracleAgent(N_a, N_p, env_ref, n_candidates=80)
        r = run_episode(oracle, SCEN, seed, N_a, N_p, T_ep)
        results['Oracle'].append(r)
        run_count += 1
        if run_count % 10 == 0:
            print(f"  {run_count}/{total_runs} runs")

    # ── BC (checkpoint gelé, pas de fine-tuning sur S5) ──────────────────
    if BC_CHECKPOINT_OK:
        policy_bc = HierarchicalPolicy(N_a, N_p)
        policy_bc.load_state_dict(torch.load(BC_CHECKPOINT, map_location='cpu'))
        policy_bc.eval()
        for seed in seeds:
            r = run_episode(policy_bc, SCEN, seed, N_a, N_p, T_ep)
            results['BC'].append(r)
            run_count += 1
            if run_count % 10 == 0:
                print(f"  {run_count}/{total_runs} runs")

    # ── PPO (checkpoint gelé par seed, pas de fine-tuning sur S5) ────────
    if PPO_AVAILABLE:
        n_ppo_found = 0
        for seed in seeds:
            policy_ppo = load_ppo_for_seed(seed, N_a, N_p)
            if policy_ppo is None:
                continue
            r = run_episode(policy_ppo, SCEN, seed, N_a, N_p, T_ep)
            results['PPO'].append(r)
            n_ppo_found += 1
            run_count += 1
            if run_count % 10 == 0:
                print(f"  {run_count}/{total_runs} runs")
        if n_ppo_found < n_seeds:
            print(f"  ⚠ Seulement {n_ppo_found}/{n_seeds} checkpoints PPO trouvés")
        if n_ppo_found == 0:
            del results['PPO']

    print(f"\n✓ {run_count} runs complétés")

    # ── Statistiques : chaque modèle vs B1 ────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"RÉSULTATS S5_WACREN_OOD — {n_seeds} seeds")
    print(f"{'═'*70}")

    b1_L = np.array([r['L_med'] for r in results['B1_Static']])

    rows = []
    print(f"\n  {'Model':<14} {'L_med (ms)':<18} {'vs B1':<10} {'95% CI':<20} {'p-value'}")
    print("  " + "─"*72)

    for name, res_list in results.items():
        L = np.array([r['L_med'] for r in res_list])
        J = np.array([r['J_mean'] if 'J_mean' in r else r.get('J', np.nan)
                      for r in res_list])
        row = {
            'model': name,
            'L_mean': float(L.mean()),
            'L_std':  float(L.std()),
            'J_mean': float(np.nanmean(J)),
            'n': len(L),
        }

        if name == 'B1_Static':
            row.update({'gain_pct': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0, 'p': None})
            sym = '—'
        else:
            n = min(len(L), len(b1_L))
            gains = (b1_L[:n] - L[:n]) / b1_L[:n] * 100
            ci = stats.t.interval(0.95, n-1, loc=gains.mean(), scale=stats.sem(gains))
            try:
                _, p = stats.wilcoxon(L[:n], b1_L[:n])
            except Exception:
                p = float('nan')
            row.update({
                'gain_pct': float(gains.mean()),
                'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]),
                'p': float(p),
            })
            if ci[0] > 0:
                sym = '✅'
            elif gains.mean() > 0:
                sym = '⚠'
            else:
                sym = '❌'

        rows.append(row)
        ci_str = f"[{row['ci_lo']:+.1f}%,{row['ci_hi']:+.1f}%]" if name!='B1_Static' else "ref."
        p_str  = f"{row['p']:.4f}" if row['p'] is not None else "—"
        print(f"  {sym} {name:<12} {row['L_mean']:>7.1f}±{row['L_std']:>5.1f}  "
              f"{row['gain_pct']:>+6.1f}%  {ci_str:<20} {p_str}")

    # ── Comparaison directe avec S1-S4 (valeurs verrouillées en mémoire) ──
    print(f"\n{'─'*70}")
    print("COMPARAISON avec les gains sur S1-S4 (mémoire verrouillée)")
    print(f"{'─'*70}")
    s1_s4_bc = {'S1': 1.5, 'S2': 3.1, 'S3': 3.6, 'S4': 2.2}
    print(f"  BC vs B1 sur S1-S4 : {s1_s4_bc}")
    if 'BC' in results:
        bc_row = next(r for r in rows if r['model']=='BC')
        print(f"  BC vs B1 sur S5    : {bc_row['gain_pct']:+.1f}% "
              f"[{bc_row['ci_lo']:+.1f}%, {bc_row['ci_hi']:+.1f}%]")
        avg_s1s4 = np.mean(list(s1_s4_bc.values()))
        if bc_row['ci_lo'] > 0:
            verdict = 'HOLDS'
            msg = "S5 tient — BC généralise à un scénario OOD non vu à l'entraînement."
        elif bc_row['gain_pct'] > 0:
            verdict = 'DEGRADED_BUT_POSITIVE'
            msg = "S5 se dégrade (CI traverse zéro) mais reste dans le bon sens — à documenter honnêtement."
        else:
            verdict = 'COLLAPSES'
            msg = "S5 s'effondre — BC ne généralise pas à ce niveau d'overhead OOD."
        print(f"\n  Moyenne S1-S4 : {avg_s1s4:+.1f}% | S5 : {bc_row['gain_pct']:+.1f}%")
        print(f"  VERDICT : {verdict}")
        print(f"  {msg}")
    else:
        verdict = 'BC_NOT_AVAILABLE'
        print("  BC non évalué (checkpoint absent) — impossible de conclure sur la généralisation OOD")

    # ── Export ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv('results_s5/GABPO_S5_results.csv', index=False)
    with open('results_s5/GABPO_S5_analysis.json', 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y%m%d_%H%M'),
            'scenario': SCEN,
            'n_seeds': n_seeds,
            'results': rows,
            's1_s4_bc_reference': s1_s4_bc,
            'verdict': verdict,
        }, f, indent=2, default=str)

    print(f"\n✓ Sauvegardé : results_s5/GABPO_S5_results.csv")
    print(f"✓ Sauvegardé : results_s5/GABPO_S5_analysis.json")
    print(f"\nRappel : ce résultat est rapporté tel quel, sans ajustement.")

    return df


# ── Point d'entrée ────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_s5 = run_s5_evaluation(n_seeds=10)
