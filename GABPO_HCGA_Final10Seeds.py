# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Test décisif HCGA : 10 seeds appariés                        ║
# ║                                                                        ║
# ║  Dernier test avant de fermer ou garder HCGA pour AJSE. Protocole      ║
# ║  strict, aucune modification d'architecture depuis le pilote 2 seeds   ║
# ║  (échelle √d déjà fixée, ne plus y toucher).                           ║
# ║                                                                        ║
# ║  Pour chaque seed 0..9 : entraîner BC_sans_HCGA et BC_avec_HCGA sur    ║
# ║  le MÊME dataset Oracle, même epochs, même lr, même batch size,        ║
# ║  même décodeur, même évaluation S1/S3/S4.                              ║
# ║                                                                        ║
# ║  Δ_s = Gain_HCGA,s − Gain_BC,s   pour chaque seed s, chaque scénario   ║
# ║                                                                        ║
# ║  Analyse : moyenne Δ, médiane Δ, SD, IC95%, Wilcoxon apparié,          ║
# ║  P5-P95, ET variance inter-seed de chaque modèle séparément (pas       ║
# ║  seulement le delta) — pour distinguer "HCGA moins performant" de      ║
# ║  "HCGA moins performant mais plus stable".                             ║
# ║                                                                        ║
# ║  Règle de décision (pas automatique sur le seul signe de Δ) :          ║
# ║    Δ>0 et significatif       → HCGA devient contribution démontrée     ║
# ║    Δ<0 et significatif       → HCGA fermé                              ║
# ║    Δ≈0 (non significatif)    → HCGA retiré de la contribution centrale ║
# ║    Δ non signif. mais SD_HCGA << SD_BC → analyse robustesse à part,    ║
# ║      pas de conclusion automatique sur la seule variance               ║
# ║                                                                        ║
# ║  Prérequis session : P0b → extract_features patch → BC →               ║
# ║  GABPO_Ablation_B3B6.py → GABPO_HCGA_SanityCheck.py → échelle √d        ║
# ║  déjà restaurée (vérifié automatiquement au démarrage).                ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_HCGA_Final10Seeds.py').read())                     ║
# ║    df, verdict = run_hcga_final(n_seeds=10)                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import inspect
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_hcga_final', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Test décisif HCGA : 10 seeds appariés      ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    _ = BGPDynamicEnv, HierarchicalAction, extract_features
    _ = HierarchicalPolicy, HierarchicalPolicy_HCGA
    _ = generate_dataset_with_obs, run_episode, B1_Static
    print("✓ Dépendances chargées")
except NameError as e:
    raise RuntimeError(
        f"Exécuter P0b → extract_features patch → BC → "
        f"GABPO_Ablation_B3B6.py → GABPO_HCGA_SanityCheck.py d'abord : {e}")

# ── Vérifier l'échelle √d avant de lancer quoi que ce soit ────────────────
_test_model = HierarchicalPolicy_HCGA(5, 3)
_hcga_src = inspect.getsource(_test_model._hcga)
if '/ 2.0' in _hcga_src or '/2.0' in _hcga_src:
    raise RuntimeError(
        "ARRÊT : l'échelle /2.0 est active — réappliquer le patch √d "
        "avant de lancer le test décisif. Ne pas changer l'architecture "
        "en cours de route, comme convenu.")
elif 'self.d ** 0.5' in _hcga_src or 'd**0.5' in _hcga_src:
    print("✓ Échelle √d confirmée — architecture gelée pour ce test")
else:
    print("⚠ Échelle non identifiée automatiquement — vérifier manuellement "
          "avant de continuer")
del _test_model


def get_logits_generic(model, obs):
    """Interface unifiée BC sans HCGA (forward(feat)) / avec HCGA (get_logits(obs))."""
    if hasattr(model, 'get_logits'):
        return model.get_logits(obs)
    else:
        feat = model._get_features(obs)
        return model.forward(feat)


def train_bc_style(model, dataset, n_epochs=100, batch_size=128, lr=1e-3):
    """Entraînement BC identique pour les deux modèles — ne pas modifier."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(dataset)
    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start+batch_size]
            losses = []
            for i in batch_idx:
                d = dataset[i]
                noop_l, pop_l, type_l, amp_l = get_logits_generic(model, d['obs_raw'])
                l_noop = F.cross_entropy(noop_l.unsqueeze(0), torch.LongTensor([d['noop']]))
                l_pop  = F.cross_entropy(pop_l.unsqueeze(0),  torch.LongTensor([d['pop']]))
                l_type = F.cross_entropy(type_l.unsqueeze(0), torch.LongTensor([d['type']]))
                l_amp  = F.cross_entropy(amp_l.unsqueeze(0),  torch.LongTensor([d['amp']]))
                loss = 2.0*l_noop + 1.5*l_pop + 1.0*l_type + 0.5*l_amp
                losses.append(loss)
            batch_loss = torch.stack(losses).mean()
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def run_hcga_final(n_seeds:int=10, n_episodes:int=100, n_epochs:int=100,
                   N_a:int=50, N_p:int=12):
    """
    Test décisif : 10 seeds appariés BC_sans_HCGA vs BC_avec_HCGA.
    Dataset Oracle régénéré une fois, partagé entre les deux modèles
    ET entre tous les seeds (déterministe — seed=si*1000+ep dans
    generate_dataset_with_obs, indépendant du seed d'entraînement ici).
    """
    scenarios_train = ['S3_PoP_Failure', 'S4_BGP_Anomaly']
    all_scenarios   = scenarios_train + ['S1_Nominal']

    print(f"\n[Test décisif] {n_seeds} seeds appariés, {n_epochs} epochs")
    print("=" * 60)

    print(f"\n[Dataset Oracle] Génération (partagé)...")
    dataset = generate_dataset_with_obs(
        scenarios_train, n_episodes, N_a, N_p, T_ep=72)
    print(f"  ✓ {len(dataset)} samples")

    b1 = B1_Static(N_a, N_p)
    all_evals = []

    for model_name, ModelClass in [('BC_sans_HCGA', HierarchicalPolicy),
                                     ('BC_avec_HCGA', HierarchicalPolicy_HCGA)]:
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")

        for seed in range(n_seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            model = ModelClass(N_a, N_p)
            model = train_bc_style(model, dataset, n_epochs=n_epochs)

            ck = f'results_hcga_final/{model_name}_seed{seed}.pt'
            torch.save(model.state_dict(), ck)

            for scen in all_scenarios:
                r_m  = run_episode(model, scen, seed+900, N_a, N_p, T_ep=288)
                r_b1 = run_episode(b1,    scen, seed+900, N_a, N_p, T_ep=288)
                imp  = (r_b1['L_med']-r_m['L_med'])/r_b1['L_med']*100
                all_evals.append({
                    'model': model_name, 'seed': seed, 'scenario': scen,
                    'L_med': r_m['L_med'], 'B1_L_med': r_b1['L_med'],
                    'improvement_pct': imp,
                })

            imps = [r['improvement_pct'] for r in all_evals
                    if r['model']==model_name and r['seed']==seed]
            print(f"  Seed {seed} : " +
                  " | ".join(f"{s}={i:+.1f}%" for s, i in
                             zip(all_scenarios, imps)))

        # Sauvegarde intermédiaire après chaque modèle — sécurité coupure
        pd.DataFrame(all_evals).to_csv(
            'results_hcga_final/GABPO_HCGA_final_partial.csv', index=False)
        print(f"  ✓ Sauvegarde intermédiaire OK")

    df = pd.DataFrame(all_evals)
    df.to_csv('results_hcga_final/GABPO_HCGA_final_results.csv', index=False)

    # ══════════════════════════════════════════════════════════════════
    # ANALYSE — Δ appariés + variance inter-seed séparée
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*76}")
    print("ANALYSE DÉCISIVE — Δ appariés (HCGA − BC), variance, tests")
    print(f"{'='*76}")

    summary_rows = []
    for scen in all_scenarios:
        bc_vals   = df[(df.model=='BC_sans_HCGA') & (df.scenario==scen)
                       ].sort_values('seed').improvement_pct.values
        hcga_vals = df[(df.model=='BC_avec_HCGA') & (df.scenario==scen)
                       ].sort_values('seed').improvement_pct.values

        deltas = hcga_vals - bc_vals  # apparié par seed
        mean_delta   = float(deltas.mean())
        median_delta = float(np.median(deltas))
        sd_delta     = float(deltas.std())
        ci_delta     = stats.t.interval(0.95, len(deltas)-1,
                                        loc=mean_delta, scale=stats.sem(deltas))
        p5_95        = (float(np.percentile(deltas, 5)),
                        float(np.percentile(deltas, 95)))
        try:
            _, p_wilcoxon = stats.wilcoxon(hcga_vals, bc_vals)
        except Exception:
            p_wilcoxon = float('nan')

        sd_bc   = float(bc_vals.std())
        sd_hcga = float(hcga_vals.std())

        print(f"\n  {scen}")
        print(f"    BC   : {bc_vals.mean():+.2f}% ± {sd_bc:.2f}% "
              f"(seeds: {np.round(bc_vals,1).tolist()})")
        print(f"    HCGA : {hcga_vals.mean():+.2f}% ± {sd_hcga:.2f}% "
              f"(seeds: {np.round(hcga_vals,1).tolist()})")
        print(f"    Δ moyen    : {mean_delta:+.3f}%")
        print(f"    Δ médian   : {median_delta:+.3f}%")
        print(f"    Δ SD       : {sd_delta:.3f}%")
        print(f"    Δ IC95%    : [{ci_delta[0]:+.3f}%, {ci_delta[1]:+.3f}%]")
        print(f"    Δ P5-P95   : [{p5_95[0]:+.3f}%, {p5_95[1]:+.3f}%]")
        print(f"    Wilcoxon p : {p_wilcoxon:.4f}")
        print(f"    SD_BC={sd_bc:.2f}% vs SD_HCGA={sd_hcga:.2f}% "
              f"(ratio HCGA/BC = {sd_hcga/(sd_bc+1e-8):.2f})")

        summary_rows.append({
            'scenario': scen, 'bc_mean': bc_vals.mean(), 'bc_sd': sd_bc,
            'hcga_mean': hcga_vals.mean(), 'hcga_sd': sd_hcga,
            'delta_mean': mean_delta, 'delta_median': median_delta,
            'delta_sd': sd_delta, 'ci_lo': ci_delta[0], 'ci_hi': ci_delta[1],
            'p5': p5_95[0], 'p95': p5_95[1], 'wilcoxon_p': p_wilcoxon,
            'sd_ratio_hcga_bc': sd_hcga/(sd_bc+1e-8),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('results_hcga_final/GABPO_HCGA_final_summary.csv', index=False)

    # ── Verdict global (pas automatique sur signe seul) ──────────────────
    print(f"\n{'='*76}")
    print("VERDICT GLOBAL")
    print(f"{'='*76}")

    avg_delta   = df_summary.delta_mean.mean()
    all_sig_neg = all((r.wilcoxon_p < 0.05 and r.delta_mean < 0)
                       for r in df_summary.itertuples())
    all_sig_pos = all((r.wilcoxon_p < 0.05 and r.delta_mean > 0)
                       for r in df_summary.itertuples())
    any_sig     = any(r.wilcoxon_p < 0.05 for r in df_summary.itertuples())
    avg_sd_ratio = df_summary.sd_ratio_hcga_bc.mean()

    print(f"  Δ moyen (tous scénarios) : {avg_delta:+.3f}%")
    print(f"  Ratio SD moyen HCGA/BC   : {avg_sd_ratio:.2f} "
          f"({'HCGA plus stable' if avg_sd_ratio<0.8 else 'similaire' if avg_sd_ratio<1.2 else 'HCGA moins stable'})")

    if all_sig_pos:
        verdict = 'HCGA_VALIDATED'
        msg = ("HCGA améliore significativement le gain sur tous les scénarios "
               "(p<0.05, Δ>0). → HCGA devient une contribution démontrée, "
               "reconstruire le manuscrit AJSE en conséquence.")
    elif all_sig_neg:
        verdict = 'HCGA_CLOSED'
        msg = ("HCGA dégrade significativement le gain sur tous les scénarios "
               "(p<0.05, Δ<0). → HCGA fermé définitivement pour cette version. "
               "AJSE se recentre sur C3 (décomposition hiérarchique + BC).")
    elif not any_sig:
        if avg_sd_ratio < 0.7:
            verdict = 'HCGA_STABILITY_TRADEOFF'
            msg = ("Aucune différence de gain significative (tous p≥0.05), mais "
                   "HCGA réduit nettement la variance inter-seed (ratio SD "
                   f"{avg_sd_ratio:.2f}). → Analyse robustesse/stabilité "
                   "spécifique nécessaire avant de trancher — PAS une "
                   "conclusion automatique en faveur de HCGA.")
        else:
            verdict = 'HCGA_NOT_JUSTIFIED'
            msg = ("Aucune différence de gain significative ET pas de gain "
                   "de stabilité notable. → HCGA retiré de la contribution "
                   "centrale. AJSE se recentre sur C3 (décomposition "
                   "hiérarchique + BC) uniquement.")
    else:
        verdict = 'MIXED'
        msg = ("Résultats mixtes entre scénarios (significatif dans un sens "
               "sur certains, pas sur d'autres). → Examiner scénario par "
               "scénario avant de conclure globalement.")

    print(f"\n  VERDICT : {verdict}")
    print(f"  {msg}")

    print(f"\n✓ Sauvegardé : results_hcga_final/GABPO_HCGA_final_results.csv")
    print(f"✓ Sauvegardé : results_hcga_final/GABPO_HCGA_final_summary.csv")
    print(f"\n→ Télécharger maintenant :")
    print(f"   from google.colab import files")
    print(f"   files.download('results_hcga_final/GABPO_HCGA_final_summary.csv')")

    return df, df_summary, verdict


if __name__ == "__main__":
    df, df_summary, verdict = run_hcga_final(n_seeds=10)
