# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Audit niveau-seed v2 : correction pour données groupées      ║
# ║                                                                        ║
# ║  Corrige le script précédent : les 50 observations (10 seeds × 5      ║
# ║  scénarios) NE SONT PAS indépendantes (clustering par seed ET par      ║
# ║  scénario). Une corrélation Spearman brute sur ces 50 lignes           ║
# ║  surestimerait la significativité si on la traitait comme preuve.     ║
# ║                                                                        ║
# ║  Ce script fournit, dans l'ordre de rigueur croissante :               ║
# ║    0. Diagnostics explicites (doublons seed/scénario, cohérence des    ║
# ║       seeds entre TE1/BC/NOOP)                                        ║
# ║    1. Spearman brut n=50 — PUREMENT DESCRIPTIF, jamais cité comme      ║
# ║       preuve statistique dans le manuscrit                            ║
# ║    2. Corrélation intra-scénario (10 obs. chacune) — le signal le      ║
# ║       plus interprétable, déjà présent dans la v1                     ║
# ║    3. Régression à effets fixes scénario (dummy variables) — sépare    ║
# ║       l'effet du NO-OP rate de l'effet du scénario lui-même           ║
# ║    4. Test de permutation (10,000 permutations, respectant la          ║
# ║       structure de clusters par scénario) — p-value non-paramétrique  ║
# ║       robuste au clustering                                           ║
# ║    5. IC bootstrap (resampling par CLUSTER scénario, pas par ligne)    ║
# ║                                                                        ║
# ║  Réutilise les résultats déjà calculés — aucun recalcul de TE1/BC.     ║
# ║                                                                        ║
# ║  Usage :                                                               ║
# ║    exec(open('GABPO_SeedLevel_Audit_v2.py').read())                    ║
# ║    df_joined, report = run_seedlevel_audit_v2()                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import pandas as pd
import numpy as np
from scipy import stats

os.makedirs('results_seedlevel_audit', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Audit niveau-seed v2 (corrigé clustering)  ║")
print("╚══════════════════════════════════════════════════════╝")

NOOP_PATH = 'results_noop_audit/bc_noop_stats.csv'
TE1_PATH  = 'results_te/TE1_OSPF_results.csv'
BC_MATCHED_PATH = 'results_te_vs_bc/BC_matched_results.csv'

for path in [NOOP_PATH, TE1_PATH, BC_MATCHED_PATH]:
    if not os.path.exists(path):
        raise RuntimeError(f"Fichier manquant : {path}")

ALL_SCENARIOS = ['S1_Nominal', 'S2_FlashCrowd', 'S3_PoP_Failure',
                  'S4_BGP_Anomaly', 'S5_WACREN_OOD']


def run_seedlevel_audit_v2(n_permutations:int=10000, n_bootstrap:int=5000,
                           random_state:int=42):
    df_noop = pd.read_csv(NOOP_PATH)
    df_te1  = pd.read_csv(TE1_PATH)
    df_bc   = pd.read_csv(BC_MATCHED_PATH)

    rng = np.random.RandomState(random_state)

    # ══════════════════════════════════════════════════════════════════
    # 0. DIAGNOSTICS — avant toute analyse
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n0. DIAGNOSTICS\n{'='*70}")

    for name, df in [('NOOP', df_noop), ('TE1', df_te1), ('BC_matched', df_bc)]:
        dups = df.duplicated(subset=['seed', 'scenario']).sum()
        seeds = sorted(df.seed.unique())
        scens = sorted(df.scenario.unique())
        print(f"  {name:<12} : {len(df)} lignes, {dups} doublons (seed,scenario), "
              f"seeds={seeds}, scénarios={len(scens)}")
        if dups > 0:
            print(f"    ⚠ ATTENTION : {dups} doublons détectés dans {name} — "
                  f"vérifier la source avant de continuer")

    seeds_noop = set(df_noop.seed.unique())
    seeds_te1  = set(df_te1.seed.unique())
    seeds_bc   = set(df_bc.seed.unique())
    common_seeds = seeds_noop & seeds_te1 & seeds_bc
    print(f"\n  Seeds communs aux 3 fichiers : {sorted(common_seeds)} "
          f"({len(common_seeds)} seeds)")
    if len(common_seeds) < 10:
        print(f"  ⚠ Moins de 10 seeds communs — l'analyse portera sur "
              f"{len(common_seeds)} seeds seulement, pas 10")

    # ── Jointure ─────────────────────────────────────────────────────────
    merged = df_noop.merge(df_te1[['seed', 'scenario', 'TE_L_med']],
                            on=['seed', 'scenario'], how='inner')
    merged = merged.merge(df_bc[['seed', 'scenario', 'BC_L_med']],
                           on=['seed', 'scenario'], how='inner')
    merged = merged[merged.seed.isin(common_seeds)].reset_index(drop=True)

    dup_after_merge = merged.duplicated(subset=['seed', 'scenario']).sum()
    print(f"\n  Après jointure : {len(merged)} observations, "
          f"{dup_after_merge} doublons résiduels")
    if dup_after_merge > 0:
        raise RuntimeError("Doublons détectés après jointure — corriger "
                            "la source avant de continuer l'analyse.")

    merged['delta_te1_vs_bc'] = (
        (merged['BC_L_med'] - merged['TE_L_med']) / merged['BC_L_med'] * 100
    )
    merged.to_csv('results_seedlevel_audit/seedlevel_joined_v2.csv', index=False)

    n = len(merged)
    n_scenarios = merged.scenario.nunique()
    n_seeds = merged.seed.nunique()
    print(f"\n  Structure des données : {n} observations = "
          f"{n_seeds} seeds × {n_scenarios} scénarios — "
          f"NON INDÉPENDANTES (clustering par seed et par scénario)")

    # ══════════════════════════════════════════════════════════════════
    # 1. Spearman BRUT — PUREMENT DESCRIPTIF
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n1. Spearman brut (n={n}) — DESCRIPTIF UNIQUEMENT, PAS UNE PREUVE\n{'='*70}")
    rho_raw, p_raw_naive = stats.spearmanr(merged.noop_rate, merged.delta_te1_vs_bc)
    print(f"  ρ={rho_raw:+.3f}, p_naïf={p_raw_naive:.4f} (n={n})")
    print(f"  ⚠ Ce p-value suppose l'indépendance des {n} observations, "
          f"ce qui est FAUX ici (clustering par scénario). Ne pas citer "
          f"ce p-value dans le manuscrit sans la correction ci-dessous.")

    # ══════════════════════════════════════════════════════════════════
    # 2. Corrélation INTRA-scénario
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n2. Corrélation intra-scénario (seule véritablement indépendante "
          f"au sein d'un cluster)\n{'='*70}")
    intra_results = []
    for scen in ALL_SCENARIOS:
        sub = merged[merged.scenario == scen]
        if len(sub) < 3:
            continue
        rho_s, p_s = stats.spearmanr(sub.noop_rate, sub.delta_te1_vs_bc)
        intra_results.append({'scenario': scen, 'rho': rho_s, 'p': p_s, 'n': len(sub)})
        print(f"  {scen:<18} ρ={rho_s:+.3f}  p={p_s:.3f}  n={len(sub)}")

    # ══════════════════════════════════════════════════════════════════
    # 3. Régression à EFFETS FIXES scénario — sépare NO-OP de l'effet scénario
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n3. Régression à effets fixes scénario\n{'='*70}")
    print(f"  Modèle : delta_te1_vs_bc = β·noop_rate + Σγ_s·1(scenario=s) + ε")

    dummies = pd.get_dummies(merged.scenario, prefix='scen', drop_first=True)
    X = pd.concat([merged[['noop_rate']].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    X.insert(0, 'intercept', 1.0)
    y = merged.delta_te1_vs_bc.values.astype(float)
    X_mat = X.values.astype(float)

    # OLS via moindres carrés (pas de dépendance statsmodels nécessaire)
    beta, residuals, rank, sv = np.linalg.lstsq(X_mat, y, rcond=None)
    y_pred = X_mat @ beta
    resid = y - y_pred
    dof = len(y) - X_mat.shape[1]
    mse = (resid ** 2).sum() / dof if dof > 0 else np.nan
    XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
    se_beta = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = beta / se_beta
    p_values_ols = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

    noop_idx = list(X.columns).index('noop_rate')
    beta_noop, se_noop, p_noop = beta[noop_idx], se_beta[noop_idx], p_values_ols[noop_idx]
    print(f"  Coefficient NO-OP rate (contrôlé par scénario) : "
          f"β={beta_noop:.3f}, SE={se_noop:.3f}, p={p_noop:.4f}")
    print(f"  Interprétation : effet du NO-OP rate sur Δ(TE1-BC) APRÈS avoir "
          f"retiré l'effet moyen de chaque scénario (dummies) — si p reste "
          f"petit ici, l'effet NO-OP n'est pas juste un artefact de scénario.")

    # ══════════════════════════════════════════════════════════════════
    # 4. Test de PERMUTATION (respecte la structure par cluster scénario)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n4. Test de permutation ({n_permutations} permutations, "
          f"par cluster scénario)\n{'='*70}")

    def compute_stat(df_):
        return stats.spearmanr(df_.noop_rate, df_.delta_te1_vs_bc)[0]

    observed_rho = compute_stat(merged)
    perm_rhos = np.zeros(n_permutations)
    scenario_groups = merged.groupby('scenario')

    for i in range(n_permutations):
        # Permute delta_te1_vs_bc INDÉPENDAMMENT DANS CHAQUE CLUSTER scénario
        # (préserve la structure de cluster, teste seulement l'association
        # NO-OP↔delta AU SEIN de chaque scénario, agrégée ensuite)
        permuted = merged.copy()
        for scen, idx in scenario_groups.groups.items():
            permuted.loc[idx, 'delta_te1_vs_bc'] = rng.permutation(
                merged.loc[idx, 'delta_te1_vs_bc'].values)
        perm_rhos[i] = compute_stat(permuted)

    p_perm = (np.sum(np.abs(perm_rhos) >= np.abs(observed_rho)) + 1) / (n_permutations + 1)
    print(f"  ρ observé = {observed_rho:+.3f}")
    print(f"  p (permutation, {n_permutations} tirages, structure cluster préservée) "
          f"= {p_perm:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # 5. IC bootstrap — resampling PAR CLUSTER (scénario), pas par ligne
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n5. IC bootstrap ({n_bootstrap} tirages, resampling par cluster "
          f"scénario)\n{'='*70}")

    scenarios_list = merged.scenario.unique()
    boot_rhos = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_scenarios = rng.choice(scenarios_list, size=len(scenarios_list),
                                       replace=True)
        boot_df = pd.concat([merged[merged.scenario == s] for s in sampled_scenarios],
                            ignore_index=True)
        if boot_df.noop_rate.nunique() > 1:
            boot_rhos[i] = stats.spearmanr(boot_df.noop_rate,
                                           boot_df.delta_te1_vs_bc)[0]
        else:
            boot_rhos[i] = np.nan

    boot_rhos = boot_rhos[~np.isnan(boot_rhos)]
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    print(f"  ρ observé = {observed_rho:+.3f}")
    print(f"  IC95% bootstrap (cluster-level resampling) = [{ci_lo:+.3f}, {ci_hi:+.3f}]")

    # ══════════════════════════════════════════════════════════════════
    # RAPPORT FINAL — tableau exploitable pour le manuscrit
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\nRAPPORT FINAL — pour intégration manuscrit\n{'='*70}")
    report = {
        'n_observations': n, 'n_seeds': n_seeds, 'n_scenarios': n_scenarios,
        'spearman_rho_raw': observed_rho,
        'spearman_p_naive_DO_NOT_CITE': p_raw_naive,
        'ols_beta_noop_controlled_for_scenario': beta_noop,
        'ols_p_noop_controlled_for_scenario': p_noop,
        'permutation_p_cluster_aware': p_perm,
        'bootstrap_ci95_lo': ci_lo, 'bootstrap_ci95_hi': ci_hi,
    }
    for k, v in report.items():
        print(f"  {k:<45} {v}")

    pd.DataFrame([report]).to_csv('results_seedlevel_audit/report_v2.csv', index=False)
    pd.DataFrame(intra_results).to_csv(
        'results_seedlevel_audit/intra_scenario_v2.csv', index=False)

    print(f"\n✓ Sauvegardé : results_seedlevel_audit/report_v2.csv")
    print(f"✓ Sauvegardé : results_seedlevel_audit/intra_scenario_v2.csv")
    print(f"\n→ Télécharger : from google.colab import files; "
          f"files.download('results_seedlevel_audit/report_v2.csv')")

    return merged, report


if __name__ == "__main__":
    df_joined, report = run_seedlevel_audit_v2()
