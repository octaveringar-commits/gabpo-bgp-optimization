# ╔══════════════════════════════════════════════════════════════════════╗
# ║ GABPO — Audit niveau-seed v3                                            ║
# ║ NO-OP rate ↔ avantage TE1−BC                                            ║
# ║                                                                          ║
# ║ OBJECTIF                                                                 ║
# ║ --------                                                                 ║
# ║ Tester si le NO-OP rate individuel de Behavioral Cloning (BC) est        ║
# ║ associé à l'avantage individuel de TE1 par rapport à BC, tout en        ║
# ║ contrôlant explicitement les deux dimensions répétées :                 ║
# ║                                                                          ║
# ║   - seed                                                                 ║
# ║   - scénario                                                             ║
# ║                                                                          ║
# ║ Unité d'observation : seed × scénario                                    ║
# ║                                                                          ║
# ║ 10 seeds × 5 scénarios = 50 observations attendues                       ║
# ║                                                                          ║
# ║ IMPORTANT                                                                ║
# ║ ---------                                                                ║
# ║ Les 50 observations ne sont PAS traitées comme 50 réplications          ║
# ║ indépendantes.                                                           ║
# ║                                                                          ║
# ║ ANALYSES                                                                 ║
# ║ --------                                                                 ║
# ║ 0. Diagnostics complets                                                  ║
# ║ 1. Spearman global n=50 : descriptif uniquement                          ║
# ║ 2. Spearman intra-scénario : association entre seeds dans chaque         ║
# ║    scénario                                                              ║
# ║ 3. OLS avec effets fixes SEED + SCÉNARIO                                 ║
# ║ 4. Test de permutation par PROFIL DE SEED                                ║
# ║ 5. Bootstrap par SEED ENTIER                                             ║
# ║ 6. Analyse de sensibilité : Pearson / Spearman sur résidus               ║
# ║ 7. Rapport automatique exploitable pour le manuscrit                     ║
# ║                                                                          ║
# ║ AUCUN recalcul de TE1 ou BC                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


import os
import warnings
import numpy as np
import pandas as pd

from scipy import stats


# ============================================================================
# CONFIGURATION
# ============================================================================

os.makedirs("results_seedlevel_audit", exist_ok=True)

NOOP_PATH = "results_noop_audit/bc_noop_stats.csv"
TE1_PATH = "results_te/TE1_OSPF_results.csv"
BC_MATCHED_PATH = "results_te_vs_bc/BC_matched_results.csv"

ALL_SCENARIOS = [
    "S1_Nominal",
    "S2_FlashCrowd",
    "S3_PoP_Failure",
    "S4_BGP_Anomaly",
    "S5_WACREN_OOD"
]

EXPECTED_N_SEEDS = 10
EXPECTED_N_SCENARIOS = 5

DEFAULT_N_PERMUTATIONS = 10000
DEFAULT_N_BOOTSTRAP = 5000
DEFAULT_RANDOM_STATE = 42


# ============================================================================
# UTILITAIRES
# ============================================================================

def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check_required_files():
    for path in [NOOP_PATH, TE1_PATH, BC_MATCHED_PATH]:
        if not os.path.exists(path):
            raise RuntimeError(
                f"\nFichier manquant : {path}\n"
                f"Ce script ne recalcule ni TE1 ni BC.\n"
                f"Vérifier les résultats précédents avant de continuer."
            )


def validate_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{name}: colonnes manquantes : {missing}\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )


def clean_key_columns(df, name):
    df = df.copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    if df["seed"].isna().any():
        raise RuntimeError(f"{name}: certaines valeurs de seed sont invalides.")
    df["seed"] = df["seed"].astype(int)
    df["scenario"] = df["scenario"].astype(str).str.strip()
    return df


def assert_unique_seed_scenario(df, name):
    duplicates = df[df.duplicated(subset=["seed", "scenario"], keep=False)]
    duplicates = duplicates.sort_values(["seed", "scenario"])
    if len(duplicates) > 0:
        print(f"\n⚠ {name}: doublons détectés sur (seed, scenario)")
        print(duplicates[["seed", "scenario"]].to_string(index=False))
        raise RuntimeError(
            f"{name}: doublons (seed, scenario). "
            f"Impossible de garantir une observation individuelle unique."
        )


def zscore_safe(x):
    x = np.asarray(x, dtype=float)
    sd = np.std(x, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - np.mean(x)) / sd


# ============================================================================
# OLS
# ============================================================================

def build_fixed_effect_design(df):
    """
    Construit : intercept, NOOP, effets fixes seed, effets fixes scenario.
    Une catégorie de référence est supprimée pour éviter la colinéarité.
    """
    seed_dummies = pd.get_dummies(df["seed"], prefix="seed",
                                   drop_first=True, dtype=float)
    scenario_dummies = pd.get_dummies(df["scenario"], prefix="scenario",
                                       drop_first=True, dtype=float)
    X = pd.concat([
        df[["noop_rate"]].reset_index(drop=True),
        seed_dummies.reset_index(drop=True),
        scenario_dummies.reset_index(drop=True)
    ], axis=1)
    X.insert(0, "intercept", 1.0)
    return X


def ols_fit(X, y):
    """OLS par moindres carrés. Retourne beta, résidus, SE, t, p, R²."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    if n <= k:
        raise RuntimeError(f"Pas assez d'observations : n={n}, paramètres={k}")

    beta, residuals_lstsq, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    dof = n - rank
    sse = np.sum(resid ** 2)
    mse = sse / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    covariance = XtX_inv * mse
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=dof)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - sse / sst if sst > 0 else np.nan

    return {"beta": beta, "fitted": fitted, "resid": resid, "dof": dof,
            "rank": rank, "se": se, "t": t_stats, "p": p_values,
            "r2": r2, "sse": sse}


def fit_two_way_fixed_effects(df):
    """
    Modèle : Δ = β·NOOP + α_seed + γ_scenario + ε
    β est estimé après contrôle des deux dimensions simultanément.
    """
    X = build_fixed_effect_design(df)
    y = df["delta_te1_vs_bc"].values.astype(float)
    result = ols_fit(X, y)

    noop_idx = list(X.columns).index("noop_rate")
    result["beta_noop"] = result["beta"][noop_idx]
    result["se_noop"] = result["se"][noop_idx]
    result["t_noop"] = result["t"][noop_idx]
    result["p_noop"] = result["p"][noop_idx]
    result["columns"] = list(X.columns)

    critical = stats.t.ppf(0.975, result["dof"])
    result["ci95_lo"] = result["beta_noop"] - critical * result["se_noop"]
    result["ci95_hi"] = result["beta_noop"] + critical * result["se_noop"]

    return result


def residualize_by_fixed_effects(df, variable):
    tmp = df.copy()
    X = build_fixed_effect_design(tmp)
    y = tmp[variable].values.astype(float)
    fit = ols_fit(X, y)
    return fit["resid"]


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def run_diagnostics(df_noop, df_te1, df_bc):
    print_section("0. DIAGNOSTICS")
    datasets = [("NOOP", df_noop), ("TE1", df_te1), ("BC_matched", df_bc)]

    for name, df in datasets:
        duplicates = df.duplicated(subset=["seed", "scenario"]).sum()
        seeds = sorted(df["seed"].unique())
        scenarios = sorted(df["scenario"].unique())
        print(f"\n{name:<12}: {len(df)} lignes | {len(seeds)} seeds | "
              f"{len(scenarios)} scénarios | {duplicates} doublon(s)")
        print(f"  Seeds      : {seeds}")
        print(f"  Scénarios  : {scenarios}")
        if duplicates > 0:
            raise RuntimeError(f"{name}: doublons détectés.")

    for name, df in datasets:
        missing_scenarios = set(ALL_SCENARIOS) - set(df["scenario"].unique())
        if missing_scenarios:
            print(f"⚠ {name}: scénarios manquants : {sorted(missing_scenarios)}")

    seeds_noop = set(df_noop["seed"].unique())
    seeds_te1 = set(df_te1["seed"].unique())
    seeds_bc = set(df_bc["seed"].unique())
    common_seeds = seeds_noop & seeds_te1 & seeds_bc

    print(f"\nSeeds communes aux trois sources : {sorted(common_seeds)}")
    if len(common_seeds) != EXPECTED_N_SEEDS:
        print(f"⚠ Nombre attendu : {EXPECTED_N_SEEDS}; obtenu : {len(common_seeds)}")

    return common_seeds


# ============================================================================
# JOINTURE
# ============================================================================

def create_merged_dataset(df_noop, df_te1, df_bc, common_seeds):
    print_section("CONSTRUCTION DU DATASET APPARIÉ")

    merged = df_noop.merge(
        df_te1[["seed", "scenario", "TE_L_med"]],
        on=["seed", "scenario"], how="inner", validate="one_to_one")
    merged = merged.merge(
        df_bc[["seed", "scenario", "BC_L_med"]],
        on=["seed", "scenario"], how="inner", validate="one_to_one")
    merged = merged[merged["seed"].isin(common_seeds)].copy()
    merged.reset_index(drop=True, inplace=True)

    if len(merged) == 0:
        raise RuntimeError("La jointure est vide.")

    required_numeric = ["noop_rate", "TE_L_med", "BC_L_med"]
    for col in required_numeric:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        if merged[col].isna().any():
            bad = merged[merged[col].isna()][["seed", "scenario", col]]
            print(bad)
            raise RuntimeError(f"Valeurs invalides dans {col}.")

    if (merged["BC_L_med"] <= 0).any():
        raise RuntimeError("BC_L_med contient une valeur <= 0.")

    merged["delta_te1_vs_bc"] = (
        (merged["BC_L_med"] - merged["TE_L_med"]) / merged["BC_L_med"] * 100
    )

    n = len(merged)
    n_seeds = merged["seed"].nunique()
    n_scenarios = merged["scenario"].nunique()
    expected = n_seeds * n_scenarios

    print(f"\nObservations appariées : {n}")
    print(f"Structure : {n_seeds} seeds × {n_scenarios} scénarios = {expected}")
    if n != expected:
        print("⚠ Dataset non équilibré.")

    assert_unique_seed_scenario(merged, "Dataset merged")

    output = "results_seedlevel_audit/seedlevel_joined_v3.csv"
    merged.to_csv(output, index=False)
    print(f"\n✓ Dataset sauvegardé : {output}")

    return merged


# ============================================================================
# 1. SPEARMAN GLOBAL — DESCRIPTIF
# ============================================================================

def run_global_spearman(df):
    print_section(f"1. SPEARMAN GLOBAL — DESCRIPTIF UNIQUEMENT (n={len(df)})")
    rho, p = stats.spearmanr(df["noop_rate"], df["delta_te1_vs_bc"])
    print(f"ρ = {rho:+.4f}")
    print(f"p naïf = {p:.6f}")
    print("\n⚠ Cette p-value suppose l'indépendance des observations "
          "et n'est PAS utilisée comme preuve confirmatoire.")
    return {"rho": rho, "p_naive": p}


# ============================================================================
# 2. SPEARMAN INTRA-SCÉNARIO
# ============================================================================

def run_intra_scenario(df):
    print_section("2. ASSOCIATIONS INTRA-SCÉNARIO")
    results = []
    for scenario in ALL_SCENARIOS:
        sub = df[df["scenario"] == scenario].copy()
        if len(sub) < 3:
            print(f"{scenario:<20} : insuffisant")
            continue
        if sub["noop_rate"].nunique() < 2 or sub["delta_te1_vs_bc"].nunique() < 2:
            print(f"{scenario:<20} : variance insuffisante")
            continue
        rho, p = stats.spearmanr(sub["noop_rate"], sub["delta_te1_vs_bc"])
        results.append({"scenario": scenario, "rho_spearman": rho,
                        "p_naive_within_scenario": p, "n": len(sub),
                        "noop_min": sub["noop_rate"].min(),
                        "noop_max": sub["noop_rate"].max(),
                        "delta_min": sub["delta_te1_vs_bc"].min(),
                        "delta_max": sub["delta_te1_vs_bc"].max()})
        print(f"{scenario:<20} ρ={rho:+.4f} p={p:.4f} n={len(sub)} "
              f"NOOP={sub['noop_rate'].min():.3f}–{sub['noop_rate'].max():.3f}")

    result_df = pd.DataFrame(results)
    output = "results_seedlevel_audit/intra_scenario_v3.csv"
    result_df.to_csv(output, index=False)
    print(f"\n✓ Sauvegardé : {output}")
    return result_df


# ============================================================================
# 3. TWO-WAY FIXED EFFECTS
# ============================================================================

def run_two_way_fixed_effects(df):
    print_section("3. MODÈLE À EFFETS FIXES SEED + SCÉNARIO")
    print("Modèle :")
    print("Δ(TE1−BC) = β·NOOP + α_seed + γ_scenario + ε")

    result = fit_two_way_fixed_effects(df)

    print(f"\nβ_NOOP = {result['beta_noop']:+.6f}")
    print(f"SE classique = {result['se_noop']:.6f}")
    print(f"t = {result['t_noop']:+.4f}")
    print(f"p = {result['p_noop']:.6f}")
    print(f"IC95% classique = [{result['ci95_lo']:+.6f}, {result['ci95_hi']:+.6f}]")
    print(f"R² = {result['r2']:.4f}")
    print("\nInterprétation : β mesure l'association entre NO-OP rate et Δ "
          "après contrôle simultané des effets fixes seed ET scénario.")
    print("\n⚠ Le p classique est fourni comme référence de modèle OLS. "
          "La décision principale doit s'appuyer sur le test de "
          "permutation ci-dessous.")

    return result


# ============================================================================
# 4. PERMUTATION PAR PROFIL DE SEED
# ============================================================================

def run_seed_profile_permutation(df, n_permutations=DEFAULT_N_PERMUTATIONS,
                                 random_state=DEFAULT_RANDOM_STATE):
    print_section(f"4. TEST DE PERMUTATION PAR PROFIL DE SEED "
                  f"({n_permutations} permutations)")

    observed_fit = fit_two_way_fixed_effects(df)
    observed_beta = observed_fit["beta_noop"]
    print(f"β observé = {observed_beta:+.6f}")

    seeds = sorted(df["seed"].unique())
    scenarios = sorted(df["scenario"].unique())
    n_seeds = len(seeds)

    noop_matrix = df.pivot(index="seed", columns="scenario",
                            values="noop_rate").reindex(index=seeds, columns=scenarios)
    delta_matrix = df.pivot(index="seed", columns="scenario",
                             values="delta_te1_vs_bc").reindex(index=seeds, columns=scenarios)

    if noop_matrix.isna().any().any():
        raise RuntimeError("NOOP matrix non complète.")
    if delta_matrix.isna().any().any():
        raise RuntimeError("Delta matrix non complète.")

    rng = np.random.RandomState(random_state)
    permuted_betas = np.zeros(n_permutations)

    # Permute le PROFIL COMPLET d'un seed (les 5 valeurs NOOP à travers
    # les scénarios sont déplacées ensemble) — préserve la structure
    # intra-seed, teste si l'appariement seed-par-seed avec delta compte.
    for i in range(n_permutations):
        permutation = rng.permutation(n_seeds)
        perm_noop = noop_matrix.iloc[permutation].copy()
        perm_noop.index = seeds

        permuted_df = df.copy()
        values = []
        for seed in seeds:
            for scenario in scenarios:
                values.append(perm_noop.loc[seed, scenario])
        permuted_df["noop_rate"] = values

        try:
            fit = fit_two_way_fixed_effects(permuted_df)
            permuted_betas[i] = fit["beta_noop"]
        except Exception:
            permuted_betas[i] = np.nan

    permuted_betas = permuted_betas[np.isfinite(permuted_betas)]
    if len(permuted_betas) < 0.95 * n_permutations:
        warnings.warn("Plus de 5% des permutations sont invalides.")

    p_perm = (np.sum(np.abs(permuted_betas) >= np.abs(observed_beta)) + 1) / \
             (len(permuted_betas) + 1)
    null_lo, null_hi = np.percentile(permuted_betas, [2.5, 97.5])

    print(f"\nNombre de permutations valides : {len(permuted_betas)}")
    print(f"β observé = {observed_beta:+.6f}")
    print(f"p permutation = {p_perm:.6f}")
    print(f"Distribution nulle 95% = [{null_lo:+.6f}, {null_hi:+.6f}]")
    print("\nInterprétation : le test détruit l'appariement entre les "
          "profils NOOP et Δ tout en conservant la structure complète "
          "des profils par seed.")

    pd.DataFrame({"permuted_beta": permuted_betas}).to_csv(
        "results_seedlevel_audit/permutation_betas_v3.csv", index=False)

    return {"observed_beta": observed_beta, "p_permutation": p_perm,
            "null_ci_lo": null_lo, "null_ci_hi": null_hi,
            "n_valid_permutations": len(permuted_betas)}


# ============================================================================
# 5. BOOTSTRAP PAR SEED
# ============================================================================

def run_seed_bootstrap(df, n_bootstrap=DEFAULT_N_BOOTSTRAP,
                       random_state=DEFAULT_RANDOM_STATE):
    print_section(f"5. BOOTSTRAP PAR SEED ({n_bootstrap} tirages)")

    seeds = np.array(sorted(df["seed"].unique()))
    rng = np.random.RandomState(random_state + 1)

    observed_fit = fit_two_way_fixed_effects(df)
    observed_beta = observed_fit["beta_noop"]

    boot_betas = []

    for i in range(n_bootstrap):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        pieces = [df[df["seed"] == s].copy() for s in sampled_seeds]
        boot_df = pd.concat(pieces, ignore_index=True)

        boot_df["bootstrap_seed"] = np.repeat(
            np.arange(len(pieces)), [len(x) for x in pieces])

        seed_dummies = pd.get_dummies(boot_df["bootstrap_seed"], prefix="seed",
                                       drop_first=True, dtype=float)
        scenario_dummies = pd.get_dummies(boot_df["scenario"], prefix="scenario",
                                          drop_first=True, dtype=float)
        X = pd.concat([boot_df[["noop_rate"]].reset_index(drop=True),
                       seed_dummies.reset_index(drop=True),
                       scenario_dummies.reset_index(drop=True)], axis=1)
        X.insert(0, "intercept", 1.0)
        y = boot_df["delta_te1_vs_bc"].values.astype(float)

        try:
            fit = ols_fit(X.values, y)
            idx = list(X.columns).index("noop_rate")
            beta = fit["beta"][idx]
            if np.isfinite(beta):
                boot_betas.append(beta)
        except Exception:
            continue

    boot_betas = np.asarray(boot_betas, dtype=float)
    if len(boot_betas) < 100:
        raise RuntimeError("Trop peu de réplications bootstrap valides.")

    ci_lo, ci_hi = np.percentile(boot_betas, [2.5, 97.5])

    print(f"β observé = {observed_beta:+.6f}")
    print(f"IC95% bootstrap par seed = [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"Bootstrap valides : {len(boot_betas)} / {n_bootstrap}")
    print("\nLe bootstrap ré-échantillonne les seeds entières et "
          "conserve leurs cinq scénarios.")

    pd.DataFrame({"bootstrap_beta": boot_betas}).to_csv(
        "results_seedlevel_audit/bootstrap_betas_v3.csv", index=False)

    return {"observed_beta": observed_beta, "ci95_lo": ci_lo,
            "ci95_hi": ci_hi, "n_valid_bootstrap": len(boot_betas)}


# ============================================================================
# 6. ANALYSE DE SENSIBILITÉ PAR RÉSIDUALISATION
# ============================================================================

def run_residual_sensitivity(df):
    print_section("6. ANALYSE DE SENSIBILITÉ — RÉSIDUS APRÈS "
                  "EFFETS FIXES SEED + SCÉNARIO")

    noop_resid = residualize_by_fixed_effects(df, "noop_rate")
    delta_resid = residualize_by_fixed_effects(df, "delta_te1_vs_bc")

    pearson_r, pearson_p = stats.pearsonr(noop_resid, delta_resid)
    spearman_rho, spearman_p = stats.spearmanr(noop_resid, delta_resid)

    print(f"Pearson résiduel  : r={pearson_r:+.4f}, p={pearson_p:.6f}")
    print(f"Spearman résiduel : ρ={spearman_rho:+.4f}, p={spearman_p:.6f}")
    print("\n⚠ Ces p-values sont des statistiques de sensibilité et ne "
          "doivent pas être interprétées comme une preuve indépendante "
          "du test de permutation.")

    return {"pearson_residual_r": pearson_r, "pearson_residual_p": pearson_p,
            "spearman_residual_rho": spearman_rho,
            "spearman_residual_p": spearman_p}


# ============================================================================
# 7. STATISTIQUES DESCRIPTIVES
# ============================================================================

def run_descriptives(df):
    print_section("7. STATISTIQUES DESCRIPTIVES")

    print(f"NOOP rate : mean={df['noop_rate'].mean():.4f}, "
          f"median={df['noop_rate'].median():.4f}, "
          f"min={df['noop_rate'].min():.4f}, max={df['noop_rate'].max():.4f}")
    print(f"Δ TE1−BC : mean={df['delta_te1_vs_bc'].mean():+.4f}%, "
          f"median={df['delta_te1_vs_bc'].median():+.4f}%, "
          f"min={df['delta_te1_vs_bc'].min():+.4f}%, "
          f"max={df['delta_te1_vs_bc'].max():+.4f}%")

    scenario_summary = df.groupby("scenario").agg(
        n=("delta_te1_vs_bc", "size"), noop_mean=("noop_rate", "mean"),
        noop_sd=("noop_rate", "std"), delta_mean=("delta_te1_vs_bc", "mean"),
        delta_sd=("delta_te1_vs_bc", "std")).reset_index()

    print("\nRésumé par scénario :")
    print(scenario_summary.to_string(index=False))

    scenario_summary.to_csv(
        "results_seedlevel_audit/scenario_summary_v3.csv", index=False)

    return scenario_summary


# ============================================================================
# 8. RAPPORT FINAL
# ============================================================================

def build_final_report(df, global_result, intra_result, two_way_result,
                        permutation_result, bootstrap_result, residual_result):
    beta = two_way_result["beta_noop"]

    if permutation_result["p_permutation"] < 0.05 and bootstrap_result["ci95_lo"] > 0:
        conclusion = ("Association positive soutenue par le test de "
                      "permutation et par l'IC bootstrap.")
    elif permutation_result["p_permutation"] < 0.05 and bootstrap_result["ci95_hi"] < 0:
        conclusion = ("Association négative soutenue par le test de "
                      "permutation et par l'IC bootstrap.")
    elif permutation_result["p_permutation"] < 0.05:
        conclusion = ("Association détectée par permutation, mais IC "
                      "bootstrap non entièrement du même côté de zéro : "
                      "interprétation prudente.")
    else:
        conclusion = ("Pas d'évidence statistique robuste d'une "
                      "association après contrôle seed + scénario.")

    report = {
        "n_observations": len(df), "n_seeds": df["seed"].nunique(),
        "n_scenarios": df["scenario"].nunique(),
        "spearman_global_rho": global_result["rho"],
        "spearman_global_p_naive_DO_NOT_CITE": global_result["p_naive"],
        "two_way_FE_beta_NOOP": beta,
        "two_way_FE_SE_classical": two_way_result["se_noop"],
        "two_way_FE_t": two_way_result["t_noop"],
        "two_way_FE_p_classical": two_way_result["p_noop"],
        "two_way_FE_CI95_lo": two_way_result["ci95_lo"],
        "two_way_FE_CI95_hi": two_way_result["ci95_hi"],
        "two_way_FE_R2": two_way_result["r2"],
        "permutation_beta_observed": permutation_result["observed_beta"],
        "permutation_p": permutation_result["p_permutation"],
        "permutation_null95_lo": permutation_result["null_ci_lo"],
        "permutation_null95_hi": permutation_result["null_ci_hi"],
        "permutation_n_valid": permutation_result["n_valid_permutations"],
        "bootstrap_beta_observed": bootstrap_result["observed_beta"],
        "bootstrap_CI95_lo": bootstrap_result["ci95_lo"],
        "bootstrap_CI95_hi": bootstrap_result["ci95_hi"],
        "bootstrap_n_valid": bootstrap_result["n_valid_bootstrap"],
        "residual_Pearson_r": residual_result["pearson_residual_r"],
        "residual_Pearson_p": residual_result["pearson_residual_p"],
        "residual_Spearman_rho": residual_result["spearman_residual_rho"],
        "residual_Spearman_p": residual_result["spearman_residual_p"],
        "automatic_interpretation": conclusion
    }

    report_df = pd.DataFrame([report])
    output = "results_seedlevel_audit/report_v3.csv"
    report_df.to_csv(output, index=False)

    return report


def print_final_report(report):
    print_section("8. RAPPORT FINAL — AUDIT GABPO")

    print(f"Observations : {report['n_observations']}")
    print(f"Seeds : {report['n_seeds']}")
    print(f"Scénarios : {report['n_scenarios']}")

    print("\n--- DESCRIPTIF ---")
    print(f"Spearman global : ρ={report['spearman_global_rho']:+.4f}")
    print(f"p naïf : {report['spearman_global_p_naive_DO_NOT_CITE']:.6f}")
    print("⚠ p naïf NON UTILISABLE comme preuve confirmatoire.")

    print("\n--- MODÈLE PRINCIPAL : SEED + SCÉNARIO ---")
    print(f"β NOOP = {report['two_way_FE_beta_NOOP']:+.6f}")
    print(f"IC95% classique = [{report['two_way_FE_CI95_lo']:+.6f}, "
          f"{report['two_way_FE_CI95_hi']:+.6f}]")
    print(f"p OLS classique = {report['two_way_FE_p_classical']:.6f}")

    print("\n--- TEST PRINCIPAL : PERMUTATION PAR PROFIL DE SEED ---")
    print(f"β observé = {report['permutation_beta_observed']:+.6f}")
    print(f"p permutation = {report['permutation_p']:.6f}")

    print("\n--- BOOTSTRAP PAR SEED ---")
    print(f"IC95% = [{report['bootstrap_CI95_lo']:+.6f}, "
          f"{report['bootstrap_CI95_hi']:+.6f}]")

    print("\n--- SENSIBILITÉ RÉSIDUELLE ---")
    print(f"Pearson = {report['residual_Pearson_r']:+.4f}, "
          f"p={report['residual_Pearson_p']:.6f}")
    print(f"Spearman = {report['residual_Spearman_rho']:+.4f}, "
          f"p={report['residual_Spearman_p']:.6f}")

    print("\n--- INTERPRÉTATION AUTOMATIQUE ---")
    print(report["automatic_interpretation"])


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def run_seedlevel_audit_v3(n_permutations=DEFAULT_N_PERMUTATIONS,
                           n_bootstrap=DEFAULT_N_BOOTSTRAP,
                           random_state=DEFAULT_RANDOM_STATE):
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║ GABPO — Audit niveau-seed v3                                       ║")
    print("║ NO-OP rate ↔ avantage TE1−BC                                      ║")
    print("║ Two-way fixed effects + seed-profile permutation + bootstrap      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    check_required_files()

    df_noop = pd.read_csv(NOOP_PATH)
    df_te1 = pd.read_csv(TE1_PATH)
    df_bc = pd.read_csv(BC_MATCHED_PATH)

    validate_columns(df_noop, ["seed", "scenario", "noop_rate"], "NOOP")
    validate_columns(df_te1, ["seed", "scenario", "TE_L_med"], "TE1")
    validate_columns(df_bc, ["seed", "scenario", "BC_L_med"], "BC_matched")

    df_noop = clean_key_columns(df_noop, "NOOP")
    df_te1 = clean_key_columns(df_te1, "TE1")
    df_bc = clean_key_columns(df_bc, "BC_matched")

    assert_unique_seed_scenario(df_noop, "NOOP")
    assert_unique_seed_scenario(df_te1, "TE1")
    assert_unique_seed_scenario(df_bc, "BC_matched")

    common_seeds = run_diagnostics(df_noop, df_te1, df_bc)
    merged = create_merged_dataset(df_noop, df_te1, df_bc, common_seeds)

    global_result = run_global_spearman(merged)
    intra_result = run_intra_scenario(merged)
    two_way_result = run_two_way_fixed_effects(merged)
    permutation_result = run_seed_profile_permutation(
        merged, n_permutations=n_permutations, random_state=random_state)
    bootstrap_result = run_seed_bootstrap(
        merged, n_bootstrap=n_bootstrap, random_state=random_state)
    residual_result = run_residual_sensitivity(merged)
    scenario_summary = run_descriptives(merged)

    report = build_final_report(merged, global_result, intra_result,
                                 two_way_result, permutation_result,
                                 bootstrap_result, residual_result)
    print_final_report(report)

    print("\n✓ Fichiers produits dans : results_seedlevel_audit/")
    print("\nFichiers principaux :")
    print("  1. seedlevel_joined_v3.csv")
    print("  2. intra_scenario_v3.csv")
    print("  3. permutation_betas_v3.csv")
    print("  4. bootstrap_betas_v3.csv")
    print("  5. scenario_summary_v3.csv")
    print("  6. report_v3.csv")
    print("\nPour télécharger le rapport dans Colab :")
    print("from google.colab import files")
    print("files.download('results_seedlevel_audit/report_v3.csv')")

    return merged, report


if __name__ == "__main__":
    df_joined, report = run_seedlevel_audit_v3()
