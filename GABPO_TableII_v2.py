# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Table II v2 : campagne indépendante et versionnée              ║
# ║                                                                          ║
# ║  CE SCRIPT N'EST PAS UNE RECONSTRUCTION DE L'ANCIENNE TABLE II.          ║
# ║  Aucune valeur historique (Table II publiée dans ICCT v12, ni            ║
# ║  LOCKED_REFERENCE, ni LOCKED_BC) n'est lue, injectée ou comparée par ce   ║
# ║  script pendant l'exécution. La recherche exhaustive de provenance menée ║
# ║  dans la session de rédaction du manuscrit n'a retrouvé aucun artefact   ║
# ║  (script, CSV, objet Git) reproduisant les valeurs S1/S3/S4 publiées     ║
# ║  dans Table II. Ce script produit donc une NOUVELLE campagne, avec sa    ║
# ║  propre chaîne de provenance complète dès le premier jour.               ║
# ║                                                                          ║
# ║  Toute comparaison avec les anciennes valeurs doit être faite APRÈS      ║
# ║  coup, hors de ce script, sur les 40 observations brutes conservées.     ║
# ║                                                                          ║
# ║  Protocole verrouillé (non modifiable via les arguments de fonction) :   ║
# ║    - checkpoint : checkpoints_bc/bc_phase_D.pt (SHA-256 vérifié,         ║
# ║                    arrêt immédiat si désaccord)                         ║
# ║    - seeds       : range(10), SANS offset (pas de +900)                  ║
# ║    - scénarios   : S1_Nominal, S2_FlashCrowd, S3_PoP_Failure,            ║
# ║                     S4_BGP_Anomaly (S5 hors champ — traité séparément)   ║
# ║    - N_a=50, N_p=12, T_ep=288                                            ║
# ║    - gain        : G = (B1_Lmed - BC_Lmed) / B1_Lmed * 100               ║
# ║    - IC          : Student-t 95%, n=10                                   ║
# ║    - test        : Wilcoxon signé apparié, bilatéral                     ║
# ║    - correction  : Bonferroni, famille S1-S4, alpha' = 0.0125            ║
# ║                                                                          ║
# ║  Prérequis dans la session Colab (comme d'habitude) :                    ║
# ║    exec(open('GABPO_P0b_DynamicEnv.py').read())                          ║
# ║    exec(open('GABPO_P1b_BC.py').read())                                  ║
# ║    (checkpoint BC déjà entraîné : checkpoints_bc/bc_phase_D.pt)          ║
# ║    (extract_features déjà patché dans la session — requis par            ║
# ║     HierarchicalPolicy.select_action(), non fourni par ce script)        ║
# ║                                                                          ║
# ║  Usage :                                                                 ║
# ║    exec(open('GABPO_TableII_v2.py').read())                              ║
# ║    df_raw, df_summary, manifest = run_table2_v2()                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import sys
import json
import hashlib
import platform
from datetime import datetime, timezone

import numpy as np
import torch
import pandas as pd
import scipy
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_table2_v2', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Table II v2 : campagne indépendante        ║")
print("╚══════════════════════════════════════════════════════╝")
print()
print("Ce script ne lit, n'injecte et ne compare aucune valeur historique")
print("de Table II pendant l'exécution. Comparaison post-hoc uniquement.")
print()

# ── Vérification des dépendances ──────────────────────────────────────────
try:
    _ = BGPDynamicEnv, run_episode, B1_Static
    print("✓ P0b chargé (BGPDynamicEnv, run_episode, B1_Static)")
except NameError as e:
    raise RuntimeError(f"Exécuter GABPO_P0b_DynamicEnv.py d'abord : {e}")

try:
    _ = HierarchicalPolicy, extract_features
    print("✓ BC chargé (HierarchicalPolicy, extract_features)")
except NameError as e:
    raise RuntimeError(
        f"Exécuter GABPO_P1b_BC.py et patcher extract_features d'abord : {e}")

BC_CHECKPOINT = 'checkpoints_bc/bc_phase_D.pt'
if not os.path.exists(BC_CHECKPOINT):
    raise RuntimeError(f"Checkpoint BC introuvable : {BC_CHECKPOINT} — "
                        f"le restaurer avant de continuer.")

# SHA-256 attendu du checkpoint BC verrouillé.
# La campagne vérifie cette empreinte avant toute interprétation des résultats.
CHECKPOINT_SHA256_EXPECTED = (
    "faabe390e43c604dad68d8f44bf5c67fc5630a43ba1d9d34b45fa3281ae038c9"
)

CAMPAIGN_ID = "TableII_v2_20260924"

ALL_SCENARIOS = ['S1_Nominal', 'S2_FlashCrowd', 'S3_PoP_Failure', 'S4_BGP_Anomaly']
BONFERRONI_ALPHA_PRIME = 0.05 / len(ALL_SCENARIOS)   # 0.0125, 4 comparaisons

# Protocole verrouillé — non paramétrable par appel de fonction.
LOCKED_N_SEEDS = 10
LOCKED_N_A = 50
LOCKED_N_P = 12
LOCKED_T_EP = 288

DEPENDENCY_FILES = [
    'GABPO_TableII_v2.py',
    'GABPO_P0b_DynamicEnv.py',
    'GABPO_P1b_BC.py',
]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _optional_sha256(path):
    return _sha256_file(path) if os.path.exists(path) else None


def _sha256_this_script():
    """Hash du fichier source de ce script lui-même, tel qu'exécuté."""
    this_path = os.path.abspath(__file__) if '__file__' in globals() else None
    if this_path and os.path.exists(this_path):
        return _sha256_file(this_path), this_path
    return None, None


def run_table2_v2():
    """
    Campagne indépendante et versionnée, verrouillée : BC (checkpoint
    verrouillé) vs B1, S1-S4, seeds 0..9 SANS offset, N_a=50, N_p=12,
    T_ep=288. Conserve les 40 observations brutes, pas seulement les
    moyennes. Aucun paramètre n'est ajustable par appel — le protocole
    verrouillé est fixé au niveau du module, pas de la fonction.
    """
    n_seeds, N_a, N_p, T_ep = LOCKED_N_SEEDS, LOCKED_N_A, LOCKED_N_P, LOCKED_T_EP

    # ── Provenance du checkpoint — ARRÊT si désaccord ──────────────────────
    ck_sha256 = _sha256_file(BC_CHECKPOINT)
    print(f"\n[Provenance] checkpoint SHA-256 : {ck_sha256}")
    print(f"[Provenance] attendu             : {CHECKPOINT_SHA256_EXPECTED}")
    if ck_sha256 != CHECKPOINT_SHA256_EXPECTED:
        raise RuntimeError(
            "\n"
            "ERREUR DE PROVENANCE : le SHA-256 du checkpoint BC ne correspond "
            "pas au checkpoint verrouillé attendu.\n"
            f"  attendu : {CHECKPOINT_SHA256_EXPECTED}\n"
            f"  obtenu  : {ck_sha256}\n"
            "\n"
            "La campagne Table II v2 est interrompue. "
            "Aucun résultat ne doit être interprété tant que le checkpoint "
            "n'est pas vérifié."
        )
    print("  ✓ Concordance exacte avec le checkpoint verrouillé.")

    # ── Provenance du script et de ses dépendances ─────────────────────────
    script_sha256, script_path = _sha256_this_script()
    dependency_hashes = {f: _optional_sha256(f) for f in DEPENDENCY_FILES}
    print(f"[Provenance] script SHA-256       : "
          f"{script_sha256 or 'indisponible (exec() sans __file__)'}")
    for fn, h in dependency_hashes.items():
        print(f"[Provenance] {fn:<30}: {h or 'introuvable dans le répertoire courant'}")

    # ── Charger BC une fois (poids gelés) ──────────────────────────────────
    policy = HierarchicalPolicy(N_a, N_p)
    policy.load_state_dict(torch.load(BC_CHECKPOINT, map_location='cpu'))
    policy.eval()
    b1 = B1_Static(N_a, N_p)
    print("\n✓ Checkpoint BC chargé, poids gelés")

    # ── Reprise : autorisée UNIQUEMENT si le manifeste existant correspond
    #    exactement à cette campagne (même campaign_id, mêmes hash de
    #    dépendances). Sinon, on repart de zéro pour éviter tout mélange
    #    entre deux versions expérimentales. ─────────────────────────────
    raw_path = 'results_table2_v2/table2_v2_raw.csv'
    manifest_path = 'results_table2_v2/table2_v2_manifest.json'
    evals = []
    done = set()
    resume_ok = False
    if os.path.exists(raw_path) and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            prev_manifest = json.load(f)
        same_campaign = (
            prev_manifest.get('campaign_id') == CAMPAIGN_ID
            and prev_manifest.get('checkpoint_sha256_actual') == ck_sha256
            and prev_manifest.get('script_sha256') == script_sha256
            and prev_manifest.get('dependency_sha256') == dependency_hashes
        )
        if same_campaign:
            prev = pd.read_csv(raw_path)
            evals = prev.to_dict('records')
            done = set(zip(prev.seed, prev.scenario))
            resume_ok = True
            print(f"\n✓ Reprise autorisée — manifeste identique : "
                  f"{len(done)}/{n_seeds * len(ALL_SCENARIOS)} observations déjà présentes")
        else:
            print(f"\n⚠ Fichiers précédents trouvés mais manifeste DIFFÉRENT "
                  f"(script/dépendances/checkpoint modifiés depuis le dernier "
                  f"lancement) — reprise refusée pour éviter un mélange de "
                  f"campagnes. Les anciens fichiers results_table2_v2/*.csv "
                  f"seront écrasés.")

    if not resume_ok and os.path.exists(raw_path):
        # Pas de fusion silencieuse : on repart d'un état vide.
        evals = []
        done = set()

    print(f"\n[Évaluation] campaign_id={CAMPAIGN_ID} | BC vs B1 | "
          f"seeds 0..{n_seeds-1} (SANS offset) | scénarios {ALL_SCENARIOS}")
    print("=" * 70)

    for seed in range(n_seeds):
        for scen in ALL_SCENARIOS:
            if (seed, scen) in done:
                continue
            r_bc = run_episode(policy, scen, seed, N_a, N_p, T_ep)
            r_b1 = run_episode(b1, scen, seed, N_a, N_p, T_ep)
            gain = (r_b1['L_med'] - r_bc['L_med']) / r_b1['L_med'] * 100
            evals.append({
                'campaign_id': CAMPAIGN_ID,
                'seed': seed, 'scenario': scen,
                'BC_Lmed': r_bc['L_med'], 'B1_Lmed': r_b1['L_med'],
                'gain_pct': gain,
            })
            print(f"  seed={seed} {scen:<16} BC={r_bc['L_med']:.2f}ms "
                  f"B1={r_b1['L_med']:.2f}ms gain={gain:+.3f}%")
            # sauvegarde après CHAQUE observation — reprise si interrompu
            pd.DataFrame(evals).to_csv(raw_path, index=False)

    df_raw = pd.DataFrame(evals)

    # ── Validation stricte des 40 observations ─────────────────────────────
    expected_pairs = {(s, sc) for s in range(n_seeds) for sc in ALL_SCENARIOS}
    actual_pairs = set(zip(df_raw['seed'], df_raw['scenario']))
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise RuntimeError(
            f"Jeu d'observations invalide.\n"
            f"Manquantes : {missing}\n"
            f"Supplémentaires : {extra}"
        )
    if df_raw.duplicated(['seed', 'scenario']).any():
        raise RuntimeError("Doublons détectés dans les couples seed/scenario.")
    if df_raw[['BC_Lmed', 'B1_Lmed', 'gain_pct']].isna().any().any():
        raise RuntimeError("Valeurs NaN détectées dans les résultats.")
    print(f"\n✓ {len(df_raw)} observations brutes validées "
          f"(exactement {n_seeds} seeds x {len(ALL_SCENARIOS)} scénarios, "
          f"sans doublon ni NaN)")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSE — par scénario, Student-t CI + Wilcoxon + Bonferroni
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\nANALYSE — Table II v2 (BC vs B1, S1-S4)\n{'='*70}")

    summary_rows = []
    for scen in ALL_SCENARIOS:
        sub = df_raw[df_raw.scenario == scen].sort_values('seed')
        gains = sub.gain_pct.values
        n = len(gains)
        mean_g, sd_g = gains.mean(), gains.std(ddof=1)
        ci = stats.t.interval(0.95, n - 1, loc=mean_g, scale=stats.sem(gains))
        try:
            _, p_wilcoxon = stats.wilcoxon(
                sub['BC_Lmed'].to_numpy(),
                sub['B1_Lmed'].to_numpy(),
                alternative='two-sided',
                method='auto',
            )
        except Exception:
            p_wilcoxon = float('nan')
        sig = p_wilcoxon < BONFERRONI_ALPHA_PRIME if not np.isnan(p_wilcoxon) else False

        print(f"\n  {scen} (n={n})")
        print(f"    Gain moyen : {mean_g:+.3f}% ± {sd_g:.3f}%")
        print(f"    95% CI     : [{ci[0]:+.3f}%, {ci[1]:+.3f}%]")
        print(f"    Wilcoxon p : {p_wilcoxon:.6f}")
        print(f"    Significatif (Bonferroni α'={BONFERRONI_ALPHA_PRIME}) : {sig}")

        summary_rows.append({
            'scenario': scen, 'n': n,
            'B1_mean_ms': sub.B1_Lmed.mean(), 'B1_sd_ms': sub.B1_Lmed.std(ddof=1),
            'BC_mean_ms': sub.BC_Lmed.mean(), 'BC_sd_ms': sub.BC_Lmed.std(ddof=1),
            'gain_pct': mean_g, 'gain_sd': sd_g,
            'ci_lo': ci[0], 'ci_hi': ci[1],
            'wilcoxon_p': p_wilcoxon,
            'bonferroni_alpha_prime': BONFERRONI_ALPHA_PRIME,
            'significant': bool(sig),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('results_table2_v2/table2_v2_summary.csv', index=False)

    # ══════════════════════════════════════════════════════════════════════
    # MÉTADONNÉES ET MANIFESTE DE PROVENANCE
    # ══════════════════════════════════════════════════════════════════════
    metadata = {
        'campaign_id': CAMPAIGN_ID,
        'campaign_name': 'GABPO_TableII_v2',
        'is_reconstruction_of_historical_table_ii': False,
        'note': ("Independent, versioned re-evaluation campaign. Does not "
                 "read, inject, or compare against any historical Table II "
                 "value (ICCT v12, LOCKED_REFERENCE, or LOCKED_BC) during "
                 "execution. Historical S1/S3/S4 values could not be traced "
                 "to any script, CSV, or Git object despite exhaustive "
                 "provenance search; see manuscript Reproducibility note."),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'torch_version': torch.__version__,
        'pandas_version': pd.__version__,
    }
    manifest = {
        'campaign_id': CAMPAIGN_ID,
        'script_sha256': script_sha256,
        'script_path': script_path,
        'dependency_sha256': dependency_hashes,
        'checkpoint_path': BC_CHECKPOINT,
        'checkpoint_sha256_actual': ck_sha256,
        'checkpoint_sha256_expected': CHECKPOINT_SHA256_EXPECTED,
        'checkpoint_sha256_match': True,
        'seeds': list(range(n_seeds)),
        'seed_offset': 0,
        'scenarios': ALL_SCENARIOS,
        'N_a': N_a, 'N_p': N_p, 'T_ep': T_ep,
        'n_observations': int(len(df_raw)),
        'n_expected_observations': n_seeds * len(ALL_SCENARIOS),
        'gain_formula': "G = (B1_Lmed - BC_Lmed) / B1_Lmed * 100",
        'ci_method': "Student-t, 95%, n=10 per scenario",
        'statistical_test': ("paired Wilcoxon signed-rank, two-sided, "
                              "method='auto', on (BC_Lmed, B1_Lmed)"),
        'correction': f"Bonferroni, family=S1-S4, alpha_prime={BONFERRONI_ALPHA_PRIME}",
        'metadata': metadata,
    }
    with open('results_table2_v2/table2_v2_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*70}\nRÉSUMÉ\n{'='*70}")
    print(df_summary.to_string(index=False))

    print(f"\n✓ Sauvegardé : results_table2_v2/table2_v2_raw.csv (40 observations)")
    print(f"✓ Sauvegardé : results_table2_v2/table2_v2_summary.csv")
    print(f"✓ Sauvegardé : results_table2_v2/table2_v2_metadata.json")
    print(f"✓ Sauvegardé : results_table2_v2/table2_v2_manifest.json")
    print(f"\n→ Télécharger :")
    print(f"   from google.colab import files")
    print(f"   for fn in ['table2_v2_raw.csv', 'table2_v2_summary.csv',")
    print(f"              'table2_v2_metadata.json', 'table2_v2_manifest.json']:")
    print(f"       files.download(f'results_table2_v2/{{fn}}')")

    return df_raw, df_summary, manifest


if __name__ == "__main__":
    df_raw, df_summary, manifest = run_table2_v2()
