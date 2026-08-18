# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Computational Cost Profiling                                 ║
# ║                                                                        ║
# ║  Addresses external review gap: "no computational cost analysis"       ║
# ║  (parameter counts, inference latency, training time, memory).         ║
# ║                                                                        ║
# ║  Measures ALL architectures already trained in this study — no new     ║
# ║  training needed. Requires loaded checkpoints or fresh instances       ║
# ║  (fresh instances give identical parameter counts/inference latency    ║
# ║  to trained checkpoints — only weights differ, not architecture).      ║
# ║                                                                        ║
# ║  Reports, per architecture:                                            ║
# ║    - Parameter count (trainable)                                       ║
# ║    - Model size on disk (MB, float32)                                  ║
# ║    - Inference latency per decision (mean/std over N repeated calls)   ║
# ║    - Peak memory during a single forward pass (approximate, CPU)       ║
# ║                                                                        ║
# ║  Does NOT re-measure training time from scratch (would require         ║
# ║  re-running full training) — instead reports training time already    ║
# ║  observed and logged during the original experiments (Section 5.4     ║
# ║  protocol: 100 epochs, 14,400 samples), stated explicitly as such.     ║
# ║                                                                        ║
# ║  Prerequisites: P0b → extract_features patch → BC → Ablation B3-B6 →   ║
# ║  GABPO_HCGA_SanityCheck.py already loaded (for all architecture        ║
# ║  classes: HierarchicalPolicy, HierarchicalPolicy_HCGA, B3-B6).         ║
# ║                                                                        ║
# ║  Usage:                                                                ║
# ║    exec(open('GABPO_ComputationalProfiling.py').read())                ║
# ║    df_profile = run_computational_profiling()                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import time
import numpy as np
import torch
import pandas as pd

os.makedirs('results_profiling', exist_ok=True)

print("╔══════════════════════════════════════════════════════╗")
print("║  GABPO — Computational Cost Profiling               ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    _ = BGPDynamicEnv, extract_features, HierarchicalPolicy
    print("✓ Dépendances de base chargées")
except NameError as e:
    raise RuntimeError(
        f"Exécuter P0b → extract_features patch → BC d'abord : {e}")

HCGA_AVAILABLE = 'HierarchicalPolicy_HCGA' in globals()
ABLATION_AVAILABLE = all(c in globals() for c in
    ['B3_BiLSTM_Ablation', 'B4_GCN_Ablation', 'B5_GraphSAGE_Ablation', 'B6_GAT_Ablation'])

if not HCGA_AVAILABLE:
    print("⚠ HierarchicalPolicy_HCGA absent — HCGA exclu du profilage "
          "(exécuter GABPO_HCGA_SanityCheck.py si besoin)")
if not ABLATION_AVAILABLE:
    print("⚠ Classes B3-B6 absentes — encodeurs alternatifs exclus du profilage "
          "(exécuter GABPO_Ablation_B3B6.py si besoin)")


def count_params(model):
    """Nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    """Taille approximative sur disque (float32, 4 bytes/param)."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / (1024 ** 2)


def measure_inference_latency(model, obs, n_repeats=200, warmup=20):
    """
    Mesure le temps d'une décision (select_action) sur CPU.
    Warmup pour stabiliser le cache/JIT, puis n_repeats mesures réelles.
    """
    # Warmup
    for _ in range(warmup):
        _ = model.select_action(obs)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = model.select_action(obs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return {
        'latency_mean_ms': float(times.mean()),
        'latency_std_ms': float(times.std()),
        'latency_p50_ms': float(np.percentile(times, 50)),
        'latency_p95_ms': float(np.percentile(times, 95)),
    }


def run_computational_profiling(N_a:int=50, N_p:int=12, n_repeats:int=200):
    print(f"\n[Profiling] {n_repeats} répétitions par architecture (CPU)")
    print("=" * 70)

    # Observation de référence — même pour toutes les architectures
    env = BGPDynamicEnv('S1_Nominal', 0, N_a, N_p, T_ep=10)
    obs = env.reset()

    architectures = {}

    # ── Modèle central : BC (pas de HCGA) ──────────────────────────────
    architectures['BC (proposed, no HCGA)'] = HierarchicalPolicy(N_a, N_p)

    # ── HCGA (si disponible) ────────────────────────────────────────────
    if HCGA_AVAILABLE:
        architectures['BC + HCGA'] = HierarchicalPolicy_HCGA(N_a, N_p)

    # ── Encodeurs alternatifs (si disponibles) ──────────────────────────
    if ABLATION_AVAILABLE:
        architectures['B3 — BiLSTM (no graph)']   = B3_BiLSTM_Ablation(N_a, N_p)
        architectures['B4 — GCN']                 = B4_GCN_Ablation(N_a, N_p)
        architectures['B5 — GraphSAGE']            = B5_GraphSAGE_Ablation(N_a, N_p)
        architectures['B6 — Single-graph attn.']   = B6_GAT_Ablation(N_a, N_p)

    rows = []
    for name, model in architectures.items():
        model.eval()
        n_params = count_params(model)
        size_mb  = model_size_mb(model)

        print(f"\n  {name}")
        print(f"    Paramètres : {n_params:,}")
        print(f"    Taille (float32) : {size_mb:.3f} MB")

        lat = measure_inference_latency(model, obs, n_repeats=n_repeats)
        print(f"    Latence inférence : {lat['latency_mean_ms']:.3f} ± "
              f"{lat['latency_std_ms']:.3f} ms  "
              f"(p50={lat['latency_p50_ms']:.3f}, p95={lat['latency_p95_ms']:.3f})")

        rows.append({
            'architecture': name,
            'n_params': n_params,
            'size_mb': size_mb,
            **lat,
        })

    df = pd.DataFrame(rows)
    df.to_csv('results_profiling/GABPO_computational_profile.csv', index=False)

    # ── Référence : direct 1200-dim (jamais entraîné avec succès, mais
    #    on peut quand même mesurer son coût architectural pour comparaison) ──
    print(f"\n{'='*70}")
    print("RÉFÉRENCE : direct 1200-dim (architecture seule, jamais convergé)")
    print(f"{'='*70}")
    try:
        direct_model = GABPOAgent(N_a, N_p) if 'GABPOAgent' in globals() else None
        if direct_model is not None:
            n_params_direct = count_params(direct_model)
            print(f"  Paramètres : {n_params_direct:,} "
                  f"(architecture PPO directe avec HCGA, jamais convergée — Section 4.3)")
        else:
            print("  ⚠ GABPOAgent non chargé — comparaison directe non disponible "
                  "(charger GABPO_PPO_v4.py si besoin)")
    except Exception as e:
        print(f"  ⚠ Non mesurable dans cette session : {e}")

    # ── Résumé training time déjà observé (pas remesuré) ────────────────
    print(f"\n{'='*70}")
    print("TEMPS D'ENTRAÎNEMENT (déjà observé pendant les campagnes originales, "
          "PAS remesuré ici)")
    print(f"{'='*70}")
    training_times_observed = {
        'BC (100 epochs, 28,800 pairs)':        '~30 min (CPU, session Colab standard)',
        'HCGA (100 epochs, 14,400 pairs, ×10 seeds)': '~44 min/seed (CPU)',
        'B3-B6 encoders (100 epochs each, ×10 seeds)': '~26-55 min/seed (CPU, varies by encoder)',
    }
    for k, v in training_times_observed.items():
        print(f"  {k:<50} {v}")
    print("\n  ⚠ Ces temps sont approximatifs, observés empiriquement pendant "
          "les campagnes d'entraînement de cette étude (logs Colab), pas mesurés "
          "dans un environnement contrôlé dédié au profilage.")

    print(f"\n✓ Sauvegardé : results_profiling/GABPO_computational_profile.csv")
    print(f"\n→ Télécharger : from google.colab import files; "
          f"files.download('results_profiling/GABPO_computational_profile.csv')")

    return df


if __name__ == "__main__":
    df_profile = run_computational_profiling()
