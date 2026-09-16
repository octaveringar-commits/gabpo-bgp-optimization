# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GABPO — Traffic Engineering Baselines (TE1, TE2, TE3)                ║
# ║                                                                        ║
# ║  Addresses reviewer point #5 (ICCT + 2 independent Computer Networks   ║
# ║  reviews): B1 (static) and B2 (greedy-by-load) alone are judged        ║
# ║  insufficient. This script adds three baselines grounded in           ║
# ║  established Traffic Engineering literature, adapted to the GABPO     ║
# ║  dual-graph (AS/PoP) BGP action space (Section III of the manuscript).║
# ║                                                                        ║
# ║  TE1 — OSPF-like shortest path: deterministic, routes toward the      ║
# ║        PoP with lowest static RTT — no adaptation to load or state.   ║
# ║        Represents the conventional non-adaptive baseline referenced   ║
# ║        by the reviewers.                                              ║
# ║                                                                        ║
# ║  TE2 — MATE-adapted (Elwalid et al., 2001, "MATE: MPLS Adaptive        ║
# ║        Traffic Engineering"): iteratively shifts traffic AWAY from     ║
# ║        congested PoPs and TOWARD underutilized ones, proportional to  ║
# ║        the *gradient* of observed congestion — a smoothed, adaptive   ║
# ║        multi-path load-balancing update, not a one-shot greedy pick.  ║
# ║        Adapted from link-level congestion gradient (original MATE) to ║
# ║        PoP-level congestion gradient (this environment's action       ║
# ║        space is PoP-indexed, not link-indexed).                       ║
# ║                                                                        ║
# ║  TE3 — TeXCP-adapted (Kandula et al., 2005, "Walking the Tightrope:    ║
# ║        Responsive Yet Stable Traffic Engineering"): utilization-      ║
# ║        based multipath splitting — redistributes traffic in           ║
# ║        proportion to each PoP's INVERSE utilization ratio, aiming to  ║
# ║        equalize relative load across PoPs rather than following a     ║
# ║        congestion gradient (TE2) or a static shortest path (TE1).     ║
# ║        This distinguishes TE3 from TE2 by objective (load EQUALITY)   ║
# ║        rather than congestion AVOIDANCE.                              ║
# ║                                                                        ║
# ║  All three operate directly on the flat (N_p*N_a*2,) BGP action        ║
# ║  vector, identical interface to B1_Static / B2_Greedy, so they plug   ║
# ║  directly into run_episode() without any environment changes.         ║
# ║                                                                        ║
# ║  Honesty note: none of these are literal reimplementations of the      ║
# ║  original algorithms (which target MPLS/intra-domain multipath, not   ║
# ║  BGP/Anycast) — they are adaptations preserving each algorithm's       ║
# ║  core control principle within GABPO's action space. This is stated   ║
# ║  explicitly in the manuscript text, not left implicit.                ║
# ║                                                                        ║
# ║  Prerequisites: P0b (BGPDynamicEnv, run_episode) already loaded.       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import numpy as np


class TE1_OSPF:
    """
    Deterministic shortest-path-like baseline. Routes all AS traffic
    toward whichever PoP has the lowest static RTT to the WACREN
    region — analogous to OSPF selecting the lowest-cost path and
    never revisiting that choice regardless of current load.
    No adaptation to congestion or state: this is the conventional,
    non-adaptive point of comparison.
    """
    def __init__(self, N_a, N_p, static_rtt=None):
        self.N_a, self.N_p = N_a, N_p
        # Static RTT per PoP — fixed at construction, never updated.
        # If not provided, use a deterministic proxy ordering (PoP index
        # as a stand-in for geographic distance, consistent across seeds).
        if static_rtt is None:
            static_rtt = np.linspace(1.0, 2.0, N_p)
        self.best_pop = int(np.argmin(static_rtt))

    def select_action(self, obs):
        a = np.zeros(self.N_p * self.N_a * 2, dtype=np.float32)
        off = self.N_p * self.N_a
        for i in range(self.N_a):
            a[off + self.best_pop * self.N_a + i] = 1.0
        return a

    def name(self):
        return 'TE1_OSPF'


class TE2_MATE:
    """
    MATE-adapted: congestion-gradient-based multipath adjustment.
    At each step, computes the load imbalance between the most and
    least loaded PoPs and shifts a LOCAL_PREF adjustment proportional
    to that gradient — congested PoPs are actively de-preferred,
    underloaded PoPs are actively preferred, smoothly rather than via
    a one-shot top-k selection (contrast with B2_Greedy).
    """
    def __init__(self, N_a, N_p, gain=0.5):
        self.N_a, self.N_p = N_a, N_p
        self.gain = gain  # step size on the congestion gradient

    def select_action(self, obs):
        load = obs.get('load', np.zeros(self.N_p))[:self.N_p]
        a = np.zeros(self.N_p * self.N_a * 2, dtype=np.float32)
        off = self.N_p * self.N_a

        mean_load = load.mean()
        std_load = load.std() + 1e-8
        # Normalized congestion gradient per PoP (positive = overloaded)
        gradient = (load - mean_load) / std_load

        most_congested = int(np.argmax(gradient))
        least_congested = int(np.argmin(gradient))

        # Only act if there is a meaningful imbalance to correct
        if gradient[most_congested] - gradient[least_congested] > 0.1:
            amplitude = float(np.clip(self.gain * abs(gradient[most_congested]), 0.1, 1.0))
            for i in range(self.N_a):
                # De-prefer the congested PoP
                a[off + most_congested * self.N_a + i] = -amplitude
                # Prefer the underloaded PoP
                a[off + least_congested * self.N_a + i] = amplitude
        return a

    def name(self):
        return 'TE2_MATE'


class TE3_TeXCP:
    """
    TeXCP-adapted: utilization-equalization multipath splitting.
    Rather than reacting to a congestion gradient (TE2), TE3 aims to
    equalize the *relative utilization ratio* across all PoPs by
    redistributing preference in proportion to each PoP's inverse
    utilization — PoPs far below the mean utilization gain preference
    proportionally, not just the single least-loaded one.
    """
    def __init__(self, N_a, N_p, capacity=None):
        self.N_a, self.N_p = N_a, N_p
        # Assumed per-PoP capacity (uniform if not provided) — utilization
        # is computed as load / capacity, following TeXCP's utilization
        # ratio formulation rather than raw load.
        self.capacity = capacity if capacity is not None else np.ones(N_p)

    def select_action(self, obs):
        load = obs.get('load', np.zeros(self.N_p))[:self.N_p]
        a = np.zeros(self.N_p * self.N_a * 2, dtype=np.float32)
        off = self.N_p * self.N_a

        utilization = load / (self.capacity[:self.N_p] + 1e-8)
        mean_util = utilization.mean()

        # Preference proportional to how far BELOW the mean each PoP's
        # utilization sits — under-utilized PoPs get positive LP,
        # over-utilized PoPs get negative LP, scaled by deviation.
        deviation = mean_util - utilization  # positive = underutilized
        max_dev = np.abs(deviation).max() + 1e-8

        for p in range(self.N_p):
            if abs(deviation[p]) > 0.05 * max_dev:  # ignore negligible imbalance
                amplitude = float(np.clip(abs(deviation[p]) / max_dev, 0.1, 1.0))
                sign = 1.0 if deviation[p] > 0 else -1.0
                for i in range(self.N_a):
                    a[off + p * self.N_a + i] = sign * amplitude
        return a

    def name(self):
        return 'TE3_TeXCP'


# ══════════════════════════════════════════════════════════════════════════
# Sanity check — verify all three produce valid (N_p*N_a*2,) actions
# ══════════════════════════════════════════════════════════════════════════

def run_te_sanity_check(N_a=50, N_p=12):
    """
    Structural check only — no evaluation, no training. Confirms the
    three baselines produce correctly shaped actions on a real
    observation before running any seeded evaluation.
    """
    print("\n[Sanity Check] TE1/TE2/TE3 baselines")
    print("=" * 60)
    try:
        _ = BGPDynamicEnv
    except NameError:
        raise RuntimeError("Exécuter GABPO_P0b_DynamicEnv.py d'abord.")

    env = BGPDynamicEnv('S1_Nominal', 0, N_a, N_p, T_ep=10)
    obs = env.reset()

    baselines = {
        'TE1_OSPF':  TE1_OSPF(N_a, N_p),
        'TE2_MATE':  TE2_MATE(N_a, N_p),
        'TE3_TeXCP': TE3_TeXCP(N_a, N_p),
    }

    all_ok = True
    for name, model in baselines.items():
        try:
            action = model.select_action(obs)
            shape_ok = (action.shape == (N_p * N_a * 2,))
            nonzero = int((action != 0).sum())
            status = '✓' if shape_ok else '✗'
            if not shape_ok:
                all_ok = False
            print(f"  {status} {name:<12} shape={action.shape} "
                  f"nonzero_entries={nonzero}")
        except Exception as e:
            all_ok = False
            print(f"  ✗ {name:<12} ERREUR: {type(e).__name__}: {e}")

    print(f"\n{'✓ Toutes les baselines TE sont fonctionnelles' if all_ok else '✗ ÉCHEC — corriger avant évaluation'}")
    return all_ok


if __name__ == "__main__":
    run_te_sanity_check()
