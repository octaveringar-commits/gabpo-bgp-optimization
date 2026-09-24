# GABPO — Locked TE vs BC Audit Dataset

This directory contains the reproducible locked results for the
TE-baseline versus BC comparison.

## Experimental design

- TE baselines: TE1_OSPF, TE2_MATE, TE3_TeXCP
- Scenarios: S1–S5
- Seeds: 0–9
- Environment seed: `seed + 900`
- N_a = 50
- N_p = 12
- Episode length: T_ep = 288
- TE observations: 150
- TE-vs-BC comparisons: 15

## Statistical analysis

Paired Wilcoxon signed-rank tests were computed for the 15
TE-vs-BC comparisons.

Multiple testing correction:
Holm-Bonferroni, alpha = 0.05.

## Three historically missing comparisons

| Comparison | Scenario | Gain (%) | Raw p | Holm p |
|---|---|---:|---:|---:|
| TE1_OSPF vs BC | S2_FlashCrowd | +0.664818 | 0.232422 | 0.464844 |
| TE2_MATE vs BC | S3_PoP_Failure | +0.476946 | 0.105469 | 0.316406 |
| TE3_TeXCP vs BC | S3_PoP_Failure | -0.009084 | 0.769531 | 0.769531 |

All three comparisons are non-significant after
Holm-Bonferroni correction.

## Important provenance note

The historical value p = 0.2324 for TE1/S2 versus BC is confirmed
by direct reproduction using the current BC checkpoint and the
10-seed TE campaign.

The older `LOCKED_BC` values embedded in legacy evaluation code
must not be interpreted as the source of the present TE-vs-BC
statistics.

## Checkpoint

The BC checkpoint used for direct comparison is:

`bc_phase_D.pt`

Its SHA256 is recorded in `PROVENANCE.json`.

## Files

- `TE1_OSPF_results.csv`
- `TE2_MATE_results.csv`
- `TE3_TeXCP_results.csv`
- `GABPO_TE_all_results_LOCKED.csv`
- `TE_vs_BC_Wilcoxon_raw_LOCKED.csv`
- `TE_vs_BC_HolmBonferroni_LOCKED.csv`
- `TE_BC_three_target_comparisons_LOCKED.csv`
- `bc_phase_D.pt`
- `PROVENANCE.json`

These files are intended as an audit/provenance snapshot and
should not be silently overwritten by subsequent experiments.
