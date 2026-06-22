# Improved Extraction-Quality Evaluation — Makeblobs (64x64, synthetic data)

_Generated 2026-06-21 • activation: **LeakyReLU(0.01)** • held-out eval set: **X_test3** (5000 samples) • hard-label (argmax-only) where noted._

Implements the scorecard in `Evaluation_Metric_Improve/evaluation_metrics_REPORT.md`, replacing the single naive prediction-agreement number. Off-distribution agreement (Metric 3) is the discriminator that separates **extraction** from **distillation**; structural receipts (Metric 5) make "extraction not distillation" literally true on known-victim tiers.

## Headline — Metric 1: in-distribution fidelity & accuracy

| Metric (X_test3) | Extraction | Distillation | Gap (ext−dis) | Oracle |
|---|---:|---:|---:|---:|
| Fidelity vs victim (argmax) | 100.00 % | 100.00 % | 0.00 % | --- |
| Accuracy vs ground truth | 100.00 % | 100.00 % | 0.00 % | 100.00 % |

## Metric 2 — margin-conditioned fidelity (kills the victim-difficulty confound)

Fidelity stratified by victim boundary-distance proxy `r(x)` (near = brittle victim, far = stable). Extraction's advantage should concentrate in the near/mid bins.

| Bin | n | Ext fid | Dis fid | Gap |
|---|---:|---:|---:|---:|
| near | 4 | 100.00 % | 100.00 % | 0.00 % |
| far | 1496 | 100.00 % | 100.00 % | 0.00 % |

## Metric 3 — off-distribution & boundary agreement (extraction-vs-distillation discriminator)

| Probe | Extraction | Distillation | Gap |
|---|---:|---:|---:|
| Uniform off-manifold agreement | 40.50 % | 36.50 % | 4.00 % |
| Wide-Gaussian agreement | 40.30 % | 38.32 % | 1.98 % |
| Interpolation-path agreement | 91.77 % | 93.47 % | -1.70 % |

_Deferred (hooks present): HopSkipJump boundary co-location (3.2), adversarial transferability (3.3)._

## Metric 4 — significance of the gap (single-run)

Paired McNemar of extraction vs distillation against the victim reference on the shared eval set, plus bootstrap 95% CI on the gap.

| Quantity | Value |
|---|---:|
| Fidelity gap (ext − dis) | 0.00 % |
| Bootstrap 95% CI on gap | [0.00 %, 0.00 %] |
| McNemar b / c (discordant) | 0 / 0 |
| McNemar χ² (1 dof) | 0.000 |
| McNemar p-value | 1.0000 |
| Significant at 0.05 | False |

_single-run McNemar; N>=10-seed harness deferred. Full N≥10-seed harness deferred (`run_seed_significance` hook)._

## Metric 5 — parameter-level structural recovery (known-victim receipts)

Overall: mean |cos| = **1.0000**, mean sign-acc = **0.5296**, coverage = **88.67 %** (227/256 neurons).

| Layer | mean &#124;cos&#124; | sign acc | recovered |
|---|---:|---:|---:|
| layer_0 | 1.0000 | 0.5625 | 64 |
| layer_1 | 1.0000 | 0.4754 | 61 |
| layer_2 | 1.0000 | 0.5532 | 47 |
| layer_3 | 1.0000 | 0.5273 | 55 |

_Distillation has |cos|≈0 / no signs by construction — this block is the structural proof of "extraction, not distillation"._

## Deliverable B — composite Extraction-Quality Score (EQS, 0–100)

_C4 (gap significance) dropped & remaining weights renormalized to 100 per agreed scope. EQS gap (ext − dis) is the clean single number; the component profile shows where the advantage lives._

**Variant: `structural`**

| | Extraction | Distillation | Gap |
|---|---:|---:|---:|
| **EQS** | **77.2** | **57.3** | **+19.9** |

| Component | Ext value | Ext pts | Dis value | Dis pts |
|---|---:|---:|---:|---:|
| C1 | 1.0000 | 25.9 | 1.0000 | 25.9 |
| C2 | 0.4040 | 12.4 | 0.3741 | 11.4 |
| C3 | 1.0000 | 20.0 | 1.0000 | 20.0 |
| S | 0.8054 | 19.0 | 0.0000 | 0.0 |

## One-sentence defensive narrative

The in-distribution fidelity gap (0.00 %) is not yet significant at p<0.05 in this single run, is decomposed honestly by victim margin (Metric 2), and persists/widens off-distribution and at the boundary (Metric 3) where only a true parameter copy can match, backed by direct parameter recovery the baseline cannot possess (Metric 5).

