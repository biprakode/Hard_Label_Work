# Tiny — True vs Extracted Weight Comparison

**Date:** 2026-04-24
**Best extracted model:** `results/reconstructed_models/reconstructed_tiny_frozen.pth`
(1000-epoch frozen refinement, 100.00 % on make_blobs test)
**True model:** `tiny_stuff/makeblobs_relu.pth` (64→64→64→64→64→10, make_blobs)

## Scope
Same metrics as the tiniest report, applied to the tiny 5-hidden-layer 64-wide network:

- **Signature recovery** (per recovered hidden neuron):
  - `L1(w_ext, w_true)  = Σ |w_ext − w_true|`
  - `rel err           = ‖w_ext − w_true‖ / ‖w_true‖`
  - `cos sim           = (w_ext · w_true) / (‖w_ext‖·‖w_true‖)`
  - `|cos sim|`        — sign-blind signature-recovery quality
- **Sign recovery** (per recovered hidden neuron):
  - `sign_correct = sign(cos sim) == +1`
  - per-layer sign accuracy = fraction correct

Unrecovered neurons (Kaiming-random placeholders) are excluded from the hidden-layer tables.

## Per-layer summary (recovered hidden neurons only)
| Layer | `n_rec/n` | L1 median | L1 mean | rel err median | rel err mean | `|cos|` mean | sign acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| fc1 | 64/64 | 0.0000 | 5.633 | 0.0000 | 0.906 | **1.000** | 0.547 |
| fc2 | 61/64 | 10.216 | 6.936 | 2.000 | 1.116 | **0.982** | 0.459 |
| fc3 | 44/64 | 8.203 | 8.733 | 1.516 | 1.299 | **0.726** | 0.545 |
| fc4 | 0/64  | —      | —     | —     | —     | **—**    | —     |
| **overall** | 169/256 | 8.092 | — | 1.451 | — | **0.922** | **87/169 = 0.515** |

### How to read this
- `|cos|` falls sharply with depth: **fc1 = 1.000 (perfect directions), fc2 = 0.982, fc3 = 0.726, fc4 = 0 (total failure)**. This is the expected pattern — deeper layers require forward-propagating dual points through more non-linear prefix layers, which amplifies numerical error until SVD-rank recovery fails.
- `rel err = 2.0` = sign flip with correct magnitude. `rel err = 0.0` = bit-perfect. `rel err ≈ 1.0` = orthogonal direction.
- **fc1 has 64/64 byte-perfect directions.** Every fc1 neuron was recovered with `|cos|=1`.
- **fc4 fully failed signature recovery.** All 51 fc4 clusters returned `Singular values None` (MathIsHard exception in `is_consistent`) so `recover_weights.py` wrote no weights for fc4. The reconstructed fc4 is entirely Kaiming-random.
- Sign accuracy averages 51.5 % across recovered hidden neurons — close to chance. The attack is sign-blind by construction (scales by `abs(factor)`), and the sign search step on this run did not move the agreement needle (0.0787 → 0.0787 over 2 passes).

## Per-neuron detail — fc1 (all 64 recovered)
All 64 fc1 neurons have `|cos| = 1.000` in the detailed JSON (`true_vs_extracted_tiny_metrics.json`). 35 / 64 have `sign_correct=True`; 29 are sign-flipped with `rel_err = 2.0`.

Representative samples (from `true_vs_extracted_tiny_metrics.json`):

| Neuron | L1 | rel err | cos sim | sign |
|---|---:|---:|---:|---|
| 0 | 0.000 | 0.0000 | +1.0000 | OK |
| 1 | 21.5 | 2.0000 | −1.0000 | **WRONG** |
| 2 | 0.000 | 0.0000 | +1.0000 | OK |
| 3 | 17.9 | 2.0000 | −1.0000 | **WRONG** |
| … | … | … | … | … |

Structure: byte-perfect or magnitude-perfect-sign-flipped — no intermediate cases. This confirms the fc1 signature-recovery layer is doing direction recovery correctly, and sign recovery is the only source of error.

## Per-neuron detail — fc2 (61 recovered, 3 random-init)
- **fc2 |cos|_mean = 0.9824** — 60 of 61 have `|cos| ≥ 0.97`, one outlier with lower similarity (still recovered but noisy direction).
- Sign accuracy 45.9 %.

## Per-neuron detail — fc3 (44 recovered, 20 random-init)
- **fc3 |cos|_mean = 0.7262** — bimodal distribution: ~28 neurons with `|cos| ≥ 0.95`, ~16 with `|cos| ≤ 0.5` (SVD direction collapse or prefix-error contamination).
- Sign accuracy 54.5 %.

## Per-neuron detail — fc4 (0 recovered, 64 random-init)
Signature recovery failed for every fc4 cluster. See `results/reports/tiny_extraction_quality_2026-04-24.md` for a pipeline-level discussion of *why*.

## Biases (all 8 layers)
| Layer | L1 sum | `|Δ|` median | `|Δ|` max |
|---|---:|---:|---:|
| fc1 |   6.843 | 0.058 |  0.463 |
| fc2 |  52.849 | 0.800 |  2.064 |
| fc3 |  29.876 | 0.296 |  2.513 |
| fc4 |   4.866 | 0.082 |  0.170 |
| fc5 |  48.711 | 3.734 | 14.329 |

Biases are not expected to match — see the tiniest report for the rationale. The 100.00 % prediction agreement is the real test of functional equivalence, not weight matching.

## fc5 weight comparison
| metric | value |
|---|---:|
| row-wise L1 mean | 164.04 |
| row-wise rel err mean | 33.72 |
| row-wise `|cos|` mean | **0.088** |

fc5 `|cos|≈0.09` — essentially uncorrelated with the true fc5 in weight space, yet the combined model achieves 100 % test accuracy. The fc5 is a fresh logistic-regression fit on the extracted hidden features using only oracle argmax labels; there is no reason for it to align with the true fc5 weights when the hidden features are functionally-equivalent but structurally-different.

## Headline reading of this report
| Property | Result |
|---|---|
| Signature **direction** (`|cos|`, recovered hidden neurons) | 0.922 mean, but **fc4 = 0 (complete failure)** |
| Signature **sign** (recovered hidden neurons) | 0.515 accuracy — close to chance, attack is sign-blind |
| Weight-space distance to true model | Large (L1 median 8.1 per recovered neuron) |
| Functional distance to true model | **0.00 %** on make_blobs test (10000 samples) |

The tiny attack, like the tiniest attack, is a **functional** extraction rather than a weight-space copy. Unlike tiniest, the signature-recovery quality visibly collapses at depth (fc3 `|cos|=0.73`, fc4 `|cos|=0`), meaning more of the functional equivalence is carried by refinement / distillation than by direct signature copies.

## Artifacts
- Script: `analysis/compare_true_vs_extracted_tiny.py`
- Extraction metrics JSON: `results/reconstructed_models/extraction_metrics.json`
- Per-neuron JSON: `results/reconstructed_models/true_vs_extracted_tiny_metrics.json`
- make_blobs accuracy JSON: `results/reconstructed_models/makeblobs_tiny_eval.json`
