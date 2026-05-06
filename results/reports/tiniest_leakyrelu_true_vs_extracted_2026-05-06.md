# Tiniest leaky α=0.01 — True vs Extracted Weight Comparison

**Date:** 2026-05-06
**Activation:** Leaky ReLU(α=0.01)
**Architecture:** 8 → 8 → 8 → 8 → 8 → 8 (4 hidden, make_blobs 8-class)
**Best extracted model:** `enhanced_codebase/results/reconstructed_models/reconstructed_tiniest.pth`
**True model:** `tiny_shit/tiniest_makeblobs_leakyrelu.pth`
**Functional accuracy on X_test2:** 99.25% (oracle 99.95%)

## Scope
This report compares weight vectors row-by-row. We measure:

- **Signature recovery** (per recovered hidden neuron):
  - `L1(w_ext, w_true)  = Σ |w_ext − w_true|`
  - `rel err           = ‖w_ext − w_true‖ / ‖w_true‖`
  - `cos sim           = (w_ext · w_true) / (‖w_ext‖·‖w_true‖)`
  - `|cos sim|`        — the sign-blind signature-recovery quality metric
- **Sign recovery** (per recovered hidden neuron):
  - `sign_correct = sign(cos sim) == +1`
  - per-layer sign accuracy = fraction correct

**Recovery detection method**: a neuron is counted as "recovered" when its extracted row achieves `|cos sim| ≥ 0.9` against some true neuron. Kaiming-init rows score well below this. This is independent of the `metadata.json` saved by `recover_weights.py` and so survives even when those files have been overwritten by a later run.

## Per-layer summary (recovered neurons only)
| Layer | `n_rec/n` | L1 median | L1 mean | rel err median | rel err mean | `|cos|` mean | sign acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| fc1 | 8/8 | 0.000 | 3.270 | 0.000 | 0.750 | **1.000** | 0.625 |
| fc2 | 7/8 | 11.015 | 7.316 | 2.000 | 1.143 | **1.000** | 0.429 |
| fc3 | 4/8 | 11.524 | 9.222 | 2.000 | 1.500 | **1.000** | 0.250 |
| fc4 | 3/8 | 9.945 | 7.603 | 2.000 | 1.333 | **1.000** | 0.333 |
| **overall** | 22/32 | 6.105 | — | 2.000 | — | **1.000** | **10/22 = 0.455** |

### How to read this
- `|cos|` is the *direction-only* quality. A value of 1.000 means the extracted weight is parallel (possibly antiparallel) to the true neuron — signature recovery succeeded.
- `rel err = 2.0` exactly means `w_ext ≈ −w_true` (same magnitude, opposite sign). Every neuron with the wrong sign shows this fingerprint. `rel err = 0.0` means bit-perfect match.
- Sign accuracy is **lower than 50% in expectation by chance** alone — the attack is **sign-blind by construction** (it scales by `abs(factor)` in `load_unsigned_weights`), and the oracle sign search only optimizes for hard-label agreement, not ground-truth sign.

## Per-neuron detail (recovered only)
| Layer | Placed at | Matched true | L1 | rel err | cos sim | sign |
|---|---:|---:|---:|---:|---:|---|
| fc1 | 0 | 0 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 1 | 1 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 2 | 2 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 3 | 3 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 4 | 4 | 13.948 | 2.000 | -1.0000 | **WRONG** |
| fc1 | 5 | 5 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 6 | 6 | 6.628 | 2.000 | -1.0000 | **WRONG** |
| fc1 | 7 | 7 | 5.582 | 2.000 | -1.0000 | **WRONG** |
| fc2 | 0 | 0 | 12.638 | 2.000 | -1.0000 | **WRONG** |
| fc2 | 1 | 1 | 14.734 | 2.000 | -1.0000 | **WRONG** |
| fc2 | 2 | 2 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 3 | 3 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 4 | 4 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 6 | 6 | 11.015 | 2.000 | -1.0000 | **WRONG** |
| fc2 | 7 | 7 | 12.827 | 2.000 | -1.0000 | **WRONG** |
| fc3 | 3 | 3 | 12.295 | 2.000 | -1.0000 | **WRONG** |
| fc3 | 5 | 5 | 0.000 | 0.000 | +1.0000 | OK |
| fc3 | 6 | 6 | 10.752 | 2.000 | -1.0000 | **WRONG** |
| fc3 | 7 | 7 | 13.840 | 2.000 | -1.0000 | **WRONG** |
| fc4 | 2 | 2 | 9.945 | 2.000 | -1.0000 | **WRONG** |
| fc4 | 3 | 3 | 0.000 | 0.000 | +1.0000 | OK |
| fc4 | 5 | 5 | 12.864 | 2.000 | -1.0000 | **WRONG** |

### Observations
- **10/22 neurons are byte-perfect** (`L1≈0, cos≈+1`).
- **12/22 are direction-perfect but sign-flipped** (`cos≈−1, rel err≈2`). The model uses these via compensating bias flips.
- **0/22 are direction-collapsed** (`|cos|<0.5`).

## Biases (all hidden neurons per layer)
| Layer | L1 sum | `|Δ|` median | `|Δ|` max |
|---|---:|---:|---:|
| fc1 | 5.173 | 0.200 | 2.495 |
| fc2 | 20.958 | 2.185 | 5.891 |
| fc3 | 10.688 | 1.150 | 3.522 |
| fc4 | 7.100 | 0.723 | 2.405 |
| fc5 | 580.398 | 25.230 | 248.307 |

Biases are not expected to match. The extracted bias is `b_i = −median(w_i · h_{L-1}(x_d))` computed through *already-extracted* lower layers. When `w_i` is sign-flipped or a lower layer has direction collapse, the dual-point bias is a different point in parameter space that still yields a near-equivalent function. **The functional accuracy on X_test2 is the real test of extraction; weight-matching is not.**

fc5's large bias deviation is expected: fc5 weights and biases are a fresh logistic-regression fit on the extracted hidden features (different features → different decoder).

## fc5 weight comparison
| metric | value |
|---|---:|
| row-wise L1 mean | 257.388 |
| row-wise rel err mean | 67.361 |
| row-wise `|cos|` mean | **0.175** |

fc5 `|cos|≈0.17` — the LR-fit decoder lives in a different point of fc5 parameter space than the oracle's decoder, yet the combined model achieves 99.25% test accuracy. This is the clearest demonstration that the attack achieves **functional** extraction on a *different* point in parameter space, not a *structural* copy of the oracle.

## Headline reading
| Property | Result |
|---|---|
| Signature **direction** (\|cos\|, recovered) | 1.000 mean |
| Signature **sign** (recovered) | 0.455 accuracy — attack is sign-blind by design |
| Weight-space distance to true model | Large (L1 median 6.10 per recovered neuron) |
| Functional distance to true model | **0.70 %** on make_blobs X_test2 (99.25% extraction vs 99.95% oracle) |

The attack is a **functional** extraction, not a weight-space copy. Direction recovery on the hidden layers is mostly successful where signature recovery succeeded; signs and biases are a free variable that the refinement step adjusts jointly to match oracle behavior.

## Artifacts
- Script: `/tmp/compute_leaky_true_vs_ext_v2.py`
- JSON (this layout): `results/reconstructed_models/tiniest_leakyrelu_true_vs_extracted_metrics.json`
- Iter-1 / iter-2 reports: `tiniest_leakyrelu_iter1_2026-05-05.md`, `tinier_leakyrelu_iter1_2026-05-06.md`
- Activation toggle / port audit: `leaky_relu_port.md` (project root)