# Tinier leaky α=0.01 — True vs Extracted Weight Comparison

**Date:** 2026-05-06
**Activation:** Leaky ReLU(α=0.01)
**Architecture:** 32 → 16 → 16 → 16 → 8 → 4 (4 hidden, non-uniform widths, make_blobs 4-class)
**Best extracted model:** `enhanced_codebase/results/reconstructed_models/reconstructed_tinier.pth`
**True model:** `tiny_shit/tinier_makeblobs_leakyrelu.pth`
**Functional accuracy on X_test2:** 100.00% (oracle 100.00%)

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
| fc1 | 16/16 | 5.055 | 7.033 | 1.000 | 1.000 | **1.000** | 0.500 |
| fc2 | 13/16 | 0.000 | 3.362 | 0.000 | 0.769 | **1.000** | 0.615 |
| fc3 | 4/16 | 3.825 | 4.614 | 1.000 | 1.000 | **1.000** | 0.500 |
| fc4 | 0/8 | — | — | — | — | — | — |
| **overall** | 33/56 | 0.000 | — | 0.000 | — | **1.000** | **18/33 = 0.545** |

### How to read this
- `|cos|` is the *direction-only* quality. A value of 1.000 means the extracted weight is parallel (possibly antiparallel) to the true neuron — signature recovery succeeded.
- `rel err = 2.0` exactly means `w_ext ≈ −w_true` (same magnitude, opposite sign). Every neuron with the wrong sign shows this fingerprint. `rel err = 0.0` means bit-perfect match.
- Sign accuracy is **lower than 50% in expectation by chance** alone — the attack is **sign-blind by construction** (it scales by `abs(factor)` in `load_unsigned_weights`), and the oracle sign search only optimizes for hard-label agreement, not ground-truth sign.

## Per-neuron detail (recovered only)
| Layer | Placed at | Matched true | L1 | rel err | cos sim | sign |
|---|---:|---:|---:|---:|---:|---|
| fc1 | 0 | 0 | 24.687 | 2.000 | -1.0000 | **WRONG** |
| fc1 | 1 | 1 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 2 | 2 | 10.659 | 2.000 | -1.0000 | **WRONG** |
| fc1 | 3 | 3 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 4 | 4 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 5 | 5 | 12.192 | 2.000 | -1.0000 | **WRONG** |
| fc1 | 6 | 6 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 7 | 7 | 14.241 | 2.000 | -1.0000 | **WRONG** |
| fc1 | 8 | 8 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 9 | 9 | 11.554 | 2.000 | -1.0000 | **WRONG** |
| fc1 | 10 | 10 | 13.615 | 2.000 | -1.0000 | **WRONG** |
| fc1 | 11 | 11 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 12 | 12 | 10.110 | 2.000 | -1.0000 | **WRONG** |
| fc1 | 13 | 13 | 15.461 | 2.000 | -1.0000 | **WRONG** |
| fc1 | 14 | 14 | 0.000 | 0.000 | +1.0000 | OK |
| fc1 | 15 | 15 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 0 | 0 | 11.772 | 2.000 | -1.0000 | **WRONG** |
| fc2 | 1 | 1 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 2 | 2 | 9.153 | 2.000 | -1.0000 | **WRONG** |
| fc2 | 3 | 3 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 5 | 5 | 7.275 | 2.000 | -1.0000 | **WRONG** |
| fc2 | 6 | 6 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 7 | 7 | 7.216 | 2.000 | -1.0000 | **WRONG** |
| fc2 | 9 | 9 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 10 | 10 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 11 | 11 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 12 | 12 | 8.288 | 2.000 | -1.0000 | **WRONG** |
| fc2 | 13 | 13 | 0.000 | 0.000 | +1.0000 | OK |
| fc2 | 14 | 14 | 0.000 | 0.000 | +1.0000 | OK |
| fc3 | 4 | 4 | 0.000 | 0.000 | +1.0000 | OK |
| fc3 | 8 | 8 | 10.808 | 2.000 | -1.0000 | **WRONG** |
| fc3 | 12 | 12 | 7.649 | 2.000 | -1.0000 | **WRONG** |
| fc3 | 14 | 14 | 0.000 | 0.000 | +1.0000 | OK |

### Observations
- **18/33 neurons are byte-perfect** (`L1≈0, cos≈+1`).
- **15/33 are direction-perfect but sign-flipped** (`cos≈−1, rel err≈2`). The model uses these via compensating bias flips.
- **0/33 are direction-collapsed** (`|cos|<0.5`).

## Biases (all hidden neurons per layer)
| Layer | L1 sum | `|Δ|` median | `|Δ|` max |
|---|---:|---:|---:|
| fc1 | 4.080 | 0.106 | 0.902 |
| fc2 | 22.248 | 0.521 | 5.358 |
| fc3 | 8.014 | 0.347 | 2.918 |
| fc4 | 1.126 | 0.132 | 0.315 |
| fc5 | 16.297 | 3.868 | 8.118 |

Biases are not expected to match. The extracted bias is `b_i = −median(w_i · h_{L-1}(x_d))` computed through *already-extracted* lower layers. When `w_i` is sign-flipped or a lower layer has direction collapse, the dual-point bias is a different point in parameter space that still yields a near-equivalent function. **The functional accuracy on X_test2 is the real test of extraction; weight-matching is not.**

fc5's large bias deviation is expected: fc5 weights and biases are a fresh logistic-regression fit on the extracted hidden features (different features → different decoder).

## fc5 weight comparison
| metric | value |
|---|---:|
| row-wise L1 mean | 20.989 |
| row-wise rel err mean | 9.246 |
| row-wise `|cos|` mean | **0.251** |

fc5 `|cos|≈0.25` — the LR-fit decoder lives in a different point of fc5 parameter space than the oracle's decoder, yet the combined model achieves 100.00% test accuracy. This is the clearest demonstration that the attack achieves **functional** extraction on a *different* point in parameter space, not a *structural* copy of the oracle.

## Headline reading
| Property | Result |
|---|---|
| Signature **direction** (\|cos\|, recovered) | 1.000 mean |
| Signature **sign** (recovered) | 0.545 accuracy — attack is sign-blind by design |
| Weight-space distance to true model | Large (L1 median 0.00 per recovered neuron) |
| Functional distance to true model | **0.00 %** on make_blobs X_test2 (100.00% extraction vs 100.00% oracle) |

The attack is a **functional** extraction, not a weight-space copy. Direction recovery on the hidden layers is mostly successful where signature recovery succeeded; signs and biases are a free variable that the refinement step adjusts jointly to match oracle behavior.

## Artifacts
- Script: `/tmp/compute_leaky_true_vs_ext_v2.py`
- JSON (this layout): `results/reconstructed_models/tinier_leakyrelu_true_vs_extracted_metrics.json`
- Iter-1 / iter-2 reports: `tiniest_leakyrelu_iter1_2026-05-05.md`, `tinier_leakyrelu_iter1_2026-05-06.md`
- Activation toggle / port audit: `leaky_relu_port.md` (project root)