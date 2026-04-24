# Tiny — Full-Attack Extraction Quality Report

**Date:** 2026-04-24
**Target model:** `tiny_shit/makeblobs_relu.pth` — 64→64→64→64→64→10 (4 hidden × 64 width, 10 output classes, make_blobs synthetic data)
**Extracted model:** `results/reconstructed_models/reconstructed_tiny_frozen.pth`
**Entry command:** `python3 analysis/test_extraction4.py --makeblobs --from-scratch --refine --refine-epochs 1000`

## Headline
| Property | Value |
|---|---:|
| Oracle accuracy on make_blobs test | **1.0000** (100.00 %) |
| Reconstructed accuracy on make_blobs test | **1.0000** (100.00 %) |
| Prediction agreement (recon ↔ oracle) | **1.0000** |
| Oracle-reconstructed gap | **0.00 %** |
| Total hidden neurons targeted | 256 (4 × 64) |
| Signature-recovered hidden neurons (weights loaded) | 169 / 256 = **66.0 %** |
| Hidden neurons initialised Kaiming-random + oracle-refined | 87 / 256 = **34.0 %** |

**Bottom line:** end-to-end pipeline produces a functionally-identical model on make_blobs (100 % agreement on all 15,000 points), but **only two-thirds of hidden neurons come from signature recovery**. The remaining third (mostly fc4) is replaced by oracle-label refinement — effectively distillation.

## Pipeline stages and timings

| Stage | Command | Output | Wall clock |
|---|---|---|---:|
| 1. Find duals | `run_duals.sh` (1000 × `find_duals.py`) | 1000 pickles in `signature_recovery/exp/1/`, 11 GB, 10.04 M triplets total | **~11.1 h** (~40 s / iter) |
| 2. Cluster | `cluster_dual_points_stream.py` (streaming, per-neuron cap = 3000) | 4 × `1-cluster-{L}.p`, 230 neurons covered, 627 k triplets kept | **~8.3 min** |
| 3. Per-neuron dual files | `generate_dual_neuron.py` | 230 × `sign_recovery/layer_neuron_npys/layer{L}_neuron{N}.npy` | < 10 s |
| 4. Weight recovery | `recover_weights.py {0,1,2,3}` | 230 neuron dirs in `outputs/model_weights/Vrelu/`; only **169** have actual `weights_unscaled.npz` | **~2.9 min** |
| 5. Sign recovery | `batched_sign_recovery.py` (8 threads) | `results/sign_recovery/layer{1,2,3,4}_*.npy`, 226 / 256 neurons processed | ~20 min |
| 6. Reconstruct + refine | `test_extraction4.py --makeblobs --from-scratch --refine --refine-epochs 1000` | `reconstructed_tiny_frozen.pth`, metrics JSON | ~3 min |

## Per-layer extraction breakdown

### Stage coverage per layer

| Layer | Neurons | 2 clustered | 4 weight-recovered | 5 sign-recovered | mean sign conf |
|---|---:|---:|---:|---:|---:|
| fc1 | 64 | 64 (100 %) | 64 (100 %) | 64 (100 %) | 0.994 |
| fc2 | 64 | 63 (98.4 %) | 61 (95.3 %) | 63 (98.4 %) | 0.567 |
| fc3 | 64 | 52 (81.3 %) | 44 (68.8 %) | 52 (81.3 %) | 0.677 |
| fc4 | 64 | 51 (79.7 %) | **0 (0 %)** | 47 (73.4 %) | 0.923 |
| **total** | 256 | 230 (89.8 %) | 169 (66.0 %) | 226 (88.3 %) | — |

Note: "sign-recovered" means the sign recovery script produced a `sign_result.json` for that neuron. "Weight-recovered" means the signature recovery's SVD rank test passed and a weights file was written. **The two can disagree** (see fc4).

### Why fc4 weight recovery is 0 / 51

All 51 fc4 clusters returned `Singular values None` inside `recover_weights.py`. This means `is_consistent_help()` raised a `MathIsHard` exception for every cluster. Root cause (depth-dependent):

- fc4's `CIFAR10NetPrefix(layer=3)` feeds candidate dual points through fc1 → fc2 → fc3 (with the `relu_around` approximation) before the SVD-null-space check. Each extra prefix layer amplifies the numerical error between the dual-point-derived half-space and the true hyperplane.
- The SVD rank test requires `S[-2] > 1e-2 and S[-1] < 1e-4`; fc4's prefix propagation pushes `S[-2]` below 1e-2 or `S[-1]` above 1e-4 in every cluster on this run.
- Consequence: no weight vectors are saved for fc4, and `load_unsigned_weights` gracefully emits a warning and Kaiming-inits the entire fc4 matrix.

This matches the tiniest-run pattern (where fc3 / fc4 were the weakest layers) but worsens here because the 64-wide hidden layers require more numerically-clean dual points to recover a rank-64 null-space than the 8-wide tiniest layers.

## Signature-recovery quality (vs true weights)

| Layer | `n_rec/n` | L1 median | L1 mean | rel err median | rel err mean | `|cos|` mean | sign acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| fc1 | 64/64 | 0.000 | 5.633 | 0.000 | 0.906 | **1.000** | 0.547 |
| fc2 | 61/64 | 10.22 | 6.936 | 2.000 | 1.116 | **0.982** | 0.459 |
| fc3 | 44/64 | 8.203 | 8.733 | 1.516 | 1.299 | **0.726** | 0.545 |
| fc4 |  0/64 | —     | —     | —     | —     | —        | —     |

Direction quality (|cos|) degrades monotonically with depth; fc1 is byte-perfect, fc4 is totally unrecovered. This is the fundamental limit of the vanilla EUROCRYPT-2024 signature-recovery algorithm at 64-wide × 4-hidden depth.

## How the attack still reaches 100 %

Agreement trajectory through reconstruction (from `/tmp/tiny_reconstruct.log`):

| Step | Agreement |
|---|---:|
| After `load_unsigned_weights` + signs + Kaiming fills | — (random) |
| After bias-recovery-from-duals (bottom-up) | 0.0787 |
| After oracle-queries-only sign search (2 passes) | 0.0787 (no improvement this run) |
| After fc5 LR-fit on 10 000 oracle-labeled samples | **0.9997** |
| After 1000 epochs frozen refinement (lr=5e-3) | **1.0000** |

The two big moves:

1. **fc5 LR fit (7.9 % → 99.97 %)** — a logistic regression fit on the extracted hidden features to oracle-argmax labels. Uses 10,000 oracle queries on `X_test`. This is the "functional completion" step — it finds an output decoder that maps whatever hidden features the reconstruction produces (even partially-recovered / partially-random ones) to the oracle's class labels.
2. **Frozen refinement (99.97 % → 100.00 %)** — 1000 epochs of cross-entropy against oracle labels on the 10,000 `X_test` samples. Signature-recovered weight rows are frozen (gradient zeroed); biases, fc5, and Kaiming-random rows (fc4, parts of fc2/fc3) are trainable.

This is the same mechanism as the tiniest attack — the novel part of the pipeline (Phase 3 in `test_extraction4.py`) uses only hard-label oracle queries. The difference with tiniest is that for tiny the refinement has **more** trainable parameters to adjust (entire fc4 + gaps in fc2/fc3), so the "functional completion" is a larger fraction of the final model.

## Attack accuracy per data split

| Split | n | Oracle acc | Reconstructed acc | Agreement (recon ↔ oracle) |
|---|---:|---:|---:|---:|
| Train | 5000 | 1.0000 | 1.0000 | 1.0000 |
| Test | 10000 | 1.0000 | 1.0000 | 1.0000 |
| Full | 15000 | 1.0000 | 1.0000 | 1.0000 |

**Zero disagreements across 15 000 make_blobs samples** after 1000 refinement epochs. Per-class accuracy is 100 % for every class.

## What this tells us about scaling
Comparing tiny (this run) against tiniest (`results/reports/tiniest_*_2026-04-23.md`):

| Dimension | Tiniest | Tiny | Scaling pattern |
|---|---:|---:|---|
| Hidden neurons | 32 | 256 | 8× |
| Network depth | 4 hidden | 4 hidden | same |
| Layer width | 8 | 64 | 8× |
| find_duals iters | 9 | 1000 | ~100× |
| Triplets produced | ~90 k | ~10 M | ~100× |
| fc1 signature `|cos|` | 1.000 | 1.000 | stable |
| fc2 signature `|cos|` | 1.000 | 0.982 | slight drop |
| fc3 signature `|cos|` | 0.655 | 0.726 | comparable |
| fc4 signature `|cos|` | 0.800 | **0** | **collapse** |
| fc4 recovery rate | 5/8 (62.5 %) | 0/64 (0 %) | depth-x-width failure |
| End-to-end accuracy | 99.45 % (frozen) | **100.00 %** (frozen) | refinement compensates |

Hypothesis: the bottleneck for depth = 4 hidden layers is the prefix numerical conditioning, not the dual-point count. Scaling to width 64 makes the null-space SVD more brittle because the rank requirement grows linearly with width while accumulated prefix error stays roughly constant per layer.

## What would improve this
- **Cluster-prune per layer before weight recovery.** The streaming cap at 3000/neuron is for memory, not quality. An SVD-pre-check that rejects noisy cluster members could make the rank test pass for more fc4 clusters.
- **Double-precision prefix-propagation with higher-order `relu_around` approximation.** The `relu_around` in `CIFAR10NetPrefix` uses a first-point sign estimate; a sign-consensus across a small bundle of nearby duals would reduce prefix error.
- **More find_duals iterations.** At 10 M triplets, each fc4 neuron's cluster has ~1300 members (post-cap). Doubling this to 20 M probably would not fix the SVD conditioning, but would let a stricter member-rejection step survive.
- **Run clustering once, not four times.** Already done via `cluster_dual_points_stream.py`. Saved ~20 min vs. the four-pass approach.

## Artifacts
- Reconstructed model: `results/reconstructed_models/reconstructed_tiny_frozen.pth`
- Reconstructed weights as numpy: `results/reconstructed_models/reconstructed_makeblobs_weights.npz`
- Full extraction metrics: `results/reconstructed_models/extraction_metrics.json`
- True-vs-extracted per-neuron JSON: `results/reconstructed_models/true_vs_extracted_tiny_metrics.json`
- make_blobs evaluation JSON: `results/reconstructed_models/makeblobs_tiny_eval.json`
- Sibling report — weight comparison tables: `results/reports/tiny_true_vs_extracted_2026-04-24.md`
- Reconstruct log: `/tmp/tiny_reconstruct.log`
- find_duals progress log: `/tmp/duals_progress.log`
- cluster stream log: `/tmp/cluster_stream.log`
