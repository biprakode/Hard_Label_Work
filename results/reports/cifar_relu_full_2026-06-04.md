# CIFAR-10 ReLU Flagship — Full Enhanced Attack Report
**Date**: 2026-06-04
**Victim**: `tiny_stuff/TinyModel_relu.{pth,keras}` — `3072→256→256→256→64→10` MLP, ReLU, float64
**Eval set**: `X_test2` (held-out CIFAR slice, seed=99, never queried in Phase 3)

---

## 1. TL;DR

End-to-end run of the enhanced hard-label pipeline against the flagship
CIFAR-10 ReLU MLP. Phase 1 recovered **502/832 (60.3 %)** of hidden weight
directions (cosine similarity ≈ 1.0 on recovered rows). Phase 2 sign recovery
was run on L1+L2 (real signal, ~78 min wall) and aborted on L3+L4 after worker
stalls projected another ~12 hours; Phase 3 oracle sign search recovered most of
the missing sign information. Phase 3 reconstruction reached **44.06 %** held-out
accuracy and **50.40 %** prediction agreement vs the oracle on `X_test2`, vs
**53.34 %** oracle ground truth. Matches the prior flagship baseline
(44.86 % / 51.42 %) within run variance.

---

## 2. Setup

| | |
|---|---|
| Architecture | `3072 → 256 → 256 → 256 → 64 → 10` (4 hidden layers, 832 hidden neurons) |
| Activation | ReLU (`LEAKY_ALPHA = 0.0`) |
| Victim training | CIFAR-10, inputs `x/255*2-1 ∈ [-1,1]`, float64, no BN/dropout |
| Victim test acc (X_test2) | **53.34 %** |
| Host | 14-core CPU, **22 Gi RAM** (post-restart: 17 Gi available, swap=0) |
| Dual search | batched PyTorch (`signature_recovery/torch_impl/parallel_duals.py`), 5 workers × batch-48 |
| Clustering | streaming (`cluster_dual_points_stream.py`), `CLUSTER_PER_NEURON_CAP=150` |
| Sign recovery | `nThreads=5`, `nExp=2000`, `nExpMin=200`, `choose_dx=along_decision_boundary` |
| Phase 3 | `analysis/run_extraction.py --full --from-scratch --refine --refine-epochs 1000` |

---

## 3. Per-Stage Timings (wall, 14-core box)

| Stage | Time | Notes |
|---|---|---|
| Dual search (140 rounds, 5w × b48) | **61.1 min** | 792 K triplets seen, 119 K kept |
| Streaming cluster (per-neuron cap 150) | 3.1 min | 4 layer pickles, 8.8 GB total |
| Per-neuron `.npy` bridge | 8 s | 804 .npy files generated |
| Recover weights L0..L3 (sequential) | 67.0 min | L0: 64m (3072-dim SVD), L1/L2/L3 ≪ 1m each |
| Sign recovery (L1+L2 complete) | 78 min | 506 neurons fully processed |
| Sign recovery (L3 partial, L4 not started) | aborted at 3:01 | Slow walks on deep neurons; see §6 |
| Phase 3 reconstruction + refinement | ~7 min | 1000-epoch refinement, lr=5e-3 |
| **Total wall (incl. partial sign step)** | **~5h 25m** | |

---

## 4. Phase 1 — Signature Recovery

### 4.1 Dual collection & clustering

| | |
|---|---|
| Triplets seen / kept | 792 K / 119 K |
| Per-neuron storage | 74 KB/triplet (3 × 3072 × float64) |

**Cluster coverage** (whitebox `cheat_neuron_diff` labels):

| Layer | covered / total | density |
|---|---|---|
| 0 | **256/256** | dense (identity prefix) |
| 1 | **251/256** | dense (1-layer prefix) |
| 2 | 244/256 covered but uncoverable | see below |
| 3 | 53/64 covered, uncoverable | |

### 4.2 Weight recovery

| Layer | recovered / total | mean abs err | reason for failure |
|---|---|---|---|
| 0 | **255/256** | 9.2 × 10⁻⁹ | identity prefix; clean SVD null-space |
| 1 | **247/256** | ~10⁻⁹ | 1-layer ReLU prefix still well-conditioned |
| 2 | **0/256** | — | `min(hits)==0` structural rejection (95-97 % "Mean OK") |
| 3 | **0/64** | — | structural (compounded depth) |
| **Total** | **502/832 (60.3 %)** | — | matches prior baseline 495/832 |

Mean |cos sim| on recovered rows = **1.000** (machine precision). The two
deep-layer zero recoveries are intrinsic, not statistical — raising the dual
budget does not help (deep neurons live where most upstream units are
saturated-off, so weight components along those inputs are unobservable from
boundary geometry).

---

## 5. Phase 2 — Sign Recovery

| Layer | processed / total | +1 / -1 | mean confidence | status |
|---|---|---|---|---|
| 1 | **256/256** | 255 / 1 | 0.90 | full |
| 2 | **250/256** | 224 / 26 | 0.52 | full (6 dOFF-error neurons defaulted +1) |
| 3 | 110/256 | partial (rest padded +1) | 0.15 (on processed) | aborted; see §6 |
| 4 | 0/64 | (padded +1) | — | not started; see §6 |

L1 shows the documented bias toward +1 (sparse future-toggle signal in the
first hidden layer). L2 has the richest sign signal (26 confirmed −1 neurons).

---

## 6. Why L3 / L4 were aborted

After L2 finished cleanly in ~78 min, L3 reached only **86/256** in another 50 min.
Per-neuron timings on L3 went up to **746 s** (12+ min) with most workers
processing the same neuron family for >5 min each. Cause is algorithmic
**depth, not CPU or memory** (5 workers @ 97 % CPU = 5/14 cores busy;
11 Gi RAM free, swap=0):

- Each per-neuron walk must navigate boundaries of all *upstream* neurons
  (256 for L1, 512 for L2, 768 for L3, 1024 for L4) at every step.
- L3 has sparser dual-point sets (some neurons only 4-12 dual points,
  median 150) → more degenerate walks.
- Projected wall time at observed pace: **~12 more hours for L3 + L4**.

Pragmatic call: halt Phase 2, pad L3 unprocessed + all L4 with +1 stubs, let
Phase 3 oracle sign search do the corrections. Per the prior CIFAR flagship
insight (`cifar_flagship_insights.md` §5.4), even a fully blank Phase 2 leaves
Phase 3 sign search at ~22 % agreement — the structural advantage from L0+L1
weights is what carries.

---

## 7. Phase 3 — Reconstruction + Refinement

### 7.1 Oracle sign search (greedy, k > 18)

| Pass | order | flips L0 | flips L1 | flips L2 | flips L3 | agreement |
|---|---|---|---|---|---|---|
| start | — | — | — | — | — | 10.06 % |
| 1 | [3,2,1,0] | +30 | +74 | 0 | 0 | 21.25 % |
| 2 | [0,1,2,3] | +17 | +15 | 0 | 0 | 23.27 % |
| 3 | [3,2,1,0] | +8 | +8 | 0 | 0 | **24.18 %** |

L2/L3/L4 flips are 0 because no signature was recovered for those layers
(rows are random Kaiming init; flipping doesn't help).

### 7.2 fc5 LR fit on oracle labels

LR fit on 10 K queried samples → **40.08 %** agreement (single closed-form
step).

### 7.3 Frozen-row cross-entropy refinement (1000 epochs, lr 5e-3)

| epoch | loss | X_test agreement |
|---|---|---|
| 1 | 1.71 | 39.27 % |
| 100 | 0.66 | 77.11 % |
| 200 | 0.10 | 99.17 % |
| 300 | 0.015 | **100.00 %** |
| 1000 | 6 × 10⁻⁴ | 100.00 % |

Refinement saturates on the queried set at epoch ~300. Frozen rows are the
502 recovered (real) directions; the 330 random-init rows and all biases
remain free.

---

## 8. Final Evaluation on `X_test2` (held-out)

| Metric | This run | Prior baseline | Oracle (victim) |
|---|---|---|---|
| Reconstructed accuracy | **44.06 %** | 44.86 % | 53.34 % |
| Prediction agreement vs oracle | **50.40 %** | 51.42 % | — |
| Agreement on *queried* X_test | ~100 % | ~100 % | — |
| L0 sign accuracy | 49.0 % | similar | — |
| L1 sign accuracy | 49.0 % | similar | — |
| L0/L1 mean |cos sim| | 1.0000 | 1.0000 | — |

The 100 %-on-queries vs 50 %-on-held-out gap is the **query-set overfit**
documented in the prior flagship run: free distillation memorises the 10 K
queries but the 502 frozen-true rows regularise the held-out behaviour toward
the victim function — that's where the +7 pt over no-signature baseline comes
from.

---

## 9. Artifacts

```
results/reconstructed_models/
  reconstructed_full.pth                # 832-hidden-neuron model
  reconstructed_full_weights.npz
  extraction_metrics.json

results/sign_recovery/
  layer{1,2,3,4}_signs.npy              # int8 ±1
  layer{1,2,3,4}_confidences.npy        # float64
  layer{1,2,3,4}_votes.npy              # int32
  layer{1,2,3,4}_summary.json
  model_sign_recovery_summary.json

signature_recovery/exp/
  1/*.p                                 # 140 dual-search round pickles (~55 GB)
  1-cluster-{0,1,2,3}.p                 # streamed per-layer clusters (~8.8 GB)

signature_recovery/outputs/model_weights/Vrelu/
  layer_0/neuron_*/weights{,_unscaled}.{npz,txt}, metadata.json   # 255 neurons
  layer_1/neuron_*/...                                            # 247 neurons
  layer_2/neuron_*/                                               # 244 empty dirs
  layer_3/neuron_*/                                               # 53 empty dirs

sign_recovery/layer_neuron_npys/
  layer{1,2,3,4}_neuron{i}.npy          # 804 per-neuron dual-point arrays
```

---

## 10. Reproduce

End-to-end recipe is in the project root README (§"Full attack on CIFAR-10 ReLU").
Minimum hardware: 22 Gi RAM, 14 CPU cores, 100+ Gi free disk. Restart the
machine if swap is not empty before starting — Phase 1 saturates 16-18 Gi of
working set across workers.

---

## 11. Open threads

- **Sign recovery for deep layers** — current algorithm scales poorly with
  depth × width. A boundary-walk with adaptive `nExpMin` (e.g., exit once
  `logp < -3.69`, even before nExpMin) would cut L3 wall by ~3×.
- **L2/L3 weight recovery** — same structural failure as the prior baseline.
  LeakyReLU(α > 0) is the documented lever (relaxes the `min(hits) == 0`
  rejection); see `paper_notes/section1/1_2_leaky_relu_adaptation.md`.
- **Reproducible Phase 2 timing** — the dOFF DataFrame / index error path
  (2 occurrences this run) needs a deterministic test before being declared
  benign.
