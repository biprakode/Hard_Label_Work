# Cheat Removal — Results

**Aim:** make the extraction attack treat the victim as a pure black box — the
only victim signal allowed is the **argmax hard label**. Tested on tiniest
(8-8-8-8-8-8), ReLU and LeakyReLU(0.01). All code in `cheat_remove/`.

**Headline:** the attack runs **end-to-end with the victim as a pure black box**
(every victim touch is `argmax`) and reconstructs a model at **99.65 % prediction
agreement on X_test2** for both ReLU and Leaky — matching the documented
cheat-based tiniest result (98.95 %). The dominant whitebox cheats are removed
and replaced with validated argmax-only primitives. One genuinely hard piece —
**clean black-box layer separation** — is characterised honestly below: it is the
crux the original `cheat_neuron_diff` shortcut hides, and it does not cleanly
solve on this make_blobs manifold.

## What is now black-box (the cheats removed)

| Cheat (audit) | Replacement | Module | Validated |
|---|---|---|---|
| `gapt`/`gap` boundary detection + autograd normal | argmax bisection + finite-difference normal | `bb_core.py` | ✅ duals land on single-neuron crossings (0 multi-flips) |
| single-walk find_duals | **batched** argmax walk, lane-compacted, plugged into the torch `parallel_duals` harness (`--impl blackbox`) | `bb_find_duals.py` | ✅ identical triplet format; runs W-way parallel |
| `transfer_weights(cheat_net_cpu)` true prefix (layer 0) | identity prefix (no truth) | `bb_recover.py` | ✅ |
| `cheat_solution` scaling + neuron match | gauge `‖w‖=1` + arbitrary ids | `bb_recover.py` | ✅ |
| `cheat_neuron_diff` clustering (within a layer) | SVD-consistency seed-and-grow | `bb_recover.py` | ✅ same/diff neuron separate by ~3 orders (2.7e-6 vs 7.4e-2) |
| Phase-2 `whitebox.*` sign reads | dropped; signs via hard-label Phase-3 `oracle_sign_search` | (Phase 3) | ✅ hard-label |

## Validation gates (cheat used ONLY as a grader, never fed into the attack)

1. **Black-box dual points** — one walk: 5 714 duals, argmax-only; of single-flip
   triplets, **0 multi-neuron crossings**. Batched finder: 0 multi-flips, same format. ✅
2. **Black-box SVD weight recovery** — on a *pure* layer-0 cluster (formed by the
   grader), recovery from argmax-only normals + identity prefix gives
   **|cos| = 1.000** (2–3 triplets already saturate; 1 triplet is noisy). ✅
3. **Consistency clustering separates neurons** — normalised smallest singular
   value: same-neuron ≈ **2.7e-6**, different-neuron ≈ **7.4e-2** (clean 1e-4
   threshold). ✅ *within a layer.*
4. **End-to-end black-box functional agreement** — see table. ✅

## End-to-end black-box results (X_test2, seed=99, eval-only)

| Victim | Pipeline | Prediction agreement | Reconstructed acc | Oracle acc |
|---|---|---|---|---|
| tiniest **ReLU** | bb duals + bb layer-0 sig + hard-label Phase 3 | **99.65 %** | 99.70 % | 99.95 % |
| tiniest ReLU | *baseline:* no signature, hard-label Phase 3 only | 99.65 % | 99.70 % | — |
| tiniest **Leaky(0.01)** | bb duals + bb layer-0 sig + hard-label Phase 3 | **99.65 %** | 99.70 % | 99.95 % |
| tiniest Leaky | *baseline:* no signature, hard-label Phase 3 only | 99.65 % | 99.70 % | — |

Oracle-query budget (stage 1): ≈ **20 M** argmax queries for ReLU (≈15 M Leaky),
almost all spent on the finite-difference boundary **normals** (IDIM serial
boundary searches per point — the price of not reading the autograd gradient).
Phase 3 adds the same 3 distinct argmax invocations on X_test as the cheat
pipeline. Every query is `argmax(victim(x))`.

## The honest finding: black-box *layer separation* is the hard core

The consistency test cleanly groups duals that share a hyperplane **in input
space**, but it cannot tell a **layer-0** neuron from a **deeper** neuron, because
on this make_blobs manifold a deeper neuron is often **globally linear over the
data region** (its upstream neurons keep a fixed activation pattern there). Direct
measurements on a tiniest-ReLU run:

- A true layer-1 neuron produced a 508-dual cluster (506/508 pure) that looked
  **globally coplanar in input space** — 509 global projection-inliers, *more*
  than real layer-0 neurons.
- Neither spatial spread nor full-cluster SVD ratio separated layer-0 from this
  deeper neuron (L0 spreads 0.2–8.3 overlap L1's 0.5; SVD ratios overlap too).

So "take the 8 biggest globally-consistent input-space directions as layer 0"
mislabels deep-but-locally-linear neurons (only 1/8 matched a true layer-0
neuron at |cos|>0.95 on the ReLU run). This is **exactly why the original code
cheats with `cheat_neuron_diff`** — which reads the true hidden-activation sign
flip and therefore *knows* the layer/neuron for free. Removing it surfaces the
genuine difficulty that motivates the hard-label paper.

Peeling cannot bootstrap past this without first cleanly isolating layer 0
(ReLU sign is not a free gauge, so the layer-1 prefix needs *correct, signed*
layer-0 weights). On this manifold that clean isolation is the open problem.

## What this means for the threat model

- **Functional extraction is achievable fully black-box.** With the victim as a
  pure argmax oracle, hard-label Phase 3 (geometric bias recovery → oracle sign
  search → fc5 LR fit → frozen-row refinement) reconstructs a model at ~99.65 %
  agreement on this task. The black-box dual finder + black-box weight recovery
  are validated and feed it cleanly.
- **Structural (cryptanalytic) black-box extraction is bounded by layer
  separation.** On easy, low-dimensional make_blobs the functional gap is closed
  by Phase-3 distillation regardless of structural recovery, so signature
  recovery does not change the functional number here (baseline = with-signature
  = 99.65 %). On a harder task (e.g. CIFAR-10) where distillation alone is weak,
  structural recovery would matter — and there the layer-separation problem is
  what a black-box attack must solve.

## Files

| File | Role | Status |
|---|---|---|
| `bb_core.py` | Oracle (argmax-only) + boundary/normal/dual primitives | ✅ |
| `bb_find_duals.py` | batched black-box dual finder (torch-harness compatible) | ✅ |
| `bb_recover.py` | consistency clustering + SVD weight recovery, peel-aware prefix | ✅ (within-layer) |
| `bb_pipeline.py` | end-to-end: bb duals → bb layer-0 sig → Phase-3 format | ✅ |
| `signature_recovery/torch_impl/parallel_duals.py` | `--impl blackbox` runs `bb_find_duals` W-way parallel | ✅ |

## Reproduce

```bash
# (utils.py LEAKY_ALPHA = 0.0 for ReLU, 0.01 for Leaky; TINIEST=True)
cd cheat_remove
python3 bb_pipeline.py --target 3000          # stage 1: argmax-only signature -> Vrelu/layer_0
cd ..
python3 analysis/run_extraction.py --tiniest --from-scratch --refine --refine-epochs 1000
# stage 2: hard-label Phase 3 -> reconstructed model + agreement on X_test2
```

## Honest limitations / future work

- **Clean black-box layer separation** (and therefore full peeling to layers
  1–3) is unsolved here. A real black-box solution likely needs to probe
  *off the data manifold* to observe deeper neurons bending across layer-0
  hyperplanes, or a global piecewise-linear fit — both beyond this iteration.
- **`bb_sign.py` / `bb_bias.py`** (per-layer local sign + bias during peeling)
  were designed (see CHEAT_REMOVE_APPROACH.md) but are only needed once layer
  separation works; layer 0 needs no biases and Phase-3 supplies the rest.
- The finite-difference normal is query-heavy; a batched multi-point normal
  estimator would cut the ~20 M-query budget substantially.
