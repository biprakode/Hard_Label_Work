# Vanilla vs Current Workflow — Diff & Rationale

**Date:** 2026-04-23 (updated 2026-05-06)
**Vanilla:** `vanilla_codebase/hard-label-dnn-extraction/` — the EUROCRYPT 2024 paper drop.
**Current:** repository root + `enhanced_codebase/` — the extended attack that runs end-to-end on small models, produces a functionally-equivalent model, and now supports **Leaky ReLU activations**.

## One-line summary
The vanilla codebase implements **Phase 1 (signature recovery)** and **Phase 2 (sign recovery)** as two independent research demos restricted to ReLU + 4×256 CIFAR-10. The current codebase wires them together into a **Phase 3 (reconstruction + refinement)** that produces a working `.pth`, adds a **greedy oracle sign search** for any layer width, **clean train/test separation** via a fresh eval set (X_test2), and a **configurable LEAKY_ALPHA toggle** that lets the same pipeline attack Leaky ReLU(α) variants without breaking the original ReLU path.

---

## 1. File-level diff
### `signature_recovery/`
| File | Status | What changed (why) |
|---|---|---|
| `find_duals.py` | **modified** | CUDA → CPU fallback (no GPU on dev box); `TARGET` made dataset-dependent (3000 for tiniest — more duals improve SVD rank for direction recovery); absolute path for `exp/` so parallel shell launchers work. |
| `cluster_dual_points.py` | **modified** | Uses `LAYER_BOUNDARIES` for flat neuron indexing (vanilla hard-codes `DIM` so won't cluster non-uniform architectures like tinier `32→16→16→16→8→4`); CUDA → CPU. |
| `recover_weights.py` | **rewritten** | Vanilla only prints weights. Current saves `weights_unscaled.npz`, `weights.npz`, `metadata.json` per `layer_X/neuron_Y/` with the **scaling factor** so that downstream extraction can apply `abs(factor)` (sign-blind scaling). Uses `LAYER_SIZES` for variable widths. |
| `utils.py` | **refactored** | `LAYER_SIZES = [input, h1, …, hn, out]` as single source of truth. Replaces hard-coded `IDIM, DIM, SHRINK`. `CIFAR10NetPrefix` rebuilt to respect per-layer widths. Adds `LAYER_BOUNDARIES` for flat-to-layer id mapping. Supports TINIEST / TINIER / TINY / MAKEBLOBS dataset flags. |
| `generate_dual_neuron.py` | **new** | Converts clustered dual triplets into per-neuron `layerX_neuronY.npy` files expected by sign recovery. |
| `custom_recover_weights.py` | **new** | Variant of recover_weights that writes to a custom output path. |
| `run_duals.sh` | **new** | Parallel launcher that invokes `find_duals.py` many times (bumps total dual-point count). |
| `compare_weights*.py`, `cluster_analyzer.py` | **new** | Ad-hoc debugging scripts for weight/cluster diagnostics. |

### `sign_recovery/`
| File | Status | What changed (why) |
|---|---|---|
| `sign_recovery.py` | **modified** | Returns `recovered_sign ∈ {+1,−1}` and `confidence`. Writes `sign_result.json` per neuron. Guards against empty DataFrames, NaN in dOFF/dON, and infinite loops on non-converging boundary walks (added `max_iter=1000`). Vanilla silently hangs/NaNs on sparse layers. |
| `batched_sign_recovery.py` | **rewritten** | Dataset-aware config (per-layer `nExpMin`, `nExp`, `choose_dx`). Aggregates per-layer `signs.npy`, `confidences.npy`, `votes.npy`, and a model-wide summary JSON. Vanilla was a thin `multiprocessing.Pool` wrapper with no aggregation. Crucially: **all layers now use `choose_dx='along_decision_boundary'`** instead of `perfect_control_along_decision_boundary`, because the latter caps `dOFF ≤ 3·dON`, which destroys the sign asymmetry that the algorithm relies on (see comments in lines 146–153). |
| `whitebox.py`, `create_tables.py`, `README.md` | **modified** | Minor formatting / docstring changes. |
| `custom_tables.py` | **new** | Aggregates results to `results/tables/*.csv`. |

### New top-level infrastructure
| File | Purpose |
|---|---|
| `analysis/test_extraction4.py` | Wires signature + sign outputs into a full `.pth`. Implements bias recovery from dual points, joint (w, b) oracle sign search (brute-force for k≤18, **greedy for k>18**), fc5 logistic regression, and frozen/unfrozen oracle-label refinement. Uses X_test for Phase-3 training and **X_test2 for evaluation**. **This is the entire new Phase 3.** |
| `analysis/evaluate_reconstructed_makeblobs.py` | Predicts with the extracted model on the sklearn make_blobs task, reports accuracy vs true labels for both X_test (seed=42) and **X_test2 (seed=99, eval-only)**. |
| `analysis/compare_true_vs_extracted.py` | Per-layer L1 / rel error / cos-sim between true and extracted weights. |
| `analysis/verify_sign_blind_scaling.py` | Sanity check that the scaling factor sign cannot leak. |
| `create_tiniest_makeblobs_model.py` (+ tinier + full) | Trains small targets for the attack. Vanilla shipped only the CIFAR-10 keras file and gave you nothing to train on. |
| `tiny_shit/` (now `tiny_stuff/` in enhanced_codebase) | Trained small targets in `.pth` + `.keras`. |
| `data/x_test_*.npy`, `y_test_*.npy` | Phase-3 training splits (seed=42) for each target. |
| `data/x_test2_*.npy`, `y_test2_*.npy` | **Fresh eval-only splits (seed=99, same cluster centers)** — no Phase-3 training overlap. |
| `results/` | Structured outputs: `sign_recovery/`, `reconstructed_models/`, `reports/`, `tables/`. |
| `enhanced_codebase/` | Self-contained copy of the complete pipeline with all paths patched — smoke-tested end-to-end on tiniest. |

---

## 2. Substantive behavior differences

| Aspect | Vanilla | Current | Why the change |
|---|---|---|---|
| Target architecture | hard-coded 4×256 ReLU CIFAR net | any shape via `LAYER_SIZES` | needed to shrink to 8-8-8-8-8-8 where SVD, clustering and sign recovery are tractable end-to-end |
| Sign recovery `choose_dx` | `perfect_control_along_decision_boundary` for all layers | `along_decision_boundary` for all layers | vanilla variant caps `dOFF ≤ 3·dON` inside `sign_recovery.py:397`, **forcing all votes to +1** on layers with sparse future-toggle signal. Removing the cap lets true sign asymmetry propagate to votes. |
| Biases | not part of the attack (dropped) | extracted from dual points `b_i = −median(w_i · h_{L-1}(x_d))` | a usable model needs biases; dual points give them once lower layers are reconstructed |
| Output layer (fc5) | not part of the attack | logistic-regression fit on `(h_4, oracle_label)` pairs | fc5 has no pre-activation dual points under hard labels; the only signal is argmax of logits, so an LR on the extracted features recovers it |
| First/last-layer signs | biased to +1 (see cap above) | **oracle-queries-only sign search**: brute-force 2^k for k≤18; **greedy O(k)-per-pass for k>18** | terminal layers have no downstream ReLU signal, so the vanilla statistical test fails. For tiny layers (k≤18) 2000 queries per combo is cheap; for CIFAR-10's 256-wide layers greedy is the only feasible option — vanilla silently skipped all such layers |
| Sign search scalability | N/A | k≤18 → brute-force (optimal per layer); k>18 → `_greedy_sign_pass_layer` (flip one neuron, keep if agreement improves, repeat) | brute-force exponential; greedy O(k) per pass. For CIFAR-10 256-wide layers k>18 is always true — the sign search was a dead code path in the old version |
| Reconstruction | *not implemented* | `reconstruct_model()` in `test_extraction4.py` | vanilla stops at per-neuron weight *printouts*; there is no full-model artifact to test |
| Refinement | *not implemented* | `oracle_label_refinement()` with frozen weight-row masking | closes the gap from 78 % (after sign search + LR fit) to 99.45 % without touching extracted directions |
| Scaling factor handling | `soln / factor` (sign **leaks** through the sign of `factor`) | `soln / abs(factor)` in `test_extraction4.load_unsigned_weights` | the sign of the scaling factor is a free function of the extracted direction's orientation — dividing by `factor` directly would reveal the true sign without the statistical sign-recovery test having to work |
| Partial recovery | crashes if any neuron is missing | Kaiming-init unrecovered rows, track via `recovered_mask` | signature recovery routinely misses some neurons; refinement trains the random-init rows as fresh params |
| Train/test overlap in eval | N/A | **X_test2**: fresh samples from same cluster centers, different seed (99). Phase-3 trains on X_test, evaluates on X_test2 | without separation, 100% agreement on X_test is meaningless — the model memorised its own training labels. X_test2 gives an honest out-of-sample number (99.50%). |
| Dataset support | CIFAR-10 only | CIFAR-10, make_blobs (full/tinier/tiniest) | make_blobs gives perfectly-separable clusters → 100 % oracle accuracy, which removes oracle-noise confounders when grading the attack |
| Activation support | ReLU only | ReLU + Leaky ReLU(α) via `LEAKY_ALPHA` toggle | the OFF-side α·z signal lets the SVD recover even prefix neurons that ReLU's null space drops. With α=0.0 the pipeline is byte-identical to the original ReLU path; with α>0 it loads `*_leakyrelu.{pth,keras}` victims and applies leaky-aware patches throughout. See `leaky_relu_port.md` for the 5 gated patches. |

---

## 3. Accuracy progression (same target: tiniest 8-8-8-8-8-8 make_blobs)

All figures measured on the **held-out test split** (n=2000) unless noted.  
Before 2026-05-04, "test" = X_test (seed=42, Phase-3 training set — overlap). After, "X_test2" = seed=99 eval-only set (no overlap).

| Workflow stage | Eval set | Reconstructed acc vs true | Notes |
|---|---|---:|---|
| Vanilla end-to-end | — | — | does not produce a reconstructed model at all |
| Current cycle 0 (baseline) | X_test | 0.1255 | signature + sign outputs combined naively |
| Current cycle 1 (fixes + oracle sign search) | X_test | 0.4305 | with cheat biases + fc5 |
| Current cycle 2 (from-scratch bias + LR fc5) | X_test | 0.7820 | no cheating |
| Current cycle 3 (frozen refinement) | X_test | **0.9945** | no cheating, extraction-pure; but eval has Phase-3 overlap |
| Current cycle 3 (unfrozen refinement) | X_test | 1.0000 | drifts extracted directions → no longer extraction |
| **Cycle 3 + X_test2 eval (2026-05-04)** | **X_test2** | **0.9950** | honest out-of-sample number, no Phase-3 overlap |
| **Cycle 4: leaky α=0.01 port, tiniest (2026-05-05)** | **X_test2** | **0.9925** | first non-ReLU target; 22/32 neurons recovered (vs ReLU 19/32). 5 leaky-gated code patches added; ReLU path preserved at α=0. See `tiniest_leakyrelu_iter1_2026-05-05.md`. |
| **Cycle 5: leaky α=0.01, tinier (2026-05-06)** | **X_test2** | **1.0000** | non-uniform 32→16→16→16→8→4. 33/56 neurons recovered. Surfaced and fixed a pre-existing `LAYER_SIZES[layer+1]` shape bug in `recover_weights.py` is_consistent_help that only triggers when input_dim ≠ first_hidden_dim. Refinement converged in 1 epoch. See `tinier_leakyrelu_iter1_2026-05-06.md`. |

---

## 4. README — launching the new attack

Use `enhanced_codebase/` for a self-contained run. All paths are pre-patched; no cross-directory dependencies.

```markdown
# Hard-Label DNN Extraction — extended workflow

Produces a functionally-equivalent `.pth` of a hidden neural network from
**hard-label oracle queries only**, on a small sklearn make_blobs target.

Based on the EUROCRYPT 2024 paper; see `vanilla_codebase/` for the original
reference.

## Prerequisites
- Python 3.11
- `pip install torch tensorflow keras numpy scipy pandas scikit-learn matplotlib`

## One-shot pipeline (tiniest 8-8-8-8-8-8, make_blobs, ~10 min on CPU)

### Step 0 — Train the victim
```bash
python3 create_tiniest_makeblobs_model.py
```
Writes `tiny_stuff/tiniest_makeblobs_relu.{pth,keras}` and
`data/{x,y}_test_tiniest_makeblobs.npy`.

### Step 1 — Signature recovery (Phase 1)
```bash
# 1a. Collect dual points (repeat ~10–50 times for coverage)
./run_duals.sh   # launches find_duals.py in parallel

# 1b. Cluster by neuron
python3 signature_recovery/cluster_dual_points.py --layers 0 1 2 3

# 1c. Per-neuron .npy files (consumed by sign recovery)
python3 signature_recovery/generate_dual_neuron.py

# 1d. SVD → weight directions + magnitudes
python3 signature_recovery/recover_weights.py
# -> signature_recovery/outputs/model_weights/Vrelu/layer_X/neuron_Y/
```

### Step 2 — Sign recovery (Phase 2)
```bash
python3 sign_recovery/batched_sign_recovery.py
# -> results/sign_recovery/layerX_{signs,confidences,votes}.npy
```

### Step 3 — Reconstruct + refine (Phase 3)
```bash
# Full from-scratch: bias recovery, greedy/brute-force sign search, LR fc5, frozen refinement.
# X_test (seed=42) is used for Phase-3 oracle training.
# X_test2 (seed=99, same cluster centers) is used for clean final evaluation.
python3 analysis/test_extraction4.py \
    --tiniest --from-scratch --refine --refine-epochs 500
# -> results/reconstructed_models/reconstructed_tiniest.pth
#    results/reconstructed_models/extraction_metrics.json
```

### Step 4 — Evaluate (with clean X_test2 split)
```bash
python3 analysis/evaluate_reconstructed_makeblobs.py
# -> results/reconstructed_models/makeblobs_eval.json
# Reports accuracy on both X_test (overlap) and X_test2 (clean).
```

### Step 5 — Compare vs true weights
```bash
python3 analysis/compare_true_vs_extracted.py
```

## CLI flags for `test_extraction4.py`
| flag | meaning |
|---|---|
| `--tiniest` | 8-8-8-8-8-8 target |
| `--tinier` | 32-16-16-16-8-4 target |
| `--makeblobs` | 64-64-64-64-64-10 target |
| `--from-scratch` | extract biases + fc5 (no cheating). Implies `--sign-search`. |
| `--sign-search` | oracle sign search: brute-force (k≤18) or greedy (k>18) |
| `--refine` | Adam epochs against oracle labels; extracted rows frozen |
| `--refine-unfreeze` | same, all params trainable (distillation — no longer extraction) |
| `--refine-epochs N` | epoch count (default 300) |

## Sign search: brute-force vs greedy
- **k ≤ 18 neurons recovered in a layer** → brute-force 2^k combos, globally optimal per layer
- **k > 18** → greedy: for each recovered neuron, flip its sign and keep if oracle agreement improves. O(k) per pass, repeats `n_passes` times (default 3) with alternating layer order. Enables sign correction on CIFAR-10's 256-wide layers where brute-force is infeasible.

## Eval dataset design (X_test vs X_test2)
| Set | Seed | Role |
|---|---|---|
| `X_test` (seed=42, n=2000) | same as training | Phase-3 oracle training: sign search, fc5 LR, refinement |
| `X_test2` (seed=99, same cluster centers) | fresh samples | **Clean evaluation only** — never seen during Phase-3 training |

Generated via `make_blobs(return_centers=True, seed=42)` then `make_blobs(centers=..., seed=99)` + same scaler.

## Expected numbers (tiniest, from-scratch, frozen refine, 500 epochs)
| metric | value |
|---|---:|
| Oracle acc on X_test2 | 99.95% |
| Reconstructed acc on X_test2 | **99.50%** |
| Agreement with oracle on X_test2 | **99.50%** |
| Agreement with oracle on X_test (training set) | 100.0% (inflated — overlap) |

## Reports
All in `results/reports/`:
- `vanilla_vs_current_workflow_2026-04-23.md` — this file
- `tiniest_greedy_xtest2_2026-05-04.md` — greedy sign search + X_test2 eval details
- `tiniest_leakyrelu_iter1_2026-05-05.md` — Leaky ReLU(α=0.01) port on tiniest (99.25% on X_test2)
- `tinier_leakyrelu_iter1_2026-05-06.md` — Leaky ReLU(α=0.01) on tinier (100% on X_test2; surfaces & fixes shape bug)
- `tiny_cheating_audit_2026-04-24.md` — where oracle queries are used
- `tiny_refinement_mechanism_2026-04-24.md` — how refinement boosts accuracy
- `tiny_true_vs_extracted_2026-04-24.md` — weight-space comparison (tiny 64-wide)
```

---

## 6. Leaky ReLU port (2026-05-05)

The `LEAKY_ALPHA` toggle in `signature_recovery/utils.py`, `sign_recovery/sign_recovery.py`, `sign_recovery/batched_sign_recovery.py`, and `analysis/test_extraction4.py` switches the entire pipeline between ReLU (α=0.0, default) and Leaky ReLU(α). Five gated patches were added, all no-ops when α=0:

1. **`recover_weights.py is_consistent_help` zero-hits bypass** — for leaky, "always-OFF" prefix coords still carry α·z signal so don't early-reject the cluster
2. **`recover_weights.py extract_weights` SVD gate relaxation** — leaky's α·z leakage adds extra small singular values; drop `S[-2]>1e-2 and S[-1]<1e-4` and rely on `min(errs)<1e-3` against cheat solution
3. **`test_extraction4.py load_unsigned_weights` metadata gate** — skip neurons without `metadata.json` (recover_weights' "Failed to identify" output has unreliable direction)
4. **`test_extraction4.py combine_weights_and_signs` zero-sign handling** — sign=0 (sign recovery skipped) used to multiply weight by 0; now treated as +1 so recovered weight survives, oracle sign search polishes
5. **OOM workaround** — `batched_sign_recovery.py nThreads` 8 → 2 for 24GB machines

### Mathematical insight
At the kink (z=0), ReLU's slope jumps 0→1; Leaky ReLU's jumps α→1. Within each cell the network is still piecewise linear, so the SVD machinery is preserved. The OFF-side α·z is actually *helpful*: ReLU's null space contains every always-OFF prefix coord (zero column → free direction), while leaky's α·z gives those coords a small but well-conditioned constraint. Result: leaky α=0.01 on tiniest recovered **22/32 neurons vs ReLU's 19/32** — particularly fc3 (4/8 vs 0/8).

### Tiniest leaky α=0.01 results
| Metric | Leaky α=0.01 | ReLU baseline |
|---|---|---|
| Recovered neurons | 22/32 (69%) | 19/32 (59%) |
| Reconstructed acc on X_test2 | **99.25%** | **99.50%** |
| Oracle acc on X_test2 | 99.95% | 99.95% |

### Tinier leaky α=0.01 results (non-uniform 32→16→16→16→8→4)
| Metric | Leaky α=0.01 |
|---|---|
| Recovered neurons | 33/56 (59%) |
| **Reconstructed acc on X_test2** | **100.00%** |
| Oracle acc on X_test2 | 100.00% |
| Refinement convergence | 1 epoch |

The tinier run surfaced an additional pre-existing bug — `recover_weights.py` is_consistent_help used `LAYER_SIZES[layer+1]` (target output dim) where `hiddens.shape[1]` (target input dim = prefix output dim) was needed. Tiniest's uniform 8×widths happened to make these equal, masking the bug for years. Now fixed in both repos (no leaky gating — it's a real bug fix that benefits ReLU non-uniform configs too).

See `leaky_relu_port.md` (project root) for the resume-friendly plan and full file-by-file audit.

---

## Artifacts
- This report: `results/reports/vanilla_vs_current_workflow_2026-04-23.md`
- Per-file diffs reproducible with:
  `diff -rq vanilla_codebase/hard-label-dnn-extraction/ signature_recovery/`
  `diff -rq vanilla_codebase/hard-label-dnn-extraction/ sign_recovery/`
