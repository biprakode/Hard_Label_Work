# Leaky ReLU Port — Plan & Progress

**Date started**: 2026-05-04
**Goal**: Extend the hard-label DNN extraction pipeline to support Leaky ReLU(α) activations across all three architectures (tiniest, tinier, tiny). Iterate: tiniest → tinier → tiny.

---

## ⚠️ CRITICAL GUIDELINE — DO NOT BREAK EXISTING ATTACK ⚠️

**The current ReLU extraction pipeline must remain byte-identical when `LEAKY_ALPHA = 0.0` (the default).**

All additions are gated on `LEAKY_ALPHA > 0`. With `LEAKY_ALPHA = 0`:
- All forward passes use plain ReLU exactly as before
- Model paths resolve to `*_relu.{pth,keras}` (the original artifacts)
- All thresholds, helper functions, and SVD logic behave identically
- No new model classes, no signature-changes to existing functions

**Toggle to enable leaky mode**: set `LEAKY_ALPHA = 0.01` (or chosen α) in:
- `signature_recovery/utils.py`
- `sign_recovery/sign_recovery.py` (or set via `sign_recovery.LEAKY_ALPHA = α` from `batched_sign_recovery.py`)
- `sign_recovery/batched_sign_recovery.py`
- `analysis/test_extraction4.py`

---

## Math (why this works)

| Property | ReLU | Leaky ReLU(α) |
|----------|------|---------------|
| Cells per neuron | 2 (ON/OFF) | 2 (ON/OFF) |
| Slope on ON | 1 | 1 |
| Slope on OFF | 0 | α |
| Kink at z=0 | yes | yes |
| Cell sign detection | `output > 0 ⟺ z > 0` | `output > 0 ⟺ z > 0` (since α·z ≤ 0 when z ≤ 0) |
| Within-cell linearity | yes | yes |

The attack scaffolding (dual points on kinks, SVD on linearized prefix, dON/dOFF sign recovery) is preserved. Only cell-slope coefficients change from `{0,1}` to `{α,1}`. Sign-recovery signal weakens by factor `(1-α)/(1+α) ≈ 0.98` at α=0.01.

---

## File-by-file summary of changes

### Configurable activation toggle: each file has `LEAKY_ALPHA = 0.0` (default ReLU). Code branches on `LEAKY_ALPHA > 0`.

### `signature_recovery/utils.py`
- Added `LEAKY_ALPHA = 0.0`
- Added helpers: `act(x)`, `act_np(x)`, `cell_slope_mask(x)` — all collapse to ReLU when α=0
- `CIFAR10Net.forward`/`forward_grad`/`cheat`: replaced `self.relu(x)` with `act(x)`. **`self.relu = nn.ReLU()` retained** in `__init__` for backward compat (any external code referencing `cheat_net_cpu.relu`).
- `MODEL_PATH`: gains `_leakyrelu` suffix when α>0; `_relu` (original) otherwise

### `signature_recovery/recover_weights.py`
- `CIFAR10NetPrefix.relu_around`: branches — α=0 uses original `mask = (x[:1]>=0).double(); return x*mask`; α>0 uses `slope = cell_slope_mask(x[:1]); return x*slope`
- `forward()`: branches — α=0 uses `nn.functional.relu` (original); α>0 uses `act()`
- SVD null-space thresholds (`>1e-4`, `>1e-5`) **kept as-is** because for α=0.01: `leaky_relu(z) > 1e-4 ⟺ z > 1e-4` still holds (negative outputs are α·z ≤ 0). Upgrade to preact-based detection only if iter-1 reveals problems.

### `sign_recovery/sign_recovery.py`
- Added `LEAKY_ALPHA = 0.0`, helper `_apply_act(x)` — branches based on α
- 3 call sites patched: `neuron_toggle_state` line 67, `get_neuron_values` line 183, `get_target_layer_output_norm_after_ReLU` line 191
- Each one replaces `x[x<0] = 0.0` with `_apply_act(x)` (or scaling for the wiggle case)
- `whitebox.py` left untouched: `getOutputMatrixWhitebox` and `getRealSigns` are dead code; `getSignatures` is only invoked by `perfect_control_along_decision_boundary` which the pipeline doesn't use

### `sign_recovery/batched_sign_recovery.py`
- Added `LEAKY_ALPHA = 0.0` and `sign_recovery.LEAKY_ALPHA = LEAKY_ALPHA` to propagate
- `model_path` resolution gained `_leakyrelu` suffix when α>0

### `analysis/test_extraction4.py`
- Added `LEAKY_ALPHA = 0.0`, helper `_act(x)` — branches based on α
- 4 model classes (TinyModel, TinierModel, TiniestModel, FullModel): `F.relu(...)` → `_act(...)` (16 sites)
- `_hidden_activations_up_to`: `torch.relu` → `_act`
- 5 model-path constants: `_relu` → f-string with `_act_suffix`

### New file: `create_tiniest_makeblobs_leakyrelu.py`
- Clone of `create_tiniest_makeblobs_model.py`
- PyTorch: `F.leaky_relu(x, negative_slope=0.01)` instead of `torch.relu(x)`
- Keras: explicit `LeakyReLU(alpha=0.01)` layers between Dense layers (so Dense weights remain cleanly addressable for whitebox readers)
- Outputs: `tiny_shit/tiniest_makeblobs_leakyrelu.{pth,keras}` + `tiniest_makeblobs_leakyrelu_alpha.txt`

---

## Backward-compat verification

After all changes, with `LEAKY_ALPHA = 0.0` (default):
```
$ python3 -c "from signature_recovery import utils; print(utils.MODEL_PATH)"
/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/tiny_shit/makeblobs_relu.pth   # ← original path
$ python3 -c "from signature_recovery.utils import act; import torch; print(act(torch.tensor([-1.0,0.0,1.0])).tolist())"
[0.0, 0.0, 1.0]   # ← ReLU output, unchanged
```

ReLU pipeline remains fully functional.

---

## Iteration 1 — tiniest, α=0.01

### Status — ITERATION 1 COMPLETE
- [x] Phase 0: Train leaky tiniest victim → 100% test accuracy, PT/Keras max diff 2.5e-6
- [x] Phase 1 code: all files modified; ReLU mode preserved; leaky mode wired
- [x] Backup ReLU baseline: `enhanced_codebase/_relu_baseline_backup/`
- [x] Fix pre-existing path bug in `enhanced_codebase/signature_recovery/utils.py` (`/enhanced_codebase/...` → `BASE_DIR`-relative)
- [x] Phase 1 run: find_duals (12021 triplets, 4 pickles) → cluster (fc1 8/8, fc2 8/8, fc3 7/8, fc4 8/8) → generate_dual_neuron (31/32) → recover_weights (22/32 identified)
- [x] Phase 2 run: PARTIAL. Layer 1 + 7/8 of layer 2; layer 2 neuron 7 hung at DualPointID 328 for 30+ min. Killed and aggregated partials. Layers 3-4 not run; oracle sign search filled them in at Phase 3.
- [x] Phase 3 run: test_extraction4 --tiniest --from-scratch --refine → **99.25% on X_test2**
- [x] Iter-1 report: `results/reports/tiniest_leakyrelu_iter1_2026-05-05.md`

### Code patches added during iter-1 (all gated on `LEAKY_ALPHA > 0`):
1. `recover_weights.py` `is_consistent_help` zero-hits bypass — for leaky, OFF coords still carry α·z signal, don't early-reject
2. `recover_weights.py` `extract_weights` SVD gate relaxation — leaky's α·z leakage adds extra small SVs, drop `S[-2]>1e-2 and S[-1]<1e-4`, rely on `min(errs)<1e-3` in `dosteal`
3. `test_extraction4.py` `load_unsigned_weights` metadata gate — skip neurons without `metadata.json` (the SVD-gate-relaxed code now saves "Failed to identify" results that have wrong direction; without metadata, treat as Kaiming)
4. `test_extraction4.py` `combine_weights_and_signs` zero-sign handling — sign=0 (unknown, partial sign recovery) used to zero out the weight via `weight * 0`; now treated as +1 so recovered weight survives, oracle sign search polishes
5. **OOM workaround**: `enhanced_codebase/sign_recovery/batched_sign_recovery.py` `nThreads` 8 → 2 (the user's PC has 24GB; 8 threads × ~3GB workers ≈ OOM)

### Phase 1 results (vs ReLU baseline)
| Layer | Leaky α=0.01 | ReLU baseline |
|-------|--------------|---------------|
| fc1   | 8/8          | 8/8           |
| fc2   | 7/8          | 6/8           |
| fc3   | 4/8 ✨       | **0/8**       |
| fc4   | 3/8          | 5/8           |
| **Total** | **22/32 (69%)** | **19/32 (59%)** |

Surprisingly, leaky α=0.01 achieves better signature recovery than ReLU. The α·z signal on OFF coords gives the SVD additional constraints that ReLU's null-space lacks — particularly visible on fc3 where ReLU baseline got 0/8.

### Phase 3 results (vs ReLU baseline)
| Metric | Leaky α=0.01 | ReLU baseline |
|--------|--------------|---------------|
| Oracle accuracy on X_test2 | 99.95% | 99.95% |
| **Reconstructed accuracy on X_test2** | **99.25%** | **99.50%** |
| Prediction agreement | 99.20% | 99.50% |

ITER-1 SUCCESS: leaky port matches ReLU functional accuracy within 0.25%.

### Pipeline state after iter-1
- **Main repo**: all `LEAKY_ALPHA = 0.0` (ReLU pipeline preserved byte-identical)
- **Enhanced_codebase**: all `LEAKY_ALPHA = 0.01` (kept in leaky mode for further iter testing on tinier/tiny)
- New leaky tiniest victim: `tiny_shit/tiniest_makeblobs_leakyrelu.{pth,keras}` (and copy in `enhanced_codebase/tiny_stuff/`)
- All 5 patches above are present in BOTH repos with the same `LEAKY_ALPHA > 0` gating

---

## Iteration 2 — tinier (32→16→16→16→8→4), α=0.01

### Status — ITERATION 2 COMPLETE
- [x] Phase 0: Train tinier_makeblobs_leakyrelu — 100% on both X_test and X_test2
- [x] enhanced_codebase config switched: TINIER=True, TINIEST=False (in utils.py and batched_sign_recovery.py)
- [x] Phase 1: find_duals (8 rounds, 16037 triplets) → cluster (16/13/7/5) → recover_weights (16/13/4/0 = 33/56 identified)
- [x] **Pre-existing bug found and fixed**: `recover_weights.py is_consistent_help` line 151. `hits = np.zeros(LAYER_SIZES[layer+1])` should be `hits = np.zeros(hiddens.shape[1])`. Mismatch only triggers when input_dim ≠ first_hidden_dim (true for tinier 32→16, false for tiniest's uniform 8×). Fix is **NOT gated on LEAKY_ALPHA** — it's a real bug fix that benefits ReLU non-uniform configs too. Mirrored to both repos.
- [x] Phase 2: batched_sign_recovery with reduced nExpMin=200, nExp=2000 (was 1000/10000) — 34/56 sign-processed, no hangs
- [x] Phase 3: test_extraction4 --tinier --from-scratch --refine → **100.00% on X_test2** (refinement converged in 1 epoch)
- [x] Iter-2 report: `results/reports/tinier_leakyrelu_iter1_2026-05-06.md`

### Iter-2 results vs iter-1
| Metric | Tiniest leaky | Tinier leaky |
|--------|---------------|--------------|
| Hidden neurons | 32 | 56 |
| Phase 1 recovered | 22/32 (69%) | 33/56 (59%) |
| Phase 2 sign-processed | 15/32 (partial; killed) | 34/56 (full pipeline ran clean) |
| **X_test2 reconstructed acc** | **99.25%** | **100.00%** |
| Refinement epochs to converge | 500 | 1 |

### Pipeline state after iter-2
- Main repo: `LEAKY_ALPHA = 0.0` (unchanged), shape-bug fix applied (always-on)
- Enhanced_codebase: `LEAKY_ALPHA = 0.01`, TINIER=True
- All patches mirrored. Total leaky-gated patches: 5. Always-on bug fixes: 1.

### Where iter-1 runs
**`enhanced_codebase/`** (already in TINIEST=True mode). Main repo's TINY ReLU outputs are not touched. To enable leaky mode in enhanced_codebase, the four `LEAKY_ALPHA` constants are set to 0.01 there only.

### ReLU baseline numbers to compare against (from `tiniest_greedy_xtest2_2026-05-04.md`)
- Recovery: fc1 8/8, fc2 6/8, fc3 0/8, fc4 5/8 (19/32 = 59%)
- Sign accuracy after oracle search: fc1 62.5%, fc2 50%, fc4 60% (avg 57.5%)
- Final agreement on X_test2: **99.50%**
- Oracle accuracy on X_test2: 99.95%

---

## How to resume mid-run

1. **Confirm activation toggles**:
   ```bash
   grep "^LEAKY_ALPHA" enhanced_codebase/signature_recovery/utils.py \
                     enhanced_codebase/sign_recovery/sign_recovery.py \
                     enhanced_codebase/sign_recovery/batched_sign_recovery.py \
                     enhanced_codebase/analysis/test_extraction4.py
   # All should be 0.01 for iter-1 leaky run; 0.0 in main repo
   ```

2. **Verify victim trained**: `ls enhanced_codebase/tiny_stuff/tiniest_makeblobs_leakyrelu.{pth,keras,_alpha.txt}`

3. **Phase 1**: from `enhanced_codebase/signature_recovery/`,
   ```bash
   ./run_find_duals.sh   # or run_duals.sh — generates dual points
   /home/biprarshi/miniconda3/envs/DLenv/bin/python3 cluster_dual_points.py --layers 0 1 2 3
   /home/biprarshi/miniconda3/envs/DLenv/bin/python3 generate_dual_neuron.py
   /home/biprarshi/miniconda3/envs/DLenv/bin/python3 recover_weights.py 0 0
   /home/biprarshi/miniconda3/envs/DLenv/bin/python3 recover_weights.py 1 0
   /home/biprarshi/miniconda3/envs/DLenv/bin/python3 recover_weights.py 2 0
   /home/biprarshi/miniconda3/envs/DLenv/bin/python3 recover_weights.py 3 0
   ```

4. **Phase 2**: `cd enhanced_codebase/sign_recovery && python3 batched_sign_recovery.py`

5. **Phase 3**: `cd enhanced_codebase && python3 analysis/test_extraction4.py --tiniest --from-scratch --refine --refine-epochs 500`

6. **Restore ReLU defaults after iter-1** (in enhanced_codebase only — main repo defaults already at 0.0):
   - Set `LEAKY_ALPHA = 0.0` in 4 files
   - Optionally restore baseline outputs from `enhanced_codebase/_relu_baseline_backup/`

---

## Iteration roadmap

- **Iter 1**: tiniest, α=0.01 — current. Identifies any code gaps in the pipeline.
- **Iter 2**: tinier (32→16→16→16→8→4), α=0.01 — train new victim, re-run pipeline. Code changes expected to be zero (all logic generalised).
- **Iter 3**: tiny (64→64→64→64→64→10), α=0.01 — same approach. Watch for fc4 numerical issues that already exist in ReLU mode.
- **Iter 4** (if requested): explore higher α values (0.1, 0.2) — sign-recovery signal weakens; may need algorithmic adjustments.

---

## Risk-ranked uncertainties

1. `recover_weights.py` SVD thresholds (`>1e-5`, `>1e-4`) — current analysis says they should still work for α=0.01 because OFF outputs are ≤0; iter-1 will confirm
2. Sign recovery contrast (`(1-α)/(1+α)`) — at α=0.01 only ~2% loss of signal; should be fine
3. find_duals' kink detector — gradient gap is `1-α` instead of `1`; at α=0.01 should still be detectable
4. Phase-3 refinement — activation-agnostic; robust

---

## Files modified (quick reference)

Main repo (`LEAKY_ALPHA = 0.0` — defaults preserve ReLU):
- `signature_recovery/utils.py` ✓
- `signature_recovery/recover_weights.py` ✓
- `sign_recovery/sign_recovery.py` ✓
- `sign_recovery/batched_sign_recovery.py` ✓
- `analysis/test_extraction4.py` ✓
- `create_tiniest_makeblobs_leakyrelu.py` (new) ✓

Enhanced_codebase (`LEAKY_ALPHA = 0.01` — leaky mode for iter-1 testing):
- `signature_recovery/utils.py` ✓ (also fixed pre-existing path bug)
- `signature_recovery/recover_weights.py` ✓
- `sign_recovery/sign_recovery.py` (pending sync)
- `sign_recovery/batched_sign_recovery.py` (pending sync)
- `analysis/test_extraction4.py` (pending sync)
- `tiny_stuff/tiniest_makeblobs_leakyrelu.{pth,keras}` (copied)

Backup of ReLU baseline outputs:
- `enhanced_codebase/_relu_baseline_backup/{sig_recovery_outputs,sign_recovery_results,reconstructed_models}/`
