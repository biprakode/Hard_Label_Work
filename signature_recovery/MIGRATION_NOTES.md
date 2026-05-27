# Signature Recovery — NumPy → PyTorch Migration Notes (Phase A)

**Goal:** pure code migration. Same inputs, same outputs (format/shape/dtype/range — *not*
bit-identical values, since the search is randomised), faster wall clock. **No algorithm changes.**

**Environment:** `torch 2.9.0+cpu`, **no CUDA**, 14 CPU cores. Python
`/home/biprarshi/miniconda3/envs/MLenv/bin/python3` (only env with torch).
The torch code must run on `device='cpu'`; `device='cuda'` should work by changing one arg but is untested here.

**Config used for this profile:** `TINIEST=True`, `MAKEBLOBS=True`, `LEAKY_ALPHA=0.0` (tiniest ReLU,
8-8-8-8-8-8). Set in `signature_recovery/utils.py`. This matches the documented ReLU baseline
`paper_notes/section3/...` / `results/reports/tiniest_greedy_xtest2_2026-05-04.md`.
**NOTE:** I changed the toggle from the prior `TINY=True / LEAKY_ALPHA=0.01` to profile tiniest;
left at tiniest-ReLU because that is the Phase-C test target. Restore before any tiny run.

---

## DATAFLOW: Signature Recovery Pipeline

The shell driver is `run_one_model.sh <arch> <activation>` (STEP 2–5 are Phase 1). It runs
find_duals.py **N times in a sequential shell loop** (`for i in $(seq 1 DUAL_ITERS)`), then one
cluster pass, one generate pass, and recover_weights once per hidden layer.

### Step 1: `find_duals.py`  ← **THE BOTTLENECK (>90 % of wall time)**

- **Entry point:** module-level `main()` call at import time (line 240). No `if __name__`.
  Invoked as `python3 find_duals.py` with **no args** → `SEED=1` (utils.py line 84:
  `SEED = 1 if len(sys.argv) < 3 else int(sys.argv[1])`).
- **Input:** the oracle model (`utils.cheat_net_cpu` / `cheat_net_cuda`, loaded from
  `MODEL_PATH`), `LAYER_SIZES`/`IDIM` from utils. No file input.
- **Output:** `signature_recovery/exp/{SEED}/duals_{rand08d}.p` — one pickle per process run.
  Filename suffix is `random.randint(0,1000000)` after `random.seed(None)`.
- **Output format:** `list[ (left, middle, right) ]` where each element is an `np.ndarray`,
  **shape `(IDIM,)`, dtype `float64`**. `middle` is the recovered dual/critical point;
  `left`/`right` bracket it along the walk (`left = boundary + step_dir·midstep/2` of pair i,
  `right = boundary + step_dir·midstep/2` of pair i+1). Built in `main()` lines 224–231 by zipping
  consecutive entries of `find_dual_points()`'s `middle_points` (each a 2-tuple
  `(half_step_pt, dual_pt)`).
- **Bottleneck:** `find_dual_points()` (lines 52–207) walks **one** decision-boundary path,
  sequentially. Per dual point it issues *hundreds* of single-sample oracle/grad calls:
  - `find_decision_boundary()` (utils 286): ~50 iters of `bmodel(mid)` bisection.
  - `get_normal()` (utils 483): one autograd forward+`backward()` (`USE_GRADIENT=True`).
  - upper-bound sweep (line 111): `for step_size in 10**np.arange(-5,5,.1)` = 100 steps, each
    `is_on_decision_boundary()` → `gapt()` forward.
  - binary search (line 146): ~30 iters of `is_on_decision_boundary()`.
  - `refine_to_decision_boundary_cheat()` (Newton, ~10 iters of `gapt`).
  `main()` repeats whole paths until `len(all_points) >= TARGET` (TINIEST=3000, TINIER=2000, else 10000).
- **NumPy/torch calls:** `np.random.normal`, `np.dot`, `np.sum`, `norm()`; oracle via
  `bmodel`/`model`; gradient via `gapt(x,grad=True).backward()`. All **single-sample**.
- **Parallelisable:** **YES, two independent axes.**
  1. *Across paths/iterations* — every `find_dual_points()` call (and every `find_duals.py`
     process) is fully independent and writes its own pickle → embarrassingly parallel.
     Currently serialised by the shell loop. Multiprocessing alone ≈ 14× on this box.
  2. *Across B simultaneous walks* — run B independent walks in lockstep, replacing each
     single-sample `bmodel(x)`/`gapt(x)` with a batched `bmodel(X_batch)`/`gapt(X_batch)`.
     One batched forward replaces B sequential forwards (the real GPU-style win; also helps CPU
     by amortising Python/dispatch overhead). Walks that converge early are masked out.

### Step 2: `cluster_dual_points_stream.py`

- **Entry point:** `if __name__ == '__main__': stream_cluster_all()`. No args.
  (`cluster_dual_points.py` is the older non-streaming variant; takes `layer` argv,
  optional `slow`. The pipeline uses the **stream** version.)
- **Input:** all pickles in `exp/1/` (streamed one file at a time to bound RAM).
- **Output:** `exp/1-cluster-{0..n_hidden-1}.p`, one per hidden layer.
- **Output format:** `dict[flat_neuron_idx:int] -> list[(left,middle,right)]` (same triplet type
  as Step 1). Per-neuron cap `PER_NEURON_CAP=3000`.
- **Bottleneck:** per-triplet `cheat_neuron_diff_cuda(left, right)` (utils 255) — stacks the two
  points, runs `cheat_cuda` (a forward returning padded pre-activations), compares sign patterns,
  keeps triplets that flip **exactly one** neuron. Python for-loop over every triplet.
- **NumPy/torch calls:** `cheat_neuron_diff_cuda` does `torch.tensor(np.stack([a,b]))` + a forward
  + `torch.where`. One call **per triplet** — the round-trip + tiny forward per triplet is the cost.
- **Parallelisable:** **YES.** `cheat_neuron_diff` is a pure function of (left,right). Batch K
  triplets → one `cheat()` forward of shape `(2K, IDIM)` → vectorised sign-flip count + routing.
  Eliminates the per-triplet CPU↔tensor round-trip. Profiled cost is already small (2 s) so this
  is a secondary target.

### Step 3: `generate_dual_neuron.py`

- **Entry point:** top-level script body (no `main`). No args.
- **Input:** `exp/1-cluster-{0..4}.p` (hard-coded list; missing files skipped).
- **Output:** `sign_recovery/layer_neuron_npys/layer{L+1}_neuron{local}.npy`.
- **Output format:** `np.ndarray` shape `(n_duals, IDIM)` float64 — **the middle points only**
  (`triplet[1]`), filtered to shape `(IDIM,)`.
- **Bottleneck:** trivial reshuffle (`np.array`, `np.save`). 1 s. **Not worth migrating.**
- **Parallelisable:** n/a (I/O bound, already instant).

### Step 4: `recover_weights.py`

- **Entry point:** `if __name__ == '__main__'`; `python3 recover_weights.py <LAYER>` (argv[1] = layer 0..3).
  Also opens `{argv[1]}_weight_vectors.txt` for logging at import (line 9).
- **Input:** `exp/1-cluster-{LAYER}.p`; `LAYER_SIZES`; `cheat_net_cpu` (for the prefix transfer);
  `cheat_solution` (ground-truth weights, used only for the post-hoc match/scale + metadata).
- **Output:** `outputs/model_weights/Vrelu/layer_{LAYER}/neuron_{cluster_id}/` containing
  `weights_unscaled.npz`, `weights_unscaled.txt`, `weights.npz`, `weights.txt`, `metadata.json`.
- **Output format:** `.npz` saved with positional `np.savez` → key **`arr_0`**, a `(IDIM,)`-length
  float64 vector (the recovered, L2-normalised null-space direction; `weights.npz` is
  magnitude-corrected by `/best_factor`). `metadata.json` keys:
  `{matched_neuron:int, scaling_factor:float, absolute_error:float, cluster_id:int}`.
- **Bottleneck:** per cluster, `extract_weights → is_consistent → is_consistent_help`
  (lines 124–215). For each of up to ~1200 triplets in a cluster it computes
  `get_normal(left)`, `get_normal(right)` (two autograd passes), an `intersect()`
  (`np.linalg.lstsq` + `scipy.linalg.null_space`), samples points on the null subspace,
  runs them through `prefix.forward_around`, then a **big `np.linalg.svd`** /
  `torch.linalg.svdvals` on the stacked centred samples. SVD + the per-triplet `get_normal`
  pair are the cost.
- **NumPy/torch calls:** `np.linalg.svd`, `np.linalg.lstsq`, `scipy.linalg.null_space`,
  `torch.linalg.svdvals`, `np.vstack`, `np.concatenate`, `get_normal` (autograd).
- **Parallelisable:** **YES** across clusters (each `dosteal` cluster is independent → batched
  SVD over clusters padded to uniform shape). Profiled cost is small (4 s for 4 layers at tiniest)
  but grows at tiny scale (more clusters, IDIM=64). Secondary target.

---

## INTERFACE CONTRACT (DO NOT CHANGE)

| Boundary | Contract |
|---|---|
| Pipeline input | trained `.pth` oracle at `utils.MODEL_PATH` + `LAYER_SIZES` from `utils.py` toggles |
| find_duals output | `list[(np.ndarray, np.ndarray, np.ndarray)]`, each `(IDIM,)` `float64`, pickled to `exp/{SEED}/duals_{rand}.p` |
| cluster output | `dict[int -> list[(ndarray,ndarray,ndarray)]]` pickled to `exp/1-cluster-{L}.p`, per-neuron ≤ 3000 |
| per-neuron npy | `np.ndarray (n_duals, IDIM)` float64 = middle points, at `sign_recovery/layer_neuron_npys/layer{L+1}_neuron{n}.npy` |
| weight npz | `np.savez` positional → key `arr_0`, vector len `IDIM`; dir `outputs/model_weights/Vrelu/layer_{L}/neuron_{id}/` |
| metadata.json | keys `matched_neuron, scaling_factor, absolute_error, cluster_id` |

**Numerical constants that MUST be preserved verbatim in the torch port:**
- Decision-boundary bisection stop: `|zero-one|` sum `> 1e-16` (point) / `> 1e-14` (batched).
- find_dual walk: step sweep `10**np.arange(-5,5,.1)`; "too big" guard `step_size > 10`;
  "too small" guard `step_size <= 1e-4`; binary-search stop `|upper-lower| > 1e-8`;
  `a_bit_past` offset `+1e-4`; path-end delta `|dist-last| < 1e-4`.
- `is_on_decision_boundary_cheat`: `|gap| < 1e-10`.
- `refine_to_decision_boundary_cheat`: `tolerance=1e-13`, `max_iterations=10`, derivative `h=1e-6`.
- `get_normal`: `USE_GRADIENT=True` → autograd of `gapt` (random scalar × grad, then L2 normalise).
- recover SVD gates — **ReLU mode**: `S[-2] > 1e-2 and S[-1] < 1e-4`; **Leaky mode**: return soln
  whenever SVD ran (gate moved downstream to `min(errs) < 1e-3`).
- recover thresholds: `hiddens > 1e-4`; `np.abs(samples) < 1e-5` (all-zero); `> 1e-5` (shared coords).
- `vectorized_check_closest_pair_distance`: reject if max pairwise sq-dist `< 1`.
- Activation: `LEAKY_ALPHA` single source of truth in utils; `act`/`act_np`/`cell_slope_mask`.

**Randomness:** find_duals calls `np.random.seed(None); random.seed(None)` at the top of `main()`
→ **non-deterministic by design**. The torch port must NOT add seeding. Output values will differ
run-to-run; equivalence is judged on **format + shape + dtype + value range + recovery rate**.

---

## BASELINE TIMINGS (tiniest, 8-wide, ReLU; this machine, CPU)

| Step | Time | Notes |
|---|---|---|
| `find_duals.py` (1 full run) | **14.76 s** | TARGET=3000 triplets; peak RSS 285 MB |
| `find_duals.py` × 8 (sequential) | 60 s | high variance — some runs `exit(0)` early ("Hit end of the road"), produce fewer files |
| `cluster_dual_points_stream.py` | **2 s** | 12 017 triplets seen, 10 706 kept |
| `generate_dual_neuron.py` | **1 s** | 29 `.npy` files written |
| `recover_weights.py` × 4 layers | **4 s** | recovered: L0 8/8, L1 7/7, L2 4, L3 5 |
| **Phase-1 total (9 find_duals + downstream)** | **~75–135 s** | find_duals = >90 % of it |

**Recovery sanity:** tiniest-ReLU here = L0 8/8, L1 7/7, L2 4, L3 5 (24/32), at/above the documented
baseline (fc1 8/8, fc2 6/8, fc3 0/8, fc4 5/8). The profiling run is healthy.

### Extrapolation to tiny (the real target)
find_duals at tiny is IDIM=64 (vs 8), TARGET=10000 (vs 3000), and 1000 iterations (vs 9) — the
documented tiny run took **~18 h**, essentially all in find_duals. cluster/generate/recover stay
seconds-to-minutes. **So the migration ROI is almost entirely in find_duals.**

## SPEED COMPARISON — MEASURED (Phase C, tiniest, 9 rounds, CPU)

| Configuration | Wall | Triplets | Speedup |
|---|---|---|---|
| NumPy single-thread (original) | ~75–135 s | ~12–27 k | 1× |
| Torch 1 worker (batched finder) | 42 s | ~36 k | ~2–3× |
| Torch 4 workers | 7 s | ~32 k | ~11–19× |
| Torch 8 workers | 5–9 s | ~33 k | ~10–25× |

Full results, equivalence checks and the lane-compaction / `max_outer` fixes are written up in
**`MIGRATION_RESULTS.md`**. Headline: format-identical, recovery 24–25/32 (≥ NumPy's 24/32),
~10–25× faster at 8 workers.

---

## MIGRATION PLAN (proposed for Phase B — confirm before I write code)

All new code in `signature_recovery/torch_impl/`. Originals untouched. `torch.set_default_dtype(torch.float64)`
at the top of every torch file. CPU-first.

1. **`find_duals_torch.py`** — `BatchDualPointFinder`: B walks in lockstep, batched oracle/grad.
   Preserve every constant above. Output identical pickle format.
2. **`parallel_duals.py`** — `torch.multiprocessing` (spawn) wrapper: W workers, each a finder,
   each writing its own `exp/{SEED}/duals_{rand}.p`. This is the dominant, certain win on a
   14-core CPU box (replaces the sequential shell loop).
3. **`cluster_torch.py`** *(secondary)* — batch K triplets per `cheat()` forward. Only if find_duals
   win exposes clustering as the next bottleneck.
4. **`recover_weights_torch.py`** *(secondary)* — batched SVD across clusters. Same SVD gates.
5. **`run_duals_torch.sh`** — drop-in for the STEP-2 shell loop; writes to the same `exp/{SEED}/`.

**Primary recommendation:** land #1 + #2 first (find_duals batched + multiprocess), validate on
tiniest (Phase C), then decide whether #3/#4 are worth it given they are already only ~6 s at tiniest.

### Risk-ranked uncertainties
1. **Batched walk masking** — walks converge at different steps; must freeze converged lanes
   without perturbing their recorded dual points. Medium risk; the lockstep step-sweep and binary
   search need careful masked-update logic.
2. **Autograd in batch** — `gapt(X,grad=True)` then per-row `.backward()`; need per-sample
   gradients (`torch.autograd.grad` with batched outputs or vmap). Must reproduce
   `random_scalar × grad → normalise` semantics row-wise.
3. **Early `exit(0)`** (find_duals line 177) — kills the whole process mid-walk in NumPy. In a
   batched/worker model this must become "end this lane/path", never `sys.exit`. Behavioural, not
   numerical — preserve the *effect* (drop that path), not the literal exit.
4. **`refine_to_decision_boundary` fallback chain** — Newton → random-direction → recurse. Batching
   the fallback is fiddly; may keep it per-lane (scalar) for correctness, batch only the hot path.
5. **CPU multiprocessing model sharing** — `model.share_memory_()` or per-worker reload; float64
   throughout to keep SVD/gap thresholds valid.
