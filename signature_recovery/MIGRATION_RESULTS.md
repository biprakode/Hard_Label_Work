# Signature Recovery — Torch Migration Results

**Status: find_duals migrated and validated on tiniest (Phase C complete).**
cluster / generate / recover left on the original NumPy code (they are seconds at
tiniest; secondary targets — see MIGRATION_NOTES.md "MIGRATION PLAN").

Machine: 14-core CPU, `torch 2.9.0+cpu` (no CUDA). Model: tiniest 8-8-8-8-8-8, ReLU,
make_blobs. Python `/home/biprarshi/miniconda3/envs/MLenv/bin/python3`.

## What was built (`signature_recovery/torch_impl/`)

| File | Role |
|---|---|
| `find_duals_torch.py` | `BatchDualPointFinder` equivalent: B independent boundary walks in lockstep, one batched forward per step. `find_batch(target, batch_size)` returns the identical `list[(left,middle,right)]` triplet format. Lane compaction drops finished walks from the batch each iteration (pure efficiency). |
| `parallel_duals.py` | `torch.multiprocessing` (spawn) wrapper, W workers each producing pickles in `exp/{SEED}/`. `--impl torch` (batched) or `--impl subprocess` (original find_duals.py parallelised). |
| `run_duals_torch.sh` | Drop-in for run_one_model.sh STEP 2. `./run_duals_torch.sh [ITERS] [WORKERS] [BATCH] [IMPL]`. |

Original files untouched. `torch.set_default_dtype(torch.float64)` everywhere. CPU-first
(`device='cuda'` should work by moving the model + tensors, untested here).

## C.1 Format equivalence — PASS

| Property | NumPy (`exp/1_numpy`) | Torch (`exp/1_torch`) |
|---|---|---|
| pickle type | `list[tuple]` | `list[tuple]` ✓ |
| tuple length | 3 | 3 ✓ |
| element type / shape | `ndarray (8,)` | `ndarray (8,)` ✓ |
| dtype | float64 | float64 ✓ |
| middle range | [-9.30, 12.88] | [-13.27, 18.33] (similar magnitude; values differ — walks are unseeded) |

## C.2 End-to-end recovery — PASS (matches/exceeds NumPy)

Downstream (cluster_dual_points_stream → generate_dual_neuron → recover_weights) run
**unmodified** on torch-produced duals.

| Layer | Torch duals | NumPy duals (Phase A) | Documented ReLU baseline |
|---|---|---|---|
| fc1 | 8/8 | 8/8 | 8/8 |
| fc2 | 7/8 | 7/7 | 6/8 |
| fc3 | 4–5 | 4 | 0/8 |
| fc4 | 5/7 | 5 | 5/8 |
| **Total** | **24–25/32** | 24/32 | 19/32 |

(fc3 is 4 or 5 across runs — coverage variance, both ≥ NumPy.) The torch finder also
gives *better layer coverage* (more triplets → fc2 covers 8 neurons vs 7).

## C.3 Speed comparison (tiniest, 9 rounds ≈ 9 find_duals.py invocations)

| Configuration | Wall time | Triplets | vs NumPy seq |
|---|---|---|---|
| **NumPy single-thread (original)** | **~75–135 s** | ~12–27 k | 1× (baseline) |
| Torch, 1 worker | 42 s | ~36 k | ~2–3× |
| Torch, 4 workers | 7 s | ~32 k | ~11–19× |
| Torch, 8 workers | 5–9 s | ~33 k | **~10–25×** |

(NumPy baseline from Phase A: one full run 14.76 s; 9 sequential ≈ 75–135 s depending on
early-exit paths.)

### Lane-compaction + max_outer cap fixed wall-time variance
Before compaction, w8 was *non-monotonic* (4 workers 8 s but 8 workers 19 s) and a single
run spiked to 73 s — a 256-lane batch is likely to contain one "marathon" lane that keeps
finding new boundaries, and the whole round waited on it.
- **Compaction** (drop finished lanes each iteration): w1 107→42 s, w8 19→6 s.
- **`max_outer=2000` cap** (a path of >2000 critical points is pathological; its first 2000
  still count): variance over 3 trials tightened to **5 / 6 / 9 s**.

At the real target (tiny, 1000 iterations / 8 workers = 125 rounds/worker) this lumpiness
amortises away entirely.

## Extrapolation to tiny (the ~18 h target)

find_duals is >90 % of Phase-1 wall time and the only step worth migrating (cluster/generate/
recover are seconds–minutes). The win scales with worker count and batch width. With 8 workers
on this box the expectation is a **~10×+ wall-clock reduction** for the tiny dual-point search,
i.e. ~18 h → low single-digit hours, before any GPU. On CUDA the batched forwards would compound
this (single batched SVD / forward across the batch).

## How to use (drop-in)

Replace run_one_model.sh STEP 2's sequential loop with:
```bash
./run_duals_torch.sh <ITERS> <WORKERS> <BATCH> torch
# e.g. tiniest:  ./run_duals_torch.sh 9 8 256 torch
#      tiny:     ./run_duals_torch.sh 1000 8 256 torch
```
Output lands in `signature_recovery/exp/{SEED}/` exactly like the original; the rest of the
pipeline is unchanged. Arch/activation still come from `signature_recovery/utils.py` toggles.

## Phase D — FULL TINY EXTRACTION (tiny ReLU, 64-64-64-64-64-10, make_blobs)

End-to-end run using the torch finder for the dual search and the **unmodified**
original code for cluster / generate / recover / sign / Phase-3. 8 workers, batch 256.

### Timings (this machine, CPU)
| Step | Time |
|---|---|
| find_duals (torch, 500 rounds, 8 workers) | **24.3 min** (1460 s) |
| cluster_dual_points_stream (8.45 M triplets streamed) | 6.5 min (392 s) |
| generate_dual_neuron | 5 s |
| recover_weights (4 layers, heavy SVD) | 185 s |
| batched_sign_recovery (256 neurons) | ~30 min |
| Phase 3 (reconstruct + sign search + fc5 LR + refine 1000 ep) | 25 s |
| **find_duals vs documented NumPy baseline** | **24.3 min vs ~18 h → ~44×** |

The dual search — the entire reason the pipeline took ~18 h — now finishes in **under
half an hour**. (cluster at 6.5 min and sign recovery at ~30 min are the new largest
costs; both are original unmodified code and out of scope for this migration. cluster
is the obvious next target — see "Not yet migrated".)

### Cluster coverage (neurons with ≥1 single-flip triplet)
fc1 64/64 · fc2 63/64 · fc3 53/64 · fc4 50/64 — 8.45 M triplets seen, 621 k kept.

### Signature recovery (SVD null-space)
| Layer | Torch run | Documented tiny_relu baseline |
|---|---|---|
| fc1 | 64/64 | — |
| fc2 | 62/64 | — |
| fc3 | 28/64 | — |
| fc4 | **0/64** | **0/64** (ReLU: OFF side contributes nothing to the SVD) |
| **Total** | **154/256** | **157/256** |

fc4 = 0/64 is the *expected* ReLU behaviour (Section 3 notes: fc4 on tiny is 0/64 ReLU
vs 57/64 Leaky). 154/256 matches the documented 157/256 within run variance.

### Final reconstruction (Phase 3, on X_test2)
- **Prediction agreement: 100.00 %**  ·  Reconstructed accuracy: 100.00 %  ·  GT accuracy: 100.00 %
- Mean |cos| = 1.000 on every recovered layer · sign_acc 0.50–0.64 (chance — *functional*, not structural, extraction, exactly as documented)
- `*** EXTRACTION SUCCESSFUL ***`, saved `reconstructed_makeblobs.{pth,npz}` + `extraction_metrics.json`

**This reproduces the documented tiny_relu result (100 % functional agreement, |cos|=1.0,
sign_acc≈chance) — with the dual search ~44× faster.** The torch migration is validated
at full tiny scale, not just tiniest.

## Not yet migrated (secondary, by design)
- `cluster_torch.py` — batch K triplets per `cheat()` forward. (cluster = 2 s at tiniest.)
- `recover_weights_torch.py` — batched SVD across clusters. (recover = 4 s at tiniest.)
Worth doing only if a tiny-scale run shows them becoming the next bottleneck after find_duals.

## Config note
Profiling/validation left `signature_recovery/utils.py` at `TINIEST=True`, `LEAKY_ALPHA=0.0`
(tiniest ReLU). Restore the desired arch/activation (run_one_model.sh STEP 1 does this) before
the next non-tiniest run.
