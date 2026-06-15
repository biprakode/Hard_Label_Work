# CIFAR-10 Flagship (ReLU) — Error & Incident Report

Target: `3072-256-256-256-64-10` ReLU victim (the model `vanilla_codebase`
extracts). Host: 14-core CPU, **22 GB RAM (≈10 GB usable, swap full)**, 109 GB
free disk. Pipeline: parallel torch dual-search → cluster → recover_weights →
Phase-3. This documents every error hit during the run and the fix applied.

---

## 1. Disk/RAM cannot hold "500 runs of find_duals" (sizing constraint)

- **Symptom / reality:** each dual triplet is `3 × 3072 × float64 = 74 KB`. The
  literal request "500 runs × 10 000 triplets" ≈ **500 GB** vs **109 GB** free.
- **Compounding:** the clusterer holds all kept triplets **in RAM**; its own
  docstring warns it "OOMs a 24 GB machine". 22 GB RAM is the binding limit, not
  disk.
- **Resolution:** sized collection to **140 rounds ≈ 700 K triplets ≈ 56 GB**,
  and capped per-neuron clustering so the in-RAM dict stays bounded. This
  saturates the 256 layer-0 neurons; deeper layers get what the walks find.

## 2. Dual-search thread oversubscription (perf, not crash)

- **Symptom:** first launch (4 workers, torch default 14 threads each → 56
  threads on 14 cores) → ~146 s/round, ~112 triplets/s aggregate (~100 min).
- **Root cause:** `find_duals_torch` sets dtype but not thread count; each spawn
  worker grabbed all cores.
- **Fix:** relaunch with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`, 5 workers →
  ~100 s/round, stable; full collection in **~68 min**.

## 3. Orphaned spawn workers survive `pkill`

- **Symptom:** after `pkill -f parallel_duals`, stray python processes kept
  running and **kept writing dual files** + held ~2.5 GB RAM.
- **Root cause:** `torch.multiprocessing` spawn children have cmdline
  `python -c "from multiprocessing.spawn import spawn_main…"` — they do **not**
  match `parallel_duals`, so `pkill -f parallel_duals` misses them.
- **Fix:** kill by PID tree / `pkill -f multiprocessing.spawn`; verify with
  `ps aux | grep '[M]Lenv/bin/python3'` before each new stage.

## 4. OOM-kill in `recover_weights` (the main incident) — twice

- **Symptom A (1st attempt):** killed mid-layer-0 after ~100/256 clusters; looked
  like gradual growth.
- **Symptom B (2nd attempt, with per-cluster gc):** killed after only **2**
  clusters — i.e. a **transient spike**, not a slow leak.
- **Root cause:** `np.linalg.svd(centered_samples)` defaults to
  `full_matrices=True`, allocating `U` of shape **(n, n)** where n ≈ 15 K sample
  rows ⇒ **~1.9 GB** — even though only `Vt[-1]` (smallest right singular vector)
  is used. Combined with the 2.8 GB cluster dict held in RAM and **full swap**
  (no spill room), free RAM (~6 GB at the time) couldn't absorb the spike →
  hard OOM-kill.
- **Fixes (all in `recover_weights.py`):**
  1. `np.linalg.svd(centered_samples, full_matrices=False)` + `del U` — kills the
     (n,n) allocation. Mathematically identical `Vt[-1]` since n ≫ d.
  2. `gc.collect()` + `del maybe, clean` after each cluster — flat memory.
  3. `CLUSTER_START/CLUSTER_END` env slicing → run each layer in **64-cluster
     chunks per process** (each process exits and frees everything).
  4. Truncated cluster dicts to **50 duals/neuron** (then 350 for deep layers) so
     the held dict shrank 2.8 GB → 0.9 GB.
- **Verified:** peak RSS dropped 7.1 GB → **5.1 GB**; full layer-0/1 recovery
  completed with 12 GB free throughout.

## 5. Over-aggressive truncation broke deep-layer recovery

- **Symptom:** at cap 50, **layers 2–3 recovered 0/245 and 0/53** ("Not enough to
  fully extract" for every cluster).
- **Root cause:** `is_consistent_help` rejects a cluster when
  `np.min(hits)==0` (any prefix-output coord never active) **in ReLU mode**.
  Through a 2–3-layer ReLU prefix, with only 50 duals many upstream neurons are
  never seen active → universal rejection. (Layers 0–1 survived: identity / 1-layer
  prefix is dense.)
- **Fix attempt:** added a layer filter to the clusterer
  (`CLUSTER_LAYERS=2,3`, `CLUSTER_PER_NEURON_CAP=350`) and re-clustered just the
  deep layers at **7× more duals** (RAM-bounded to those 298 neurons).
- **Outcome:** **still 0/245, 0/53.** Diagnostics showed "Mean OK 0.40–0.92" —
  8–60 % of the 256 upstream neurons are **structurally** inactive across a deep
  neuron's entire dual region, regardless of dual count. This is the genuine
  hard part of deep-layer signature recovery (matches the known baseline where
  `fc3` recovers 0/8), **not** a bug. Deep layers are deferred to Phase-3.
  *Note:* Leaky mode bypasses this rejection (the α·z leak carries signal), so a
  Leaky run is the lever to actually recover deep layers.

## 6. Keras `.keras` export is wrong (sign-recovery blocker — deferred)

- **Symptom:** trained victim's `PT↔Keras` max diff = **3.4e7**; a single bare
  Dense layer disagrees with `x@W+b` by ~7 even with identical weights.
- **Root cause:** `keras.backend.set_floatx('float64')` was **not** called before
  building the model in the trainer — a dtype/policy issue makes the loaded model
  compute the wrong forward.
- **Status:** **deferred, not blocking.** Sign recovery (Phase-2) loads the
  `.keras` oracle, but Phase-3 `--from-scratch` does its own hard-label greedy
  sign search, so the flagship runs without Phase-2. A standalone re-export
  (`reexport_keras.py`, sets float64 first + asserts `<1e-9`) is written but was
  not needed for this ReLU run.

## 7. Buffered stdout hid progress

- **Symptom:** background logs stayed empty for minutes; looked stalled.
- **Cause:** Python block-buffers stdout when not a TTY; TF C++ logs (stderr)
  appeared but `print()` did not.
- **Fix:** launch with `PYTHONUNBUFFERED=1`; otherwise confirm liveness via
  process CPU/RSS (`ps aux`) rather than the log.

---

## Net effect on results

None of these blocked the run; they shaped it. The OOM fixes (#4) made recovery
reliable; the coverage limit (#5) is a real property of the problem, not an
error. Final ReLU signature recovery: **495/832** (layer 0 254/256, layer 1
241/250, layers 2–3 0). See `cifar_flagship_insights.md` for the full results.
