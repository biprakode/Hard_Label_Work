# Cheating Ablation Study — Reproduction Guide

Scoped **only** to this study. For the rest of the codebase, see the root
`README.md`/`EXPLANATIONS.md`.

**This is the only file a reviewer needs to reproduce this study.** This
folder (`cheating_ablation/`) is intentionally kept minimal in git: this
file plus the shell scripts that regenerate everything (`run_ablation.sh`,
`run_one_cheat_sweep.sh`, `run_prefix_init_degradation_sweep.sh`,
`scripts/`). The generated write-ups (`cheat_disable_map.md`, every
`reports/<cheat>/results_table.md` + `observations.md` + raw artifacts/logs)
are **not** committed — they were extracted into a separate archive
(`cheating_ablation_data.zip`, alongside this repo, not tracked by git) to
keep the repo small; running the scripts below regenerates `reports/`
in place from scratch. If you only want to read the pre-computed
conclusions rather than rerun anything, ask for that archive rather than
digging through git history — it was never committed.

## What a "cheat" means here

A whitebox read: any point in the Phase 1 (signature recovery) or Phase 2
(statistical sign recovery) pipeline that reads the true victim model's
parameters (weights, biases) or internal activations instead of relying
purely on hard-label oracle queries. Taxonomy used throughout this study:

- **Weight/bias read** — reads a true numeric parameter value. All 6 cheats
  below are this type.
- **Activation read** — reads a true intermediate hidden-layer output
  (e.g. which neuron toggled). Cheat #4 (clustering).
- **True-label read** — would read a ground-truth class label rather than
  the oracle's hard-label output. None of the in-scope cheats are this
  type (Phase 3's oracle-driven sign search, which does something
  related but still oracle-only, is explicitly out of scope — see below).
- **Eval-only** — reading truth purely to *score* a result, never fed back
  into the attack. Not a cheat. Every report's "raw artifacts" folder keeps
  true-weight comparisons for grading only, same discipline as the rest of
  this codebase.

## Scope

**In scope**: Phase 1 (dual-point discovery, clustering, weight recovery)
and Phase 2 (statistical distance-to-toggle sign recovery). Six
`make_blobs` victims: tiniest / tinier / tiny, each ReLU and LeakyReLU.
Canonical 2026-06-21 parameters (`run_makeblobs_batch_2026-06-21.sh` /
`run_one_model_enhanced.sh`) held fixed for everything not under test, using
the parallel/batched torch dual search.

**Explicitly NOT run, and why**: oracle-driven sign search (brute-force,
greedy, SA, PT, margin-smoothed) — this is Phase 3 and already
hard-label-honest in its query pattern, just out of this study's Phase-1/2
focus; bias recovery — Phase 3; output-layer logistic-regression fit or
cryptanalytic fc5 extraction — Phase 3 / separate technique, both out of
scope; frozen-row ML refinement — Phase 3; CIFAR-10 (`full` architecture) —
out of scope per instruction, make_blobs only.

## Full cheat inventory and status

The line-referenced detail table (`cheat_disable_map.md`) lives in the
archive, not git (see top of this file). Summary, sufficient to reproduce
and interpret every result:

| # | Cheat | Status |
|---|---|---|
| 1 | Root loader (`cheat_net_cpu`/`cheat_net_cuda`) | Foundational, not independently testable |
| 2 | Boundary detection via true logit gap | Tested |
| 3 | Boundary refinement via true gradients | Tested |
| 4 | Neuron clustering via true activations | Tested |
| 5 | Neuron identity mapping / signature scaling (`NO_SIG_CHEAT`) | Tested (rerun; already-existing flag) |
| 6 | Prefix initialization with true weights | Locked (not removed by design). Two experiments: (a) confirmatory single-layer swap against a static baseline — null result, see below; (b) non-confirmatory recursive/compounding chain with honest scaling — real degradation found, write-up pending |
| 7 | Sign-walk whitebox parameterization | Tested |

## Reproducing each cheat's ON/OFF ablation

Every cheat's sweep script lives directly under `cheating_ablation/` or
`cheating_ablation/scripts/` (the latter for the sweeps needing extra
resource caps). All commands assume `cwd` = `Hard_Label_Work/`.

```bash
# 2. Boundary detection
./cheating_ablation/run_one_cheat_sweep.sh HONEST_BOUNDARY_DETECT boundary_detection

# 3. Boundary refinement
./cheating_ablation/run_one_cheat_sweep.sh HONEST_BOUNDARY_REFINE boundary_refinement

# 4. Neuron clustering (needs its own script: extra resource caps)
./cheating_ablation/scripts/neuron_clustering_run_sweep.sh

# 5. Signature scaling / NO_SIG_CHEAT rerun
./cheating_ablation/run_one_cheat_sweep.sh NO_SIG_CHEAT signature_scaling_rerun

# 6a. Prefix-init CONFIRMATORY (manual, two-pass, per victim/layer, swaps one
#     layer's prefix against a static already-recovered baseline; locked/
#     confirmatory status, not a single-command sweep):
STOP_AFTER_PHASE2=1 ./run_one_model_enhanced.sh <arch> <act>   # pass 1: canonical
python3 signature_recovery/recover_weights_recovered_prefix.py <LAYER> <arch_key>  # pass 2: swap one layer

# 6b. Prefix-init DEGRADATION (non-confirmatory rerun: honest scaling +
#     genuine recursive/compounding chain across all 3 non-trivial layers,
#     6 victims x 2 arms, one command):
./cheating_ablation/run_prefix_init_degradation_sweep.sh

# 7. Sign-walk (needs its own script: extra resource caps)
./cheating_ablation/scripts/sign_walk_run_sweep.sh

# Utility: resume boundary_detection for whichever victims aren't done yet
# (used mid-study after an interruption; safe to ignore if running fresh)
./cheating_ablation/scripts/boundary_detection_run_remaining.sh
```

Or run everything end-to-end with `run_ablation.sh` (below).

Each sweep writes `reports/<cheat>/raw/` (per-run Stage-0 JSON + archived
artifacts + recovery-count logs), `reports/<cheat>/logs/` (full driver
logs), and `reports/<cheat>/results_table.md` + `observations.md` (the
human-readable writeup this guide's tables are drawn from) — regenerated
fresh under `cheating_ablation/reports/` each time a script runs; this
directory is gitignored (see `.gitignore`), matching the archived copy
described at the top of this file.

## Run parameters: dual points and sign-recovery budget

All values below are the canonical 2026-06-21 defaults (`run_one_model_enhanced.sh`),
held fixed across every cheat's ON and OFF arm in this study except where a
resource-only cap is explicitly noted (see "Resource-only adaptations" below).
"Seen/kept" dual-point counts are representative single-run samples pulled
from this study's own logs (`cheating_ablation/reports/prefix_init_degradation/logs/`,
2026-07-09) — the parallel dual search is randomized per run, so exact
counts vary run-to-run by low double-digit percentages; round counts, target/round,
workers, and batch size do not vary.

### Dual-point search (`torch_impl/parallel_duals.py --impl torch`, Phase 1 STEP 2)

| Arch | Rounds (`DUAL_ITERS`) | Target triplets/round | Workers | Batch size | Representative seen/kept (ReLU) | Representative seen/kept (LeakyReLU) |
|---|---|---|---|---|---|---|
| tiniest | 6  | 3,000  | 7 | 256 | seen=22,102 kept=4,135 | seen=19,121 kept=4,055 |
| tinier  | 8  | 2,000  | 7 | 256 | seen=20,161 kept=5,448 | seen=18,910 kept=5,100 |
| tiny    | 20 | 10,000 | 7 | 256 | seen=324,184 kept=30,331 | seen=248,336 kept=30,955 |

("Rounds × target/round" is the search *budget* — e.g. tiny requests up to
20 × 10,000 = 200,000 candidate triplets across its 20 parallel rounds;
"seen" is how many raw candidates the search actually generated before
filtering, "kept" is how many survived the boundary-consistency filter and
were written to `signature_recovery/exp/1/*.p` for clustering. `TARGET` is
architecture-keyed in `find_duals_torch.py`: `3000 if TINIEST else (2000 if
TINIER else 10000)` — `full`/CIFAR-10 is out of this study's scope and uses
different values entirely, see `run_one_model_enhanced.sh`'s header comment.)

### Sign recovery (`sign_recovery/batched_sign_recovery.py`, Phase 2 STEP 6)

Canonical per-neuron experiment budget, identical across all three
architectures (no arch-specific branching in `batched_sign_recovery.py` —
only layer-specific):

| Layer            | `nExpMin` | `nExp` | `choose_dx`               |
| ---------------- | --------- | ------ | ------------------------- |
| 1 (first hidden) | 200       | 2,000  | `along_decision_boundary` |
| 2                | 200       | 2,000  | `along_decision_boundary` |
| 3                | 200       | 2,000  | `along_decision_boundary` |
| 4 (last hidden)  | 100       | 1,000  | `along_decision_boundary` |

`nThreads` (worker-pool size) is hardcoded to 48 in
`batched_sign_recovery.py` (sized for a 56-thread cloud box). Every sweep
script in this study temporarily patches it to 6 for the sweep's duration
(restored on exit) — see "Resource-only adaptations" below; this is a
parallelism cap only, it does not change `nExp`/`nExpMin` or which neurons
get processed.

**Deviation used for exactly one cheat's OFF arm**: `HONEST_SIGN_WALK`'s
honest reconstruction produces a substantially higher per-experiment
exclusion rate for some neurons than the canonical (true-weight) walk this
`nExp`/`nExpMin` pairing was tuned against, causing multi-hour stalls on
individual neurons. `reports/sign_walk/run_sweep.sh` sets
`SIGN_NEXP_CAP=500` for that arm only (`nExp` capped to 500, `nExpMin`
reduced proportionally to `min(nExpMin, 50)`) plus `STEP6_TIMEOUT=300` as an
outer wall-clock backstop. No other cheat's sweep sets `SIGN_NEXP_CAP`; all
11 other arms (ON and OFF) across all 6 cheats/reports ran at the canonical
200-2000/100-1000 budget above.

## Metrics computed, and how

- **Recovered neuron count**: `recover_weights.py`'s own per-layer STEP-5
  log summary, or (where that log line's branch differs — see the
  `NO_SIG_CHEAT` report's measurement note) `ablation_tiny/ablation_harness.py`
  Stage 0's `Recovered X/Y neurons` line, which parses `metadata.json`
  presence directly and is correct regardless of which code branch wrote it.
- **\|cos\| / sign accuracy**: `analysis/extraction_pipeline/metrics.py`'s
  `compute_weight_metrics_v2`, invoked inside `ablation_harness.py`'s Stage 0
  reconstruction — no new metric code was written for this study.
- **Functional agreement/accuracy**: `ablation_tiny/ablation_harness.py`
  Stage 0 exactly — recovered directions + Phase-2 statistical signs,
  biases zeroed, fc5 Kaiming-random (no LR fit), evaluated on held-out
  `X_test3`. This is the "Phases I and II alone" methodology; no new eval
  code was written, this study only calls the existing Stage-0 path.
- **Wall-clock**: `date +%s` around each driver invocation, in every sweep
  script.

## Resource-only adaptations (not method changes)

Several environment-variable knobs were added or reused to keep otherwise-
intractable honest arms bounded on this study's CPU-only dev box. None
change what's measured for a run that completes normally; they only bound
worst-case cost. Full rationale for each is in `cheat_disable_map.md`'s
"Notes / deviations" section and the relevant per-cheat `observations.md`:

- `STOP_AFTER_PHASE2` — skip Phase 3 entirely (this study's scope).
- `CLUSTER_SLOW_MAX_SEEDS`, `CLUSTER_SLOW_MAX_INNER`, `VERBOSE_IS_CONSISTENT`
  — cap `cluster_slow`'s O(n²) search and its debug-print I/O.
- `SIGN_NTHREADS` (pre-existing, `ablation_tiny/run_ablation.sh`) — lower
  `batched_sign_recovery.py`'s worker-pool size for this box.
- `SIGN_NEXP_CAP`, `STEP6_TIMEOUT` — bound the honest sign-walk's per-neuron
  experiment count and the step's overall wall-clock.

## Bugs found and fixed (all in previously-dead or newly-exercised code paths)

A full list with file:line references is in `cheat_disable_map.md`. None of
these were introduced by this study's method changes — they were latent,
either in code nobody had run before (`cluster_slow`, five bugs) or exposed
by exercising a code path (`dosteal`'s `layer`/`LAYER` global-scoping quirk,
`load_unsigned_weights`'s `layer_offset` requirement) for the first time in
a context other than the one it was originally written for.

## What's in git vs. what's regenerated/archived

Tracked in `cheating_ablation/` (this is everything a fresh clone has):

- `REPRODUCE.md` — this file, the single entry point.
- `run_ablation.sh` — one-command end-to-end reproduction of every tested
  cheat's sweep.
- `run_one_cheat_sweep.sh`, `run_prefix_init_degradation_sweep.sh` — shared/
  standalone sweep drivers.
- `scripts/` — the 3 sweep drivers needing cheat-specific resource caps
  (`neuron_clustering_run_sweep.sh`, `sign_walk_run_sweep.sh`) plus one
  resume utility (`boundary_detection_run_remaining.sh`).

**Not tracked** (gitignored, regenerated by the scripts above, or provided
separately as `cheating_ablation_data.zip`):

- `reports/<cheat>/` — per-cheat `results_table.md`, `observations.md`,
  `raw/` (artifacts + Stage-0 JSON), `logs/` (full driver output).
- `cheat_disable_map.md` — the authoritative, line-referenced status table
  summarizing every cheat, its file:line, and its finding.
