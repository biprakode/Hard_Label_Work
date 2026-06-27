# EXPLANATIONS — how the enhanced hard-label DNN extraction codebase works

This file holds the **conceptual / understanding** material for the codebase:
what each piece is, how the three phases fit together, why Leaky ReLU helps, the
Phase-3 module layout, how the batched dual search works, and the known caveats.

For **how to actually launch the attack** (commands, flags, per-arch parameters,
sign-search options, expected results) see **[README.md](README.md)**.

---

## What this codebase is

Self-contained fork of the EUROCRYPT-2024 "Polynomial Time Cryptanalytic
Extraction of DNNs in the Hard-Label Setting" reference code, with six additions:

1. **Streaming clustering** (`cluster_dual_points_stream.py`) that processes
   the 10M+ triplet corpus in one memory-bounded pass (was OOMing the
   vanilla `cluster_dual_points.py`).
2. **Phase 3 reconstruction** (`analysis/extraction_pipeline/`, entry point
   `analysis/run_extraction.py`) — a hard-label post-processing stage that
   takes Phases 1+2 outputs, solves for biases geometrically from dual
   points, brute-force / greedy / metaheuristic sign-searches against oracle
   argmax, LR-fits fc5 on oracle hard labels, and polishes with a frozen-row
   cross-entropy refinement loop. Closes the gap from ~8 % to 99–100 %
   functional agreement.
3. **Per-model smoke scripts** — `run_extract.sh` + `evaluate_*` +
   `compare_true_vs_extracted*` so an end-to-end run produces both a
   reconstructed `.pth` and the two written reports (true-vs-extracted
   and extraction-quality).
4. **Leaky ReLU support** via a single `LEAKY_ALPHA` toggle (default `0.0` =
   plain ReLU, byte-identical to the original pipeline). Set `> 0` to attack
   `*_leakyrelu.{pth,keras}` victims. Five activation-aware patches are gated
   on `α > 0`; the ReLU path is never touched. See "How Leaky ReLU works" below.
5. **Modular Phase-3 layout** — the original 1500-line
   `analysis/test_extraction4.py` was cosmetically split into a
   `analysis/extraction_pipeline/` package (config, architectures,
   data_loading, metrics, weight_assembly, bias_recovery,
   output_layer_recovery, sign_search, refinement, workflow). The legacy
   `test_extraction4.py` remains as a thin re-export shim, so any existing
   call site (including `run_extract.sh`) keeps working unchanged. The new
   recommended entry point is `python3 analysis/run_extraction.py …`. See
   "Phase-3 module layout" below for the full map.
6. **Batched PyTorch dual search** (`signature_recovery/torch_impl/`,
   `run_duals_torch.sh`) — a drop-in replacement for the Phase-1 bottleneck.
   `find_duals.py`'s single-sample boundary walk is reimplemented as B
   independent walks running in lockstep, so every oracle/gradient call
   becomes one batched forward pass; a `torch.multiprocessing` wrapper runs W
   workers in parallel. **No algorithm changes** — same constants, same
   `(left, middle, right)` pickle format; the rest of the pipeline consumes
   the output unchanged. On a 14-core CPU the **tiny dual search dropped from
   ~18 h to ~24 min (≈44×)**. See "How the batched dual search works" below.

## Pipeline explanation (the three phases)

```
                                     oracle model (whitebox access)
                                              │
                ┌─────────────────────────────┴──────────────────────────────┐
                │                                                             │
        Phase 1 (signature recovery)                       Phase 3 (Phase-3 reconstruction)
                │                                                             │
 find_duals ─► cluster ─► recover_weights                    hard-label oracle (argmax only)
 (decision-boundary                    │                                      │
  walks)                               ▼                                      ▼
                             per-neuron weights       bias-recov ─► sign-search ─► fc5 LR fit ─► refine
                                      │                                                       │
                                      ▼                                                       │
                               Phase 2 (sign recovery)                                        │
                                      │                                                       │
                             per-neuron signs                                                 │
                                      │                                                       │
                                      └─────────────► signed weights ─────────────────────────┘
                                                                                              │
                                                                                              ▼
                                                                                 reconstructed_<model>.pth
```

- **Phase 1** extracts the *direction* and *magnitude* of every hidden weight
  row (not the sign). Quality depends on dual-point count and network depth.
  On tiniest you get `|cos|≈1.0` for most rows; on tiny the last hidden
  layer's SVD rank test typically fails because prefix propagation error
  grows with depth.
- **Phase 2** recovers the sign (+/-) for each row via statistical tests on
  decision-boundary walks. Reliability: high on middle layers, biased on
  layer 1 (no past-layer toggles) and weak on the last hidden layer (no
  future-layer toggles).
- **Phase 3** is what this codebase adds. It takes whatever Phase 1+2
  produced (including partial failures) and uses *only* hard-label oracle
  queries on `X_test` to:
  1. geometrically fix biases from dual points,
  2. brute-force / metaheuristic-fix wrong signs (see the sign-search options
     in [README.md](README.md#sign-search-methods-metaheuristic-sign-search)),
  3. replace fc5 with a fresh LR decoder on top of the extracted features,
  4. fine-tune biases + unrecovered rows + fc5 against oracle labels, with
     signature-recovered rows frozen.

## What is in this folder

```
enhanced_codebase/
├── README.md                      # technical launch guide (how to run the attack)
├── EXPLANATIONS.md                # this file (how the codebase works)
├── ATTACK_PROMPT.md               # few-shot LLM prompt: logs -> 2 reports
├── leaky_relu_port.md             # leaky-port plan, status, all 5 gated patches + 1 always-on fix
├── run_extract.sh                 # one-shot: duals -> cluster -> recover -> sign -> reconstruct
├── run_duals_torch.sh             # drop-in batched/parallel replacement for STEP-2 find_duals
├── create_tiniest_makeblobs_leakyrelu.py   # trainer for tiniest LeakyReLU(0.01) victim
├── create_tinier_makeblobs_leakyrelu.py    # trainer for tinier LeakyReLU(0.01) victim
│
├── signature_recovery/            # Phase 1 — extract weight directions + magnitudes
│   ├── utils.py                   # single source of truth: LAYER_SIZES, model path, x_test path
│   │                              # *** contains LEAKY_ALPHA toggle + act()/cell_slope_mask helpers ***
│   │                              # contains cheat_net_{cpu,cuda} (whitebox scaffolding, DEBUG-gated)
│   ├── find_duals.py              # decision-boundary walker → pickle of (left, middle, right) triplets
│   ├── cluster_dual_points_stream.py   # streaming, memory-bounded clustering  (USE THIS)
│   ├── cluster_dual_points.py     # original (loads everything in RAM — OOMs on tiny+)
│   ├── generate_dual_neuron.py    # cluster pickles → layer{L}_neuron{i}.npy per-neuron files
│   ├── recover_weights.py         # per-layer SVD null-space → unsigned weight rows
│   │                              # *** has 3 leaky-gated bypasses + 1 always-on shape-bug fix ***
│   ├── run_duals.sh               # bash loop: for i in 1..1000: python find_duals.py
│   ├── torch_impl/                # batched PyTorch port of the dual search (≈44× on tiny)
│   │   ├── find_duals_torch.py    # B boundary walks in lockstep; identical triplet format
│   │   └── parallel_duals.py      # torch.multiprocessing wrapper (--impl torch | subprocess)
│   ├── MIGRATION_NOTES.md         # Phase-A dataflow + interface contract + baseline profile
│   └── MIGRATION_RESULTS.md       # validation: format/recovery equivalence, speed, full tiny run
│
├── sign_recovery/                 # Phase 2 — recover signs via decision-boundary statistics
│   ├── sign_recovery.py           # per-neuron sign via d_on vs d_off walks
│   │                              # *** LEAKY_ALPHA toggle + _apply_act helper ***
│   ├── batched_sign_recovery.py   # parallel runner over all neurons, per-layer aggregation
│   │                              # *** LEAKY_ALPHA toggle, model_path picks _leakyrelu when α>0 ***
│   ├── whitebox.py                # reads true weights of the keras model (inherited scaffolding)
│   ├── blackbox.py                # coordinate transforms in affine-layer space
│   └── common.py                  # shared argparse / file-management
│
├── analysis/                      # Phase 3 — reconstruction + evaluation
│   ├── run_extraction.py          # thin CLI entry point → extraction_pipeline.workflow.main
│   ├── test_extraction4.py        # legacy shim: re-exports the modular package + main(); kept
│   │                              # so run_extract.sh and existing scripts keep working
│   ├── extraction_pipeline/       # modular split of the old 1500-line test_extraction4.py
│   │   ├── __init__.py            # module map (overview)
│   │   ├── config.py              # paths + LEAKY_ALPHA toggle + _act/_act_suffix helpers
│   │   ├── architectures.py       # TinyModel / TinierModel / TiniestModel / FullModel
│   │   ├── data_loading.py        # X_test / X_test2 loaders + ground-truth model loader
│   │   ├── metrics.py             # three-tier (sign / magnitude / combined) metrics + accuracy test
│   │   ├── weight_assembly.py     # build a model from extracted values (load_unsigned_weights,
│   │   │                          # load_signs, combine_weights_and_signs, reconstruct_model,
│   │   │                          # save_reconstructed_model)
│   │   ├── bias_recovery.py       # recover biases from dual points + _hidden_activations_up_to
│   │   ├── output_layer_recovery.py # fc5 LR fit on oracle hard labels
│   │   ├── sign_search.py         # oracle_sign_search + greedy + tabu/SA/PT metaheuristics
│   │   ├── refinement.py          # oracle_label_refinement (frozen-rows distillation)
│   │   └── workflow.py            # main() orchestration: data → reconstruct → bias-recov →
│   │                              # sign-search → fc5 LR fit → refine → eval → save
│   ├── compare_true_vs_extracted.py       # tiniest per-neuron weight comparison (ReLU baseline)
│   ├── compare_true_vs_extracted_tiny.py  # tiny (64x5->10) per-neuron weight comparison
│   ├── evaluate_reconstructed_makeblobs.py # accuracy/per-class/confusion-matrix on tiniest
│   └── evaluate_reconstructed_tiny.py     # same, for tiny
│
├── tiny_stuff/                    # oracle models used as the attack target (the victims)
├── data/                          # test data (x_test / x_test2 / x_test3 per arch)
└── results/                       # pipeline outputs land here
    ├── reports/                   # tiny / tiniest / tinier reports (ReLU + leakyrelu variants)
    ├── reconstructed_models/      # reconstructed_<model>.pth, extraction_metrics.json
    └── sign_recovery/             # layer{L}_signs.npy, layer{L}_confidences.npy, summary.json
```

## What each file does

### Phase 1 — signature recovery (`signature_recovery/`)

| File | One-line purpose |
|---|---|
| `utils.py` | Loads the oracle model as `cheat_net_{cpu,cuda}`, exposes `cheat_solution` (true weights), sets `LAYER_SIZES`/`LAYER_BOUNDARIES` from the three `TINIEST`/`TINIER`/`TINY` flags. Everything downstream imports from here. |
| `find_duals.py` | Sample a random `x` near the class boundary, walk along the boundary, and at each ReLU toggle record a `(left, middle, right)` triplet with `middle` sitting on a hidden-neuron hyperplane. Saves as a pickle under `exp/{SEED}/duals_XXXX.p`. |
| `cluster_dual_points_stream.py` | Stream every pickle once, call `cheat_neuron_diff_cuda(left, right)` to find which neuron flipped; group triplets by `(layer, flat_neuron_idx)`. Caps each bucket so memory stays bounded on large runs. Output: `exp/1-cluster-{0..3}.p`. |
| `cluster_dual_points.py` | The original EUROCRYPT reference clusterer. Loads all 10M triplets into one Python list — OOMs >22 GB of RAM on the 64x5 tiny model. Kept only for reference. |
| `generate_dual_neuron.py` | Unpack the per-layer cluster pickles into `sign_recovery/layer_neuron_npys/layer{L}_neuron{i}.npy` — one file per clustered neuron containing just the `middle` dual points. |
| `recover_weights.py` | For each layer L, build a `CIFAR10NetPrefix(L)` initialised to the true lower-layer weights, forward-propagate each cluster's `middle` points, SVD the centred matrix, keep the last right-singular vector if `S[-2] > 1e-2 and S[-1] < 1e-4`. Writes `neuron_{id}/weights_unscaled.npz` + `metadata.json` with the scaling factor. |
| `run_duals.sh` | Bash loop that calls `find_duals.py` N times. Parallelisable across SEEDs if you pass an argv. |

### Phase 2 — sign recovery (`sign_recovery/`)

| File | One-line purpose |
|---|---|
| `sign_recovery.py` | For a single `(layer, neuron)` pair, walk along the decision boundary on both sides of the target neuron's hyperplane, track the distance to the next ReLU toggle. `d_on > d_off` or vice-versa determines the sign. |
| `batched_sign_recovery.py` | Parallel runner: for each neuron, spawn a worker calling `sign_recovery.main(...)`. Aggregates results into `results/sign_recovery/layer{L}_{signs,confidences,votes}.npy` + `layer{L}_summary.json`. |
| `whitebox.py` | Inherited EUROCRYPT helper: reads true `keras_model.layers[i].get_weights()`. Used by `sign_recovery.py:302` and `:728`. |
| `blackbox.py` | Coordinate transforms from input space into the affine output space of a given layer. Blackbox *if* you pass it reconstructed weights — whitebox as currently wired. |
| `common.py` | Shared argparse + `df.pkl` / `df.csv` / `df.md` save helpers. |

### Phase 3 — reconstruction (`analysis/extraction_pipeline/`)

Phase 3 is a package rather than a single file. The legacy `test_extraction4.py`
is preserved as a thin re-export shim, so any existing call (including
`run_extract.sh`) still works. The package modules and the function each owns:

| Module | Public functions | Oracle interaction |
|---|---|---|
| `config.py` | `LEAKY_ALPHA`, `_act`, `_act_suffix`, all paths (`SIGNATURE_WEIGHTS_PATH`, `TINIEST_MODEL_PTH`, `X_TEST*_PATH`, …) | none |
| `architectures.py` | `TinyModel`, `TinierModel`, `TiniestModel`, `FullModel` (forwards routed through `_act`) | none |
| `data_loading.py` | `load_test_data` / `load_test2_data` / `load_ground_truth_model` | none |
| `metrics.py` | `compute_weight_metrics_v2` (three-tier: sign / magnitude / combined), `test_model_accuracy` | none |
| `weight_assembly.py` | `load_unsigned_weights` (sign-blind via `abs(scaling_factor)`); `load_signs`; `combine_weights_and_signs` (`sign==0` ⇒ `+1` so partial sign recovery doesn't zero-out recovered rows); `reconstruct_model` (Kaiming-init unrecovered rows); `save_reconstructed_model` | none |
| `bias_recovery.py` | `_hidden_activations_up_to`; `recover_biases_from_duals` (`b_i = median(-w_i · h_{L-1}(x_d))` over 30 dual points, bottom-up) | none — uses reconstructed forward |
| `output_layer_recovery.py` | `recover_output_layer` (fc5 LR fit: forward X_test through reconstructed fc1..fc4 → `h_4`; query `oracle(X_test).argmax`; multinomial LR from `h_4` to those labels; overwrite `fc5.{weight,bias}`) | hard-label only |
| `sign_search.py` | `oracle_sign_search` (≤18 recovered neurons: enumerate 2^k sign flips + joint bias flip; pick combo maximising hard-label agreement); `greedy_oracle_sign_search` (O(k)-per-pass for `k > 18`); **MetaHeuristic combinatorial search** `_metaheuristic_oracle_sign_search` dispatching to `tabu_oracle_sign_search` / `sa_oracle_sign_search` / `pt_oracle_sign_search` (greedy-warm-started, best-true-agreement guarded) with the shared `_flip_neuron` move, `_score_objective` (`agree`/`margin`), and `_saturated` watchdog | hard-label only |
| `refinement.py` | `oracle_label_refinement` (Adam CE against `oracle(X_test).argmax`; freezes rows with `recovered_mask[i]==True`; biases, fc5, random-init rows stay trainable; `--refine-unfreeze` opens everything for full distillation) | hard-label only |
| `workflow.py` | `main()` — wires every stage in order: data → ground-truth oracle → reconstruct → bias-recov (if `--from-scratch`) → sign-search → fc5 LR fit (if `--from-scratch`) → refinement → eval → save model + extraction_metrics.json | hard-label only |

### Analysis helpers

| File | Produces |
|---|---|
| `compare_true_vs_extracted_tiny.py` | Per-neuron `L1`, relative error, `cos sim`, `sign_correct` for the 64x5 tiny model. Dumps JSON + stdout table. |
| `compare_true_vs_extracted.py` | Same, for tiniest 8-8-8-8-8-8. |
| `evaluate_reconstructed_tiny.py` | Regenerates the make_blobs splits (seed=42), checks `oracle_acc`, `reconstructed_acc`, `agreement` on train/test/full + per-class + confusion matrix. |
| `evaluate_reconstructed_makeblobs.py` | Same, for tiniest. |

## How Leaky ReLU works

The pipeline supports both ReLU and Leaky ReLU(α) victims via a single
`LEAKY_ALPHA` toggle. With `α = 0` the codebase is byte-identical to the
original ReLU pipeline; with `α > 0` it switches model paths and applies
five activation-aware patches gated on `LEAKY_ALPHA > 0`.

### The math (why α > 0 *helps*)

At the kink `z = 0`, ReLU's slope jumps `0 → 1`; Leaky ReLU's jumps `α → 1`.
The attack scaffolding (dual-point detection, SVD null-space, sign walks) still
works because the kink itself is preserved. Surprisingly, α > 0 actually
**helps** signature recovery — the small α·z signal on "OFF" prefix coordinates
gives the SVD additional well-conditioned constraints that ReLU's pure null
space lacks. On tiniest, 22/32 recovered with α=0.01 vs 19/32 for ReLU; the
effect grows with depth/scale (tiny fc4: ReLU 0/64 → leaky 54/64).

### Five gated patches (all no-op when α = 0)

| Where | What |
|---|---|
| `signature_recovery/utils.py` | `act(x)` / `act_np(x)` / `cell_slope_mask(x)` helpers; `CIFAR10Net.forward` and `cheat()` use `act` instead of `self.relu`. |
| `signature_recovery/recover_weights.py` | `relu_around` linearisation uses `cell_slope_mask` (1 on ON cells, α on OFF cells); `is_consistent_help` bypasses the `np.min(hits) == 0` reject because OFF coords still carry α·z signal; `extract_weights` drops the `S[-2]>1e-2 and S[-1]<1e-4` SVD gate (leaky's α·z leakage adds extra small SVs); the real quality check happens downstream in `dosteal` via `min(errs) < 1e-3`. |
| `sign_recovery/sign_recovery.py` | `_apply_act(x)` helper replaces `x[x<0] = 0.0` with `x[x<0] *= α` at three sites; OFF-side wiggle masking uses `α * dy` instead of `0`. |
| `sign_recovery/batched_sign_recovery.py` | Resolves `*_leakyrelu.keras` model paths when `α > 0`; propagates `LEAKY_ALPHA` to the imported `sign_recovery` module. |
| `analysis/extraction_pipeline/` (legacy `test_extraction4.py` is now a re-export shim) | `_act` helper used in all 4 model classes' forwards (16 sites in `architectures.py`) and in `_hidden_activations_up_to` (`bias_recovery.py`); model-path suffix toggle in `config.py`. **Plus two non-leaky-specific safety patches**: (a) `weight_assembly.load_unsigned_weights` skips neurons without `metadata.json`; (b) `weight_assembly.combine_weights_and_signs` treats `sign == 0` (unknown) as `+1` instead of zeroing the weight. |

Always-on bug fixes surfaced during the leaky port (apply to ReLU mode too):
- `recover_weights.py is_consistent_help` had `hits = np.zeros(LAYER_SIZES[layer+1])`
  (target output dim), but the loop indexed `hits[coord]` with `coord ∈
  hiddens.shape[1]` (prefix output dim). Tiniest's uniform 8× widths made these
  accidentally equal; tinier's 32→16 broke broadcasting. Fixed to
  `hits = np.zeros(hiddens.shape[1])`. Also benefits ReLU non-uniform configs.
- `generate_dual_neuron.py` and `recover_weights.py::dosteal` now shape-filter
  triplets against `LAYER_SIZES[0]` (the active arch's input dim) before stacking,
  so stale triplets from a prior architecture run can't raise
  `ValueError: inhomogeneous shape`. No-op for clean runs; dropped count logged.

## Phase-3 module layout

The Phase-3 pipeline lives at `analysis/extraction_pipeline/`. The split is
purely cosmetic — every function preserves its original signature, the CLI
flags are unchanged, and the legacy `test_extraction4.py` is now a thin shim
that re-exports the same names from the new package. This means:

- New code → import from `extraction_pipeline.<module>` directly.
- Old code → keeps working unchanged via `from test_extraction4 import …`.
- `run_extract.sh` now calls `analysis/run_extraction.py` (the modular CLI).

Dependency graph (top → bottom, no cycles):

```
                    config.py    (paths, LEAKY_ALPHA, _act, _act_suffix)
                       │
                ┌──────┼──────────┐
                ▼      ▼          ▼
       architectures  metrics   data_loading
                │      │          │
                └──────┴────┬─────┘
                            ▼
                    weight_assembly  ──▶  bias_recovery  ──▶  output_layer_recovery
                                                    │
                                                    └──▶  sign_search
                                                              │
                                                              ▼
                                                         refinement
                                                              │
                                                              ▼
                                                         workflow (main)
```

Smoke-tested on this refactor (byte-equivalent output structure to the
pre-refactor pipeline — same metrics keys, same model file format, same
per-layer numbers within oracle-sign-search noise tolerance):
- Tiniest **ReLU** via `run_extraction.py --tiniest --sign-search --refine`:
  98.95 % on X_test2 (ground truth 99.95 %), 99.00 % agreement.
- Tinier **LeakyReLU(0.01)** via `run_extraction.py --tinier --sign-search --refine`:
  100.00 % on X_test2, 100 % agreement.
- Tiniest **LeakyReLU(0.01)** full pipeline (`--from-scratch --refine --refine-epochs 500`):
  98.60 % on X_test2, 98.55 % agreement, 24/32 neurons recovered with |cos|=1.0.
  (Phase 2 batched_sign_recovery hangs on layer 2 — known issue; Phase 3's
  oracle sign search brute-forces 2^8 combos per layer to fill in the
  unaggregated layers, so the run completes successfully anyway.)

## How the batched dual search works

`find_duals.py` is >90 % of Phase-1 wall time: it walks the decision boundary
one point at a time, issuing hundreds of single-sample oracle/gradient calls
per dual point, and the original driver runs it in a *sequential* shell loop.
`signature_recovery/torch_impl/` reimplements exactly this walk with **B
independent walks advancing in lockstep**, so each single-sample call becomes
one batched forward pass, and `parallel_duals.py` runs **W workers** at once.

This is a pure migration — **no algorithm changes**. Every numerical constant
(step sweep `10**arange(-5,5,.1)`, guards `>10`/`≤1e-4`, binary-search `1e-8`,
`|gap|<1e-10`, Newton `1e-13`/10 iters, refine fallback) is preserved, the walk
is unseeded exactly like the original, and the output is the byte-compatible
`list[(left, middle, right)]` pickle in `exp/{SEED}/`. Downstream
(`cluster_dual_points_stream.py` onward) is unchanged. `float64` throughout;
CPU-first (also runs on `device='cuda'` by moving the model).

Two efficiency refinements (recovery-neutral): **lane compaction** drops
finished walks from the batch each iteration, and a **`max_outer` cap** bounds
a rare "marathon lane" that would otherwise hold up a round. Full write-up in
`signature_recovery/MIGRATION_RESULTS.md`; dataflow + interface contract in
`signature_recovery/MIGRATION_NOTES.md`.

**Not yet ported (secondary).** After the dual search,
`cluster_dual_points_stream.py` (~6.5 min on tiny) and `batched_sign_recovery.py`
(~30 min) become the largest costs — both original code, out of scope for the
port. Clustering is the obvious next batching target (batch K triplets per
`cheat()` forward).

For how to *run* the batched dual search (the `run_duals_torch.sh` invocation and
its validated speedup table) see [README.md](README.md#batched-pytorch-dual-search).

## Known caveats

- **Whitebox scaffolding is still present** in `utils.py` (`cheat_*`,
  `cheat_solution`) and `sign_recovery/whitebox.py`. The workflow is
  **hard-label-clean in Phase 3**, but Phases 1 and 2 inherit the vanilla
  EUROCRYPT reference code's whitebox reads. See
  `results/reports/tiny_cheating_audit_2026-04-24.md` for the full audit.
- `find_duals.py` uses `SEED = 1` by default; running it N times in quick
  succession may produce the same random filename and overwrite. Pass a
  different integer as `argv[1]` to change the seed.
- Layer-1 sign recovery is biased positive because there are no past-layer
  ReLU toggles. Known limitation of the sign-recovery algorithm; affects both
  ReLU and Leaky modes equally. Phase-3 sign search fixes it.
- Refinement overfits to `X_test`. If your downstream evaluation uses the
  same `X_test` the refinement saw, agreement numbers are upper bounds.
  Use `X_test2` / `X_test3` for an honest out-of-sample number.
- Leaky α > 0.05 — untested. The leaky port was validated at α=0.01. Larger
  α weakens the dON/dOFF asymmetry by `(1-α)/(1+α)`; at α=0.2 sign recovery
  may need recalibration.

## License / credits

Attack algorithm: Carlini, Chen, Choquette-Choo, Kos, Tramèr,
"Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the
Hard-Label Setting", EUROCRYPT 2024. Original reference code in
`../vanilla_codebase/` (not shipped in this folder).
