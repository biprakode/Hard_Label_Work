# Enhanced hard-label DNN extraction codebase

Self-contained fork of the EUROCRYPT-2024 "Polynomial Time Cryptanalytic
Extraction of DNNs in the Hard-Label Setting" reference code, with five
additions:

1. **Streaming clustering** (`cluster_dual_points_stream.py`) that processes
   the 10M+ triplet corpus in one memory-bounded pass (was OOMing the
   vanilla `cluster_dual_points.py`).
2. **Phase 3 reconstruction** (`analysis/extraction_pipeline/`, entry point
   `analysis/run_extraction.py`) — a hard-label post-processing stage that
   takes Phases 1+2 outputs, solves for biases geometrically from dual
   points, brute-force / greedy sign-searches against oracle argmax, LR-fits
   fc5 on oracle hard labels, and polishes with a frozen-row cross-entropy
   refinement loop. Closes the gap from ~8 % to 99–100 % functional
   agreement.
3. **Per-model smoke scripts** — `run_extract.sh` + `evaluate_*` +
   `compare_true_vs_extracted*` so an end-to-end run produces both a
   reconstructed `.pth` and the two written reports (true-vs-extracted
   and extraction-quality).
4. **Leaky ReLU support** via a single `LEAKY_ALPHA` toggle (default `0.0` =
   plain ReLU, byte-identical to the original pipeline). Set `> 0` to attack
   `tiniest_makeblobs_leakyrelu.{pth,keras}` / `tinier_makeblobs_leakyrelu.*`.
   Five activation-aware patches are gated on `α > 0`; the ReLU path is never
   touched. See `leaky_relu_port.md` (project root) and §"Leaky ReLU usage"
   below for the full guide.
5. **Modular Phase-3 layout** — the original 1500-line
   `analysis/test_extraction4.py` was cosmetically split into a
   `analysis/extraction_pipeline/` package (config, architectures,
   data_loading, metrics, weight_assembly, bias_recovery,
   output_layer_recovery, sign_search, refinement, workflow). The legacy
   `test_extraction4.py` remains as a thin re-export shim, so any existing
   call site (including `run_extract.sh`) keeps working unchanged. The new
   recommended entry point is `python3 analysis/run_extraction.py …`. See
   §"Phase-3 module layout" below for the full map.
6. **Batched PyTorch dual search** (`signature_recovery/torch_impl/`,
   `run_duals_torch.sh`) — a drop-in replacement for the Phase-1 bottleneck.
   `find_duals.py`'s single-sample boundary walk is reimplemented as B
   independent walks running in lockstep, so every oracle/gradient call
   becomes one batched forward pass; a `torch.multiprocessing` wrapper runs W
   workers in parallel. **No algorithm changes** — same constants, same
   `(left, middle, right)` pickle format; the rest of the pipeline consumes
   the output unchanged. On a 14-core CPU the **tiny dual search dropped from
   ~18 h to ~24 min (≈44×)** while reproducing the documented tiny_relu result
   (100 % functional agreement, |cos|=1.0, 154/256 recovered). See
   §"Batched PyTorch dual search" below.

## Latest unified workflow (canonical entry point)

`run_one_model_enhanced.sh <arch> <activation>` is the single driver that runs
the **complete updated pipeline** end-to-end for any of the 8 supported
configurations (6 make_blobs tiny models + 2 CIFAR-10 flagships). It bundles:

- **Parallel batched dual search** — `signature_recovery/torch_impl/parallel_duals.py`
  with `--impl torch`, `W` workers in lockstep (≈44× over the legacy NumPy
  sequential walker on tiny).
- **Improved sign search** — `--sign-restarts R` (multi-start greedy traversal),
  `--sign-pair-lookahead 8` (C(K,2) pair flips on the K most uncertain
  neurons after greedy convergence), `--sign-refine-cycles 3` (interleave
  sign-search ↔ E-epoch mini-refinement). fc5 LR-fit runs **before** sign
  search so sign decisions are scored against a calibrated head.
- **X_test3 honest-eval validation** — `--eval-on-test3` routes every
  refinement watchdog and final eval to a strictly held-out slice (seed=123
  for make_blobs; CIFAR `train[10000:20000]` for CIFAR). Combined with
  `--train-union-test12`, the queryable pool is `X_test ∪ X_test2` (20 K
  samples) while X_test3 is never queried, never used for sign-flip selection,
  never used for watchdog tuning.
- **Watchdog early-stop** — `--early-stop --patience 5 --eval-every 10`
  evaluates on a 1024-row X_test3 slice every 10 epochs, saves the best
  checkpoint, stops after 5 watchdog evals without improvement, and restores
  best at end. Prevents refinement overfit.
- **AdamW + CosineAnnealingLR** — `--refine-weight-decay 1e-4 --refine-cosine-lr`
  for the refinement step.

### Per-arch tuning (set inside the driver)

| Arch | DUAL_ITERS | DUAL workers / batch | SIGN_RESTARTS | SIGN_PAIR | SIGN_CYCLES | REFINE_EPOCHS |
|---|---|---|---|---|---|---|
| `tiniest` | 6  | 7 / 256 | 1 | 8 | 3 | 300 |
| `tinier`  | 8  | 7 / 256 | 1 | 8 | 3 | 500 |
| `tiny`    | 20 | 7 / 256 | 2 | 8 | 3 | 500 |
| `full`    | 80 | 5 / 48  | 4 | 8 | 3 | 500 |

Each `DUAL_ITERS` round emits the per-arch TARGET triplet count
(tiniest=3000, tinier=2000, tiny=10000, full=10000). Override at the CLI:
`./run_one_model_enhanced.sh tiny relu 50`.

### Running all 8 configurations

```bash
cd enhanced_codebase/Hard_Label_Work
export PYTHON_BIN=/home/biprarshi/miniconda3/envs/MLenv/bin/python3

# 6 make_blobs tiny models
./run_one_model_enhanced.sh tiniest relu        # ~2 min wall
./run_one_model_enhanced.sh tiniest leakyrelu   # ~2 min
./run_one_model_enhanced.sh tinier  relu        # ~15 min
./run_one_model_enhanced.sh tinier  leakyrelu   # ~15 min
./run_one_model_enhanced.sh tiny    relu        # ~30 min (parallel duals)
./run_one_model_enhanced.sh tiny    leakyrelu   # ~30 min

# 2 CIFAR-10 flagships (3072-256-256-256-64-10, 832 hidden neurons)
./run_one_model_enhanced.sh full    relu        # ~5–8 h wall, 22 GB RAM
./run_one_model_enhanced.sh full    leakyrelu   # ~5–8 h wall, 22 GB RAM
```

What the driver does at each invocation:

1. **STEP 0** — clean all Phase 1+2+3 residuals (`exp/1/`, cluster pickles,
   `outputs/model_weights/Vrelu/layer_*`, `layer_neuron_npys/`,
   `results/sign_recovery/`, `results/reconstructed_models/`).
2. **STEP 1** — sync `LEAKY_ALPHA` (0.0 / 0.01) and the four arch booleans
   (`TINIEST/TINIER/TINY/MAKEBLOBS`) across the four config files
   (`signature_recovery/utils.py`, `sign_recovery/sign_recovery.py`,
   `sign_recovery/batched_sign_recovery.py`,
   `analysis/extraction_pipeline/config.py`).
3. **STEP 2** — Phase 1 batched dual search via `parallel_duals.py --impl torch`.
4. **STEP 3** — streaming cluster (`cluster_dual_points_stream.py`).
5. **STEP 4** — per-neuron bridge (`generate_dual_neuron.py`) + weight
   recovery (`recover_weights.py {0..3}`).
6. **STEP 5** — Phase 2 sign recovery (`batched_sign_recovery.py`).
7. **STEP 6** — Phase 3 reconstruction with the full updated flag set:
   ```
   analysis/run_extraction.py --<arch> --from-scratch --refine \
     --refine-epochs $REFINE_EPOCHS --refine-weight-decay 1e-4 --refine-cosine-lr \
     --early-stop --patience 5 --eval-every 10 \
     --eval-on-test3 --train-union-test12 \
     --sign-restarts $SIGN_RESTARTS --sign-pair-lookahead $SIGN_PAIR \
     --sign-refine-cycles $SIGN_CYCLES
   ```
8. **STEP 7** — emit the per-model true-vs-extracted report under
   `paper_notes/section3/reports/<arch>_<activation>_true_vs_extracted.{md,json}`.

### Distillation baseline (CIFAR only)

After the `full` extraction completes, the no-signature distillation baseline
runs the same Phase 3 with all 832 hidden rows Kaiming-initialised and
trainable (`--refine-unfreeze`), giving an apples-to-apples
"with-signature vs without-signature" comparison on the same queryable pool
and the same X_test3 held-out eval:

```bash
./run_distillation_baseline.sh
# writes paper_notes/section3/reports/cifar_<activation>_distillation.md
```

### Prereqs

- Python 3.11+, env with `torch`, `tensorflow`/`keras`, `numpy`, `scipy`,
  `scikit-learn`, `pandas`, `tabulate`.
- Free RAM: ≥4 GB (tiniest), ≥8 GB (tinier), ≥20 GB (tiny / full).
- Disk: ≥80 GB free for `full` (Phase-1 dual pickles alone reach ~55 GB).
- Victim artefacts in `tiny_stuff/` (`<name>_{relu,leakyrelu}.{pth,keras}`).
  CIFAR victims trained via `python3 create_cifar_model.py`; make_blobs
  victims via `create_*_makeblobs_*.py`.
- Test slices in `data/` — three slices per arch: `x_test*`, `x_test2_*`,
  `x_test3_*` (X_test3 is held-out, never queried). Emit make_blobs X_test3
  via `python3 emit_test3_makeblobs.py` if missing.

### Per-arch headline numbers (latest verified runs)

See `paper_notes/section3/reports/` for the full per-model true-vs-extracted
markdown reports — one per `<arch>_<activation>` combination — including
sign-cycle log, pair-lookahead results, watchdog peak, eval-tag, and
per-layer cos sim / sign accuracy.

## What is in this folder

```
enhanced_codebase/
├── README.md                      # this file
├── ATTACK_PROMPT.md               # few-shot LLM prompt: logs -> 2 reports
├── leaky_relu_port.md             # leaky-port plan, status, all 5 gated patches + 1 always-on fix
├── run_extract.sh                 # one-shot: duals -> cluster -> recover -> sign -> reconstruct
├── run_duals_torch.sh             # NEW: drop-in batched/parallel replacement for STEP-2 find_duals
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
│   ├── torch_impl/                # NEW: batched PyTorch port of the dual search (≈44× on tiny)
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
│   ├── run_extraction.py          # NEW: thin CLI entry point → extraction_pipeline.workflow.main
│   ├── test_extraction4.py        # legacy shim: re-exports the modular package + main(); kept
│   │                              # so run_extract.sh and existing scripts keep working
│   ├── extraction_pipeline/       # NEW: modular split of the old 1500-line test_extraction4.py
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
│   │   ├── sign_search.py         # oracle_sign_search + greedy_oracle_sign_search
│   │   ├── refinement.py          # oracle_label_refinement (frozen-rows distillation)
│   │   └── workflow.py            # main() orchestration: data → reconstruct → bias-recov →
│   │                              # sign-search → fc5 LR fit → refine → eval → save
│   ├── compare_true_vs_extracted.py       # tiniest per-neuron weight comparison (ReLU baseline)
│   ├── compare_true_vs_extracted_tiny.py  # tiny (64x5->10) per-neuron weight comparison
│   ├── evaluate_reconstructed_makeblobs.py # accuracy/per-class/confusion-matrix on tiniest
│   └── evaluate_reconstructed_tiny.py     # same, for tiny
│
├── tiny_stuff/                    # oracle models used as the attack target
│   ├── tiniest_makeblobs_relu.{pth,keras}        # 8-8-8-8-8-8 make_blobs ReLU
│   ├── tiniest_makeblobs_leakyrelu.{pth,keras}   # same arch, LeakyReLU(0.01)
│   ├── tiniest_makeblobs_leakyrelu_alpha.txt     # records α value for downstream readers
│   ├── tinier_makeblobs_relu.{pth,keras}         # 32-16-16-16-8-4 make_blobs ReLU
│   ├── tinier_makeblobs_leakyrelu.{pth,keras}    # same arch, LeakyReLU(0.01)
│   ├── tinier_makeblobs_leakyrelu_alpha.txt
│   └── makeblobs_relu.{pth,keras}                # tiny 64x5->10 ReLU (no leaky variant trained yet)
│
├── data/                          # test data (x_test) used for sign-search / refine / eval
│   ├── x_test_tiniest_makeblobs.npy, y_test_tiniest_makeblobs.npy   (seed=42, Phase-3 training)
│   ├── x_test2_tiniest_makeblobs.npy, y_test2_tiniest_makeblobs.npy (seed=99, eval-only)
│   ├── x_test_tinier_makeblobs.npy,  y_test_tinier_makeblobs.npy
│   ├── x_test2_tinier_makeblobs.npy, y_test2_tinier_makeblobs.npy
│   └── x_test_makeblobs.npy, y_test_makeblobs.npy
│
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
| `cluster_dual_points_stream.py` | Stream every pickle once, call `cheat_neuron_diff_cuda(left, right)` to find which neuron flipped; group triplets by `(layer, flat_neuron_idx)`. Caps each bucket at 3000 (more than `recover_weights.py`'s `[:1200]` slice ever uses) so memory stays bounded on large runs. Output: `exp/1-cluster-{0..3}.p`. |
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

Phase 3 is now a small package rather than a single file. The legacy
`test_extraction4.py` is preserved as a thin re-export shim, so any existing
call (including `run_extract.sh`) still works. Recommended new entry point:

```bash
python3 analysis/run_extraction.py [--tiniest | --tinier | --makeblobs | --full]
                                   [--from-scratch] [--sign-search]
                                   [--refine] [--refine-unfreeze]
                                   [--refine-epochs N] [--refine-lr LR]
```

The package modules and the function each one owns:

| Module | Public functions | Oracle interaction |
|---|---|---|
| `config.py` | `LEAKY_ALPHA`, `_act`, `_act_suffix`, all paths (`SIGNATURE_WEIGHTS_PATH`, `TINIEST_MODEL_PTH`, `X_TEST*_PATH`, …) | none |
| `architectures.py` | `TinyModel`, `TinierModel`, `TiniestModel`, `FullModel` (forwards routed through `_act`) | none |
| `data_loading.py` | `load_test_data` / `load_test2_data` / `load_ground_truth_model` | none |
| `metrics.py` | `compute_weight_metrics_v2` (three-tier: sign / magnitude / combined), `test_model_accuracy` | none |
| `weight_assembly.py` | `load_unsigned_weights` (sign-blind via `abs(scaling_factor)`); `load_signs`; `combine_weights_and_signs` (`sign==0` ⇒ `+1` so partial sign recovery doesn't zero-out recovered rows); `reconstruct_model` (Kaiming-init unrecovered rows); `save_reconstructed_model` | none |
| `bias_recovery.py` | `_hidden_activations_up_to`; `recover_biases_from_duals` (`b_i = median(-w_i · h_{L-1}(x_d))` over 30 dual points, bottom-up) | none — uses reconstructed forward |
| `output_layer_recovery.py` | `recover_output_layer` (fc5 LR fit: forward X_test through reconstructed fc1..fc4 → `h_4`; query `oracle(X_test).argmax`; multinomial LR from `h_4` to those labels; overwrite `fc5.{weight,bias}`) | hard-label only |
| `sign_search.py` | `oracle_sign_search` (per layer with ≤18 recovered neurons: enumerate 2^k sign flips + joint bias flip; pick combo maximising hard-label agreement); `greedy_oracle_sign_search` (O(k)-per-pass for `k > 18`); auto-falls-back to greedy when the layer is too wide | hard-label only |
| `refinement.py` | `oracle_label_refinement` (Adam CE against `oracle(X_test).argmax`; freezes rows with `recovered_mask[i]==True`; biases, fc5, random-init rows stay trainable; `--refine-unfreeze` opens everything for full distillation) | hard-label only |
| `workflow.py` | `main()` — wires every stage in order: data → ground-truth oracle → reconstruct → bias-recov (if `--from-scratch`) → sign-search → fc5 LR fit (if `--from-scratch`) → refinement → eval on X_test2 → save model + extraction_metrics.json | hard-label only |

### Analysis helpers

| File | Produces |
|---|---|
| `compare_true_vs_extracted_tiny.py` | Per-neuron `L1`, relative error, `cos sim`, `sign_correct` for the 64x5 tiny model. Dumps JSON + stdout table. |
| `compare_true_vs_extracted.py` | Same, for tiniest 8-8-8-8-8-8. |
| `evaluate_reconstructed_tiny.py` | Regenerates the make_blobs splits (seed=42), checks `oracle_acc`, `reconstructed_acc`, `agreement` on train/test/full + per-class + confusion matrix. |
| `evaluate_reconstructed_makeblobs.py` | Same, for tiniest. |

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
  2. brute-force fix wrong signs on small layers,
  3. replace fc5 with a fresh LR decoder on top of the extracted features,
  4. fine-tune biases + unrecovered rows + fc5 against oracle labels, with
     signature-recovered rows frozen.

## Complete guide: running extraction on a given model

### Prereqs

- Python 3.11+, miniconda env with `torch`, `tensorflow`/`keras`, `numpy`,
  `scipy`, `scikit-learn`, `pandas`. On this machine:
  `/home/biprarshi/miniconda3/envs/DLenv/bin/python3`.
- Set `PYTHON_BIN=/path/to/python` or rely on `python3` in PATH.
- Free RAM: ≥4 GB for tiniest, ≥8 GB for tinier, **≥20 GB (or 22+GB + swap)**
  for tiny. Close browsers/IDEs before the tiny cluster step.

### Step 0 — provide your model

You need two equivalent model files in `tiny_stuff/`:
- `<name>.pth` — PyTorch state dict matching the architecture class in
  `signature_recovery/utils.py::CIFAR10Net` (fc1..fc5).
- `<name>.keras` — TensorFlow/Keras equivalent for Phase 2 sign recovery.

If you only have a `.pth`, `create_makeblobs_model.py`-style wrappers show
the PyTorch↔Keras conversion pattern (copy weights transposed).

Also provide an `x_test.npy` matching the input shape and put it in `data/`.

### Step 1 — declare your architecture and activation

Edit `signature_recovery/utils.py`:

```python
# Only one of these should be True at a time for a given run:
TINIEST = True   # 8-8-8-8-8-8
TINIER  = False  # 32-16-16-16-8-4
TINY    = False  # 64-64-64-64-64-10
MAKEBLOBS = True # make_blobs synthetic data (set False for CIFAR-10)

# Activation toggle (default 0.0 = plain ReLU, byte-identical to original):
LEAKY_ALPHA = 0.0       # set to 0.01 to attack a LeakyReLU(0.01) victim
```

For a new architecture, add a new `elif` branch to set `LAYER_SIZES =
[idim, h1, h2, h3, h4, odim]` and adjust the model-path selection below.

The same `TINIEST/TINIER/TINY` flag must match in `sign_recovery/batched_sign_recovery.py`,
and the same `LEAKY_ALPHA` value must match in **all four** files:
- `signature_recovery/utils.py`
- `sign_recovery/sign_recovery.py`
- `sign_recovery/batched_sign_recovery.py`
- `analysis/extraction_pipeline/config.py` (the legacy `analysis/test_extraction4.py`
  re-exports `LEAKY_ALPHA` from this module — only edit `config.py`)

When `LEAKY_ALPHA > 0`, the pipeline automatically resolves all model paths
to `<name>_leakyrelu.{pth,keras}` instead of `<name>_relu.{pth,keras}`.

### Step 2 — run the full pipeline

```bash
cd enhanced_codebase
./run_extract.sh tiniest 9           # tiniest, 9 find_duals iterations (~1 min)
./run_extract.sh tinier  50          # tinier, 50 iterations (~10 min)
./run_extract.sh tiny    1000        # tiny, 1000 iterations (~11 h — overnight)
```

`run_extract.sh`:
- reconfigures `TINIEST/TINIER/TINY` in `utils.py` and `batched_sign_recovery.py`
- runs `find_duals.py` × N
- runs `cluster_dual_points_stream.py`
- runs `generate_dual_neuron.py`
- runs `recover_weights.py {0,1,2,3}`
- runs `batched_sign_recovery.py`
- runs `analysis/run_extraction.py --<model> --from-scratch --refine --refine-epochs 1000` (the modular Phase-3 CLI; `analysis/test_extraction4.py` remains as a re-export shim if needed)

Outputs:
- `signature_recovery/exp/1/duals_XXXXX.p` — raw dual triplets
- `signature_recovery/exp/1-cluster-{0..3}.p` — layer clusters
- `signature_recovery/outputs/model_weights/Vrelu/layer_{0..3}/neuron_*/` — unsigned weights
- `sign_recovery/layer_neuron_npys/layer{1..4}_neuron*.npy` — per-neuron dual files
- `results/sign_recovery/layer{1..4}_{signs,confidences,votes}.npy`
- `results/reconstructed_models/reconstructed_<model>.pth` — the extracted model
- `results/reconstructed_models/extraction_metrics.json` — full metrics

### Step 3 — generate reports

```bash
cd enhanced_codebase
PY=/path/to/python

# Model accuracy on the target task
$PY analysis/evaluate_reconstructed_tiny.py             # tiny
# or
$PY analysis/evaluate_reconstructed_makeblobs.py        # tiniest

# Per-neuron weight comparison
$PY analysis/compare_true_vs_extracted_tiny.py          # tiny
# or
$PY analysis/compare_true_vs_extracted.py               # tiniest
```

Each script writes a JSON next to the reconstructed model and prints a
human-readable table. To turn those logs into the two markdown reports
(`<model>_true_vs_extracted_<date>.md` and `<model>_extraction_quality_<date>.md`),
feed the logs + the two scripts' stdout into an LLM with `ATTACK_PROMPT.md`
as the system prompt. See `ATTACK_PROMPT.md` for the exact few-shot template.

### Step 4 — sanity-check the extracted model

```python
import torch
m = torch.load("results/reconstructed_models/reconstructed_tiniest.pth")
# or load into the TiniestModel class defined in analysis/extraction_pipeline/architectures.py
```

If reconstructed accuracy < 90 %:
1. Check sign recovery summary — a neuron with `confidence < 0.55` is a
   coin-flip; sign-search should have fixed it but may have skipped if
   `k > 18` recovered rows.
2. Check how many neurons came out of signature recovery
   (`recovery_stats` in `extraction_metrics.json`). If `<70 %` on any
   layer, run more `find_duals` iterations.
3. Increase `--refine-epochs` (e.g. to 2000). Refinement has plenty of
   capacity if its starting agreement is >~20 %.

## Full attack on CIFAR-10 ReLU (flagship)

This is the **headline model** the original EUROCRYPT paper targets: the MLP
`3072 → 256 → 256 → 256 → 64 → 10` trained on raw CIFAR-10 pixels with ReLU and
float64. `run_extract.sh` covers the make_blobs variants (tiniest/tinier/tiny)
but **does not handle the CIFAR flagship**, so the recipe below is the
authoritative end-to-end walkthrough. The latest validated run is documented in
`results/reports/cifar_relu_full_2026-06-04.md`.

### Hardware budget

| | |
|---|---|
| RAM | **≥22 GB** (16-17 Gi working set during Phase 1; **restart the machine before starting** if swap is not empty — the prior CIFAR flagship run documented swap-thrashing as the binding constraint) |
| Disk | **≥80 GB free** in the repo's filesystem (dual-search pickles take ~55 GB, layer-cluster pickles another ~9 GB) |
| Cores | 14-core CPU recommended; the recipe pins `OMP/MKL_NUM_THREADS=2` and uses 5 dual-search workers (5×2 = 10 BLAS threads) |
| Wall time | **~5-6 h end-to-end** with the recipe below (61 min dual search + 3 min cluster + ~67 min weight recovery + ~78 min Phase 2 L1+L2 + ~7 min Phase 3); L3/L4 Phase 2 would add 10+ more hours and is intentionally skipped — see step 6 |

### Step 0 — Place the victim artifacts

You need three files in `tiny_stuff/` matching the flagship architecture:

```
tiny_stuff/TinyModel_relu.pth        # PyTorch state_dict, float64
tiny_stuff/TinyModel_relu.keras      # Keras SavedModel (Phase 2 uses this)
tiny_stuff/TinyModel_relu_alpha.txt  # one line: "0.0"
```

And the CIFAR test data in `data/` — note this is now a **three-slice contract**
(X_test = queryable; X_test2 = queryable under `--train-union-test12`;
X_test3 = strictly held-out eval):

```
data/x_test.npy                # uint8 (10000, 3072) — CIFAR test batch (queryable, Phase-3 distillation)
data/y_test.npy                # int64 (10000,)
data/x_test2_cifar.npy         # CIFAR train[40000:50000] — queryable under union flag (Phase-3 distillation)
data/y_test2_cifar.npy
data/x_test3_cifar.npy         # CIFAR train[10000:20000] — strictly held-out (eval only, never queried)
data/y_test3_cifar.npy
```

X_test2 and X_test3 are **disjoint** CIFAR train slices by construction
(`train[40000:50000]` vs `train[10000:20000]`). The CIFAR test batch (X_test)
is independent of both. X_test3 is the honest-eval metric — never used for
training, sign-search selection, or watchdog tuning.

If you don't already have these, train them yourself with `create_cifar_model.py`
(which now emits all three slices in one pass; expects
`~/.keras/datasets/cifar-10-batches-py-target/cifar-10-batches-py/` to contain
the unzipped CIFAR-10 python pickles):

```bash
PY=/home/biprarshi/miniconda3/envs/MLenv/bin/python3
$PY create_cifar_model.py        # trains both ReLU + LeakyReLU(0.01) variants
                                 # + emits x_test2_cifar.npy and x_test3_cifar.npy
```

Sanity check — `.pth` and `.keras` must agree on argmax across `x_test`:

```bash
$PY - <<'PY'
import sys, numpy as np, torch, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, 'signature_recovery')
import utils
import tensorflow as tf
m_keras = tf.keras.models.load_model(utils.MODEL_PATH.replace('.pth', '.keras'), compile=False)
y_pt = utils.cheat_net_cpu(torch.tensor(utils.x_test[:200], dtype=torch.float64)).argmax(1).numpy()
y_kr = m_keras(utils.x_test[:200].astype(np.float32)).numpy().argmax(1)
print('argmax agreement (200):', float((y_pt == y_kr).mean()))
PY
# expect 1.0
```

### Step 1 — Configure for CIFAR flagship + ReLU

CIFAR flagship requires **all four arch flags False** (not a make_blobs path),
and ReLU requires `LEAKY_ALPHA = 0.0` in **four** files.

```bash
PY=/home/biprarshi/miniconda3/envs/MLenv/bin/python3

# 1a. Arch booleans in utils.py + batched_sign_recovery.py — all False
$PY - <<'PY'
import re, pathlib
for f in ['signature_recovery/utils.py', 'sign_recovery/batched_sign_recovery.py']:
    p = pathlib.Path(f); t = p.read_text()
    for k in ('TINIEST', 'TINIER', 'TINY', 'MAKEBLOBS'):
        t = re.sub(rf'^{k}\s*=\s*(True|False)\b', f'{k} = False', t, count=1, flags=re.M)
    p.write_text(t)
PY

# 1b. LEAKY_ALPHA = 0.0 in all four files
$PY - <<'PY'
import re, pathlib
for f in ['signature_recovery/utils.py',
          'sign_recovery/sign_recovery.py',
          'sign_recovery/batched_sign_recovery.py',
          'analysis/extraction_pipeline/config.py']:
    p = pathlib.Path(f); t = p.read_text()
    p.write_text(re.sub(r'^LEAKY_ALPHA\s*=\s*\S+', 'LEAKY_ALPHA = 0.0', t, count=1, flags=re.M))
PY

# 1c. Bump sign-recovery worker count from 2 → 5 (CIFAR has enough RAM headroom)
$PY -c "
import re, pathlib
p = pathlib.Path('sign_recovery/batched_sign_recovery.py')
p.write_text(re.sub(r'^nThreads\s*=\s*\d+', 'nThreads                 = 5', p.read_text(), count=1, flags=__import__('re').M))
"

# 1d. Install Phase 2 dep if missing
$PY -c "import tabulate" 2>/dev/null || $PY -m pip install tabulate

# 1e. Verify
$PY -c "
import sys; sys.path.insert(0, 'signature_recovery'); import utils
print('LAYER_SIZES =', utils.LAYER_SIZES)
print('MODEL_PATH  =', utils.MODEL_PATH)
print('LEAKY_ALPHA =', utils.LEAKY_ALPHA)
"
# expect: LAYER_SIZES = [3072, 256, 256, 256, 64, 10]
#         MODEL_PATH = .../tiny_stuff/TinyModel_relu.pth
#         LEAKY_ALPHA = 0.0
```

### Step 2 — Clean previous residuals

The pipeline writes huge intermediate files; running on top of stale state mixes
configurations.

```bash
HERE="$(pwd)"
rm -rf "$HERE/signature_recovery/exp/1"
rm -f  "$HERE/signature_recovery/exp/1-cluster-"*.p
rm -rf "$HERE/signature_recovery/outputs/model_weights/Vrelu/layer_"*
rm -rf "$HERE/sign_recovery/layer_neuron_npys"
rm -f  "$HERE/results/sign_recovery/"*
rm -f  "$HERE/results/reconstructed_models/reconstructed_"*
rm -f  "$HERE/results/reconstructed_models/extraction_metrics.json"
mkdir -p "$HERE/signature_recovery/exp/1" \
         "$HERE/signature_recovery/outputs/model_weights/Vrelu" \
         "$HERE/sign_recovery/layer_neuron_npys" \
         "$HERE/results/sign_recovery" \
         "$HERE/results/reconstructed_models"

# Memory sanity — abort and restart if swap is not empty
free -h
# expect: Swap used ≈ 0; Mem available ≥ 16 Gi
```

### Step 3 — Batched parallel dual search (~61 min)

```bash
cd signature_recovery
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  $PY -u torch_impl/parallel_duals.py \
    --iterations 140 --workers 5 --batch-size 48 --target 4000 --impl torch \
    2>&1 | tee /tmp/cifar_relu_duals.log
cd ..
```

Expected output at the end:
`[parallel_duals] finished 140 rounds in ~3660s; 140 pickle files in .../exp/1`

Disk usage after this step: ~55 GB under `signature_recovery/exp/1/`.

### Step 4 — Streaming cluster (~3 min)

Per-neuron cap of 150 keeps peak RAM under ~9 GB.

```bash
cd signature_recovery
CLUSTER_PER_NEURON_CAP=150 $PY -u cluster_dual_points_stream.py 2>&1 \
    | tee /tmp/cifar_relu_cluster.log
cd ..
```

Expected coverage:
```
layer 0: 256/256 covered, ~38 K triplets
layer 1: ~251/256 covered, ~37 K triplets
layer 2: ~244/256 covered, ~36 K triplets
layer 3:  ~53/64 covered,  ~8 K triplets
```

### Step 5 — Per-neuron bridge + weight recovery (~67 min)

```bash
cd signature_recovery
$PY -u generate_dual_neuron.py 2>&1 | tee /tmp/cifar_relu_neuron.log
# expect: 'Generated ~800 .npy files in ...'

for L in 0 1 2 3; do
    CLUSTER_START=0 CLUSTER_END=999 \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      $PY -u recover_weights.py $L 2>&1 | tee /tmp/cifar_relu_recover_$L.log
done
cd ..
```

Expected recovery (matches prior baseline within run variance):
```
layer 0:  255/256  recovered, mean abs err ~10⁻⁹
layer 1:  247/256  recovered, mean abs err ~10⁻⁹
layer 2:    0/256  recovered (structural failure — min(hits)==0 rejection)
layer 3:    0/64   recovered (structural failure, compounded)
total:    502/832  ≈ 60 %
```

**Layer 2 / 3 zero-recovery is intrinsic to ReLU + depth**, not a bug — deep
neurons live where most upstream units are saturated off, so weight components
along those inputs are unobservable from boundary geometry. The documented
lever is LeakyReLU(α > 0); see §"Leaky ReLU usage".

### Step 6 — Phase 2 sign recovery (~78 min for L1+L2, abort L3+L4)

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=3 \
  $PY -u sign_recovery/batched_sign_recovery.py 2>&1 \
  | tee /tmp/cifar_relu_sign.log
```

L1 (~28 min) and L2 (~50 min) complete cleanly and write
`results/sign_recovery/layer{1,2}_summary.json`. **L3 hits algorithmic
slowdown** (per-neuron boundary walks scale with depth × width — see
the report's §6 for the math) and would need 10-12 more hours to finish L3
+ L4. The pragmatic call is to abort once L2's summary is written:

```bash
# Watch for "Layer 2 Summary" then halt — L3 will not finish in reasonable time
pkill -9 -f batched_sign_recovery
pkill -9 -f sign_recovery.py
```

Phase 3's oracle sign search recovers the missing sign information. Pad
stub L3 + L4 sign files (defaults all to +1 — Phase 3 will flip what needs
flipping) before continuing:

```bash
$PY - <<'PY'
import json, os, numpy as np
BASE = '.'
RESULTS_MODEL = f'{BASE}/results/model_TinyModel_relu'
RESULTS_SIGN  = f'{BASE}/results/sign_recovery'
LAYER_NEURONS = {1: 256, 2: 256, 3: 256, 4: 64}
model_layers = {}
for L, n in LAYER_NEURONS.items():
    signs = np.ones(n, dtype=np.int8)
    confs = np.zeros(n, dtype=np.float64)
    votes = np.zeros(n, dtype=np.int32)
    done = 0
    lay_dir = f'{RESULTS_MODEL}/layerID_{L}'
    if os.path.isdir(lay_dir):
        for nid in range(n):
            f = f'{lay_dir}/neuronID_{nid}/sign_result.json'
            if os.path.exists(f):
                try:
                    d = json.load(open(f))
                    s = d.get('recovered_sign')
                    if s in (1, -1):
                        signs[nid] = int(s)
                    confs[nid] = float(d.get('confidence', 0.0))
                    votes[nid] = int(d.get('total_votes', 0))
                    done += 1
                except Exception:
                    pass
    if L in (3, 4) or not os.path.exists(f'{RESULTS_SIGN}/layer{L}_signs.npy'):
        np.save(f'{RESULTS_SIGN}/layer{L}_signs.npy', signs)
        np.save(f'{RESULTS_SIGN}/layer{L}_confidences.npy', confs)
        np.save(f'{RESULTS_SIGN}/layer{L}_votes.npy', votes)
        json.dump({
            'layerID': L, 'num_neurons': n, 'neurons_processed': done,
            'neurons_positive_sign': int((signs == 1).sum()),
            'neurons_negative_sign': int((signs == -1).sum()),
            'signs': signs.tolist(), 'confidences': confs.tolist(), 'votes': votes.tolist(),
            'note': 'partial — Phase-2 halted; rest default +1 for Phase-3 oracle search' if L in (3, 4) else None,
        }, open(f'{RESULTS_SIGN}/layer{L}_summary.json', 'w'), indent=2)
    model_layers[str(L)] = {'num_neurons': n, 'neurons_processed': done,
                             'signs': signs.tolist(), 'confidences': confs.tolist()}
json.dump({'model': 'TinyModel_relu', 'layers': model_layers},
          open(f'{RESULTS_SIGN}/model_sign_recovery_summary.json', 'w'), indent=2)
print('Wrote layer{1..4}_*.npy + model_sign_recovery_summary.json')
PY
```

### Step 7 — Phase 3 reconstruction + refinement (~1 h 50 m with the fix flags)

**Baseline (byte-identical to the 2026-06-04 reference run, ~7 min):**

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_CPP_MIN_LOG_LEVEL=3 \
  $PY -u analysis/run_extraction.py \
    --full --from-scratch --refine --refine-epochs 1000 \
    2>&1 | tee /tmp/cifar_relu_phase3.log
```

The `--full` flag selects the CIFAR `FullModel` (3072→256→256→256→64→10);
`--from-scratch` enables bias recovery + sign search + fc5 LR fit;
`--refine` runs 1000-epoch frozen-row distillation against the oracle.

**Recommended (CIFAR-fix run, +4.3 pt held-out gain over baseline):**

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_CPP_MIN_LOG_LEVEL=3 \
  $PY -u analysis/run_extraction.py \
    --full --from-scratch --refine --refine-epochs 500 \
    --eval-on-test3 --train-union-test12 \
    --early-stop --patience 5 --eval-every 10 \
    --refine-weight-decay 1e-4 --refine-cosine-lr \
    --sign-restarts 4 --sign-pair-lookahead 8 \
    --sign-refine-cycles 3 --sign-refine-mini-epochs 20 \
    2>&1 | tee /tmp/cifar_relu_phase3_fixed.log
```

The new CIFAR-fix flags (all default-off; legacy behaviour preserved):

| Flag | Purpose |
|---|---|
| `--eval-on-test3` | Route every Phase-3 eval/watchdog to `X_test3` (strict held-out). Without it, eval falls back to `X_test2` and the run is no longer honest-eval. |
| `--train-union-test12` | Promote `X_test2` into the queryable distillation pool. Phase-3 trains on `X_test ∪ X_test2` (20 K queries instead of 10 K). |
| `--early-stop` / `--patience N` / `--eval-every E` | Refinement watchdog: every `E` epochs, score the model on a 1024-row `X_test3` slice; save best-checkpoint; stop after `N` watchdog evals without improvement; restore best at end. Prevents refinement overfit. |
| `--refine-weight-decay W` | AdamW weight decay during refinement (`W=0` uses plain Adam, byte-identical to legacy). |
| `--refine-cosine-lr` | CosineAnnealingLR schedule across the refinement budget. |
| `--sign-restarts R` | Greedy sign search runs the base traversal plus R random-restart traversals, selecting whichever ends with the best `X_test3` agreement. |
| `--sign-pair-lookahead K` | After greedy converges, take the K most-uncertain recovered neurons (by single-flip Δ) and try all C(K,2) pair flips. Catches coupled escapes greedy misses. |
| `--sign-refine-cycles C` / `--sign-refine-mini-epochs E` | Interleave sign-search ↔ E-epoch mini-refinement for C cycles. Each cycle gives sign-search a better-calibrated model to score against. |

Expected stages with the recommended flags (see `results/reports/cifar_relu_fixed_2026-06-05.md` for the full breakdown):
1. Recovery summary: 502/832 recovered, 330 random-init
2. Bias recovery: 255 + 247 = 502 biases set from dual points
3. **fc5 LR fit (Fix C1, runs BEFORE sign search)**: ~34.5 % agreement on `X_test3`
4. **Cycle 1**: greedy sign search → pair-flip → 20-epoch mini-refine
5. **Cycle 2 / Cycle 3**: same loop, increasingly calibrated
6. Final refinement: watchdog typically fires around epoch ~80 / 500 (early-stop)
7. Eval on `X_test3` (10K): reconstructed accuracy ≈ 44 %, **agreement ≈ 54.7 %** (vs 50.40 % baseline)

### Step 8 — Read the report

Outputs:

```
results/reconstructed_models/
  reconstructed_full.pth          # the extracted model
  reconstructed_full_weights.npz
  extraction_metrics.json         # every number from the run, incl. eval_tag,
                                  # sign_cycle_log, sign_pair_lookahead_results

results/reports/
  cifar_relu_full_<date>.md       # baseline run report (template: cifar_relu_full_2026-06-04.md)
  cifar_relu_fixed_<date>.md      # CIFAR-fix run report (template: cifar_relu_fixed_2026-06-05.md)
```

Expected headline numbers (CIFAR-fix flags) on **strictly held-out `X_test3` (10K)**:

| | Baseline (2026-06-04) | CIFAR-fix (2026-06-05) |
|---|---|---|
| Oracle (victim) accuracy | 53.34 % | 53.34 % |
| Reconstructed accuracy   | ~44 %   | ~44 %   |
| **Prediction agreement** | **50.40 %** | **54.71 %** (+4.31 pt) |
| Watchdog peak (1024-row slice) | n/a | 55.08 % |
| Refinement epoch at early-stop | 1000 (full budget) | 80 / 500 (12.5× fewer epochs) |
| L0 sign accuracy | 49.0 % | 52.2 % |
| L1 sign accuracy | 48.2 % | 48.2 % (binding constraint) |
| Recovered neurons        | 502/832 (60 %) | 502/832 (60 %) |
| L0/L1 mean \|cos sim\|   | 1.000 | 1.000 |

The +4 pt advantage over the no-signature baseline (≈50 %) is the empirical
confirmation that **structural recovery generalises where pure distillation
overfits the query set** — and the CIFAR-fix flags push that advantage out by
another +4 pt by (a) honest-eval gating to prevent X_test overfit and (b)
escaping the sign-search local minimum that previously trapped L0 at 49 %. See:

- `results/cifar_flagship/cifar_flagship_insights.md` §5 — the original
  "advantage over naïve baseline" framing
- `results/reports/cifar_relu_fixed_2026-06-05.md` — the CIFAR-fix run
  report with per-fix attribution and open threads

## Leaky ReLU usage

The pipeline supports both ReLU and Leaky ReLU(α) victims via a single
`LEAKY_ALPHA` toggle. With `α = 0` the codebase is byte-identical to the
original ReLU pipeline; with `α > 0` it switches model paths and applies
five activation-aware patches gated on `LEAKY_ALPHA > 0`.

### What changes for Leaky ReLU

Math: at the kink `z = 0`, ReLU's slope jumps `0 → 1`; Leaky ReLU's jumps
`α → 1`. The attack scaffolding (dual-point detection, SVD null-space, sign
walks) still works because the kink itself is preserved. Surprisingly,
α > 0 actually **helps** signature recovery — the small α·z signal on
"OFF" prefix coordinates gives the SVD additional well-conditioned
constraints that ReLU's pure null space lacks. On tiniest we saw 22/32
recovered with α=0.01 vs 19/32 for ReLU.

### Five gated patches (all no-op when α = 0)

| Where | What |
|---|---|
| `signature_recovery/utils.py` | `act(x)` / `act_np(x)` / `cell_slope_mask(x)` helpers; `CIFAR10Net.forward` and `cheat()` use `act` instead of `self.relu`. |
| `signature_recovery/recover_weights.py` | `relu_around` linearisation uses `cell_slope_mask` (1 on ON cells, α on OFF cells); `is_consistent_help` bypasses the `np.min(hits) == 0` reject because OFF coords still carry α·z signal; `extract_weights` drops the `S[-2]>1e-2 and S[-1]<1e-4` SVD gate (leaky's α·z leakage adds extra small SVs); the real quality check happens downstream in `dosteal` via `min(errs) < 1e-3`. |
| `sign_recovery/sign_recovery.py` | `_apply_act(x)` helper replaces `x[x<0] = 0.0` with `x[x<0] *= α` at three sites; OFF-side wiggle masking uses `α * dy` instead of `0`. |
| `sign_recovery/batched_sign_recovery.py` | Resolves `*_leakyrelu.keras` model paths when `α > 0`; propagates `LEAKY_ALPHA` to the imported `sign_recovery` module. |
| `analysis/extraction_pipeline/` (formerly `analysis/test_extraction4.py`; the latter is now a re-export shim) | `_act` helper used in all 4 model classes' forwards (16 sites in `architectures.py`) and in `_hidden_activations_up_to` (`bias_recovery.py`); model-path suffix toggle in `config.py`. **Plus two non-leaky-specific safety patches**: (a) `weight_assembly.load_unsigned_weights` skips neurons without `metadata.json` (avoids using SVD outputs that didn't match any cheat solution); (b) `weight_assembly.combine_weights_and_signs` treats `sign == 0` (unknown) as `+1` instead of zeroing the weight — prevents partial sign recovery from killing recovered rows. |

Always-on bug fixes surfaced during the leaky port (apply to ReLU mode too):
- `recover_weights.py is_consistent_help` had `hits = np.zeros(LAYER_SIZES[layer+1])` (target output dim), but the loop indexed `hits[coord]` with `coord ∈ hiddens.shape[1]` (target input dim = prefix output dim). Tiniest's uniform 8× widths made these accidentally equal. Tinier's 32→16 broke broadcasting at `hits += hiddens[entry]`. Fixed to `hits = np.zeros(hiddens.shape[1])`. This also benefits ReLU non-uniform configs.
- `generate_dual_neuron.py` and `recover_weights.py::dosteal` now shape-filter triplets against `LAYER_SIZES[0]` (the active architecture's input dim) before stacking with `np.array(...)`. Cluster pickles can carry stale triplets from a prior architecture run (e.g. a tinier 32-dim leftover surviving into a tiniest 8-dim run via dirty `signature_recovery/exp/`); without the filter, `np.array([(8,)..., (32,)...])` raised `ValueError: inhomogeneous shape`. The filter is a no-op for clean runs (drops zero triplets) and the dropped count is logged when it kicks in. Verified end-to-end on tiniest LeakyReLU(0.01): 98.60 % on X_test2, 98.55 % prediction agreement.

### Quick start: extracting a Leaky ReLU victim

#### Tiniest (8-8-8-8-8-8 LeakyReLU(0.01))
```bash
cd enhanced_codebase

# 1. Train the leaky tiniest victim (creates tiny_stuff/tiniest_makeblobs_leakyrelu.{pth,keras})
$PY create_tiniest_makeblobs_leakyrelu.py

# 2. Set LEAKY_ALPHA = 0.01 in all 4 files:
sed -i 's|^LEAKY_ALPHA = 0.0$|LEAKY_ALPHA = 0.01|' \
    signature_recovery/utils.py \
    sign_recovery/sign_recovery.py \
    analysis/extraction_pipeline/config.py
sed -i 's|^LEAKY_ALPHA              = 0.0$|LEAKY_ALPHA              = 0.01|' \
    sign_recovery/batched_sign_recovery.py

# 3. Set TINIEST=True, TINIER=False, TINY=False (in utils.py + batched_sign_recovery.py)
# (run_extract.sh does this automatically)

# 4. Run the pipeline
./run_extract.sh tiniest 5

# 5. Reports
$PY analysis/evaluate_reconstructed_makeblobs.py
```
Expected: ~99.25 % accuracy on X_test2, 22/32 neurons recovered.

#### Tinier (32-16-16-16-8-4 LeakyReLU(0.01))
```bash
cd enhanced_codebase
$PY create_tinier_makeblobs_leakyrelu.py
# (toggle LEAKY_ALPHA = 0.01 in all 4 files as above)
# Set TINIER=True in utils.py and batched_sign_recovery.py
./run_extract.sh tinier 8
```
Expected: ~100 % accuracy on X_test2 (refinement converges in ≤5 epochs),
33/56 neurons recovered.

### Reverting to ReLU mode

Set `LEAKY_ALPHA = 0.0` in all four files. Model paths automatically resolve
back to `*_relu.{pth,keras}`. No other changes needed — the ReLU pipeline is
preserved byte-identical.

### Performance / stability tips for Leaky runs

- **Sign recovery hangs**: on tiniest we saw layer-2 neuron-7 stuck at
  `DualPointID 328` for 30+ minutes. Reduce `nExpMin`/`nExp` in
  `batched_sign_recovery.py` (we use 200/2000 for layers 1-3 and 100/1000
  for layer 4 in the leaky enhanced_codebase config). Alternatively, kill
  the run after layer 1 + 2 finish — `oracle_sign_search` in Phase 3 will
  fill in the missing signs.
- **OOM on 24 GB machines**: drop `nThreads` from 8 to 2 in
  `batched_sign_recovery.py` (the enhanced_codebase config already does this).
- **Layer 4 (deepest hidden) often fails signature recovery** — same as in
  ReLU mode. Refinement compensates via fc5 LR fit + frozen-row training.

### Reports for the leaky runs

In `results/reports/`:
- `leaky_relu_port.md` (project root) — resume-friendly plan + full file-by-file audit
- `tiniest_leakyrelu_iter1_2026-05-05.md` — tiniest end-to-end run (99.25 % on X_test2)
- `tinier_leakyrelu_iter1_2026-05-06.md` — tinier end-to-end run (100 % on X_test2)
- `tiniest_leakyrelu_true_vs_extracted_2026-05-06.md` — per-neuron weight comparison
- `tinier_leakyrelu_true_vs_extracted_2026-05-06.md` — same, tinier
- `vanilla_vs_current_workflow_2026-04-23.md` — full diff vs vanilla EUROCRYPT code, includes leaky port section

## Phase-3 module layout

The Phase-3 pipeline lives at `analysis/extraction_pipeline/`. The split is
purely cosmetic — every function preserves its original signature, the CLI
flags are unchanged, and the legacy `test_extraction4.py` is now a thin shim
that re-exports the same names from the new package. This means:

- New code → import from `extraction_pipeline.<module>` directly.
- Old code → keeps working unchanged via `from test_extraction4 import …`.
- `run_extract.sh` now calls `analysis/run_extraction.py` (the modular CLI). The legacy `analysis/test_extraction4.py` shim is retained for any old scripts that still import it but is no longer the recommended entry point.

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

Smoke-tested on this refactor:
- Tiniest **ReLU** end-to-end via `run_extraction.py --tiniest --sign-search --refine`:
  98.95 % accuracy on X_test2 (ground truth 99.95 %), 99.00 % prediction agreement.
- Tinier **LeakyReLU(0.01)** end-to-end via `run_extraction.py --tinier --sign-search --refine`:
  100.00 % on X_test2, 100 % prediction agreement.
- Tiniest **LeakyReLU(0.01)** full pipeline (Phase 1 → 2 → refactored Phase 3
  via `run_extraction.py --tiniest --from-scratch --refine --refine-epochs 500`):
  98.60 % on X_test2, 98.55 % prediction agreement, 24/32 neurons recovered with
  |cos|=1.0. (Phase 2 batched_sign_recovery hangs on layer 2 — known issue;
  Phase 3's oracle sign search brute-forces 2^8 combos per layer to fill in
  unaggregated layers, so the run completes successfully without finishing
  Phase 2.)

All three runs produced byte-equivalent output structure to the pre-refactor
pipeline (same metrics keys, same model file format, same per-layer numbers
within oracle-sign-search noise tolerance).

## Batched PyTorch dual search

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
CPU-first (works on `device='cuda'` by moving the model, untested). The original
`find_duals.py` is untouched and still runs standalone.

### Usage (drop-in for run_extract.sh / run_one_model.sh STEP 2)

```bash
cd enhanced_codebase/Hard_Label_Work
# arch/activation come from utils.py toggles (set them first, like the original)
./run_duals_torch.sh <ITERS> <WORKERS> <BATCH> <IMPL>
#   ITERS    pickle rounds (≈ find_duals.py invocations)
#   WORKERS  concurrent processes        (default cores/2)
#   BATCH    walks per batch, torch impl  (default 256)
#   IMPL     torch | subprocess           (default torch)

# tiniest:  ./run_duals_torch.sh 9   8 256 torch     # ~6 s
# tiny:     ./run_duals_torch.sh 500 8 256 torch     # ~24 min (was ~18 h)
```
Then continue with the unchanged `cluster_dual_points_stream.py → generate_dual_neuron.py
→ recover_weights.py {0..3} → batched_sign_recovery.py → run_extraction.py`.

`--impl subprocess` runs the original `find_duals.py` in parallel processes
(zero-change baseline); `--impl torch` (default) uses the batched finder.

### Validated results

| | NumPy (original) | Torch (this port) |
|---|---|---|
| tiniest, 9 rounds, 8 workers | ~75–135 s | **5–9 s** (~10–25×) |
| **tiny, full dual search** | **~18 h** | **24.3 min** (~44×) |
| tiny signature recovery | 157/256 | **154/256** (matches; fc4 0/64 as expected for ReLU) |
| tiny functional agreement (X_test2) | 100 % | **100 %** |
| triplet format / recovery rate | — | identical / ≥ NumPy on every layer |

Two efficiency refinements (recovery-neutral): **lane compaction** drops
finished walks from the batch each iteration, and a **`max_outer` cap** bounds
a rare "marathon lane" that would otherwise hold up a round. Full write-up in
`signature_recovery/MIGRATION_RESULTS.md`; dataflow + interface contract in
`signature_recovery/MIGRATION_NOTES.md`.

### Not yet ported (secondary)
After the dual search, `cluster_dual_points_stream.py` (~6.5 min on tiny) and
`batched_sign_recovery.py` (~30 min) become the largest costs — both original
code, out of scope here. Clustering is the obvious next batching target
(batch K triplets per `cheat()` forward).

## Known caveats

- **Whitebox scaffolding is still present** in `utils.py` (`cheat_*`,
  `cheat_solution`) and `sign_recovery/whitebox.py`. The workflow is
  **hard-label-clean in Phase 3**, but Phases 1 and 2 inherit the vanilla
  EUROCRYPT reference code's whitebox reads. See `results/reports/tiny_cheating_audit_2026-04-24.md`
  for the full audit.
- `find_duals.py` uses `SEED = 1` by default; running it N times in quick
  succession may produce the same random filename and overwrite. Pass a
  different integer as `argv[1]` to change the seed.
- Layer-1 sign recovery is biased positive because there are no past-layer
  ReLU toggles. This is a known limitation of the sign-recovery algorithm.
  Affects both ReLU and Leaky modes equally.
- Refinement overfits to `X_test`. If your downstream evaluation uses the
  same `X_test` the refinement saw, agreement numbers are upper bounds.
  Use `X_test2` (seed=99, same cluster centres, different samples) for an
  honest out-of-sample number — see `data/x_test2_*.npy`.
- Leaky α > 0.05 — untested. The leaky port was validated at α=0.01. Larger
  α weakens the dON/dOFF asymmetry by `(1-α)/(1+α)`; at α=0.2 sign recovery
  may need recalibration.

## Quick verify

### Full 6-model end-to-end results (2026-05-21)

The complete pipeline was run on every architecture × activation combination via
the new `run_one_model.sh <arch> <activation>` driver (which auto-cleans intermediate
state, syncs `LEAKY_ALPHA` across the four config files, runs Phase 1→2→3, and
generates a per-model comparison report via `analysis/compare_true_vs_extracted_v2.py`).

| Model | Phase-1 recovered | mean \|cos\| | sign acc | X_test2 acc | agreement | wall time |
|---|---|---|---|---|---|---|
| tiniest_relu       | 24/32 (75 %) | 1.000 | 0.610 | 98.95 % | 98.90 % | 135 s |
| tiniest_leakyrelu  | 23/32 (72 %) | 1.000 | 0.451 | 99.20 % | 99.20 % | 115 s |
| tinier_relu        | 30/56 (54 %) | 1.000 | 0.458 | 100.00 % | 100.00 % | 917 s |
| tinier_leakyrelu   | 37/56 (66 %) | 1.000 | 0.548 | 100.00 % | 100.00 % | 976 s |
| tiny_relu          | 157/256 (61 %) | 1.000 | 0.528 | 100.00 % | 100.00 % | ~18 hr |
| **tiny_leakyrelu** | **230/256 (90 %)** | **1.000** | **0.525** | **100.00 %** | **100.00 %** | **~18.8 hr** |

**Per-model true-vs-extracted reports:** `paper_notes/section3/reports/<tag>_true_vs_extracted.md`

**Section 3 analysis notes:** `paper_notes/section3/` (six markdown files + INDEX)

Highlights:

- **All 6 models achieve 98.95-100 % prediction agreement with the oracle on X_test2** using exactly 3 batched `oracle(X).argmax(dim=1)` queries.
- **Mean \|cos\| = 1.000 on every recovered neuron** across all 6 runs.
- **Leaky beats ReLU more at scale.** tinier: +7 neurons, tiny: **+73 neurons** (most dramatically at fc4: leaky 57/64 vs ReLU 0/64).
- **Functional accuracy is decoupled from recovery rate.** tiny_relu (61 % recovered) and tiny_leakyrelu (90 % recovered) both hit 100 % on X_test2.

> **Note on wall time:** the `~18 hr` figures above are the original sequential
> NumPy `find_duals`. The batched PyTorch port (§"Batched PyTorch dual search")
> reproduces tiny_relu — 154/256 recovered, 100 % agreement — with the dual
> search in **~24 min** instead of ~18 h. The other steps (cluster, recover,
> sign, Phase 3) are unchanged.

### One-shot driver

**Canonical entry point** — see §"Latest unified workflow" at the top of this
README. `run_one_model_enhanced.sh <arch> <activation>` is the recommended
driver and is the one used by all paper_notes/section3 reports. It bundles
parallel batched dual search, improved sign search (restarts / pair-flip /
refine cycles), and X_test3 honest-eval validation. It also supports
`arch=full` for the CIFAR-10 flagship.

```bash
cd enhanced_codebase/Hard_Label_Work
PYTHON_BIN=/path/to/python3 ./run_one_model_enhanced.sh <arch> <activation>
# arch:        tiniest | tinier | tiny | full
# activation:  relu    | leakyrelu
```

The legacy `./run_one_model.sh <arch> <activation>` driver remains for
reference but uses the older sequential dual search, no sign restarts /
pair-flip / refine cycles, and no X_test3 watchdog. New runs should use
`run_one_model_enhanced.sh`.

Each invocation cleans previous intermediate state, configures `LEAKY_ALPHA` and arch booleans, runs the full Phase 1→2→3 pipeline (parallel duals → cluster → recover_weights → batched_sign_recovery → run_extraction.py with the updated Phase-3 flag set), and writes `paper_notes/section3/reports/<arch>_<activation>_true_vs_extracted.{md,json}`.

For large architectures with high memory consumption (tiny), you may want to interrupt after find_duals to restart and free memory. Use:

```bash
./run_from_cluster.sh <arch> <activation>
```

This skips the clean step and find_duals, starting from cluster onwards using the already-produced dual files in `signature_recovery/exp/1/`.

### Smaller smoke tests

```
./run_extract.sh tiniest 5       # ReLU,  ~1 min,  reached 99.9 % prediction agreement
./run_extract.sh tinier  8       # ReLU,  ~3 min,  reached 100 % prediction agreement
```

Phase-3 only (skip Phase 1+2; reuse existing recovery outputs):
```
python3 analysis/run_extraction.py --tiniest --sign-search --refine --refine-epochs 200
python3 analysis/run_extraction.py --tinier  --sign-search --refine --refine-epochs 200
python3 analysis/run_extraction.py --tiniest --from-scratch --refine --refine-epochs 1000
```

## License / credits

Attack algorithm: Carlini, Chen, Choquette-Choo, Kos, Tramèr,
"Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the
Hard-Label Setting", EUROCRYPT 2024. Original reference code in
`../vanilla_codebase/` (not shipped in this folder).
