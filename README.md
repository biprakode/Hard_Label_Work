# Enhanced hard-label DNN extraction codebase

Self-contained fork of the EUROCRYPT-2024 "Polynomial Time Cryptanalytic
Extraction of DNNs in the Hard-Label Setting" reference code, with four
additions:

1. **Streaming clustering** (`cluster_dual_points_stream.py`) that processes
   the 10M+ triplet corpus in one memory-bounded pass (was OOMing the
   vanilla `cluster_dual_points.py`).
2. **Phase 3 reconstruction** (`analysis/test_extraction4.py`) — a hard-label
   post-processing stage that takes Phases 1+2 outputs, solves for biases
   geometrically from dual points, brute-force / greedy sign-searches against
   oracle argmax, LR-fits fc5 on oracle hard labels, and polishes with a
   frozen-row cross-entropy refinement loop. Closes the gap from ~8 % to
   99–100 % functional agreement.
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

## What is in this folder

```
enhanced_codebase/
├── README.md                      # this file
├── ATTACK_PROMPT.md               # few-shot LLM prompt: logs -> 2 reports
├── leaky_relu_port.md             # leaky-port plan, status, all 5 gated patches + 1 always-on fix
├── run_extract.sh                 # one-shot: duals -> cluster -> recover -> sign -> reconstruct
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
│   └── run_duals.sh               # bash loop: for i in 1..1000: python find_duals.py
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
│   ├── test_extraction4.py        # main: load signature+signs, bias-recover, sign-search,
│   │                              # fc5 LR fit, oracle-label refinement, save reconstructed_*.pth
│   │                              # *** LEAKY_ALPHA toggle + _act helper, 16 model-class call sites ***
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

### Phase 3 — reconstruction (`analysis/test_extraction4.py`)

The single file `analysis/test_extraction4.py` orchestrates:

| Function | Purpose | Oracle interaction |
|---|---|---|
| `load_unsigned_weights` | Load `neuron_{id}/weights_unscaled.npz`, apply `abs(scaling_factor)` to kill the sign leak. Returns `(W, recovered_mask)`. | none |
| `load_signs` | Load `layer{L}_signs.npy` and merge with `W` → signed weight matrix. | none |
| `reconstruct_model` | Build `TinyModel`/`TinyModelReLU`/etc, fill each layer with `signed_weights`, Kaiming-init the unrecovered rows. | none |
| `recover_biases_from_duals` | For each recovered neuron *i* in layer L: `b_i = -median(w_i · h_{L-1}(x_d))` over 30 dual points. Bottom-up. | none — uses reconstructed forward, not oracle |
| `oracle_sign_search` | Per layer with ≤18 recovered neurons, enumerate 2^k sign flips (+ joint bias flip via `b_i = -w_i · h`), pick flip combo maximising hard-label agreement with `oracle(X_test).argmax`. | hard-label only |
| `recover_output_layer` | fc5 LR fit: forward `X_test` through reconstructed fc1..fc4 → features `h_4`; query `oracle(X_test).argmax`; fit multinomial logistic regression from `h_4` to those labels; overwrite `fc5.{weight,bias}`. | hard-label only |
| `oracle_label_refinement` | 1000 epochs Adam cross-entropy against `oracle(X_test).argmax` labels. **Freezes** rows whose `recovered_mask[i]` is True (zeroes their gradient each step). Leaves biases, fc5, and random-init rows trainable. | hard-label only |

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
- `analysis/test_extraction4.py`

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
- runs `analysis/test_extraction4.py --<model> --from-scratch --refine --refine-epochs 1000`

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
# or load into the TiniestModel class defined in analysis/test_extraction4.py
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
| `analysis/test_extraction4.py` | `_act` helper used in all 4 model classes' forwards (16 sites) and in `_hidden_activations_up_to`; model paths suffix toggle. **Plus two non-leaky-specific safety patches**: (a) `load_unsigned_weights` skips neurons without `metadata.json` (avoids using SVD outputs that didn't match any cheat solution); (b) `combine_weights_and_signs` treats `sign == 0` (unknown) as `+1` instead of zeroing the weight — prevents partial sign recovery from killing recovered rows. |

One always-on bug fix surfaced during the leaky port:
- `recover_weights.py is_consistent_help` had `hits = np.zeros(LAYER_SIZES[layer+1])` (target output dim), but the loop indexed `hits[coord]` with `coord ∈ hiddens.shape[1]` (target input dim = prefix output dim). Tiniest's uniform 8× widths made these accidentally equal. Tinier's 32→16 broke broadcasting at `hits += hiddens[entry]`. Fixed to `hits = np.zeros(hiddens.shape[1])`. This also benefits ReLU non-uniform configs.

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
    analysis/test_extraction4.py
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

The codebase was smoke-tested end-to-end on tiniest and tinier:
```
./run_extract.sh tiniest 5       # ReLU,  ~1 min,  reached 99.9 % prediction agreement
./run_extract.sh tinier  8       # ReLU,  ~3 min,  reached 100 % prediction agreement

# Leaky — set LEAKY_ALPHA=0.01 in all 4 files first:
./run_extract.sh tiniest 5       # Leaky, ~5 min,  99.25 % on X_test2
./run_extract.sh tinier  8       # Leaky, ~8 min,  100 %   on X_test2
```

## License / credits

Attack algorithm: Carlini, Chen, Choquette-Choo, Kos, Tramèr,
"Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the
Hard-Label Setting", EUROCRYPT 2024. Original reference code in
`../vanilla_codebase/` (not shipped in this folder).
