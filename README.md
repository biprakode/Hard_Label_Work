# Enhanced hard-label DNN extraction codebase

Self-contained fork of the EUROCRYPT-2024 "Polynomial Time Cryptanalytic
Extraction of DNNs in the Hard-Label Setting" reference code, with three
additions:

1. **Streaming clustering** (`cluster_dual_points_stream.py`) that processes
   the 10M+ triplet corpus in one memory-bounded pass (was OOMing the
   vanilla `cluster_dual_points.py`).
2. **Phase 3 reconstruction** (`analysis/test_extraction4.py`) — a hard-label
   post-processing stage that takes Phases 1+2 outputs, solves for biases
   geometrically from dual points, brute-force sign-searches against oracle
   argmax, LR-fits fc5 on oracle hard labels, and polishes with a frozen-row
   cross-entropy refinement loop. Closes the gap from ~8 % to 99-100 %
   functional agreement.
3. **Per-model smoke scripts** — `run_extract.sh` + `evaluate_*` +
   `compare_true_vs_extracted*` so an end-to-end run produces both a
   reconstructed `.pth` and the two written reports (true-vs-extracted
   and extraction-quality).

## What is in this folder

```
enhanced_codebase/
├── README.md                      # this file
├── ATTACK_PROMPT.md               # few-shot LLM prompt: logs -> 2 reports
├── run_extract.sh                 # one-shot: duals -> cluster -> recover -> sign -> reconstruct
│
├── signature_recovery/            # Phase 1 — extract weight directions + magnitudes
│   ├── utils.py                   # single source of truth: LAYER_SIZES, model path, x_test path
│   │                              # contains cheat_net_{cpu,cuda} (whitebox scaffolding, DEBUG-gated)
│   ├── find_duals.py              # decision-boundary walker → pickle of (left, middle, right) triplets
│   ├── cluster_dual_points_stream.py   # streaming, memory-bounded clustering  (USE THIS)
│   ├── cluster_dual_points.py     # original (loads everything in RAM — OOMs on tiny+)
│   ├── generate_dual_neuron.py    # cluster pickles → layer{L}_neuron{i}.npy per-neuron files
│   ├── recover_weights.py         # per-layer SVD null-space → unsigned weight rows
│   └── run_duals.sh               # bash loop: for i in 1..1000: python find_duals.py
│
├── sign_recovery/                 # Phase 2 — recover signs via decision-boundary statistics
│   ├── sign_recovery.py           # per-neuron sign via d_on vs d_off walks
│   ├── batched_sign_recovery.py   # parallel runner over all neurons, per-layer aggregation
│   ├── whitebox.py                # reads true weights of the keras model (inherited scaffolding)
│   ├── blackbox.py                # coordinate transforms in affine-layer space
│   └── common.py                  # shared argparse / file-management
│
├── analysis/                      # Phase 3 — reconstruction + evaluation
│   ├── test_extraction4.py        # main: load signature+signs, bias-recover, sign-search,
│   │                              # fc5 LR fit, oracle-label refinement, save reconstructed_*.pth
│   ├── compare_true_vs_extracted.py       # tiniest (8-8-8-8-8-8) per-neuron weight comparison
│   ├── compare_true_vs_extracted_tiny.py  # tiny (64x5->10) per-neuron weight comparison
│   ├── evaluate_reconstructed_makeblobs.py # accuracy/per-class/confusion-matrix on tiniest
│   └── evaluate_reconstructed_tiny.py     # same, for tiny
│
├── tiny_shit/                     # oracle models used as the attack target
│   ├── tiniest_makeblobs_relu.{pth,keras}   # 8-8-8-8-8-8 make_blobs
│   └── makeblobs_relu.{pth,keras}           # 64x5->10 make_blobs
│
├── data/                          # test data (x_test) used for sign-search / refine / eval
│   ├── x_test_tiniest_makeblobs.npy, y_test_tiniest_makeblobs.npy
│   └── x_test_makeblobs.npy, y_test_makeblobs.npy
│
└── results/                       # pipeline outputs land here
    ├── reports/                   # example reports from the 64x5 tiny run
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

### Step 1 — declare your architecture

Edit `signature_recovery/utils.py`:

```python
# Only one of these should be True at a time for a given run:
TINIEST = True   # 8-8-8-8-8-8
TINIER  = False  # 32-16-16-16-8-4
TINY    = False  # 64-64-64-64-64-10
MAKEBLOBS = True # make_blobs synthetic data (set False for CIFAR-10)
```

For a new architecture, add a new `elif` branch to set `LAYER_SIZES =
[idim, h1, h2, h3, h4, odim]` and adjust the model-path selection below.

Do the same flag edit in `sign_recovery/batched_sign_recovery.py` (top of
file).

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
- Refinement overfits to `X_test`. If your downstream evaluation uses the
  same `X_test` the refinement saw, agreement numbers are upper bounds.

## Quick verify

The codebase was smoke-tested end-to-end on tiniest in this repo:
```
./run_extract.sh tiniest 5       # ~1 minute, reached 99.9 % prediction agreement
```

## License / credits

Attack algorithm: Carlini, Chen, Choquette-Choo, Kos, Tramèr,
"Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the
Hard-Label Setting", EUROCRYPT 2024. Original reference code in
`../vanilla_codebase/` (not shipped in this folder).
