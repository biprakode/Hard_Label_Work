# Enhanced hard-label DNN extraction — launch guide

Technical guide to **running the attack**: the driver scripts, every flag/option,
per-architecture parameters, the sign-search options, and expected results.

This is a self-contained fork of the EUROCRYPT-2024 "Polynomial Time
Cryptanalytic Extraction of DNNs in the Hard-Label Setting" reference code, with
a parallel dual search, a hard-label Phase-3 reconstruction stage (bias recovery
+ metaheuristic sign search + fc5 LR fit + frozen-row refinement), Leaky-ReLU
support, and an improved evaluation scorecard.

➡ **For how the codebase works** (the three phases, file-by-file map, Leaky-ReLU
internals, Phase-3 module layout, caveats) see **[EXPLANATIONS.md](EXPLANATIONS.md)**.

---

## Cheating ablation study (read this first if you're reviewing)

Phase 1 (signature recovery) and Phase 2 (statistical sign recovery) each
contain a handful of points that read the true victim model's weights,
biases, or activations directly, instead of relying purely on hard-label
oracle queries — necessary engineering shortcuts for tractable experimentation,
documented and individually ablated (ON = cheat active / OFF = honest
hard-label-only replacement) across six `make_blobs` victims.

➡ **[cheating_ablation/REPRODUCE.md](cheating_ablation/REPRODUCE.md)** is the
single entry point: what each cheat is, why it exists, exact reproduction
commands, and where the measured impact of removing each one is written up.

---

## Prereqs

- Python 3.11+, env with `torch`, `tensorflow`/`keras`, `numpy`, `scipy`,
  `scikit-learn`, `pandas`, `tabulate`. Point the scripts at it with
  `export PYTHON_BIN=/path/to/python3` (or rely on `python3` in PATH).
- Free RAM: ≥4 GB (tiniest), ≥8 GB (tinier), ≥20 GB (tiny / full).
- Disk: ≥80 GB free for `full` (Phase-1 dual pickles alone reach ~55 GB).
- Victim artefacts in `tiny_stuff/` (`<name>_{relu,leakyrelu}.{pth,keras}`).
  CIFAR victims via `python3 create_cifar_model.py`; make_blobs victims via
  `create_*_makeblobs_*.py`.
- Test slices in `data/` — three per arch: `x_test*`, `x_test2_*`, `x_test3_*`
  (X_test3 is held-out, never queried). Emit make_blobs X_test3 via
  `python3 emit_test3_makeblobs.py` if missing.

---

## Canonical entry point: `run_one_model_enhanced.sh`

`run_one_model_enhanced.sh <arch> <activation>` runs the **complete pipeline
end-to-end** for any of the 8 supported configurations (6 make_blobs tiny models
+ 2 CIFAR-10 flagships). It bundles: parallel batched dual search → streaming
cluster → per-neuron bridge → weight recovery → Phase-2 sign recovery → Phase-3
reconstruct+refine → report → improved eval scorecard.

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
# CANONICAL: the exact GCE V100 attack is run_cifar_kaggle.sh (SA+margin, retuned
# defaults) — NOT run_one_model_enhanced.sh. Needs an fp64 GPU + ~180 GB scratch.
bash run_cifar_kaggle.sh relu                       # relu victim (workers=1 on GPU)
DUAL_WORKERS=3 bash run_cifar_kaggle.sh leakyrelu   # leaky victim (3 CUDA contexts)
```

> The two CIFAR flagship rows in the [Attack-parameters table](#attack-parameters-validated-against-the-2026-06-21-make_blobs-runs)
> are read straight from these runs' `extraction_metrics.json`; see
> [Reproduce the canonical CIFAR attack](#reproduce-the-canonical-cifar-gce-attack).

Key flags it sets internally (all overridable by env / the relevant CLI flag):

- **Parallel batched dual search** — `parallel_duals.py --impl torch`, `W` workers in lockstep.
- **MetaHeuristic / combinatorial sign search** — Phase-3 sign step defaults to **SA + margin**
  (`SIGN_METHOD=sa SIGN_OBJ=margin`); set `SIGN_METHOD=pt` for the widest layers
  (CIFAR `full`), `SIGN_METHOD=greedy SIGN_OBJ=agree` for legacy. Full menu:
  [Sign-search methods](#sign-search-methods-metaheuristic-sign-search). Supporting
  knobs: `--sign-restarts R`, `--sign-pair-lookahead 8`, `--sign-refine-cycles 3`.
- **X_test3 honest-eval** — `--eval-on-test3 --train-union-test12`: queryable pool is
  `X_test ∪ X_test2` (20 K), while X_test3 is never queried / never used for sign-flip
  selection / never used for watchdog tuning.
- **Watchdog early-stop** — `--early-stop --patience 5 --eval-every 10` (1024-row X_test3 slice).
- **AdamW + CosineAnnealingLR** — `--refine-weight-decay 1e-4 --refine-cosine-lr`.
- **Improved eval scorecard (step 9)** — `evaluate_extraction_quality.py` runs automatically,
  auto-builds the distillation baseline, auto-detects activation. See [Step 9](#improved-evaluation-step-9).

### Per-arch tuning (set inside the driver)

| Arch | DUAL_ITERS | DUAL workers / batch | SIGN_RESTARTS | SIGN_PAIR | SIGN_CYCLES | REFINE_EPOCHS |
|---|---|---|---|---|---|---|
| `tiniest` | 6  | 7 / 256 | 1 | 8 | 3 | 300 |
| `tinier`  | 8  | 7 / 256 | 1 | 8 | 3 | 500 |
| `tiny`    | 20 | 7 / 256 | 2 | 8 | 3 | 500 |
| `full`    | 80 | 5 / 48  | 4 | 8 | 3 | 500 |

Each `DUAL_ITERS` round emits the per-arch TARGET triplet count
(tiniest=3000, tinier=2000, tiny=10000, full=10000). Override the round count at
the CLI: `./run_one_model_enhanced.sh tiny relu 50`.

### Attack parameters (validated against the 2026-06-21 make_blobs runs)

Canonical end-to-end parameters. The make_blobs columns are hardcoded in
`run_one_model_enhanced.sh` and cross-checked against the per-run
`extraction_metrics.json` under `paper_notes/section3/reports/2026-06-21/`.

**†** The **`full (CIFAR)` column is the canonical GCE V100 run** (not
`run_one_model_enhanced.sh` — the flagship uses `run_cifar_kaggle.sh`, whose
defaults were retuned to these values). Numbers are read back from the
downloaded `extraction_metrics.json` of the two canonical runs:
`paper_notes/section3/reports/cifar_gce_relu_2026-06-25/relu/relu_extraction_metrics.json`
(cifar_relu, GCE V100, 2026-06-25) and
`paper_notes/section3/reports/cifar_kaggle_2026-06-26/full_leakyrelu_extraction_metrics.json`
(cifar_leakyrelu, GCE V100, 2026-06-26). Kaggle-era CIFAR runs are **not**
canonical.

| Phase | Parameter | tiniest | tinier | tiny | full (CIFAR) † |
|---|---|---|---|---|---|
| **1 — dual search** | `DUAL_ITERS` (rounds) | 6 | 8 | 20 | 150 (`DUAL_CHUNK=20`, cluster+trim per chunk) |
| | workers / batch | 7 / 256 | 7 / 256 | 7 / 256 | 1 / 256 (relu) · 3 / 256 (leaky) — GPU one CUDA context |
| | implementation | torch (`parallel_duals.py --impl torch`) | ← | ← | ← (CUDA float64, V100) |
| | TARGET triplets / round | 3000 | 2000 | 10000 | 10000 |
| **2 — sign recovery** | runner | `batched_sign_recovery.py` (float64) | ← | ← | ← (TF thread-capped to 1/worker) |
| **3 — reconstruction** | sign-search method | SA+margin (default); PT+margin (A/B arm) | ← | ← | SA+margin (PT intractable on the 256-wide arch) |
| | `--sign-restarts` | 1 | 1 | 2 | 0 (relu) / 1 (leaky) |
| | `--sign-pair-lookahead` | 8 | 8 | 8 | 4 |
| | `--sign-refine-cycles` | 3 | 3 | 3 | 1 |
| | mini-refine (epochs / lr / wd) | 20 / 5e-3 / 1e-4 | ← | ← | ← |
| | `--refine-epochs` | 300 | 500 | 500 | 500 |
| | refine optimiser | AdamW, `--refine-weight-decay 1e-4`, `--refine-cosine-lr` | ← | ← | ← (lr 5e-3) |
| | watchdog | `--early-stop --patience 5 --eval-every 10` | ← | ← | ← |
| | eval gating | `--eval-on-test3` + `--train-union-test12` (queryable = X_test ∪ X_test2 = 20 K) | ← | ← | ← |
| **activation** | `LEAKY_ALPHA` | 0.0 (ReLU) / 0.01 (Leaky) | ← | ← | ← |
| **oracle cost** | batched `argmax` queries / model | 3 (cached; sign-search + fc5 LR fit + refinement) | ← | ← | ← |

The make_blobs rows are reproduced by `run_makeblobs_batch_2026-06-21.sh`
(ARM A = SA+margin full pipeline, ARM B = PT+margin Phase-3 re-run on ARM A's
on-disk artifacts). The CIFAR `full` column is the **canonical GCE V100 attack**
— fire it with `bash run_cifar_kaggle.sh relu` / `DUAL_WORKERS=3 bash
run_cifar_kaggle.sh leakyrelu` (the live script's defaults already encode every
`full` value above; see [Reproduce the canonical CIFAR attack](#reproduce-the-canonical-cifar-gce-attack)).
The manual CIFAR walkthrough further below is an older 2026-06-04/05 Kaggle run
and is **not** the canonical source for these parameters.

### What the driver does per invocation

1. **STEP 0** — clean all Phase 1+2+3 residuals.
2. **STEP 1** — sync `LEAKY_ALPHA` (0.0 / 0.01) and the four arch booleans across
   the four config files.
3. **STEP 2** — Phase-1 batched dual search (`parallel_duals.py --impl torch`).
4. **STEP 3** — streaming cluster (`cluster_dual_points_stream.py`).
5. **STEP 4** — per-neuron bridge (`generate_dual_neuron.py`) + weight recovery
   (`recover_weights.py {0..3}`).
6. **STEP 5** — Phase-2 sign recovery (`batched_sign_recovery.py`).
7. **STEP 6** — Phase-3 reconstruction:
   ```
   analysis/run_extraction.py --<arch> --from-scratch --refine \
     --refine-epochs $REFINE_EPOCHS --refine-weight-decay 1e-4 --refine-cosine-lr \
     --early-stop --patience 5 --eval-every 10 \
     --eval-on-test3 --train-union-test12 \
     --sign-restarts $SIGN_RESTARTS --sign-pair-lookahead $SIGN_PAIR \
     --sign-refine-cycles $SIGN_CYCLES \
     --sign-search-method $SIGN_METHOD --sign-search-objective $SIGN_OBJ
   ```
8. **STEP 7** — per-model true-vs-extracted report under
   `paper_notes/section3/reports/<arch>_<activation>_true_vs_extracted.{md,json}`.

### Distillation baseline (CIFAR only)

After the `full` extraction completes, the no-signature distillation baseline
runs the same Phase 3 with all 832 hidden rows Kaiming-initialised and trainable
(`--refine-unfreeze`) — an apples-to-apples "with-signature vs without-signature"
comparison on the same queryable pool and X_test3 held-out eval:

```bash
./run_distillation_baseline.sh
# writes paper_notes/section3/reports/cifar_<activation>_distillation.md
```

### Per-arch headline numbers

See `paper_notes/section3/reports/` for the full per-model true-vs-extracted
markdown reports (one per `<arch>_<activation>`), incl. sign-cycle log,
pair-lookahead results, watchdog peak, eval-tag, per-layer cos-sim / sign acc.
Latest verified end-to-end numbers are in [Results](#results) below.

---

## Sign-search methods (MetaHeuristic Sign Search)

Phase-3 fixes Phase-2's noisy/near-chance signs by treating each layer's sign
assignment as a search that maximises agreement against the **cached**
`oracle(X_test).argmax` vector. **Every candidate sign vector is scored by a
forward pass of the *reconstructed* model, not an oracle query — so the entire
sign search (any method, any number of candidates) costs exactly ONE oracle call
per invocation.** That is what makes heavier combinatorial search free.

Two regimes and four methods:

- **`k ≤ 18` recovered neurons (small layers):** exact **brute-force** — enumerate
  all `2^k` sign combos + joint bias flip, pick the best. Automatic, method-independent.
- **`k > 18` (wide layers, e.g. CIFAR `full`'s 256-wide fc1–fc3):** `2^256` is
  infeasible, so a per-layer optimiser is selected by `--sign-search-method`:

| `--sign-search-method` | What it does | Cost (per run) | When to use |
|---|---|---|---|
| `greedy` | Legacy single-flip: flip→test→keep-if-better→revert, multi-pass. **Cannot** escape pairwise/k-flip-coupled local optima. | cheapest | legacy baseline / A-B control only |
| `tabu` | Best-move-by-`opt_score` each sweep even if worsening, with a tabu tenure + aspiration. Only beats greedy **with `margin`**. | ~greedy×sweeps | rarely the best pick |
| `sa` | **Simulated annealing** — accept worsening flips w.p. `exp(Δ/T)` on a cooling schedule; reaches pair/k-coupled configs greedy can't. **Reliable win; recommended default.** | ≈4–5k fwd / ~16 s | **default** for all victims |
| `pt` | **Parallel tempering** — M replicas on a temperature ladder with replica-exchange swaps; highest ceiling, ~7× SA's cost. | ≈19.5k fwd / ~114 s | the widest/hardest **unsaturated** layers (CIFAR `full`) |

Objective (`--sign-search-objective`, ignored for `greedy`):

| value | climb signal | note |
|---|---|---|
| `agree` | raw 0/1 oracle agreement | **flat landscape** — most single flips change no argmax (Δ=0), starving the optimiser of direction |
| `margin` | `mean( logit[oracle] − max_{j≠oracle} logit[j] )` on the **reconstructed model's own logits** | gradient-like guidance through flat regions; still hard-label wrt the victim. **Recommended.** |

**Acceptance and final selection always use true 0/1 agreement**; `margin` only
smooths the *search direction*. Every non-greedy method is **warm-started from
greedy** and **best-true-agreement guarded**, so the result is **≥ greedy by
construction**. A **saturation watchdog** short-circuits the search once
agreement hits 1.0 (skips entirely on already-saturated tiers; ~halves forwards
mid-search with identical sign accuracy), which is why it is safe to default-on.

**Benchmark** (corrupt 50 % of true signs on a 256-wide victim, recover; mean
sign-recovery accuracy over 8 seeds):

| method | mean sign-acc |
|---|---|
| greedy (baseline) | 0.557 |
| SA + agree | 0.569 |
| SA + margin | **0.639** |
| PT + margin | **0.666** |

`SA == PT` on every tier whose reconstruction is already saturated before sign
search (watchdog no-op); PT earns its ~7× cost only where the per-layer search
starts unsaturated (empirically `tiny_relu`). On the CIFAR `full` flagship PT is
theoretically favoured but **computationally intractable** (256-wide layers,
10–30 h with no convergence), so the canonical attack falls back to SA+margin.

**How to choose / run:**

```bash
# default (fires automatically in every attack driver script): SA + margin
./run_one_model_enhanced.sh tiny leakyrelu

# canonical CIFAR flagship (the exact GCE V100 attack — SA+margin, not PT):
bash run_cifar_kaggle.sh relu                       # workers=1 on GPU
DUAL_WORKERS=3 bash run_cifar_kaggle.sh leakyrelu   # 3 CUDA contexts on the V100

# reproduce legacy greedy baseline:
SIGN_METHOD=greedy SIGN_OBJ=agree ./run_one_model_enhanced.sh tiny relu
```

> **PT is intractable on the CIFAR `full` 256-wide arch** (projected 10–30 h, no
> convergence) — the canonical GCE runs use **SA+margin**, which is now the
> `run_cifar_kaggle.sh` default. Do **not** pass `SIGN_METHOD=pt` for `full`.

**Note on defaults.** `run_extraction.py`'s own argparse defaults are
`--sign-search-method greedy --sign-search-objective agree` (the legacy A/B
control). **All attack driver scripts** (`run_one_model_enhanced.sh`,
`run_extract.sh`, `run_from_cluster.sh`, `run_cifar_kaggle.sh`)
override this to **`sa`/`margin`** so the combinatorial search fires by default.
The chosen method/objective are recorded in `extraction_metrics.json`
(`sign_search_method`, `sign_search_objective`). Algorithm detail:
`paper_notes/section2/2_3_oracle_sign_search.md` §2.3.7–§2.3.12; benchmark
provenance: `paper_notes/section3/3_10_combinatorial_sign_search_results.md`.

---

## Generic single-model extraction: `run_extract.sh`

A lighter one-shot driver for the make_blobs tiers (tiniest/tinier/tiny). Does
**not** handle the CIFAR flagship — for that use the [CIFAR walkthrough](#cifar-10-flagship-walkthrough-relu).

### Step 0 — provide your model

Two equivalent victim files in `tiny_stuff/`: `<name>.pth` (PyTorch state dict
matching `signature_recovery/utils.py::CIFAR10Net`, fc1..fc5) and `<name>.keras`
(Keras equivalent, used by Phase-2 sign recovery). Put a matching `x_test.npy` in
`data/`.

### Step 1 — declare architecture + activation

Edit `signature_recovery/utils.py` (exactly one arch flag True):

```python
TINIEST = True   # 8-8-8-8-8-8
TINIER  = False  # 32-16-16-16-8-4
TINY    = False  # 64-64-64-64-64-10
MAKEBLOBS = True # make_blobs synthetic data (False for CIFAR-10)
LEAKY_ALPHA = 0.0   # set 0.01 to attack a LeakyReLU(0.01) victim
```

The same `TINIEST/TINIER/TINY` flag must match in
`sign_recovery/batched_sign_recovery.py`, and the same `LEAKY_ALPHA` must match
in **all four** files: `signature_recovery/utils.py`,
`sign_recovery/sign_recovery.py`, `sign_recovery/batched_sign_recovery.py`,
`analysis/extraction_pipeline/config.py`. When `LEAKY_ALPHA > 0`, model paths
resolve to `<name>_leakyrelu.{pth,keras}` automatically.

### Step 2 — run the full pipeline

```bash
cd enhanced_codebase
./run_extract.sh tiniest 9           # tiniest, 9 find_duals iterations (~1 min)
./run_extract.sh tinier  50          # tinier, 50 iterations (~10 min)
./run_extract.sh tiny    1000        # tiny, 1000 iterations (~11 h — overnight)
```

`run_extract.sh` reconfigures the arch flags, runs `find_duals.py` × N →
`cluster_dual_points_stream.py` → `generate_dual_neuron.py` →
`recover_weights.py {0,1,2,3}` → `batched_sign_recovery.py` →
`analysis/run_extraction.py --<model> --from-scratch --refine` (with
`--sign-search-method sa --sign-search-objective margin`).

Outputs: raw dual triplets (`exp/1/`), layer clusters (`exp/1-cluster-{0..3}.p`),
unsigned weights (`outputs/model_weights/Vrelu/`), per-neuron dual files,
`results/sign_recovery/layer{1..4}_*.npy`, and
`results/reconstructed_models/{reconstructed_<model>.pth, extraction_metrics.json}`.

### Step 3 — generate reports

```bash
PY=/path/to/python
$PY analysis/evaluate_reconstructed_tiny.py             # tiny (accuracy on task)
$PY analysis/evaluate_reconstructed_makeblobs.py        # tiniest
$PY analysis/compare_true_vs_extracted_tiny.py          # tiny (per-neuron weight comparison)
$PY analysis/compare_true_vs_extracted.py               # tiniest
```

Each writes a JSON next to the reconstructed model and prints a table. To turn
those logs into the two markdown reports, feed the logs + stdout into an LLM with
`ATTACK_PROMPT.md` as the system prompt.

### Step 4 — sanity-check / troubleshoot

If reconstructed accuracy < 90 %:
1. Check the sign-recovery summary — a neuron with `confidence < 0.55` is a
   coin-flip; sign-search should have fixed it but may have skipped if `k > 18`.
2. Check `recovery_stats` in `extraction_metrics.json`. If `<70 %` on any layer,
   run more `find_duals` iterations.
3. Increase `--refine-epochs` (e.g. to 2000) — refinement has capacity if its
   starting agreement is `>~20 %`.

---

## Reproduce the canonical CIFAR GCE attack

The two CIFAR flagship victims were extracted on a **self-managed GCE V100 VM**
(Tesla V100-SXM2-16GB, float64), and **only those runs are canonical** — the
`full (CIFAR)` column of the [attack-parameters table](#attack-parameters-validated-against-the-2026-06-21-make_blobs-runs)
is read back from their `extraction_metrics.json`. The live `run_cifar_kaggle.sh`
already bakes in every canonical value (SA+margin, `DUAL_ROUNDS=150`,
`DUAL_CHUNK=20`, `DISK_CAP_GB=180`, `--sign-restarts 1 --sign-pair-lookahead 4
--sign-refine-cycles 1 --refine-epochs 500`), so the reviewer fires the exact
attack with a single command per victim:

```bash
cd enhanced_codebase/Hard_Label_Work
export PYTHON_BIN=/home/biprarshi/miniconda3/envs/MLenv/bin/python3

bash run_cifar_kaggle.sh relu                       # cifar_relu  (GPU workers=1)
DUAL_WORKERS=3 bash run_cifar_kaggle.sh leakyrelu   # cifar_leakyrelu (3 CUDA contexts)
```

Requires an **fp64-capable GPU** (V100 ideal; T4/L4/P100 are 1/32 fp64 and will
crawl) and **~180 GB scratch**. Each run writes `full_<act>_extraction_metrics.json`,
`full_<act>_true_vs_extracted.{md,json}`, and `full_<act>_eval_scorecard.md` under
`paper_notes/section3/reports/cifar_kaggle_<date>/`. Canonical result artifacts:
`cifar_gce_relu_2026-06-25/` and `cifar_kaggle_2026-06-26/`. Headline recovery:
cifar_relu **499/832** (fc3/fc4 SVD-saturated to 0), cifar_leakyrelu **807/832**
(fc3 243/256, fc4 58/64). Full VM build/babysit/teardown playbook:
`enhanced_codebase/kaggle_cifar/GCE_VM_RESUME.md` + `CIFAR_RELU_RESUME.md`.

## CIFAR-10 flagship walkthrough (ReLU)

The **headline model**: the MLP `3072 → 256 → 256 → 256 → 64 → 10` on raw CIFAR-10
pixels, ReLU, float64. `run_extract.sh` does not handle it, so this is the
authoritative manual end-to-end recipe. (For the **canonical automated path**, use
`bash run_cifar_kaggle.sh relu` — see [Reproduce the canonical CIFAR attack](#reproduce-the-canonical-cifar-gce-attack)
above.)

### Hardware budget

| | |
|---|---|
| RAM | **≥22 GB** (16–17 Gi working set during Phase 1; **restart the machine first** if swap is non-empty — swap-thrashing was the binding constraint on the original run) |
| Disk | **≥80 GB free** (dual pickles ~55 GB, layer-cluster pickles ~9 GB) |
| Cores | 14-core CPU recommended; recipe pins `OMP/MKL_NUM_THREADS=2`, 5 dual-search workers |
| Wall time | **~5–6 h** with this recipe (61 min duals + 3 min cluster + ~67 min recover + ~78 min Phase-2 L1+L2 + ~7 min Phase-3); L3/L4 Phase-2 skipped — see Step 6 |

### Step 0 — place the victim + data

Three victim files in `tiny_stuff/`: `TinyModel_relu.pth`, `TinyModel_relu.keras`,
`TinyModel_relu_alpha.txt` (one line: `0.0`). CIFAR data in `data/` is a
**three-slice contract**:

```
data/x_test.npy        # uint8 (10000,3072) — CIFAR test (queryable, Phase-3 distillation)
data/x_test2_cifar.npy # CIFAR train[40000:50000] — queryable under --train-union-test12
data/x_test3_cifar.npy # CIFAR train[10000:20000] — strictly held-out (eval only, never queried)
# (+ matching y_*.npy)
```

X_test2 and X_test3 are disjoint train slices; X_test is the independent test
batch. Train them with `create_cifar_model.py` (emits all three slices, both ReLU
+ LeakyReLU variants). Sanity-check `.pth` vs `.keras` argmax agreement on
`x_test` — expect `1.0`:

```bash
$PY - <<'PY'
import sys, numpy as np, torch, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; sys.path.insert(0, 'signature_recovery')
import utils, tensorflow as tf
m_keras = tf.keras.models.load_model(utils.MODEL_PATH.replace('.pth', '.keras'), compile=False)
y_pt = utils.cheat_net_cpu(torch.tensor(utils.x_test[:200], dtype=torch.float64)).argmax(1).numpy()
y_kr = m_keras(utils.x_test[:200].astype(np.float32)).numpy().argmax(1)
print('argmax agreement (200):', float((y_pt == y_kr).mean()))   # expect 1.0
PY
```

### Step 1 — configure for CIFAR + ReLU

All four arch flags **False** (not a make_blobs path), `LEAKY_ALPHA = 0.0` in all
four files, bump sign-recovery workers, install `tabulate`:

```bash
PY=/home/biprarshi/miniconda3/envs/MLenv/bin/python3

# 1a. arch booleans all False (utils.py + batched_sign_recovery.py)
$PY - <<'PY'
import re, pathlib
for f in ['signature_recovery/utils.py', 'sign_recovery/batched_sign_recovery.py']:
    p = pathlib.Path(f); t = p.read_text()
    for k in ('TINIEST','TINIER','TINY','MAKEBLOBS'):
        t = re.sub(rf'^{k}\s*=\s*(True|False)\b', f'{k} = False', t, count=1, flags=re.M)
    p.write_text(t)
PY

# 1b. LEAKY_ALPHA = 0.0 in all four files
$PY - <<'PY'
import re, pathlib
for f in ['signature_recovery/utils.py','sign_recovery/sign_recovery.py',
          'sign_recovery/batched_sign_recovery.py','analysis/extraction_pipeline/config.py']:
    p = pathlib.Path(f); t = p.read_text()
    p.write_text(re.sub(r'^LEAKY_ALPHA\s*=\s*\S+', 'LEAKY_ALPHA = 0.0', t, count=1, flags=re.M))
PY

# 1c. sign-recovery worker count → 5  ;  1d. install dep
$PY -c "import re,pathlib; p=pathlib.Path('sign_recovery/batched_sign_recovery.py'); p.write_text(re.sub(r'^nThreads\s*=\s*\d+','nThreads                 = 5',p.read_text(),count=1,flags=re.M))"
$PY -c "import tabulate" 2>/dev/null || $PY -m pip install tabulate

# 1e. verify → LAYER_SIZES=[3072,256,256,256,64,10], MODEL_PATH=.../TinyModel_relu.pth, LEAKY_ALPHA=0.0
$PY -c "import sys; sys.path.insert(0,'signature_recovery'); import utils; print(utils.LAYER_SIZES, utils.MODEL_PATH, utils.LEAKY_ALPHA)"
```

### Step 2 — clean residuals (running on stale state mixes configs)

```bash
HERE="$(pwd)"
rm -rf "$HERE/signature_recovery/exp/1"
rm -f  "$HERE/signature_recovery/exp/1-cluster-"*.p
rm -rf "$HERE/signature_recovery/outputs/model_weights/Vrelu/layer_"*
rm -rf "$HERE/sign_recovery/layer_neuron_npys"
rm -f  "$HERE/results/sign_recovery/"* "$HERE/results/reconstructed_models/reconstructed_"* \
       "$HERE/results/reconstructed_models/extraction_metrics.json"
mkdir -p "$HERE/signature_recovery/exp/1" "$HERE/signature_recovery/outputs/model_weights/Vrelu" \
         "$HERE/sign_recovery/layer_neuron_npys" "$HERE/results/sign_recovery" \
         "$HERE/results/reconstructed_models"
free -h    # abort + restart if Swap used is not ≈ 0 / Mem available < 16 Gi
```

### Step 3 — batched parallel dual search (~61 min)

```bash
cd signature_recovery
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  $PY -u torch_impl/parallel_duals.py \
    --iterations 140 --workers 5 --batch-size 48 --target 4000 --impl torch \
    2>&1 | tee /tmp/cifar_relu_duals.log
cd ..
# end: "finished 140 rounds in ~3660s; 140 pickle files in .../exp/1"  (~55 GB on disk)
```

### Step 4 — streaming cluster (~3 min)

Per-neuron cap 150 keeps peak RAM under ~9 GB.

```bash
cd signature_recovery
CLUSTER_PER_NEURON_CAP=150 $PY -u cluster_dual_points_stream.py 2>&1 | tee /tmp/cifar_relu_cluster.log
cd ..
# expect ~ layer0 256/256, layer1 ~251/256, layer2 ~244/256, layer3 ~53/64 covered
```

### Step 5 — per-neuron bridge + weight recovery (~67 min)

```bash
cd signature_recovery
$PY -u generate_dual_neuron.py 2>&1 | tee /tmp/cifar_relu_neuron.log   # ~800 .npy files
for L in 0 1 2 3; do
    CLUSTER_START=0 CLUSTER_END=999 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      $PY -u recover_weights.py $L 2>&1 | tee /tmp/cifar_relu_recover_$L.log
done
cd ..
```

Expected recovery: `layer0 255/256`, `layer1 247/256`, `layer2 0/256`,
`layer3 0/64`, **total 502/832 ≈ 60 %**. **Layer 2/3 zero-recovery is intrinsic
to ReLU + depth, not a bug** — the documented lever is LeakyReLU(α > 0) (see
[EXPLANATIONS.md → How Leaky ReLU works](EXPLANATIONS.md#how-leaky-relu-works)).

### Step 6 — Phase-2 sign recovery (~78 min for L1+L2, abort L3+L4)

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=3 \
  $PY -u sign_recovery/batched_sign_recovery.py 2>&1 | tee /tmp/cifar_relu_sign.log
```

L1 (~28 min) and L2 (~50 min) complete and write `layer{1,2}_summary.json`. **L3
hits algorithmic slowdown** (10–12 h to finish L3+L4); abort once L2's summary is
written — Phase-3 sign search recovers the missing signs anyway:

```bash
# watch for "Layer 2 Summary", then:
pkill -9 -f batched_sign_recovery ; pkill -9 -f sign_recovery.py
```

Pad stub L3+L4 sign files (default all +1; Phase-3 flips what's needed):

```bash
$PY - <<'PY'
import json, os, numpy as np
RESULTS_MODEL='./results/model_TinyModel_relu'; RESULTS_SIGN='./results/sign_recovery'
LAYER_NEURONS={1:256,2:256,3:256,4:64}; model_layers={}
for L,n in LAYER_NEURONS.items():
    signs=np.ones(n,dtype=np.int8); confs=np.zeros(n); votes=np.zeros(n,dtype=np.int32); done=0
    lay_dir=f'{RESULTS_MODEL}/layerID_{L}'
    if os.path.isdir(lay_dir):
        for nid in range(n):
            f=f'{lay_dir}/neuronID_{nid}/sign_result.json'
            if os.path.exists(f):
                try:
                    d=json.load(open(f)); s=d.get('recovered_sign')
                    if s in (1,-1): signs[nid]=int(s)
                    confs[nid]=float(d.get('confidence',0.0)); votes[nid]=int(d.get('total_votes',0)); done+=1
                except Exception: pass
    if L in (3,4) or not os.path.exists(f'{RESULTS_SIGN}/layer{L}_signs.npy'):
        np.save(f'{RESULTS_SIGN}/layer{L}_signs.npy',signs)
        np.save(f'{RESULTS_SIGN}/layer{L}_confidences.npy',confs)
        np.save(f'{RESULTS_SIGN}/layer{L}_votes.npy',votes)
        json.dump({'layerID':L,'num_neurons':n,'neurons_processed':done,
                   'neurons_positive_sign':int((signs==1).sum()),'neurons_negative_sign':int((signs==-1).sum()),
                   'signs':signs.tolist(),'confidences':confs.tolist(),'votes':votes.tolist(),
                   'note':'partial — Phase-2 halted; rest default +1 for Phase-3 oracle search' if L in (3,4) else None},
                  open(f'{RESULTS_SIGN}/layer{L}_summary.json','w'),indent=2)
    model_layers[str(L)]={'num_neurons':n,'neurons_processed':done,'signs':signs.tolist(),'confidences':confs.tolist()}
json.dump({'model':'TinyModel_relu','layers':model_layers},open(f'{RESULTS_SIGN}/model_sign_recovery_summary.json','w'),indent=2)
print('Wrote layer{1..4}_*.npy + model_sign_recovery_summary.json')
PY
```

### Step 7 — Phase-3 reconstruction + refinement

**Baseline** (byte-identical to the 2026-06-04 reference, ~7 min):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_CPP_MIN_LOG_LEVEL=3 \
  $PY -u analysis/run_extraction.py --full --from-scratch --refine --refine-epochs 1000 \
    2>&1 | tee /tmp/cifar_relu_phase3.log
```

`--full` = CIFAR `FullModel`; `--from-scratch` = bias recovery + sign search +
fc5 LR fit; `--refine` = frozen-row distillation against the oracle.

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
    --sign-search-method sa --sign-search-objective margin \
    2>&1 | tee /tmp/cifar_relu_phase3_fixed.log
```

For the widest CIFAR layers, swap `--sign-search-method sa` → `pt` (parallel
tempering crosses barriers SA can miss). The CIFAR-fix flags (all default-off):

| Flag | Purpose |
|---|---|
| `--eval-on-test3` | Route every Phase-3 eval/watchdog to `X_test3` (strict held-out). Without it, eval falls back to `X_test2` and the run is no longer honest-eval. |
| `--train-union-test12` | Promote `X_test2` into the queryable pool. Phase-3 trains on `X_test ∪ X_test2` (20 K queries instead of 10 K). |
| `--early-stop` / `--patience N` / `--eval-every E` | Refinement watchdog: every `E` epochs score a 1024-row `X_test3` slice; save best; stop after `N` evals without improvement; restore best. |
| `--refine-weight-decay W` | AdamW weight decay (`W=0` = plain Adam, byte-identical to legacy). |
| `--refine-cosine-lr` | CosineAnnealingLR schedule over the refinement budget. |
| `--sign-search-method {greedy,tabu,sa,pt}` | Per-layer sign optimiser for `k > 18` layers. Driver scripts default `sa`; `pt` for widest CIFAR layers. Zero extra oracle queries. See [Sign-search methods](#sign-search-methods-metaheuristic-sign-search). |
| `--sign-search-objective {agree,margin}` | Climb signal: `agree` (flat 0/1) or `margin` (smoothed). Acceptance always on true agreement. Default `margin`. Ignored for `greedy`. |
| `--sign-restarts R` | Base traversal + R random-restart traversals; pick best `X_test3` agreement. |
| `--sign-pair-lookahead K` | After convergence, try all C(K,2) pair flips of the K most-uncertain neurons. Catches coupled escapes. |
| `--sign-refine-cycles C` / `--sign-refine-mini-epochs E` | Interleave sign-search ↔ E-epoch mini-refinement for C cycles. |

Expected stages (recommended flags): recovery 502/832 (330 random-init) →
biases 502 set → fc5 LR fit ~34.5 % on X_test3 → 3 sign-search/pair-flip/mini-refine
cycles → final refinement (early-stop ~epoch 80/500) → **X_test3 (10K) agreement
≈ 54.7 %** (vs 50.40 % baseline).

### Step 8 — read the report

Outputs land in `results/reconstructed_models/` (`reconstructed_full.pth`,
`extraction_metrics.json` with `eval_tag`, `sign_cycle_log`,
`sign_pair_lookahead_results`) and `results/reports/cifar_relu_{full,fixed}_<date>.md`.

Expected headline numbers on **strictly held-out `X_test3` (10K)**:

| | Baseline (2026-06-04) | CIFAR-fix (2026-06-05) |
|---|---|---|
| Oracle (victim) accuracy | 53.34 % | 53.34 % |
| Reconstructed accuracy   | ~44 %   | ~44 %   |
| **Prediction agreement** | **50.40 %** | **54.71 %** (+4.31 pt) |
| Watchdog peak (1024-row slice) | n/a | 55.08 % |
| Refinement epoch at early-stop | 1000 (full) | 80 / 500 |
| L0 / L1 sign accuracy | 49.0 % / 48.2 % | 52.2 % / 48.2 % |
| Recovered neurons | 502/832 (60 %) | 502/832 (60 %) |
| L0/L1 mean \|cos sim\| | 1.000 | 1.000 |

### Improved evaluation (Step 9)

The single "agreement" number cannot tell **extraction** apart from **pure
distillation** and is confounded by the victim's ~53 % accuracy. The improved
scorecard replaces it with a multi-metric suite. **This runs automatically as
step 9 of `run_one_model_enhanced.sh`**; run standalone on any prior extraction:

```bash
python3 analysis/evaluate_extraction_quality.py --full       # CIFAR flagship
python3 analysis/evaluate_extraction_quality.py --makeblobs  # tiny relu (synthetic)
python3 analysis/evaluate_extraction_quality.py --tiniest    # smoke / structural
```

- **Activation is auto-detected** from the extraction metrics' `leaky_alpha` field
  (falling back to the `*_alpha.txt` sidecar / filename), eliminating silent
  ReLU↔LeakyReLU mismatch.
- **The distillation baseline is mandatory** — every report is a two-arm
  extraction-vs-distillation comparison. If `reconstructed_<arch>_distillation.pth`
  is missing the driver builds it (`--refine-unfreeze`, same regularisers),
  caches it, and reuses it; force a rebuild with `--force-distill`. Skip only via
  the explicit `--allow-single-arm`.

Metrics: (1) in-dist fidelity + accuracy, (2) margin-conditioned fidelity,
(3) off-distribution + interpolation agreement (the extraction-vs-distillation
discriminator), (4) paired McNemar + bootstrap CI on the gap, (5) structural
receipts (|cos|, sign-acc, coverage), plus a composite **EQS (0–100)**. Canonical
CIFAR run: in-dist gap **+6.1 pt**, off-distribution gap **+14 pt**, McNemar
p≪1e-20, EQS gap ≈ +23. Reports → `results/reports/eval_<arch>_<date>.md`.

---

## Leaky ReLU — quick start

The pipeline supports Leaky ReLU(α) victims via a single `LEAKY_ALPHA` toggle
(default `0.0` = ReLU, byte-identical). With the enhanced driver, just pass
`leakyrelu` (it sets `LEAKY_ALPHA=0.01` across all four files automatically):

```bash
./run_one_model_enhanced.sh tiniest leakyrelu
```

For the manual `run_extract.sh` path:

```bash
cd enhanced_codebase

# Tiniest (8-8-8-8-8-8 LeakyReLU(0.01))
$PY create_tiniest_makeblobs_leakyrelu.py          # train victim
sed -i 's|^LEAKY_ALPHA = 0.0$|LEAKY_ALPHA = 0.01|' \
    signature_recovery/utils.py sign_recovery/sign_recovery.py analysis/extraction_pipeline/config.py
sed -i 's|^LEAKY_ALPHA              = 0.0$|LEAKY_ALPHA              = 0.01|' sign_recovery/batched_sign_recovery.py
# set TINIEST=True, TINIER=False, TINY=False (run_extract.sh does this)
./run_extract.sh tiniest 5
$PY analysis/evaluate_reconstructed_makeblobs.py    # expect ~99.25 % on X_test2, 22/32 recovered

# Tinier (32-16-16-16-8-4 LeakyReLU(0.01))
$PY create_tinier_makeblobs_leakyrelu.py            # + toggle LEAKY_ALPHA, set TINIER=True
./run_extract.sh tinier 8                           # expect ~100 % on X_test2, 33/56 recovered
```

**Reverting to ReLU:** set `LEAKY_ALPHA = 0.0` in all four files — paths resolve
back to `*_relu.{pth,keras}`, ReLU pipeline byte-identical.

(How the leaky patches work: [EXPLANATIONS.md → How Leaky ReLU works](EXPLANATIONS.md#how-leaky-relu-works).)

### Performance / stability tips for leaky runs

- **Sign recovery hangs** (e.g. tiniest layer-2 neuron-7): reduce `nExpMin`/`nExp`
  in `batched_sign_recovery.py` (leaky config uses 200/2000 for layers 1–3,
  100/1000 for layer 4), or kill after layers 1+2 finish — Phase-3
  `oracle_sign_search` fills in the missing signs.
- **OOM on 24 GB machines**: drop `nThreads` in `batched_sign_recovery.py`.
- **Layer 4 often fails signature recovery** (same as ReLU) — refinement
  compensates via fc5 LR fit + frozen-row training.

---

## Batched PyTorch dual search

Drop-in replacement for the Phase-1 bottleneck (`find_duals.py`): B independent
boundary walks in lockstep + W parallel workers, **no algorithm changes**, same
triplet pickle format. (How it works: [EXPLANATIONS.md](EXPLANATIONS.md#how-the-batched-dual-search-works).)

### Usage (drop-in for `run_extract.sh` STEP 2)

```bash
cd enhanced_codebase/Hard_Label_Work
# arch/activation come from utils.py toggles (set them first)
./run_duals_torch.sh <ITERS> <WORKERS> <BATCH>
#   ITERS    pickle rounds (≈ find_duals.py invocations)
#   WORKERS  concurrent processes        (default cores/2)
#   BATCH    walks per batch, torch impl  (default 256)

# tiniest:  ./run_duals_torch.sh 9   8 256     # ~6 s
# tiny:     ./run_duals_torch.sh 500 8 256     # ~24 min (was ~18 h)
```

Then continue with `cluster_dual_points_stream.py → generate_dual_neuron.py →
recover_weights.py {0..3} → batched_sign_recovery.py → run_extraction.py`.
The batched torch finder (`parallel_duals.py --impl torch`) is the only
supported dual search.

### Validated results

| | NumPy (original) | Torch (this port) |
|---|---|---|
| tiniest, 9 rounds, 8 workers | ~75–135 s | **5–9 s** (~10–25×) |
| **tiny, full dual search** | **~18 h** | **24.3 min** (~44×) |
| tiny signature recovery | 157/256 | **154/256** (matches; fc4 0/64 as expected for ReLU) |
| tiny functional agreement (X_test2) | 100 % | **100 %** |
| triplet format / recovery rate | — | identical / ≥ NumPy on every layer |

---

## Results

### Full 6-model end-to-end (2026-05-21, sequential NumPy dual search)

| Model | Phase-1 recovered | mean \|cos\| | sign acc | X_test2 acc | agreement | wall time |
|---|---|---|---|---|---|---|
| tiniest_relu       | 24/32 (75 %) | 1.000 | 0.610 | 98.95 % | 98.90 % | 135 s |
| tiniest_leakyrelu  | 23/32 (72 %) | 1.000 | 0.451 | 99.20 % | 99.20 % | 115 s |
| tinier_relu        | 30/56 (54 %) | 1.000 | 0.458 | 100.00 % | 100.00 % | 917 s |
| tinier_leakyrelu   | 37/56 (66 %) | 1.000 | 0.548 | 100.00 % | 100.00 % | 976 s |
| tiny_relu          | 157/256 (61 %) | 1.000 | 0.528 | 100.00 % | 100.00 % | ~18 hr |
| **tiny_leakyrelu** | **230/256 (90 %)** | **1.000** | **0.525** | **100.00 %** | **100.00 %** | **~18.8 hr** |

- All 6 hit 98.95–100 % agreement with exactly **3 batched `oracle(X).argmax` queries**.
- Mean **|cos| = 1.000** on every recovered neuron.
- **Leaky beats ReLU more at scale**: tinier +7, tiny **+73** neurons (fc4: leaky 57/64 vs ReLU 0/64).
- The `~18 hr` figures are the original sequential NumPy finder; the batched port
  reproduces tiny_relu (154/256, 100 % agreement) with the dual search in **~24 min**.

(Canonical make_blobs results are the 2026-06-21 SA/PT run —
`paper_notes/section3/3_10_combinatorial_sign_search_results.md`.)

---

## Drivers reference

| Script | Use |
|---|---|
| `run_one_model_enhanced.sh <arch> <act>` | **Canonical** end-to-end driver (parallel duals, SA-margin sign search, X_test3 watchdog, step-9 scorecard). `arch ∈ {tiniest,tinier,tiny,full}`, `act ∈ {relu,leakyrelu}`. |
| `run_cifar_kaggle.sh <act>` | CIFAR `full` monolithic driver (disk-chunked duals, SA phase-3); env: `SIGN_METHOD`, `DUAL_WORKERS`, `DUAL_CHUNK`. |
| `run_extract.sh <arch> <iters>` | Lighter make_blobs one-shot (sequential or torch duals). |
| `run_from_cluster.sh <arch> <act>` | Resume from cluster onward (skips clean + find_duals; reuses `signature_recovery/exp/1/`). |
| `run_distillation_baseline.sh` | No-signature distillation baseline for the two-arm comparison. |

```bash
PYTHON_BIN=/path/to/python3 ./run_one_model_enhanced.sh full relu
```

### Smaller smoke tests

```bash
./run_extract.sh tiniest 5       # ReLU, ~1 min, ~99.9 % agreement
./run_extract.sh tinier  8       # ReLU, ~3 min, 100 % agreement

# Phase-3 only (reuse existing recovery outputs):
python3 analysis/run_extraction.py --tiniest --sign-search --refine --refine-epochs 200
python3 analysis/run_extraction.py --tinier  --sign-search --refine --refine-epochs 200
python3 analysis/run_extraction.py --tiniest --from-scratch --refine --refine-epochs 1000
```

---

## Credits

Attack algorithm: Carlini, Chen, Choquette-Choo, Kos, Tramèr, "Polynomial Time
Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting",
EUROCRYPT 2024. See [EXPLANATIONS.md](EXPLANATIONS.md) for how this fork works.
