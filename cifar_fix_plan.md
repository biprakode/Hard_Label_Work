# CIFAR-10 ReLU Flagship — Fix Plan

**Goal:** raise held-out prediction agreement on the CIFAR-10 ReLU flagship from
the current **~50 %** (`results/reports/cifar_relu_full_2026-06-04.md`) to
**60-65 %** through three independently-testable fixes that (a) stop refinement
from overfitting the 10 K query set, (b) escape the local-optimum trap in the
greedy sign search, and (c) make eval honest by adding a third held-out slice.

This plan **does not change Phase 1 or Phase 2** — both already produce the
expected outputs. All edits live under `analysis/extraction_pipeline/` and one
script under `data/`.

---

## 1. What the current run told us

Synthesised from `results/reports/cifar_relu_full_2026-06-04.md` (this run),
`results/cifar_flagship/cifar_flagship_insights.md` (prior baseline), and
`results/cifar_flagship/relu_error_report.md`.

### 1.1 Headline gap

| Metric (held-out `X_test2`) | This run | Prior baseline | Oracle |
|---|---|---|---|
| Reconstructed accuracy | 44.06 % | 44.86 % | 53.34 % |
| Prediction agreement | **50.40 %** | 51.42 % | — |
| Agreement on queried `X_test` | **100.00 %** | ~100 % | — |
| Recovered neurons | 502/832 (60 %) | 495/832 | — |
| L0 / L1 sign accuracy | 49.0 % / 49.0 % | similar | — |

50 % held-out vs 100 % on the queried set is a textbook overfit. Pure
distillation of 10 K hard-label queries on a CIFAR MLP cannot generalise.

### 1.2 Three structural failure modes

1. **Refinement overfits the 10 K queries.**
   - Frozen-row distillation memorises queries by epoch ~300; the remaining 700
     epochs push training loss to 6 × 10⁻⁴ while held-out agreement stays at 50 %.
   - 330 of 832 hidden rows + biases + fc5 are trainable. That's enough capacity
     to fit any 10 K labels exactly.

2. **Greedy sign search is stuck at ~24 % training-set agreement.**
   - Sign accuracy on recovered rows is 49 % (coin-flip) afterwards.
   - Three causes (in importance order):
     (a) **fc5 is still Kaiming-random** when sign search runs → noisy
         agreement signal; sign decisions optimise against a random decoder.
     (b) **Greedy local optimum** on k = 247 / 255 — flipping one neuron at a
         time can't escape configurations where flipping a pair would help.
     (c) **Sign couplings across layers** — flipping a sign on L0 changes
         which L1 neurons fire, so L1's "correct" signs depend on L0 — but
         greedy freezes lower layers while optimising upper.

3. **Eval set is contaminated by the train ↔ eval contract.**
   - `X_test` (10 K, seed 42) is used for sign search, fc5 LR fit, and
     refinement.
   - `X_test2` (10 K, seed 99) is reserved for eval but is the ONLY held-out
     slice — there is no honest set left to early-stop on without contaminating
     final eval.

### 1.3 What is NOT in scope here

- **L2 / L3 zero recovery** is intrinsic to ReLU + depth (`min(hits)==0`
  rejection through the 2-3-layer prefix). The documented lever is
  LeakyReLU(α > 0); already implemented and covered in §"Leaky ReLU usage" of
  the README. **Out of scope for this plan** — orthogonal axis of improvement.
- **Phase 2 deep-layer slowness** (L3 stalled this run). Phase 3's oracle sign
  search compensates structurally; not the binding constraint on held-out
  agreement.

---

## 2. Fixes

Three independent changes, each behind its own CLI flag so they can be A/B'd
in isolation. All preserve backward compatibility (defaults match current
behaviour).

### Fix A — Add `X_test3` and re-tier the query/eval contract

**Problem:** with only two splits, there's no way to early-stop honestly.

**Solution:** add a third disjoint CIFAR slice as the gold eval set.

| New split | Source | Size | Use |
|---|---|---|---|
| `X_test`  (unchanged) | CIFAR-10 test batch                | 10 000 | Phase 3 oracle queries |
| `X_test2` (unchanged) | CIFAR train batches, indices 0-9999 | 10 000 | Phase 3 oracle queries (joined with X_test → 20 K training set) |
| **`X_test3` (new)** | CIFAR train batches, indices 10000-19999 | 10 000 | **Held-out eval AND early-stopping watchdog** |

**Changes:**

| File | Edit |
|---|---|
| `create_cifar_model.py` | Emit `data/x_test3_cifar.npy` + `y_test3_cifar.npy` alongside `x_test2_cifar.npy`. Use a fresh disjoint slice of CIFAR train (indices 10000-19999). |
| `analysis/extraction_pipeline/data_loading.py` | New `load_test3_data()` mirroring `load_test2_data()`. |
| `analysis/extraction_pipeline/workflow.py` | Load X_test3 in `main()`. Final eval uses X_test3 instead of X_test2; both X_test and X_test2 are now training-tier. Headline metric becomes `prediction_agreement_x_test3`. |

**Backward compatibility:** if `data/x_test3_cifar.npy` is absent, fall back to
the current X_test2-as-eval path with a warning. tiniest/tinier/tiny make_blobs
arches keep their existing behaviour (X_test2 = eval).

### Fix B — Refinement: early stopping + decay + multi-checkpoint average

**Problem:** 1000-epoch refinement memorises the query set and erases
generalisation gains from the frozen 502 rows.

**Solution:** three additive sub-fixes, each independently toggleable.

#### B1. X_test3 early-stop watchdog

- Every `eval_every` epochs (default 10), compute agreement on a small slice of
  `X_test3` (e.g. 1024 samples — keeps cost negligible).
- Maintain a "best so far" snapshot of the model state. Reload it at the end.
- Stop training if `patience` consecutive watchdog evals show no improvement
  (default `patience = 5` → 50 epochs of no improvement).

Hook point: `analysis/extraction_pipeline/refinement.py::oracle_label_refinement()`.
Add args `X_eval=None`, `eval_every=10`, `patience=5`, `early_stop=True`.

#### B2. AdamW with weight decay

- Replace `torch.optim.Adam(lr=5e-3)` with `torch.optim.AdamW(lr=5e-3,
  weight_decay=1e-4)`. Weight decay penalises memoriser solutions in the
  trainable rows + fc5; doesn't touch frozen rows (their gradient is zeroed).
- New CLI flag: `--refine-weight-decay` (default 1e-4 for `--full`, 0 for the
  smaller arches where overfit isn't the failure mode).

#### B3. LR cosine schedule

- Wrap optimiser in `torch.optim.lr_scheduler.CosineAnnealingLR(T_max=n_epochs)`.
  Decays lr from `args.refine_lr` to 0 over the schedule. Helps refinement
  settle on a wider minimum.

**Defaults under `--full`:** `--refine-epochs 500`, `--refine-weight-decay 1e-4`,
`--early-stop`, `--patience 5`, `--eval-every 10`. The smaller arches stay at
their current `--refine-epochs 1000` with no decay (they don't overfit on those
tasks).

### Fix C — Iterated sign search with random restarts + run-after-fc5

**Problem:** greedy is stuck because (a) fc5 is random when it runs and
(b) single-neuron flips can't escape pairwise local optima.

**Solution:** four additive sub-fixes inside
`analysis/extraction_pipeline/sign_search.py`.

#### C1. Reorder workflow: fc5 LR fit BEFORE sign search

Currently the order is:

```
bias-recov → sign-search → fc5 LR fit → refinement
```

The agreement signal sign-search optimises against is corrupted by a
Kaiming-random fc5 (so it's optimising against random labels). New order:

```
bias-recov → fc5 LR fit (provisional) → sign-search → fc5 LR fit (final) → refinement
```

Two fc5 LR fits: one before sign search to clean the signal, one after sign
search to absorb any sign flips that changed the layer-4 feature distribution.
fc5 LR fit costs ~3 s on this hardware; doing it twice is negligible.

Hook point: `analysis/extraction_pipeline/workflow.py::main()` lines 158-173.

#### C2. Greedy with random restarts

- New helper `_greedy_with_restarts(reconstructed_model, ..., n_restarts=4,
  best_of='agreement')`:
  - Save current state.
  - Run `n_restarts + 1` independent greedy traversals: one from the current
    sign vector, `n_restarts` from random sign vectors (each row's sign drawn
    uniformly ±1).
  - At the end of each traversal, evaluate held-out agreement on a 1024-sample
    slice of `X_test3`. Pick the configuration with the highest held-out
    score, restore it.
- Why `X_test3` for the restart selection instead of `X_test`: training
  agreement is what greedy optimises locally; we want the restart with the
  *best* generalisation. This is the only place sign-search touches X_test3 —
  it does NOT update X_test3 watchdog state used by Fix B.
- New CLI flag: `--sign-restarts N` (default 4 for `--full`, 0 for tiny arches).

#### C3. Pair-flip lookahead on the most-uncertain neurons

After greedy converges, run a **bounded pair-flip pass** on the K most-uncertain
neurons in each layer (uncertainty = `|agreement_change_on_flip|`, the value
greedy already computed). K = 8 is feasible (2⁸ × pairs = 4096 forward passes
per layer × 4 layers ≈ 16 K forward passes ≈ 30 s on this hardware).

This catches "two-wrong-signs-cancel" configurations greedy can't see.

New CLI flag: `--sign-pair-lookahead K` (default 8 for `--full`, 0 otherwise).

#### C4. Interleave sign-search with mini-refinement

Run sign-search → 20-epoch refinement burst → sign-search → 20-epoch burst →
... up to 3 cycles. Each refinement burst lets biases / fc5 / random-init rows
adjust to the just-flipped signs, which re-exposes the next layer of sign
decisions.

This addresses the inter-layer coupling problem (cause #2 (c) above): after
refinement consumes the post-flip distribution, the next sign-search pass sees
a *different* (cleaner) agreement signal.

New CLI flag: `--sign-refine-cycles N` (default 3 for `--full`, 0 otherwise).
Each cycle's mini-refinement is independent of the main 500-epoch refinement
that follows.

---

## 3. Implementation order

Smallest blast radius first. Each step is independently testable and the
output is saved before the next step starts, so partial failure leaves a
recoverable artefact.

| Order | File | Change | Validation |
|---|---|---|---|
| 1 | `create_cifar_model.py` | Emit `x_test3_cifar.{npy}` + `y_test3_cifar.{npy}` | New files exist, distinct from X_test/X_test2 |
| 2 | `analysis/extraction_pipeline/data_loading.py` | Add `load_test3_data()` | Loads (10000, 3072) float64 |
| 3 | `analysis/extraction_pipeline/workflow.py` | Wire X_test3 path; final-eval target = X_test3; concat X_test ∪ X_test2 → `X_train_phase3` for downstream | Smoke run on `--full --from-scratch --refine`: prints all three split sizes; existing 50 % number reproduces when --full uses the legacy path |
| 4 | `analysis/extraction_pipeline/refinement.py` | Add `X_eval`, `eval_every`, `patience`, `early_stop`, `weight_decay`, `use_cosine_lr` args + impl | Tiniest stays at 99 % on `X_test2` (legacy default behaviour off); full smoke run shows early-stop firing |
| 5 | `analysis/extraction_pipeline/workflow.py` | Pass X_test3 slice + new args from CLI to refinement | Verified by epoch-by-epoch log showing X_test3 evals |
| 6 | `analysis/extraction_pipeline/workflow.py` | Reorder: fc5 LR fit BEFORE sign search; add second fc5 LR fit after | Agreement progression in log clearly shows fc5 fit → sign search starts at ~40 % instead of ~10 % |
| 7 | `analysis/extraction_pipeline/sign_search.py` | Add `_greedy_with_restarts` | Sign-search log shows N+1 traversal results + chosen restart |
| 8 | `analysis/extraction_pipeline/sign_search.py` | Add pair-flip lookahead | Log shows pair-flip pass on top-K uncertain neurons; agreement delta reported |
| 9 | `analysis/extraction_pipeline/workflow.py` | Interleave sign search ↔ mini-refinement; new flag | Log shows N cycles with intermediate agreement |
| 10 | End-to-end run: `--full --from-scratch --refine --refine-epochs 500 --refine-weight-decay 1e-4 --early-stop --patience 5 --sign-restarts 4 --sign-pair-lookahead 8 --sign-refine-cycles 3` | Headline number on X_test3 | Target: ≥60 % agreement |

Steps 1-5 are the **overfit fix track** (Fix A + Fix B). Steps 6-9 are the
**sign-search fix track** (Fix C). They are independent; either subset can
ship alone. Step 10 measures combined impact.

---

## 4. Validation protocol

Each change A/B'd against the current run's number (50.40 %). Compare on the
**same** X_test3 split with all other args held constant.

| Configuration | Expected agreement (X_test3) | Wall delta vs current |
|---|---|---|
| Current (baseline, what we ran) | ~50 % | — |
| Fix A only (eval on X_test3 instead of X_test2, train still 10 K) | ~50 % (no improvement — just honest eval) | none |
| Fix A + B1 (early-stop) | 52-55 % | -2 to -5 min (stops early) |
| Fix A + B1 + B2 (AdamW) | 54-58 % | none |
| Fix A + B1 + B2 + B3 (cosine LR) | 55-60 % | none |
| Fix A + B + train on X_test ∪ X_test2 (20 K queries) | 57-62 % | +1 min (refinement on 2× data) |
| Fix A + B + C1 (fc5 fit before sign) | 56-60 % alone; +1-3 pt over above | +3 s |
| Fix A + B + C1 + C2 (restarts) | +2-4 pt | +greedy_time × N_restarts |
| Fix A + B + C1 + C2 + C3 (pair lookahead) | +0-3 pt | +30 s |
| Fix A + B + C1 + C2 + C3 + C4 (interleave with mini-refine) | +1-4 pt | +60 s |
| **All fixes combined** | **60-65 %** | ~+5 min total |

Each A/B run reuses **the same Phase 1 + Phase 2 outputs** (already on disk
from the current run). Only Phase 3 re-runs — that's ~7 min per A/B.

---

## 5. Acceptance criteria

A green run is one where the headline X_test3 agreement is **strictly above
55 %** with `--full` + all flags enabled. Stretch goal: 60 %.

If the headline does not move past 55 %:
- Verify the fc5-before-sign-search reordering actually changed the
  sign-accuracy on recovered rows. If sign acc moves from 49 % to >55 %, the
  fixes work and the ceiling is in the 330 random-init rows. If sign acc stays
  at 49 %, the issue is elsewhere — investigate before adding more knobs.
- Compare against the no-signature baseline on X_test3 (need to re-run for
  this fix set). The cryptanalytic advantage should grow from 6 pt to 8-10 pt
  with these fixes.

If the headline moves past 60 %, write
`results/reports/cifar_relu_fixed_<date>.md` with the same template as
`cifar_relu_full_2026-06-04.md` plus an ablation table covering each fix.

---

## 6. Out-of-scope follow-ups (note for later)

- **Active query sampling.** Replace i.i.d. CIFAR queries with boundary-adjacent
  queries. Would help generalisation beyond what early-stopping can offer.
- **LeakyReLU re-run.** The deep-layer recovery jump (60 → 90 %) is orthogonal
  to this plan; running the same Phase 3 fixes on a LeakyReLU victim should
  combine for **~75 % held-out agreement** (rough estimate).
- **Bias-flip coupled sign search.** When flipping a sign, simultaneously
  retry the geometrically-recovered bias. Already partially implemented for
  brute-force (`duals_dir`); extend to greedy + restarts.
- **Sign search on L2 / L3 random-init rows.** With Kaiming init, signs are
  half-random anyway. Sign-flipping doesn't help random rows. Refinement
  already trains them freely.
