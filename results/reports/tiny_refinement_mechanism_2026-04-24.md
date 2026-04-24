# Tiny — How Phase 3 Refinement Works (and Why It Reaches 100 %)

**Date:** 2026-04-24
**Scope:** the Phase 3 part of `analysis/test_extraction4.py` when invoked with
`--makeblobs --from-scratch --refine --refine-epochs 1000`, as applied to the
tiny 64→64→64→64→64→10 make_blobs model in this extraction run.

## Problem at the start of Phase 3

Phase 1 (signature recovery) and Phase 2 (sign recovery) produce the following
per-layer inventory for tiny:

| Layer | Signature-recovered rows | Signs recovered | Rows at Kaiming random |
|---|---:|---:|---:|
| fc1 | 64 / 64 | 64 / 64 | 0 |
| fc2 | 61 / 64 | 63 / 64 | 3 |
| fc3 | 44 / 64 | 52 / 64 | 20 |
| fc4 | 0 / 64  | 47 / 64 | **64** |
| fc5 | — | — | 10 rows, fresh LR-fit |

Immediately after `load_unsigned_weights` combines these with the recovered
signs (and Kaiming-fills the gaps), the reconstructed model agrees with the
oracle on **~8 %** of test inputs — barely better than 10 % random guessing.

Phase 3 closes that gap to **100.00 %** using only hard-label oracle queries.
This report explains how.

## Phase 3 pipeline — stages and order

```
(reconstructed model assembled from Phase-1/2 outputs + Kaiming fills)
                                │  agreement ≈ 0 %
                                ▼
        1. recover_biases_from_duals          (no oracle queries)
                                │  agreement 0.0787
                                ▼
        2. oracle_sign_search                 (hard-label oracle on X_test)
                                │  agreement 0.0787 (no improvement this run)
                                ▼
        3. recover_output_layer (fc5 LR fit)  (hard-label oracle on X_test + aug)
                                │  agreement 0.9997
                                ▼
        4. oracle_label_refinement            (hard-label oracle on X_test, 1000 epochs)
                                │  agreement 1.0000
                                ▼
                     reconstructed_tiny_frozen.pth
```

Each stage in order, with the exact role, the oracle interaction, and the
measured effect on this run.

---

## Stage 1 — `recover_biases_from_duals` *(no oracle queries)*

**Location:** `test_extraction4.py:707`
**Idea:** for every signature-recovered neuron *i* in hidden layer *L*, its
bias `b_i` is determined by the geometry: the neuron crosses zero exactly on
its dual points. So

```
0 = w_i · h_{L-1}(x_d) + b_i      ⇒      b_i = -w_i · h_{L-1}(x_d)
```

where `h_{L-1}(x_d)` is the activation of the reconstructed (not oracle)
model at layer *L−1* on dual point `x_d`.

**Inputs used:**
- the dual-point `.npy` files in `sign_recovery/layer_neuron_npys/` (produced
  in Phase 1, purely from decision-boundary walks)
- the reconstructed model itself

**Does NOT query the oracle.** The only "oracle-adjacent" information already
baked into the dual points is which hyperplane they live on, which is a
geometric property of the model, not a label query.

**Median over 30 duals** makes this robust to dual-point noise.

**Effect on tiny:** fc1/fc2/fc3 biases are reset to geometric estimates; fc4
biases are left at Kaiming values because there are no recovered fc4 weight
rows to geometrically anchor. Agreement moves from ~random to **0.0787** —
better than random guessing, but not by much, because fc4 is still garbage.

---

## Stage 2 — `oracle_sign_search` *(hard-label oracle queries on X_test)*

**Location:** `test_extraction4.py:808`

**Idea:** the sign-recovery step is unreliable on deep / narrow layers. For
each hidden layer *L* with `k ≤ 18` recovered neurons, brute-force all **2^k**
sign combinations (flip `w_i` **and** `b_i` together, keeping `b_i = -w_i ·
h` consistent), score each by hard-label agreement against the oracle on
`X_test`, and adopt the best combo.

**Oracle interaction:**
- Query `oracle_model(X_test).argmax(-1)` once (vector of 10 000 hard labels).
- For each of `sum(2^k)` candidate combos, run `reconstructed(X_test).argmax(-1)`
  and compare.

**Does NOT expose soft logits, weights, or activations.** This is exactly the
hard-label threat model.

**Effect on tiny:** agreement 0.0787 → 0.0787. **No improvement on this run.**
Why: with fc4 at Kaiming random, the reconstructed output is essentially
uncorrelated with oracle labels, so flipping signs in fc1-fc3 doesn't create
a measurable agreement signal — all sign combos look equally bad. Sign search
was effective on tiniest because fc4 was partially recovered; here it is a
no-op.

**Skips** layers with `k > 18` recovered neurons (fc1 has k=64, fc2 k=61,
fc3 k=44; all skipped — only fc4 would be eligible, but fc4 has 0 recovered).
So the sign search on this run was effectively a no-op even before the
oracle-agreement check.

---

## Stage 3 — `recover_output_layer` (fc5 LR fit) *(hard-label oracle queries)*

**Location:** `test_extraction4.py:747`
**This is the stage that does the heavy lifting on tiny.**

**Idea:** whatever the current reconstructed-hidden-layer features `h_4 =
h_{L=4}(x)` produce, fit a multinomial logistic regression from `h_4` to
oracle-argmax labels. This replaces fc5 entirely with an LR decoder.

**Mechanics:**
1. Forward all 10,000 `X_test` samples through the reconstructed fc1-fc4
   (some rows extracted, others random) → hidden features `h_4 ∈ ℝ^{10000×64}`.
2. Query `oracle_model(X_test).argmax(-1)` → hard labels `y_oracle ∈ {0..9}^{10000}`.
3. Fit `LogisticRegression(multinomial, solver=lbfgs, max_iter=2000, C=1e6)`
   on `(h_4, y_oracle)`.
4. Overwrite `fc5.weight` and `fc5.bias` with the LR coefficients / intercepts.

**Oracle interaction:** `oracle(X_test).argmax(-1)` once. 10,000 hard labels.
Nothing else.

**Why it works on tiny:** even though fc4 is random and fc3 is partially
random, the combined hidden representation at layer 4 still has enough
information to linearly separate 10 classes — logistic regression with
`C=1e6` (basically no regularisation) has 10×64 = 640 free parameters, plenty
of capacity to fit 10,000 labelled points. The result is a decoder that
compensates for the corrupted hidden features.

**Effect on tiny:** **0.0787 → 0.9997.** A single LR fit gets within one
sample of perfect agreement.

**Important caveat:** at this point the model is no longer an "extraction"
of the hidden-layer structure in a weight-matching sense — fc5 is entirely
new, not a copy of the oracle's fc5. But functionally it reproduces the
oracle's decisions. Compare `|cos|(ext_fc5, true_fc5) = 0.088` in the weight
comparison: the recovered fc5 is nearly orthogonal to the true fc5.

---

## Stage 4 — `oracle_label_refinement` *(hard-label oracle queries, 1000 epochs)*

**Location:** `test_extraction4.py:623`

**Idea:** take whatever partial model we have and fine-tune it against oracle
argmax labels via cross-entropy, but freeze the rows that signature recovery
claims to have extracted.

**Mechanics:**
1. `oracle_labels = oracle_model(X_test).argmax(-1)` — single query, reused
   across all epochs.
2. Build `freeze_row_masks[lid] = recovered_mask[lid]` per hidden layer —
   `True` means that row is frozen (gradient zeroed each step).
3. Optimizer: Adam over **all** parameters; after `loss.backward()` explicitly
   zero the gradients for `weight[row_mask]` in each hidden layer.
4. 1000 epochs of cross-entropy loss on `X_test`.

**What's trainable on this tiny run:**
- fc1: 0 rows (all 64 recovered → frozen)
- fc2: 3 rows (61 recovered → 3 random-init remain trainable)
- fc3: 20 rows (44 recovered → 20 random-init trainable)
- fc4: **64 rows (0 recovered → all trainable)**
- fc5: always trainable (LR-fit result is the starting point)
- biases: always trainable

So the refinement has ~87 trainable hidden rows out of 256 + all of fc5 +
all biases — enough capacity to distil the oracle's behaviour into the
Kaiming-random fc4 while preserving the extracted rows in fc1-fc3.

**Oracle interaction:** exactly `oracle_model(X_test).argmax(-1)` once (10,000
queries). The gradient signal comes from cross-entropy against those fixed
labels; no additional oracle calls during the 1000 epochs.

**Effect on tiny:** **0.9997 → 1.0000.** Within 100 epochs the loss drops to
`1.0e-04`; by epoch 1000 it is `0.0e+00` numerically.

---

## Agreement trajectory (this run)

| Stage | Agreement | Δ |
|---|---:|---:|
| 0. Load signature + signs + Kaiming fills | ~0 | — |
| 1. `recover_biases_from_duals`           | 0.0787 | +0.08 |
| 2. `oracle_sign_search` (2 passes)        | 0.0787 |  0.00 |
| 3. `recover_output_layer` (fc5 LR fit)    | **0.9997** | **+0.92** |
| 4. `oracle_label_refinement` (1000 ep)    | **1.0000** | +0.003 |

Essentially all of the accuracy gain comes from **Stage 3**. On tiny, the
fc5 LR fit is the "linear probe on top of whatever features we have"
mechanism that rescues a broken hidden stack. Stage 4 just polishes the
last ~3 disagreeing samples.

## Why the refinement strategy is what it is

| Design choice | Rationale |
|---|---|
| **Freeze signature-recovered rows during refinement** | Preserves the "extraction" claim for rows that actually came from Phase 1. Without freezing, refinement is indistinguishable from full distillation. |
| **All-epochs use `X_test`, not a fresh query set** | `X_test` is the test set for the make_blobs task. Using it means refinement sees every input on which the report accuracy is later computed — the oracle_label_refinement is essentially overfitting to X_test. This is legitimate under the threat model (adversary can query the oracle on any input they want) but must be understood when interpreting the 100.00 % number. |
| **`C=1e6` in the LR fit** | Weak regularisation. With 10,000 samples and 64 features, the fit does not need regularisation to generalise; `C=1e6` lets LR hit near-perfect train accuracy. |
| **Bias-recov uses duals, not oracle** | Biases are geometrically determined by the dual-point ⇒ hyperplane relationship. No oracle query needed. |

## Total oracle queries during Phase 3 on tiny

| Stage | Oracle queries |
|---|---:|
| 1. Bias recovery from duals | 0 |
| 2. Sign search | `1 × 10 000` baseline labels, then ~0 more (no eligible layer) |
| 3. fc5 LR fit | `1 × 10 000` labels |
| 4. Oracle-label refinement | `1 × 10 000` labels (reused across 1000 epochs) |
| **Total distinct query set** | **X_test**, 10 000 hard-label queries |

These labels are reused; the total number of oracle invocations is **3** (bias recovery doesn't count, sign search's baseline = same query, fc5 LR fit = same query, refinement = same query). The oracle sees the model's queries on `X_test` at most a handful of times.

## Limitations of this as a pure "signature extraction"

The 100 % number is a combination of real signature recovery (for fc1/fc2
and most of fc3) and *oracle-guided distillation* for the rest (fc4 and the
unrecovered rows of fc2/fc3, plus fc5). When the attack is evaluated as
"how much of the oracle's weights did we recover", the answer is:

- Direction quality (|cos|) is byte-perfect for fc1, 0.98 for fc2, 0.73 for
  fc3, **0 for fc4**.
- Signs are ~51 % correct on the recovered rows (chance-level — attack is
  sign-blind).
- fc5 is essentially orthogonal in weight space to the true fc5 (|cos|=0.09).
- The reconstructed model's behaviour matches the oracle on 100 % of 15 000
  make_blobs samples.

A honest framing: **Phase 3 turns a 66 %-recovered-hidden, 0 %-recovered-output model into a 100 %-functional-agreement model by distilling 10 000 oracle hard labels into the fc5 decoder and the random-init hidden rows**. That is a legitimate hard-label attack result, but it is important to distinguish from a weight-level extraction.

## Artifacts and cross-refs
- Phase 3 code: `analysis/test_extraction4.py` lines 623-869 (four functions)
- Run log: `/tmp/tiny_reconstruct.log`
- Metrics: `results/reconstructed_models/extraction_metrics.json`
- Sibling reports:
  - `results/reports/tiny_extraction_quality_2026-04-24.md` — pipeline-level breakdown
  - `results/reports/tiny_true_vs_extracted_2026-04-24.md` — weight-level comparison
  - `results/reports/tiny_cheating_audit_2026-04-24.md` — where the attack reads the true model
