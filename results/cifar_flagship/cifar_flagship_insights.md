# CIFAR-10 Flagship Extraction — Insights & Results

Hard-label extraction attack run on the **flagship CIFAR-10 MLP**
`3072-256-256-256-64-10` (the model the original `vanilla_codebase` targets),
motivated by the question: *on a task hard enough that hard-label distillation
cannot trivially close the functional gap, does structural signature recovery
still help?*

**Headline result (ReLU):** Yes. On held-out CIFAR, structural recovery of the
shallow layers gives **+7.1 points of prediction agreement** over pure
distillation — and pure distillation badly **overfits the query set** (100 % on
the queried points, 44 % on held-out), which it did *not* do on the easy
make_blobs task (99.65 %). CIFAR is exactly the regime where structure matters.

> This run is the **whitebox-aided (vanilla) pipeline**, not `cheat_remove`.
> Clustering uses `cheat_neuron_diff` and recovery uses the true-weight prefix +
> `cheat_solution` matching, as in the codebase being reproduced. The only
> hard-label-clean stage is Phase-3 sign/output recovery.

---

## 1. Setup

| | |
|---|---|
| Architecture | `3072 → 256 → 256 → 256 → 64 → 10` (4 hidden layers, 832 hidden neurons) |
| Activations | ReLU (done) and LeakyReLU(0.01) (pending) |
| Victim training | CIFAR-10 (cached pickles), inputs `x/255*2-1 ∈ [-1,1]`, float64, no BN/dropout (pure piecewise-linear) |
| Victim test acc | ReLU **53.7 %**, Leaky **53.0 %** (raw-pixel MLP ceiling; train acc → 100 %, sharp boundaries) |
| Eval set | `X_test2` = held-out CIFAR **train** slice (10 K), never queried in Phase-3 |
| Host | 14-core CPU, **22 GB RAM (~10 GB usable, swap full)**, 109 GB disk |
| Dual search | parallelized torch rewrite (`parallel_duals.py --impl torch`), real-image boundary seeding |

## 2. Dual collection & clustering (ReLU)

| stage | number |
|---|---|
| Dual-search throughput | 66.8 triplets/s/proc; 5 workers → ~230/s |
| Rounds collected | 140 (≈68 min, ~56 GB on disk) |
| Triplets seen / single-flip kept | 807,542 / 119,230 |
| Per-neuron storage | 74 KB/triplet (3×3072×float64) |

**Cluster coverage** (whitebox `cheat_neuron_diff` labels each dual by the neuron
it toggles):

| Layer | neurons covered | quality |
|---|---|---|
| 0 | **256/256** | dense (identity prefix) |
| 1 | **250/256** | dense (1-layer prefix) |
| 2 | 245/256 covered, but **uncoverable** (see below) | |
| 3 | 53/64 covered, **uncoverable** | |

## 3. Signature recovery (ReLU) — 495/832 (60 %)

| Layer | recovered | why |
|---|---|---|
| 0 | **254/256** | identity prefix; SVD null-space clean |
| 1 | **241/250** | 1-layer ReLU prefix still well-conditioned |
| 2 | **0/256** | structural coverage failure |
| 3 | **0/64** | structural coverage failure (compounded) |

**Why deep layers give 0 (a result, not a bug):** recovery rejects a neuron's
cluster when any prefix-output coordinate is never active across its duals
(`min(hits)==0`). Diagnostics showed **8–60 % of the 256 upstream neurons are
inactive across a deep neuron's *entire* dual region** ("Mean OK 0.40–0.92"),
even after raising the budget 7× (50 → 350 duals/neuron). This is intrinsic:
deep neurons live where most upstream units are saturated-off, so their weight
components along those inputs are unobservable from boundary geometry. Matches
the documented baseline (tiniest: `fc3` recovers 0/8).

## 4. Phase-3 reconstruction — the core result

Phase-3 `--full --from-scratch --refine` (1000 epochs): geometric bias recovery
→ greedy hard-label sign search → fc5 logistic-regression fit on oracle labels →
frozen-recovered-row refinement. Evaluated on held-out `X_test2`.

| Metric (held-out CIFAR) | **with-signature** | no-sig baseline | victim |
|---|---|---|---|
| Prediction agreement | **51.42 %** | 44.32 % | — |
| Reconstructed accuracy | **44.86 %** | 38.43 % | 53.34 % |
| Agreement on the *queried* set (X_test) | ~100 % | 99.98 % | — |

**+7.1 pt agreement / +6.4 pt accuracy** from structural recovery, despite a
heavily handicapped sign step (below). Both pipelines fit the 10 K queries almost
perfectly but generalize far worse — the with-signature run because its frozen
true layer-0/1 features regularize it toward the victim's actual function; the
baseline because nothing constrains it.

## 5. Key insights (results-focused)

1. **Distillation overfits the query set on CIFAR.** Refinement reached **100 %
   agreement on the 10 K queried inputs by epoch 300**, yet only **44–51 % on
   held-out** data. The train↔held-out gap *is* the functional gap distillation
   leaves open on a hard task.

2. **Easy vs hard task is the whole story.** On make_blobs (prior work in this
   repo) hard-label Phase-3 distillation alone reaches **99.65 %** held-out, equal
   to with-signature — structure adds nothing because distillation already
   generalizes. On CIFAR the same Phase-3 only reaches **44 %**, and structure
   adds **+7 pt**. This is the empirical confirmation that *the value of
   cryptanalytic structural recovery shows up precisely when distillation
   can't generalize.*

3. **Structural recovery = generalizable regularization.** Freezing 495 *true*
   shallow-layer feature directions constrains the reconstruction to the victim's
   real early representation. The free-distillation baseline has more capacity,
   overfits more, and generalizes worse (44.3 % vs 51.4 %).

4. **The advantage is a lower bound — sign recovery is the weak link.** Greedy
   hard-label sign search on 256-wide layers reached only **22.8 %** agreement
   before fc5, so many frozen layer-0/1 rows carry *wrong signs* (which don't
   generalize). A proper Phase-2 sign recovery (dON/dOFF walks) would very likely
   widen the with-signature gap. Width, not the method, is the bottleneck: greedy
   flip-one is weak at k=256.

5. **Depth is the recovery frontier.** Shallow layers recover almost perfectly
   (99 %/94 %); deep layers (ReLU) recover 0 due to structural upstream
   inactivity. Raising the dual budget did not help — coverage is structural, not
   statistical. **LeakyReLU is the lever**: its recovery path skips the
   `min(hits)==0` rejection (the α·z leak keeps "off" coordinates informative),
   so a Leaky run is the way to actually recover layers 2–3. (Leaky run was set up
   and then stopped per request; this is the predicted next finding.)

6. **The 22 GB RAM box, not disk, is the real constraint** for the flagship.
   Per-neuron clustering and the recover SVD must be RAM-bounded
   (`full_matrices=False`, capped dicts, chunked processes); see
   `relu_error_report.md`.

## 6. Artifacts

```
results/cifar_flagship/
  relu_comparison.json        # the with-sig vs baseline numbers above
  relu_withsig_metrics.json   # full Phase-3 metrics (with signature)
  relu_baseline_metrics.json  # full Phase-3 metrics (no signature)
  relu_withsig_phase3.log     # full Phase-3 log (incl. refinement curve)
  relu_baseline_phase3.log
  relu_error_report.md        # incidents & fixes
  cifar_flagship_insights.md  # this file
```

## 7. Reproduce (ReLU)

```bash
# config: utils.py + extraction_pipeline/config.py  LEAKY_ALPHA=0.0,
#         utils.py TINY=False TINIER=False TINIEST=False MAKEBLOBS=False
python3 create_cifar_model.py                                   # train victims + stage data
cd signature_recovery
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 torch_impl/parallel_duals.py \
    --iterations 140 --workers 5 --batch-size 48 --target 4000 --impl torch
CLUSTER_PER_NEURON_CAP=150 python3 cluster_dual_points_stream.py
python3 generate_dual_neuron.py
# recover (chunked, RAM-safe): each layer in CLUSTER_START/END windows of 64
for L in 0 1 2 3; do CLUSTER_START=0 CLUSTER_END=999 python3 recover_weights.py $L; done
cd ..
python3 analysis/run_extraction.py --full --from-scratch --refine --refine-epochs 1000
# baseline:
python3 analysis/run_extraction.py --full --from-scratch --refine --refine-epochs 1000 \
    --signature-path /tmp/empty_sig
```

## 8. Open threads / next

- **Leaky run** (deep layers expected to recover) — set up, stopped on request.
- **Proper Phase-2 sign recovery** — would lift the with-signature number off its
  lower bound (needs the `.keras` export fix in `reexport_keras.py`).
- **Query diversity / early-stopping** in refinement — current 1000-epoch refine
  overfits the query set; fewer epochs or more/disperse queries may improve
  held-out generalization for both arms.
