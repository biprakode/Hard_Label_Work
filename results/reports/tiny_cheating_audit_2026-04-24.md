# Tiny — Cheating Audit

**Date:** 2026-04-24
**Run:** `python3 analysis/test_extraction4.py --makeblobs --from-scratch --refine --refine-epochs 1000`
**Target:** tiny 64→64→64→64→64→10 make_blobs model

## TL;DR
The Phase-3 reconstruction on this run is **hard-label-clean** — `--from-scratch` gates off the two bias/fc5 cheats and the only oracle-queries beyond that are `oracle(X_test).argmax(-1)`, which is the legal hard-label API.

Phase 1 (signature recovery) and Phase 2 (sign recovery) still read the true model in the same places as the vanilla EUROCRYPT-2024 reference code. These are inherited cheats, not introduced by the current workflow, but they are present and should be called out.

- **Phase 1 (signature recovery):** whitebox-assisted via `signature_recovery/utils.py` (`cheat_*`, `cheat_solution`)
- **Phase 2 (sign recovery):** whitebox-assisted via `sign_recovery/whitebox.py`
- **Phase 3 (reconstruction + refinement):** **fully hard-label** on this run (oracle argmax only)

## Cheat inventory for this tiny run

Legend:
- **W** = reads true weights or biases directly
- **A** = reads true hidden activations (needs internal state, not hard labels)
- **T** = reads true class label (not a cheat for this attack since the oracle is 100 % on make_blobs: `oracle argmax == true label`)
- **E** = used only for grading, does not affect the extraction output

### Phase 1 — `signature_recovery/` (whitebox-assisted, inherited from vanilla)

| File | Line(s) | What | Type | Severity | Used on this run? |
|---|---|---|---|---|---|
| `utils.py` | 163–171 | Loads entire true `makeblobs_relu.pth` as `cheat_net_cpu` / `cheat_net_cuda`, exposes true weights via `cheat_solution` | **W, A** | **HIGH** | **Yes** — loaded at import time |
| `utils.py` | 180–194 | `cheat(x)`, `cheat_cuda(x)`, `gap(x)`, `gapt(x)` return true hidden activations / true logit gap. Guarded by `DEBUG` but `DEBUG=True` | **W, A** | HIGH | **Yes** — `gapt()` called inside `find_duals.is_on_decision_boundary_cheat` |
| `utils.py` | 208–227 | `cheat_neuron_diff(a,b)`, `cheat_neuron_diff_cuda(a,b)`, `cheat_num_flips(a,b)` — compare true-activation sign between two inputs | **A** | HIGH | **Yes** — `cheat_neuron_diff_cuda` called 10,037,894 times during streaming clustering (one per dual triplet) |
| `find_duals.py` | 4–16 | `is_on_decision_boundary` calls `is_on_decision_boundary_cheat` → `gapt(...)` (true logit gap) for boundary detection | **W** | **HIGH** | **Yes** — every dual-point search iteration; ran 1000 × ~250 calls = ~250k boundary tests |
| `find_duals.py` | 20–50 | `refine_to_decision_boundary_cheat` uses gradient of true logits (`gapt(..., grad=True)`) to Newton-refine points onto the hyperplane | **W** | **HIGH** | **Yes** — every boundary refinement |
| `cluster_dual_points_stream.py` | (this run's streaming variant) | `cheat_neuron_diff_cuda(left, right)` — the clustering signal: which true neuron toggled between two inputs. **The** signal that groups duals by neuron. | **A** | **CRITICAL** | **Yes** — ran on all 10 M triplets |
| `recover_weights.py` | 222 | `transfer_weights(cheat_net_cpu, prefix)` — **prefix network is initialised with true lower-layer weights** before forward-propagating candidate dual points | **W** | **CRITICAL** | **Yes** — ran 4× (once per hidden layer) |
| `recover_weights.py` | 255 | `factor = np.median(soln / cheat_solution[LAYER][maybe_neuron, :])` — computes scaling factor by **dividing extracted solution by the true weight vector**. Both magnitude and sign of `factor` leak. | **W** | **CRITICAL** | **Yes** — ran for every successfully-extracted cluster |
| `recover_weights.py` | 256–257 | `errs.append(|soln / factor - cheat_solution[LAYER][maybe_neuron, :]|)` — picks which true neuron each cluster corresponds to by minimising distance to true weights | **W** | **HIGH** | **Yes** — 169 cluster→neuron mappings produced this way |

Mitigation partially present:
- `test_extraction4.py:261` applies `abs(factor)` at load time — kills the sign leak from `cheat_solution` at the downstream consumer. The scaling **magnitude** still leaks through the saved `scaling_factor` in `metadata.json`, and the cluster → true-neuron mapping still leaks (directory names use the matched true-neuron id).

**Count on this run:** 10,037,894 calls to `cheat_neuron_diff_cuda` in clustering; ~250,000 calls to `gapt` during find_duals; ~250 calls to `transfer_weights(cheat_net_cpu, ...)` in recover_weights.

### Phase 2 — `sign_recovery/` (whitebox-assisted, inherited from vanilla)

| File | Line(s) | What | Type | Severity | Used on this run? |
|---|---|---|---|---|---|
| `sign_recovery.py` | 41 | `import whitebox` | — | HIGH | **Yes** |
| `sign_recovery.py` | 302 | `whitebox.getSignatures(model, layerId)` — returns true weights of lower layers after "simulating" signature recovery (= copying them) | **W** | HIGH | **Yes** — called inside `get_dx` for every experiment; runs 256 × (~1000 exps) times in total |
| `sign_recovery.py` | 728 | `whitebox.getWeightsAndBiases(model, ...)` — reads true weights & biases of entire keras model | **W** | HIGH | **Yes** — called once per neuron at setup |
| `whitebox.py` | (entire file) | All functions read `model.layers[i].get_weights()` | **W** | — | **Yes** — Phase-2 scaffolding |
| `blackbox.py` | (entire file) | Coordinate transforms using *passed* weights — blackbox *if* passed weights are reconstructed, not true | — | LOW | Passed weights are true (from `getWeightsAndBiases(keras_model, ...)`), so blackbox routines are effectively whitebox this run |

**Impact on tiny:** Phase 2's decision-boundary walks are parameterised by true lower-layer weights. When lower-layer extraction has errors (e.g. fc3 `|cos|=0.73`, fc4 not recovered at all), the vanilla sign-recovery would cascade those errors. By using truth here, Phase 2 is **given an easier problem than it would face in a real attack.** The 226 / 256 neurons that got a sign out of Phase 2 would almost certainly be fewer if the Phase-1 reconstructed weights were fed in instead.

### Phase 3 — `analysis/test_extraction4.py` (hard-label clean on this run)

Gates are active because of `--from-scratch`.

| Line(s) | What | Type | Severity | Active on this run? |
|---|---|---|---|---|
| 146–196 | `load_ground_truth_model` — loads true `makeblobs_relu.pth` into memory | **W** | — | **Loaded** but used only in ways listed below |
| 333–403 | `compute_weight_metrics_v2(extracted, true)` — per-neuron cos/rel-err | **E** | none | **Yes — grading only**, does not flow back into the reconstruction |
| 502–504 | `true_model = load_ground_truth_model(...)` | **W** | — | Loaded unconditionally |
| 546–548, 571 | `true_weights = true_layer.weight.data.numpy()` for metrics | **E** | none | Grading only |
| **581–584** | `layer.bias.data = true_layer.bias.data.clone()` when `copy_true_biases=True` | **W, writes into reconstruction** | HIGH when active | **Gated off by `--from-scratch`** ✓ |
| **590–593** | `model.fc5.weight.data = true_model.fc5.weight.data.clone()` when `copy_true_output=True` | **W, writes into reconstruction** | HIGH when active | **Gated off by `--from-scratch`** ✓ |
| 1105–1106 | `true_accuracy = test_model_accuracy(true_model, ...)` | **E** | none | Grading only |
| 1147–1157 | `oracle_sign_search`, `recover_output_layer` receive `true_model` — used only for `oracle(X).argmax(-1)` | **T** (hard-label equivalent) | none | **Yes — hard-label only** ✓ |
| 1173–1196 | Per-layer cos/rel-err metrics against true weights | **E** | none | Grading only |
| 1201 | `true_preds = true_model(X_test).argmax(...)` — for agreement metric | **T** | none | Grading only |
| 623–692 | `oracle_label_refinement` uses `oracle_model(X).argmax(-1)` | **T** (hard-label) | none | **Yes — hard-label only** ✓ |
| 707–744 | `recover_biases_from_duals` — uses dual points and the reconstructed (not true) model | — | none | **Yes — not a cheat, no oracle call** |

**Count with `--from-scratch`:** 0 weight / bias values copied into the reconstruction. All `true_model(...)` calls are `.argmax(...)` → equivalent to an oracle-argmax API. All cos / rel-err readouts are post-hoc grading.

Phase 3 on this tiny run is **hard-label-clean**.

## Oracle-query budget on this tiny run (the hard-label-legal queries)

| Phase-3 consumer | Oracle invocations | Queried inputs |
|---|---:|---|
| `oracle_sign_search` baseline labels | 1 | `X_test` (10 000) |
| `oracle_sign_search` candidate-combo evaluation | ~0 (no eligible layer) | — |
| `recover_output_layer` fc5 LR fit | 1 | `X_test` (10 000) — same labels reused |
| `oracle_label_refinement` 1000-epoch fine-tune | 1 | `X_test` (10 000) — same labels reused, reused across all epochs |
| **Total distinct query set** | **X_test — 10 000 inputs, 3 distinct invocations** | — |

The oracle is queried only on the already-public `X_test` makeblobs set, and only for `argmax` labels. This is as conservative a hard-label access pattern as possible.

## Where the tiny run differs from the tiniest run

Almost everywhere, the cheat pattern is identical (same source files). The scale differs:

| Metric | Tiniest | Tiny |
|---|---:|---:|
| `cheat_neuron_diff_cuda` calls in clustering | ~90 k | 10,037,894 |
| `gapt` calls in find_duals | ~5 k | ~250 k |
| `transfer_weights(cheat_net_cpu, prefix)` calls | ~120 | ~250 |
| `whitebox.getSignatures` calls in Phase 2 | ~2500 (32 × ~80 exps) | ~15 k (256 × ~60 exps) |
| Phase-3 oracle queries (distinct invocations) | 3 | 3 |
| Phase-3 oracle inputs | 2000 (tiniest X_test) | 10000 (tiny X_test) |
| Bias/fc5 copies when `--from-scratch` | 0 | 0 |

Phase 3 stays clean at any scale because the cheats are gated off symbolically by `--from-scratch` and the oracle-argmax pattern is the same.

## What would make Phase 1 + Phase 2 hard-label-clean

Same list as in the tiniest cheating audit:

| Cheat | Black-box replacement | Estimated cost |
|---|---|---|
| `gapt(x)` for boundary detection | Walk until `argmax(oracle(x))` changes, binary-search for exact toggle | ~2× more queries, no numerical change |
| `cheat_neuron_diff(a, b)` clustering | `is_consistent()`-style SVD consistency test on triplets (already in `cluster_slow`, just slower) | 10-100× more compute, no truth reads |
| `cheat_solution[LAYER][n]` for scaling | Fix scaling arbitrarily (e.g. `‖w‖=1`), absorb the gauge in fc5 LR fit | 0 query cost, needs LR on every layer |
| `cheat_solution[LAYER]` for cluster → neuron mapping | Arbitrary stable permutation — reconstructed model is isomorphic to the true one up to permutation anyway | 0 query cost, needs robust output-layer fitting |
| `whitebox.getWeightsAndBiases` in sign recovery | Pass Phase-1 reconstructed weights instead of true weights | 0 query cost, needs Phase-1 quality to be good |

None of these would change the *end-to-end accuracy numbers* for tiny qualitatively — Stage 3 of Phase 3 (the fc5 LR fit + refinement) will still close the agreement gap because it is genuinely black-box. What would change is the **signature-recovery quality** — the current `|cos|=0.98 / 0.73 / 0` for fc2 / fc3 / fc4 would likely get worse without the whitebox scaffolding, because Phase 2's sign recovery cascade would compound Phase 1's errors rather than restart from truth.

## Honest framing of this tiny attack

> **What is genuinely black-box on this run:**
> Phase 3 — the bias recovery from duals (no oracle), sign search (hard-label), fc5 LR fit (hard-label), and refinement (hard-label) — is the novel contribution and is clean. It turns a 8 %-agreement partially-extracted model into a 100 %-agreement model using only `oracle(X_test).argmax(-1)`.
>
> **What is still whitebox-assisted:**
> Phase 1 and Phase 2, which produced the partially-extracted model that Phase 3 inherits, both read true weights and true activations through the `cheat_*` / `whitebox` scaffolding shipped with the EUROCRYPT reference code.
>
> **What this implies for the paper's threat model:**
> The pipeline-as-run is **not end-to-end hard-label**. It is end-to-end hard-label **conditional on** Phases 1 and 2 doing their job in a real attack. For tiny specifically, the inherited whitebox cheats in Phase 1 are what allow `find_duals` to terminate cleanly; without them, the `is_on_decision_boundary` walk would need the hard-label-only replacement (walk-until-argmax-change), which is ~2× slower but equivalent numerically.

## Files
- Reports cross-reference:
  - `results/reports/tiny_extraction_quality_2026-04-24.md` — what the pipeline produced
  - `results/reports/tiny_true_vs_extracted_2026-04-24.md` — weight-level comparison
  - `results/reports/tiny_refinement_mechanism_2026-04-24.md` — how Phase 3 closes the gap
  - `results/reports/cheating_audit_2026-04-23.md` — the equivalent audit for tiniest
- Verify Phase-3 cleanliness on this run:
  `grep -n "true_model\|cheat\|copy_true" analysis/test_extraction4.py`
- `utils.py` `DEBUG` flag gates Phase-1 cheats on/off (line 16). Currently `True`.
