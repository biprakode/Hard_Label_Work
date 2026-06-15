# Cheat Removal — Context & Plan

**Goal:** make the extraction attack treat the victim as a pure **black box** —
the only thing the attacker may learn from the victim is the **hard label**
`argmax(victim(x))`. No reading of true weights, true biases, true hidden
activations, or true logits/logit-gaps anywhere in the pipeline.

**Scope of this work:** tiniest (8-8-8-8-8-8), ReLU **and** LeakyReLU(0.01).
Everything lives in `enhanced_codebase/Hard_Label_Work/cheat_remove/`.

## Source of truth: the cheating audit

`results/reports/tiny_cheating_audit_2026-04-24.md` enumerates every whitebox
read. Summary of what is a cheat and the black-box replacement we implement:

| # | Cheat (file) | What it reads | Black-box replacement |
|---|---|---|---|
| 1 | `find_duals.py` boundary detection via `gapt`/`gap` (utils) | true logit gap + autograd gradient | argmax bisection for boundary points; finite-difference normal from argmax-only boundary searches (`get_gradient_dir` math) |
| 2 | `cluster_dual_points*.py` `cheat_neuron_diff_cuda` | true hidden-activation sign flip (which neuron toggled) | SVD-consistency grouping: two duals share a neuron iff the null-space test is consistent in the *recovered*-prefix space |
| 3 | `recover_weights.py` `transfer_weights(cheat_net_cpu,…)` | true lower-layer weights to build the prefix | **layer peeling**: prefix = identity for layer 0, then prefix built from *recovered* weights for deeper layers |
| 4 | `recover_weights.py` `cheat_solution` scaling + neuron match | true weight vectors (for scale + cluster→neuron id) | gauge fix `‖w‖=1`; arbitrary stable cluster ids (Phase 3 fc5 LR fit absorbs scale + permutation) |
| 5 | `sign_recovery/whitebox.py` (Phase 2) | true weights/biases | drop Phase 2 entirely; recover signs in Phase 3 via `oracle_sign_search` (already hard-label; tiniest layers are ≤8 wide so brute force 2^k is trivial) |

Phase 3 (`analysis/extraction_pipeline/`) is already hard-label-clean under
`--from-scratch` (audit §Phase 3). We reuse it unchanged for bias recovery,
sign search, fc5 LR fit, and refinement.

## The black-box boundary

`cheat_remove/bb_core.py` defines `Oracle`, which loads the victim `.pth` and
exposes **only** `label(X) -> argmax`. It counts queries. Nothing else in
`cheat_remove/` touches the victim's parameters or internal activations. The
true model is used in exactly one place — *grading* — and that path is clearly
marked and never feeds back into the extraction (same discipline as Phase 3's
post-hoc metrics).

## Pipeline (all argmax-only)

```
random argmax-differing points ─► bisect ─► boundary point
        └─ walk along tangent (finite-diff normal) ─► detect kink ─► dual point
                                   │
                                   ▼  (argmax-only triplets)
   peel layer 0 (prefix=Id) ─► SVD-consistency cluster ─► null-space weight
        └─ build prefix from recovered L0 ─► peel layer 1 ─► …
                                   │
                                   ▼  unsigned, ‖w‖=1, arbitrary ids
            Phase 3 (hard-label): bias recovery ─► oracle sign search
                       ─► fc5 LR fit ─► refinement ─► reconstructed model
```

## Files in this folder

- `CHEAT_REMOVE_CONTEXT.md` — this file (plan + cheat map).
- `bb_core.py` — `Oracle` (argmax-only) + black-box geometric primitives
  (boundary bisection, finite-difference normal, dual-point walk).
- `bb_find_duals.py` — produce dual-point triplets using only the oracle.
- `bb_recover.py` — layer-peeling SVD weight recovery (gauge-fixed, no truth).
- `bb_pipeline.py` — end-to-end driver: duals → peel-recover → Phase 3 → eval.
- `CHEAT_REMOVE_RESULTS.md` — measured results (filled after runs).

## Status

- [x] bb_core primitives validated (boundary, normal, dual points) — argmax only
- [x] batched bb find_duals, integrated with torch `parallel_duals` (`--impl blackbox`)
- [x] black-box SVD weight recovery validated on a pure cluster (|cos|=1.0)
- [x] consistency clustering separates neurons within a layer (~3 orders)
- [x] end-to-end tiniest ReLU — **99.65 %** agreement, fully black-box
- [x] end-to-end tiniest LeakyReLU(0.01) — **99.65 %** agreement, fully black-box
- [x] results written → `CHEAT_REMOVE_RESULTS.md`
- [~] **clean black-box layer separation** — OPEN (deeper neurons look globally
      linear over the make_blobs manifold; this is the hard core the
      `cheat_neuron_diff` shortcut hides). Peeling to layers 1–3 blocked on it.

See `CHEAT_REMOVE_RESULTS.md` for the full write-up and the honest threat-model
framing. The dominant whitebox cheats are removed and replaced with validated
argmax-only primitives; functional extraction is fully black-box.
