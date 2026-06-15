# Cheat Removal — Approach (corrected peeling design)

This documents the **black-box layer-peeling** attack. The victim is touched
only through `bb_core.Oracle.label` (argmax hard label). No true weights,
biases, activations, or logits are ever read by the attack.

## Why peeling (and why signs must be recovered per layer)

Signature recovery of layer L pushes dual points through a **prefix** = the
network's layers 0…L-1. The original code cheats by loading the *true* weights
+ biases into that prefix (`transfer_weights(cheat_net_cpu, …)`). To remove the
cheat we must build the prefix from **recovered** layers — so layers must be
recovered **bottom-up (peeling)**.

Two facts force a per-layer sign step *inside* the peel:

1. **ReLU sign is not a free gauge.** `ReLU(z) ≠ ReLU(−z)`. Only the per-neuron
   *scale* is free (the next layer absorbs `diag(s)`, `s>0`). So to build the
   prefix for layer L+1 you need layer L's **signed** weights.
2. **Phase-3 `oracle_sign_search` needs the downstream layers** to be functional
   (it flips signs and measures *full-model* agreement). During the peel,
   layers L+1…output don't exist yet, so it cannot resolve layer-L signs. Signs
   must come from a **local** method that needs no downstream.

Therefore we do **not** "defer all signs to Phase 3". We keep a sign step, but
make it black-box: the legitimate Carlini Phase-2 algorithm (local dON/dOFF
decision-boundary walk) fed the **recovered** (not true) lower layers.

## Per-layer loop (bottom-up, L = 0,1,2,3)

```
prefix = recovered layers 0..L-1   (identity for L=0; no biases needed there)

1. SIGNATURE  (bb_recover) — black-box:
     dual points → cluster (SVD-consistency in prefix-output space)
     → SVD null-space per cluster → w_L directions, gauge ‖w‖=1, arbitrary ids
     Boundary normals via argmax finite-difference (bb_core.boundary_normal).

2. SIGN  (bb_sign) — black-box, local, no downstream:
     for each recovered neuron j, displace the dual to the +w_j and -w_j sides,
     walk along the OUTPUT decision boundary, measure distance to the next
     toggle; d_on vs d_off asymmetry → sign(w_j). Uses recovered prefix, argmax.

3. BIAS  (bb_bias) — black-box, geometric, no oracle:
     b_j = -median(w_j · prefix(x_dual)) over that neuron's duals, using the
     reconstructed (recovered) forward. (Same math as Phase-3
     recover_biases_from_duals, applied per layer during the peel.)

assemble signed+biased layer L  →  extend prefix  →  L+1
```

After all hidden layers are peeled:

```
4. OUTPUT + POLISH  (default Phase 3, already hard-label-clean):
     fc5 LR-fit on oracle argmax labels  →  frozen-recovered-row refinement
     (Adam CE vs oracle labels; recovered rows frozen, rest trainable).
     Optional final global oracle_sign_search as a cheap joint polish.
```

## What each black-box primitive replaces (vs the cheating original)

| Cheat (audit) | Black-box replacement | Module |
|---|---|---|
| `gapt`/`gap` boundary detection + autograd normal | argmax bisection + finite-diff normal | `bb_core` |
| `cheat_neuron_diff_cuda` clustering | SVD-consistency / projection-peak in prefix-output space | `bb_recover` |
| `transfer_weights(cheat_net_cpu, prefix)` (true prefix) | prefix built from **recovered** signed+biased layers (peeling) | `bb_recover` + `bb_peel` |
| `cheat_solution` scaling + neuron match | gauge `‖w‖=1`; arbitrary stable cluster ids (Phase-3 fc5 absorbs scale+perm) | `bb_recover` |
| `whitebox.getSignatures/getWeightsAndBiases` (Phase 2) | local dON/dOFF sign walk fed **recovered** lower layers | `bb_sign` |

## Gauge / correctness notes

- **Scale gauge `‖w‖=1` is consistent through the peel** *iff signs are correct*:
  with `s>0`, `ReLU(s·z)=s·ReLU(z)`, so the per-neuron scale passes through and
  is absorbed by the next layer's recovered direction. A wrong sign (`s<0`)
  breaks this — hence step 2 is mandatory before extending the prefix.
- **Biases needed before peeling deeper**: the prefix forward computes
  preactivations, which need `b`. Step 3 supplies them per layer.
- **Leaky ReLU(α)**: the dON/dOFF asymmetry weakens by ≈`(1-α)/(1+α)` (≈0.98 at
  α=0.01) — still resolvable. Signature recovery is actually *easier* under
  leaky (the α·z OFF leakage conditions the SVD), as already seen in Section 3.

## Modules

| File | Role | Status |
|---|---|---|
| `bb_core.py` | Oracle (argmax-only) + boundary/normal/dual primitives | ✅ done, validated |
| `bb_find_duals.py` | batched black-box dual finder (torch-harness compatible) | ✅ done, validated |
| `bb_recover.py` | signature recovery: clustering + SVD, peel-aware prefix | ✅ within-layer (SVD |cos|=1; layer separation open — see RESULTS) |
| `bb_sign.py` | local black-box sign recovery (dON/dOFF), recovered prefix | ⏳ deferred — only needed once layer separation works |
| `bb_bias.py` | per-layer geometric bias recovery from duals | ⏳ deferred — Phase 3 supplies biases end-to-end |
| `bb_peel.py` | orchestrator: per-layer signature→sign→bias, extend prefix | ⏳ blocked on clean layer-0 isolation |
| `bb_pipeline.py` | end-to-end: bb duals → bb layer-0 sig → Phase-3 → eval (relu+leaky) | ✅ 99.65 % both |

**Key finding (see `CHEAT_REMOVE_RESULTS.md`):** the corrected peeling design is
right, but its first prerequisite — cleanly isolating layer 0 black-box — does
not hold on this make_blobs victim, because deeper neurons appear *globally
linear over the data manifold* and are indistinguishable from layer-0 neurons by
input-space geometry. That is precisely the difficulty `cheat_neuron_diff`
hides. Functional extraction is nonetheless fully black-box via hard-label
Phase 3 (99.65 %, ReLU and Leaky).

## Validation gates (grading uses truth ONLY for scoring, never fed back)

1. L0 signature: 8/8 clusters, mean |cos| vs true W0 ≈ 1.0
2. L0 signs: sign accuracy vs truth
3. L0 bias: bias rel-err vs truth
4. Peel L1 with assembled L0: L1 |cos|
5. Full peel L0..L3 recovery summary
6. End-to-end functional agreement on X_test2 (tiniest ReLU and Leaky)

Results recorded in `CHEAT_REMOVE_RESULTS.md`.
