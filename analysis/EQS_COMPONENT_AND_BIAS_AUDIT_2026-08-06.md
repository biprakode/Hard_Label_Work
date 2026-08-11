# EQS component breakdown + bias-recovery audit — 2026-08-06

Read-only audit. No code was changed; the only computation run was a comparison of two
already-saved CIFAR checkpoint pairs (§4). Written to support the composite-EQS table in
the paper (`Paper tex files/cscml26/`, `\label{tab:eqs}`).

Companion documents:
- `paper_notes/section3/EQS_component_audit.md` — the *formulas* for C1/C2/C3/S, line-referenced.
- `paper_notes/section3/3_9_scorecard_results.md` — the aggregated scorecard results.

This file holds the *numbers*: the per-component decomposition behind every row of the paper
table, the three parts of S, and what is and is not available for a bias relative-error column.

---

## 1. Provenance — which scorecard backs which table row

All paths relative to `enhanced_codebase/paper_notes/section3/reports/`.

| Table row | Scorecard |
|---|---|
| tiniest_relu, tiniest_leakyrelu, tinier_relu, tinier_leakyrelu, tiny_leakyrelu | `2026-06-21/<model>_sa_margin_eval_scorecard.md` |
| tiny_relu (SA) / (PT) | `2026-06-21/tiny_relu_{sa,pt}_margin_eval_scorecard.md` |
| cifar_relu | `cifar_gce_relu_2026-06-25/relu/eqs/relu_eval_scorecard.md` |
| cifar_leakyrelu | `cifar_kaggle_2026-06-26/full_leakyrelu_eval_scorecard.md` |

The `sa_margin` and `pt_margin` scorecards are identical in every EQS field for every model
**except tiny_relu**. The SA/PT distinction therefore only produces a distinct table row for
that one model, where it moves C2 (0.4335 → 0.4900) and S (0.6819 → 0.6871).

---

## 2. EQS component values (variant `structural`)

Raw values in [0,1], reported as **extraction / distillation**.

| Model | C1 | C2 | C3 | S | EQS ext | EQS dis |
|---|---|---|---|---|---|---|
| tiniest_relu | 0.9930 / 0.9610 | 0.2166 / 0.3304 | 0.9910 / 0.9731 | 0.7421 / 0 | 69.6 | 54.4 |
| tiniest_leakyrelu | 1.0000 / 0.9860 | 0.5880 / 0.3756 | 1.0000 / 1.0000 | 0.8507 / 0 | 83.9 | 57.0 |
| tinier_relu | 1.0000 / 1.0000 | 0.5297 / 0.5295 | 1.0000 / 1.0000 | 0.6217 / 0 | 76.7 | 62.1 |
| tinier_leakyrelu | 1.0000 / 1.0000 | 0.6123 / 0.5132 | 1.0000 / 1.0000 | 0.6959 / 0 | 81.0 | 61.6 |
| tiny_relu (SA) | 1.0000 / 1.0000 | 0.4335 / 0.3338 | 1.0000 / 1.0000 | 0.6819 / 0 | 75.2 | 56.1 |
| tiny_relu (PT) | 1.0000 / 1.0000 | 0.4900 / 0.3338 | 1.0000 / 1.0000 | 0.6871 / 0 | 77.0 | 56.1 |
| tiny_leakyrelu | 1.0000 / 1.0000 | 0.2835 / 0.3741 | 1.0000 / 1.0000 | 0.7943 / 0 | 73.2 | 57.3 |
| cifar_relu | 0.5640 / 0.5033 | 0.3365 / 0.2173 | 0.6012 / 0.5271 | 0.7102 / 0 | 53.6 | 30.2 |
| cifar_leakyrelu | 0.5829 / 0.5286 | 0.3383 / 0.2246 | 0.6292 / 0.5486 | 0.8357 / 0 | 57.7 | 31.5 |

### Weighted points

Weights `{C1:22, C2:26, C3:17, S:20}`, sum 85, renormalised by 100/85 to
**25.88 / 30.59 / 20.00 / 23.53** (`analysis/extraction_pipeline/eval_metrics.py:445-448`;
applied in `compute_eqs` :464-487). Points are `weight_renormalized x clip(value,0,1)`.

| Model | C1 pts | C2 pts | C3 pts | S pts | = ext | dis |
|---|---|---|---|---|---|---|
| tiniest_relu | 25.7 / 24.9 | 6.6 / 10.1 | 19.8 / 19.5 | 17.5 / 0 | 69.6 | 54.4 |
| tiniest_leakyrelu | 25.9 / 25.5 | 18.0 / 11.5 | 20.0 / 20.0 | 20.0 / 0 | 83.9 | 57.0 |
| tinier_relu | 25.9 / 25.9 | 16.2 / 16.2 | 20.0 / 20.0 | 14.6 / 0 | 76.7 | 62.1 |
| tinier_leakyrelu | 25.9 / 25.9 | 18.7 / 15.7 | 20.0 / 20.0 | 16.4 / 0 | 81.0 | 61.6 |
| tiny_relu (SA) | 25.9 / 25.9 | 13.3 / 10.2 | 20.0 / 20.0 | 16.0 / 0 | 75.2 | 56.1 |
| tiny_relu (PT) | 25.9 / 25.9 | 15.0 / 10.2 | 20.0 / 20.0 | 16.2 / 0 | 77.0 | 56.1 |
| tiny_leakyrelu | 25.9 / 25.9 | 8.7 / 11.4 | 20.0 / 20.0 | 18.7 / 0 | 73.2 | 57.3 |
| cifar_relu | 14.6 / 13.0 | 10.3 / 6.6 | 12.0 / 10.5 | 16.7 / 0 | 53.6 | 30.2 |
| cifar_leakyrelu | 15.1 / 13.7 | 10.3 / 6.9 | 12.6 / 11.0 | 19.7 / 0 | 57.7 | 31.5 |

### Three observations that the composite hides

**(a) The distillation baseline is structurally capped at 76.5/100.** S is 0 for distillation by
construction (no parameters to compare against), so 23.53 points are unreachable for it. Part of
every EQS gap in the paper table is definitional. The *earned* gap — C1+C2+C3 only — is:

| Model | earned gap (pts) |
|---|---|
| tiniest_relu | +0.6 |
| tiniest_leakyrelu | +6.9 |
| tinier_relu | +0.0 |
| tinier_leakyrelu | +3.0 |
| tiny_relu (SA) | +3.1 |
| tiny_relu (PT) | +4.8 |
| tiny_leakyrelu | **−2.7** |
| cifar_relu | +6.7 |
| cifar_leakyrelu | +6.5 |

On tiny_leakyrelu the extraction arm is *behind* the distillation baseline on black-box
behaviour; its reported +15.9 comes entirely from S.

**(b) C1 and C3 are saturated on four models.** Both are exactly 1.0000 for extraction *and*
distillation on tinier_relu, tinier_leakyrelu, tiny_relu and tiny_leakyrelu — 45.9 points that
discriminate nothing. Only C2 and S move on those rows. The cifar rows are the only ones where
C1/C3 carry signal.

**(c) The paper's "off-manifold gap" column is the uniform probe alone, not C2.** C2 averages
the uniform and wide-Gaussian pools (`eval_metrics.py:229`, divisor 2). Examples:

| Model | uniform gap | wide-Gaussian gap |
|---|---|---|
| tiniest_relu | −10.98 pt | −11.78 pt |
| tiny_relu (PT) | +20.44 pt | +10.80 pt |
| cifar_relu | +14.58 pt | +9.26 pt |
| cifar_leakyrelu | +13.26 pt | +9.48 pt |

Interpolation-path agreement is computed and printed in the scorecards but deliberately excluded
from C2. Worth stating in the table caption, or a reader will try to reconcile that column
against C2 and fail.

---

## 3. The three parts of S

`eval_metrics.py:417-419` — an unweighted mean of three quantities, no weighting, no clipping:

```python
parts = [v for v in (mean_cos, mean_sign, coverage) if v is not None]
structural_score = float(np.mean(parts)) if parts else None
```

| Model | mean \|cos\| | mean sign-acc | coverage | S | recovered |
|---|---|---|---|---|---|
| tiniest_relu | 1.0000 | 0.6012 | 0.6250 | 0.7421 | 20/32 |
| tiniest_leakyrelu | 1.0000 | 0.7708 | 0.7812 | 0.8507 | 25/32 |
| tinier_relu | 1.0000 | 0.3472 | 0.5179 | 0.6217 | 29/56 |
| tinier_leakyrelu | 1.0000 | 0.4627 | 0.6250 | 0.6959 | 35/56 |
| tiny_relu (SA) | 1.0000 | 0.5028 | 0.5430 | 0.6819 | 139/256 |
| tiny_relu (PT) | 1.0000 | 0.5184 | 0.5430 | 0.6871 | 139/256 |
| tiny_leakyrelu | 1.0000 | 0.5002 | 0.8828 | 0.7943 | 226/256 |
| cifar_relu | 1.0000 | 0.5307 | 0.5998 | 0.7102 | 499/832 |
| cifar_leakyrelu | 1.0000 | 0.5372 | 0.9700 | 0.8357 | 807/832 |

All nine reproduce to 4 dp, e.g. cifar_leakyrelu `(1.0000 + 0.5372 + 0.9700)/3 = 0.8357`.

**mean |cos| is 1.0000 on all nine models** — a fixed 1/3 of every S, i.e. a constant 7.84 EQS
points handed to the extraction arm on every row. It is 1.0000 because the mean is taken only
over matched neurons: `structural_metrics` skips the `_all` layer entries precisely because those
include the He-initialised unrecovered neurons (`eval_metrics.py:392-394`).

Consequence for the paper: S is a deterministic function of two columns already printed,
`S = (1 + sign-acc + coverage)/3`. It adds no independent evidence, and the constant 1.0000 term
inflates it. tinier_relu is the sharp case — sign accuracy 0.347 is *below* chance and coverage
is barely half, yet S = 0.6217 and contributes 14.6 points, because the |cos| term alone floors
S at 0.333. Either state the definition in the caption or report mean |cos| as its own column so
the floor is visible.

---

## 4. Bias recovery — what exists, and a null result on CIFAR

### 4.1 What is stored

`compare_true_vs_extracted_v2.py:241-243` writes three bias fields per layer into
`<model>_true_vs_extracted.json`:

```python
'b_l1_sum':       float(np.sum(np.abs(b_ext - b_true))),
'b_delta_median': float(np.median(np.abs(b_ext - b_true))),
'b_delta_max':    float(np.max(np.abs(b_ext - b_true))),
```

Two limitations, both load-bearing:

1. **Absolute, not relative.** No denominator anywhere in the file.
2. **Unmasked.** The weight metrics beside them are restricted to recovered neurons
   (`compare_layer` takes `idx = np.where(mask)[0]`, :124), but these three lines sit *outside*
   that loop and run over all `n` neurons in the layer — including the He-random-init ones. They
   are therefore not a bias-recovery measurement.

Stored `b_delta_median / b_delta_max`, per layer (fc1 → fc4):

| Model | fc1 | fc2 | fc3 | fc4 |
|---|---|---|---|---|
| tiniest_relu | 0.466 / 0.974 | 3.219 / 5.504 | 0.639 / 0.967 | 0.322 / 2.745 |
| tiniest_leakyrelu | 0.052 / 1.185 | 0.412 / 3.138 | 0.662 / 2.137 | 3.367 / 10.19 |
| tinier_relu | 0.212 / 0.840 | 0.473 / 2.743 | 0.247 / 0.448 | 0.099 / 0.280 |
| tinier_leakyrelu | 0.118 / 0.732 | 0.566 / 5.337 | 0.316 / 1.147 | 0.117 / 1.710 |
| tiny_relu (SA) | 0.081 / 0.488 | 0.611 / 1.943 | 0.116 / 1.399 | 0.092 / 0.168 |
| tiny_relu (PT) | 0.053 / 0.443 | 0.582 / 1.854 | 0.101 / 1.424 | 0.088 / 0.198 |
| tiny_leakyrelu | 0.094 / 0.485 | 1.029 / 2.099 | 0.430 / 7.660 | 0.454 / 1.902 |
| cifar_leakyrelu | 0.247 / 1.115 | 2.664 / 17.98 | 1.763 / 12.01 | 0.935 / 4.576 |

**cifar_relu is absent.** The only `full_relu_true_vs_extracted.json` is from
`cifar_kaggle_2026-06-22`, which recovered 0/832 neurons and is *not* the run in the paper table
(that is `cifar_gce_relu_2026-06-25`, 499/832).

### 4.2 What cannot be recomputed

The extracted checkpoints for the seven small-model rows are **gone**. Every
`true_vs_extracted.json` points at
`Hard_Label_Work/results/reconstructed_models/reconstructed_{tiniest,tinier,tiny}.pth`; that
directory now contains `.json` files only.

The masks are gone with them. `get_recovered_masks` (:108-119) reads the live scratch directory
`signature_recovery/outputs/model_weights/Vrelu`, which is overwritten by every run; it currently
holds a 64-input / 64-neuron model dated 2026-07-14, unrelated to any table row. The `.npz` files
there store a weight row only (`arr_0`, shape `(in_dim,)`) — no bias.

So a bias rel-err column for those seven rows requires re-running the extractions.

### 4.3 What can be recomputed: the two CIFAR rows

Both victims and both reconstructions survive:

| Model | victim | extracted |
|---|---|---|
| cifar_relu | `Hard_Label_Work/tiny_stuff/TinyModel_relu.pth` | `paper_notes/section3/reports/cifar_gce_relu_2026-06-25/relu/relu_reconstructed_full.pth` |
| cifar_leakyrelu | `Hard_Label_Work/tiny_stuff/TinyModel_leakyrelu.pth` | `paper_notes/section3/reports/cifar_kaggle_2026-06-26/full_leakyrelu_reconstructed_full.pth` |

The mask was reconstructed as `|cos(w_ext[i], w_true[i])| > 0.999` — reconstruction places a
recovered neuron at its own index (`compare_layer` records `'placed_at': int(i)`). This
reproduces the published figures exactly, which validates the method:

| Model | per-layer recovered | total | published coverage | sign-acc derived | published |
|---|---|---|---|---|---|
| cifar_leakyrelu | 254 / 252 / 243 / 58 | 807/832 | 97.00 % | 0.5372 | 0.5372 |
| cifar_relu | 252 / 247 / 0 / 0 | 499/832 | 59.98 % | 0.5307 | 0.5307 |

### 4.4 Result — biases are not recovered in these runs

Metric: **scale-free hyperplane offset**, `o = b/‖w‖`, with the extracted offset sign-aligned to
the weight row (`o_ext · sign(cos)`), so the per-neuron scale freedom and the sign ambiguity both
cancel. Restricted to recovered neurons. Median relative error:

| Model | fc1 | fc2 | fc3 | fc4 | all recovered |
|---|---|---|---|---|---|
| cifar_relu | 0.453 | 21.09 | — | — | **2.83** |
| cifar_leakyrelu | 1.957 | 19.31 | 7.539 | 4.893 | **5.76** |

Zero neurons, on any layer of either model, land below 1e-3.

This is **not** a scaling artefact. On the same neurons:

- `‖w_ext‖ / ‖w_true‖` median = 0.9999 on every layer of both models;
- the stored weight `rel_err_median` is 1e-4 on every layer;
- mean |cos| = 1.0000.

The weight rows, magnitude included, are essentially exact. The biases are wrong in *magnitude*,
not merely in sign: `|b_ext| / |b_true|` has median 1.83 on fc1 and **19.0** on fc2
(cifar_leakyrelu). Sign alignment does not rescue it — on fc1 the aligned median (1.96) is no
better than the raw one (1.87).

### 4.5 Before this goes in the paper

Adding the column as measured would report a negative result on bias recovery. That may be the
honest thing to publish, but it should first be established which of two things is true:

1. bias recovery genuinely fails in the Hard_Label_Work pipeline; or
2. these checkpoints predate working bias recovery and the metric is stale.

Note that the HLW2 canonical pipeline derives the bias jointly with its row under the same
*matched ∧ resolved* condition and rescales both by a single MAG_FIX factor (preserving the ReLU
hyperplane) — a different mechanism from whatever produced these checkpoints. The two are not
interchangeable evidence.

---

## 5. Reproducing §4.3–4.4

`python3` in the base environment has no torch; use the MLenv interpreter.

```bash
cd enhanced_codebase
~/miniconda3/envs/MLenv/bin/python3 - <<'PY'
import torch, numpy as np

def sd(p):
    o = torch.load(p, map_location='cpu', weights_only=False)
    if hasattr(o, 'state_dict'): o = o.state_dict()
    return {k: v.double().numpy() for k, v in o.items()}

PAIRS = [
 ('cifar_relu',
  'Hard_Label_Work/tiny_stuff/TinyModel_relu.pth',
  'paper_notes/section3/reports/cifar_gce_relu_2026-06-25/relu/relu_reconstructed_full.pth'),
 ('cifar_leakyrelu',
  'Hard_Label_Work/tiny_stuff/TinyModel_leakyrelu.pth',
  'paper_notes/section3/reports/cifar_kaggle_2026-06-26/full_leakyrelu_reconstructed_full.pth'),
]

for tag, tp, ep in PAIRS:
    T, E = sd(tp), sd(ep); print('==', tag); acc = []
    for L in range(1, 5):
        wt, we = T[f'fc{L}.weight'], E[f'fc{L}.weight']
        bt, be = T[f'fc{L}.bias'],   E[f'fc{L}.bias']
        nt, ne = np.linalg.norm(wt, axis=1), np.linalg.norm(we, axis=1)
        cos = (wt * we).sum(1) / (nt * ne)
        m = np.abs(cos) > 0.999                  # recovered-neuron mask
        if not m.sum():
            print(f'  fc{L}: none recovered'); continue
        o_t = bt[m] / nt[m]                      # scale-free offset
        o_e = (be[m] / ne[m]) * np.sign(cos[m])  # sign-aligned
        rel = np.abs(o_e - o_t) / np.maximum(np.abs(o_t), 1e-15)
        acc.append(rel)
        print(f'  fc{L} rec {m.sum():3d}/{len(m)}  sign+ {(cos[m] > 0).mean():.4f}  '
              f'|w| ratio {np.median(ne[m] / nt[m]):.4f}  offset relerr med {np.median(rel):.4g}')
    r = np.concatenate(acc)
    print(f'  ALL median {np.median(r):.4g}  frac<1e-3 {np.mean(r < 1e-3):.4f}')
PY
```

The interpreter emits a harmless `_distutils_hack` warning on stderr; filter it if it is noisy.
