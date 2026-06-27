# Additive Phase-3 Ablation — make_blobs victims

_Generated 2026-06-27. Produced by `ablation_tiny/run_ablation.sh` → `ablation_harness.py` (read-only on all pipeline/method code) → `aggregate_ablation.py`._

Every number is an actual evaluation on the held-out **X_test3** (never queried; training pool = X_test ∪ X_test2). Stages are **cumulative**: stage *k* includes all components of stages < *k*. EQS is the structural-variant composite (C1=22, C2=26, C3=17, S=20).

## 1. Per-victim ablation tables

### tiniest_relu

_Victim `tiniest_makeblobs_relu.pth` · oracle acc (X_test3) = 99.90% · recovered 21/32 neurons · LeakyReLU α=0.0._

| Stage | Agreement | Ext acc | Sign acc | EQS | ΔAgreement | ΔEQS |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 — RAW (Phase 1+2 load) | 12.40% | 12.30% | 39.02% | 23.6 | n/a | n/a |
| Stage 1 — + BIAS recovery | 22.80% | 22.70% | 39.02% | 28.9 | +10.40 | +5.3 |
| Stage 2 — + LR FIT (fc5) | 92.40% | 92.40% | 39.02% | 66.8 | +69.60 | +37.9 |
| Stage 3 — + SIGN SEARCH (SA+margin) | 99.40% | 99.40% | 39.02% | 70.7 | +7.00 | +3.9 |
| Stage 4 — + FROZEN REFINE (full pipeline) | 99.70% | 99.70% | 39.02% | 71.2 | +0.30 | +0.5 |
| Distillation baseline (non-staged) | 96.00% | 95.90% | n/a | 54.4 | — | — |

_Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +69.60 pp, ΔEQS +37.9._

### tiniest_leakyrelu

_Victim `tiniest_makeblobs_leakyrelu.pth` · oracle acc (X_test3) = 100.00% · recovered 23/32 neurons · LeakyReLU α=0.01._

| Stage | Agreement | Ext acc | Sign acc | EQS | ΔAgreement | ΔEQS |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 — RAW (Phase 1+2 load) | 12.50% | 12.50% | 66.67% | 30.9 | n/a | n/a |
| Stage 1 — + BIAS recovery | 2.80% | 2.80% | 66.67% | 24.4 | -9.70 | -6.4 |
| Stage 2 — + LR FIT (fc5) | 98.00% | 98.00% | 66.67% | 70.9 | +95.20 | +46.5 |
| Stage 3 — + SIGN SEARCH (SA+margin) | 97.90% | 97.90% | 66.67% | 70.8 | -0.10 | -0.1 |
| Stage 4 — + FROZEN REFINE (full pipeline) | 98.90% | 98.90% | 66.67% | 71.5 | +1.00 | +0.7 |
| Distillation baseline (non-staged) | 98.60% | 98.60% | n/a | 57.1 | — | — |

_Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +95.20 pp, ΔEQS +46.5._

### tinier_relu

_Victim `tinier_makeblobs_relu.pth` · oracle acc (X_test3) = 100.00% · recovered 29/56 neurons · LeakyReLU α=0.0._

| Stage | Agreement | Ext acc | Sign acc | EQS | ΔAgreement | ΔEQS |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 — RAW (Phase 1+2 load) | 23.30% | 23.30% | 48.08% | 32.5 | n/a | n/a |
| Stage 1 — + BIAS recovery | 0.00% | 0.00% | 48.08% | 17.5 | -23.30 | -15.0 |
| Stage 2 — + LR FIT (fc5) | 99.94% | 99.94% | 48.08% | 75.6 | +99.94 | +58.0 |
| Stage 3 — + SIGN SEARCH (SA+margin) | 100.00% | 100.00% | 51.92% | 79.6 | +0.06 | +4.0 |
| Stage 4 — + FROZEN REFINE (full pipeline) | 100.00% | 100.00% | 51.92% | 81.3 | +0.00 | +1.7 |
| Distillation baseline (non-staged) | 100.00% | 100.00% | n/a | 60.7 | — | — |

_Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +99.94 pp, ΔEQS +58.0._

### tinier_leakyrelu

_Victim `tinier_makeblobs_leakyrelu.pth` · oracle acc (X_test3) = 100.00% · recovered 34/56 neurons · LeakyReLU α=0.01._

| Stage | Agreement | Ext acc | Sign acc | EQS | ΔAgreement | ΔEQS |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 — RAW (Phase 1+2 load) | 25.00% | 25.00% | 48.43% | 35.4 | n/a | n/a |
| Stage 1 — + BIAS recovery | 6.58% | 6.58% | 48.43% | 25.9 | -18.42 | -9.5 |
| Stage 2 — + LR FIT (fc5) | 99.92% | 99.92% | 48.43% | 74.7 | +93.34 | +48.8 |
| Stage 3 — + SIGN SEARCH (SA+margin) | 100.00% | 100.00% | 50.51% | 77.9 | +0.08 | +3.2 |
| Stage 4 — + FROZEN REFINE (full pipeline) | 100.00% | 100.00% | 50.51% | 78.8 | +0.00 | +1.0 |
| Distillation baseline (non-staged) | 100.00% | 100.00% | n/a | 60.2 | — | — |

_Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +93.34 pp, ΔEQS +48.8._

### tiny_relu

_Victim `makeblobs_relu.pth` · oracle acc (X_test3) = 100.00% · recovered 139/256 neurons · LeakyReLU α=0.0._

| Stage | Agreement | Ext acc | Sign acc | EQS | ΔAgreement | ΔEQS |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 — RAW (Phase 1+2 load) | 3.00% | 3.00% | 53.77% | 19.2 | n/a | n/a |
| Stage 1 — + BIAS recovery | 14.34% | 14.34% | 53.77% | 24.8 | +11.34 | +5.6 |
| Stage 2 — + LR FIT (fc5) | 100.00% | 100.00% | 53.77% | 71.0 | +85.66 | +46.2 |
| Stage 3 — + SIGN SEARCH (SA+margin) | 100.00% | 100.00% | 56.15% | 74.1 | +0.00 | +3.1 |
| Stage 4 — + FROZEN REFINE (full pipeline) | 100.00% | 100.00% | 56.15% | 75.3 | +0.00 | +1.2 |
| Distillation baseline (non-staged) | 100.00% | 100.00% | n/a | 55.2 | — | — |

_Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +85.66 pp, ΔEQS +46.2._

### tiny_leakyrelu

_Victim `makeblobs_leakyrelu.pth` · oracle acc (X_test3) = 100.00% · recovered 228/256 neurons · LeakyReLU α=0.01._

| Stage | Agreement | Ext acc | Sign acc | EQS | ΔAgreement | ΔEQS |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 — RAW (Phase 1+2 load) | 10.68% | 10.68% | 53.45% | 26.7 | n/a | n/a |
| Stage 1 — + BIAS recovery | 12.60% | 12.60% | 53.45% | 27.7 | +1.92 | +1.0 |
| Stage 2 — + LR FIT (fc5) | 100.00% | 100.00% | 53.45% | 72.5 | +87.40 | +44.8 |
| Stage 3 — + SIGN SEARCH (SA+margin) | 100.00% | 100.00% | 53.45% | 75.7 | +0.00 | +3.2 |
| Stage 4 — + FROZEN REFINE (full pipeline) | 100.00% | 100.00% | 53.45% | 77.1 | +0.00 | +1.4 |
| Distillation baseline (non-staged) | 100.00% | 100.00% | n/a | 55.3 | — | — |

_Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +87.40 pp, ΔEQS +44.8._

## 2. Per-victim largest-lift summary

- **tiniest_relu** — Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +69.60 pp, ΔEQS +37.9.
- **tiniest_leakyrelu** — Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +95.20 pp, ΔEQS +46.5.
- **tinier_relu** — Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +99.94 pp, ΔEQS +58.0.
- **tinier_leakyrelu** — Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +93.34 pp, ΔEQS +48.8.
- **tiny_relu** — Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +85.66 pp, ΔEQS +46.2.
- **tiny_leakyrelu** — Largest lift carried by **fc5 LR fit** (Stage 2): ΔAgreement +87.40 pp, ΔEQS +44.8.

## 3. Stage-4 sanity check (vs headline)

Stage 4 is the full pipeline and should reproduce the headline numbers produced by the canonical driver on the same Phase-1/2 artifacts. Phase-1 dual search is stochastic, so small deviations are expected; gaps > 2.0 pp are flagged.

**tiniest_relu**

| Headline reference | Headline | Harness Stage 4 | Δ |
|---|---:|---:|---:|
| driver run_extraction.py prediction_agreement (X_test3) | 99.70% | 99.70% | +0.00 pp |
| driver step-9 scorecard fidelity (X_test3) | 99.70% | 99.70% | +0.00 pp |
| driver step-9 scorecard EQS (structural) | 71.2 | 71.2 | +0.0 |

**tiniest_leakyrelu**

| Headline reference | Headline | Harness Stage 4 | Δ |
|---|---:|---:|---:|
| driver run_extraction.py prediction_agreement (X_test3) | 98.90% | 98.90% | -0.00 pp |
| driver step-9 scorecard fidelity (X_test3) | 98.90% | 98.90% | +0.00 pp |
| driver step-9 scorecard EQS (structural) | 71.5 | 71.5 | +0.0 |

**tinier_relu**

| Headline reference | Headline | Harness Stage 4 | Δ |
|---|---:|---:|---:|
| driver run_extraction.py prediction_agreement (X_test3) | 100.00% | 100.00% | +0.00 pp |
| driver step-9 scorecard fidelity (X_test3) | 100.00% | 100.00% | +0.00 pp |
| driver step-9 scorecard EQS (structural) | 81.3 | 81.3 | +0.0 |

**tinier_leakyrelu**

| Headline reference | Headline | Harness Stage 4 | Δ |
|---|---:|---:|---:|
| driver run_extraction.py prediction_agreement (X_test3) | 100.00% | 100.00% | +0.00 pp |
| driver step-9 scorecard fidelity (X_test3) | 100.00% | 100.00% | +0.00 pp |
| driver step-9 scorecard EQS (structural) | 78.8 | 78.8 | +0.0 |

**tiny_relu**

| Headline reference | Headline | Harness Stage 4 | Δ |
|---|---:|---:|---:|
| driver run_extraction.py prediction_agreement (X_test3) | 100.00% | 100.00% | +0.00 pp |
| driver step-9 scorecard fidelity (X_test3) | 100.00% | 100.00% | +0.00 pp |
| driver step-9 scorecard EQS (structural) | 75.3 | 75.3 | +0.0 |

**tiny_leakyrelu**

| Headline reference | Headline | Harness Stage 4 | Δ |
|---|---:|---:|---:|
| driver run_extraction.py prediction_agreement (X_test3) | 100.00% | 100.00% | +0.00 pp |
| driver step-9 scorecard fidelity (X_test3) | 100.00% | 100.00% | +0.00 pp |
| driver step-9 scorecard EQS (structural) | 77.1 | 77.1 | +0.0 |

_No stage-4 discrepancies beyond tolerance / rounding._

## 4. Config (reproducibility)

Identical canonical SA+margin configuration for every victim (per-arch only `refine_epochs` differs):

| Knob | Value |
|---|---|
| sign-search method / objective | sa / margin |
| sign pair-lookahead K | 8 |
| sign refine cycles / mini-epochs | 3 / 20 |
| refine optimiser | AdamW, weight_decay=0.0001, cosine_lr=True |
| refine lr | 0.005 |
| refine epochs (tiniest/tinier/tiny) | 300 / 500 / 500 |
| early-stop patience / eval-every | 5 / 10 |
| train pool / eval set | X_test ∪ X_test2 / X_test3 |
| query budget | 20000 |
| EQS variant | structural (C1=22,C2=26,C3=17,S=20) |
| reconstruct seed / scorecard seed | 42 / 0 |
| margin-proxy subsample (n_boundary) | 1000 |

**Victim checkpoints** (`tiny_stuff/`):

- `tiniest_relu` → `tiny_stuff/tiniest_makeblobs_relu.pth`
- `tiniest_leakyrelu` → `tiny_stuff/tiniest_makeblobs_leakyrelu.pth`
- `tinier_relu` → `tiny_stuff/tinier_makeblobs_relu.pth`
- `tinier_leakyrelu` → `tiny_stuff/tinier_makeblobs_leakyrelu.pth`
- `tiny_relu` → `tiny_stuff/makeblobs_relu.pth`
- `tiny_leakyrelu` → `tiny_stuff/makeblobs_leakyrelu.pth`

_Distillation baseline: all hidden rows Kaiming-initialised and trainable (`--refine-unfreeze`), same query budget + refinement settings; sign acc n/a (no recovered rows), EQS structural S-block = 0._
