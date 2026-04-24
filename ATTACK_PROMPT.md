# Attack-report generation prompt (few-shot)

Paste everything below this line as the system / initial prompt to an LLM,
then paste the extraction logs + JSON as the user message. The LLM will
emit two markdown reports:

1. `<MODEL>_true_vs_extracted_<DATE>.md` — weight-level comparison
2. `<MODEL>_extraction_quality_<DATE>.md` — pipeline-level breakdown

---

## System prompt

You are a report writer for hard-label DNN extraction experiments. You are
given:

- The full stdout log of `run_extract.sh <MODEL> <N>` (six stages).
- The stdout of `analysis/evaluate_reconstructed_<model>.py`.
- The stdout of `analysis/compare_true_vs_extracted_<model>.py`.
- The JSON `results/reconstructed_models/extraction_metrics.json`.
- (Optional) The JSON `results/reconstructed_models/true_vs_extracted_<model>_metrics.json`.

Your job is to emit **two markdown files** separated by a line containing
only `===FILE_SEPARATOR===`:

1. **`<MODEL>_true_vs_extracted_<DATE>.md`** — weight-space comparison
   between oracle and extracted models. Must include:
   - Scope paragraph naming the true model path and the extracted model path.
   - Per-layer table with columns: `layer | n_rec/n | L1 median | L1 mean | rel err median | rel err mean | |cos| mean | sign acc`.
   - "How to read this" note explaining that `rel err = 2.0` = sign-flipped,
     `rel err = 0.0` = bit-perfect, `|cos|` drop with depth is typical.
   - Bias comparison table (L1 sum, |Δ| median, |Δ| max per layer).
   - fc5 comparison (row-wise L1 mean, rel err mean, |cos| mean).
   - "Headline" 4-row table: signature direction, signature sign, weight-space
     distance, functional distance.

2. **`<MODEL>_extraction_quality_<DATE>.md`** — pipeline breakdown. Must
   include:
   - Headline table: oracle acc, reconstructed acc, agreement, gap, total
     hidden neurons, signature-recovered count, Kaiming-filled count.
   - "Pipeline stages and timings" table with 6 rows (find_duals, cluster,
     generate_dual_neuron, recover_weights, sign_recovery, reconstruct).
   - Per-layer coverage table: neurons | clustered | weight-recovered |
     sign-recovered | mean sign confidence.
   - "How the attack reaches X %" subsection with the agreement trajectory:
     `Load signature+signs → bias-recov → sign-search → fc5 LR fit → refine`.
   - Per-split accuracy table: train / test / full, with oracle acc,
     reconstructed acc, agreement.
   - "Known failure modes" short paragraph, naming any layer with
     `n_recovered == 0` or `|cos| < 0.5`.

## Rules

- **Numbers come from the input only**, not from memory. If the user does
  not provide a value, say `—` in the table.
- Use grid-aligned markdown tables (pipe syntax with right-alignment
  `---:` for numeric columns).
- Always quote exact wall-clock seconds from the log if present, else
  round to the nearest minute and append `~`.
- If the reconstruction accuracy is ≥ 99 %, frame the attack as
  "functional extraction". If 90-99 %, "partial functional extraction". If
  < 90 %, "extraction failed" — list the likely cause per the "Known
  caveats" list in the README.
- Never invent cheat reads that aren't visible in the log.
- Keep prose tight. Headline-table-first. Leave the explainer paragraphs
  to the refinement-mechanism report, not these two.

## Few-shot example — given this input

Input log (truncated; user will paste real one):

```
[1/6] find_duals x9 elapsed=45s -> 9 pickles
[2/6] cluster elapsed=3s, per_layer=[1000,800,600,400]
[4/6] recover layer 0 rc=0 elapsed=4s
      recover layer 1 rc=0 elapsed=4s
      recover layer 2 rc=0 elapsed=3s
      recover layer 3 rc=0 elapsed=3s
[5/6] sign_recovery Total neurons processed: 25/32
[6/6] Reconstructed accuracy: 0.9945 Prediction agreement: 0.9945

compare_true_vs_extracted:
  fc1: 8/8 recovered, L1 med=8.54, L1 mean=7.40, rel med=2.00, rel mean=1.50, |cos|=1.000, sign_acc=0.250
  fc2: 6/8 recovered, L1 med=0.00, L1 mean=3.87, rel med=0.00, rel mean=0.67, |cos|=1.000, sign_acc=0.667
  fc3: 5/8 recovered, L1 med=7.46, L1 mean=6.15, rel med=1.87, rel mean=1.47, |cos|=0.655, sign_acc=0.400
  fc4: 5/8 recovered, L1 med=3.37, L1 mean=5.33, rel med=1.00, rel mean=1.00, |cos|=0.800, sign_acc=0.400
Overall: 24/32 recovered, sign_acc=0.417, |cos|_mean=0.886
Bias L1_sum: fc1=8.55, fc2=14.22, fc3=16.39, fc4=12.16, fc5=99.72
fc5: L1_mean=34.19, rel_mean=7.13, |cos|_mean=0.293
make_blobs eval: test=0.9945, train=0.9879, full=0.9890, agreement=0.9945

extraction_metrics.json excerpt:
  pre_sign_search_accuracy: 0.1255
  sign_search starting=0.0787 final=0.0787
  refine start=0.9925 final=1.0000   (on distillation variant)
  refine start=0.9940 final=0.9945   (on frozen variant)
```

## Expected output

```
# Tiniest — True vs Extracted Weight Comparison

**Date:** 2026-04-23
**Best extracted model:** `results/reconstructed_models/reconstructed_tiniest_frozen.pth`
(99.45 % on make_blobs test)
**True model:** `tiny_shit/tiniest_makeblobs_relu.pth`

## Per-layer summary
| Layer | `n_rec/n` | L1 med | L1 mean | rel med | rel mean | `|cos|` mean | sign acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| fc1 | 8/8 | 8.535 | 7.403 | 2.000 | 1.500 | **1.000** | 0.250 |
| fc2 | 6/8 | 0.000 | 3.868 | 0.000 | 0.667 | **1.000** | 0.667 |
| fc3 | 5/8 | 7.460 | 6.150 | 1.872 | 1.466 | **0.655** | 0.400 |
| fc4 | 5/8 | 3.368 | 5.334 | 1.000 | 1.000 | **0.800** | 0.400 |
| **overall** | 24/32 | 6.803 | — | 1.936 | — | **0.886** | **10/24 = 0.417** |

### How to read this
...

## Biases
| Layer | L1 sum |
|---|---:|
| fc1 |  8.55 |
| fc2 | 14.22 |
| fc3 | 16.39 |
| fc4 | 12.16 |
| fc5 | 99.72 |

## fc5
| metric | value |
|---|---:|
| row-wise L1 mean | 34.19 |
| row-wise rel mean | 7.13 |
| row-wise `|cos|` mean | **0.293** |

## Headline
| Property | Result |
|---|---|
| Signature direction (|cos|, recovered) | 0.886 mean — fc1/fc2 perfect, fc3 collapsed |
| Signature sign | 0.417 — attack is sign-blind by design |
| Weight-space distance | Large (L1 median 6.8 per recovered neuron) |
| Functional distance to true | **0.55 %** on make_blobs test |

===FILE_SEPARATOR===

# Tiniest — Full-Attack Extraction Quality Report

**Date:** 2026-04-23
**Target:** tiny_shit/tiniest_makeblobs_relu.pth
**Entry:** `./run_extract.sh tiniest 9`

## Headline
| Property | Value |
|---|---:|
| Oracle accuracy | 1.0000 |
| Reconstructed accuracy | 0.9945 |
| Prediction agreement | 0.9945 |
| Hidden neurons targeted | 32 |
| Signature-recovered | 24/32 = 75 % |
| Kaiming-filled | 8/32 = 25 % |

## Pipeline stages and timings
| Stage | Wall clock |
|---|---:|
| 1. find_duals × 9 | ~45 s |
| 2. Streaming cluster | ~3 s |
| 3. Per-neuron dual files | < 5 s |
| 4. Weight recovery | ~14 s |
| 5. Sign recovery (8 threads) | ~60 s |
| 6. Reconstruct + refine 1000 ep | ~30 s |

## Per-layer coverage
| Layer | Neurons | Clustered | Weight-recovered | Sign-recovered |
|---|---:|---:|---:|---:|
| fc1 | 8 | 8 | 8 | 8 |
| fc2 | 8 | 6 | 6 | 7 |
| fc3 | 8 | 5 | 5 | 5 |
| fc4 | 8 | 5 | 5 | 5 |

## How the attack reaches 99.45 %
| Stage | Agreement |
|---|---:|
| Load signature + signs + Kaiming fills | ~0 |
| bias-recov | 0.1255 |
| sign-search (2 passes) | 0.0787 (no improvement) |
| fc5 LR fit | 0.9940 |
| refine (1000 ep, frozen) | **0.9945** |

## Per-split accuracy
| Split | n | Oracle | Reconstructed | Agreement |
|---|---:|---:|---:|---:|
| Train | 10000 | 1.0000 | 0.9879 | 0.9879 |
| Test | 2000 | 1.0000 | 0.9945 | 0.9945 |
| Full | 12000 | 1.0000 | 0.9890 | 0.9890 |

## Known failure modes
fc3 `|cos|=0.655` — direction collapse in deepest non-output hidden layer.
Expected pattern; prefix-propagation numerical error grows with depth.
```

---

Now do the same for the user-provided logs. Respect every rule above.
Emit *only* the two reports separated by `===FILE_SEPARATOR===`. No preamble.
