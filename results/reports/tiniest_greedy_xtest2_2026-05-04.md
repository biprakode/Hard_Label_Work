# Tiniest Model Extraction Report — Greedy Sign Search + X_test2 Eval
**Date**: 2026-05-04  
**Model**: Tiniest (8→8→8→8→8→8, 8 classes, make_blobs, float64)  
**Changes introduced**: (1) greedy oracle sign search for k>18, (2) X_test2 fresh eval set

---

## 1. What Changed

### 1.1 Greedy Oracle Sign Search
Previously, the oracle sign search (brute-force 2^k) skipped any layer with k>18 recovered neurons.  
For CIFAR-10's 256-wide layers this meant **all layers were skipped** and the sign flipping step was a no-op.

**New behaviour**: when k>18, fall through to greedy O(k)-per-pass flip search (`_greedy_sign_pass_layer`).  
- For each recovered neuron: try flipping its sign, keep if oracle agreement improves  
- Repeat for `n_passes` (default 3), alternating layer order  
- No 2^k restriction — works for any architecture  
- For k≤18 (like tiniest, k=5-8 per layer): brute-force still used (globally optimal per layer)

### 1.2 X_test2 — Fresh Eval Set (Train/Test Overlap Fix)
Previously, Phase-3 training (sign search, fc5 LR fit, oracle refinement) and final evaluation all used the **same** `X_test` (seed=42, n=2000).  
This means perfect memorization on X_test shows up as 100% "accuracy" — not genuine generalisation.

**New behaviour**:
- `X_test` (seed=42, n=2000): used only for Phase-3 training (oracle queries)
- `X_test2` (same cluster centers, seed=99, n=2000): used only for final evaluation

**X_test2 generation**: `make_blobs(return_centers=True)` with seed=42 extracts cluster centers.  
`X_test2 = make_blobs(centers=seed42_centers, random_state=99)` + same scaler fit on seed=42 training data.  
Oracle accuracy on X_test2: **99.95%** (confirms labels are consistent).

---

## 2. Extraction Results (Tiniest, `--from-scratch --refine`)

| Dataset         | Oracle acc | Reconstructed acc | Agreement vs oracle | Note |
|-----------------|-----------|-------------------|---------------------|------|
| X_test (seed=42) | 100.0%    | **100.0%**        | **100.0%**          | Phase-3 training set (overlap!) |
| X_test2 (seed=99) | 99.95%  | **99.50%**        | **99.50%**          | Clean eval (no overlap) |
| Train (seed=42) | 100.0%    | 99.57%            | 99.57%              | |
| Full dataset    | 100.0%    | 99.64%            | 99.64%              | |

**Key finding**: the 100% agreement on X_test is inflated by train/test overlap. The honest number is **99.50% on X_test2**.  
This is still very strong — the extraction generalises well to unseen data.

---

## 3. Per-Class Breakdown (X_test2, seed=99)

| Class | n  | Oracle | Reconstructed | Gap |
|-------|----|--------|---------------|-----|
| 0     | 250 | 100%  | 98.80%        | -1.20% |
| 1     | 250 | 100%  | 98.40%        | -1.60% |
| 2     | 250 | 99.6% | 99.60%        | 0.00% |
| 3     | 250 | 100%  | 99.20%        | -0.80% |
| 4–7   | 250 | 100%  | **100%**      | 0.00% |

Errors concentrated in classes 0-3 (10 disagreements total). Classes 4-7 perfect.

---

## 4. Weight-Space Recovery Metrics

### Phase-1 Signature Recovery
| Layer | Recovered | |cos| (recovered) | Sign acc (pre-search) |
|-------|-----------|-------------|----------------------|
| fc1   | 8/8 (100%)| 1.00       | 50%                  |
| fc2   | 6/8 (75%) | 1.00       | 33%                  |
| fc3   | 0/8 (0%)  | N/A (Kaiming) | N/A              |
| fc4   | 5/8 (62%) | 0.80       | 20%                  |
| **Total** | **19/32 (59%)** | — | — |

### Phase-2 Sign Recovery (from batched_sign_recovery.py)
Signs from sign recovery were all +1 (biased — layer 1 has no past-layer toggles, last layer has no future toggles). Sign accuracy **before** oracle sign search was ~50%.

### Phase-3 Oracle Sign Search (brute-force, k≤8)
Post-sign-search metrics (recovered neurons only):
| Layer | Sign acc | |cos| | Mag rel err |
|-------|----------|------|-------------|
| fc1   | 62.5%    | 1.00 | 0.00        |
| fc2   | 50.0%    | 1.00 | 0.00        |
| fc4   | 60.0%    | 0.80 | 0.20        |
| **Avg** | **57.5%** | **0.93** | **0.07** |

Note: fc3 is 0% recovered (all Kaiming init) — sign accuracy not applicable.  
Despite low sign accuracy, the model achieves **99.5% functional accuracy** — see §5 for why.

---

## 5. Why Low Sign Accuracy → High Functional Accuracy

The tiniest model has only 8 neurons per layer and the oracle refinement step (500 epochs, Adam, frozen recovered rows) is free to:
- Adjust biases of all neurons (recovered and random)
- Retrain fc5 from scratch (LR fit + fine-tuning)
- Retrain all random-init neurons (fc3 entirely, 2 neurons in fc2, 3 in fc4)

This is essentially **knowledge distillation** on top of partial extraction: the functional behaviour is copied from the oracle via hard labels, not purely from weight-space recovery.

---

## 6. Greedy Sign Search — Relevance

For the **tiniest model** (k=5-8 per layer), the brute-force search is still used (k≤18).  
The greedy path is exercised when k>18, which applies to:
- **CIFAR-10 full model**: layers with 64 or 256 neurons — previously all skipped
- Any model wider than 18 neurons per layer

To verify greedy is working, run:
```bash
python3 analysis/test_extraction4.py --full --from-scratch  # triggers greedy on all layers
```

---

## 7. Files Added / Modified

| File | Change |
|------|--------|
| `analysis/test_extraction4.py` | Added `_greedy_sign_pass_layer`, `greedy_oracle_sign_search`; `_run_one_pass` uses greedy for k>18; added `load_test2_data`; main uses X_test for training, X_test2 for eval |
| `analysis/evaluate_reconstructed_makeblobs.py` | Added `X_TEST2_PATH`; includes 'test2' split in all evaluation tables |
| `data/x_test2_tiniest_makeblobs.npy` | 2000 fresh eval samples (seed=99, same cluster centers) |
| `data/y_test2_tiniest_makeblobs.npy` | Corresponding labels |
| `data/x_test2_makeblobs.npy` | 5000 fresh eval samples for makeblobs 64-wide model |
| `data/y_test2_makeblobs.npy` | Corresponding labels |
| All files mirrored in `enhanced_codebase/` | Synced |
