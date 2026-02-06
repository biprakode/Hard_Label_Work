# Complete DNN Extraction Workflow

This guide walks through the complete process of extracting a neural network's weights using only hard-label (black-box) access.

---

## Supported Models

| Model | Architecture | Input | Hidden Dims | Output | Dataset |
|-------|-------------|-------|-------------|--------|---------|
| **Tinier** | 32->16->16->16->8->4 | 32 | [16,16,16,8] | 4 | make_blobs |
| **Tiny** | 64->64->64->64->64->10 | 64 | [64,64,64,64] | 10 | make_blobs/CIFAR-10 |
| **Full** | 3072->256->256->256->64->10 | 3072 | [256,256,256,64] | 10 | CIFAR-10 |

The **Tinier** model has non-uniform hidden layer widths and is designed for fast iteration.

```
Input(32) -> fc1 -> [16] -> fc2 -> [16] -> fc3 -> [16] -> fc4 -> [8] -> fc5 -> Output(4)
              ReLU          ReLU          ReLU          ReLU
Layer boundaries: [0, 16, 32, 48, 56]  (56 total hidden neurons)
```

---

## Phase 0: Model Creation

### Create Tinier Model
```bash
python create_tinier_makeblobs_model.py
```

**Outputs**:
- `tiny_shit/tinier_makeblobs_relu.pth` - PyTorch model
- `tiny_shit/tinier_makeblobs_relu.keras` - Keras model
- `data/x_test_tinier_makeblobs.npy` - Test features (10000, 32)
- `data/y_test_tinier_makeblobs.npy` - Test labels (10000,)

**Expected**: Test accuracy > 80%

### Configuration

Set flags in `signature_recovery/utils.py`:
```python
TINIER = True       # Non-uniform hidden widths
MAKEBLOBS = True    # Use make_blobs dataset
# This sets LAYER_SIZES = [32, 16, 16, 16, 8, 4]
```

All dimension-dependent code reads from `LAYER_SIZES`:
- `IDIM = LAYER_SIZES[0]` (input dim)
- `DIM = max(LAYER_SIZES[1:-1])` (max hidden width)
- `LAYER_BOUNDARIES = [0, 16, 32, 48, 56]` (cumulative hidden dims)

---

## Phase 1: Signature Recovery

### Step 1.1: Generate Dual Points

```bash
cd signature_recovery

# Single run
python find_duals.py

# Batch run (1000 iterations)
./run_duals.sh
```

**Output**: `exp/1/duals_XXXXXXXX.p`

### Step 1.2: Cluster Dual Points by Neuron

```bash
# Cluster for each layer (0-4)
for layer in 0 1 2 3 4; do
    python cluster_dual_points.py $layer
done
```

Uses `LAYER_BOUNDARIES` to map flat neuron indices to layers (no more `neuron_idx // DIM`).

**Output**: `exp/1-cluster-{0,1,2,3,4}.p`

### Step 1.3: Generate Per-Neuron Dual Files

```bash
python generate_dual_neuron.py
```

Computes local neuron index via `neuron_idx - LAYER_BOUNDARIES[layer]` instead of `neuron_idx % 64`.

**Output**: `../sign_recovery/layer_neuron_npys/layer{X}_neuron{Y}.npy`

Per-layer neuron counts for tinier model:
- Layer 1: neurons 0-15 (16 neurons)
- Layer 2: neurons 0-15 (16 neurons)
- Layer 3: neurons 0-15 (16 neurons)
- Layer 4: neurons 0-7 (8 neurons)

### Step 1.4: Recover Unsigned Weight Vectors

```bash
for layer in 0 1 2 3; do
    python recover_weights.py $layer
done
```

`CIFAR10NetPrefix` builds layers from `LAYER_SIZES` (handles non-uniform widths).

**Output**: `outputs/model_weights/Vrelu/layer_{X}/neuron_{Y}/weights.{npz,txt}`

---

## Phase 2: Sign Recovery

### Step 2.1: Configure and Run

Edit `sign_recovery/batched_sign_recovery.py`:
```python
TINIER = True
MAKEBLOBS = True
# Automatically sets:
#   LAYER_NEURON_COUNTS = {1: 16, 2: 16, 3: 16, 4: 8}
#   model_path = "tinier_makeblobs_relu.keras"
```

```bash
cd sign_recovery
python batched_sign_recovery.py
```

Per-layer parameters:
| Layer | nExp | choose_dx | Notes |
|-------|------|-----------|-------|
| 1 | 10,000 | perfect_control | No past-layer toggles |
| 2-3 | 10,000 | along_decision_boundary | Standard |
| 4 | 100 | perfect_control | No future toggles |

**Output**: `results/sign_recovery/layer{X}_{signs,confidences,votes}.npy`

### Step 2.2: Aggregate Results

```bash
python custom_tables.py
```

**Output**: `results/tables/`

---

## Phase 3: Model Reconstruction & Verification

### Step 3.1: Reconstruct and Verify

```bash
cd analysis

# For tinier model (three-tier metrics)
python test_extraction4.py --tinier

# For tiny/makeblobs model
python test_extraction4.py --makeblobs

# For full model
python test_extraction4.py --full
```

### Three-Tier Metrics

The verification script reports three separate metrics:

1. **SIGN accuracy**: `sign(cosine_similarity)` per neuron - did we get the +/- right?
2. **MAGNITUDE relative error**: after sign-aligning each neuron - is the magnitude right?
3. **COMBINED relative error**: without alignment - overall quality

This separates "is the direction right?" from "is the sign right?" to identify root causes.

### Step 3.2: Investigate Sign Issues

```bash
python investigate_sign_recovery.py --tinier
```

Reports:
- Ground truth sign distribution per layer
- Per-neuron recovered vs true sign comparison
- Dual point availability
- Structural analysis of layer 1 and layer 4 failures

---

## Quick Reference: Full Pipeline

```bash
# === Phase 0: Create Model ===
python create_tinier_makeblobs_model.py

# === Phase 1: Signature Recovery ===
cd signature_recovery
./run_duals.sh
for layer in 0 1 2 3 4; do python cluster_dual_points.py $layer; done
python generate_dual_neuron.py
for layer in 0 1 2 3; do python recover_weights.py $layer; done

# === Phase 2: Sign Recovery ===
cd ../sign_recovery
python batched_sign_recovery.py
python custom_tables.py

# === Phase 3: Reconstruction ===
cd ../analysis
python test_extraction4.py --tinier
python investigate_sign_recovery.py --tinier
```

---

## Known Issues

### Layer 1: Signs never flip
The sign recovery algorithm measures asymmetry via previous-layer toggles. Layer 1 has no previous hidden layers, so there's no toggle-based signal. All votes tend the same direction, producing biased results.

**Mitigation**: Use `perfect_control_along_decision_boundary` mode with high nExp.

### Last hidden layer: Signs show '?'
Sign recovery measures distance to future-layer toggles. The last hidden layer feeds directly into the output (no more ReLUs), so there are no future toggles.

**Mitigation**: Solve signs algebraically using known output layer weights.

---

## Directory Structure

```
hard-label-dnn-extraction/
├── create_tinier_makeblobs_model.py   # Phase 0
├── tiny_shit/
│   ├── tinier_makeblobs_relu.pth      # Tinier model (PyTorch)
│   └── tinier_makeblobs_relu.keras    # Tinier model (Keras)
├── data/
│   ├── x_test_tinier_makeblobs.npy    # Test data
│   └── y_test_tinier_makeblobs.npy    # Test labels
├── signature_recovery/
│   ├── utils.py                        # LAYER_SIZES, LAYER_BOUNDARIES
│   ├── find_duals.py
│   ├── cluster_dual_points.py
│   ├── generate_dual_neuron.py
│   ├── recover_weights.py
│   ├── exp/                            # Dual points + clusters
│   └── outputs/model_weights/Vrelu/    # Recovered weights
├── sign_recovery/
│   ├── batched_sign_recovery.py        # Per-layer neuron counts
│   ├── custom_tables.py
│   └── layer_neuron_npys/              # Per-neuron dual points
├── analysis/
│   ├── test_extraction4.py             # Three-tier metrics
│   └── investigate_sign_recovery.py    # Diagnostic tool
└── results/
    ├── sign_recovery/                  # Aggregated signs
    ├── tables/                         # Summary tables
    └── reconstructed_models/           # Final output
```
