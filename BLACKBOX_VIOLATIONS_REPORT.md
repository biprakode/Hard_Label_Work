
# Comprehensive Blackbox Violation Report

## Executive Summary

This report documents all instances where the DNN extraction attack implementation violates the **blackbox oracle assumption**. In a true blackbox setting, an attacker should only have access to:
- **Input**: Arbitrary vectors to query the model
- **Output**: Hard-label predictions (classification labels only, no confidence scores or probabilities)

However, for computational efficiency during research/development, this codebase "cheats" by directly accessing model internals (weights, gradients, activations) that would be unavailable in a realistic attack scenario.

---

## Critical Violations by Phase

### Phase 1: Signature Recovery (`signature_recovery/`)

#### 1. Ground Truth Weight Access for Validation
**Location**: `signature_recovery/recover_weights.py:246-256`

```python
# VIOLATION: Direct access to ground truth weights
errs = []
for maybe_neuron in range(DIM):
    factor = np.median(soln/cheat_solution[LAYER][maybe_neuron, :])
    errs.append(np.sum(np.abs(soln / factor - cheat_solution[LAYER][maybe_neuron, :])))
if min(errs) < 1e-3:
    print("Successfully extracted neuron", np.argmin(errs), 'with abs err', np.min(errs))
```

**What it does**: Compares recovered weights against ground truth model weights to compute accuracy metrics.

**Why it's cheating**: In a blackbox attack, you don't have access to the victim model's weights, so you cannot compute this error metric. The attacker wouldn't even know which neuron was successfully extracted.

**Legitimate alternative**: In a true blackbox setting, validation would require testing the reconstructed model's predictions against the oracle on a test set.

---

#### 2. Gradient Computation via Backpropagation
**Location**: `signature_recovery/utils.py:425-432`

```python
def get_normal(x, step_size=1e-6):
    if USE_GRADIENT:
        x = torch.tensor(x, requires_grad=True)
        out = gapt(x.cpu(), grad=True)
        out[0].backward()  # VIOLATION: Backpropagation requires model internals
        real = np.random.normal(0, 1) * x.grad.cpu().numpy()
        real = norm(real)
        return real
```

**What it does**: Computes exact gradients of the decision boundary using PyTorch's autograd (backpropagation).

**Why it's cheating**: Computing gradients requires:
- Access to model architecture and weights
- Ability to compute partial derivatives through all layers
- This is fundamentally a **whitebox** operation

**Legitimate alternative**: Use finite difference approximations based purely on querying predictions at nearby points (the `else` branch does this, but it's not used when `USE_GRADIENT=True`).

---

#### 3. Direct Model Forward Pass Access
**Location**: `signature_recovery/utils.py:128-136`

```python
# VIOLATION: Loading and storing the complete victim model
cheat_net_cuda = CIFAR10Net().to(device).double()
cheat_net_cuda = load_converted_model(MODEL_PATH, cheat_net_cuda, device)

cheat_net_cpu = CIFAR10Net()
cheat_net_cpu = load_converted_model(MODEL_PATH, cheat_net_cpu, device)

cheat_solution = [x.cpu().detach().numpy() for x in cheat_net_cpu.parameters()][::2]
```

**What it does**: Loads the complete victim model into memory and extracts all weight matrices.

**Why it's cheating**: In a blackbox scenario, you only have query access to the model via an API, not direct access to the model file or parameters.

---

#### 4. Layer-wise Activation Inspection
**Location**: `signature_recovery/utils.py:84-101` (the `cheat()` method)

```python
@torch.no_grad
def cheat(self, x):
    o = []
    def relu(x):
        if x.size(-1) < DIM:
            padding = DIM - x.size(-1)
            xx = torch.nn.functional.pad(x, (0, padding))
            xx[:, -padding:] = 1
            o.append(xx)  # VIOLATION: Storing intermediate layer activations
        else:
            o.append(x)
        return self.relu(x)

    x = x.view(-1, IDIM)
    x = relu(self.fc1(x))
    x = relu(self.fc2(x))
    x = relu(self.fc3(x))
    x = relu(self.fc4(x))
    return torch.stack(o)
```

**What it does**: Returns activations from all intermediate layers after each ReLU.

**Why it's cheating**: Layer activations are internal to the model. A blackbox API only returns final predictions, not intermediate values.

**Used in**: `signature_recovery/utils.py:145-151` via `cheat()` and `cheat_cuda()` functions.

---

#### 5. Decision Boundary Gradient Cheating
**Location**: `signature_recovery/find_duals.py:14-17`

```python
def is_on_decision_boundary_cheat(point, delta):
    # VIOLATION: Using internal gap function requiring model access
    return torch.abs(gapt(torch.tensor(point))) < 1e-10
```

**What it does**: Checks if a point is on the decision boundary by computing the gap between top-2 logits, which requires full model forward pass access.

**Why it's cheating**: The `gapt()` function requires accessing raw model outputs (logits) before the argmax, not just the final label.

---

#### 6. Neuron Toggle Detection via Activation Inspection
**Location**: `signature_recovery/utils.py:174-196`

```python
def cheat_num_flips(a,b):
    if not DEBUG: raise
    return np.sum((cheat(a)>0) != (cheat(b)>0))  # VIOLATION: Counting neuron toggles

def cheat_neuron_diff(a,b):
    if not DEBUG: raise
    return np.where((cheat(a)>0).flatten() != (cheat(b)>0).flatten())[0]

def cheat_neuron_diff_cuda(a,b):
    if not DEBUG: raise
    ab = torch.tensor(np.stack([a,b])).double().cpu()
    out = cheat_cuda(ab)>0
    a = out[:,0,:]
    b = out[:,1,:]
    return torch.where(a.flatten() != b.flatten())[0].cpu().numpy()
```

**What it does**: Directly inspects which neurons toggled between two inputs by comparing layer activations.

**Why it's cheating**: Requires access to intermediate layer outputs to determine which specific neurons flipped from OFF to ON or vice versa.

---

### Phase 2: Sign Recovery (`sign_recovery/`)

#### 7. Direct Weight and Bias Extraction
**Location**: `sign_recovery/whitebox.py:10-25`

```python
def getWeightsAndBiases(model, layer_indices=None):
    weights = []
    biases = []

    if layer_indices is None:
        layer_indices = range(len(model.layers))

    for l in layer_indices:
        layer = model.layers[l]
        params = layer.get_weights()  # VIOLATION: Direct weight access
        if len(params) == 2:
            w, b = params
            weights.append(np.copy(w))
            biases.append(np.copy(b))

    return weights, biases
```

**What it does**: Extracts all weights and biases directly from the Keras model via `layer.get_weights()`.

**Why it's cheating**: This is literally the definition of whitebox access - directly reading model parameters.

**Used in**: Multiple functions throughout `whitebox.py` including:
- `getRealSigns()` - line 27-30
- `getSignatures()` - line 32-38
- `getNeuronWeightBias()` - line 92-104
- `getOutputMatrixWhitebox()` - line 232-242

---

#### 8. Layer Output Inspection
**Location**: `sign_recovery/whitebox.py:122-142`

```python
def getLayerOutputs(model, testInput, onlyLayerID=None):
    outputOfAllLayers = []

    for layerID, layer in enumerate(model.layers):
        if onlyLayerID is not None and layerID != onlyLayerID:
            continue

        # VIOLATION: Creating intermediate models to extract layer outputs
        intermediateLayerModel = Model(inputs=model.input,
                                      outputs=model.get_layer(layer.name).output)
        intermediateOutput = intermediateLayerModel(testInput).numpy()
        outputOfAllLayers.append(intermediateOutput)
```

**What it does**: Creates intermediate Keras models to extract outputs from specific layers.

**Why it's cheating**: Extracting layer-specific outputs requires model architecture access and the ability to create custom computation graphs.

---

#### 9. Ground Truth Sign Validation
**Location**: `sign_recovery/whitebox.py:27-30`

```python
def getRealSigns(model, layerID):
    weights, biases = getWeightsAndBiases(model, range(1, layerID + 1))
    signsLayer = np.sign(weights[-1][0])  # VIOLATION: Computing true signs
    return signsLayer
```

**What it does**: Computes the ground truth signs of all neurons for validation.

**Why it's cheating**: Requires direct weight access. In a blackbox attack, you don't know the true signs to compare against.

---

#### 10. Sign Correctness Testing
**Location**: `sign_recovery/whitebox.py:40-41`

```python
def signIsCorrect(neuronID, w, w0):
    return (w[:,neuronID]==w0[:,neuronID]).all()  # VIOLATION: Comparing against ground truth
```

**What it does**: Checks if the recovered weight vector's sign is correct by comparing with ground truth.

**Why it's cheating**: Requires access to the true weight matrix for comparison.

---

## Configuration Flags for Cheating

### USE_GRADIENT Flag
**Location**: `signature_recovery/utils.py` (used throughout)

When `USE_GRADIENT=True`:
- Enables gradient computation via backpropagation (violation #2)
- Uses `get_normal()` with PyTorch autograd
- Significantly speeds up dual point discovery

When `USE_GRADIENT=False`:
- Uses finite difference approximations (legitimate blackbox approach)
- Much slower but doesn't require model internals

### DEBUG Flag
**Location**: `signature_recovery/utils.py` (lines 146, 154, 163, 175, etc.)

```python
def cheat(x):
    if not DEBUG: raise  # Guard to prevent accidental use in production
    return cheat_net_cpu.cheat(torch.tensor(x).double()).numpy()
```

The `DEBUG` flag gates most cheating functions. When `DEBUG=False`, these functions raise exceptions. This suggests the authors were aware of the violations and intended to replace them with blackbox alternatives.

---

## Impact on Attack Complexity

### Computational Speedup from Cheating

| Operation | Blackbox Method | Whitebox Cheat | Speedup Factor |
|-----------|----------------|----------------|----------------|
| Gradient computation | Finite differences (~2N queries for N-dim gradient) | Backprop (1 forward + 1 backward pass) | ~100-1000x |
| Decision boundary detection | Random sampling + binary search | Direct gap computation | ~50-100x |
| Neuron toggle detection | Exhaustive testing + prediction comparison | Direct activation inspection | ~1000x+ |
| Validation | Model comparison on test set | Direct weight comparison | N/A (impossible in blackbox) |

### What Would Change in True Blackbox

1. **No validation metrics**: Couldn't compute "abs err" or know which neuron was recovered
2. **Slower dual point discovery**: Would need finite differences instead of gradients
3. **More queries**: Each gradient approximation needs O(n) queries instead of 1 forward pass
4. **Less precision**: Finite differences are numerically less stable than exact gradients
5. **No ground truth comparison**: Success would be measured by reconstructed model's prediction accuracy only

---

## Legitimate Blackbox Alternatives

### For Gradient Computation
```python
# Instead of: x.backward() to get exact gradients
# Use: Finite difference approximation
def finite_difference_gradient(x, f, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad
```

### For Decision Boundary Detection
```python
# Instead of: checking if |gap(x)| < threshold using logits
# Use: Random perturbations and label comparison
def is_on_boundary_blackbox(x, model_query, delta=1e-4, n_samples=100):
    base_label = model_query(x)
    for _ in range(n_samples):
        perturbation = np.random.randn(*x.shape) * delta
        if model_query(x + perturbation) != base_label:
            return True
    return False
```

### For Validation
```python
# Instead of: comparing recovered weights to ground truth
# Use: Test set prediction comparison
def validate_extraction_blackbox(reconstructed_model, oracle_query, test_set):
    agreement = 0
    for x in test_set:
        if reconstructed_model(x) == oracle_query(x):
            agreement += 1
    return agreement / len(test_set)
```

---

## Summary Table of Violations

| # | Violation Type | Location | Blackbox? | Used For |
|---|---------------|----------|-----------|----------|
| 1 | Ground truth weight access | `recover_weights.py:246-256` | ❌ | Validation/metrics |
| 2 | Gradient computation (backprop) | `utils.py:425-432` | ❌ | Finding dual points |
| 3 | Model file loading | `utils.py:128-136` | ❌ | Everything |
| 4 | Layer activation inspection | `utils.py:84-101` | ❌ | Neuron toggle detection |
| 5 | Decision boundary via gap | `find_duals.py:14-17` | ❌ | Boundary refinement |
| 6 | Neuron toggle detection | `utils.py:174-196` | ❌ | Clustering dual points |
| 7 | Direct weight extraction | `whitebox.py:10-25` | ❌ | Sign recovery validation |
| 8 | Layer output extraction | `whitebox.py:122-142` | ❌ | Activation analysis |
| 9 | Ground truth sign access | `whitebox.py:27-30` | ❌ | Validation |
| 10 | Sign correctness testing | `whitebox.py:40-41` | ❌ | Accuracy metrics |

---

## Recommendations for True Blackbox Implementation

1. **Set `USE_GRADIENT=False`** throughout the codebase
2. **Remove all `cheat_*` functions** or guard them more strictly
3. **Replace gradient computations** with finite difference methods
4. **Remove weight comparisons** and use prediction agreement for validation
5. **Track query budget** - count every call to `model()` or `bmodel()`
6. **Accept longer runtime** - blackbox attacks are inherently slower
7. **Accept lower precision** - finite differences are less accurate than exact gradients

The paper's theoretical contributions remain valid, but this implementation trades correctness for speed by using whitebox shortcuts.
