"""
Complete Model Extraction Verification Script

This script:
1. Loads unsigned weight vectors from signature recovery
2. Loads recovered signs from sign recovery
3. Combines them: signed_weights = unsigned_weights * signs
4. Reconstructs the full model
5. Compares with ground truth model
6. Reports detailed accuracy metrics (three-tier: sign, magnitude, combined)

Three-tier metrics:
    - SIGN: sign(cosine_sim) per neuron -> sign accuracy
    - MAGNITUDE: relative error after sign-aligning -> magnitude accuracy
    - COMBINED: relative error without alignment -> overall accuracy

Usage:
    python test_extraction4.py [--tiny] [--full] [--makeblobs] [--tinier]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
from pathlib import Path
import argparse

# ========== Configuration ========== #
BASE_DIR = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase"

# Activation toggle. Must match signature_recovery/utils.py LEAKY_ALPHA.
#   LEAKY_ALPHA = 0.0  -> plain ReLU (DEFAULT, original pipeline preserved exactly)
#   LEAKY_ALPHA > 0    -> Leaky ReLU(alpha)
LEAKY_ALPHA = 0.01


def _act(x):
    """ReLU mode (LEAKY_ALPHA == 0): F.relu. Leaky mode: F.leaky_relu(x, alpha)."""
    if LEAKY_ALPHA > 0:
        return F.leaky_relu(x, negative_slope=LEAKY_ALPHA)
    return F.relu(x)


_act_suffix = "leakyrelu" if LEAKY_ALPHA > 0 else "relu"

# Signature recovery outputs (unsigned weights)
SIGNATURE_WEIGHTS_PATH = os.path.join(BASE_DIR, "signature_recovery/outputs/model_weights/Vrelu")

# Sign recovery outputs
SIGN_RECOVERY_PATH = os.path.join(BASE_DIR, "results/sign_recovery")

# Ground truth models — suffix swaps to "leakyrelu" when LEAKY_ALPHA > 0
TINY_MODEL_PTH = os.path.join(BASE_DIR, f"tiny_stuff/TinyModel_{_act_suffix}.pth")
TINY_MODEL_KERAS = os.path.join(BASE_DIR, f"tiny_stuff/TinyModel_{_act_suffix}.keras")
MAKEBLOBS_MODEL_PTH = os.path.join(BASE_DIR, f"tiny_stuff/makeblobs_{_act_suffix}.pth")
TINIER_MODEL_PTH = os.path.join(BASE_DIR, f"tiny_stuff/tinier_makeblobs_{_act_suffix}.pth")
TINIEST_MODEL_PTH = os.path.join(BASE_DIR, f"tiny_stuff/tiniest_makeblobs_{_act_suffix}.pth")
FULL_MODEL_PTH = os.path.join(BASE_DIR, "signature_recovery/models/converted_model.pth")

# Test data (used for Phase-3 oracle training: sign search, fc5 LR fit, refinement)
X_TEST_PATH = os.path.join(BASE_DIR, "data/x_test.npy")
X_TEST_MAKEBLOBS_PATH = os.path.join(BASE_DIR, "data/x_test_makeblobs.npy")
Y_TEST_MAKEBLOBS_PATH = os.path.join(BASE_DIR, "data/y_test_makeblobs.npy")
X_TEST_TINIER_PATH = os.path.join(BASE_DIR, "data/x_test_tinier_makeblobs.npy")
Y_TEST_TINIER_PATH = os.path.join(BASE_DIR, "data/y_test_tinier_makeblobs.npy")
X_TEST_TINIEST_PATH = os.path.join(BASE_DIR, "data/x_test_tiniest_makeblobs.npy")
Y_TEST_TINIEST_PATH = os.path.join(BASE_DIR, "data/y_test_tiniest_makeblobs.npy")

# X_test2: fresh eval-only set (seed=99, same scaler) — no Phase-3 training overlap
X_TEST2_TINIEST_PATH = os.path.join(BASE_DIR, "data/x_test2_tiniest_makeblobs.npy")
Y_TEST2_TINIEST_PATH = os.path.join(BASE_DIR, "data/y_test2_tiniest_makeblobs.npy")
X_TEST2_MAKEBLOBS_PATH = os.path.join(BASE_DIR, "data/x_test2_makeblobs.npy")
Y_TEST2_MAKEBLOBS_PATH = os.path.join(BASE_DIR, "data/y_test2_makeblobs.npy")

# Output path for reconstructed models
OUTPUT_PATH = os.path.join(BASE_DIR, "results/reconstructed_models")
# ================================== #


class TinyModel(nn.Module):
    """5-layer tiny model for testing (64x64 dimensions)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 10)  # Output layer
        self.double()

    def forward(self, x):
        x = x.view(-1, 64)
        x = _act(self.fc1(x))
        x = _act(self.fc2(x))
        x = _act(self.fc3(x))
        x = _act(self.fc4(x))
        x = self.fc5(x)
        return x


class TinierModel(nn.Module):
    """5-layer tinier model with non-uniform hidden widths (32->16->16->16->8->4)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.double()

    def forward(self, x):
        x = x.view(-1, 32)
        x = _act(self.fc1(x))
        x = _act(self.fc2(x))
        x = _act(self.fc3(x))
        x = _act(self.fc4(x))
        x = self.fc5(x)
        return x


class TiniestModel(nn.Module):
    """Tiniest 8-8-8-8-8-8 make_blobs model."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 8)
        self.fc5 = nn.Linear(8, 8)
        self.double()

    def forward(self, x):
        x = x.view(-1, 8)
        x = _act(self.fc1(x))
        x = _act(self.fc2(x))
        x = _act(self.fc3(x))
        x = _act(self.fc4(x))
        x = self.fc5(x)
        return x


class FullModel(nn.Module):
    """Full CIFAR-10 model (3072 -> 256 -> ... -> 10)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 10)
        self.double()

    def forward(self, x):
        x = x.view(-1, 3072)
        x = _act(self.fc1(x))
        x = _act(self.fc2(x))
        x = _act(self.fc3(x))
        x = _act(self.fc4(x))
        x = self.fc5(x)
        return x


def load_ground_truth_model(model_path, model_class, device='cpu'):
    """Load the ground truth model from a .pth file."""
    model = model_class().to(device)
    state_dict = torch.load(model_path, map_location=device)

    # Handle different naming conventions
    rename_maps = [
        # TinyModel naming
        {
            "fc1.weight": "fc1.weight", "fc1.bias": "fc1.bias",
            "fc2.weight": "fc2.weight", "fc2.bias": "fc2.bias",
            "fc3.weight": "fc3.weight", "fc3.bias": "fc3.bias",
            "fc4.weight": "fc4.weight", "fc4.bias": "fc4.bias",
            "output.weight": "fc5.weight", "output.bias": "fc5.bias",
        },
        # Alternative naming with hidden_layer
        {
            "hidden_layer1.weight": "fc1.weight", "hidden_layer1.bias": "fc1.bias",
            "hidden_layer2.weight": "fc2.weight", "hidden_layer2.bias": "fc2.bias",
            "hidden_layer3.weight": "fc3.weight", "hidden_layer3.bias": "fc3.bias",
            "hidden_layer4.weight": "fc4.weight", "hidden_layer4.bias": "fc4.bias",
            "output.weight": "fc5.weight", "output.bias": "fc5.bias",
        },
    ]

    # Try direct load first (works for models saved with matching key names)
    loaded = False
    try:
        model.load_state_dict(state_dict)
        loaded = True
    except RuntimeError:
        pass

    if not loaded:
        for rename_map in rename_maps:
            try:
                new_state_dict = {}
                for old_key, new_key in rename_map.items():
                    if old_key in state_dict:
                        new_state_dict[new_key] = state_dict[old_key]
                if new_state_dict:
                    model.load_state_dict(new_state_dict, strict=False)
                    loaded = True
                    break
            except Exception:
                continue

    if not loaded:
        print(f"Warning: Could not load model from {model_path}")

    return model


def load_unsigned_weights(signature_path, layer_id, num_neurons, input_dim, use_random_init=True, layer_offset=0):
    """
    Load unsigned weight vectors from signature recovery output.

    Uses weights_unscaled.npz + abs(scaling_factor) to ensure
    the scaling does NOT reveal sign information (only magnitude).

    `layer_offset`: subtracted from parsed neuron id when directory names use
    global (flat) neuron IDs rather than per-layer IDs.
    """
    weights = {}
    metadata_dict = {}
    recovered_mask = np.zeros(num_neurons, dtype=bool)

    layer_dir = os.path.join(signature_path, f"layer_{layer_id}")
    if not os.path.exists(layer_dir):
        print(f"  Warning: Layer directory not found: {layer_dir}")
        weight_matrix = _kaiming_init(num_neurons, input_dim)
        return weight_matrix, recovered_mask, metadata_dict

    for neuron_dir in Path(layer_dir).glob("neuron_*"):
        try:
            raw_id = int(neuron_dir.name.split("_")[1])
            # Handle global (flat) neuron IDs by shifting into [0, num_neurons).
            neuron_id = raw_id - layer_offset if raw_id >= num_neurons else raw_id

            if neuron_id < 0 or neuron_id >= num_neurons:
                continue

            # Load metadata to get scaling factor.
            # If metadata.json is absent the neuron came out of recover_weights as
            # "Failed to identify" — i.e. the SVD returned a vector that didn't match
            # any cheat solution, so its direction is unreliable. Skip it so the
            # loader treats it as unrecovered (Kaiming init); otherwise the bad
            # direction poisons the prefix and refinement can't recover.
            metadata_path = neuron_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                scaling_factor = meta.get('scaling_factor', 1.0)
                metadata_dict[neuron_id] = meta
            else:
                continue

            # Load UNSCALED weights and apply abs(scaling_factor)
            weight_files = [
                (neuron_dir / "weights_unscaled.npz", True),
                (neuron_dir / "weights_unscaled.npy", True),
                (neuron_dir / "weights.npz", False),
                (neuron_dir / "weights.npy", False),
                (neuron_dir / "weights.txt", False),
            ]

            for wf, is_unscaled in weight_files:
                if wf.exists():
                    if wf.suffix == '.npz':
                        data = np.load(wf)
                        w = data[list(data.keys())[0]]
                    elif wf.suffix == '.npy':
                        w = np.load(wf)
                    else:
                        w = np.loadtxt(wf)

                    w = w.flatten()

                    if is_unscaled:
                        w = w / np.abs(scaling_factor)
                    break

            weights[neuron_id] = w
            recovered_mask[neuron_id] = True
        except Exception as e:
            print(f"Warning: Error loading weights from {neuron_dir}: {e}")

    # Build weight matrix - always use input_dim for correct shape
    if use_random_init:
        weight_matrix = _kaiming_init(num_neurons, input_dim)
    else:
        weight_matrix = np.zeros((num_neurons, input_dim), dtype=np.float64)

    for nid, w in weights.items():
        if nid < num_neurons and len(w) == input_dim:
            weight_matrix[nid] = w
        elif nid < num_neurons:
            print(f"  Warning: Neuron {nid} weight dim {len(w)} != expected {input_dim}, skipping")
            recovered_mask[nid] = False

    recovered_count = np.sum(recovered_mask)
    total = num_neurons
    print(f"  Recovered {recovered_count}/{total} neurons ({100*recovered_count/total:.1f}%)")
    if recovered_count < total and use_random_init:
        print(f"  Using Kaiming/He initialization for {total - recovered_count} unrecovered neurons")

    return weight_matrix, recovered_mask, metadata_dict


def _kaiming_init(num_neurons, input_dim, nonlinearity='relu'):
    """Kaiming/He initialization for ReLU networks."""
    gain = np.sqrt(2.0) if nonlinearity == 'relu' else 1.0
    std = gain / np.sqrt(input_dim)
    return np.random.randn(num_neurons, input_dim).astype(np.float64) * std


def load_signs(sign_path, layer_id):
    """Load recovered signs from sign recovery output."""
    npy_path = os.path.join(sign_path, f"layer{layer_id}_signs.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)

    json_path = os.path.join(sign_path, f"layer{layer_id}_summary.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return np.array(data.get('signs', []), dtype=np.int8)

    model_summary_path = os.path.join(sign_path, "model_sign_recovery_summary.json")
    if os.path.exists(model_summary_path):
        with open(model_summary_path, 'r') as f:
            data = json.load(f)
        layer_data = data.get('layers', {}).get(str(layer_id), {})
        if 'signs' in layer_data:
            return np.array(layer_data['signs'], dtype=np.int8)

    return None


def combine_weights_and_signs(unsigned_weights, signs):
    """Combine unsigned weights with recovered signs.

    Sign values: +1 / -1 are known, 0 means unknown (sign recovery didn't process
    this neuron). Treat 0 as +1 so the recovered weight is preserved; oracle
    sign search will flip it later if -1 is correct. Multiplying by 0 would
    zero out an otherwise-valid recovered weight.
    """
    if unsigned_weights is None or signs is None:
        return None

    num_neurons = unsigned_weights.shape[0]
    if len(signs) < num_neurons:
        signs = np.concatenate([signs, np.ones(num_neurons - len(signs), dtype=np.int8)])

    # Replace 0 (unknown) with +1; oracle sign search will polish.
    sign_eff = np.where(signs[:num_neurons] == 0, np.int8(1), signs[:num_neurons])
    return unsigned_weights * sign_eff[:, np.newaxis]


def compute_weight_metrics_v2(extracted_weights, true_weights):
    """
    Three-tier metrics separating sign from magnitude analysis.

    Returns dict with:
        - sign_accuracy: fraction of neurons with correct sign (from cosine_sim sign)
        - magnitude_mean_rel_error: relative error after sign-aligning each neuron
        - combined_mean_rel_error: relative error without alignment (overall)
        - per_neuron: list of per-neuron dicts
    """
    if extracted_weights is None or true_weights is None:
        return None
    if extracted_weights.shape != true_weights.shape:
        print(f"Shape mismatch: extracted {extracted_weights.shape} vs true {true_weights.shape}")
        return None

    n_neurons = extracted_weights.shape[0]
    per_neuron = []
    correct_signs = 0
    magnitude_errors = []
    combined_errors = []

    for i in range(n_neurons):
        ext = extracted_weights[i]
        true = true_weights[i]
        ext_norm = np.linalg.norm(ext)
        true_norm = np.linalg.norm(true)

        if ext_norm < 1e-12 or true_norm < 1e-12:
            per_neuron.append({
                'neuron': i, 'cosine_sim': 0.0, 'sign_correct': False,
                'magnitude_rel_error': 1.0, 'combined_rel_error': 1.0
            })
            combined_errors.append(1.0)
            magnitude_errors.append(1.0)
            continue

        cos_sim = np.dot(ext, true) / (ext_norm * true_norm)
        sign_correct = cos_sim > 0

        # Magnitude error: align signs first, then measure relative error
        aligned_ext = ext if cos_sim > 0 else -ext
        mag_rel_error = np.linalg.norm(aligned_ext - true) / true_norm

        # Combined error: no alignment
        comb_rel_error = np.linalg.norm(ext - true) / true_norm

        if sign_correct:
            correct_signs += 1

        magnitude_errors.append(mag_rel_error)
        combined_errors.append(comb_rel_error)

        per_neuron.append({
            'neuron': i,
            'cosine_sim': float(cos_sim),
            'sign_correct': bool(sign_correct),
            'magnitude_rel_error': float(mag_rel_error),
            'combined_rel_error': float(comb_rel_error),
        })

    return {
        'sign_accuracy': correct_signs / n_neurons if n_neurons > 0 else 0,
        'magnitude_mean_rel_error': float(np.mean(magnitude_errors)) if magnitude_errors else 1.0,
        'magnitude_median_rel_error': float(np.median(magnitude_errors)) if magnitude_errors else 1.0,
        'combined_mean_rel_error': float(np.mean(combined_errors)) if combined_errors else 1.0,
        'combined_median_rel_error': float(np.median(combined_errors)) if combined_errors else 1.0,
        'mean_abs_cosine_sim': float(np.mean([abs(p['cosine_sim']) for p in per_neuron])),
        'n_neurons': n_neurons,
        'per_neuron': per_neuron,
    }


def test_model_accuracy(model, X_test, Y_test, model_name="Model"):
    """Test model accuracy on test data."""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = outputs.argmax(dim=1)
        correct = (predictions == Y_test).sum().item()
        accuracy = correct / len(Y_test)

    print(f"{model_name} Accuracy: {accuracy:.4f} ({correct}/{len(Y_test)})")
    return accuracy


def load_test_data(tiny=True, makeblobs=False, tinier=False, tiniest=False):
    """Load and preprocess test data."""
    if tiniest:
        if os.path.exists(X_TEST_TINIEST_PATH):
            x_test = np.load(X_TEST_TINIEST_PATH).astype(np.float64)
            y_test = np.load(Y_TEST_TINIEST_PATH) if os.path.exists(Y_TEST_TINIEST_PATH) else np.zeros(len(x_test), dtype=np.int64)
        else:
            print(f"Tiniest test data not found at {X_TEST_TINIEST_PATH}")
            return None, None
        return torch.tensor(x_test, dtype=torch.float64), torch.tensor(y_test, dtype=torch.long)

    if tinier:
        if os.path.exists(X_TEST_TINIER_PATH):
            x_test = np.load(X_TEST_TINIER_PATH).astype(np.float64)
            y_test = np.load(Y_TEST_TINIER_PATH) if os.path.exists(Y_TEST_TINIER_PATH) else np.zeros(len(x_test), dtype=np.int64)
        else:
            print(f"Tinier test data not found at {X_TEST_TINIER_PATH}")
            return None, None
        return torch.tensor(x_test, dtype=torch.float64), torch.tensor(y_test, dtype=torch.long)

    if makeblobs:
        if os.path.exists(X_TEST_MAKEBLOBS_PATH):
            x_test = np.load(X_TEST_MAKEBLOBS_PATH).astype(np.float64)
            y_test = np.load(Y_TEST_MAKEBLOBS_PATH) if os.path.exists(Y_TEST_MAKEBLOBS_PATH) else np.zeros(len(x_test), dtype=np.int64)
        else:
            print(f"Makeblobs test data not found at {X_TEST_MAKEBLOBS_PATH}")
            return None, None
        return torch.tensor(x_test, dtype=torch.float64), torch.tensor(y_test, dtype=torch.long)

    # CIFAR-10 data
    if not os.path.exists(X_TEST_PATH):
        try:
            import tensorflow as tf
            (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            np.save(X_TEST_PATH, x_test)
        except Exception as e:
            print(f"Could not load test data: {e}")
            return None, None

    x_test = np.load(X_TEST_PATH)

    if tiny:
        if len(x_test.shape) == 4 and x_test.shape[-1] == 3:
            x_test = x_test.mean(axis=-1)
        if x_test.shape[1] > 8:
            x_test = x_test[:, ::4, ::4]
        x_test = x_test.reshape(-1, 64)
    else:
        x_test = x_test.reshape(-1, 3072)

    x_test = x_test.astype(np.float64)
    x_test = x_test / 255.0 * 2 - 1

    try:
        import tensorflow as tf
        (_, _), (_, y_test) = tf.keras.datasets.cifar10.load_data()
        y_test = y_test.squeeze()
    except Exception:
        y_test = np.zeros(len(x_test), dtype=np.int64)

    return torch.tensor(x_test, dtype=torch.float64), torch.tensor(y_test, dtype=torch.long)


def load_test2_data(tiny=True, makeblobs=False, tinier=False, tiniest=False):
    """Load fresh eval-only set (seed=99, same scaler). Returns (X_test2, Y_test2) tensors."""
    if tiniest:
        if os.path.exists(X_TEST2_TINIEST_PATH):
            x = np.load(X_TEST2_TINIEST_PATH).astype(np.float64)
            y = np.load(Y_TEST2_TINIEST_PATH) if os.path.exists(Y_TEST2_TINIEST_PATH) else np.zeros(len(x), dtype=np.int64)
            return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.long)
    if makeblobs:
        if os.path.exists(X_TEST2_MAKEBLOBS_PATH):
            x = np.load(X_TEST2_MAKEBLOBS_PATH).astype(np.float64)
            y = np.load(Y_TEST2_MAKEBLOBS_PATH) if os.path.exists(Y_TEST2_MAKEBLOBS_PATH) else np.zeros(len(x), dtype=np.int64)
            return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.long)
    return None, None


def reconstruct_model(signature_path, sign_path, model_class, layer_config, true_model_path=None, random_seed=42,
                      copy_true_biases=True, copy_true_output=True):
    """
    Reconstruct a model by combining signature recovery and sign recovery outputs.

    Handles partial recovery: unrecovered neurons get Kaiming/He random initialization.
    Returns (model, metrics, recovery_stats, recovered_masks_by_layer).
    """
    np.random.seed(random_seed)

    model = model_class()
    metrics = {}
    recovery_stats = {
        'total_neurons': 0,
        'recovered_neurons': 0,
        'random_init_neurons': 0,
        'per_layer': {}
    }
    recovered_masks_by_layer = {}

    true_model = None
    if true_model_path and os.path.exists(true_model_path):
        true_model = load_ground_truth_model(true_model_path, model_class)

    layers = [model.fc1, model.fc2, model.fc3, model.fc4]

    # Precompute layer_offsets for flat neuron ID conversion
    layer_sizes = [v[0] for v in layer_config.values()]
    layer_offsets = [sum(layer_sizes[:i]) for i in range(len(layer_sizes))]

    for layer_id, (layer, (num_neurons, input_dim)) in enumerate(zip(layers, layer_config.values()), start=0):
        print(f"\n--- Layer {layer_id} ({num_neurons} neurons, {input_dim} inputs) ---")

        unsigned_weights, recovered_mask, metadata = load_unsigned_weights(
            signature_path, layer_id, num_neurons, input_dim,
            use_random_init=True, layer_offset=layer_offsets[layer_id]
        )
        recovered_masks_by_layer[layer_id] = recovered_mask

        recovered_count = int(np.sum(recovered_mask))
        recovery_stats['per_layer'][layer_id] = {
            'total': num_neurons,
            'recovered': recovered_count,
            'random_init': num_neurons - recovered_count,
            'recovery_rate': recovered_count / num_neurons
        }
        recovery_stats['total_neurons'] += num_neurons
        recovery_stats['recovered_neurons'] += recovered_count
        recovery_stats['random_init_neurons'] += (num_neurons - recovered_count)

        # Load signs (sign recovery uses 1-indexed layers)
        signs = load_signs(sign_path, layer_id + 1)
        if signs is not None:
            print(f"  Loaded signs: {len(signs)} neurons, {np.sum(signs == 1)} positive, {np.sum(signs == -1)} negative, {np.sum(signs == 0)} unknown")
        else:
            print(f"  Warning: No signs found for layer {layer_id + 1}, using +1 for all")
            signs = np.ones(num_neurons, dtype=np.int8)

        signed_weights = combine_weights_and_signs(unsigned_weights, signs)

        if signed_weights is not None:
            with torch.no_grad():
                layer.weight.data = torch.tensor(signed_weights, dtype=torch.float64)

            if true_model is not None:
                true_layer = [true_model.fc1, true_model.fc2, true_model.fc3, true_model.fc4][layer_id]
                true_weights = true_layer.weight.data.numpy()

                # Compute three-tier metrics (only for recovered neurons)
                recovered_signed = signed_weights[recovered_mask]
                recovered_true = true_weights[recovered_mask]

                if len(recovered_signed) > 0:
                    layer_metrics = compute_weight_metrics_v2(recovered_signed, recovered_true)
                    if layer_metrics:
                        layer_metrics['num_recovered'] = recovered_count
                        layer_metrics['num_random_init'] = num_neurons - recovered_count
                        # Remove per_neuron for summary (too verbose)
                        per_neuron_data = layer_metrics.pop('per_neuron', [])
                        metrics[f'layer_{layer_id}'] = layer_metrics
                        metrics[f'layer_{layer_id}_per_neuron'] = per_neuron_data
                        print(f"  [Recovered neurons only - three-tier metrics]")
                        print(f"    SIGN accuracy:      {layer_metrics['sign_accuracy']:.4f}")
                        print(f"    MAGNITUDE rel err:  {layer_metrics['magnitude_mean_rel_error']:.4f} (median: {layer_metrics['magnitude_median_rel_error']:.4f})")
                        print(f"    COMBINED rel err:   {layer_metrics['combined_mean_rel_error']:.4f} (median: {layer_metrics['combined_median_rel_error']:.4f})")
                        print(f"    Mean |cos sim|:     {layer_metrics['mean_abs_cosine_sim']:.4f}")

                # Also compute metrics for ALL neurons (including random init)
                all_metrics = compute_weight_metrics_v2(signed_weights, true_weights)
                if all_metrics:
                    all_metrics.pop('per_neuron', None)
                    metrics[f'layer_{layer_id}_all'] = all_metrics
                    print(f"  [All neurons (incl. random init)]")
                    print(f"    SIGN accuracy:      {all_metrics['sign_accuracy']:.4f}")
                    print(f"    COMBINED rel err:   {all_metrics['combined_mean_rel_error']:.4f}")
        else:
            print(f"  Using full random initialization for weights")

        # Copy biases from true model (only if explicitly requested — cheating)
        if copy_true_biases and true_model is not None:
            true_layer = [true_model.fc1, true_model.fc2, true_model.fc3, true_model.fc4][layer_id]
            with torch.no_grad():
                layer.bias.data = true_layer.bias.data.clone()
        else:
            with torch.no_grad():
                layer.bias.data.zero_()

    # Copy output layer from true model (only if explicitly requested — cheating)
    if copy_true_output and true_model is not None:
        with torch.no_grad():
            model.fc5.weight.data = true_model.fc5.weight.data.clone()
            model.fc5.bias.data = true_model.fc5.bias.data.clone()
        print(f"\n--- Output Layer ---")
        print(f"  Copied from true model (cheat)")
    else:
        # Initialize fc5 with Kaiming; will be fit via LR on oracle labels
        fan_in = model.fc5.weight.shape[1]
        std = np.sqrt(2.0 / fan_in)
        with torch.no_grad():
            model.fc5.weight.data = torch.tensor(
                np.random.randn(*model.fc5.weight.shape).astype(np.float64) * std,
                dtype=torch.float64
            )
            model.fc5.bias.data.zero_()
        print(f"\n--- Output Layer ---")
        print(f"  Kaiming init (will be LR-fit from oracle labels)")

    # Print recovery summary
    print(f"\n--- Recovery Summary ---")
    total = recovery_stats['total_neurons']
    recovered = recovery_stats['recovered_neurons']
    print(f"  Total neurons: {total}")
    print(f"  Recovered: {recovered} ({100*recovered/total:.1f}%)")
    print(f"  Random init: {total - recovered} ({100*(total-recovered)/total:.1f}%)")

    return model, metrics, recovery_stats, recovered_masks_by_layer


DUAL_POINTS_DIR = os.path.join(BASE_DIR, "sign_recovery/layer_neuron_npys")


def oracle_label_refinement(reconstructed_model, oracle_model, X_train,
                             recovered_masks, n_epochs=300, lr=5e-3,
                             freeze_recovered_weights=True, verbose=True):
    """
    Polish the reconstructed model against oracle hard labels.

    When freeze_recovered_weights=True, weight rows for signature-recovered
    neurons have their gradients zeroed — only biases, fc5, and rows of
    random-init (unrecovered) neurons update. This keeps the attack's
    extracted identity intact while allowing non-extracted components
    (biases, output layer, and neurons never reached by find_duals) to
    absorb oracle-label information.
    """
    reconstructed_model.train()
    oracle_model.eval()
    with torch.no_grad():
        oracle_labels = oracle_model(X_train).argmax(dim=1)

    hidden_layers = [reconstructed_model.fc1, reconstructed_model.fc2,
                     reconstructed_model.fc3, reconstructed_model.fc4]

    # Build boolean grad-freeze masks per layer (True = frozen / zero grad)
    freeze_row_masks = {}
    if freeze_recovered_weights:
        for lid, mask in recovered_masks.items():
            freeze_row_masks[lid] = torch.tensor(mask, dtype=torch.bool)

    optimizer = torch.optim.Adam(reconstructed_model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    n_loggings = 10
    log_every = max(1, n_epochs // n_loggings)

    with torch.no_grad():
        preds0 = reconstructed_model(X_train).argmax(dim=1)
        start_agree = (preds0 == oracle_labels).float().mean().item()
    if verbose:
        print(f"  [refine] start agreement={start_agree:.4f}, "
              f"{'frozen recovered weights' if freeze_recovered_weights else 'all params trainable'}, "
              f"n_epochs={n_epochs}, lr={lr}")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = reconstructed_model(X_train)
        loss = loss_fn(preds, oracle_labels)
        loss.backward()

        if freeze_recovered_weights:
            for lid, layer in enumerate(hidden_layers):
                row_mask = freeze_row_masks.get(lid)
                if row_mask is not None and layer.weight.grad is not None:
                    layer.weight.grad[row_mask] = 0.0

        optimizer.step()

        if verbose and (epoch == 0 or (epoch + 1) % log_every == 0 or epoch == n_epochs - 1):
            with torch.no_grad():
                preds = reconstructed_model(X_train).argmax(dim=1)
                agree = (preds == oracle_labels).float().mean().item()
            print(f"  [refine] epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}  agreement={agree:.4f}")

    reconstructed_model.eval()
    with torch.no_grad():
        preds = reconstructed_model(X_train).argmax(dim=1)
        final_agree = (preds == oracle_labels).float().mean().item()
    return {
        'start_agreement': float(start_agree),
        'final_agreement': float(final_agree),
        'freeze_recovered_weights': bool(freeze_recovered_weights),
        'n_epochs': int(n_epochs),
        'lr': float(lr),
    }


def _hidden_activations_up_to(reconstructed_model, x, up_to_layer):
    """Forward x through fc1..fc{up_to_layer-1} with ReLU. Returns h (no ReLU on boundary layer)."""
    layers = [reconstructed_model.fc1, reconstructed_model.fc2,
              reconstructed_model.fc3, reconstructed_model.fc4]
    h = x
    for l_idx in range(up_to_layer):
        h = _act(layers[l_idx](h))
    return h


def recover_biases_from_duals(reconstructed_model, duals_dir, recovered_masks,
                               layer_ids=(0, 1, 2, 3), n_duals=30, verbose=True):
    """
    For each recovered neuron i in layer L: b_i = -w_i · h_{L-1}(x_d) where x_d
    is any dual point of neuron i. Uses the already-reconstructed lower layers
    for h. Called bottom-up so lower layers are stable.

    Median over n_duals dual points for robustness.
    """
    layers = [reconstructed_model.fc1, reconstructed_model.fc2,
              reconstructed_model.fc3, reconstructed_model.fc4]

    for lid in sorted(layer_ids):
        layer = layers[lid]
        mask = recovered_masks.get(lid)
        if mask is None:
            continue
        biases = layer.bias.data.clone()
        n_set = 0
        for i in range(len(mask)):
            if not mask[i]:
                continue
            dual_path = os.path.join(duals_dir, f"layer{lid+1}_neuron{i}.npy")
            if not os.path.exists(dual_path):
                continue
            duals = np.load(dual_path)
            if len(duals) == 0:
                continue
            x_d = torch.tensor(duals[:n_duals], dtype=torch.float64)
            with torch.no_grad():
                h = _hidden_activations_up_to(reconstructed_model, x_d, lid)
                w_i = layer.weight.data[i]
                b_candidates = -(h @ w_i)
                biases[i] = b_candidates.median()
            n_set += 1
        layer.bias.data = biases
        if verbose:
            print(f"  [bias-recov] Layer {lid}: set {n_set}/{int(mask.sum())} biases from dual points")


def recover_output_layer(reconstructed_model, oracle_model, X_samples, verbose=True, n_aug=8000):
    """
    Recover fc5 using hard-label oracle queries.

    Queries the oracle on X_samples + augmented samples (Gaussian perturbations
    + random coverage) to build (h_4, label) training set, then fits 8-way
    multinomial logistic regression.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("sklearn not available, skipping output layer recovery")
        return

    oracle_model.eval()
    reconstructed_model.eval()

    # LR fit on X_samples directly. Out-of-distribution augmentation (uniform
    # coverage, wide Gaussian) was found to distort the fit away from the X_test
    # region without improving in-distribution accuracy.
    X_big = X_samples.numpy().astype(np.float64)
    X_big_t = X_samples

    with torch.no_grad():
        oracle_labels = oracle_model(X_big_t).argmax(dim=1).numpy()
        h4 = _hidden_activations_up_to(reconstructed_model, X_big_t, up_to_layer=4).numpy()

    # multinomial logistic regression; uses only hard labels
    n_classes = int(oracle_labels.max()) + 1
    lr = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        max_iter=2000, C=1e6,  # very weak regularization
        fit_intercept=True,
    )
    lr.fit(h4, oracle_labels)

    fc5 = reconstructed_model.fc5
    out_dim = fc5.weight.shape[0]
    # LR coef_ has shape (n_classes_used, input_dim). Expand to full output shape.
    coef = np.zeros((out_dim, h4.shape[1]), dtype=np.float64)
    intercept = np.zeros(out_dim, dtype=np.float64)
    for idx, cls in enumerate(lr.classes_):
        if cls < out_dim:
            coef[cls] = lr.coef_[idx]
            intercept[cls] = lr.intercept_[idx]

    with torch.no_grad():
        fc5.weight.data = torch.tensor(coef, dtype=torch.float64)
        fc5.bias.data = torch.tensor(intercept, dtype=torch.float64)

    if verbose:
        with torch.no_grad():
            small_oracle = oracle_model(X_samples).argmax(dim=1).numpy()
            small_recon = reconstructed_model(X_samples).argmax(dim=1).numpy()
            lr_preds = lr.predict(h4)
        print(f"  [fc5-recov] LR fit on {len(X_big)} samples ({len(X_samples)} original + augmented), "
              f"{n_classes} classes seen; LR train acc vs oracle = "
              f"{(lr_preds == oracle_labels).mean():.4f}; "
              f"reconstructed vs oracle on original X_test = {(small_recon == small_oracle).mean():.4f}")


def oracle_sign_search(reconstructed_model, oracle_model, X_test, recovered_masks,
                        layer_ids=(0, 1, 2, 3), n_passes=3, order='both', verbose=True,
                        duals_dir=None):
    """
    Brute-force sign search using only hard-label oracle queries.

    For each layer, enumerate all 2^k sign flips over the k recovered neurons
    and pick the flip combo that maximizes label agreement with the hard-label
    oracle on X_test. Iterate n_passes times with alternating direction
    (top-down, bottom-up) until convergence — this matters because downstream
    errors mask upstream signs and vice versa.

    This is genuinely black-box: only oracle(X_test).argmax is used.
    """
    reconstructed_model.eval()
    oracle_model.eval()
    with torch.no_grad():
        oracle_labels = oracle_model(X_test).argmax(dim=1)

    layers = [reconstructed_model.fc1, reconstructed_model.fc2,
              reconstructed_model.fc3, reconstructed_model.fc4]

    results = {}
    prev_agree = -1.0

    def _current_agreement():
        with torch.no_grad():
            preds = reconstructed_model(X_test).argmax(dim=1)
            return (preds == oracle_labels).float().mean().item()

    start_agree = _current_agreement()
    if verbose:
        print(f"  [sign-search] starting oracle agreement: {start_agree:.4f}")

    for pass_i in range(n_passes):
        if order == 'top-down':
            this_order = list(reversed(layer_ids))
        elif order == 'bottom-up':
            this_order = list(layer_ids)
        elif order == 'both':
            this_order = list(reversed(layer_ids)) if pass_i % 2 == 0 else list(layer_ids)
        else:
            this_order = list(layer_ids)

        if verbose:
            print(f"  [sign-search] pass {pass_i+1}/{n_passes} order={this_order}")
        _run_one_pass(reconstructed_model, layers, recovered_masks, this_order,
                      X_test, oracle_labels, results, verbose,
                      duals_dir=duals_dir)

        cur_agree = _current_agreement()
        if verbose:
            print(f"  [sign-search] pass {pass_i+1} agreement: {cur_agree:.4f}")
        if cur_agree <= prev_agree + 1e-6:
            if verbose:
                print(f"  [sign-search] converged (no improvement), stopping early")
            break
        prev_agree = cur_agree

    results['final_agreement'] = _current_agreement()
    results['starting_agreement'] = start_agree
    return results


def _run_one_pass(reconstructed_model, layers, recovered_masks, layer_order,
                  X_test, oracle_labels, results, verbose, duals_dir=None):
    """Single pass of per-layer brute-force sign search. Mutates layers in place.

    If duals_dir is provided, biases are kept consistent with weight signs via
    b_i = -w_i · h_{L-1}(x_d).median() — so flipping w_i also flips b_i.
    """
    for lid in layer_order:
        layer = layers[lid]
        mask = recovered_masks.get(lid)
        if mask is None:
            continue
        recovered_idx = np.where(mask)[0]
        k = len(recovered_idx)
        if k == 0:
            continue
        if k > 18:
            if verbose:
                print(f"  [sign-search] Layer {lid}: {k} recovered neurons — brute force infeasible (2^{k}), using greedy")
            n_flipped = _greedy_sign_pass_layer(
                reconstructed_model, layers, lid, recovered_masks,
                X_test, oracle_labels, duals_dir=duals_dir,
            )
            cur_agree = (reconstructed_model(X_test).argmax(dim=1) == oracle_labels).float().mean().item()
            results[lid] = {
                'recovered': int(k),
                'flipped': int(n_flipped),
                'best_agreement': float(cur_agree),
                'method': 'greedy',
            }
            if verbose:
                print(f"  [sign-search] Layer {lid}: greedy flipped {n_flipped}/{k}, agreement {cur_agree:.4f}")
            continue

        original_weight = layer.weight.data.clone()
        original_bias = layer.bias.data.clone()
        # Baseline: current agreement with no changes at all to this layer
        with torch.no_grad():
            preds0 = reconstructed_model(X_test).argmax(dim=1)
            baseline_agree = (preds0 == oracle_labels).float().mean().item()
        best_agree = -1.0
        best_combo = 0

        # Precompute h·|w_i| medians if recomputing biases from duals.
        # proj[neuron_idx] = median of (original_weight[neuron_idx] · h_{L-1}(x_d_i))
        # so that b_i_for_current_weight_sign = -proj[neuron_idx].
        projections = {}
        if duals_dir is not None:
            with torch.no_grad():
                for neuron_idx in recovered_idx:
                    dpath = os.path.join(duals_dir, f"layer{lid+1}_neuron{int(neuron_idx)}.npy")
                    if not os.path.exists(dpath):
                        continue
                    duals = np.load(dpath)
                    if len(duals) == 0:
                        continue
                    x_d = torch.tensor(duals[:30], dtype=torch.float64)
                    h = _hidden_activations_up_to(reconstructed_model, x_d, lid)
                    projections[int(neuron_idx)] = float((h @ original_weight[int(neuron_idx)]).median())

        if verbose:
            mode = "w+b (joint)" if duals_dir else "w only"
            print(f"  [sign-search] Layer {lid}: searching 2^{k}={2**k} combos ({mode})")

        with torch.no_grad():
            for combo in range(2 ** k):
                new_weight = original_weight.clone()
                new_bias = original_bias.clone()
                for bit_idx, neuron_idx in enumerate(recovered_idx):
                    if (combo >> bit_idx) & 1:
                        new_weight[int(neuron_idx)] = -original_weight[int(neuron_idx)]
                    if duals_dir is not None and int(neuron_idx) in projections:
                        sign = -1.0 if (combo >> bit_idx) & 1 else 1.0
                        # b_i = -w_i · h = -(sign * |w_i|) · h = -sign * (w_orig · h)
                        new_bias[int(neuron_idx)] = -sign * projections[int(neuron_idx)]
                layer.weight.data = new_weight
                layer.bias.data = new_bias
                preds = reconstructed_model(X_test).argmax(dim=1)
                agree = (preds == oracle_labels).float().mean().item()
                if agree > best_agree:
                    best_agree = agree
                    best_combo = combo

        # Safety: if best combo doesn't beat baseline, revert entirely — protects
        # against bias-recomputation using drifted lower layers making things worse.
        if best_agree < baseline_agree - 1e-6:
            layer.weight.data = original_weight
            layer.bias.data = original_bias
            results[lid] = {
                'recovered': int(k),
                'flipped': 0,
                'best_agreement': float(baseline_agree),
                'best_combo': 0,
                'reverted': True,
            }
            if verbose:
                print(f"  [sign-search] Layer {lid}: best combo {best_agree:.4f} < baseline {baseline_agree:.4f}, reverted")
            continue

        # Apply best combo permanently
        final_weight = original_weight.clone()
        final_bias = original_bias.clone()
        n_flipped = 0
        for bit_idx, neuron_idx in enumerate(recovered_idx):
            if (best_combo >> bit_idx) & 1:
                final_weight[int(neuron_idx)] = -original_weight[int(neuron_idx)]
                n_flipped += 1
            if duals_dir is not None and int(neuron_idx) in projections:
                sign = -1.0 if (best_combo >> bit_idx) & 1 else 1.0
                final_bias[int(neuron_idx)] = -sign * projections[int(neuron_idx)]
        layer.weight.data = final_weight
        layer.bias.data = final_bias

        results[lid] = {
            'recovered': int(k),
            'flipped': int(n_flipped),
            'best_agreement': float(best_agree),
            'best_combo': int(best_combo),
        }
        if verbose:
            print(f"  [sign-search] Layer {lid}: best agreement {best_agree:.4f} (baseline {baseline_agree:.4f}), flipped {n_flipped}/{k} signs")

    return results


def _greedy_sign_pass_layer(reconstructed_model, layers, lid, recovered_masks,
                              X_train, oracle_labels, duals_dir=None):
    """
    One greedy pass over recovered neurons in a single layer.

    For each neuron: flip its sign (and optionally recompute bias from duals).
    Keep the flip if oracle agreement improves; else revert.
    Returns number of neurons flipped.
    """
    layer = layers[lid]
    mask = recovered_masks.get(lid)
    if mask is None:
        return 0
    recovered_idx = np.where(mask)[0]
    if len(recovered_idx) == 0:
        return 0

    n_flipped = 0
    with torch.no_grad():
        for neuron_idx_t in recovered_idx:
            neuron_idx = int(neuron_idx_t)

            preds_curr = reconstructed_model(X_train).argmax(dim=1)
            agree_curr = (preds_curr == oracle_labels).float().mean().item()

            orig_w = layer.weight.data[neuron_idx].clone()
            orig_b = layer.bias.data[neuron_idx].clone()

            layer.weight.data[neuron_idx] = -orig_w

            if duals_dir is not None:
                dpath = os.path.join(duals_dir, f"layer{lid+1}_neuron{neuron_idx}.npy")
                if os.path.exists(dpath):
                    duals = np.load(dpath)
                    if len(duals) > 0:
                        x_d = torch.tensor(duals[:30], dtype=torch.float64)
                        h = _hidden_activations_up_to(reconstructed_model, x_d, lid)
                        layer.bias.data[neuron_idx] = -(h @ layer.weight.data[neuron_idx]).median()

            preds_flip = reconstructed_model(X_train).argmax(dim=1)
            agree_flip = (preds_flip == oracle_labels).float().mean().item()

            if agree_flip > agree_curr + 1e-7:
                n_flipped += 1
            else:
                layer.weight.data[neuron_idx] = orig_w
                layer.bias.data[neuron_idx] = orig_b

    return n_flipped


def greedy_oracle_sign_search(reconstructed_model, oracle_model, X_train, recovered_masks,
                               layer_ids=(0, 1, 2, 3), n_passes=5, verbose=True,
                               duals_dir=None):
    """
    Greedy O(k)-per-pass sign search using hard-label oracle queries.

    For each pass: for each layer (alternating direction), for each recovered
    neuron, flip its sign and keep if oracle agreement improves.
    Works for any layer width — no 2^k restriction.
    This function is called automatically from oracle_sign_search when k > 18.
    """
    reconstructed_model.eval()
    oracle_model.eval()
    with torch.no_grad():
        oracle_labels = oracle_model(X_train).argmax(dim=1)

    layers = [reconstructed_model.fc1, reconstructed_model.fc2,
              reconstructed_model.fc3, reconstructed_model.fc4]

    def _cur_agree():
        with torch.no_grad():
            return (reconstructed_model(X_train).argmax(dim=1) == oracle_labels).float().mean().item()

    start_agree = _cur_agree()
    if verbose:
        print(f"  [greedy-sign-search] starting agreement: {start_agree:.4f}")

    prev_agree = -1.0
    results = {'starting_agreement': float(start_agree), 'passes': []}

    for pass_i in range(n_passes):
        this_order = list(reversed(layer_ids)) if pass_i % 2 == 0 else list(layer_ids)
        if verbose:
            print(f"  [greedy-sign-search] pass {pass_i+1}/{n_passes} order={this_order}")

        total_flipped = 0
        for lid in this_order:
            n_flip = _greedy_sign_pass_layer(
                reconstructed_model, layers, lid, recovered_masks,
                X_train, oracle_labels, duals_dir=duals_dir,
            )
            total_flipped += n_flip
            if verbose and n_flip > 0:
                mask = recovered_masks.get(lid)
                k = int(mask.sum()) if mask is not None else 0
                print(f"    layer {lid}: flipped {n_flip}/{k}, agree={_cur_agree():.4f}")

        cur_agree = _cur_agree()
        if verbose:
            print(f"  [greedy-sign-search] pass {pass_i+1} agreement: {cur_agree:.4f} ({total_flipped} flips)")

        results['passes'].append({'pass': pass_i + 1, 'agreement': float(cur_agree),
                                   'total_flipped': total_flipped})

        if cur_agree <= prev_agree + 1e-6 and total_flipped == 0:
            if verbose:
                print(f"  [greedy-sign-search] converged, stopping early")
            break
        prev_agree = cur_agree

    results['final_agreement'] = _cur_agree()
    return results


def save_reconstructed_model(model, output_path, name="reconstructed_model"):
    """Save the reconstructed model in multiple formats."""
    Path(output_path).mkdir(parents=True, exist_ok=True)

    pth_path = os.path.join(output_path, f"{name}.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"Saved PyTorch model to: {pth_path}")

    weights_path = os.path.join(output_path, f"{name}_weights.npz")
    weights_dict = {}
    for name_param, param in model.named_parameters():
        weights_dict[name_param] = param.detach().numpy()
    np.savez(weights_path, **weights_dict)
    print(f"Saved weights to: {weights_path}")


def main():
    parser = argparse.ArgumentParser(description="Model Extraction Verification")
    parser.add_argument('--tiny', action='store_true', default=True, help="Use tiny model (64x64)")
    parser.add_argument('--full', action='store_true', help="Use full model (3072x256)")
    parser.add_argument('--makeblobs', action='store_true', help="Use makeblobs model (64x64, synthetic data)")
    parser.add_argument('--tinier', action='store_true', help="Use tinier model (32->16->16->16->8->4)")
    parser.add_argument('--tiniest', action='store_true', help="Use tiniest model (8-8-8-8-8-8, make_blobs)")
    parser.add_argument('--signature-path', type=str, default=SIGNATURE_WEIGHTS_PATH)
    parser.add_argument('--sign-path', type=str, default=SIGN_RECOVERY_PATH)
    parser.add_argument('--output-path', type=str, default=OUTPUT_PATH)
    parser.add_argument('--sign-search', action='store_true',
                        help="After reconstruction, brute-force sign combos per layer using only hard-label oracle queries on X_test")
    parser.add_argument('--from-scratch', action='store_true',
                        help="Rebuild model from scratch: no cheat biases, no cheat fc5. Implies --sign-search with joint w+b flipping, plus fc5 LR-fit on oracle labels")
    parser.add_argument('--duals-dir', type=str, default=DUAL_POINTS_DIR,
                        help="Directory holding layer{L}_neuron{i}.npy dual point files")
    parser.add_argument('--refine', action='store_true',
                        help="After sign search + fc5 LR fit, polish the model against oracle hard labels. Freezes extracted weight rows; only biases, fc5, and unrecovered neurons' rows are updated")
    parser.add_argument('--refine-unfreeze', action='store_true',
                        help="When combined with --refine, unfreeze ALL weights (full distillation). Strays furthest from 'extraction' but pushes accuracy closer to 100%")
    parser.add_argument('--refine-epochs', type=int, default=300)
    parser.add_argument('--refine-lr', type=float, default=5e-3)
    args = parser.parse_args()
    if args.from_scratch:
        args.sign_search = True

    print("="*70)
    print("MODEL EXTRACTION VERIFICATION (v2 - Three-Tier Metrics)")
    print("="*70)

    # Determine model configuration
    if args.tiniest:
        model_class = TiniestModel
        true_model_path = TINIEST_MODEL_PTH
        tiny = False
        makeblobs = False
        tinier = False
        tiniest = True
        layer_config = {0: (8, 8), 1: (8, 8), 2: (8, 8), 3: (8, 8)}
    elif args.tinier:
        model_class = TinierModel
        true_model_path = TINIER_MODEL_PTH
        tiny = False
        makeblobs = False
        tinier = True
        tiniest = False
        # Non-uniform layer config: (num_neurons, input_dim)
        layer_config = {0: (16, 32), 1: (16, 16), 2: (16, 16), 3: (8, 16)}
    elif args.full:
        model_class = FullModel
        true_model_path = FULL_MODEL_PTH
        tiny = False
        makeblobs = False
        tinier = False
        tiniest = False
        layer_config = {0: (256, 3072), 1: (256, 256), 2: (256, 256), 3: (64, 256)}
    elif args.makeblobs:
        model_class = TinyModel
        true_model_path = MAKEBLOBS_MODEL_PTH
        tiny = True
        makeblobs = True
        tinier = False
        tiniest = False
        layer_config = {0: (64, 64), 1: (64, 64), 2: (64, 64), 3: (64, 64)}
    else:
        model_class = TinyModel
        true_model_path = TINY_MODEL_PTH
        makeblobs = False
        tiny = True
        tinier = False
        tiniest = False
        layer_config = {0: (64, 64), 1: (64, 64), 2: (64, 64), 3: (64, 64)}

    if tiniest:
        model_type_str = "Tiniest (8-8-8-8-8-8, make_blobs)"
    elif tinier:
        model_type_str = "Tinier (32->16->16->16->8->4, make_blobs)"
    elif makeblobs:
        model_type_str = "Makeblobs (64x64, synthetic data)"
    elif tiny:
        model_type_str = "Tiny (64x64, CIFAR-10)"
    else:
        model_type_str = "Full (3072x256, CIFAR-10)"

    print(f"\nModel type: {model_type_str}")
    print(f"Architecture: {list(layer_config.values())}")
    print(f"Ground truth model: {true_model_path}")
    print(f"Signature weights path: {args.signature_path}")
    print(f"Sign recovery path: {args.sign_path}")

    # Load test data
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    # X_test: used for Phase-3 oracle training (sign search, fc5 LR, refinement)
    X_test, Y_test = load_test_data(tiny=tiny, makeblobs=makeblobs, tinier=tinier, tiniest=tiniest)
    if X_test is None:
        print("Failed to load test data")
        return
    print(f"X_test (Phase-3 training) shape: {X_test.shape}")

    # X_test2: fresh eval-only set (seed=99) — no overlap with Phase-3 training
    X_test2, Y_test2 = load_test2_data(tiny=tiny, makeblobs=makeblobs, tinier=tinier, tiniest=tiniest)
    if X_test2 is None:
        print("  Warning: X_test2 not found, falling back to X_test for evaluation")
        X_test2, Y_test2 = X_test, Y_test
    else:
        print(f"X_test2 (eval-only, seed=99) shape: {X_test2.shape}")

    # Load and test ground truth model
    print("\n" + "="*70)
    print("GROUND TRUTH MODEL")
    print("="*70)
    true_model = load_ground_truth_model(true_model_path, model_class)
    true_accuracy = test_model_accuracy(true_model, X_test2, Y_test2, "Ground Truth (on X_test2)")

    # Reconstruct model
    print("\n" + "="*70)
    print("RECONSTRUCTING MODEL FROM EXTRACTION")
    print("="*70)
    print("\nNOTE: Unrecovered neurons use Kaiming/He initialization")
    print("      Scaling uses abs(factor) to ensure sign is NOT revealed\n")
    reconstructed_model, layer_metrics, recovery_stats, recovered_masks_by_layer = reconstruct_model(
        args.signature_path,
        args.sign_path,
        model_class,
        layer_config,
        true_model_path,
        random_seed=42,
        copy_true_biases=not args.from_scratch,
        copy_true_output=not args.from_scratch,
    )

    # For --from-scratch: recover biases from duals bottom-up *before* sign search,
    # so sign search operates on a consistent (w, b) starting point.
    if args.from_scratch:
        print("\n" + "="*70)
        print("BIAS RECOVERY FROM DUAL POINTS (bottom-up)")
        print("="*70)
        recover_biases_from_duals(
            reconstructed_model, args.duals_dir, recovered_masks_by_layer,
            layer_ids=tuple(range(len(layer_config))), verbose=True,
        )

    # Pre-sign-search accuracy on eval set (no training overlap)
    pre_search_accuracy = test_model_accuracy(reconstructed_model, X_test2, Y_test2, "Pre-sign-search (X_test2)")

    sign_search_results = None
    if args.sign_search:
        print("\n" + "="*70)
        print("ORACLE-QUERIES-ONLY SIGN SEARCH")
        print("="*70)
        print("Brute-forcing 2^k sign combos per layer using only hard-label oracle queries on X_test")
        duals_for_search = args.duals_dir if args.from_scratch else None
        sign_search_results = oracle_sign_search(
            reconstructed_model, true_model, X_test, recovered_masks_by_layer,
            layer_ids=tuple(range(len(layer_config))), verbose=True,
            duals_dir=duals_for_search,
        )

        # For --from-scratch: after sign search, fit fc5 via LR on oracle labels
        if args.from_scratch:
            print("\n" + "="*70)
            print("OUTPUT LAYER (fc5) RECOVERY via LR on oracle labels")
            print("="*70)
            recover_output_layer(reconstructed_model, true_model, X_test, verbose=True)

    refine_results = None
    if args.refine:
        print("\n" + "="*70)
        print("ORACLE-LABEL REFINEMENT")
        print("="*70)
        refine_results = oracle_label_refinement(
            reconstructed_model, true_model, X_test, recovered_masks_by_layer,
            n_epochs=args.refine_epochs, lr=args.refine_lr,
            freeze_recovered_weights=not args.refine_unfreeze,
            verbose=True,
        )

        # Recompute per-layer metrics against true weights after sign flips
        print("\n--- Post-sign-search per-layer metrics ---")
        layers_list = [reconstructed_model.fc1, reconstructed_model.fc2,
                       reconstructed_model.fc3, reconstructed_model.fc4]
        true_layers = [true_model.fc1, true_model.fc2, true_model.fc3, true_model.fc4]
        for lid in range(len(layer_config)):
            mask = recovered_masks_by_layer.get(lid)
            if mask is None or not mask.any():
                continue
            ext = layers_list[lid].weight.data.numpy()
            true_w = true_layers[lid].weight.data.numpy()
            m = compute_weight_metrics_v2(ext[mask], true_w[mask])
            if m:
                m.pop('per_neuron', None)
                m['num_recovered'] = int(mask.sum())
                m['num_random_init'] = int(len(mask) - mask.sum())
                layer_metrics[f'layer_{lid}'] = m
                print(f"  layer_{lid}: sign_acc={m['sign_accuracy']:.4f}  "
                      f"|cos|={m['mean_abs_cosine_sim']:.4f}  "
                      f"mag_rel_err={m['magnitude_mean_rel_error']:.4f}")

    # Final evaluation on X_test2 (fresh, no Phase-3 training overlap)
    print("\n" + "="*70)
    print("RECONSTRUCTED MODEL EVALUATION (on X_test2 — fresh eval set, seed=99)")
    print("="*70)
    recon_accuracy = test_model_accuracy(reconstructed_model, X_test2, Y_test2, "Reconstructed (X_test2)")

    # Compare predictions on X_test2
    print("\n--- Prediction Comparison (X_test2) ---")
    with torch.no_grad():
        true_preds = true_model(X_test2).argmax(dim=1)
        recon_preds = reconstructed_model(X_test2).argmax(dim=1)
        pred_agreement = (true_preds == recon_preds).float().mean().item()
    print(f"Prediction agreement (X_test2): {pred_agreement:.4f}")

    # Summary with three-tier metrics
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY (Three-Tier Metrics)")
    print("="*70)

    print("\n--- Per-Layer Metrics (Recovered Neurons Only) ---")
    summary_layers = {k: v for k, v in layer_metrics.items() if not k.endswith('_per_neuron') and not k.endswith('_all')}
    for layer_name, m in sorted(summary_layers.items()):
        if 'sign_accuracy' not in m:
            continue
        print(f"\n{layer_name} ({m.get('num_recovered', '?')}/{m.get('num_recovered', 0) + m.get('num_random_init', 0)} recovered):")
        print(f"  SIGN accuracy:      {m['sign_accuracy']:.4f}")
        print(f"  MAGNITUDE rel err:  {m['magnitude_mean_rel_error']:.4f} (median: {m['magnitude_median_rel_error']:.4f})")
        print(f"  COMBINED rel err:   {m['combined_mean_rel_error']:.4f} (median: {m['combined_median_rel_error']:.4f})")
        print(f"  Mean |cos sim|:     {m['mean_abs_cosine_sim']:.4f}")

    # Overall averages
    if summary_layers:
        valid = [m for m in summary_layers.values() if 'sign_accuracy' in m]
        if valid:
            avg_sign = np.mean([m['sign_accuracy'] for m in valid])
            avg_mag = np.mean([m['magnitude_mean_rel_error'] for m in valid])
            avg_comb = np.mean([m['combined_mean_rel_error'] for m in valid])
            avg_cos = np.mean([m['mean_abs_cosine_sim'] for m in valid])

            print(f"\n--- Overall Averages (across layers) ---")
            print(f"  SIGN accuracy:      {avg_sign:.4f}")
            print(f"  MAGNITUDE rel err:  {avg_mag:.4f}")
            print(f"  COMBINED rel err:   {avg_comb:.4f}")
            print(f"  Mean |cos sim|:     {avg_cos:.4f}")

    print(f"\n--- Model Performance ---")
    print(f"Ground truth accuracy: {true_accuracy:.4f}")
    print(f"Reconstructed accuracy: {recon_accuracy:.4f}")
    print(f"Accuracy difference: {abs(true_accuracy - recon_accuracy):.4f}")
    print(f"Prediction agreement: {pred_agreement:.4f}")

    extraction_success = pred_agreement > 0.95 and recon_accuracy > 0.9 * true_accuracy
    print(f"\n*** EXTRACTION {'SUCCESSFUL' if extraction_success else 'NEEDS IMPROVEMENT'} ***")

    # Save reconstructed model
    print("\n" + "="*70)
    print("SAVING RECONSTRUCTED MODEL")
    print("="*70)
    if tiniest:
        model_name = "reconstructed_tiniest"
    elif tinier:
        model_name = "reconstructed_tinier"
    elif makeblobs:
        model_name = "reconstructed_makeblobs"
    elif tiny:
        model_name = "reconstructed_tiny"
    else:
        model_name = "reconstructed_full"
    save_reconstructed_model(reconstructed_model, args.output_path, model_name)

    # Save metrics
    metrics_path = os.path.join(args.output_path, "extraction_metrics.json")
    # Filter out per-neuron data for JSON serialization
    serializable_metrics = {}
    for k, v in layer_metrics.items():
        if k.endswith('_per_neuron'):
            continue  # Skip per-neuron lists for the summary JSON
        if isinstance(v, dict):
            serializable_metrics[k] = {
                kk: float(vv) if isinstance(vv, (float, np.floating)) else int(vv) if isinstance(vv, (int, np.integer)) else vv
                for kk, vv in v.items()
            }
        else:
            serializable_metrics[k] = v

    all_metrics = {
        'model_type': model_type_str,
        'model_name': model_name,
        'layer_config': {str(k): list(v) for k, v in layer_config.items()},
        'layer_metrics': serializable_metrics,
        'recovery_stats': {
            'total_neurons': recovery_stats['total_neurons'],
            'recovered_neurons': recovery_stats['recovered_neurons'],
            'random_init_neurons': recovery_stats['random_init_neurons'],
            'overall_recovery_rate': recovery_stats['recovered_neurons'] / max(1, recovery_stats['total_neurons']),
            'per_layer': {str(k): v for k, v in recovery_stats['per_layer'].items()}
        },
        'true_accuracy': float(true_accuracy),
        'reconstructed_accuracy': float(recon_accuracy),
        'pre_sign_search_accuracy': float(pre_search_accuracy),
        'prediction_agreement': float(pred_agreement),
        'extraction_success': extraction_success,
        'sign_search_applied': bool(args.sign_search),
        'sign_search_results': sign_search_results,
        'refinement_applied': bool(args.refine),
        'refinement_results': refine_results,
        'from_scratch': bool(args.from_scratch),
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
