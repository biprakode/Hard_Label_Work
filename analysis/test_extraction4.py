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
BASE_DIR = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction"

# Signature recovery outputs (unsigned weights)
SIGNATURE_WEIGHTS_PATH = os.path.join(BASE_DIR, "signature_recovery/outputs/model_weights/Vrelu")

# Sign recovery outputs
SIGN_RECOVERY_PATH = os.path.join(BASE_DIR, "results/sign_recovery")

# Ground truth models
TINY_MODEL_PTH = os.path.join(BASE_DIR, "tiny_shit/TinyModel_relu.pth")
TINY_MODEL_KERAS = os.path.join(BASE_DIR, "tiny_shit/TinyModel_relu.keras")
MAKEBLOBS_MODEL_PTH = os.path.join(BASE_DIR, "tiny_shit/makeblobs_relu.pth")
TINIER_MODEL_PTH = os.path.join(BASE_DIR, "tiny_shit/tinier_makeblobs_relu.pth")
FULL_MODEL_PTH = os.path.join(BASE_DIR, "signature_recovery/models/converted_model.pth")

# Test data
X_TEST_PATH = os.path.join(BASE_DIR, "data/x_test.npy")
X_TEST_MAKEBLOBS_PATH = os.path.join(BASE_DIR, "data/x_test_makeblobs.npy")
Y_TEST_MAKEBLOBS_PATH = os.path.join(BASE_DIR, "data/y_test_makeblobs.npy")
X_TEST_TINIER_PATH = os.path.join(BASE_DIR, "data/x_test_tinier_makeblobs.npy")
Y_TEST_TINIER_PATH = os.path.join(BASE_DIR, "data/y_test_tinier_makeblobs.npy")

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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
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


def load_unsigned_weights(signature_path, layer_id, num_neurons, input_dim, use_random_init=True):
    """
    Load unsigned weight vectors from signature recovery output.

    Uses weights_unscaled.npz + abs(scaling_factor) to ensure
    the scaling does NOT reveal sign information (only magnitude).
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
            neuron_id = int(neuron_dir.name.split("_")[1])

            if neuron_id >= num_neurons:
                continue

            # Load metadata to get scaling factor
            metadata_path = neuron_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                scaling_factor = meta.get('scaling_factor', 1.0)
                metadata_dict[neuron_id] = meta
            else:
                scaling_factor = 1.0
                metadata_dict[neuron_id] = {'scaling_factor': 1.0}

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
    """Combine unsigned weights with recovered signs."""
    if unsigned_weights is None or signs is None:
        return None

    num_neurons = unsigned_weights.shape[0]
    if len(signs) < num_neurons:
        signs = np.concatenate([signs, np.ones(num_neurons - len(signs), dtype=np.int8)])

    return unsigned_weights * signs[:num_neurons, np.newaxis]


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


def load_test_data(tiny=True, makeblobs=False, tinier=False):
    """Load and preprocess test data."""
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


def reconstruct_model(signature_path, sign_path, model_class, layer_config, true_model_path=None, random_seed=42):
    """
    Reconstruct a model by combining signature recovery and sign recovery outputs.

    Handles partial recovery: unrecovered neurons get Kaiming/He random initialization.
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

    true_model = None
    if true_model_path and os.path.exists(true_model_path):
        true_model = load_ground_truth_model(true_model_path, model_class)

    layers = [model.fc1, model.fc2, model.fc3, model.fc4]

    for layer_id, (layer, (num_neurons, input_dim)) in enumerate(zip(layers, layer_config.values()), start=0):
        print(f"\n--- Layer {layer_id} ({num_neurons} neurons, {input_dim} inputs) ---")

        unsigned_weights, recovered_mask, metadata = load_unsigned_weights(
            signature_path, layer_id, num_neurons, input_dim, use_random_init=True
        )

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

        # Copy biases from true model (not recovered in this attack)
        if true_model is not None:
            true_layer = [true_model.fc1, true_model.fc2, true_model.fc3, true_model.fc4][layer_id]
            with torch.no_grad():
                layer.bias.data = true_layer.bias.data.clone()

    # Copy output layer from true model
    if true_model is not None:
        with torch.no_grad():
            model.fc5.weight.data = true_model.fc5.weight.data.clone()
            model.fc5.bias.data = true_model.fc5.bias.data.clone()
        print(f"\n--- Output Layer ---")
        print(f"  Copied from true model")

    # Print recovery summary
    print(f"\n--- Recovery Summary ---")
    total = recovery_stats['total_neurons']
    recovered = recovery_stats['recovered_neurons']
    print(f"  Total neurons: {total}")
    print(f"  Recovered: {recovered} ({100*recovered/total:.1f}%)")
    print(f"  Random init: {total - recovered} ({100*(total-recovered)/total:.1f}%)")

    return model, metrics, recovery_stats


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
    parser.add_argument('--signature-path', type=str, default=SIGNATURE_WEIGHTS_PATH)
    parser.add_argument('--sign-path', type=str, default=SIGN_RECOVERY_PATH)
    parser.add_argument('--output-path', type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    print("="*70)
    print("MODEL EXTRACTION VERIFICATION (v2 - Three-Tier Metrics)")
    print("="*70)

    # Determine model configuration
    if args.tinier:
        model_class = TinierModel
        true_model_path = TINIER_MODEL_PTH
        tiny = False
        makeblobs = False
        tinier = True
        # Non-uniform layer config: (num_neurons, input_dim)
        layer_config = {0: (16, 32), 1: (16, 16), 2: (16, 16), 3: (8, 16)}
    elif args.full:
        model_class = FullModel
        true_model_path = FULL_MODEL_PTH
        tiny = False
        makeblobs = False
        tinier = False
        layer_config = {0: (256, 3072), 1: (256, 256), 2: (256, 256), 3: (64, 256)}
    elif args.makeblobs:
        model_class = TinyModel
        true_model_path = MAKEBLOBS_MODEL_PTH
        tiny = True
        makeblobs = True
        tinier = False
        layer_config = {0: (64, 64), 1: (64, 64), 2: (64, 64), 3: (64, 64)}
    else:
        model_class = TinyModel
        true_model_path = TINY_MODEL_PTH
        makeblobs = False
        tiny = True
        tinier = False
        layer_config = {0: (64, 64), 1: (64, 64), 2: (64, 64), 3: (64, 64)}

    if tinier:
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
    X_test, Y_test = load_test_data(tiny=tiny, makeblobs=makeblobs, tinier=tinier)
    if X_test is None:
        print("Failed to load test data")
        return

    print(f"Test data shape: {X_test.shape}")

    # Load and test ground truth model
    print("\n" + "="*70)
    print("GROUND TRUTH MODEL")
    print("="*70)
    true_model = load_ground_truth_model(true_model_path, model_class)
    true_accuracy = test_model_accuracy(true_model, X_test, Y_test, "Ground Truth")

    # Reconstruct model
    print("\n" + "="*70)
    print("RECONSTRUCTING MODEL FROM EXTRACTION")
    print("="*70)
    print("\nNOTE: Unrecovered neurons use Kaiming/He initialization")
    print("      Scaling uses abs(factor) to ensure sign is NOT revealed\n")
    reconstructed_model, layer_metrics, recovery_stats = reconstruct_model(
        args.signature_path,
        args.sign_path,
        model_class,
        layer_config,
        true_model_path,
        random_seed=42
    )

    # Test reconstructed model
    print("\n" + "="*70)
    print("RECONSTRUCTED MODEL EVALUATION")
    print("="*70)
    recon_accuracy = test_model_accuracy(reconstructed_model, X_test, Y_test, "Reconstructed")

    # Compare predictions
    print("\n--- Prediction Comparison ---")
    with torch.no_grad():
        true_preds = true_model(X_test).argmax(dim=1)
        recon_preds = reconstructed_model(X_test).argmax(dim=1)
        pred_agreement = (true_preds == recon_preds).float().mean().item()
    print(f"Prediction agreement: {pred_agreement:.4f}")

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
    if tinier:
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
        'prediction_agreement': float(pred_agreement),
        'extraction_success': extraction_success,
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
