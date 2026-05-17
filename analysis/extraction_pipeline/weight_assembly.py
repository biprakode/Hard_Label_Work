"""
Build a model from signature recovery + sign recovery outputs.

Pipeline
--------
    load_unsigned_weights    -> per-neuron magnitudes/directions (sign-blind)
    load_signs               -> +1/-1 per neuron from sign recovery
    combine_weights_and_signs -> signed weight rows
    reconstruct_model        -> a fresh model_class with all 4 hidden layers wired
                                 + biases / fc5 either copied (cheat) or zero/Kaiming
    save_reconstructed_model -> persist .pth + .npz
"""

import os
import json
from pathlib import Path

import numpy as np
import torch

from .config import _act, OUTPUT_PATH  # noqa: F401  (OUTPUT_PATH re-exported)
from .data_loading import load_ground_truth_model
from .metrics import compute_weight_metrics_v2


def _kaiming_init(num_neurons, input_dim, nonlinearity='relu'):
    """Kaiming/He initialization for ReLU networks (also fine for leaky)."""
    gain = np.sqrt(2.0) if nonlinearity == 'relu' else 1.0
    std = gain / np.sqrt(input_dim)
    return np.random.randn(num_neurons, input_dim).astype(np.float64) * std


def load_unsigned_weights(signature_path, layer_id, num_neurons, input_dim,
                          use_random_init=True, layer_offset=0):
    """
    Load unsigned weight rows from `signature_path/layer_<L>/neuron_<i>/`.

    Uses weights_unscaled.npz + abs(scaling_factor) so the scaling does not
    leak sign information (only magnitude).

    Neurons without metadata.json (recover_weights "Failed to identify") are
    skipped — their direction is unreliable; they fall back to Kaiming init.

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
            neuron_id = raw_id - layer_offset if raw_id >= num_neurons else raw_id

            if neuron_id < 0 or neuron_id >= num_neurons:
                continue

            metadata_path = neuron_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                scaling_factor = meta.get('scaling_factor', 1.0)
                metadata_dict[neuron_id] = meta
            else:
                continue

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


def load_signs(sign_path, layer_id):
    """Load recovered signs (+1/-1, 0=unknown) from sign-recovery output dir."""
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
    """
    Multiply each neuron's unsigned weight row by its sign (+1/-1).

    Sign value 0 means "unknown" (sign recovery did not process this neuron).
    Treat 0 as +1 so the recovered direction is preserved; oracle sign search
    will flip later if -1 is correct. Multiplying by 0 would zero out a
    perfectly valid recovered direction.
    """
    if unsigned_weights is None or signs is None:
        return None

    num_neurons = unsigned_weights.shape[0]
    if len(signs) < num_neurons:
        signs = np.concatenate([signs, np.ones(num_neurons - len(signs), dtype=np.int8)])

    sign_eff = np.where(signs[:num_neurons] == 0, np.int8(1), signs[:num_neurons])
    return unsigned_weights * sign_eff[:, np.newaxis]


def reconstruct_model(signature_path, sign_path, model_class, layer_config,
                      true_model_path=None, random_seed=42,
                      copy_true_biases=True, copy_true_output=True):
    """
    Reconstruct a model by combining signature + sign recovery outputs.

    Unrecovered neurons get Kaiming/He random init.
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

    layer_sizes = [v[0] for v in layer_config.values()]
    layer_offsets = [sum(layer_sizes[:i]) for i in range(len(layer_sizes))]

    for layer_id, (layer, (num_neurons, input_dim)) in enumerate(zip(layers, layer_config.values()), start=0):
        print(f"\n--- Layer {layer_id} ({num_neurons} neurons, {input_dim} inputs) ---")

        unsigned_weights, recovered_mask, _meta = load_unsigned_weights(
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
        recovery_stats['total_neurons']     += num_neurons
        recovery_stats['recovered_neurons'] += recovered_count
        recovery_stats['random_init_neurons'] += (num_neurons - recovered_count)

        # Sign recovery uses 1-indexed layers
        signs = load_signs(sign_path, layer_id + 1)
        if signs is not None:
            print(f"  Loaded signs: {len(signs)} neurons, "
                  f"{np.sum(signs == 1)} positive, "
                  f"{np.sum(signs == -1)} negative, "
                  f"{np.sum(signs == 0)} unknown")
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

                recovered_signed = signed_weights[recovered_mask]
                recovered_true = true_weights[recovered_mask]

                if len(recovered_signed) > 0:
                    layer_metrics = compute_weight_metrics_v2(recovered_signed, recovered_true)
                    if layer_metrics:
                        layer_metrics['num_recovered'] = recovered_count
                        layer_metrics['num_random_init'] = num_neurons - recovered_count
                        per_neuron_data = layer_metrics.pop('per_neuron', [])
                        metrics[f'layer_{layer_id}'] = layer_metrics
                        metrics[f'layer_{layer_id}_per_neuron'] = per_neuron_data
                        print(f"  [Recovered neurons only - three-tier metrics]")
                        print(f"    SIGN accuracy:      {layer_metrics['sign_accuracy']:.4f}")
                        print(f"    MAGNITUDE rel err:  {layer_metrics['magnitude_mean_rel_error']:.4f} (median: {layer_metrics['magnitude_median_rel_error']:.4f})")
                        print(f"    COMBINED rel err:   {layer_metrics['combined_mean_rel_error']:.4f} (median: {layer_metrics['combined_median_rel_error']:.4f})")
                        print(f"    Mean |cos sim|:     {layer_metrics['mean_abs_cosine_sim']:.4f}")

                all_metrics = compute_weight_metrics_v2(signed_weights, true_weights)
                if all_metrics:
                    all_metrics.pop('per_neuron', None)
                    metrics[f'layer_{layer_id}_all'] = all_metrics
                    print(f"  [All neurons (incl. random init)]")
                    print(f"    SIGN accuracy:      {all_metrics['sign_accuracy']:.4f}")
                    print(f"    COMBINED rel err:   {all_metrics['combined_mean_rel_error']:.4f}")
        else:
            print(f"  Using full random initialization for weights")

        # Biases: cheat-copy or zero
        if copy_true_biases and true_model is not None:
            true_layer = [true_model.fc1, true_model.fc2, true_model.fc3, true_model.fc4][layer_id]
            with torch.no_grad():
                layer.bias.data = true_layer.bias.data.clone()
        else:
            with torch.no_grad():
                layer.bias.data.zero_()

    # Output layer fc5: cheat-copy or Kaiming (for later LR fit)
    if copy_true_output and true_model is not None:
        with torch.no_grad():
            model.fc5.weight.data = true_model.fc5.weight.data.clone()
            model.fc5.bias.data = true_model.fc5.bias.data.clone()
        print(f"\n--- Output Layer ---")
        print(f"  Copied from true model (cheat)")
    else:
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

    print(f"\n--- Recovery Summary ---")
    total = recovery_stats['total_neurons']
    recovered = recovery_stats['recovered_neurons']
    print(f"  Total neurons: {total}")
    print(f"  Recovered: {recovered} ({100*recovered/total:.1f}%)")
    print(f"  Random init: {total - recovered} ({100*(total-recovered)/total:.1f}%)")

    return model, metrics, recovery_stats, recovered_masks_by_layer


def save_reconstructed_model(model, output_path, name="reconstructed_model"):
    """Save the reconstructed model in .pth and .npz formats."""
    Path(output_path).mkdir(parents=True, exist_ok=True)

    pth_path = os.path.join(output_path, f"{name}.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"Saved PyTorch model to: {pth_path}")

    weights_path = os.path.join(output_path, f"{name}_weights.npz")
    weights_dict = {n: p.detach().numpy() for n, p in model.named_parameters()}
    np.savez(weights_path, **weights_dict)
    print(f"Saved weights to: {weights_path}")
