"""
Sign Obfuscation Module for Sign Recovery Phase

This module obfuscates the signs of weight vectors from signature recovery
to ensure that approximately 50% of neurons require sign flips during the
sign recovery phase, making the attack more realistic and challenging.

Background:
-----------
In the DNN extraction attack:
1. Signature Recovery extracts weight vectors WITHOUT signs (magnitude + direction only)
2. Sign Recovery must determine if each neuron's sign is +1 or -1

Problem:
--------
In practice, especially for first and last layers, very few neurons naturally
have incorrect signs after signature recovery, making sign recovery trivial.

Solution:
---------
Before sign recovery, we strategically obfuscate signs to ensure ~50% need flips.
This simulates a more challenging attack scenario and tests the robustness of
the sign recovery algorithm.
"""

import numpy as np
import os
from pathlib import Path


def obfuscate_layer_signs(weights, biases, target_flip_ratio=0.5, strategy='random', seed=None):
    """
    Obfuscate signs of weight vectors to ensure target_flip_ratio of neurons need sign flips.

    Args:
        weights (np.ndarray): Weight matrix of shape [in_features, out_features]
        biases (np.ndarray): Bias vector of shape [out_features]
        target_flip_ratio (float): Target ratio of neurons with flipped signs (default: 0.5)
        strategy (str): Obfuscation strategy - 'random', 'structured', 'adversarial'
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (obfuscated_weights, obfuscated_biases, flip_mask)
            - obfuscated_weights: Weight matrix with obfuscated signs
            - obfuscated_biases: Bias vector with obfuscated signs
            - flip_mask: Boolean array indicating which neurons were flipped
    """
    if seed is not None:
        np.random.seed(seed)

    weights = weights.copy()
    biases = biases.copy()
    n_neurons = weights.shape[-1]

    # Determine which neurons to flip
    if strategy == 'random':
        # Randomly select neurons to flip
        flip_mask = np.random.rand(n_neurons) < target_flip_ratio

    elif strategy == 'structured':
        # Flip alternate neurons (creates structured pattern)
        flip_mask = np.zeros(n_neurons, dtype=bool)
        flip_mask[::2] = True  # Flip every other neuron
        # Adjust to match target ratio
        n_flips_target = int(n_neurons * target_flip_ratio)
        n_flips_current = np.sum(flip_mask)
        if n_flips_current < n_flips_target:
            # Need to flip more
            unflipped_indices = np.where(~flip_mask)[0]
            additional_flips = np.random.choice(unflipped_indices,
                                               n_flips_target - n_flips_current,
                                               replace=False)
            flip_mask[additional_flips] = True
        elif n_flips_current > n_flips_target:
            # Need to flip fewer
            flipped_indices = np.where(flip_mask)[0]
            remove_flips = np.random.choice(flipped_indices,
                                           n_flips_current - n_flips_target,
                                           replace=False)
            flip_mask[remove_flips] = False

    elif strategy == 'adversarial':
        # Flip neurons with largest magnitude (hardest to recover)
        neuron_magnitudes = np.linalg.norm(weights, axis=0)
        n_flips = int(n_neurons * target_flip_ratio)
        largest_indices = np.argsort(neuron_magnitudes)[-n_flips:]
        flip_mask = np.zeros(n_neurons, dtype=bool)
        flip_mask[largest_indices] = True

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Apply sign flips
    for neuron_id in range(n_neurons):
        if flip_mask[neuron_id]:
            weights[:, neuron_id] *= -1
            biases[neuron_id] *= -1

    return weights, biases, flip_mask


def obfuscate_model_signs(all_weights, all_biases, target_flip_ratio=0.5,
                          layer_specific_ratios=None, strategy='random', seed=None):
    """
    Obfuscate signs for all layers in a model.

    Args:
        all_weights (list): List of weight matrices for each layer
        all_biases (list): List of bias vectors for each layer
        target_flip_ratio (float): Default target flip ratio for all layers
        layer_specific_ratios (dict): Optional dict mapping layer_id -> flip_ratio
        strategy (str): Obfuscation strategy
        seed (int): Random seed

    Returns:
        tuple: (obfuscated_weights, obfuscated_biases, flip_masks_dict)
    """
    if seed is not None:
        np.random.seed(seed)

    obfuscated_weights = []
    obfuscated_biases = []
    flip_masks_dict = {}

    for layer_id, (weights, biases) in enumerate(zip(all_weights, all_biases)):
        # Use layer-specific ratio if provided
        if layer_specific_ratios and layer_id in layer_specific_ratios:
            flip_ratio = layer_specific_ratios[layer_id]
        else:
            flip_ratio = target_flip_ratio

        # Obfuscate this layer
        obs_w, obs_b, flip_mask = obfuscate_layer_signs(
            weights, biases, flip_ratio, strategy,
            seed=(seed + layer_id) if seed else None
        )

        obfuscated_weights.append(obs_w)
        obfuscated_biases.append(obs_b)
        flip_masks_dict[layer_id] = flip_mask

        print(f"  Layer {layer_id}: Flipped {np.sum(flip_mask)}/{len(flip_mask)} "
              f"neurons ({np.mean(flip_mask):.1%})")

    return obfuscated_weights, obfuscated_biases, flip_masks_dict


def save_obfuscation_info(output_dir, flip_masks_dict, strategy, target_ratio):
    """
    Save obfuscation information for later analysis.

    Args:
        output_dir (Path): Directory to save info
        flip_masks_dict (dict): Dictionary mapping layer_id -> flip_mask
        strategy (str): Obfuscation strategy used
        target_ratio (float): Target flip ratio
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save flip masks as numpy arrays
    for layer_id, flip_mask in flip_masks_dict.items():
        mask_path = output_dir / f"layer{layer_id}_flip_mask.npy"
        np.save(mask_path, flip_mask)

    # Save metadata
    metadata = {
        'strategy': strategy,
        'target_ratio': target_ratio,
        'layers': {int(lid): {
            'n_neurons': int(len(mask)),
            'n_flipped': int(np.sum(mask)),
            'flip_ratio': float(np.mean(mask))
        } for lid, mask in flip_masks_dict.items()}
    }

    import json
    metadata_path = output_dir / "obfuscation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved obfuscation info to: {output_dir}")


def compare_with_ground_truth(obfuscated_weights, ground_truth_weights, flip_masks_dict):
    """
    Compare obfuscated weights with ground truth to verify obfuscation worked correctly.

    Args:
        obfuscated_weights (list): Obfuscated weight matrices
        ground_truth_weights (list): Ground truth weight matrices
        flip_masks_dict (dict): Dictionary of flip masks

    Returns:
        dict: Comparison statistics per layer
    """
    stats = {}

    for layer_id, (obs_w, gt_w, flip_mask) in enumerate(
        zip(obfuscated_weights, ground_truth_weights,
            [flip_masks_dict[i] for i in range(len(obfuscated_weights))])
    ):
        # Check if signs match for each neuron
        sign_matches = np.zeros(obs_w.shape[1], dtype=bool)

        for neuron_id in range(obs_w.shape[1]):
            # Compare signs (check if they point in same direction)
            dot_product = np.dot(obs_w[:, neuron_id], gt_w[:, neuron_id])
            sign_matches[neuron_id] = dot_product > 0

        # Verify flip mask is correct
        expected_mismatches = flip_mask
        actual_mismatches = ~sign_matches

        stats[layer_id] = {
            'total_neurons': len(sign_matches),
            'correct_signs': int(np.sum(sign_matches)),
            'incorrect_signs': int(np.sum(~sign_matches)),
            'flip_ratio': float(np.mean(~sign_matches)),
            'obfuscation_accuracy': float(np.mean(expected_mismatches == actual_mismatches))
        }

        print(f"  Layer {layer_id}: {stats[layer_id]['incorrect_signs']}/{stats[layer_id]['total_neurons']} "
              f"signs incorrect ({stats[layer_id]['flip_ratio']:.1%})")

    return stats


# ============================================================================
# Example usage and testing
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SIGN OBFUSCATION MODULE TEST")
    print("=" * 70)

    # Create dummy weights for testing
    print("\n[Test 1] Random obfuscation")
    np.random.seed(42)
    weights = np.random.randn(64, 64)
    biases = np.random.randn(64)

    obs_w, obs_b, flip_mask = obfuscate_layer_signs(weights, biases,
                                                     target_flip_ratio=0.5,
                                                     strategy='random',
                                                     seed=42)

    print(f"  Flipped {np.sum(flip_mask)}/{len(flip_mask)} neurons "
          f"({np.mean(flip_mask):.1%})")

    # Verify signs were actually flipped
    for i in range(min(5, len(flip_mask))):
        if flip_mask[i]:
            assert np.allclose(obs_w[:, i], -weights[:, i]), "Sign flip verification failed!"
    print("  ✓ Sign flip verification passed")

    print("\n[Test 2] Structured obfuscation")
    obs_w2, obs_b2, flip_mask2 = obfuscate_layer_signs(weights, biases,
                                                        target_flip_ratio=0.5,
                                                        strategy='structured',
                                                        seed=42)
    print(f"  Flipped {np.sum(flip_mask2)}/{len(flip_mask2)} neurons "
          f"({np.mean(flip_mask2):.1%})")

    print("\n[Test 3] Adversarial obfuscation")
    obs_w3, obs_b3, flip_mask3 = obfuscate_layer_signs(weights, biases,
                                                        target_flip_ratio=0.5,
                                                        strategy='adversarial',
                                                        seed=42)
    print(f"  Flipped {np.sum(flip_mask3)}/{len(flip_mask3)} neurons "
          f"({np.mean(flip_mask3):.1%})")

    # Verify adversarial flips largest neurons
    magnitudes = np.linalg.norm(weights, axis=0)
    top_32 = set(np.argsort(magnitudes)[-32:])
    flipped_neurons = set(np.where(flip_mask3)[0])
    overlap = len(top_32 & flipped_neurons)
    print(f"  Adversarial overlap with top 32 largest: {overlap}/32 ({overlap/32:.1%})")

    print("\n[Test 4] Multi-layer obfuscation")
    all_weights = [np.random.randn(64, 64) for _ in range(4)]
    all_biases = [np.random.randn(64) for _ in range(4)]

    obs_weights, obs_biases, flip_masks = obfuscate_model_signs(
        all_weights, all_biases,
        target_flip_ratio=0.5,
        strategy='random',
        seed=42
    )

    print("\n✓ All tests passed!")
