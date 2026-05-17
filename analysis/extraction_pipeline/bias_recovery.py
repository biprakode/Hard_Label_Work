"""
Bias recovery from dual points.

Math: a dual point x_d for neuron i in layer L lies exactly on its ReLU
hyperplane, so  w_i · h_{L-1}(x_d) + b_i = 0  =>  b_i = -w_i · h_{L-1}(x_d).
We use the already-reconstructed lower layers to compute h_{L-1} and take the
median over n_duals dual points for robustness against drift in deeper layers.

Called bottom-up so each layer's prefix is stable when its biases are set.
"""

import os
import numpy as np
import torch

from .config import _act


def _hidden_activations_up_to(reconstructed_model, x, up_to_layer):
    """Forward x through fc1..fc{up_to_layer-1} with activation. No activation on boundary."""
    layers = [reconstructed_model.fc1, reconstructed_model.fc2,
              reconstructed_model.fc3, reconstructed_model.fc4]
    h = x
    for l_idx in range(up_to_layer):
        h = _act(layers[l_idx](h))
    return h


def recover_biases_from_duals(reconstructed_model, duals_dir, recovered_masks,
                               layer_ids=(0, 1, 2, 3), n_duals=30, verbose=True):
    """
    For each recovered neuron i in layer L: b_i = median_d(-w_i · h_{L-1}(x_d)).

    Uses the already-reconstructed lower layers for h. Should be called
    bottom-up so lower layers are stable when their biases are computed.
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
