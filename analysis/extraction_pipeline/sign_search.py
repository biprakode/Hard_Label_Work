"""
Oracle-queries-only sign search.

Two strategies:
    * brute force (k <= 18): enumerate all 2^k flip combos for the layer
    * greedy   (k  > 18):    flip-one-keep-if-better, multiple passes

Both use *only* hard-label oracle queries on X_test, so this remains a
black-box attack. When duals_dir is provided, biases are kept consistent
with weight signs via b_i = -w_i · h_{L-1}(x_d).median(), so flipping w_i
also flips b_i (joint w+b search).
"""

import os
import numpy as np
import torch

from .bias_recovery import _hidden_activations_up_to


# ----------------------------------------------------------------- public API --

def oracle_sign_search(reconstructed_model, oracle_model, X_test, recovered_masks,
                        layer_ids=(0, 1, 2, 3), n_passes=3, order='both', verbose=True,
                        duals_dir=None):
    """
    Per-layer brute-force / greedy sign search using hard-label oracle queries.

    Iterates `n_passes` times in alternating direction (top-down / bottom-up)
    until convergence — downstream errors mask upstream signs and vice versa.
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
                      X_test, oracle_labels, results, verbose, duals_dir=duals_dir)

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


def greedy_oracle_sign_search(reconstructed_model, oracle_model, X_train, recovered_masks,
                               layer_ids=(0, 1, 2, 3), n_passes=5, verbose=True,
                               duals_dir=None):
    """
    O(k)-per-pass greedy sign search using hard-label oracle queries.

    For each pass: for each layer (alternating direction), for each recovered
    neuron, flip its sign and keep iff oracle agreement improves. Works for
    any layer width — no 2^k restriction. Called automatically from
    `oracle_sign_search` when k > 18.
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


# ---------------------------------------------------------------- internals --

def _run_one_pass(reconstructed_model, layers, recovered_masks, layer_order,
                  X_test, oracle_labels, results, verbose, duals_dir=None):
    """Single pass of per-layer brute-force sign search. Mutates layers in place.

    If duals_dir is provided, biases are kept consistent with weight signs via
    b_i = -w_i · h_{L-1}(x_d).median() — flipping w_i also flips b_i.
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
        with torch.no_grad():
            preds0 = reconstructed_model(X_test).argmax(dim=1)
            baseline_agree = (preds0 == oracle_labels).float().mean().item()
        best_agree = -1.0
        best_combo = 0

        # Precompute h·w_orig medians for joint w+b flipping
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
                        new_bias[int(neuron_idx)] = -sign * projections[int(neuron_idx)]
                layer.weight.data = new_weight
                layer.bias.data = new_bias
                preds = reconstructed_model(X_test).argmax(dim=1)
                agree = (preds == oracle_labels).float().mean().item()
                if agree > best_agree:
                    best_agree = agree
                    best_combo = combo

        # Safety: revert if no combo beats baseline
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
    """One greedy pass: flip each recovered neuron; keep flip iff oracle agreement improves."""
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
