"""
Oracle-queries-only sign search.

Two strategies:
    * brute force (k <= 18): enumerate all 2^k flip combos for the layer
    * greedy   (k  > 18):    flip-one-keep-if-better, multiple passes

Both use *only* hard-label oracle queries on X_test, so this remains a
black-box attack. When duals_dir is provided, biases are kept consistent
with weight signs via b_i = -w_i · h_{L-1}(x_d).median(), so flipping w_i
also flips b_i (joint w+b search).

Fix C additions (all default-off):
    * `greedy_oracle_sign_search_with_restarts` — N+1 traversals (current +
      N random sign vectors); pick by held-out X_eval agreement. Escapes the
      single-flip local optimum greedy gets stuck in.
    * `pair_flip_lookahead` — after greedy converges on a layer, try all
      C(K,2) pair-flips on the top-K most-uncertain neurons. Catches the
      "two-wrong-signs-cancel" case greedy can't see.
"""

import copy
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


# --------------------------------------------------- Fix C internals (helpers) --

def _reproject_bias_for_neuron(reconstructed_model, layer, neuron_idx, lid, duals_dir):
    """When duals_dir is provided, re-project the bias from the (new) weight row.
    No-op if the dual file is missing or empty. Caller must hold no_grad."""
    if duals_dir is None:
        return
    dpath = os.path.join(duals_dir, f"layer{lid+1}_neuron{neuron_idx}.npy")
    if not os.path.exists(dpath):
        return
    duals = np.load(dpath)
    if len(duals) == 0:
        return
    x_d = torch.tensor(duals[:30], dtype=torch.float64)
    h = _hidden_activations_up_to(reconstructed_model, x_d, lid)
    layer.bias.data[neuron_idx] = -(h @ layer.weight.data[neuron_idx]).median()


def _randomize_signs(reconstructed_model, layers, recovered_masks, duals_dir, rng):
    """Independently flip each recovered row with probability 0.5 (in-place).
    Bias is re-projected when duals_dir is provided so w+b stays consistent."""
    with torch.no_grad():
        for lid, layer in enumerate(layers):
            mask = recovered_masks.get(lid)
            if mask is None:
                continue
            recovered_idx = np.where(mask)[0]
            for nidx_t in recovered_idx:
                nidx = int(nidx_t)
                if rng.random() < 0.5:
                    layer.weight.data[nidx] = -layer.weight.data[nidx]
                    _reproject_bias_for_neuron(reconstructed_model, layer, nidx, lid, duals_dir)


def _agreement_on(model, X, oracle_labels):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == oracle_labels).float().mean().item()


# ---------------------------------------------------- Fix C2: random restarts --

def greedy_oracle_sign_search_with_restarts(
    reconstructed_model, oracle_model, X_train, recovered_masks,
    X_eval=None, layer_ids=(0, 1, 2, 3), n_passes=5,
    n_restarts=4, eval_sample=1024, verbose=True, duals_dir=None, seed=0,
):
    """Run greedy_oracle_sign_search `n_restarts + 1` times:
        * traversal 0  starts from current sign vector
        * traversals 1..N start from random sign vectors (each recovered row
          flipped i.i.d. ±1 with p=0.5)
    Pick the traversal whose held-out X_eval agreement is highest; restore
    that model state.

    X_eval is OPTIONAL — if None, fall back to scoring by X_train agreement
    (same as plain greedy_oracle_sign_search, but with restarts). The plan
    recommends X_eval = X_test3[:eval_sample] for honest restart selection.
    """
    layers = [reconstructed_model.fc1, reconstructed_model.fc2,
              reconstructed_model.fc3, reconstructed_model.fc4]

    oracle_model.eval()
    with torch.no_grad():
        oracle_labels_train = oracle_model(X_train).argmax(dim=1)
        if X_eval is not None:
            X_eval_use = X_eval[:eval_sample]
            oracle_labels_eval = oracle_model(X_eval_use).argmax(dim=1)
        else:
            X_eval_use, oracle_labels_eval = X_train, oracle_labels_train

    rng = np.random.RandomState(seed)
    start_state = copy.deepcopy(reconstructed_model.state_dict())
    best_state = start_state
    best_score = _agreement_on(reconstructed_model, X_eval_use, oracle_labels_eval)
    best_traversal_train = _agreement_on(reconstructed_model, X_train, oracle_labels_train)
    if verbose:
        tag = "X_eval" if X_eval is not None else "X_train"
        print(f"  [restarts] baseline state held-out {tag} agreement={best_score:.4f}, "
              f"X_train agreement={best_traversal_train:.4f}")

    per_restart = []
    for r in range(n_restarts + 1):
        # Reset to baseline state for traversals 0 and 1..N alike.
        reconstructed_model.load_state_dict(copy.deepcopy(start_state))
        if r > 0:
            _randomize_signs(reconstructed_model, layers, recovered_masks, duals_dir, rng)
            init_train = _agreement_on(reconstructed_model, X_train, oracle_labels_train)
            if verbose:
                print(f"  [restarts] restart {r}/{n_restarts}: random init "
                      f"agreement={init_train:.4f} → running greedy ...")
        else:
            if verbose:
                print(f"  [restarts] traversal {r}/{n_restarts}: from current state → running greedy ...")
        traversal_result = greedy_oracle_sign_search(
            reconstructed_model, oracle_model, X_train, recovered_masks,
            layer_ids=layer_ids, n_passes=n_passes, verbose=False,
            duals_dir=duals_dir,
        )
        train_agree = traversal_result['final_agreement']
        eval_agree = _agreement_on(reconstructed_model, X_eval_use, oracle_labels_eval)
        per_restart.append({
            'restart': r,
            'train_agreement': float(train_agree),
            'eval_agreement': float(eval_agree),
        })
        if verbose:
            print(f"  [restarts] traversal {r} train={train_agree:.4f} eval={eval_agree:.4f}")
        if eval_agree > best_score + 1e-6:
            best_score = eval_agree
            best_state = copy.deepcopy(reconstructed_model.state_dict())
            best_traversal_train = train_agree

    reconstructed_model.load_state_dict(best_state)
    final_train = _agreement_on(reconstructed_model, X_train, oracle_labels_train)
    final_eval = _agreement_on(reconstructed_model, X_eval_use, oracle_labels_eval)
    if verbose:
        print(f"  [restarts] selected restart with eval_agreement={best_score:.4f} "
              f"(after restore: train={final_train:.4f} eval={final_eval:.4f})")
    return {
        'n_restarts': int(n_restarts),
        'used_X_eval': bool(X_eval is not None),
        'per_restart': per_restart,
        'final_agreement': float(final_train),
        'final_eval_agreement': float(final_eval),
    }


# -------------------------------------------- Fix C3: pair-flip lookahead -----

def pair_flip_lookahead(
    reconstructed_model, oracle_model, X_train, recovered_masks,
    K=8, layer_ids=(0, 1, 2, 3), verbose=True, duals_dir=None,
):
    """For each layer:
        1. Re-score each recovered neuron's |flip_agree - baseline_agree|.
        2. Pick the K with smallest absolute change (most uncertain).
        3. For every pair (i, j) in those K, simultaneously flip both. Keep
           the joint flip iff agreement strictly improves over the current
           best (greedy local-optimum escape via pair coupling).

    This is a single pass per layer — small enough (C(K,2) forward passes per
    layer; K=8 → 28 passes/layer) that it's cheap enough to interleave with
    sign-search cycles.
    """
    layers = [reconstructed_model.fc1, reconstructed_model.fc2,
              reconstructed_model.fc3, reconstructed_model.fc4]

    oracle_model.eval()
    with torch.no_grad():
        oracle_labels = oracle_model(X_train).argmax(dim=1)

    results = {}
    for lid in layer_ids:
        layer = layers[lid]
        mask = recovered_masks.get(lid)
        if mask is None:
            results[lid] = {'skipped': 'no_mask'}
            continue
        recovered_idx = np.where(mask)[0]
        if len(recovered_idx) < 2:
            results[lid] = {'skipped': 'fewer_than_2_recovered', 'recovered': int(len(recovered_idx))}
            continue

        # 1. Score uncertainty per neuron.
        with torch.no_grad():
            baseline_agree = _agreement_on(reconstructed_model, X_train, oracle_labels)
            deltas = []
            for nidx_t in recovered_idx:
                nidx = int(nidx_t)
                orig_w = layer.weight.data[nidx].clone()
                orig_b = layer.bias.data[nidx].clone()
                layer.weight.data[nidx] = -orig_w
                if duals_dir is not None:
                    _reproject_bias_for_neuron(reconstructed_model, layer, nidx, lid, duals_dir)
                flip_agree = _agreement_on(reconstructed_model, X_train, oracle_labels)
                deltas.append((nidx, abs(flip_agree - baseline_agree)))
                layer.weight.data[nidx] = orig_w
                layer.bias.data[nidx] = orig_b

        # 2. Top-K most-uncertain (smallest |Δ|).
        deltas.sort(key=lambda x: x[1])
        top_k = [d[0] for d in deltas[:max(2, K)]]

        # 3. Pair-flip pass.
        n_pair_flips = 0
        cur_agree = baseline_agree
        with torch.no_grad():
            for i in range(len(top_k)):
                for j in range(i + 1, len(top_k)):
                    ni, nj = top_k[i], top_k[j]
                    orig_wi = layer.weight.data[ni].clone()
                    orig_bi = layer.bias.data[ni].clone()
                    orig_wj = layer.weight.data[nj].clone()
                    orig_bj = layer.bias.data[nj].clone()

                    layer.weight.data[ni] = -orig_wi
                    layer.weight.data[nj] = -orig_wj
                    if duals_dir is not None:
                        _reproject_bias_for_neuron(reconstructed_model, layer, ni, lid, duals_dir)
                        _reproject_bias_for_neuron(reconstructed_model, layer, nj, lid, duals_dir)

                    test_agree = _agreement_on(reconstructed_model, X_train, oracle_labels)
                    if test_agree > cur_agree + 1e-7:
                        cur_agree = test_agree
                        n_pair_flips += 1
                    else:
                        layer.weight.data[ni] = orig_wi
                        layer.bias.data[ni] = orig_bi
                        layer.weight.data[nj] = orig_wj
                        layer.bias.data[nj] = orig_bj

        results[lid] = {
            'recovered': int(len(recovered_idx)),
            'K': int(len(top_k)),
            'baseline_agreement': float(baseline_agree),
            'final_agreement': float(cur_agree),
            'pair_flips_accepted': int(n_pair_flips),
        }
        if verbose:
            print(f"  [pair-flip] layer {lid}: K={len(top_k)} most-uncertain, "
                  f"baseline={baseline_agree:.4f} → after pair pass={cur_agree:.4f} "
                  f"({n_pair_flips} pair flips accepted)")

    return results
