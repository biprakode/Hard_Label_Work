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
import math
import os
import numpy as np
import torch


# Sign search optimises oracle agreement. Once agreement == 1.0 the objective is
# flat (every input already matches the oracle), so no flip can improve it and
# further search is wasted compute. This watchdog short-circuits the search the
# moment agreement saturates.
_AGREE_SAT = 1.0 - 1e-12


def _saturated(agree):
    return agree >= _AGREE_SAT

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

    # Watchdog: agreement already saturated → no flip can improve it, skip search.
    if _saturated(start_agree):
        if verbose:
            print(f"  [sign-search] agreement already 1.0 — skipping (saturated)")
        results['final_agreement'] = start_agree
        results['starting_agreement'] = start_agree
        results['saturated'] = True
        return results

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
        if _saturated(cur_agree):
            if verbose:
                print(f"  [sign-search] agreement hit 1.0 — stopping (saturated)")
            results['saturated'] = True
            break
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

    # Watchdog: agreement already saturated → no flip can improve it, skip.
    if _saturated(start_agree):
        if verbose:
            print(f"  [greedy-sign-search] agreement already 1.0 — skipping (saturated)")
        results['final_agreement'] = float(start_agree)
        results['saturated'] = True
        return results

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

        if _saturated(cur_agree):
            if verbose:
                print(f"  [greedy-sign-search] agreement hit 1.0 — stopping (saturated)")
            results['saturated'] = True
            break
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

            # Watchdog: agreement saturated → no flip can improve, stop this layer.
            if _saturated(agree_curr):
                break

            orig_w = layer.weight.data[neuron_idx].clone()
            orig_b = layer.bias.data[neuron_idx].clone()

            _flip_neuron(reconstructed_model, layer, neuron_idx, lid, duals_dir)

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


def _flip_neuron(reconstructed_model, layer, neuron_idx, lid, duals_dir):
    """The single shared 'flip move' used by every optimizer (greedy/tabu/SA/PT).

    Negates a recovered neuron's weight row (an involution) and, when duals_dir is
    provided, re-projects its bias from the NEW row via b = -median(h_{L-1} . w),
    so the joint w+b update stays identical across all search strategies.
    Caller must hold no_grad."""
    layer.weight.data[neuron_idx] = -layer.weight.data[neuron_idx]
    if duals_dir is not None:
        _reproject_bias_for_neuron(reconstructed_model, layer, neuron_idx, lid, duals_dir)


def _score_objective(reconstructed_model, X, oracle_labels, mode='agree'):
    """Return (select_score, opt_score).

    select_score : true 0/1 argmax agreement vs the cached oracle labels. ALWAYS
                   used to accept/select the final configuration (honest, hard-label).
    opt_score    : the signal the optimizer climbs.
        mode='agree'  -> opt_score == select_score (raw 0/1 agreement).
        mode='margin' -> mean( logit[oracle] - max_{j!=oracle} logit[j] ) on the
                         reconstructed model's OWN logits (still hard-label w.r.t.
                         the victim, which only ever provided argmax labels). Gives
                         gradient-like signal through the flat 0/1-agreement regions.
    Caller must hold no_grad."""
    logits = reconstructed_model(X)
    preds = logits.argmax(dim=1)
    agree = (preds == oracle_labels).float().mean().item()
    if mode == 'agree':
        return agree, agree
    idx = oracle_labels.view(-1, 1)
    true_logit = logits.gather(1, idx).squeeze(1)
    masked = logits.scatter(1, idx, float('-inf'))     # new tensor; leaves logits intact
    runner_up = masked.max(dim=1).values
    margin = (true_logit - runner_up).mean().item()
    return agree, margin


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

        # Watchdog: traversal 0 already at agreement 1.0 → it is objective-optimal;
        # random restarts cannot beat it and only risk held-out overfitting. Stop.
        if r == 0 and _saturated(train_agree):
            if verbose:
                print(f"  [restarts] traversal 0 at agreement 1.0 — skipping random restarts (saturated)")
            break

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


# =====================================================================
# sign_search_improve — Track A: metaheuristics on the true objective
# (tabu search, simulated annealing). Both reuse the shared flip move
# (`_flip_neuron`) and objective (`_score_objective`) so the move,
# joint-w+b bias update, and safety-revert are identical to greedy.
# They differ only in the ACCEPTANCE RULE / MEMORY, which is exactly what
# lets them escape the single-flip local optimum greedy stalls in.
# =====================================================================

def _trial_score(reconstructed_model, layer, nidx, lid, duals_dir,
                 X, oracle_labels, objective):
    """Flip neuron nidx, score (select, opt), then exactly restore it.
    Save/restore is explicit (not the involution) so it is bit-exact even
    in the duals_dir joint-w+b regime. Caller holds no_grad."""
    ow = layer.weight.data[nidx].clone()
    ob = layer.bias.data[nidx].clone()
    _flip_neuron(reconstructed_model, layer, nidx, lid, duals_dir)
    sel, opt = _score_objective(reconstructed_model, X, oracle_labels, objective)
    layer.weight.data[nidx] = ow
    layer.bias.data[nidx] = ob
    return sel, opt


def _net_flips_vs(layer, ref_weight, recovered_idx):
    """How many recovered rows now point opposite to a reference snapshot."""
    n = 0
    for nidx in recovered_idx:
        if (layer.weight.data[int(nidx)] * ref_weight[int(nidx)]).sum().item() < 0:
            n += 1
    return n


def tabu_sign_pass_layer(reconstructed_model, layers, lid, recovered_masks,
                         X_train, oracle_labels, duals_dir=None,
                         tabu_tenure=5, max_sweeps=4, objective='agree'):
    """One per-layer tabu-search pass.

    Each sweep: evaluate every single-flip, take the BEST move by the
    optimization score even if it worsens agreement, but forbid re-flipping a
    neuron for `tabu_tenure` iterations (aspiration overrides tabu when a move
    sets a new global best true-agreement). Returns net #flips vs pass entry.
    The configuration restored at the end is the best-seen by TRUE agreement,
    and is never worse than the pass-entry baseline (safety-revert)."""
    layer = layers[lid]
    mask = recovered_masks.get(lid)
    if mask is None:
        return 0
    recovered_idx = [int(i) for i in np.where(mask)[0]]
    if not recovered_idx:
        return 0

    with torch.no_grad():
        entry_w = layer.weight.data.clone()
        entry_b = layer.bias.data.clone()
        best_sel, _ = _score_objective(reconstructed_model, X_train, oracle_labels, objective)
        if _saturated(best_sel):          # watchdog: nothing to gain on this layer
            return 0
        best_w, best_b = entry_w.clone(), entry_b.clone()
        tabu_until = {i: -1 for i in recovered_idx}

        it = 0
        for _sweep in range(max_sweeps):
            if _saturated(best_sel):
                break
            best_move, best_move_opt, best_move_sel = None, -float('inf'), None
            for ni in recovered_idx:
                sel, opt = _trial_score(reconstructed_model, layer, ni, lid,
                                        duals_dir, X_train, oracle_labels, objective)
                is_tabu = tabu_until[ni] >= it
                aspires = sel > best_sel + 1e-12
                if is_tabu and not aspires:
                    continue
                if opt > best_move_opt + 1e-15:
                    best_move, best_move_opt, best_move_sel = ni, opt, sel
            if best_move is None:
                break
            _flip_neuron(reconstructed_model, layer, best_move, lid, duals_dir)
            it += 1
            tabu_until[best_move] = it + tabu_tenure
            if best_move_sel > best_sel + 1e-12:
                best_sel = best_move_sel
                best_w = layer.weight.data.clone()
                best_b = layer.bias.data.clone()

        layer.weight.data = best_w
        layer.bias.data = best_b
        return _net_flips_vs(layer, entry_w, recovered_idx)


def sa_sign_pass_layer(reconstructed_model, layers, lid, recovered_masks,
                       X_train, oracle_labels, duals_dir=None,
                       sweeps=4, t0=0.02, tend=1e-4, objective='agree', rng=None):
    """One per-layer simulated-annealing pass.

    Proposes random single flips; accepts with prob min(1, exp(Δopt / T)) on a
    geometric cooling schedule from `t0` to `tend`. At T→0 this reduces to
    greedy; at T>0 it accepts worsening flips, reaching pair/k-flip-coupled
    configurations greedy cannot. Restores the best-seen state by TRUE
    agreement; never worse than the pass-entry baseline."""
    layer = layers[lid]
    mask = recovered_masks.get(lid)
    if mask is None:
        return 0
    recovered_idx = [int(i) for i in np.where(mask)[0]]
    if not recovered_idx:
        return 0
    if rng is None:
        rng = np.random.RandomState(0)

    with torch.no_grad():
        entry_w = layer.weight.data.clone()
        cur_sel, cur_opt = _score_objective(reconstructed_model, X_train, oracle_labels, objective)
        if _saturated(cur_sel):           # watchdog: nothing to gain on this layer
            return 0
        best_sel = cur_sel
        best_w = layer.weight.data.clone()
        best_b = layer.bias.data.clone()

        n_steps = max(1, sweeps * len(recovered_idx))
        for step in range(n_steps):
            frac = step / max(1, n_steps - 1)
            T = t0 * (tend / t0) ** frac if t0 > 0 else 0.0
            ni = recovered_idx[rng.randint(len(recovered_idx))]
            ow = layer.weight.data[ni].clone()
            ob = layer.bias.data[ni].clone()
            _flip_neuron(reconstructed_model, layer, ni, lid, duals_dir)
            sel, opt = _score_objective(reconstructed_model, X_train, oracle_labels, objective)
            d = opt - cur_opt
            accept = d >= 0 or (T > 0 and rng.random() < math.exp(d / T))
            if accept:
                cur_opt, cur_sel = opt, sel
                if sel > best_sel + 1e-12:
                    best_sel = sel
                    best_w = layer.weight.data.clone()
                    best_b = layer.bias.data.clone()
                    if _saturated(best_sel):    # watchdog: reached 1.0, stop
                        break
            else:
                layer.weight.data[ni] = ow
                layer.bias.data[ni] = ob

        layer.weight.data = best_w
        layer.bias.data = best_b
        return _net_flips_vs(layer, entry_w, recovered_idx)


def _metaheuristic_oracle_sign_search(reconstructed_model, oracle_model, X_train,
                                      recovered_masks, pass_fn, layer_ids=(0, 1, 2, 3),
                                      n_passes=5, verbose=True, tag='meta'):
    """Generic multi-pass driver shared by tabu / SA. `pass_fn(model, layers,
    lid, masks, X, oracle_labels)` runs one per-layer optimization pass and
    returns net #flips. Alternates layer order per pass; early-stops on no
    improvement and no flips (same contract as greedy_oracle_sign_search)."""
    reconstructed_model.eval()
    oracle_model.eval()
    with torch.no_grad():
        oracle_labels = oracle_model(X_train).argmax(dim=1)
    layers = [reconstructed_model.fc1, reconstructed_model.fc2,
              reconstructed_model.fc3, reconstructed_model.fc4]

    def _cur():
        return _agreement_on(reconstructed_model, X_train, oracle_labels)

    start = _cur()
    if verbose:
        print(f"  [{tag}-sign-search] starting agreement: {start:.4f}")

    # GLOBAL best-by-true-agreement guard: a metaheuristic accepts temporary
    # worsening to escape local optima, so the final on-model state may be worse
    # than something already visited. We snapshot the best full-model state ever
    # seen (incl. the incoming config) and restore it at the end. This makes the
    # search monotone-safe: it can never return below where it started.
    best_agree = start
    best_state = copy.deepcopy(reconstructed_model.state_dict())

    # Watchdog: agreement already saturated → no flip can improve it, skip search.
    if _saturated(start):
        if verbose:
            print(f"  [{tag}-sign-search] agreement already 1.0 — skipping (saturated)")
        results = {'starting_agreement': float(start), 'passes': [],
                   'final_agreement': float(start), 'saturated': True}
        return results

    prev = -1.0
    saturated = False
    results = {'starting_agreement': float(start), 'passes': []}
    for pass_i in range(n_passes):
        order = list(reversed(layer_ids)) if pass_i % 2 == 0 else list(layer_ids)
        total = 0
        for lid in order:
            total += pass_fn(reconstructed_model, layers, lid, recovered_masks,
                             X_train, oracle_labels)
            a = _cur()
            if a > best_agree + 1e-12:
                best_agree = a
                best_state = copy.deepcopy(reconstructed_model.state_dict())
            if _saturated(best_agree):
                saturated = True
                break
        cur = _cur()
        if verbose:
            print(f"  [{tag}-sign-search] pass {pass_i+1}/{n_passes} "
                  f"agreement: {cur:.4f} (best {best_agree:.4f}, {total} net flips)")
        results['passes'].append({'pass': pass_i + 1, 'agreement': float(cur),
                                   'total_flipped': int(total)})
        if saturated:
            if verbose:
                print(f"  [{tag}-sign-search] agreement hit 1.0 — stopping (saturated)")
            results['saturated'] = True
            break
        if cur <= prev + 1e-6 and total == 0:
            if verbose:
                print(f"  [{tag}-sign-search] converged, stopping early")
            break
        prev = cur

    reconstructed_model.load_state_dict(best_state)
    results['final_agreement'] = float(_cur())
    return results


def tabu_oracle_sign_search(reconstructed_model, oracle_model, X_train, recovered_masks,
                            layer_ids=(0, 1, 2, 3), n_passes=5, tabu_tenure=5,
                            max_sweeps=4, objective='agree', verbose=True, duals_dir=None,
                            warm_start_greedy=True, greedy_passes=5):
    """Tabu-search sign assignment (drop-in alternative to greedy_oracle_sign_search).

    With `warm_start_greedy` (default) a greedy descent runs first, so the result
    is guaranteed >= greedy; tabu then escapes greedy's local optimum from there.
    """
    if warm_start_greedy:
        greedy_oracle_sign_search(reconstructed_model, oracle_model, X_train,
                                  recovered_masks, layer_ids=layer_ids,
                                  n_passes=greedy_passes, verbose=False, duals_dir=duals_dir)

    def _pass(model, layers, lid, masks, X, ol):
        return tabu_sign_pass_layer(model, layers, lid, masks, X, ol,
                                    duals_dir=duals_dir, tabu_tenure=tabu_tenure,
                                    max_sweeps=max_sweeps, objective=objective)
    return _metaheuristic_oracle_sign_search(
        reconstructed_model, oracle_model, X_train, recovered_masks, _pass,
        layer_ids=layer_ids, n_passes=n_passes, verbose=verbose, tag='tabu')


def sa_oracle_sign_search(reconstructed_model, oracle_model, X_train, recovered_masks,
                          layer_ids=(0, 1, 2, 3), n_passes=5, sweeps=4, t0=0.02,
                          tend=1e-4, objective='agree', seed=0, verbose=True, duals_dir=None,
                          warm_start_greedy=True, greedy_passes=5):
    """Simulated-annealing sign assignment (drop-in alternative to greedy).

    With `warm_start_greedy` (default) a greedy descent runs first, so the result
    is guaranteed >= greedy; SA then escapes via temperature-driven worsening moves.
    """
    if warm_start_greedy:
        greedy_oracle_sign_search(reconstructed_model, oracle_model, X_train,
                                  recovered_masks, layer_ids=layer_ids,
                                  n_passes=greedy_passes, verbose=False, duals_dir=duals_dir)

    rng = np.random.RandomState(seed)

    def _pass(model, layers, lid, masks, X, ol):
        return sa_sign_pass_layer(model, layers, lid, masks, X, ol,
                                  duals_dir=duals_dir, sweeps=sweeps, t0=t0,
                                  tend=tend, objective=objective, rng=rng)
    return _metaheuristic_oracle_sign_search(
        reconstructed_model, oracle_model, X_train, recovered_masks, _pass,
        layer_ids=layer_ids, n_passes=n_passes, verbose=verbose, tag='sa')


# ----------------------------------------- Track A: parallel tempering (M4) ----

def _sync_layer_to_mask(reconstructed_model, layer, entry_w, recovered_idx,
                        flip_mask, lid, duals_dir):
    """Set the layer's recovered rows to entry_w with `flip_mask` applied
    (True = negated from entry), re-projecting biases when duals_dir is given.
    Caller holds no_grad."""
    for j, nidx in enumerate(recovered_idx):
        layer.weight.data[nidx] = -entry_w[nidx] if flip_mask[j] else entry_w[nidx].clone()
    if duals_dir is not None:
        for nidx in recovered_idx:
            _reproject_bias_for_neuron(reconstructed_model, layer, nidx, lid, duals_dir)


def pt_sign_pass_layer(reconstructed_model, layers, lid, recovered_masks,
                       X_train, oracle_labels, duals_dir=None,
                       n_replicas=6, sweeps=3, t_min=1e-3, t_max=0.05,
                       objective='agree', rng=None):
    """One per-layer parallel-tempering (replica-exchange) pass.

    Runs `n_replicas` copies of the layer's sign vector on a geometric
    temperature ladder t_min..t_max. Each sweep: every replica does one local
    SA sweep (k single-flip Metropolis proposals at its own temperature), then
    adjacent replicas attempt a swap with prob min(1, exp((1/Ti-1/Tj)(Ei-Ej))),
    E = -opt. Hot replicas cross barriers single-chain SA cannot; swaps inject
    those discoveries into the cold chain. Restores the global best-by-TRUE-
    agreement seen; never worse than the pass-entry baseline."""
    layer = layers[lid]
    mask = recovered_masks.get(lid)
    if mask is None:
        return 0
    recovered_idx = [int(i) for i in np.where(mask)[0]]
    k = len(recovered_idx)
    if k == 0:
        return 0
    if rng is None:
        rng = np.random.RandomState(0)

    with torch.no_grad():
        entry_w = layer.weight.data.clone()
        temps = [t_min * (t_max / t_min) ** (i / max(1, n_replicas - 1))
                 for i in range(n_replicas)]
        # All replicas start from the entry (warm-started) config.
        repl = [np.zeros(k, dtype=bool) for _ in range(n_replicas)]
        cur_opt = [None] * n_replicas

        # Global best by true agreement (entry config is the first candidate).
        _sync_layer_to_mask(reconstructed_model, layer, entry_w, recovered_idx,
                            repl[0], lid, duals_dir)
        best_sel, opt0 = _score_objective(reconstructed_model, X_train, oracle_labels, objective)
        if _saturated(best_sel):          # watchdog: nothing to gain on this layer
            return 0
        best_mask = repl[0].copy()
        for m in range(n_replicas):
            cur_opt[m] = opt0

        for _sweep in range(sweeps):
            if _saturated(best_sel):
                break
            for mi in range(n_replicas):
                T = temps[mi]
                _sync_layer_to_mask(reconstructed_model, layer, entry_w,
                                    recovered_idx, repl[mi], lid, duals_dir)
                cur = cur_opt[mi]
                for _ in range(k):
                    j = rng.randint(k)
                    nidx = recovered_idx[j]
                    _flip_neuron(reconstructed_model, layer, nidx, lid, duals_dir)
                    sel, opt = _score_objective(reconstructed_model, X_train, oracle_labels, objective)
                    d = opt - cur
                    if d >= 0 or (T > 0 and rng.random() < math.exp(d / T)):
                        cur = opt
                        repl[mi][j] = not repl[mi][j]
                        if sel > best_sel + 1e-12:
                            best_sel = sel
                            best_mask = repl[mi].copy()
                    else:
                        _flip_neuron(reconstructed_model, layer, nidx, lid, duals_dir)
                cur_opt[mi] = cur
                if _saturated(best_sel):
                    break
            # adjacent replica-exchange swaps
            for mi in range(n_replicas - 1):
                delta = (1.0 / temps[mi] - 1.0 / temps[mi + 1]) * (cur_opt[mi] - cur_opt[mi + 1])
                # E = -opt; accept if (1/Ti-1/Tj)(Ei-Ej) >= 0 i.e. delta(as defined) >= 0
                if delta >= 0 or rng.random() < math.exp(delta):
                    repl[mi], repl[mi + 1] = repl[mi + 1], repl[mi]
                    cur_opt[mi], cur_opt[mi + 1] = cur_opt[mi + 1], cur_opt[mi]

        _sync_layer_to_mask(reconstructed_model, layer, entry_w, recovered_idx,
                            best_mask, lid, duals_dir)
        return _net_flips_vs(layer, entry_w, recovered_idx)


def pt_oracle_sign_search(reconstructed_model, oracle_model, X_train, recovered_masks,
                          layer_ids=(0, 1, 2, 3), n_passes=3, n_replicas=6, sweeps=3,
                          t_min=1e-3, t_max=0.05, objective='agree', seed=0,
                          verbose=True, duals_dir=None, warm_start_greedy=True,
                          greedy_passes=5):
    """Parallel-tempering sign assignment (strongest Track-A optimizer; reserve
    for the widest layers). Warm-started from greedy so result is >= greedy."""
    if warm_start_greedy:
        greedy_oracle_sign_search(reconstructed_model, oracle_model, X_train,
                                  recovered_masks, layer_ids=layer_ids,
                                  n_passes=greedy_passes, verbose=False, duals_dir=duals_dir)
    rng = np.random.RandomState(seed)

    def _pass(model, layers, lid, masks, X, ol):
        return pt_sign_pass_layer(model, layers, lid, masks, X, ol,
                                  duals_dir=duals_dir, n_replicas=n_replicas,
                                  sweeps=sweeps, t_min=t_min, t_max=t_max,
                                  objective=objective, rng=rng)
    return _metaheuristic_oracle_sign_search(
        reconstructed_model, oracle_model, X_train, recovered_masks, _pass,
        layer_ids=layer_ids, n_passes=n_passes, verbose=verbose, tag='pt')
