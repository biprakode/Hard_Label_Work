import re
import os
import sys
import pickle
from utils import *
from collections import defaultdict

from recover_weights import is_consistent, CIFAR10NetPrefix, transfer_weights, VERBOSE_IS_CONSISTENT

def cheat_cluster(layer):
    duals = []
    root = os.path.join(BASE_DIR, 'signature_recovery/exp/1')
    for f in sorted(os.listdir(root)):
        print(f)
        x = pickle.load(open(os.path.join(root,f),"rb"))
        duals.extend(x)

    cheating = defaultdict(list)
    for idx,(left,middle,right) in enumerate(duals):
        if idx%1000 == 0:
            print(idx, '/', len(duals))
        diff = cheat_neuron_diff_cuda(left, right)
        if len(diff) == 1:
            flat_idx = diff[0]
            # Use LAYER_BOUNDARIES to determine which layer this neuron belongs to
            if LAYER_BOUNDARIES[layer] <= flat_idx < LAYER_BOUNDARIES[layer + 1]:
                cheating[flat_idx].append((left, middle, right))
    
    pickle.dump(cheating, open(os.path.join(BASE_DIR, "signature_recovery/exp/1-cluster-%d.p")%layer, "wb"))

def refine_cluster(maybe, layer, prefix):
    maybe = np.array(maybe)
    points = np.zeros(len(maybe))
    for _ in range(10):
        order = np.arange(len(maybe))
        random.shuffle(order)
        for i in range(0, len(order)-(len(order)%3), 3):
            ok = is_consistent([maybe[x] for x in order[i:i+3]], prefix, layer, False)
            if VERBOSE_IS_CONSISTENT:
                print("ok?", ok)
            # is_consistent_help's rejection path returns the tuple (None, None),
            # not a scalar -- mirror cluster_slow's own type(S) == np.float64 guard
            # (line ~68) rather than assuming a bare float back from is_consistent.
            if type(ok) == np.float64 and ok > 1e-5:
                points[order[i:i+3]] += 1
    
    maybe = maybe[points < 6]
    return maybe

# Resource-budget cap on the outer seed loop, NOT a method change (mirrors the
# SIGN_NTHREADS precedent in ablation_tiny/run_ablation.sh: a parallelism/
# resource constant, not a change to what's measured). cluster_slow is O(n) per
# seed x up to 1000 seeds by default -- on this study's CPU dev box that is
# multi-hour per layer per victim. Default 1000 preserves original behavior
# exactly; cheating_ablation's sweep script lowers it via env var and documents
# the wall-clock/coverage tradeoff in reports/neuron_clustering/observations.md.
CLUSTER_SLOW_MAX_SEEDS = int(os.environ.get("CLUSTER_SLOW_MAX_SEEDS", "1000"))
# Same resource-budget rationale as CLUSTER_SLOW_MAX_SEEDS above: the inner
# `for j,b in enumerate(duals)` scan is O(len(duals)) per seed (the original
# 1000-into-the-inner-loop throttle only kicks in after j>1000), and is the
# actual dominant cost (an SVD-based is_consistent call per (a,b) pair), not
# print I/O. Default None preserves the original unbounded scan exactly.
_env_max_inner = os.environ.get("CLUSTER_SLOW_MAX_INNER")
CLUSTER_SLOW_MAX_INNER = int(_env_max_inner) if _env_max_inner else None


def cluster_slow(layer):
    prefix = CIFAR10NetPrefix(layer).cpu()
    transfer_weights(cheat_net_cpu, prefix)

    duals = []
    root = os.path.join(BASE_DIR, 'signature_recovery/exp/1')
    for f in sorted(os.listdir(root)):
        print(f)
        x = pickle.load(open(os.path.join(root,f),"rb"))
        duals.extend(x)

    output = {}
    # generate_dual_neuron.py routes each cluster by get_layer_index(neuron_idx),
    # which requires the dict key to fall inside this layer's LAYER_BOUNDARIES
    # window -- cluster_id (an arbitrary per-layer enumeration index starting at
    # 0) does NOT satisfy that on its own (every layer's file would collide at
    # keys 0,1,2,... and get misrouted to layer 0). Cluster IDENTITY doesn't need
    # to match a true neuron index (the network is isomorphic up to permutation),
    # it just needs a synthetic id inside the right layer's range -- offset by
    # LAYER_BOUNDARIES[layer] and cap at this layer's true width.
    next_synthetic_id = LAYER_BOUNDARIES[layer]
    layer_end = LAYER_BOUNDARIES[layer + 1]
    print("LAYER", layer)
    for cluster_id,a in enumerate(duals[:CLUSTER_SLOW_MAX_SEEDS]):
        print("idx", cluster_id)
        maybe = [a]
        inner_duals = duals if CLUSTER_SLOW_MAX_INNER is None else duals[:CLUSTER_SLOW_MAX_INNER]
        for j,b in enumerate(inner_duals):
            if j > 1000 and len(maybe) < 3000/j:
                print("Too low rate; break", j)
                break
            S = is_consistent((a,b), prefix, layer)
            # Necessary to tune 1e-5 for the appropriate TPR/FPR tradeoff
            if type(S) == np.float64 and S < 1e-5:
                print("Got close")
                print(S, cheat_neuron_diff_cuda(a[0], a[2]), cheat_neuron_diff_cuda(b[0], b[2]))
                maybe.append(b)
        print("Found set size", len(maybe))

        print("Before refine")
        for i in range(len(maybe)):
            idx = cheat_neuron_diff_cuda(maybe[i][0], maybe[i][2])
            print(idx)
        
        # OPTIONAL: increase precision, reduce recall
        if len(maybe) > 2:
            maybe = refine_cluster(maybe, layer, prefix)
        else:
            continue

        print("After refine")
        for i in range(len(maybe)):
            idx = cheat_neuron_diff_cuda(maybe[i][0], maybe[i][2])
            print(idx)

        if next_synthetic_id >= layer_end:
            print("Layer synthetic-id range exhausted; skipping remaining clusters")
            continue
        print("WRITING", next_synthetic_id)
        # refine_cluster returns an ndarray (not list/tuple); generate_dual_neuron.py
        # requires isinstance(dual_triplets, (list, tuple)). list(...) keeps each
        # element as a (3, IDIM) row (still len()==3, triplet[1] still the midpoint),
        # matching the (left, middle, right) triplet contract downstream expects.
        output[next_synthetic_id] = list(maybe)
        next_synthetic_id += 1
        # Flat exp/1-cluster-{L}.p naming (not a exp/1-cluster/ subdir) to match
        # cheat_cluster's output convention and generate_dual_neuron.py's expectation.
        pickle.dump(output, open(os.path.join(BASE_DIR, "signature_recovery/exp/1-cluster-%d.p")%layer, "wb"))

if len(sys.argv) > 2 and sys.argv[2] == 'slow':
    cluster_slow(int(sys.argv[1]))
else:
    cheat_cluster(int(sys.argv[1]))
