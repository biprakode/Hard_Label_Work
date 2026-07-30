"""
Streaming version of cluster_dual_points.py.

Instead of loading all ~10M triplets into one giant `duals` list
(which OOMs a 24GB machine because Python overhead balloons it to 20-30GB),
we stream the pickle files one at a time and process each triplet in place.

Outputs one cluster file per hidden layer:
    exp/1-cluster-{0,1,2,3}.p
Each is a dict[flat_neuron_idx] -> list[(left, middle, right)] matching the
format produced by the original cheat_cluster().
"""
import os
import sys
import gc
import pickle
import time
from collections import defaultdict

from utils import (
    cheat_neuron_diff_cuda,
    LAYER_BOUNDARIES,
    LAYER_SIZES,
    BASE_DIR,
)

ROOT = os.path.join(BASE_DIR, 'signature_recovery/exp/1')
OUT_DIR = os.path.join(BASE_DIR, 'signature_recovery/exp')

# Per-neuron cap. recover_weights.py slices to [:1200] and its hits-coverage
# subsampling effectively uses only ~2 duals per input coord, so a few hundred
# distinct-region duals per neuron already over-determine the SVD null space.
# On the full CIFAR flagship (832 hidden neurons) the cluster dict is held in RAM,
# so the cap is the dominant RAM knob: 832 * CAP * 74KB. On a 22GB box keep it
# small. CAP=150 -> ~9GB peak.  (override with env CLUSTER_PER_NEURON_CAP)
import os as _os
PER_NEURON_CAP = int(_os.environ.get("CLUSTER_PER_NEURON_CAP", "150"))

# Optional layer filter: cluster ONLY these hidden layers (comma-separated, e.g.
# "2,3"). Lets us re-cluster just the deep layers at a much higher per-neuron cap
# (RAM = sum_over_selected_layers(neurons * CAP * 74KB)) without paying for the
# wide shallow layers. Default: all layers. Unselected layers' cluster files are
# left untouched on disk.
_LAYERS_ENV = _os.environ.get("CLUSTER_LAYERS", "").strip()
ONLY_LAYERS = set(int(x) for x in _LAYERS_ENV.split(",") if x != "") if _LAYERS_ENV else None

# MERGE mode (chunked dual search, e.g. the Kaggle CIFAR run under a disk cap):
# preload any existing 1-cluster-{L}.p into the buckets before streaming the new
# pickles, so clustering a fresh chunk ACCUMULATES into the prior chunks' clusters
# (still capped at PER_NEURON_CAP) instead of overwriting them. This lets the
# driver cluster+delete raw pickles after every chunk and keep peak disk low.
MERGE = _os.environ.get("CLUSTER_MERGE", "0") == "1"


def stream_cluster_all():
    n_hidden_layers = len(LAYER_BOUNDARIES) - 1
    clusters = [defaultdict(list) for _ in range(n_hidden_layers)]

    if MERGE:
        preloaded = 0
        for L in range(n_hidden_layers):
            if ONLY_LAYERS is not None and L not in ONLY_LAYERS:
                continue
            prev_path = os.path.join(OUT_DIR, f"1-cluster-{L}.p")
            if os.path.exists(prev_path):
                with open(prev_path, 'rb') as f:
                    prev = pickle.load(f)
                for flat_idx, lst in prev.items():
                    clusters[L][flat_idx] = list(lst[:PER_NEURON_CAP])
                    preloaded += len(clusters[L][flat_idx])
                del prev
        print(f"[merge] preloaded {preloaded} triplets from existing cluster files", flush=True)

    files = sorted(os.listdir(ROOT))
    total_files = len(files)
    print(f"Streaming {total_files} dual-point files, {n_hidden_layers} layer buckets")
    print(f"LAYER_SIZES: {LAYER_SIZES}")
    print(f"LAYER_BOUNDARIES: {LAYER_BOUNDARIES}")

    total_seen = 0
    total_kept = 0
    t0 = time.time()

    for fi, fname in enumerate(files):
        path = os.path.join(ROOT, fname)
        with open(path, 'rb') as f:
            triplets = pickle.load(f)

        for (left, middle, right) in triplets:
            total_seen += 1
            diff = cheat_neuron_diff_cuda(left, right)
            if len(diff) == 1:
                flat_idx = int(diff[0])
                for L in range(n_hidden_layers):
                    if LAYER_BOUNDARIES[L] <= flat_idx < LAYER_BOUNDARIES[L + 1]:
                        if ONLY_LAYERS is not None and L not in ONLY_LAYERS:
                            break
                        bucket = clusters[L][flat_idx]
                        if len(bucket) < PER_NEURON_CAP:
                            bucket.append((left, middle, right))
                            total_kept += 1
                        break

        del triplets
        if (fi + 1) % 20 == 0 or fi + 1 == total_files:
            elapsed = time.time() - t0
            per_layer = [sum(len(v) for v in clusters[L].values()) for L in range(n_hidden_layers)]
            print(f"[{fi + 1:4d}/{total_files}] seen={total_seen} kept={total_kept} "
                  f"per_layer={per_layer} elapsed={elapsed:.1f}s", flush=True)
            gc.collect()

    print("\nWriting layer clusters...")
    for L in range(n_hidden_layers):
        if ONLY_LAYERS is not None and L not in ONLY_LAYERS:
            continue   # leave unselected layers' cluster files untouched on disk
        out_path = os.path.join(OUT_DIR, f"1-cluster-{L}.p")
        data = dict(clusters[L])
        print(f"  layer {L}: {len(data)} neurons covered, "
              f"{sum(len(v) for v in data.values())} triplets -> {out_path}")
        with open(out_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        del data
        clusters[L] = None
        gc.collect()

    print(f"\nTotals: seen={total_seen} kept={total_kept} elapsed={time.time() - t0:.1f}s")


if __name__ == '__main__':
    stream_cluster_all()
