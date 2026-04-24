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
)

ROOT = '/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/signature_recovery/exp/1'
OUT_DIR = '/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/signature_recovery/exp'

# Per-neuron cap: recover_weights.py later slices to [:1200], so anything above
# ~3000 is pure memory pressure with no signal gain.
PER_NEURON_CAP = 3000


def stream_cluster_all():
    n_hidden_layers = len(LAYER_BOUNDARIES) - 1
    clusters = [defaultdict(list) for _ in range(n_hidden_layers)]

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
