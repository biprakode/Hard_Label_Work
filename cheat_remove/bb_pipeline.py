"""
End-to-end BLACK-BOX extraction for tiniest.

Stage 1 (this file): victim accessed ONLY via argmax hard labels.
  - blackbox dual points          (bb_find_duals)
  - blackbox boundary normals      (bb_core)
  - blackbox consistency clustering + SVD weight recovery (bb_recover)
  - write recovered layer-0 directions in Phase-3 signature format
Stage 2 (run_extraction.py, separate call): hard-label Phase 3
  - bias recovery (geometric) + oracle_sign_search + fc5 LR fit + refinement.

Layer separation note: on this make_blobs victim, deeper-layer neurons can look
globally linear over the data manifold, so input-space clustering cannot cleanly
isolate layer 0 (see CHEAT_REMOVE_RESULTS.md). We therefore feed Phase 3 the
top-N globally-consistent input-space directions (frozen) and let the hard-label
Phase 3 close the functional gap. No victim parameter is ever read.

Usage:
  python3 bb_pipeline.py --target 3000 --neurons 8 --out <Vrelu_dir>
"""
import os
import sys
import json
import time
import shutil
import argparse
import numpy as np

# utils.py (imported transitively below) reads sys.argv[1] as SEED at import;
# stash our CLI args and strip argv so that import sees a clean argv.
_ARGV = sys.argv[:]
sys.argv = sys.argv[:1]

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import bb_core as bb
import bb_find_duals as bfd
import bb_recover as bbr
import utils  # attacker-side config only (IDIM, LAYER_SIZES, LEAKY_ALPHA)


def write_layer0_signatures(clusters, out_dir, n_neurons):
    """Write recovered layer-0 directions (gauge ‖w‖=1) in the format
    weight_assembly.load_unsigned_weights expects: layer_0/neuron_<i>/{
    weights_unscaled.npz, metadata.json}. Arbitrary stable ids (permutation
    gauge); scaling_factor=1.0 (no cheat_solution). Biggest clusters first."""
    layer_dir = os.path.join(out_dir, "layer_0")
    if os.path.exists(layer_dir):
        shutil.rmtree(layer_dir)
    os.makedirs(layer_dir, exist_ok=True)
    order = sorted(range(len(clusters)), key=lambda k: -len(clusters[k]['idxs']))
    written = 0
    for nid, ci in enumerate(order[:n_neurons]):
        c = clusters[ci]
        nd = os.path.join(layer_dir, f"neuron_{nid}")
        os.makedirs(nd, exist_ok=True)
        np.savez(os.path.join(nd, "weights_unscaled.npz"), c['w'].astype(np.float64))
        with open(os.path.join(nd, "metadata.json"), "w") as f:
            json.dump({"matched_neuron": nid, "scaling_factor": 1.0,
                       "absolute_error": 0.0, "cluster_id": nid,
                       "cluster_size": len(c['idxs']),
                       "blackbox": True}, f, indent=2)
        written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3000, help="dual triplets to collect")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--neurons", type=int, default=None, help="layer-0 width (default LAYER_SIZES[1])")
    ap.add_argument("--out", default=None, help="Vrelu signature dir to write into")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(_ARGV[1:])

    np.random.seed(args.seed)
    n_neurons = args.neurons or utils.LAYER_SIZES[1]
    out_dir = args.out or os.path.join(
        os.path.dirname(_THIS), "signature_recovery/outputs/model_weights/Vrelu")
    os.makedirs(out_dir, exist_ok=True)

    o = bb.Oracle()
    print(f"[bb_pipeline] arch={utils.LAYER_SIZES} alpha={utils.LEAKY_ALPHA} "
          f"target={args.target} neurons={n_neurons}", flush=True)

    t = time.time()
    triplets = bfd.find_batch(o, target=args.target, batch_size=args.batch_size, verbose=True)
    print(f"[bb_pipeline] duals: {len(triplets)} triplets, {time.time()-t:.1f}s, "
          f"{o.n_queries} oracle queries", flush=True)

    t = time.time()
    NL, NR, valid = bbr.precompute_normals(o, triplets)
    print(f"[bb_pipeline] normals: {time.time()-t:.1f}s, valid {int(valid.sum())}/{len(triplets)}, "
          f"{o.n_queries} total queries", flush=True)

    prefix = bbr.LinearizedPrefix([], alpha=utils.LEAKY_ALPHA)   # layer 0 = identity
    clusters, unassigned = bbr.recover_layer(
        o, triplets, NL, NR, valid, prefix, n_neurons=n_neurons,
        alpha=utils.LEAKY_ALPHA, verbose=True)

    written = write_layer0_signatures(clusters, out_dir, n_neurons)
    print(f"[bb_pipeline] wrote {written} layer-0 signature directions -> {out_dir}/layer_0", flush=True)
    print(f"[bb_pipeline] TOTAL oracle queries: {o.n_queries}", flush=True)


if __name__ == "__main__":
    main()
