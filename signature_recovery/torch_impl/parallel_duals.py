"""
Multiprocessing wrapper for dual-point search.

The dual-point search is embarrassingly parallel: every find_duals run (and every
boundary walk) is independent and writes its own pickle. The original driver
(run_one_model.sh STEP 2) runs find_duals.py in a SEQUENTIAL shell loop. This
wrapper runs W workers concurrently, each producing pickles in the SAME format
and SAME directory, so downstream clustering is unaware of which produced them.

Two implementations (both write identical output format):
  --impl torch       (default) batched torch finder (find_duals_torch.find_batch)
  --impl subprocess  spawn the original find_duals.py as parallel subprocesses
                     (zero algorithm change; the pure-original baseline, parallelised)

CPU-first: torch.multiprocessing spawn, each worker reloads utils (its own model
copy). float64 throughout (set by find_duals_torch). On a 14-core box, W≈cores.

Output: signature_recovery/exp/{SEED}/duals_{rand08d}.p, one per round.
"""
import os
import sys
import time
import pickle
import random
import argparse
import subprocess

import numpy as np
import torch.multiprocessing as mp

_THIS = os.path.dirname(os.path.abspath(__file__))
_SIGREC = os.path.dirname(_THIS)
if _SIGREC not in sys.path:
    sys.path.insert(0, _SIGREC)


def _torch_round(args):
    """One round of the batched torch finder -> one pickle. Runs in a worker."""
    round_id, target, batch_size, exp_dir = args
    # utils.py reads sys.argv[1] as SEED at import; strip argv so it defaults to 1.
    sys.argv = sys.argv[:1]
    # Import inside the worker so each spawned process loads its own model copy.
    import torch
    torch.set_default_dtype(torch.float64)
    from torch_impl import find_duals_torch as fdt

    np.random.seed(None)
    random.seed(None)
    triplets = fdt.find_batch(target=target, batch_size=batch_size, verbose=False)
    os.makedirs(exp_dir, exist_ok=True)
    out = os.path.join(exp_dir, "duals_%08d.p" % random.randint(0, 1000000))
    with open(out, "wb") as f:
        pickle.dump(triplets, f)
    return round_id, len(triplets), out


def _blackbox_round(args):
    """One round of the BLACK-BOX finder (cheat_remove/bb_find_duals) -> one pickle.
    Victim accessed only via argmax hard labels. Runs in a worker."""
    round_id, target, batch_size, exp_dir = args
    sys.argv = sys.argv[:1]   # utils.py reads sys.argv[1] as SEED at import
    import numpy as np
    _root = os.path.dirname(_SIGREC)
    cr = os.path.join(_root, "cheat_remove")
    if cr not in sys.path:
        sys.path.insert(0, cr)
    import bb_core, bb_find_duals
    np.random.seed(None)
    o = bb_core.Oracle()
    triplets = bb_find_duals.find_batch(o, target=target, batch_size=batch_size, verbose=False)
    os.makedirs(exp_dir, exist_ok=True)
    import random
    random.seed(None)
    out = os.path.join(exp_dir, "duals_%08d.p" % random.randint(0, 1000000))
    with open(out, "wb") as f:
        pickle.dump(triplets, f)
    return round_id, len(triplets), out


def _subprocess_round(args):
    """One round of the ORIGINAL find_duals.py as a subprocess (pure baseline)."""
    round_id, py, sigrec = args
    log = "/tmp/parallel_duals_orig_%d.log" % round_id
    with open(log, "wb") as f:
        subprocess.run([py, "find_duals.py"], cwd=sigrec, stdout=f, stderr=f, check=False)
    return round_id, -1, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=9,
                    help="number of pickle rounds to produce (≈ find_duals.py invocations)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--batch-size", type=int, default=256, help="walks per batch (torch impl)")
    ap.add_argument("--target", type=int, default=None,
                    help="triplets per round (default: find_duals_torch.TARGET for the arch)")
    ap.add_argument("--impl", choices=["torch", "subprocess", "blackbox"], default="torch")
    ap.add_argument("--output-dir", default=None,
                    help="exp dir (default: signature_recovery/exp/{SEED})")
    ap.add_argument("--python-bin", default=sys.executable)
    args = ap.parse_args()

    # utils.py reads sys.argv[1] as SEED at import; strip our flags so it defaults to 1.
    sys.argv = sys.argv[:1]
    import utils
    seed = utils.SEED
    exp_dir = args.output_dir or os.path.join(_SIGREC, "exp", str(seed))
    os.makedirs(exp_dir, exist_ok=True)

    if args.impl == "torch":
        from torch_impl import find_duals_torch as fdt
        target = args.target or fdt.TARGET
        tasks = [(i, target, args.batch_size, exp_dir) for i in range(args.iterations)]
        worker = _torch_round
        print(f"[parallel_duals] impl=torch workers={args.workers} iters={args.iterations} "
              f"batch={args.batch_size} target/round={target} -> {exp_dir}", flush=True)
    elif args.impl == "blackbox":
        _root = os.path.dirname(_SIGREC)
        sys.path.insert(0, os.path.join(_root, "cheat_remove"))
        import bb_find_duals
        target = args.target or bb_find_duals.TARGET
        bs = args.batch_size if args.batch_size != 256 else 48   # blackbox default 48
        tasks = [(i, target, bs, exp_dir) for i in range(args.iterations)]
        worker = _blackbox_round
        print(f"[parallel_duals] impl=blackbox (argmax-only) workers={args.workers} "
              f"iters={args.iterations} batch={bs} target/round={target} -> {exp_dir}", flush=True)
    else:
        tasks = [(i, args.python_bin, _SIGREC) for i in range(args.iterations)]
        worker = _subprocess_round
        print(f"[parallel_duals] impl=subprocess workers={args.workers} "
              f"iters={args.iterations} -> {exp_dir}", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(processes=args.workers) as pool:
        for round_id, n, out in pool.imap_unordered(worker, tasks):
            done += 1
            tag = f"{n} triplets" if n >= 0 else "(subprocess)"
            print(f"  round {round_id} done: {tag}  [{done}/{args.iterations}] "
                  f"{time.time() - t0:.1f}s", flush=True)

    nfiles = len([f for f in os.listdir(exp_dir) if f.endswith(".p")])
    print(f"[parallel_duals] finished {args.iterations} rounds in {time.time() - t0:.1f}s; "
          f"{nfiles} pickle files in {exp_dir}")


if __name__ == "__main__":
    main()
