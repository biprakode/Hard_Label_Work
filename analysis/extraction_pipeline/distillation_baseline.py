"""
Arch-agnostic pure-distillation baseline generator.

The distillation arm is the *same architecture with NO frozen cryptanalytic rows*:
all hidden rows Kaiming-initialised and trainable, fit to the victim's hard labels
on the same oracle query pool. It is the control the +6-7% extraction advantage is
measured against, and — per project policy — a **mandatory** companion to every
extraction eval (see `evaluate_extraction_quality.py`, which calls
`ensure_distillation_baseline` before scoring).

Mechanism (non-destructive, unlike the legacy CIFAR-only `run_distillation_baseline.sh`):
we drive the SAME Phase-3 workflow (`workflow.main`) but redirect its
`--signature-path` / `--sign-path` to *empty* temp dirs, so `reconstruct_model`
finds 0 recovered and Kaiming-inits every row; `--refine-unfreeze` then trains them
all. Output is redirected to a temp dir via `--output-path`, then archived under the
canonical `_distillation` names. The real extraction artifacts on disk are never
touched.
"""

import os
import json
import shutil
import tempfile

from .config import OUTPUT_PATH
from . import workflow as _wf
from .data_loading import load_test3_data


# arch_key -> (workflow CLI flag, reconstructed-model basename, default refine epochs)
_ARCH_FLAG = {
    'tiniest':  ('--tiniest',   'reconstructed_tiniest',   300),
    'tinier':   ('--tinier',    'reconstructed_tinier',    500),
    'makeblobs':('--makeblobs', 'reconstructed_makeblobs', 500),
    'tiny':     ('--tiny',      'reconstructed_tiny',      500),
    'full':     ('--full',      'reconstructed_full',      500),
}


def distillation_paths(arch_key):
    """Canonical on-disk paths for an arch's distillation artifacts.

    Namespaced by arch so multiple tiers can coexist. For `full` the model file
    keeps its legacy name (`reconstructed_full_distillation.pth`) so the CIFAR
    flagship tooling keeps working.
    """
    _, base, _ = _ARCH_FLAG[arch_key]
    pth = os.path.join(OUTPUT_PATH, f"{base}_distillation.pth")
    metrics = os.path.join(OUTPUT_PATH, f"extraction_metrics_{arch_key}_distillation.json")
    return pth, metrics


def _legacy_full_metrics():
    """Legacy CIFAR distillation metrics filename (pre-namespacing)."""
    return os.path.join(OUTPUT_PATH, "extraction_metrics_distillation.json")


def ensure_distillation_baseline(arch_key, force=False, refine_epochs=None,
                                 extra_argv=None, verbose=True):
    """Guarantee a distillation arm exists on disk for `arch_key`; build if missing.

    Returns (dist_pth, dist_metrics_path). Idempotent: a cached baseline is reused
    unless `force=True`.
    """
    if arch_key not in _ARCH_FLAG:
        raise ValueError(f"unknown arch_key {arch_key!r}")
    flag, base, default_epochs = _ARCH_FLAG[arch_key]
    dist_pth, dist_metrics = distillation_paths(arch_key)

    # Reuse a cached baseline (or migrate the legacy CIFAR file into place).
    if not force and os.path.isfile(dist_pth):
        if not os.path.isfile(dist_metrics) and arch_key == 'full' \
                and os.path.isfile(_legacy_full_metrics()):
            shutil.copy(_legacy_full_metrics(), dist_metrics)
        if os.path.isfile(dist_metrics):
            if verbose:
                print(f"[distillation] reusing cached baseline: {dist_pth}")
            return dist_pth, dist_metrics

    epochs = refine_epochs if refine_epochs is not None else default_epochs

    tmp_root = tempfile.mkdtemp(prefix=f"distill_{arch_key}_")
    empty_sig = os.path.join(tmp_root, "sig")
    empty_sign = os.path.join(tmp_root, "sign")
    out_dir = os.path.join(tmp_root, "out")
    for d in (empty_sig, empty_sign, out_dir):
        os.makedirs(d, exist_ok=True)

    argv = [
        flag,
        '--signature-path', empty_sig,   # empty -> 0 recovered -> Kaiming hidden
        '--sign-path', empty_sign,
        '--output-path', out_dir,
        '--refine', '--refine-unfreeze',  # all rows trainable: pure distillation
        '--refine-epochs', str(epochs),
        '--refine-weight-decay', '1e-4',  # same Fix-B regularisers as extraction arm
        '--refine-cosine-lr',
    ]
    # Honest held-out eval + 20K train-union for the CIFAR flagship (matches the
    # extraction arm's regime). X_test3 exists for make_blobs tiers too, but the
    # small-arch extraction runs train on X_test only, so we keep parity there.
    if arch_key == 'full':
        x3, _ = load_test3_data(tiny=False, makeblobs=False, tinier=False, tiniest=False)
        if x3 is not None:
            argv += ['--eval-on-test3', '--train-union-test12',
                     '--early-stop', '--patience', '5', '--eval-every', '10']
    if extra_argv:
        argv += list(extra_argv)

    if verbose:
        print(f"[distillation] building baseline for {arch_key}: "
              f"workflow {' '.join(argv)}")
    _wf.main(argv)

    # Archive the workflow's outputs under the canonical _distillation names.
    produced_pth = os.path.join(out_dir, f"{base}.pth")
    produced_metrics = os.path.join(out_dir, "extraction_metrics.json")
    if not os.path.isfile(produced_pth):
        raise RuntimeError(f"distillation run did not produce {produced_pth}")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    shutil.copy(produced_pth, dist_pth)
    if os.path.isfile(produced_metrics):
        shutil.copy(produced_metrics, dist_metrics)

    shutil.rmtree(tmp_root, ignore_errors=True)
    if verbose:
        print(f"[distillation] wrote {dist_pth}")
        print(f"[distillation] wrote {dist_metrics}")
    return dist_pth, dist_metrics
