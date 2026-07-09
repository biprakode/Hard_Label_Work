"""
Batched PyTorch port of signature_recovery/find_duals.py.

NO ALGORITHM CHANGES. Same walk, same constants, same output format. The only
difference is that B independent decision-boundary walks advance in lockstep so
that every single-sample oracle / gradient call in the original becomes ONE
batched forward pass over B points.

Original (find_duals.py) per dual point issues hundreds of single-sample calls:
  find_decision_boundary  (~50 bisections)
  get_normal              (1 autograd fwd+bwd)
  upper-bound step sweep  (100 forwards)
  binary search           (~30 forwards)
  refine (Newton)         (~10 forwards)
All single-sample. This file runs B walks at once -> one batched forward per step.

Output: list[(left, middle, right)] of np.ndarray shape (IDIM,) float64, pickled
to signature_recovery/exp/{SEED}/duals_{rand08d}.p — byte-compatible with the
original so cluster_dual_points_stream.py consumes it unchanged.

Randomness: like the original, NOT seeded (main reseeds to None). Values differ
run-to-run; equivalence is format + shape + dtype + range + recovery rate.

Constants preserved verbatim (see signature_recovery/MIGRATION_NOTES.md):
  path-end |dist-last| < 1e-4 ; step sweep 10**arange(-5,5,.1) ;
  step too-big > 10 ; too-small <= 1e-4 ; binary search |upper-lower| > 1e-8 ;
  is_on_boundary |gap| < 1e-10 ; a_bit_past offset +1e-4 ;
  refine Newton tol 1e-13 / 10 iters / h=1e-6 ; refine fallback step list .

GPU SUPPORT (Kaggle): all walk tensors and the oracle forward run on DEVICE
(cuda if available, else cpu), float64 throughout — numerically identical to the
CPU path (float64 GPU matmul differs only at the last ULP, far below the 1e-8
bisection tolerance). The GPU model is installed for THIS process only
(``utils.cheat_net_cuda``); the clustering stage imports utils fresh and keeps the
CPU model, so ``cheat_neuron_diff_cuda``'s CPU assumption is unaffected. Outputs
are moved back to CPU numpy before pickling, so the on-disk format is unchanged.
On a CPU-only box DEVICE == cpu and every ``.to(DEVICE)`` is a no-op.
"""
import os
import sys
import time
import pickle
import random
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# torch_impl/ lives inside signature_recovery/, so its parent dir holds utils.py.
_THIS = os.path.dirname(os.path.abspath(__file__))
_SIGREC = os.path.dirname(_THIS)
if _SIGREC not in sys.path:
    sys.path.insert(0, _SIGREC)

# Single source of truth for the configured oracle, dims and activation toggle.
# Importing utils does NOT run anything heavy (no __main__), it just loads the model.
import utils
from utils import IDIM, LEAKY_ALPHA, bmodel, gapt, TINIEST, TINIER

# Device for the dual search. cuda on Kaggle's GPU runtime, cpu on the dev box /
# during the local smoke test. We move the oracle onto DEVICE for THIS process
# only (bmodel/gapt close over the utils module global, so reassigning it here is
# enough); the separate clustering process keeps utils' default CPU model.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.cheat_net_cuda = utils.cheat_net_cuda.to(DEVICE).double()
if DEVICE.type == "cuda":
    print(f"[find_duals_torch] CUDA dual search on {torch.cuda.get_device_name(0)} "
          f"(float64)", flush=True)

# CIFAR (flagship) seeds boundary searches from REAL test images, mirroring the
# original find_decision_boundary()'s non-tiny branch; tiny/make_blobs seed from
# N(0,1). _CIFAR_SEED is True exactly when the original would sample x_test.
_CIFAR_SEED = not (utils.TINY or utils.TINIER or utils.TINIEST)

# Target triplets per saved pickle, matching find_duals.main()'s TARGET.
TARGET = 3000 if TINIEST else (2000 if TINIER else 10000)

# Step sweep grid — identical to the original's 10**np.arange(-5, 5, .1).
_STEP_VALS = torch.tensor(10.0 ** np.arange(-5, 5, 0.1), dtype=torch.float64, device=DEVICE)
# refine() fallback step list — identical to refine_to_decision_boundary().
_REFINE_FALLBACK_STEPS = [1e6, 2e6, 5e6, 1e5, 2e5, 5e5, 1e4, 2e4, 5e4, 1e3, 2e3, 5e3, 1e2]

# Cheating-ablation switch (default off = current behavior, byte-identical).
# ON: _is_on_boundary uses only oracle labels (bmodel), never the true logit
# margin gapt. Radii for the two call sites mirror find_duals.py's original
# is_on_decision_boundary(point, delta) usage (step sweep delta=1e-5, binary
# search delta=1e-9) so the honest predicate probes at a comparable scale.
HONEST_BOUNDARY_DETECT = os.environ.get("HONEST_BOUNDARY_DETECT", "0") == "1"

# ON: skip the Newton step entirely (uses true gapt's finite-difference
# derivative) and fall straight into the random-direction-probe + bisection
# fallback below, which already exists and is already exercised today for
# lanes where Newton fails to converge — this just makes it the only path.
HONEST_BOUNDARY_REFINE = os.environ.get("HONEST_BOUNDARY_REFINE", "0") == "1"


def _t(arr):
    """numpy/array -> float64 tensor on DEVICE (replaces torch.from_numpy)."""
    return torch.as_tensor(arr, dtype=torch.float64, device=DEVICE)


def _bmodel(X):
    """Batched oracle argmax. X: (B, IDIM) tensor on DEVICE -> labels (B,) int."""
    return bmodel(X)


def _gap(X):
    """Batched margin (top - runner-up), no grad. X: (B, IDIM) -> (B,)."""
    with torch.no_grad():
        return gapt(X, grad=False)


def _is_on_boundary(X, radius=1e-5):
    """Vectorised is_on_decision_boundary[_cheat].
    Cheat (default): |gap| < 1e-10, reads the true pre-softmax margin.
    Honest (HONEST_BOUNDARY_DETECT=1): label-agreement test bmodel(X+r) ==
    bmodel(X-r) for a random offset r of the given radius, mirroring
    find_duals.py's original oracle-only is_on_decision_boundary(point, delta).
    """
    if HONEST_BOUNDARY_DETECT:
        r = _t(np.random.normal(size=X.shape)) * radius
        return _bmodel(X + r) != _bmodel(X - r)
    return _gap(X).abs() < 1e-10


def _get_normal_batch(boundary):
    """Vectorised get_normal (USE_GRADIENT path).
    Per row: random_scalar * d(gap)/dx then L2-normalise. Since gap[i] depends
    only on x[i], summing the batch gap and back-propagating yields the correct
    per-sample gradient. Independent random sign per row reproduces B independent
    get_normal() calls.
    """
    x = boundary.detach().clone().requires_grad_(True)
    out = gapt(x, grad=True)              # (B,) — builds graph through forward_grad
    out.sum().backward()
    rnd = _t(np.random.normal(0.0, 1.0, size=(x.shape[0], 1)))
    real = rnd * x.grad
    real = real / torch.sqrt((real ** 2).sum(dim=1, keepdim=True))
    return real.detach()                 # (B, IDIM)


def _bisect_batch(zero, one, max_iters=200):
    """Per-lane decision-boundary bisection, mirroring find_decision_boundary()'s
    point-mode logic but with each lane keeping ITS OWN orig label (utils'
    find_decision_boundary_batched collapses to lane 0's label, unusable here).
    zero/one: (B, IDIM) tensors with differing argmax labels per lane.
    Returns the 'zero'-side boundary point (B, IDIM).
    """
    zero = zero.clone()
    one = one.clone()
    orig = _bmodel(zero)                  # (B,) per-lane label
    for _ in range(max_iters):
        s = (zero - one).abs().sum(dim=1)
        active = s > 1e-16
        if not active.any():
            break
        mid = (zero + one) / 2
        idx = _bmodel(mid)
        same = (idx == orig) & active
        diff = (idx != orig) & active
        zero = torch.where(same.unsqueeze(1), mid, zero)
        one = torch.where(diff.unsqueeze(1), mid, one)
    return zero


def _sample_seed_points(n):
    """Draw n seed points. CIFAR flagship: random REAL test images (matching
    find_decision_boundary()'s non-tiny branch). Tiny/make_blobs: N(0,1)."""
    if _CIFAR_SEED:
        idx = np.random.randint(0, len(utils.x_test), size=n)
        return utils.x_test[idx].astype(np.float64).copy()
    return np.random.normal(size=(n, IDIM))


def _init_boundaries(B):
    """Initialise B boundary points: sample random pairs with differing labels,
    then bisect. CIFAR seeds from real images; tiny from N(0,1) — both mirror the
    original find_decision_boundary(). Returns (boundary (B', IDIM), B') where
    B' <= B is the number of lanes that found a differing-label pair.
    """
    pts_a = _sample_seed_points(B)
    pts_b = _sample_seed_points(B)
    ta = _t(pts_a)
    tb = _t(pts_b)
    la = _bmodel(ta).cpu().numpy()
    lb = _bmodel(tb).cpu().numpy()
    for _ in range(50):
        same = (la == lb)
        if not same.any():
            break
        pts_b[same] = _sample_seed_points(int(same.sum()))
        tb = _t(pts_b)
        lb = _bmodel(tb).cpu().numpy()
    ok = la != lb
    if not ok.any():
        return None, 0
    bnd = _bisect_batch(_t(pts_a[ok]), _t(pts_b[ok]))
    return bnd, int(ok.sum())


def _refine_batch(a_bit_past):
    """Vectorised refine_to_decision_boundary[_cheat]: Newton's method on gap,
    with the original's random-direction fallback for non-converged lanes.
    Returns (refined (B, IDIM), ok (B,) bool). ok=False lanes correspond to the
    original returning None (path ends).
    """
    B = a_bit_past.shape[0]
    x = a_bit_past.clone()
    h = 1e-6
    tol = 1e-13
    if HONEST_BOUNDARY_REFINE:
        converged = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    else:
        y = _gap(x)
        converged = y.abs() < tol
        for _ in range(10):
            if converged.all():
                break
            dydx = (y - _gap(x - h)) / h          # directional deriv along all-ones, per lane
            # dy/dx == 0 would have triggered the real-fallback in the original; treat
            # as not-yet-converged and let the random fallback below handle it.
            safe = dydx != 0
            step = torch.where(safe, y / dydx, torch.zeros_like(y))
            x_new = x - step.unsqueeze(1)
            upd = (~converged) & safe
            x = torch.where(upd.unsqueeze(1), x_new, x)
            y = _gap(x)
            converged = converged | (y.abs() < tol)

    ok = converged.clone()
    # Fallback for lanes Newton did not converge: probe random directions at
    # shrinking radii until a label flip brackets the boundary, then bisect.
    need = ~ok
    if need.any():
        found = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        r_keep = torch.zeros_like(x)
        for step in _REFINE_FALLBACK_STEPS:
            todo = need & (~found)
            if not todo.any():
                break
            r = _t(np.random.normal(size=(B, IDIM))) / step
            flip = _bmodel(x + r) != _bmodel(x - r)
            newly = todo & flip
            r_keep = torch.where(newly.unsqueeze(1), r, r_keep)
            found = found | newly
        if found.any():
            fb = _bisect_batch((x + r_keep)[found], (x - r_keep)[found])
            idx = torch.nonzero(found, as_tuple=True)[0]
            x[idx] = fb
            ok = ok | found
    return x, ok


def find_batch(target=TARGET, batch_size=256, max_outer=2000, verbose=True):
    """Run batched boundary walks until `target` triplets are collected.
    Returns list[(left, middle, right)] of np.ndarray (IDIM,) float64.

    Lanes whose path has ended are COMPACTED out of the batch each iteration
    (pure efficiency — identical results), so the long tail of a few slow walks
    costs forwards over only the surviving lanes rather than the full width.
    """
    triplets = []
    t0 = time.time()
    while len(triplets) < target:
        boundary, B = _init_boundaries(batch_size)
        if B == 0:
            continue

        start = boundary.clone()
        # Fixed random unit vector per lane (find_dual_points's `rr`).
        rr = _t(np.random.normal(size=(B, IDIM)))
        rr = rr / torch.sqrt((rr ** 2).sum(dim=1, keepdim=True))

        last_dist = torch.full((B,), 1e9, dtype=torch.float64, device=DEVICE)
        ids = torch.arange(B, device=DEVICE)   # original lane id of each surviving row
        lane_pairs = {i: [] for i in range(B)}  # per lane: list of (left, dual) tensors

        def _filt(mask, *tensors):
            return tuple(t[mask] for t in tensors)

        for _outer in range(max_outer):
            if boundary.shape[0] == 0:
                break

            # --- path-end convergence: |dist - last| < 1e-4 ends the path ---
            dist = torch.sqrt(((boundary - start) ** 2).sum(dim=1))
            keep = (dist - last_dist).abs() >= 1e-4
            if not keep.any():
                break
            if not keep.all():
                boundary, start, rr, ids, dist, last_dist = _filt(
                    keep, boundary, start, rr, ids, dist, last_dist)
            last_dist = dist

            # --- tangent step direction: project rr off the normal ---
            normal = _get_normal_batch(boundary)
            dot_nr = (normal * rr).sum(dim=1, keepdim=True)
            dot_nn = (normal * normal).sum(dim=1, keepdim=True)
            step_dir = rr - normal * (dot_nr / dot_nn)
            step_dir = step_dir / torch.sqrt((step_dir ** 2).sum(dim=1, keepdim=True))

            # --- upper-bound step sweep (10**arange(-5,5,.1)) ---
            k = boundary.shape[0]
            prev = torch.full((k,), float(_STEP_VALS[0]), dtype=torch.float64, device=DEVICE)
            step_at = torch.full((k,), float('nan'), dtype=torch.float64, device=DEVICE)
            broke = torch.zeros(k, dtype=torch.bool, device=DEVICE)
            for sv in _STEP_VALS:
                if broke.all():
                    break
                on = _is_on_boundary(boundary + step_dir * sv, radius=1e-5)
                prev = torch.where((~broke) & on, sv, prev)
                newbroke = (~broke) & (~on)
                step_at = torch.where(newbroke, sv, step_at)
                broke = broke | newbroke
            # lanes still on-boundary at the largest sv (>10) -> caught by too-big guard
            step_at = torch.where(torch.isnan(step_at),
                                  torch.tensor(float(_STEP_VALS[-1]), device=DEVICE), step_at)

            # --- step guards: too big (>10) or too small (<=1e-4) end the path ---
            survive = (step_at <= 10) & (step_at > 1e-4)
            if not survive.any():
                break
            if not survive.all():
                boundary, start, rr, ids, last_dist, step_dir, prev, step_at = _filt(
                    survive, boundary, start, rr, ids, last_dist, step_dir, prev, step_at)

            # --- binary search on [prev, step_at], stop at |upper-lower| <= 1e-8 ---
            lower = prev.clone()
            upper = step_at.clone()
            mid_step = (lower + upper) / 2
            for _ in range(60):
                need = (upper - lower).abs() > 1e-8
                if not need.any():
                    break
                m = (lower + upper) / 2
                on = _is_on_boundary(boundary + step_dir * m.unsqueeze(1), radius=1e-9)
                lower = torch.where(need & on, m, lower)
                upper = torch.where(need & (~on), m, upper)
                mid_step = torch.where(need, m, mid_step)

            # --- record dual triplet halves for the surviving lanes ---
            left = boundary + step_dir * (mid_step.unsqueeze(1) / 2)
            dual = boundary + step_dir * mid_step.unsqueeze(1)
            id_list = ids.tolist()
            for j, lid in enumerate(id_list):
                lane_pairs[lid].append((left[j].clone(), dual[j].clone()))

            # --- advance to the next decision boundary (a_bit_past +1e-4) ---
            a_bit_past = boundary + step_dir * (mid_step + 1e-4).unsqueeze(1)
            nb, ok = _refine_batch(a_bit_past)
            boundary = nb
            if not ok.all():    # lanes where refine failed end (original: exit/None)
                if not ok.any():
                    break
                boundary, start, rr, ids, last_dist = _filt(
                    ok, boundary, start, rr, ids, last_dist)

        # --- zip consecutive (left,dual) pairs per lane -> (left, dual, right) ---
        # Move back to CPU numpy here so the on-disk pickle format is unchanged.
        for pairs in lane_pairs.values():
            for (l0, d0), (l1, _d1) in zip(pairs, pairs[1:]):
                triplets.append((l0.cpu().numpy(), d0.cpu().numpy(), l1.cpu().numpy()))

        if verbose:
            print(f"  [find_batch] collected {len(triplets)}/{target} "
                  f"(B={B}, {time.time() - t0:.1f}s)", flush=True)

    return triplets


def main():
    # Mirror find_duals.main(): non-deterministic, save one pickle to exp/{SEED}/.
    np.random.seed(None)
    random.seed(None)
    batch_size = int(os.environ.get("DUAL_BATCH", "256"))
    seed = utils.SEED
    exp_dir = os.path.join(_SIGREC, "exp", str(seed))
    os.makedirs(exp_dir, exist_ok=True)

    triplets = find_batch(target=TARGET, batch_size=batch_size)
    out = os.path.join(exp_dir, "duals_%08d.p" % random.randint(0, 1000000))
    with open(out, "wb") as f:
        pickle.dump(triplets, f)
    print(f"Finished: {len(triplets)} triplets -> {out}")


if __name__ == "__main__":
    main()
