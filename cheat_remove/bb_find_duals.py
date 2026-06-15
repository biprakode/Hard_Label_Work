"""
Batched BLACK-BOX dual-point finder.

Same lockstep-walk + lane-compaction structure as the torch rewrite
(signature_recovery/torch_impl/find_duals_torch.find_batch), but every victim
access is an argmax hard-label query through bb_core.Oracle — no logit gap, no
autograd gradient, no true activations. B independent boundary walks advance
together so each geometric primitive issues ONE batched oracle call over B
points instead of B serial calls.

Output: list[(left, middle, right)] np.ndarray (IDIM,) float64 — byte-compatible
with find_duals.py, so the (de-cheated) recovery consumes it unchanged.

All numerical constants match the whitebox walk in find_duals.py / bb_core.py.
"""
import os
import sys
import time
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import bb_core as bb
from bb_core import IDIM, TINIEST, TINIER

TARGET = 3000 if TINIEST else (2000 if TINIER else 10000)

_STEP_VALS = 10.0 ** np.arange(-5, 5, 0.1)       # tangent sweep grid
_BRACKET_VALS = 10.0 ** np.arange(-7, 0, 0.33)   # per-dim normal bracket search
_REFINE_STEPS = [1e6, 2e6, 5e6, 1e5, 2e5, 5e5, 1e4, 2e4, 5e4, 1e3, 2e3, 5e3, 1e2]


def _find_boundary_batch(o, A, Bp, orig, tol=1e-12, max_iter=200):
    """Per-lane argmax bisection. A has label `orig` (per lane), Bp differs."""
    a = A.copy()
    b = Bp.copy()
    for _ in range(max_iter):
        s = np.abs(a - b).sum(axis=1)
        act = s > tol
        if not act.any():
            break
        m = (a + b) / 2
        lab = o.label(m)
        same = (lab == orig) & act
        diff = (lab != orig) & act
        a[same] = m[same]
        b[diff] = m[diff]
    return a


def _init_boundaries(o, B):
    """B fresh boundary points from random differing-label pairs (argmax-only)."""
    pa = bb._sample_points(B)
    pb = bb._sample_points(B)
    la = o.label(pa)
    lb = o.label(pb)
    for _ in range(60):
        same = (la == lb)
        if not same.any():
            break
        pb[same] = bb._sample_points(int(same.sum()))
        lb = o.label(pb)
    ok = la != lb
    if not ok.any():
        return None, 0
    bnd = _find_boundary_batch(o, pa[ok], pb[ok], la[ok])
    return bnd, int(ok.sum())


def _normal_batch(o, X, step_size=1e-7):
    """Finite-difference boundary normal for every lane at once (argmax-only).
    Returns (normals (B,IDIM), valid (B,) bool)."""
    B = X.shape[0]
    orig = o.label(X)
    xp = X.copy()
    xp[:, 0] += step_size
    flip0 = o.label(xp) != orig
    xp[flip0, 0] -= 2 * step_size

    ratios = np.zeros((B, IDIM))
    valid = np.ones(B, dtype=bool)
    for i in range(IDIM):
        found = np.zeros(B, dtype=bool)
        bracket = xp.copy()
        for s in _BRACKET_VALS:
            todo = ~found
            if not todo.any():
                break
            cand = xp.copy()
            cand[:, i] += s
            lab = o.label(cand)
            same = lab == orig
            cand[same, i] -= 2 * s
            lab2 = o.label(cand)
            now = todo & (lab2 != orig)
            bracket[now] = cand[now]
            found |= now
        valid &= found
        b = _find_boundary_batch(o, xp, bracket, orig)
        ratios[:, i] = (xp[:, i] - b[:, i]) / step_size

    inv = np.divide(1.0, ratios, out=np.zeros_like(ratios), where=np.abs(ratios) >= 1e-30)
    nrm = np.sqrt((inv ** 2).sum(axis=1, keepdims=True))
    valid &= (nrm[:, 0] > 1e-30)
    nrm[nrm < 1e-30] = 1.0
    return inv / nrm, valid


def _on_boundary_batch(o, X, N, delta=1e-5):
    """Vectorised on-boundary test: stepping ±delta along the normal flips the
    hard label. (B,) bool."""
    return o.label(X + delta * N) != o.label(X - delta * N)


def _refine_batch(o, X):
    """Per-lane re-projection onto a boundary via random-direction probing +
    bisection. Returns (refined (B,IDIM), ok (B,) bool)."""
    B = X.shape[0]
    x = X.copy()
    found = np.zeros(B, dtype=bool)
    r_keep = np.zeros_like(x)
    orig_for = np.zeros(B, dtype=np.int64)
    for step in _REFINE_STEPS:
        todo = ~found
        if not todo.any():
            break
        r = np.random.normal(size=(B, IDIM)) / step
        lp = o.label(x + r)
        lm = o.label(x - r)
        now = todo & (lp != lm)
        r_keep[now] = r[now]
        orig_for[now] = lp[now]            # label of (x + r) side
        found |= now
    out = x.copy()
    if found.any():
        fb = _find_boundary_batch(o, (x + r_keep)[found], (x - r_keep)[found],
                                  orig_for[found])
        out[found] = fb
    return out, found


def find_batch(o, target=TARGET, batch_size=64, max_outer=2000, verbose=False):
    """Collect ~`target` (left, middle, right) triplets via batched black-box
    boundary walks. Mirrors find_duals_torch.find_batch with lane compaction."""
    triplets = []
    t0 = time.time()
    while len(triplets) < target:
        boundary, B = _init_boundaries(o, batch_size)
        if B == 0:
            continue
        start = boundary.copy()
        rr = np.random.normal(size=(B, IDIM))
        rr /= np.sqrt((rr ** 2).sum(axis=1, keepdims=True))
        last_dist = np.full(B, 1e9)
        ids = np.arange(B)
        lane_pairs = {i: [] for i in range(B)}

        def _filt(mask, *arrs):
            return tuple(a[mask] for a in arrs)

        for _outer in range(max_outer):
            if boundary.shape[0] == 0:
                break

            dist = np.sqrt(((boundary - start) ** 2).sum(axis=1))
            keep = np.abs(dist - last_dist) >= 1e-4
            if not keep.any():
                break
            if not keep.all():
                boundary, start, rr, ids, dist, last_dist = _filt(
                    keep, boundary, start, rr, ids, dist, last_dist)
            last_dist = dist

            normal, nvalid = _normal_batch(o, boundary)
            if not nvalid.all():
                if not nvalid.any():
                    break
                boundary, start, rr, ids, last_dist, normal = _filt(
                    nvalid, boundary, start, rr, ids, last_dist, normal)

            dot_nr = (normal * rr).sum(axis=1, keepdims=True)
            dot_nn = (normal * normal).sum(axis=1, keepdims=True)
            step_dir = rr - normal * (dot_nr / dot_nn)
            step_dir /= np.sqrt((step_dir ** 2).sum(axis=1, keepdims=True))

            k = boundary.shape[0]
            prev = np.full(k, _STEP_VALS[0])
            step_at = np.full(k, np.nan)
            broke = np.zeros(k, dtype=bool)
            for sv in _STEP_VALS:
                if broke.all():
                    break
                on = _on_boundary_batch(o, boundary + step_dir * sv, normal)
                prev = np.where((~broke) & on, sv, prev)
                newbroke = (~broke) & (~on)
                step_at = np.where(newbroke, sv, step_at)
                broke = broke | newbroke
            step_at = np.where(np.isnan(step_at), _STEP_VALS[-1], step_at)

            survive = (step_at <= 10) & (step_at > 1e-4)
            if not survive.any():
                break
            if not survive.all():
                boundary, start, rr, ids, last_dist, step_dir, normal, prev, step_at = _filt(
                    survive, boundary, start, rr, ids, last_dist, step_dir, normal, prev, step_at)

            lo = prev.copy()
            hi = step_at.copy()
            mid_step = (lo + hi) / 2
            for _ in range(60):
                need = np.abs(hi - lo) > 1e-8
                if not need.any():
                    break
                m = (lo + hi) / 2
                on = _on_boundary_batch(o, boundary + step_dir * m[:, None], normal)
                lo = np.where(need & on, m, lo)
                hi = np.where(need & (~on), m, hi)
                mid_step = np.where(need, m, mid_step)

            left = boundary + step_dir * (mid_step[:, None] / 2)
            dual = boundary + step_dir * mid_step[:, None]
            for j, lid in enumerate(ids.tolist()):
                lane_pairs[lid].append((left[j].copy(), dual[j].copy()))

            a_bit_past = boundary + step_dir * (mid_step[:, None] + 1e-4)
            nb, ok = _refine_batch(o, a_bit_past)
            boundary = nb
            if not ok.all():
                if not ok.any():
                    break
                boundary, start, rr, ids, last_dist = _filt(
                    ok, boundary, start, rr, ids, last_dist)

        for pairs in lane_pairs.values():
            for (l0, d0), (l1, _d1) in zip(pairs, pairs[1:]):
                triplets.append((l0, d0, l1))

        if verbose:
            print(f"  [bb find_batch] {len(triplets)}/{target} "
                  f"(B={B}, {time.time() - t0:.1f}s, {o.n_queries} queries)", flush=True)
    return triplets
