"""
Black-box core: the victim is accessed ONLY through argmax hard labels.

Everything an attacker is allowed to know about the victim flows through
`Oracle.label(X)`. No true weights, no true biases, no hidden activations, no
logits / logit-gaps are ever read. The geometric primitives below
(boundary bisection, finite-difference normal, dual-point walk) reproduce the
math of signature_recovery's whitebox helpers using only the hard label.

This module loads the victim via signature_recovery.utils' CIFAR10Net loader
purely as a convenience to get the architecture + weights onto disk into a
forward-only module; the ONLY method the rest of cheat_remove/ calls is
`Oracle.label`, which returns argmax and nothing else.
"""
import os
import sys
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_THIS = os.path.dirname(os.path.abspath(__file__))
_SIGREC = os.path.join(os.path.dirname(_THIS), "signature_recovery")
if _SIGREC not in sys.path:
    sys.path.insert(0, _SIGREC)

# We import utils only to (a) build the victim forward module and (b) read the
# attacker-side config (IDIM, LAYER_SIZES, MODEL_PATH, dataset sampling).
# IDIM / LAYER_SIZES are PUBLIC attack assumptions (the attacker chooses the
# architecture hypothesis); they are not secret victim parameters.
import utils
IDIM = utils.IDIM
LAYER_SIZES = utils.LAYER_SIZES
TINY = utils.TINY
TINIER = utils.TINIER
TINIEST = utils.TINIEST


class MathIsHard(Exception):
    pass


class Oracle:
    """The black box. The ONLY victim access in the whole cheat_remove pipeline."""

    def __init__(self, model=None):
        # Build a forward-only copy of the victim. After construction we only
        # ever call .label(); the parameters are never read by the attack.
        if model is None:
            model = utils.CIFAR10Net()
            model = utils.load_converted_model(utils.MODEL_PATH, model, utils.device)
            model.double().eval()
        self._model = model
        self.n_queries = 0

    def label(self, X):
        """Hard-label query. X: (B, IDIM) or (IDIM,) ndarray/tensor -> (B,) int."""
        t = torch.as_tensor(np.asarray(X, dtype=np.float64))
        if t.ndim == 1:
            t = t.unsqueeze(0)
        self.n_queries += t.shape[0]
        with torch.no_grad():
            return self._model(t).argmax(dim=1).cpu().numpy().astype(np.int64)

    def label1(self, x):
        """Convenience scalar-label for a single point."""
        return int(self.label(x)[0])


def _sample_points(n):
    """Sample candidate inputs the same way utils.find_decision_boundary does
    for the tiny/tinier/tiniest models (standard normal). Public, victim-free."""
    return np.random.normal(size=(n, IDIM))


def find_boundary(o, x0, x1, tol=1e-12, max_iter=300):
    """Bisect between two points with different hard labels -> a point on the
    decision boundary (argmax tie surface). Argmax-only."""
    a = np.array(x0, dtype=np.float64)
    b = np.array(x1, dtype=np.float64)
    la = o.label1(a)
    for _ in range(max_iter):
        if np.abs(a - b).sum() < tol:
            break
        m = (a + b) / 2
        if o.label1(m) == la:
            a = m
        else:
            b = m
    return a


def random_boundary(o, max_tries=200):
    """Find a fresh decision-boundary point from random differing-label samples."""
    for _ in range(max_tries):
        pts = _sample_points(16)
        labs = o.label(pts)
        uniq = np.unique(labs)
        if len(uniq) >= 2:
            i0 = np.where(labs == uniq[0])[0][0]
            i1 = np.where(labs == uniq[1])[0][0]
            return find_boundary(o, pts[i0], pts[i1])
    raise MathIsHard("could not find two differing labels")


def boundary_normal(o, x, step_size=1e-7):
    """Finite-difference estimate of the decision-boundary normal at boundary
    point x, using ONLY hard labels (reproduces utils.get_gradient_dir's math).

    For a local hyperplane n·x = c, moving from a near-boundary point xp along
    axis e_i hits the boundary at t_i = (c - n·xp)/n_i, so
    ratio_i = (xp_i - boundary_i)/step ∝ 1/n_i  =>  normal ∝ 1/ratios.
    """
    x = np.array(x, dtype=np.float64)
    orig = o.label1(x)
    xp = x.copy()
    xp[0] += step_size
    if o.label1(xp) != orig:
        xp[0] -= 2 * step_size

    ratios = np.empty(IDIM)
    for i in range(IDIM):
        found = False
        for step in 10.0 ** np.arange(-7, 0, 0.33):
            xp2 = xp.copy()
            xp2[i] += step
            if o.label1(xp2) == orig:
                xp2[i] -= 2 * step
            if o.label1(xp2) != orig:
                found = True
                break
        if not found:
            raise MathIsHard("no boundary bracket along dim %d" % i)
        b = find_boundary(o, xp, xp2)
        denom = (xp[i] - b[i])
        ratios[i] = denom / step_size
    # normal ∝ 1/ratios ; guard against exact zeros
    inv = np.where(np.abs(ratios) < 1e-30, 0.0, 1.0 / ratios)
    nrm = np.sqrt((inv ** 2).sum())
    if nrm < 1e-30:
        raise MathIsHard("degenerate normal")
    return inv / nrm


def on_boundary(o, x, normal, delta=1e-5):
    """Black-box 'is x on the decision boundary' test: stepping a little along
    the (locally estimated) normal flips the hard label. Replaces the whitebox
    |logit_gap(x)| < tol check."""
    return o.label1(x + delta * normal) != o.label1(x - delta * normal)


def refine_to_boundary(o, x):
    """Project x back onto a decision boundary by probing random directions at
    shrinking radii until a label flip brackets it, then bisecting. Mirrors
    utils.refine_to_decision_boundary's fallback. Argmax-only."""
    x = np.array(x, dtype=np.float64)
    for step in [1e6, 2e6, 5e6, 1e5, 2e5, 5e5, 1e4, 2e4, 5e4, 1e3, 2e3, 5e3, 1e2]:
        r = np.random.normal(size=IDIM) / step
        if o.label1(x + r) != o.label1(x - r):
            return find_boundary(o, x + r, x - r)
    return None


def find_dual_points(o, verbose=False):
    """Walk along the decision boundary; each time the boundary bends (a hidden
    neuron toggles) record a dual point. Returns list of (half_step_pt, dual_pt)
    pairs — same structure as find_duals.find_dual_points's middle_points.
    Argmax-only; constants match the whitebox walk in find_duals.py.
    """
    middle = []
    start = boundary = orig = random_boundary(o)
    last_dist = 1e9
    rr = np.random.normal(size=IDIM)
    rr /= np.sqrt((rr ** 2).sum())

    while True:
        dist = np.sqrt(((boundary - start) ** 2).sum())
        if np.abs(dist - last_dist) < 1e-4:
            break
        last_dist = dist

        try:
            n = boundary_normal(o, boundary)
        except MathIsHard:
            break

        step_dir = rr - n * (np.dot(n, rr) / np.dot(n, n))
        step_dir /= np.sqrt((step_dir ** 2).sum())

        # upper-bound sweep: how far can we step along the tangent and stay on
        # the (locally linear) boundary?
        prev_step = 10.0 ** -5
        step_size = prev_step
        for step_size in 10.0 ** np.arange(-5, 5, 0.1):
            if not on_boundary(o, boundary + step_dir * step_size, n):
                break
            prev_step = step_size

        if step_size > 10:
            break
        if step_size <= 1e-4:
            break

        # binary search for the exact kink location along the tangent
        lo, hi = prev_step, step_size
        while np.abs(hi - lo) > 1e-8:
            mid = (lo + hi) / 2
            if on_boundary(o, orig + step_dir * mid, n):
                lo = mid
            else:
                hi = mid
        mid_step = (lo + hi) / 2

        middle.append((orig + step_dir * mid_step / 2, orig + step_dir * mid_step))

        a_bit_past = orig + step_dir * (mid_step + 1e-4)
        nb = refine_to_boundary(o, a_bit_past)
        if nb is None:
            break
        boundary = orig = nb

    if verbose:
        print(f"  path found {len(middle)} dual points")
    return middle
