"""
Black-box per-layer signature recovery (peeling).

Recovers layer-L weight directions (gauge ‖w‖=1) from dual-point triplets, using
ONLY argmax-derived boundary normals (bb_core) and a prefix built from the
already-recovered lower layers (identity for L=0). Clustering is SVD-consistency
/ projection-peak in the prefix-output space — no `cheat_neuron_diff`, no true
weights, no `cheat_solution`.

The recovery SVD math mirrors signature_recovery/recover_weights.is_consistent_help
(intersect -> subspace samples -> forward_around -> SVD null-space), but every
victim touch is a hard label and the prefix is reconstructed, not true.
"""
import os
import sys
import numpy as np
import scipy.linalg

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import bb_core as bb
from bb_core import IDIM, LAYER_SIZES
import bb_find_duals as bfd


# --------------------------------------------------------------------------- #
# Reconstructed prefix (built from recovered signed+biased lower layers).
# --------------------------------------------------------------------------- #
class LinearizedPrefix:
    """Forward map through recovered layers 0..L-1. `layers` is a list of
    (W, b): W shape (n_out, n_in) signed & gauge-scaled, b shape (n_out,).
    Empty list => identity (layer 0)."""

    def __init__(self, layers, alpha):
        self.layers = layers
        self.alpha = alpha

    def forward(self, X):
        """True (nonlinear) forward through recovered layers -> layer-L input."""
        h = np.asarray(X, dtype=np.float64)
        for (W, b) in self.layers:
            z = h @ W.T + b
            h = np.where(z >= 0, z, self.alpha * z)
        return h

    def forward_around(self, X):
        """Linearised forward: activation slope frozen at row 0's sign pattern
        (the dual point's trajectory). Mirrors recover_weights.relu_around."""
        h = np.asarray(X, dtype=np.float64)
        for (W, b) in self.layers:
            z = h @ W.T + b
            on = (z[0] >= 0).astype(np.float64)           # anchor on the dual
            slope = on + self.alpha * (1.0 - on)
            h = z * slope
        return h

    @property
    def out_dim(self):
        return self.layers[-1][0].shape[0] if self.layers else IDIM


# --------------------------------------------------------------------------- #
# Pure-math helpers (identical to recover_weights).
# --------------------------------------------------------------------------- #
def _intersect(left, right, nl, nr):
    A = np.vstack((nl, nr))
    b = np.array([np.dot(nl, left), np.dot(nr, right)])
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    N = scipy.linalg.null_space(A, 1e-5)
    return x0, N


def _gen(x0, N, num):
    rv = np.random.randn(N.shape[1], num)
    return (x0[:, None] + N @ rv).T


# --------------------------------------------------------------------------- #
# Boundary normals (argmax-only) for every triplet endpoint, batched.
# --------------------------------------------------------------------------- #
def precompute_normals(o, triplets, chunk=256):
    """Returns NL, NR arrays (n,IDIM) and a `valid` mask. Boundary normals at
    each triplet's left and right, via batched argmax finite-difference."""
    n = len(triplets)
    L = np.stack([t[0] for t in triplets])
    R = np.stack([t[2] for t in triplets])
    NL = np.zeros((n, IDIM)); NR = np.zeros((n, IDIM))
    vL = np.zeros(n, bool); vR = np.zeros(n, bool)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        nl, ok_l = bfd._normal_batch(o, L[s:e])
        nr, ok_r = bfd._normal_batch(o, R[s:e])
        NL[s:e] = nl; NR[s:e] = nr; vL[s:e] = ok_l; vR[s:e] = ok_r
    return NL, NR, (vL & vR)


# --------------------------------------------------------------------------- #
# SVD recovery of one neuron from a set of (already-consistent) triplets.
# --------------------------------------------------------------------------- #
def recover_one(idxs, triplets, NL, NR, prefix, samples_per=16, cap=120):
    samples = []
    for i in idxs[:cap]:
        left, x0, right = triplets[i]
        nl, nr = NL[i], NR[i]
        x0p, N = _intersect(np.asarray(left), np.asarray(right), nl, nr)
        if N.shape[1] == 0:
            continue
        pts = _gen(np.asarray(x0), N, samples_per)
        pts = np.concatenate(([np.asarray(x0)], pts), 0)
        ha = prefix.forward_around(pts)
        samples.append(ha)
    if not samples:
        return None, None
    S = np.concatenate(samples, 0)
    S = S - S.mean(0)
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    w = Vt[-1]
    nrm = np.linalg.norm(w)
    if nrm < 1e-12:
        return None, None
    return w / nrm, sv


def _samples_for(i, triplets, NL, NR, prefix, samples_per=16):
    """Subspace samples for triplet i, pushed through the (linearised) prefix.
    Used both for recovery SVD and the pairwise consistency test."""
    left, x0, right = triplets[i]
    x0p, N = _intersect(np.asarray(left), np.asarray(right), NL[i], NR[i])
    if N.shape[1] == 0:
        return None
    pts = _gen(np.asarray(x0), N, samples_per)
    pts = np.concatenate(([np.asarray(x0)], pts), 0)
    return prefix.forward_around(pts)


def _consistency(sa, sb):
    """Normalised smallest singular value of the stacked, centred samples of two
    triplets. ~1e-6 if they lie on the same neuron's hyperplane (under this
    prefix), ~1e-1 otherwise. Pure numpy, no oracle."""
    S = np.concatenate([sa, sb], 0)
    S = S - S.mean(0)
    sv = np.linalg.svd(S, compute_uv=False)
    return sv[-1] / sv[0]


# --------------------------------------------------------------------------- #
# Cluster duals into layer-L neurons and recover each (black-box).
# --------------------------------------------------------------------------- #
def recover_layer(o, triplets, NL, NR, valid, prefix, n_neurons, alpha,
                  min_cluster=8, thresh=1e-4, max_seed_tries=600, verbose=True):
    """cluster_slow-style seed-and-grow on the SVD-consistency test.

    For a seed triplet, every other still-unassigned triplet that is *consistent*
    (shares the seed's hyperplane in prefix-output space, normalised smallest SV
    < thresh) is gathered into one neuron's cluster, then its direction is
    recovered by SVD. Duals belonging to DEEPER layers are not consistent with
    any layer-L hyperplane under this prefix, so they stay unassigned and are
    peeled at the next layer.

    Returns (clusters, unassigned) where each cluster is
    {w: (dimL,) unit dir, b_center: float, idxs: [...]}.
    """
    usable = [i for i in range(len(triplets)) if valid[i]]
    samp = {}
    for i in usable:
        s = _samples_for(i, triplets, NL, NR, prefix)
        if s is not None:
            samp[i] = s
    usable = [i for i in usable if i in samp]

    # layer-L input (nonlinear forward) per usable dual, for bias offsets
    Hin = prefix.forward(np.stack([np.asarray(triplets[i][1]) for i in usable]))
    hmap = {i: Hin[k] for k, i in enumerate(usable)}

    unassigned = set(usable)
    pool = list(usable)
    rng = np.random.default_rng(0)
    rng.shuffle(pool)

    clusters = []
    tries = 0
    for seed in pool:
        if len(clusters) >= n_neurons:
            break
        if seed not in unassigned or tries >= max_seed_tries:
            continue
        tries += 1
        sa = samp[seed]
        members = [j for j in unassigned if _consistency(sa, samp[j]) < thresh]
        if len(members) < min_cluster:
            unassigned.discard(seed)        # bad/lonely seed; don't retry it
            continue
        w, _ = recover_one(members, triplets, NL, NR, prefix)
        if w is None:
            unassigned.discard(seed)
            continue
        if any(abs(np.dot(w, c['w'])) > 0.985 for c in clusters):
            unassigned -= set(members)      # duplicate neuron; consume members
            continue
        b_center = float(np.median([np.dot(w, hmap[i]) for i in members]))
        clusters.append({'w': w, 'b_center': b_center, 'idxs': members})
        unassigned -= set(members)
        if verbose:
            print(f"    neuron {len(clusters)-1}: |cluster|={len(members)}", flush=True)

    if verbose:
        print(f"    recovered {len(clusters)} neurons, {len(unassigned)} duals unassigned")
    return clusters, sorted(unassigned)
