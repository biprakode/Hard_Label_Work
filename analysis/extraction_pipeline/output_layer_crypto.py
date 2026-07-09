"""
Cryptanalytic recovery of the output layer (fc5) in the hard-label setting.

Ported from the reference implementation of Canales-Martinez & Santos,
"Extracting Some Layers of Deep Neural Networks in the Hard-Label Setting"
(ePrint 2025/1118), Section 3 (multi-output / softmax case, Sec 3.3). The
reference source lives at
    enhanced_codebase/FC5_LR_fit_source/hard-label-contract-output/outputLayer.py
and is TensorFlow/Keras + argv-driven; here we port its
collect -> build -> solve -> sign-check chain to our torch pipeline and match
the in-place contract of `output_layer_recovery.recover_output_layer` (writes
`fc5.weight` (out_dim, d_r) and `fc5.bias` (out_dim,) as float64).

Idea (Sec 3): at an input x where two classes i, j tie, their raw output
scores are equal,
        A_i . y + b_i = A_j . y + b_j ,   y = f_{1..r}(x)   (penultimate activation)
so each tie point yields ONE linear equation in the output-layer parameters
(A, b). Collecting enough linearly-independent equations lets us SOLVE (not fit)
for fc5, up to the softmax/argmax gauge freedom.

Gauge: the output layer is determined only up to  A_i -> s (A_i + w0),
b_i -> s (b_i + b0)  for any shared row w0 in R^{d_r}, scalar b0, and scale s>0
(add-affine-to-all-outputs: d_r+1 dof; positive scale: 1 dof => rank deficiency
d_r+2). Following the reference we pin output-0's weights+bias to 0 and
output-1's first weight to 1, then least-squares solve the reduced full-rank
system. Recovering an EQUIVALENT network (identical argmax everywhere) is the
success criterion, not literal parameter equality.

Hard-label throughout: the oracle is queried by argmax only; no logits are read.
"""

import time

import numpy as np
import torch

from .bias_recovery import _hidden_activations_up_to


# --------------------------------------------------------------------------- #
#  Hard-label oracle                                                           #
# --------------------------------------------------------------------------- #
def _make_oracle(oracle_model, input_dim):
    """
    Return `oracle(x_np) -> np.ndarray[int]` that queries `oracle_model` with
    ARGMAX ONLY (ref outputLayer.py:416, `func = lambda x: np.argmax(model.predict(x))`).

    `x_np` may be shape (input_dim,) or (n, input_dim); result is shape (n,).
    No softmax / logit value ever leaves this function -> hard-label contract.
    """
    oracle_model.eval()

    def oracle(x_np):
        x = np.asarray(x_np, dtype=np.float64).reshape(-1, input_dim)
        xt = torch.from_numpy(x)
        with torch.no_grad():
            logits = oracle_model(xt)
            labels = logits.argmax(dim=1).cpu().numpy()   # <-- argmax only
        return labels

    return oracle


# --------------------------------------------------------------------------- #
#  Transition-point search (fresh, hard-label)                                 #
# --------------------------------------------------------------------------- #
def _random_point(input_dim, input_range, rng):
    return rng.uniform(-input_range, input_range, size=(input_dim,))


def _bisect_boundary(oracle, x1, c1, x2, tol=1e-13, ntries=50):
    """
    Bisect between x1 (oracle class c1) and x2 (a different class) down to the
    class-tie surface (port of aux_functions.findTransitionPoint's inner loop).
    Returns (point, class_a, class_b) with class_a != class_b, or None.

    IMPORTANT: x2 may drift to a THIRD class during bisection (the `else`
    branch), so the two adjacent classes are read FRESH from the endpoints at
    the end — mislabelling the pair corrupts the equation. Hard-label only.
    """
    x1 = x1.copy()
    x2 = x2.copy()
    precision = np.linalg.norm(x2 - x1)
    precision_old = precision + 1.0
    tries = 0
    while precision > tol:
        if precision_old == precision:
            tries += 1
            if tries > ntries:
                break                # converged as far as float64 allows
        else:
            tries = 0
        xnew = (x1 + x2) / 2.0
        cnew = int(oracle(xnew)[0])
        if cnew == c1:
            x1 = xnew
        else:                        # cnew == c2 or a third class: shrink toward x1
            x2 = xnew
        precision_old = precision
        precision = np.linalg.norm(x2 - x1)

    a = int(oracle(x1)[0])
    b = int(oracle(x2)[0])
    if a == b:
        return None                  # boundary collapsed to one class; discard
    return x1, a, b


# --------------------------------------------------------------------------- #
#  Linear-system construction / solve                                         #
# --------------------------------------------------------------------------- #
def _h4_augmented(reconstructed_model, points_np, input_dim):
    """
    Forward raw input points through the reconstructed fc1..fc4 (activation via
    `_act`, so ReLU / LeakyReLU are both handled) to get penultimate y=f_{1..r},
    then append a ones column to absorb the bias. Returns (n, d_r+1) float64.
    """
    x = torch.from_numpy(np.asarray(points_np, dtype=np.float64).reshape(-1, input_dim))
    with torch.no_grad():
        h4 = _hidden_activations_up_to(reconstructed_model, x, up_to_layer=4).cpu().numpy()
    ones = np.ones((h4.shape[0], 1), dtype=np.float64)
    return np.hstack((h4, ones))


def _equation_rows(h4_aug, class_a, class_b, d_r, n_outputs):
    """
    Encode tie equations  [y|1] placed +in block(a), -in block(b)  into rows of
    width (d_r+1)*n_outputs (ref buildSystemOfEquations, outputLayer.py:144-152).
    class_a is the reference (+) class, class_b the second (-) class.
    """
    width = (d_r + 1) * n_outputs
    n = h4_aug.shape[0]
    rows = np.zeros((n, width), dtype=np.float64)
    a0, a1 = class_a * (d_r + 1), (class_a + 1) * (d_r + 1)
    b0, b1 = class_b * (d_r + 1), (class_b + 1) * (d_r + 1)
    rows[:, a0:a1] = h4_aug
    rows[:, b0:b1] = -h4_aug
    return rows


def _shared_row_basis(d_r, n_outputs):
    """
    Basis of the additive gauge subspace: adding a shared affine row g in
    R^{d_r+1} to EVERY output block, [g,g,...,g], leaves all tie equations
    invariant (row . [g..g] = [+y|1].g + [-y|1].g = 0). Returns (width, d_r+1).
    """
    width = (d_r + 1) * n_outputs
    G = np.zeros((width, d_r + 1), dtype=np.float64)
    for j in range(d_r + 1):
        for c in range(n_outputs):
            G[c * (d_r + 1) + j, j] = 1.0
    return G


def _solve_system(big_matrix, d_r, n_outputs):
    """
    Recover the output-layer parameter vector P (per-class [W_c | b_c], stacked)
    from the homogeneous tie system  big_matrix @ P = 0.

    The true parameters P_true satisfy the tie equations exactly, so P_true lies
    in null(big_matrix). At full rank that null space is exactly
    span{P_true} (+) shared-affine-row gauge (d_r+1 dims) + scale, dimension
    d_r+2. We therefore take the null space (small right singular vectors) and
    extract the component ORTHOGONAL to the shared-affine-row subspace -> the
    P_true direction, recovered up to scale and global sign (both immaterial to
    argmax; the sign is fixed downstream by a functional vote).

    This replaces the reference's fragile heuristic pin (output-1 first weight
    := 1, outputLayer.py:183-192), which blows up whenever that particular
    weight is ~0 in the gauge-fixed truth. Returns (W (n_outputs, d_r),
    b (n_outputs,)).
    """
    # Restrict the search to the orthogonal complement of the shared-affine-row
    # gauge, then take the minimum-residual direction there. Concretely: pick an
    # orthonormal basis B of gauge^perp, and solve  min_||u||=1 ||C (B u)||  via
    # the smallest right singular vector of C B. v = B u is then automatically
    # orthogonal to the gauge and, at full rank, is exactly the P_true direction.
    # This is numerically robust (no explicit null-space dimensioning).
    G = _shared_row_basis(d_r, n_outputs)                 # (width, d_r+1)
    Ug, sg, _ = np.linalg.svd(G, full_matrices=True)      # Ug: (width, width)
    rg = int((sg > max(G.shape) * sg[0] * 1e-12).sum())   # = d_r+1
    B = Ug[:, rg:]                                         # (width, width-rg) gauge^perp

    AB = big_matrix @ B                                    # (m, width-rg)
    # full_matrices=True is REQUIRED: when m < (width-rg) the min-residual
    # direction is the (width-rg)-th right singular vector, which the economy
    # SVD would drop -> we would miss the true null direction.
    _, _, VtAB = np.linalg.svd(AB, full_matrices=True)
    v = B @ VtAB[-1]                                       # min-residual, gauge-orthogonal

    P = v.reshape((n_outputs, d_r + 1))
    P[np.abs(P) < 1e-10] = 0.0            # numerical dust
    W = P[:, :d_r]                        # (n_outputs, d_r)  == fc5.weight layout
    b = P[:, d_r]                         # (n_outputs,)
    return W, b


# --------------------------------------------------------------------------- #
#  Collection driver + orchestration                                          #
# --------------------------------------------------------------------------- #
def _build_seed_pool(oracle, input_dim, input_range, rng, n_outputs, max_samples):
    """Random-sample inputs, bucket by oracle class -> {class: [points]}."""
    seeds = {}
    for _ in range(max_samples):
        x = _random_point(input_dim, input_range, rng)
        c = int(oracle(x)[0])
        seeds.setdefault(c, []).append(x)
        # stop early once we have a healthy pool spanning many classes
        if len(seeds) >= n_outputs and sum(len(v) for v in seeds.values()) >= 4 * n_outputs:
            break
    return seeds


def _collect_and_solve(oracle, reconstructed_model, input_dim, d_r, n_outputs,
                       rank_target, budget, input_range, rng, verbose):
    """
    Collect class-tie transition points and keep only those that raise the rank
    of the stacked equation system (ref unitePartialSolutions, outputLayer.py:
    51-57). Stop at rank_target or after `budget` searches.

    Rather than bisecting two uniformly-random points (which rarely reaches
    boundaries between rare class pairs, so the rank saturates below target), we
    keep a per-class SEED POOL and deliberately bisect between seeds of two
    chosen classes. This targets diverse pairs and reaches full rank reliably.
    Returns (W, b, achieved_rank, n_searches, n_kept).
    """
    width = (d_r + 1) * n_outputs
    kept_rows = []         # actual independent equation rows (for the final solve)
    K = np.zeros((0, width), dtype=np.float64)
    Q = np.zeros((0, width), dtype=np.float64)   # orthonormal basis of kept_rows
    rank = 0
    searches = 0
    kept = 0
    t0 = time.time()
    # GS residual is only a cheap PRE-FILTER (its incremental error over-counts
    # rank, causing premature "full rank" stops). A candidate that passes it is
    # confirmed with an exact matrix_rank before being accepted, so `rank` is
    # honest and collection keeps searching toward the true reachable rank.
    gs_pre = 1e-6

    seeds = _build_seed_pool(oracle, input_dim, input_range, rng, n_outputs,
                             max_samples=max(400, 40 * n_outputs))

    # cap consecutive rank-stalls once we clearly can't grow further (the class
    # adjacency graph may be tree-like -> some scale dofs are unobservable in
    # hard-label, so the reachable rank is genuinely < rank_target).
    stall = 0
    stall_cap = max(2000, 30 * rank_target)

    while rank < rank_target and searches < budget and stall < stall_cap:
        searches += 1
        if searches % 20 == 0:        # inject fresh seeds to widen coverage
            x = _random_point(input_dim, input_range, rng)
            seeds.setdefault(int(oracle(x)[0]), []).append(x)

        classes = [c for c, v in seeds.items() if v]
        if len(classes) < 2:
            continue
        ci, cj = (classes[k] for k in rng.choice(len(classes), 2, replace=False))
        x1 = seeds[ci][rng.integers(len(seeds[ci]))]
        x2 = seeds[cj][rng.integers(len(seeds[cj]))]
        res = _bisect_boundary(oracle, x1, ci, x2)
        if res is None:
            continue
        pt, ref_c, sec_c = res        # actual adjacent classes at the boundary
        h4_aug = _h4_augmented(reconstructed_model, pt.reshape(1, -1), input_dim)
        row = _equation_rows(h4_aug, ref_c, sec_c, d_r, n_outputs)[0]  # (width,)

        # (1) cheap GS pre-filter: reject rows clearly in the current span.
        nrm0 = np.linalg.norm(row)
        if nrm0 < 1e-12:
            stall += 1
            continue
        r = row.copy()
        if Q.shape[0]:
            r = r - Q.T @ (Q @ r)
            r = r - Q.T @ (Q @ r)
        if np.linalg.norm(r) <= gs_pre * nrm0:
            stall += 1
            continue
        # (2) confirm with exact rank (only reached by GS-positive candidates).
        trial = np.vstack((K, row))
        if np.linalg.matrix_rank(trial) > rank:
            K = trial
            Q = np.vstack((Q, r / np.linalg.norm(r)))
            kept_rows.append(row)
            rank += 1
            kept += 1
            stall = 0
            if verbose and (kept % 25 == 0 or rank == rank_target):
                print(f"    [fc5-crypto] rank {rank}/{rank_target} "
                      f"({kept} eqs kept / {searches} searches, {time.time()-t0:.1f}s)")
        else:
            stall += 1

    if not kept_rows:
        raise RuntimeError("fc5-crypto: no transition points found; check input_range / oracle")

    W, b = _solve_system(np.array(kept_rows), d_r, n_outputs)
    return W, b, rank, searches, kept


def _fix_global_sign(oracle, reconstructed_model, W, b, input_dim, input_range,
                     rng, n_votes=11):
    """
    The gauge scale s>0 is fixed, but a residual global sign can still flip the
    argmax. Majority-vote over random points: if the recovered network disagrees
    with the oracle on most, negate (W, b) (ref outputLayer.py:202-213, but
    voted instead of single-sample).
    """
    disagree = 0
    for _ in range(n_votes):
        x = _random_point(input_dim, input_range, rng)
        true_c = int(oracle(x)[0])
        h4_aug = _h4_augmented(reconstructed_model, x.reshape(1, -1), input_dim)  # (1, d_r+1)
        logits = h4_aug[:, :-1] @ W.T + b        # (1, n_outputs)
        if int(np.argmax(logits[0])) != true_c:
            disagree += 1
    if disagree > n_votes // 2:
        return -W, -b, True
    return W, b, False


def recover_output_layer_cryptanalytic(reconstructed_model, oracle_model,
                                        input_dim=None, input_range=1.0,
                                        budget_mult=50, seed=0, verbose=True):
    """
    Cryptanalytically recover fc5 (ePrint 2025/1118, Sec 3) and write it into
    `reconstructed_model.fc5` in place (same contract as
    output_layer_recovery.recover_output_layer). Hidden layers fc1..fc4 must be
    already sign-recovered; they are read (for h_4) but never modified.

    Parameters
    ----------
    reconstructed_model : torch model with .fc1.. .fc5 (fc1..fc4 sign-recovered)
    oracle_model        : victim model, queried by argmax only
    input_dim           : input dimension; defaults to fc1.in_features
    input_range         : sampling half-width for transition-point search
    budget_mult         : query-budget cap = budget_mult * rank_target searches
    """
    reconstructed_model.eval()
    if input_dim is None:
        input_dim = reconstructed_model.fc1.in_features
    fc5 = reconstructed_model.fc5
    n_outputs = fc5.out_features           # d_{r+1}
    d_r = fc5.in_features                  # penultimate width
    rank_target = (d_r + 1) * (n_outputs - 1) - 1     # = d_{r+1}(d_r+1) - (d_r+2)
    budget = budget_mult * max(rank_target, 1)
    rng = np.random.default_rng(seed)

    oracle = _make_oracle(oracle_model, input_dim)

    if verbose:
        print(f"  [fc5-crypto] d_r={d_r} n_outputs={n_outputs} "
              f"rank_target={rank_target} budget={budget} (=<{budget_mult}x)")

    t0 = time.time()
    W, b, rank, searches, kept = _collect_and_solve(
        oracle, reconstructed_model, input_dim, d_r, n_outputs,
        rank_target, budget, input_range, rng, verbose)
    W, b, flipped = _fix_global_sign(
        oracle, reconstructed_model, W, b, input_dim, input_range, rng)

    with torch.no_grad():
        fc5.weight.data = torch.tensor(W, dtype=torch.float64)
        fc5.bias.data = torch.tensor(b, dtype=torch.float64)

    if verbose:
        # functional-equivalence sanity over fresh random points
        n_test, ok = 2000, 0
        for _ in range(n_test):
            x = _random_point(input_dim, input_range, rng)
            true_c = int(oracle(x)[0])
            h4_aug = _h4_augmented(reconstructed_model, x.reshape(1, -1), input_dim)
            pred_c = int(np.argmax(h4_aug[:, :-1] @ W.T + b))
            ok += (pred_c == true_c)
        status = "FULL RANK" if rank >= rank_target else "UNDER-DETERMINED (budget hit)"
        print(f"  [fc5-crypto] {status}: rank {rank}/{rank_target}, "
              f"{kept} eqs / {searches} searches, sign_flip={flipped}, "
              f"random-point agreement={ok/n_test:.4f}, {time.time()-t0:.1f}s")

    return {
        'rank': int(rank), 'rank_target': int(rank_target),
        'searches': int(searches), 'equations_kept': int(kept),
        'full_rank': bool(rank >= rank_target), 'sign_flipped': bool(flipped),
        'seconds': float(time.time() - t0),
    }
