"""
Improved evaluation-metric suite for hard-label DNN extraction.

Replaces the single "naive prediction agreement" number with the multi-metric
scorecard prescribed by ``Evaluation_Metric_Improve/evaluation_metrics_REPORT.md``
(the source-of-truth spec). Each public function maps to a Metric in that report:

    fidelity                       -> Metric 1  (in-distribution fidelity + accuracy)
    boundary_distance_bisection    -> Metric 2  (hard-label margin proxy, argmax-only)
    margin_conditioned_fidelity    -> Metric 2  (fidelity stratified by margin bin)
    off_distribution_agreement     -> Metric 3  (uniform / wide-Gaussian agreement)
    interpolation_path_agreement   -> Metric 3  (class-pair interpolation agreement)
    paired_mcnemar                 -> Metric 4  (single-run significance of the gap)
    structural_metrics             -> Metric 5  (|cos|, sign-acc, coverage; known-victim)
    compute_eqs                    -> Deliverable B (composite 0-100 Extraction-Quality Score)

All black-box metrics use ONLY the oracle argmax (``model(X).argmax(1)``), matching
the strict hard-label setting. Structural metrics need ground-truth weights and are
therefore computable only on the known-victim (make_blobs / tiny) tiers — exactly
what those tiers exist for.

Deferred (documented hooks below, not implemented in this first version per the
agreed scope):
  * HopSkipJump / RayS boundary co-location + adversarial transferability (Metric 3.2/3.3)
  * Full N>=10-seed retraining significance harness (Metric 4); single-run McNemar stands in
  * Liu-style coverage (Metric 5); recovered-fraction proxy stands in
"""

import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  Oracle helper                                                              #
# --------------------------------------------------------------------------- #

def predict(model, X, batch_size=4096):
    """Hard-label oracle: return argmax class indices (np.int64) for inputs X.

    X may be a torch.Tensor or np.ndarray. Batched so large off-distribution
    pools (CIFAR, 3072-dim) do not blow memory.
    """
    model.eval()
    if not torch.is_tensor(X):
        X = torch.as_tensor(X)
    X = X.double()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out = model(X[i:i + batch_size])
            preds.append(out.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)


# --------------------------------------------------------------------------- #
#  Metric 1 — In-distribution fidelity (+ accuracy)                           #
# --------------------------------------------------------------------------- #

def fidelity(model_preds, victim_preds, true_labels=None):
    """Metric 1.

    fidelity = mean 1[ model_pred == victim_pred ]   (agreement with victim argmax)
    accuracy = mean 1[ model_pred == true_label ]    (vs ground truth, if provided)

    Returns dict. Report BOTH fidelity and accuracy (report §2 Metric 1 caveat ii):
    the *difference* in how extraction vs distillation move these two is informative.
    """
    model_preds = np.asarray(model_preds)
    victim_preds = np.asarray(victim_preds)
    out = {
        'fidelity': float(np.mean(model_preds == victim_preds)),
        'n': int(len(model_preds)),
    }
    if true_labels is not None:
        true_labels = np.asarray(true_labels)
        out['accuracy'] = float(np.mean(model_preds == true_labels))
    return out


# --------------------------------------------------------------------------- #
#  Metric 2 — Margin-conditioned fidelity (hard-label margin proxy)           #
# --------------------------------------------------------------------------- #

def boundary_distance_bisection(victim, X, n_dirs=8, max_r=2.0, n_bisect=12,
                                seed=0, batch_size=4096):
    """Cheap, argmax-only hard-label margin proxy (report §2 Metric 2).

    For each point x we estimate the distance to the victim's decision boundary as
    the smallest radius r along a few random unit directions at which the victim's
    argmax first changes from its label at x. This is the lightweight stand-in for
    HopSkipJump/RayS: fixed, tiny query budget (n_dirs * n_bisect queries/point),
    enough to *bin* points into near/mid/far — not to estimate r precisely.

    Returns r(x) as an np.ndarray of shape (len(X),). Points whose label never flips
    within max_r along any sampled direction get r = max_r (treated as "far").
    """
    rng = np.random.default_rng(seed)
    if torch.is_tensor(X):
        X = X.double().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    base = predict(victim, X, batch_size=batch_size)              # label at x
    # Per-point running best (smallest flip radius found so far).
    r_best = np.full(n, np.inf, dtype=np.float64)

    for _ in range(n_dirs):
        # Random unit direction per point (independent directions => robust proxy).
        dirs = rng.standard_normal((n, d))
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

        lo = np.zeros(n)                          # known: argmax == base
        hi = np.full(n, max_r)                    # may or may not have flipped
        # Identify which points actually flip by max_r along this direction.
        far = predict(victim, X + max_r * dirs, batch_size=batch_size)
        flips = far != base                       # only bisect those that flip
        # Bisection on the flipping subset.
        for _ in range(n_bisect):
            mid = (lo + hi) / 2.0
            pm = predict(victim, X + mid[:, None] * dirs, batch_size=batch_size)
            flipped_now = (pm != base) & flips
            # If flipped at mid -> boundary is closer (hi = mid); else lo = mid.
            hi = np.where(flipped_now, mid, hi)
            lo = np.where(flipped_now, lo, mid)
        # For flipping points, hi approximates the flip radius.
        r_dir = np.where(flips, hi, np.inf)
        r_best = np.minimum(r_best, r_dir)

    r_best[~np.isfinite(r_best)] = max_r
    return r_best


def margin_conditioned_fidelity(victim_preds, model_preds, r, n_bins=3,
                                bin_edges=None):
    """Metric 2: fidelity stratified by victim decision margin (boundary distance r).

    Bins points into tertiles of r by default ("near" = small r = brittle victim,
    "far" = large r = stable victim). Report §2: extraction's advantage should be
    largest in the mid/near bins. Returns per-bin fidelity + the edges used.
    """
    victim_preds = np.asarray(victim_preds)
    model_preds = np.asarray(model_preds)
    r = np.asarray(r, dtype=np.float64)

    if bin_edges is None:
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(r, qs)
        # Boundary-distance proxies often clip a large mass at max_r (points that
        # never flip = high margin), making interior quantiles coincide and
        # collapsing a tertile to n=0. Dedupe interior edges so every reported bin
        # is non-empty; relabel by rank when fewer than n_bins survive.
        interior = np.unique(edges[1:-1])
        bin_edges = np.concatenate([[-np.inf], interior, [np.inf]])
    else:
        bin_edges = np.asarray(bin_edges, dtype=np.float64)

    eff_bins = len(bin_edges) - 1
    if eff_bins == 3:
        labels = ['near', 'mid', 'far']
    elif eff_bins == 2:
        labels = ['near', 'far']
    elif eff_bins == 1:
        labels = ['all']
    else:
        labels = [f'bin{i}' for i in range(eff_bins)]

    bins = {}
    idx = np.digitize(r, bin_edges[1:-1], right=False)  # 0..eff_bins-1
    for b in range(eff_bins):
        m = idx == b
        name = labels[b] if b < len(labels) else f'bin{b}'
        bins[name] = {
            'n': int(m.sum()),
            'r_lo': float(bin_edges[b]) if np.isfinite(bin_edges[b]) else None,
            'r_hi': float(bin_edges[b + 1]) if np.isfinite(bin_edges[b + 1]) else None,
            'fidelity': float(np.mean(model_preds[m] == victim_preds[m])) if m.any() else None,
        }
    return {'bins': bins, 'bin_edges': [float(e) for e in bin_edges]}


# --------------------------------------------------------------------------- #
#  Metric 3 — Off-distribution & boundary agreement (the discriminator)       #
# --------------------------------------------------------------------------- #

def input_bounds(X, pad=0.0):
    """Per-dimension (min, max) of a reference set, optionally padded. Used to
    define a sampling box that matches the victim's actual input scale across
    tiers (CIFAR [-1,1], make_blobs raw float ranges)."""
    if torch.is_tensor(X):
        X = X.double().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    if pad:
        span = hi - lo
        lo = lo - pad * span
        hi = hi + pad * span
    return lo, hi


def off_distribution_agreement(victim, model, ref_X, n=5000, seed=0,
                               gaussian_scale=3.0, batch_size=4096):
    """Metric 3.1: agreement off the data manifold.

    Two pools, both argmax-only:
      * uniform     — U(lo, hi) over the per-dim input box of ref_X
      * wide_gauss  — N(mean, (gaussian_scale * std)^2), a fatter spread than the data
    A distilled copy's agreement decays off-manifold; a frozen-row extraction holds up.
    Returns agreement (victim vs model) for each pool.
    """
    rng = np.random.default_rng(seed)
    if torch.is_tensor(ref_X):
        ref_X = ref_X.double().cpu().numpy()
    ref_X = np.asarray(ref_X, dtype=np.float64)
    d = ref_X.shape[1]
    lo, hi = input_bounds(ref_X)
    mu = ref_X.mean(axis=0)
    sd = ref_X.std(axis=0) + 1e-12

    out = {}
    pools = {
        'uniform': lo + (hi - lo) * rng.random((n, d)),
        'wide_gauss': mu + gaussian_scale * sd * rng.standard_normal((n, d)),
    }
    for name, P in pools.items():
        vp = predict(victim, P, batch_size=batch_size)
        mp = predict(model, P, batch_size=batch_size)
        out[name] = {'n': int(n), 'agreement': float(np.mean(vp == mp))}
    out['mean_agreement'] = float(np.mean([out[k]['agreement'] for k in pools]))
    return out


def interpolation_path_agreement(victim, model, X, Y, n_pairs=200, n_steps=20,
                                 seed=0, batch_size=4096):
    """Metric 3.4: agreement along straight lines between opposite-class examples.

    Picks pairs of points from different victim-predicted classes and walks the
    segment between them (where a boundary crossing lives). A true parameter copy
    matches the *crossing location*; a distilled copy may not. Argmax-only.
    Returns mean per-path agreement (fraction of interior path points where
    victim and model argmax agree).
    """
    rng = np.random.default_rng(seed)
    if torch.is_tensor(X):
        X = X.double().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    vlabels = predict(victim, X, batch_size=batch_size)

    classes = np.unique(vlabels)
    if len(classes) < 2:
        return {'n_pairs': 0, 'mean_path_agreement': None}

    # Build endpoints from distinct victim-predicted classes.
    a_idx, b_idx = [], []
    attempts = 0
    while len(a_idx) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        i, j = rng.integers(0, len(X), size=2)
        if vlabels[i] != vlabels[j]:
            a_idx.append(i)
            b_idx.append(j)
    if not a_idx:
        return {'n_pairs': 0, 'mean_path_agreement': None}

    a = X[a_idx]
    b = X[b_idx]
    ts = np.linspace(0.0, 1.0, n_steps + 2)[1:-1]    # interior points only
    pts = np.concatenate([(1 - t) * a + t * b for t in ts], axis=0)
    vp = predict(victim, pts, batch_size=batch_size)
    mp = predict(model, pts, batch_size=batch_size)
    return {
        'n_pairs': int(len(a_idx)),
        'n_steps': int(n_steps),
        'mean_path_agreement': float(np.mean(vp == mp)),
    }


# ---- Deferred Metric 3 hooks (HopSkipJump boundary co-location / adv transfer) ----

def boundary_colocation_agreement(*args, **kwargs):           # noqa: D401
    """Metric 3.2 — DEFERRED. Land points on the *victim's* boundary via
    HopSkipJump/RayS, then test whether the model flips at the same place.
    Requires a query-heavy hard-label boundary attack (ART/foolbox). Left as a
    hook so the scorecard structure matches the report."""
    raise NotImplementedError("Metric 3.2 boundary co-location is deferred "
                              "(needs HopSkipJump/RayS).")


def adversarial_transferability(*args, **kwargs):             # noqa: D401
    """Metric 3.3 — DEFERRED. Craft hard-label adversarial examples on the victim,
    measure mutual transfer to the model. Deferred with 3.2."""
    raise NotImplementedError("Metric 3.3 adversarial transfer is deferred.")


# --------------------------------------------------------------------------- #
#  Metric 4 — Significance of the gap (single-run; N-seed harness deferred)    #
# --------------------------------------------------------------------------- #

def paired_mcnemar(victim_preds, ext_preds, dis_preds, n_boot=10000, seed=0):
    """Metric 4 (single-run stand-in).

    Paired comparison of two reconstructions against the victim reference on the
    SAME held-out inputs (report §2 Metric 4 step 3). Builds the 2x2 table of
    per-sample agreement and applies the continuity-corrected McNemar test:

        b = #(extraction agrees, distillation disagrees)
        c = #(extraction disagrees, distillation agrees)
        chi2 = (|b - c| - 1)^2 / (b + c),  1 dof.

    Also reports a bootstrap 95% CI on the fidelity *gap* (ext_fid - dis_fid) by
    resampling test indices. Hard-label: operates purely on argmax agreements.

    NOTE: the full report Metric 4 also wants N>=10 retrained seeds. That harness
    is DEFERRED (see module docstring / run_seed_significance hook). This single
    run quantifies the within-test-set significance only.
    """
    from scipy.stats import chi2 as _chi2

    victim_preds = np.asarray(victim_preds)
    ext_ok = np.asarray(ext_preds) == victim_preds
    dis_ok = np.asarray(dis_preds) == victim_preds

    b = int(np.sum(ext_ok & ~dis_ok))
    c = int(np.sum(~ext_ok & dis_ok))
    n_disc = b + c
    if n_disc == 0:
        chi2_stat, p_value = 0.0, 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / n_disc
        chi2_stat = max(chi2_stat, 0.0)
        p_value = float(_chi2.sf(chi2_stat, df=1))

    ext_fid = float(np.mean(ext_ok))
    dis_fid = float(np.mean(dis_ok))
    gap = ext_fid - dis_fid

    # Bootstrap CI on the gap.
    rng = np.random.default_rng(seed)
    n = len(victim_preds)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(ext_ok[idx]) - np.mean(dis_ok[idx])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    return {
        'extraction_fidelity': ext_fid,
        'distillation_fidelity': dis_fid,
        'gap': float(gap),
        'mcnemar_b': b,
        'mcnemar_c': c,
        'mcnemar_chi2': float(chi2_stat),
        'mcnemar_p_value': p_value,
        'significant_at_0.05': bool(p_value < 0.05),
        'gap_bootstrap_ci95': [float(ci_lo), float(ci_hi)],
        'gap_over_sigma': float(gap / (np.std(boot) + 1e-12)),
        'note': 'single-run McNemar; N>=10-seed harness deferred',
    }


def run_seed_significance(*args, **kwargs):                   # noqa: D401
    """Metric 4 (full) — DEFERRED. Retrain N>=10 seeds of each arm, aggregate
    seed-level paired test + bootstrap CI on the gap. Needs N* full pipeline runs;
    out of compute scope for now. Hook kept so the harness can slot in later."""
    raise NotImplementedError("N-seed significance harness is deferred "
                              "(requires retraining both arms >=10 times).")


# --------------------------------------------------------------------------- #
#  Metric 5 — Parameter-level structural recovery (known-victim tier only)     #
# --------------------------------------------------------------------------- #

def structural_metrics(extraction_metrics):
    """Metric 5: surface the structural receipts already computed by the pipeline.

    Reads the ``extraction_metrics.json`` dict (the per-run metrics file written by
    ``workflow.main``): per-layer ``mean_abs_cosine_sim`` and ``sign_accuracy`` live
    in ``layer_metrics``; recovery coverage lives in ``recovery_stats``. These are
    the things a pure-distillation model provably lacks (|cos|~0, no signs).

    Coverage here = recovered/total fraction per layer (and overall). Full
    Liu-EUROCRYPT-2026 region coverage is a documented hook (compute_liu_coverage).
    Returns aggregated structural numbers in [0,1] for the EQS structural block.
    """
    if not extraction_metrics:
        return None
    lm = extraction_metrics.get('layer_metrics', {}) or {}
    rs = extraction_metrics.get('recovery_stats', {}) or {}

    per_layer = {}
    cos_vals, sign_vals = [], []
    for name, m in sorted(lm.items()):
        # Skip aggregate / per-neuron entries: '_all' includes random-init neurons
        # (would drag |cos| below 1.0 and double-count layers); '_per_neuron' is raw.
        if name.endswith('_all') or name.endswith('_per_neuron'):
            continue
        if not isinstance(m, dict) or 'sign_accuracy' not in m:
            continue
        cos = m.get('mean_abs_cosine_sim')
        sgn = m.get('sign_accuracy')
        per_layer[name] = {
            'mean_abs_cosine_sim': None if cos is None else float(cos),
            'sign_accuracy': None if sgn is None else float(sgn),
            'num_recovered': m.get('num_recovered'),
        }
        if cos is not None:
            cos_vals.append(float(cos))
        if sgn is not None:
            sign_vals.append(float(sgn))

    total = rs.get('total_neurons') or 0
    recovered = rs.get('recovered_neurons') or 0
    coverage = (recovered / total) if total else None

    mean_cos = float(np.mean(cos_vals)) if cos_vals else None
    mean_sign = float(np.mean(sign_vals)) if sign_vals else None

    parts = [v for v in (mean_cos, mean_sign, coverage) if v is not None]
    structural_score = float(np.mean(parts)) if parts else None

    return {
        'per_layer': per_layer,
        'mean_abs_cosine_sim': mean_cos,
        'mean_sign_accuracy': mean_sign,
        'coverage': None if coverage is None else float(coverage),
        'recovered_neurons': int(recovered),
        'total_neurons': int(total),
        'structural_score': structural_score,   # [0,1], feeds EQS_structural S block
    }


def compute_liu_coverage(*args, **kwargs):                   # noqa: D401
    """Metric 5 — DEFERRED. Liu et al. EUROCRYPT-2026 region coverage (fraction of
    the input region where all effective weights are recovered). Recovered-fraction
    proxy is used instead for now."""
    raise NotImplementedError("Liu-style region coverage is deferred.")


# --------------------------------------------------------------------------- #
#  Deliverable B — composite Extraction-Quality Score (EQS)                   #
# --------------------------------------------------------------------------- #

# C4 (gap significance) is dropped per agreed scope; remaining weights are
# renormalized to sum to 100 (scale factor 100/85 in both variants).
_EQS_WEIGHTS = {
    'blackbox':   {'C1': 25, 'C2': 30, 'C3': 20, 'C5': 10},          # sum 85
    'structural': {'C1': 22, 'C2': 26, 'C3': 17, 'S': 20},           # sum 85
}


def compute_eqs(components, variant='blackbox'):
    """Deliverable B: composite 0-100 Extraction-Quality Score (C4 renormalized out).

    ``components`` is a dict of normalized [0,1] inputs:
        C1 : in-distribution fidelity (query-disjoint set)
        C2 : off-distribution + boundary agreement (mean of available 3.x parts)
        C3 : high-margin-stratum fidelity (top margin tertile)
        C5 : query economy  = 1 - clip(queries_used / query_budget, 0, 1)   [blackbox]
        S  : structural recovery (structural_metrics['structural_score'])    [structural]

    Returns the scalar EQS (0-100) plus the weighted per-component profile so the
    report can show *where* the advantage lives (report §4.4).
    """
    weights = _EQS_WEIGHTS[variant]
    raw_sum = sum(weights.values())
    scale = 100.0 / raw_sum

    profile = {}
    eqs = 0.0
    for comp, w in weights.items():
        val = components.get(comp)
        if val is None:
            val = 0.0
        val = float(np.clip(val, 0.0, 1.0))
        contribution = w * scale * val
        profile[comp] = {
            'value': val,
            'weight_renormalized': float(w * scale),
            'contribution': float(contribution),
        }
        eqs += contribution

    return {
        'variant': variant,
        'eqs': float(eqs),
        'profile': profile,
    }
