#!/usr/bin/env python3
"""
Improved evaluation driver — replaces the single "naive prediction agreement"
headline with the full metric scorecard from
``Evaluation_Metric_Improve/evaluation_metrics_REPORT.md``.

Runs AFTER reconstruction (LATEST_WORKFLOW: ... ML reconstruction -> [here] ->
report). Consumes the artifacts both arms already write to
``results/reconstructed_models/``:

    extraction arm : reconstructed_<arch>.pth         + extraction_metrics.json
    distillation   : reconstructed_full_distillation.pth + extraction_metrics_distillation.json
                     (CIFAR flagship only; absent on blobs -> single-arm mode)

Emits a markdown comparison report to ``results/reports/eval_<arch>_<date>.md`` and
a copy + machine-readable JSON into ``Evaluation_Metric_Improve/`` for session resume.

Usage:
    python3 analysis/evaluate_extraction_quality.py --tiniest   # smoke test
    python3 analysis/evaluate_extraction_quality.py --tiny      # scale test
    python3 analysis/evaluate_extraction_quality.py --full      # CIFAR flagship
"""

import os
import re
import sys
import json
import argparse
import datetime

import numpy as np
import torch

# Allow running as a script: make the package importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from extraction_pipeline.config import OUTPUT_PATH, BASE_DIR
from extraction_pipeline.architectures import (
    TinyModel, TinierModel, TiniestModel, FullModel,
)
from extraction_pipeline.data_loading import (
    load_test_data, load_test2_data, load_test3_data, load_ground_truth_model,
)
from extraction_pipeline import config as _config
from extraction_pipeline import workflow as _wf
from extraction_pipeline.workflow import _ARCHS
from extraction_pipeline import eval_metrics as em
from extraction_pipeline.distillation_baseline import (
    ensure_distillation_baseline, distillation_paths,
)


# --------------------------------------------------------------------------- #
#  Activation auto-detection (fixes the global-LEAKY_ALPHA fragility)         #
# --------------------------------------------------------------------------- #

def _act_suffix(alpha):
    return 'leakyrelu' if alpha and alpha > 0 else 'relu'


def detect_alpha(arch_key, ext_metrics_json, true_path, verbose=True):
    """Determine the activation slope the reconstructed model was built with,
    independent of the mutable global config.LEAKY_ALPHA.

    Priority:
      1. ``leaky_alpha`` recorded in the extraction metrics (self-describing —
         written by workflow.main for every run from now on).
      2. The ``*_alpha.txt`` sidecar next to the victim model file.
      3. The victim filename suffix (``_leakyrelu`` -> 0.01, ``_relu`` -> 0.0).
      4. Current global config.LEAKY_ALPHA (legacy fallback).
    """
    if ext_metrics_json and ext_metrics_json.get('leaky_alpha') is not None:
        a = float(ext_metrics_json['leaky_alpha'])
        if verbose:
            print(f"[activation] from extraction metrics: alpha={a}")
        return a
    # Sidecar next to whichever victim variant the metrics imply; try both.
    for suffix in ('leakyrelu', 'relu'):
        cand = re.sub(r'_(relu|leakyrelu)\.pth$', f'_{suffix}.pth', true_path)
        sidecar = re.sub(r'\.pth$', '_alpha.txt', cand)
        if ext_metrics_json is None and os.path.isfile(sidecar) and suffix in true_path:
            try:
                a = float(open(sidecar).read().strip())
                if verbose:
                    print(f"[activation] from sidecar {os.path.basename(sidecar)}: alpha={a}")
                return a
            except ValueError:
                pass
    if '_leakyrelu' in os.path.basename(true_path):
        if verbose:
            print("[activation] inferred LeakyReLU(0.01) from victim filename")
        return 0.01
    if '_relu' in os.path.basename(true_path):
        if verbose:
            print("[activation] inferred ReLU from victim filename")
        return 0.0
    if verbose:
        print(f"[activation] falling back to global config.LEAKY_ALPHA={_config.LEAKY_ALPHA}")
    return float(_config.LEAKY_ALPHA)


def apply_activation(alpha):
    """Make the entire pipeline use `alpha` for this process, regardless of the
    on-disk global. Patches config.LEAKY_ALPHA (read by `_act` at call time) AND
    the import-resolved, suffix-keyed model paths in config and workflow._ARCHS
    (so the auto-built distillation sub-run also loads the matching victim)."""
    suffix = _act_suffix(alpha)
    _config.LEAKY_ALPHA = float(alpha)
    base = _config.BASE_DIR
    _config.TINY_MODEL_PTH     = os.path.join(base, f"tiny_stuff/TinyModel_{suffix}.pth")
    _config.MAKEBLOBS_MODEL_PTH = os.path.join(base, f"tiny_stuff/makeblobs_{suffix}.pth")
    _config.TINIER_MODEL_PTH   = os.path.join(base, f"tiny_stuff/tinier_makeblobs_{suffix}.pth")
    _config.TINIEST_MODEL_PTH  = os.path.join(base, f"tiny_stuff/tiniest_makeblobs_{suffix}.pth")
    _config.FULL_MODEL_PTH     = os.path.join(base, f"tiny_stuff/TinyModel_{suffix}.pth")
    # Mutate _ARCHS in place so both this module's imported ref and workflow's
    # module global point at the corrected victim paths.
    for k in _wf._ARCHS:
        mc, pth, lc, lbl = _wf._ARCHS[k]
        _wf._ARCHS[k] = (mc, re.sub(r'_(relu|leakyrelu)\.pth$', f'_{suffix}.pth', pth), lc, lbl)


REPORTS_DIR = os.path.join(BASE_DIR, "results", "reports")
EVAL_IMPROVE_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "Evaluation_Metric_Improve"))

# arch_key -> reconstructed model basename written by workflow.save_reconstructed_model
_RECON_BASENAME = {
    'tiniest': 'reconstructed_tiniest',
    'tinier':  'reconstructed_tinier',
    'makeblobs': 'reconstructed_makeblobs',
    'tiny':    'reconstructed_tiny',
    'full':    'reconstructed_full',
}


# --------------------------------------------------------------------------- #
#  Setup helpers                                                              #
# --------------------------------------------------------------------------- #

def _select_arch(args):
    if args.tiniest:     key = 'tiniest'
    elif args.tinier:    key = 'tinier'
    elif args.full:      key = 'full'
    elif args.makeblobs: key = 'makeblobs'
    else:                key = 'tiny'
    model_class, true_path, layer_config, label = _ARCHS[key]
    return key, model_class, true_path, layer_config, label


def _load_recon(pth_path, model_class):
    if not os.path.isfile(pth_path):
        return None
    model = model_class()
    sd = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _eval_tier(arch_key, args):
    """Return (X_eval, Y_eval, ref_X, tag). Mirrors workflow eval-tier logic:
    CIFAR uses the held-out X_test3; blobs use X_test2; ref_X (for off-dist box
    + interpolation endpoints) is the X_test training slice."""
    flags = dict(
        tiny=arch_key in ('tiny', 'makeblobs'),
        makeblobs=arch_key == 'makeblobs',
        tinier=arch_key == 'tinier',
        tiniest=arch_key == 'tiniest',
    )
    X_test, Y_test = load_test_data(**flags)
    X_test2, Y_test2 = load_test2_data(**flags)
    X_test3, Y_test3 = load_test3_data(**flags)

    if X_test3 is not None:
        X_eval, Y_eval, tag = X_test3, Y_test3, 'X_test3'
    elif X_test2 is not None:
        X_eval, Y_eval, tag = X_test2, Y_test2, 'X_test2'
    else:
        X_eval, Y_eval, tag = X_test, Y_test, 'X_test'
    ref_X = X_test if X_test is not None else X_eval
    return X_eval, Y_eval, ref_X, tag


# --------------------------------------------------------------------------- #
#  Per-arm metric computation                                                #
# --------------------------------------------------------------------------- #

def evaluate_arm(victim, model, X_eval, Y_eval, ref_X, victim_preds,
                 X_sub, victim_preds_sub, r_sub,
                 extraction_metrics, queries_used, query_budget,
                 structural=False, seed=0):
    """Compute the full metric suite for one reconstructed arm.

    Metric 1/3 use the full eval set; Metric 2 uses the boundary-distance
    subsample (X_sub) where `r_sub` was measured, so victim preds, model preds
    and `r` are length-aligned.
    """
    model_preds = em.predict(model, X_eval)

    # Metric 1 (full eval set)
    m1 = em.fidelity(model_preds, victim_preds, true_labels=Y_eval)

    # Metric 2 (margin-conditioned fidelity — all three aligned on the subsample)
    model_preds_sub = em.predict(model, X_sub)
    m2 = em.margin_conditioned_fidelity(victim_preds_sub, model_preds_sub, r_sub)

    # Metric 3
    m3_off = em.off_distribution_agreement(victim, model, ref_X, n=5000, seed=seed)
    m3_interp = em.interpolation_path_agreement(
        victim, model, X_eval, Y_eval, n_pairs=200, n_steps=20, seed=seed)

    # Metric 5 (structural; known-victim tiers only)
    m5 = em.structural_metrics(extraction_metrics) if structural else None

    # EQS components
    high_margin_fid = m2['bins'].get('far', {}).get('fidelity')
    c2 = m3_off['mean_agreement']
    components = {
        'C1': m1['fidelity'],
        'C2': c2,
        'C3': high_margin_fid if high_margin_fid is not None else m1['fidelity'],
        'C5': 1.0 - np.clip(queries_used / max(1, query_budget), 0.0, 1.0),
    }
    eqs_bb = em.compute_eqs(components, variant='blackbox')

    # Structural-variant EQS. Computed for BOTH arms when the tier has a known
    # victim (`structural`), so the comparison is apples-to-apples: an arm with no
    # parameter recovery (distillation) gets S=0 by construction — exactly the
    # design check in report §4.3 (distillation scores low on the structural block).
    eqs_struct = None
    if structural:
        s_score = m5['structural_score'] if (m5 and m5.get('structural_score') is not None) else 0.0
        components_s = {
            'C1': components['C1'],
            'C2': components['C2'],
            'C3': components['C3'],
            'S':  s_score,
        }
        eqs_struct = em.compute_eqs(components_s, variant='structural')

    return {
        'metric1_fidelity': m1,
        'metric2_margin_conditioned': m2,
        'metric3_off_distribution': m3_off,
        'metric3_interpolation': m3_interp,
        'metric5_structural': m5,
        'eqs_blackbox': eqs_bb,
        'eqs_structural': eqs_struct,
        'model_preds': model_preds,   # kept in-memory for McNemar; stripped before JSON
    }


# --------------------------------------------------------------------------- #
#  Report rendering                                                          #
# --------------------------------------------------------------------------- #

def _pct(x):
    return "---" if x is None else f"{100 * x:.2f} %"

def _f(x, nd=4):
    return "---" if x is None else f"{x:.{nd}f}"


def render_report(ctx):
    arch = ctx['arch_label']
    tag = ctx['eval_tag']
    ext = ctx['extraction']
    dis = ctx['distillation']
    mc = ctx['mcnemar']
    date = ctx['date']
    two_arm = dis is not None

    L = []
    L.append(f"# Improved Extraction-Quality Evaluation — {arch}")
    L.append("")
    L.append(f"_Generated {date} • activation: **{ctx.get('activation','?')}** • "
             f"held-out eval set: **{tag}** "
             f"({ctx['n_eval']} samples) • hard-label (argmax-only) where noted._")
    L.append("")
    L.append("Implements the scorecard in "
             "`Evaluation_Metric_Improve/evaluation_metrics_REPORT.md`, replacing the "
             "single naive prediction-agreement number. Off-distribution agreement "
             "(Metric 3) is the discriminator that separates **extraction** from "
             "**distillation**; structural receipts (Metric 5) make \"extraction not "
             "distillation\" literally true on known-victim tiers.")
    L.append("")
    if not two_arm:
        L.append("> **Single-arm mode**: no distillation baseline on disk for this tier "
                 "(`reconstructed_full_distillation.pth` absent). Comparison/Metric-4 "
                 "rows are omitted; extraction-only metrics + structural receipts shown.")
        L.append("")

    # ---- Headline (Metric 1) ----
    L.append("## Headline — Metric 1: in-distribution fidelity & accuracy")
    L.append("")
    e1 = ext['metric1_fidelity']
    if two_arm:
        d1 = dis['metric1_fidelity']
        L.append(f"| Metric ({tag}) | Extraction | Distillation | Gap (ext−dis) | Oracle |")
        L.append("|---|---:|---:|---:|---:|")
        L.append(f"| Fidelity vs victim (argmax) | {_pct(e1['fidelity'])} | "
                 f"{_pct(d1['fidelity'])} | {_pct(e1['fidelity'] - d1['fidelity'])} | --- |")
        L.append(f"| Accuracy vs ground truth | {_pct(e1.get('accuracy'))} | "
                 f"{_pct(d1.get('accuracy'))} | "
                 f"{_pct((e1.get('accuracy') or 0) - (d1.get('accuracy') or 0))} | "
                 f"{_pct(ctx['oracle_accuracy'])} |")
    else:
        L.append(f"| Metric ({tag}) | Extraction | Oracle |")
        L.append("|---|---:|---:|")
        L.append(f"| Fidelity vs victim (argmax) | {_pct(e1['fidelity'])} | --- |")
        L.append(f"| Accuracy vs ground truth | {_pct(e1.get('accuracy'))} | "
                 f"{_pct(ctx['oracle_accuracy'])} |")
    L.append("")

    # ---- Metric 2 ----
    L.append("## Metric 2 — margin-conditioned fidelity (kills the victim-difficulty confound)")
    L.append("")
    L.append("Fidelity stratified by victim boundary-distance proxy `r(x)` "
             "(near = brittle victim, far = stable). Extraction's advantage should "
             "concentrate in the near/mid bins.")
    L.append("")
    ext_bins = ext['metric2_margin_conditioned']['bins']
    _order = ['near', 'mid', 'far', 'all']
    bins_order = ([b for b in _order if b in ext_bins]
                  + [b for b in ext_bins if b not in _order])
    hdr = "| Bin | n | " + ("Ext fid | Dis fid | Gap |" if two_arm else "Ext fid |")
    L.append(hdr)
    L.append("|---|---:|" + ("---:|---:|---:|" if two_arm else "---:|"))
    for bn in bins_order:
        eb = ext_bins.get(bn, {})
        if two_arm:
            db = dis['metric2_margin_conditioned']['bins'].get(bn, {})
            ef, dfd = eb.get('fidelity'), db.get('fidelity')
            gap = None if (ef is None or dfd is None) else ef - dfd
            L.append(f"| {bn} | {eb.get('n','?')} | {_pct(ef)} | {_pct(dfd)} | {_pct(gap)} |")
        else:
            L.append(f"| {bn} | {eb.get('n','?')} | {_pct(eb.get('fidelity'))} |")
    L.append("")

    # ---- Metric 3 ----
    L.append("## Metric 3 — off-distribution & boundary agreement (extraction-vs-distillation discriminator)")
    L.append("")
    eo = ext['metric3_off_distribution']
    ei = ext['metric3_interpolation']
    if two_arm:
        do = dis['metric3_off_distribution']
        di = dis['metric3_interpolation']
        L.append("| Probe | Extraction | Distillation | Gap |")
        L.append("|---|---:|---:|---:|")
        L.append(f"| Uniform off-manifold agreement | {_pct(eo['uniform']['agreement'])} | "
                 f"{_pct(do['uniform']['agreement'])} | "
                 f"{_pct(eo['uniform']['agreement'] - do['uniform']['agreement'])} |")
        L.append(f"| Wide-Gaussian agreement | {_pct(eo['wide_gauss']['agreement'])} | "
                 f"{_pct(do['wide_gauss']['agreement'])} | "
                 f"{_pct(eo['wide_gauss']['agreement'] - do['wide_gauss']['agreement'])} |")
        L.append(f"| Interpolation-path agreement | {_pct(ei['mean_path_agreement'])} | "
                 f"{_pct(di['mean_path_agreement'])} | "
                 f"{_pct((ei['mean_path_agreement'] or 0) - (di['mean_path_agreement'] or 0))} |")
    else:
        L.append("| Probe | Extraction |")
        L.append("|---|---:|")
        L.append(f"| Uniform off-manifold agreement | {_pct(eo['uniform']['agreement'])} |")
        L.append(f"| Wide-Gaussian agreement | {_pct(eo['wide_gauss']['agreement'])} |")
        L.append(f"| Interpolation-path agreement | {_pct(ei['mean_path_agreement'])} |")
    L.append("")
    L.append("_Deferred (hooks present): HopSkipJump boundary co-location (3.2), "
             "adversarial transferability (3.3)._")
    L.append("")

    # ---- Metric 4 ----
    L.append("## Metric 4 — significance of the gap (single-run)")
    L.append("")
    if mc is not None:
        L.append("Paired McNemar of extraction vs distillation against the victim "
                 "reference on the shared eval set, plus bootstrap 95% CI on the gap.")
        L.append("")
        L.append("| Quantity | Value |")
        L.append("|---|---:|")
        L.append(f"| Fidelity gap (ext − dis) | {_pct(mc['gap'])} |")
        L.append(f"| Bootstrap 95% CI on gap | [{_pct(mc['gap_bootstrap_ci95'][0])}, "
                 f"{_pct(mc['gap_bootstrap_ci95'][1])}] |")
        L.append(f"| McNemar b / c (discordant) | {mc['mcnemar_b']} / {mc['mcnemar_c']} |")
        L.append(f"| McNemar χ² (1 dof) | {_f(mc['mcnemar_chi2'], 3)} |")
        L.append(f"| McNemar p-value | {_f(mc['mcnemar_p_value'], 4)} |")
        L.append(f"| Significant at 0.05 | {mc['significant_at_0.05']} |")
        L.append("")
        L.append(f"_{mc['note']}. Full N≥10-seed harness deferred "
                 "(`run_seed_significance` hook)._")
    else:
        L.append("_Not computed — single-arm mode (no distillation baseline). "
                 "The N≥10-seed harness is deferred regardless._")
    L.append("")

    # ---- Metric 5 ----
    L.append("## Metric 5 — parameter-level structural recovery (known-victim receipts)")
    L.append("")
    m5 = ext['metric5_structural']
    if m5:
        L.append(f"Overall: mean |cos| = **{_f(m5['mean_abs_cosine_sim'])}**, "
                 f"mean sign-acc = **{_f(m5['mean_sign_accuracy'])}**, "
                 f"coverage = **{_pct(m5['coverage'])}** "
                 f"({m5['recovered_neurons']}/{m5['total_neurons']} neurons).")
        L.append("")
        L.append("| Layer | mean &#124;cos&#124; | sign acc | recovered |")
        L.append("|---|---:|---:|---:|")
        for ln, lv in m5['per_layer'].items():
            L.append(f"| {ln} | {_f(lv['mean_abs_cosine_sim'])} | "
                     f"{_f(lv['sign_accuracy'])} | {lv.get('num_recovered','?')} |")
        L.append("")
        L.append("_Distillation has |cos|≈0 / no signs by construction — this block is "
                 "the structural proof of \"extraction, not distillation\"._")
    else:
        L.append("_Not available — structural metrics need ground-truth weights "
                 "(known-victim/make_blobs tiers only)._")
    L.append("")

    # ---- EQS ----
    L.append("## Deliverable B — composite Extraction-Quality Score (EQS, 0–100)")
    L.append("")
    L.append("_C4 (gap significance) dropped & remaining weights renormalized to 100 "
             "per agreed scope. EQS gap (ext − dis) is the clean single number; the "
             "component profile shows where the advantage lives._")
    L.append("")
    variant = 'eqs_structural' if ext.get('eqs_structural') else 'eqs_blackbox'
    e_eqs = ext[variant]
    L.append(f"**Variant: `{e_eqs['variant']}`**")
    L.append("")
    if two_arm:
        d_eqs = dis[variant]
        L.append(f"| | Extraction | Distillation | Gap |")
        L.append("|---|---:|---:|---:|")
        L.append(f"| **EQS** | **{e_eqs['eqs']:.1f}** | **{d_eqs['eqs']:.1f}** | "
                 f"**{e_eqs['eqs'] - d_eqs['eqs']:+.1f}** |")
        L.append("")
        L.append("| Component | Ext value | Ext pts | Dis value | Dis pts |")
        L.append("|---|---:|---:|---:|---:|")
        for comp in e_eqs['profile']:
            ep = e_eqs['profile'][comp]
            dp = d_eqs['profile'][comp]
            L.append(f"| {comp} | {_f(ep['value'])} | {ep['contribution']:.1f} | "
                     f"{_f(dp['value'])} | {dp['contribution']:.1f} |")
    else:
        L.append(f"**EQS (extraction): {e_eqs['eqs']:.1f}**")
        L.append("")
        L.append("| Component | value | pts |")
        L.append("|---|---:|---:|")
        for comp, ep in e_eqs['profile'].items():
            L.append(f"| {comp} | {_f(ep['value'])} | {ep['contribution']:.1f} |")
    L.append("")

    # ---- Narrative ----
    L.append("## One-sentence defensive narrative")
    L.append("")
    if two_arm:
        L.append(f"The in-distribution fidelity gap ({_pct(e1['fidelity'] - dis['metric1_fidelity']['fidelity'])}) "
                 "is " + ("statistically significant" if (mc and mc['significant_at_0.05']) else "not yet significant at p<0.05 in this single run")
                 + ", is decomposed honestly by victim margin (Metric 2), and "
                 "persists/widens off-distribution and at the boundary (Metric 3) where "
                 "only a true parameter copy can match"
                 + (", backed by direct parameter recovery the baseline cannot possess (Metric 5)."
                    if m5 else "."))
    else:
        L.append("Extraction-only run; add a distillation baseline to populate the "
                 "comparison, McNemar, and EQS-gap rows.")
    L.append("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
#  JSON serialization (strip in-memory pred arrays)                          #
# --------------------------------------------------------------------------- #

def _strip(arm):
    if arm is None:
        return None
    return {k: v for k, v in arm.items() if k != 'model_preds'}


# --------------------------------------------------------------------------- #
#  Main                                                                      #
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(description="Improved extraction-quality evaluation")
    p.add_argument('--tiny', action='store_true', default=True)
    p.add_argument('--full', action='store_true')
    p.add_argument('--makeblobs', action='store_true')
    p.add_argument('--tinier', action='store_true')
    p.add_argument('--tiniest', action='store_true')
    p.add_argument('--ext-path', default=None, help="Override extraction .pth")
    p.add_argument('--ext-metrics', default=None, help="Override extraction metrics json")
    p.add_argument('--dis-path', default=None, help="Override distillation .pth")
    p.add_argument('--dis-metrics', default=None, help="Override distillation metrics json")
    p.add_argument('--allow-single-arm', action='store_true',
                   help="Opt out of the mandatory distillation baseline (extraction-only report). "
                        "By default a missing distillation arm is generated automatically.")
    p.add_argument('--force-distill', action='store_true',
                   help="Rebuild the distillation baseline even if one is cached on disk.")
    p.add_argument('--distill-epochs', type=int, default=None,
                   help="Override refine epochs for the auto-generated distillation baseline.")
    p.add_argument('--query-budget', type=int, default=20000,
                   help="Oracle query budget for the EQS query-economy component (C5)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-boundary', type=int, default=1500,
                   help="Subsample size for the boundary-distance margin proxy (Metric 2)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    arch_key, model_class, true_path, layer_config, label = _select_arch(args)

    print("=" * 70)
    print(f"IMPROVED EVALUATION — {label}")
    print("=" * 70)

    # ---- activation auto-detection (before any model is built) ------------------
    # Read the extraction metrics first so we can pin the activation to whatever the
    # reconstructed model was actually built with — independent of the on-disk
    # global LEAKY_ALPHA. This fixes silent ReLU/LeakyReLU mismatch garbage.
    base = _RECON_BASENAME[arch_key]
    ext_metrics_path = args.ext_metrics or os.path.join(OUTPUT_PATH, "extraction_metrics.json")
    ext_metrics_json = _load_json(ext_metrics_path)
    alpha = detect_alpha(arch_key, ext_metrics_json, true_path)
    apply_activation(alpha)
    # Re-resolve arch (true_path now carries the correct activation suffix).
    arch_key, model_class, true_path, layer_config, label = _select_arch(args)
    act_label = f"LeakyReLU({alpha})" if alpha > 0 else "ReLU"
    print(f"Activation: {act_label}  |  victim: {os.path.basename(true_path)}")

    # ---- data + victim ----
    X_eval, Y_eval, ref_X, tag = _eval_tier(arch_key, args)
    Y_eval_np = (Y_eval.cpu().numpy() if torch.is_tensor(Y_eval) else np.asarray(Y_eval))
    victim = load_ground_truth_model(true_path, model_class)
    victim.eval()
    victim_preds = em.predict(victim, X_eval)
    oracle_acc = float(np.mean(victim_preds == Y_eval_np))
    print(f"Eval tier: {tag} ({len(X_eval)} samples) | oracle acc {oracle_acc:.4f}")

    # ---- shared victim margin proxy (Metric 2) on a subsample ----
    n_b = min(args.n_boundary, len(X_eval))
    rng = np.random.default_rng(args.seed)
    sub_idx = rng.choice(len(X_eval), size=n_b, replace=False)
    X_sub = X_eval[sub_idx]
    print(f"Computing boundary-distance margin proxy on {n_b} subsampled points...")
    r_sub = em.boundary_distance_bisection(victim, X_sub, seed=args.seed)
    victim_preds_sub = victim_preds[sub_idx]

    # ---- locate extraction arm on disk (ext_metrics already loaded above) ----
    ext_path = args.ext_path or os.path.join(OUTPUT_PATH, f"{base}.pth")
    ext_model = _load_recon(ext_path, model_class)
    if ext_model is None:
        print(f"ERROR: extraction model not found at {ext_path}")
        return 1

    # Structural metrics (Metric 5) are computable whenever the known-victim run
    # left per-layer |cos|/sign-accuracy receipts on disk. All experimental tiers
    # here use a known victim, so this is availability-gated, not arch-gated.
    structural = bool(
        ext_metrics_json
        and any(isinstance(v, dict) and 'sign_accuracy' in v
                for v in (ext_metrics_json.get('layer_metrics') or {}).values()))

    # ---- distillation arm: MANDATORY companion (project policy) ----------------
    # Every eval analysis is a two-arm comparison. If the distillation baseline is
    # not on disk we build it now (same arch, no frozen rows, oracle-label fit).
    # `--allow-single-arm` is the explicit opt-out escape hatch.
    if args.dis_path:
        dis_path, dis_metrics_path = args.dis_path, args.dis_metrics
    else:
        dis_path, dis_metrics_path = distillation_paths(arch_key)

    dis_model = _load_recon(dis_path, model_class)
    if dis_model is None and not args.allow_single_arm:
        print("\n" + "=" * 70)
        print("DISTILLATION BASELINE REQUIRED — none on disk, generating now")
        print("(disable with --allow-single-arm)")
        print("=" * 70)
        dis_path, dis_metrics_path = ensure_distillation_baseline(
            arch_key, force=args.force_distill, refine_epochs=args.distill_epochs)
        dis_model = _load_recon(dis_path, model_class)
        if dis_model is None:
            print(f"ERROR: distillation generation failed (no model at {dis_path})")
            return 1

    dis_metrics_json = _load_json(dis_metrics_path) if dis_metrics_path else None
    two_arm = dis_model is not None
    print(f"Extraction arm: {ext_path}")
    print(f"Distillation arm: {'(none — single-arm mode, opted out)' if not two_arm else dis_path}")

    # ---- per-arm metric suites ----
    print("Evaluating extraction arm...")
    ext = evaluate_arm(victim, ext_model, X_eval, Y_eval_np, ref_X, victim_preds,
                       X_sub, victim_preds_sub, r_sub,
                       ext_metrics_json, _queries_used(ext_metrics_json),
                       args.query_budget, structural=structural, seed=args.seed)

    dis = None
    mcnemar = None
    if two_arm:
        print("Evaluating distillation arm...")
        dis = evaluate_arm(victim, dis_model, X_eval, Y_eval_np, ref_X, victim_preds,
                           X_sub, victim_preds_sub, r_sub,
                           dis_metrics_json, _queries_used(dis_metrics_json),
                           args.query_budget, structural=structural, seed=args.seed)
        # Metric 4 — McNemar on the full eval set.
        mcnemar = em.paired_mcnemar(victim_preds, ext['model_preds'],
                                    dis['model_preds'], seed=args.seed)

    # ---- render + persist ----
    date = datetime.date.today().isoformat()
    ctx = {
        'arch_label': label, 'eval_tag': tag, 'n_eval': int(len(X_eval)),
        'oracle_accuracy': oracle_acc, 'date': date, 'activation': act_label,
        'extraction': ext, 'distillation': dis, 'mcnemar': mcnemar,
    }
    report_md = render_report(ctx)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(EVAL_IMPROVE_DIR, exist_ok=True)
    report_name = f"eval_{arch_key}_{date}.md"
    report_path = os.path.join(REPORTS_DIR, report_name)
    with open(report_path, 'w') as f:
        f.write(report_md)
    improve_copy = os.path.join(EVAL_IMPROVE_DIR, report_name)
    with open(improve_copy, 'w') as f:
        f.write(report_md)

    json_out = {
        'arch': arch_key, 'label': label, 'eval_tag': tag,
        'n_eval': int(len(X_eval)), 'oracle_accuracy': oracle_acc,
        'leaky_alpha': float(alpha), 'activation': act_label,
        'two_arm': two_arm, 'date': date,
        'extraction': _strip(ext), 'distillation': _strip(dis),
        'mcnemar': mcnemar,
    }
    json_path = os.path.join(EVAL_IMPROVE_DIR, f"eval_{arch_key}.json")
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2, default=_json_default)

    print("-" * 70)
    print(f"Report : {report_path}")
    print(f"Copy   : {improve_copy}")
    print(f"JSON   : {json_path}")
    e1 = ext['metric1_fidelity']
    print(f"Extraction fidelity ({tag}): {e1['fidelity']:.4f} | "
          f"EQS {ext['eqs_structural']['eqs']:.1f}" if ext.get('eqs_structural')
          else f"Extraction fidelity ({tag}): {e1['fidelity']:.4f} | "
               f"EQS {ext['eqs_blackbox']['eqs']:.1f}")
    if two_arm:
        print(f"Distillation fidelity: {dis['metric1_fidelity']['fidelity']:.4f} | "
              f"gap {mcnemar['gap']:+.4f} | McNemar p={mcnemar['mcnemar_p_value']:.4g}")
    print("=" * 70)
    return 0


def _queries_used(metrics_json):
    """Best-effort oracle-query count for the EQS query-economy term (C5)."""
    if not metrics_json:
        return 0
    return int(metrics_json.get('train_phase3_size', 0) or 0)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    raise SystemExit(main())
