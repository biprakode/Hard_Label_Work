#!/usr/bin/env python3
"""
Additive Phase-3 ablation harness  (READ-ONLY on all pipeline/method code).

This script does NOT modify, monkey-patch, or re-implement any pipeline logic. It
imports the canonical Phase-3 functions from `analysis/extraction_pipeline` and
composes them in the *exact* order and with the *exact* arguments that
`run_one_model_enhanced.sh` / `run_extraction.py --from-scratch --refine` use
(SA+margin sign search, X_test3 honest-eval, AdamW+cosine refinement). The only
addition is that it *evaluates and snapshots* the reconstructed model at five
cumulative checkpoints instead of only at the end.

It reuses the already-on-disk Phase-1 (unsigned weights) + Phase-2 (signs)
artifacts that the driver produced for the current victim, so every stage shares
one extraction; the ablation is purely over the Phase-3 components.

Cumulative stages (each includes everything before it):
  Stage 0  RAW            reconstruct only: directions in place, biases = 0,
                          fc5 = random init, signs as Phase-2 produced them.
  Stage 1  + BIAS         + recover_biases_from_duals  (Section 6.2)
  Stage 2  + LR FIT       + recover_output_layer (multinomial LR fc5)  (Section 6.3)
  Stage 3  + SIGN SEARCH  + SA+margin sign search (cycles + pair-flip + mini-refine)
                            then the canonical post-search final fc5 LR refit  (Section 5)
  Stage 4  + FROZEN REFINE+ oracle_label_refinement (frozen recovered rows)  (Section 6.4)

Plus ONE non-staged distillation baseline row (all hidden rows Kaiming + trainable,
same query budget + refinement settings).

Every metric is computed on the held-out X_test3 via the canonical scorecard
(`evaluate_extraction_quality.evaluate_arm` -> `eval_metrics`), so EQS uses the
same structural 22/26/17/20 weighting as everywhere else.

Usage:
  python ablation_harness.py --arch {tiniest,tinier,makeblobs} --act {relu,leakyrelu} \
                             --out <per_victim_results.json>
"""

import os
import sys
import json
import argparse
import datetime

import numpy as np
import torch

# --- locate the pipeline package (analysis/) -------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
# This folder lives inside the repo (Hard_Label_Work/ablation_tiny/). Resolve HLW
# whether we're inside the repo (parent has analysis/) or a sibling of it.
_parent = os.path.abspath(os.path.join(HERE, ".."))
HLW = _parent if os.path.isdir(os.path.join(_parent, "analysis")) else os.path.join(_parent, "Hard_Label_Work")
ANALYSIS = os.path.join(HLW, "analysis")
for p in (ANALYSIS, HLW):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- canonical pipeline imports (read-only use) ----------------------------- #
from extraction_pipeline import config as _config
from extraction_pipeline.config import (
    SIGNATURE_WEIGHTS_PATH, SIGN_RECOVERY_PATH, DUAL_POINTS_DIR, OUTPUT_PATH,
)
from extraction_pipeline.workflow import _ARCHS
from extraction_pipeline.data_loading import (
    load_test_data, load_test2_data, load_test3_data, load_ground_truth_model,
)
from extraction_pipeline.metrics import compute_weight_metrics_v2
from extraction_pipeline.weight_assembly import reconstruct_model
from extraction_pipeline.bias_recovery import recover_biases_from_duals
from extraction_pipeline.output_layer_recovery import recover_output_layer
from extraction_pipeline.sign_search import sa_oracle_sign_search, pair_flip_lookahead
from extraction_pipeline.refinement import oracle_label_refinement
from extraction_pipeline.distillation_baseline import ensure_distillation_baseline, distillation_paths
import extraction_pipeline.eval_metrics as em
import evaluate_extraction_quality as eeq


# --- canonical per-arch Phase-3 config (mirrors run_one_model_enhanced.sh) --- #
# SIGN_RESTARTS is unused on the SA path (workflow only consults it for greedy);
# kept here only for provenance/printing.
ARCH_CFG = {
    'tiniest':   dict(sign_restarts=1, refine_epochs=300),
    'tinier':    dict(sign_restarts=1, refine_epochs=500),
    'makeblobs': dict(sign_restarts=2, refine_epochs=500),
}
# Common canonical knobs (identical across the make_blobs tiers).
SIGN_PAIR        = 8
SIGN_CYCLES      = 3
SIGN_MINI_EPOCHS = 20
SIGN_METHOD      = 'sa'
SIGN_OBJECTIVE   = 'margin'
REFINE_LR        = 5e-3
REFINE_WD        = 1e-4
REFINE_COSINE    = True
EARLY_PATIENCE   = 5
EVAL_EVERY       = 10
QUERY_BUDGET     = 20000          # X_test ∪ X_test2 train-union pool

STAGE_NAMES = {
    0: "Stage 0  RAW",
    1: "Stage 1  + BIAS",
    2: "Stage 2  + LR FIT",
    3: "Stage 3  + SIGN SEARCH",
    4: "Stage 4  + FROZEN REFINE",
}


def _arch_flags(arch_key):
    tiny     = arch_key in ('tiny', 'makeblobs')
    return dict(
        tiny=tiny,
        makeblobs=(arch_key == 'makeblobs'),
        tinier=(arch_key == 'tinier'),
        tiniest=(arch_key == 'tiniest'),
    )


def _layer_metrics_now(model, true_model, masks, n_layers):
    """Recompute per-recovered-row structural receipts (sign acc, |cos|) at the
    current model state -- identical to workflow.main's post-refinement block."""
    lm = {}
    layers = [model.fc1, model.fc2, model.fc3, model.fc4]
    tlayers = [true_model.fc1, true_model.fc2, true_model.fc3, true_model.fc4]
    for lid in range(n_layers):
        mask = masks.get(lid)
        if mask is None or not np.any(mask):
            continue
        ext = layers[lid].weight.data.cpu().numpy()
        tw = tlayers[lid].weight.data.cpu().numpy()
        m = compute_weight_metrics_v2(ext[mask], tw[mask])
        if m:
            m.pop('per_neuron', None)
            m['num_recovered'] = int(np.sum(mask))
            lm[f'layer_{lid}'] = m
    return lm


def _eval_stage(stage_id, model, true_model, masks, recovery_stats, n_layers,
                victim, X_eval, Y_eval_np, ref_X, victim_preds,
                X_sub, victim_preds_sub, r_sub):
    """Run the canonical scorecard on the current model state; return the row."""
    lm = _layer_metrics_now(model, true_model, masks, n_layers)
    stage_metrics = {'layer_metrics': lm, 'recovery_stats': recovery_stats}
    res = eeq.evaluate_arm(
        victim, model, X_eval, Y_eval_np, ref_X, victim_preds,
        X_sub, victim_preds_sub, r_sub,
        stage_metrics, queries_used=QUERY_BUDGET, query_budget=QUERY_BUDGET,
        structural=True, seed=0,
    )
    m1 = res['metric1_fidelity']
    m5 = res.get('metric5_structural') or {}
    eqs = res.get('eqs_structural') or {}
    row = {
        'stage': stage_id,
        'stage_name': STAGE_NAMES[stage_id],
        'agreement': float(m1['fidelity']),
        'ext_acc': float(m1.get('accuracy')) if m1.get('accuracy') is not None else None,
        'sign_acc': (None if m5.get('mean_sign_accuracy') is None
                     else float(m5['mean_sign_accuracy'])),
        'eqs': float(eqs['eqs']) if eqs else None,
        'mean_abs_cos': (None if m5.get('mean_abs_cosine_sim') is None
                         else float(m5['mean_abs_cosine_sim'])),
        'coverage': (None if m5.get('coverage') is None else float(m5['coverage'])),
    }
    print(f"  [{STAGE_NAMES[stage_id]:<26}] "
          f"agree={row['agreement']:.4f}  ext_acc={row['ext_acc']:.4f}  "
          f"sign_acc={row['sign_acc'] if row['sign_acc'] is None else round(row['sign_acc'],4)}  "
          f"EQS={row['eqs']:.1f}", flush=True)
    return row


def run_victim(arch_key, act, out_path):
    print("=" * 72)
    print(f"ADDITIVE PHASE-3 ABLATION  —  arch={arch_key}  act={act}")
    print(f"  LEAKY_ALPHA(config) = {_config.LEAKY_ALPHA}")
    print("=" * 72, flush=True)

    cfg = ARCH_CFG[arch_key]
    model_class, true_path, layer_config, label = _ARCHS[arch_key]
    n_layers = len(layer_config)
    flags = _arch_flags(arch_key)

    # ---- data (three-tier contract) ----
    X_test,  Y_test  = load_test_data(**flags)
    X_test2, Y_test2 = load_test2_data(**flags)
    X_test3, Y_test3 = load_test3_data(**flags)
    assert X_test3 is not None, "X_test3 required for the honest-eval ablation"
    # Canonical extraction regime: train on X_test ∪ X_test2, eval on X_test3.
    X_train_phase3 = torch.cat([X_test, X_test2], dim=0)
    X_eval, Y_eval = X_test3, Y_test3
    Y_eval_np = Y_eval.cpu().numpy() if torch.is_tensor(Y_eval) else np.asarray(Y_eval)
    ref_X = X_test
    print(f"X_train_phase3={tuple(X_train_phase3.shape)}  X_eval(X_test3)={tuple(X_eval.shape)}",
          flush=True)

    # ---- victim + shared margin proxy (computed once; victim-only) ----
    true_model = load_ground_truth_model(true_path, model_class)
    true_model.eval()
    victim_preds = em.predict(true_model, X_eval)
    oracle_acc = float(np.mean(victim_preds == Y_eval_np))
    n_b = min(1500, len(X_eval))
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(X_eval), size=n_b, replace=False)
    X_sub = X_eval[sub_idx]
    print(f"oracle acc (X_test3) = {oracle_acc:.4f}; computing margin proxy on {n_b} pts...",
          flush=True)
    r_sub = em.boundary_distance_bisection(true_model, X_sub, seed=0)
    victim_preds_sub = victim_preds[sub_idx]

    def ev(stage_id, model, masks, rstats):
        return _eval_stage(stage_id, model, true_model, masks, rstats, n_layers,
                           true_model, X_eval, Y_eval_np, ref_X, victim_preds,
                           X_sub, victim_preds_sub, r_sub)

    stages = {}

    # ===== Build the reconstruction from scratch (biases=0, fc5 random) ===== #
    model, _lm0, recovery_stats, masks = reconstruct_model(
        SIGNATURE_WEIGHTS_PATH, SIGN_RECOVERY_PATH,
        model_class, layer_config, true_path, random_seed=42,
        copy_true_biases=False, copy_true_output=False,
    )

    # ---- STAGE 0 RAW ----
    stages['0'] = ev(0, model, masks, recovery_stats)

    # ---- STAGE 1 + BIAS ----
    recover_biases_from_duals(model, DUAL_POINTS_DIR, masks,
                              layer_ids=tuple(range(n_layers)), verbose=False)
    stages['1'] = ev(1, model, masks, recovery_stats)

    # ---- STAGE 2 + LR FIT (provisional fc5 LR fit, must precede sign search) ----
    recover_output_layer(model, true_model, X_train_phase3, verbose=False)
    stages['2'] = ev(2, model, masks, recovery_stats)

    # ---- STAGE 3 + SIGN SEARCH (SA+margin, cycles + pair-flip + mini-refine) ----
    n_cycles = max(1, SIGN_CYCLES)
    for cyc in range(n_cycles):
        sa_oracle_sign_search(
            model, true_model, X_train_phase3, masks,
            layer_ids=tuple(range(n_layers)),
            objective=SIGN_OBJECTIVE, verbose=False, duals_dir=DUAL_POINTS_DIR,
        )
        if SIGN_PAIR > 0:
            pair_flip_lookahead(
                model, true_model, X_train_phase3, masks,
                K=SIGN_PAIR, layer_ids=tuple(range(n_layers)),
                verbose=False, duals_dir=DUAL_POINTS_DIR,
            )
        if cyc < n_cycles - 1:        # mini-refinement burst between cycles
            oracle_label_refinement(
                model, true_model, X_train_phase3, masks,
                n_epochs=SIGN_MINI_EPOCHS, lr=REFINE_LR,
                freeze_recovered_weights=True, verbose=False,
                weight_decay=REFINE_WD, use_cosine_lr=False,
            )
    # canonical post-search final fc5 LR refit
    recover_output_layer(model, true_model, X_train_phase3, verbose=False)
    stages['3'] = ev(3, model, masks, recovery_stats)

    # ---- STAGE 4 + FROZEN REFINE (full pipeline) ----
    oracle_label_refinement(
        model, true_model, X_train_phase3, masks,
        n_epochs=cfg['refine_epochs'], lr=REFINE_LR,
        freeze_recovered_weights=True, verbose=False,
        X_eval=X_eval, eval_every=EVAL_EVERY, patience=EARLY_PATIENCE,
        early_stop=True, weight_decay=REFINE_WD, use_cosine_lr=REFINE_COSINE,
    )
    stages['4'] = ev(4, model, masks, recovery_stats)

    # ===== Distillation baseline (non-staged contrast row) ===== #
    print("-" * 72)
    print("Building distillation baseline (Kaiming hidden, all rows trainable, "
          "same query budget + refinement settings)...", flush=True)
    dist_pth, _dist_metrics = ensure_distillation_baseline(
        arch_key, force=True, refine_epochs=cfg['refine_epochs'],
        extra_argv=['--eval-on-test3', '--train-union-test12',
                    '--early-stop', '--patience', str(EARLY_PATIENCE),
                    '--eval-every', str(EVAL_EVERY)],
        verbose=True,
    )
    dis_model = eeq._load_recon(dist_pth, model_class)
    # Evaluate the distillation arm on X_test3 with the SAME scorecard. It has no
    # recovered rows -> structural S = 0 (EQS S-block); sign acc is n/a.
    dis_res = eeq.evaluate_arm(
        true_model, dis_model, X_eval, Y_eval_np, ref_X, victim_preds,
        X_sub, victim_preds_sub, r_sub,
        {'layer_metrics': {}, 'recovery_stats': {}},
        queries_used=QUERY_BUDGET, query_budget=QUERY_BUDGET,
        structural=True, seed=0,
    )
    dm1 = dis_res['metric1_fidelity']
    distill = {
        'agreement': float(dm1['fidelity']),
        'ext_acc': float(dm1.get('accuracy')) if dm1.get('accuracy') is not None else None,
        'sign_acc': None,    # n/a — no recovered rows
        'eqs': float((dis_res.get('eqs_structural') or {}).get('eqs', 0.0)),
    }
    print(f"  [distillation              ] agree={distill['agreement']:.4f}  "
          f"ext_acc={distill['ext_acc']:.4f}  sign_acc=n/a  EQS={distill['eqs']:.1f}",
          flush=True)

    # ===== Headline references for the stage-4 sanity check ===== #
    headline = {}
    drv_metrics = os.path.join(OUTPUT_PATH, "extraction_metrics.json")
    if os.path.isfile(drv_metrics):
        dj = json.load(open(drv_metrics))
        headline['driver_extraction_metrics'] = {
            'prediction_agreement': dj.get('prediction_agreement'),
            'reconstructed_accuracy': dj.get('reconstructed_accuracy'),
            'eval_tag': dj.get('eval_tag'),
            'sign_search_method': dj.get('sign_search_method'),
            'sign_search_objective': dj.get('sign_search_objective'),
            'from_scratch': dj.get('from_scratch'),
        }
    drv_eval = os.path.join(eeq.EVAL_IMPROVE_DIR, f"eval_{arch_key}.json")
    if os.path.isfile(drv_eval):
        ej = json.load(open(drv_eval))
        ext = ej.get('extraction') or {}
        headline['driver_scorecard'] = {
            'eval_tag': ej.get('eval_tag'),
            'fidelity': (ext.get('metric1_fidelity') or {}).get('fidelity'),
            'accuracy': (ext.get('metric1_fidelity') or {}).get('accuracy'),
            'eqs_structural': (ext.get('eqs_structural') or {}).get('eqs'),
        }

    result = {
        'arch_key': arch_key,
        'activation': act,
        'victim_label': label,
        'victim_path': true_path,
        'leaky_alpha': float(_config.LEAKY_ALPHA),
        'oracle_acc_test3': oracle_acc,
        'recovery_stats': {
            'total_neurons': recovery_stats['total_neurons'],
            'recovered_neurons': recovery_stats['recovered_neurons'],
            'random_init_neurons': recovery_stats['random_init_neurons'],
            'per_layer': {str(k): v for k, v in recovery_stats['per_layer'].items()},
        },
        'stages': stages,
        'distillation': distill,
        'headline_reference': headline,
        'config': {
            'sign_search_method': SIGN_METHOD,
            'sign_search_objective': SIGN_OBJECTIVE,
            'sign_pair_lookahead': SIGN_PAIR,
            'sign_refine_cycles': SIGN_CYCLES,
            'sign_refine_mini_epochs': SIGN_MINI_EPOCHS,
            'refine_epochs': cfg['refine_epochs'],
            'refine_lr': REFINE_LR,
            'refine_weight_decay': REFINE_WD,
            'refine_cosine_lr': REFINE_COSINE,
            'early_stop_patience': EARLY_PATIENCE,
            'eval_every': EVAL_EVERY,
            'eval_set': 'X_test3',
            'train_pool': 'X_test ∪ X_test2',
            'query_budget': QUERY_BUDGET,
            'eqs_variant': 'structural (C1=22,C2=26,C3=17,S=20)',
            'reconstruct_seed': 42,
            'scorecard_seed': 0,
            'n_boundary': n_b,
        },
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote per-victim results -> {out_path}", flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=['tiniest', 'tinier', 'makeblobs'])
    ap.add_argument('--act', required=True, choices=['relu', 'leakyrelu'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    run_victim(args.arch, args.act, args.out)


if __name__ == '__main__':
    main()
