"""
End-to-end Phase-3 reconstruction workflow.

Stages (each gated by CLI flags so the workflow stays composable):

    1.  load test data (X_test for Phase-3 training, X_test2 for eval)
    2.  load ground-truth model (oracle)
    3.  build model from extracted values   (weight_assembly.reconstruct_model)
    4.  bias recovery from dual points       [--from-scratch]
    5.  oracle-queries-only sign search      [--sign-search / --from-scratch]
    6.  fc5 LR fit on oracle hard labels     [--from-scratch]
    7.  oracle-label refinement              [--refine]
    8.  evaluation on X_test2 (fresh seed=99 set) + save model + metrics
"""

import os
import json
import argparse

import numpy as np
import torch

from .config import (
    SIGNATURE_WEIGHTS_PATH, SIGN_RECOVERY_PATH, OUTPUT_PATH,
    DUAL_POINTS_DIR,
    TINY_MODEL_PTH, MAKEBLOBS_MODEL_PTH, TINIER_MODEL_PTH, TINIEST_MODEL_PTH, FULL_MODEL_PTH,
)
from .architectures import TinyModel, TinierModel, TiniestModel, FullModel
from .data_loading import load_test_data, load_test2_data, load_ground_truth_model
from .metrics import compute_weight_metrics_v2, test_model_accuracy
from .weight_assembly import reconstruct_model, save_reconstructed_model
from .bias_recovery import recover_biases_from_duals
from .output_layer_recovery import recover_output_layer
from .sign_search import oracle_sign_search
from .refinement import oracle_label_refinement


# ---------------------------------------------------------- model selection --

_ARCHS = {
    'tiniest':  (TiniestModel,  TINIEST_MODEL_PTH,   {0: (8, 8),    1: (8, 8),    2: (8, 8),    3: (8, 8)},
                 "Tiniest (8-8-8-8-8-8, make_blobs)"),
    'tinier':   (TinierModel,   TINIER_MODEL_PTH,    {0: (16, 32),  1: (16, 16),  2: (16, 16),  3: (8, 16)},
                 "Tinier (32->16->16->16->8->4, make_blobs)"),
    'full':     (FullModel,     FULL_MODEL_PTH,      {0: (256, 3072), 1: (256, 256), 2: (256, 256), 3: (64, 256)},
                 "Full (3072x256, CIFAR-10)"),
    'makeblobs': (TinyModel,    MAKEBLOBS_MODEL_PTH, {0: (64, 64),  1: (64, 64),  2: (64, 64),  3: (64, 64)},
                 "Makeblobs (64x64, synthetic data)"),
    'tiny':     (TinyModel,     TINY_MODEL_PTH,      {0: (64, 64),  1: (64, 64),  2: (64, 64),  3: (64, 64)},
                 "Tiny (64x64, CIFAR-10)"),
}


def _select_arch(args):
    """Resolve CLI flags to a (key, model_class, true_path, layer_config, label) tuple."""
    if args.tiniest:   key = 'tiniest'
    elif args.tinier:  key = 'tinier'
    elif args.full:    key = 'full'
    elif args.makeblobs: key = 'makeblobs'
    else:              key = 'tiny'
    model_class, true_path, layer_config, label = _ARCHS[key]
    return key, model_class, true_path, layer_config, label


# --------------------------------------------------------------- CLI parser --

def build_parser():
    parser = argparse.ArgumentParser(description="Model Extraction Verification")
    parser.add_argument('--tiny', action='store_true', default=True, help="Use tiny model (64x64)")
    parser.add_argument('--full', action='store_true', help="Use full model (3072x256)")
    parser.add_argument('--makeblobs', action='store_true', help="Use makeblobs model (64x64, synthetic data)")
    parser.add_argument('--tinier', action='store_true', help="Use tinier model (32->16->16->16->8->4)")
    parser.add_argument('--tiniest', action='store_true', help="Use tiniest model (8-8-8-8-8-8, make_blobs)")
    parser.add_argument('--signature-path', type=str, default=SIGNATURE_WEIGHTS_PATH)
    parser.add_argument('--sign-path', type=str, default=SIGN_RECOVERY_PATH)
    parser.add_argument('--output-path', type=str, default=OUTPUT_PATH)
    parser.add_argument('--sign-search', action='store_true',
                        help="After reconstruction, brute-force sign combos per layer using only hard-label oracle queries on X_test")
    parser.add_argument('--from-scratch', action='store_true',
                        help="Rebuild model from scratch: no cheat biases, no cheat fc5. Implies --sign-search with joint w+b flipping, plus fc5 LR-fit on oracle labels")
    parser.add_argument('--duals-dir', type=str, default=DUAL_POINTS_DIR,
                        help="Directory holding layer{L}_neuron{i}.npy dual point files")
    parser.add_argument('--refine', action='store_true',
                        help="After sign search + fc5 LR fit, polish the model against oracle hard labels. Freezes extracted weight rows; only biases, fc5, and unrecovered neurons' rows are updated")
    parser.add_argument('--refine-unfreeze', action='store_true',
                        help="When combined with --refine, unfreeze ALL weights (full distillation). Strays furthest from 'extraction' but pushes accuracy closer to 100%")
    parser.add_argument('--refine-epochs', type=int, default=300)
    parser.add_argument('--refine-lr', type=float, default=5e-3)
    return parser


# -------------------------------------------------------------- entry point --

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.from_scratch:
        args.sign_search = True

    print("=" * 70)
    print("MODEL EXTRACTION VERIFICATION (v2 - Three-Tier Metrics)")
    print("=" * 70)

    arch_key, model_class, true_model_path, layer_config, model_type_str = _select_arch(args)
    tiny     = arch_key in ('tiny', 'makeblobs')
    makeblobs = arch_key == 'makeblobs'
    tinier   = arch_key == 'tinier'
    tiniest  = arch_key == 'tiniest'

    print(f"\nModel type: {model_type_str}")
    print(f"Architecture: {list(layer_config.values())}")
    print(f"Ground truth model: {true_model_path}")
    print(f"Signature weights path: {args.signature_path}")
    print(f"Sign recovery path: {args.sign_path}")

    # 1. Test data
    print("\n" + "=" * 70 + "\nLOADING TEST DATA\n" + "=" * 70)
    X_test, Y_test = load_test_data(tiny=tiny, makeblobs=makeblobs, tinier=tinier, tiniest=tiniest)
    if X_test is None:
        print("Failed to load test data")
        return
    print(f"X_test (Phase-3 training) shape: {X_test.shape}")

    X_test2, Y_test2 = load_test2_data(tiny=tiny, makeblobs=makeblobs, tinier=tinier, tiniest=tiniest)
    if X_test2 is None:
        print("  Warning: X_test2 not found, falling back to X_test for evaluation")
        X_test2, Y_test2 = X_test, Y_test
    else:
        print(f"X_test2 (eval-only, seed=99) shape: {X_test2.shape}")

    # 2. Ground-truth oracle
    print("\n" + "=" * 70 + "\nGROUND TRUTH MODEL\n" + "=" * 70)
    true_model = load_ground_truth_model(true_model_path, model_class)
    true_accuracy = test_model_accuracy(true_model, X_test2, Y_test2, "Ground Truth (on X_test2)")

    # 3. Build model from extracted values
    print("\n" + "=" * 70 + "\nRECONSTRUCTING MODEL FROM EXTRACTION\n" + "=" * 70)
    print("\nNOTE: Unrecovered neurons use Kaiming/He initialization")
    print("      Scaling uses abs(factor) to ensure sign is NOT revealed\n")
    reconstructed_model, layer_metrics, recovery_stats, recovered_masks_by_layer = reconstruct_model(
        args.signature_path, args.sign_path,
        model_class, layer_config,
        true_model_path, random_seed=42,
        copy_true_biases=not args.from_scratch,
        copy_true_output=not args.from_scratch,
    )

    # 4. Bias recovery from duals (--from-scratch only)
    if args.from_scratch:
        print("\n" + "=" * 70 + "\nBIAS RECOVERY FROM DUAL POINTS (bottom-up)\n" + "=" * 70)
        recover_biases_from_duals(
            reconstructed_model, args.duals_dir, recovered_masks_by_layer,
            layer_ids=tuple(range(len(layer_config))), verbose=True,
        )

    pre_search_accuracy = test_model_accuracy(reconstructed_model, X_test2, Y_test2, "Pre-sign-search (X_test2)")

    # 5. Sign search
    sign_search_results = None
    if args.sign_search:
        print("\n" + "=" * 70 + "\nORACLE-QUERIES-ONLY SIGN SEARCH\n" + "=" * 70)
        print("Brute-forcing 2^k sign combos per layer using only hard-label oracle queries on X_test")
        duals_for_search = args.duals_dir if args.from_scratch else None
        sign_search_results = oracle_sign_search(
            reconstructed_model, true_model, X_test, recovered_masks_by_layer,
            layer_ids=tuple(range(len(layer_config))), verbose=True,
            duals_dir=duals_for_search,
        )

        # 6. fc5 LR fit (--from-scratch only)
        if args.from_scratch:
            print("\n" + "=" * 70 + "\nOUTPUT LAYER (fc5) RECOVERY via LR on oracle labels\n" + "=" * 70)
            recover_output_layer(reconstructed_model, true_model, X_test, verbose=True)

    # 7. Refinement
    refine_results = None
    if args.refine:
        print("\n" + "=" * 70 + "\nORACLE-LABEL REFINEMENT\n" + "=" * 70)
        refine_results = oracle_label_refinement(
            reconstructed_model, true_model, X_test, recovered_masks_by_layer,
            n_epochs=args.refine_epochs, lr=args.refine_lr,
            freeze_recovered_weights=not args.refine_unfreeze,
            verbose=True,
        )

        print("\n--- Post-sign-search per-layer metrics ---")
        layers_list = [reconstructed_model.fc1, reconstructed_model.fc2,
                       reconstructed_model.fc3, reconstructed_model.fc4]
        true_layers = [true_model.fc1, true_model.fc2, true_model.fc3, true_model.fc4]
        for lid in range(len(layer_config)):
            mask = recovered_masks_by_layer.get(lid)
            if mask is None or not mask.any():
                continue
            ext = layers_list[lid].weight.data.numpy()
            true_w = true_layers[lid].weight.data.numpy()
            m = compute_weight_metrics_v2(ext[mask], true_w[mask])
            if m:
                m.pop('per_neuron', None)
                m['num_recovered'] = int(mask.sum())
                m['num_random_init'] = int(len(mask) - mask.sum())
                layer_metrics[f'layer_{lid}'] = m
                print(f"  layer_{lid}: sign_acc={m['sign_accuracy']:.4f}  "
                      f"|cos|={m['mean_abs_cosine_sim']:.4f}  "
                      f"mag_rel_err={m['magnitude_mean_rel_error']:.4f}")

    # 8. Evaluation + save
    print("\n" + "=" * 70 + "\nRECONSTRUCTED MODEL EVALUATION (on X_test2 — fresh eval set, seed=99)\n" + "=" * 70)
    recon_accuracy = test_model_accuracy(reconstructed_model, X_test2, Y_test2, "Reconstructed (X_test2)")

    print("\n--- Prediction Comparison (X_test2) ---")
    with torch.no_grad():
        true_preds = true_model(X_test2).argmax(dim=1)
        recon_preds = reconstructed_model(X_test2).argmax(dim=1)
        pred_agreement = (true_preds == recon_preds).float().mean().item()
    print(f"Prediction agreement (X_test2): {pred_agreement:.4f}")

    # Summary
    print("\n" + "=" * 70 + "\nEXTRACTION SUMMARY (Three-Tier Metrics)\n" + "=" * 70)
    print("\n--- Per-Layer Metrics (Recovered Neurons Only) ---")
    summary_layers = {k: v for k, v in layer_metrics.items()
                      if not k.endswith('_per_neuron') and not k.endswith('_all')}
    for layer_name, m in sorted(summary_layers.items()):
        if 'sign_accuracy' not in m:
            continue
        print(f"\n{layer_name} ({m.get('num_recovered', '?')}/{m.get('num_recovered', 0) + m.get('num_random_init', 0)} recovered):")
        print(f"  SIGN accuracy:      {m['sign_accuracy']:.4f}")
        print(f"  MAGNITUDE rel err:  {m['magnitude_mean_rel_error']:.4f} (median: {m['magnitude_median_rel_error']:.4f})")
        print(f"  COMBINED rel err:   {m['combined_mean_rel_error']:.4f} (median: {m['combined_median_rel_error']:.4f})")
        print(f"  Mean |cos sim|:     {m['mean_abs_cosine_sim']:.4f}")

    if summary_layers:
        valid = [m for m in summary_layers.values() if 'sign_accuracy' in m]
        if valid:
            avg_sign = np.mean([m['sign_accuracy'] for m in valid])
            avg_mag  = np.mean([m['magnitude_mean_rel_error'] for m in valid])
            avg_comb = np.mean([m['combined_mean_rel_error'] for m in valid])
            avg_cos  = np.mean([m['mean_abs_cosine_sim'] for m in valid])
            print(f"\n--- Overall Averages (across layers) ---")
            print(f"  SIGN accuracy:      {avg_sign:.4f}")
            print(f"  MAGNITUDE rel err:  {avg_mag:.4f}")
            print(f"  COMBINED rel err:   {avg_comb:.4f}")
            print(f"  Mean |cos sim|:     {avg_cos:.4f}")

    print(f"\n--- Model Performance ---")
    print(f"Ground truth accuracy: {true_accuracy:.4f}")
    print(f"Reconstructed accuracy: {recon_accuracy:.4f}")
    print(f"Accuracy difference: {abs(true_accuracy - recon_accuracy):.4f}")
    print(f"Prediction agreement: {pred_agreement:.4f}")

    extraction_success = pred_agreement > 0.95 and recon_accuracy > 0.9 * true_accuracy
    print(f"\n*** EXTRACTION {'SUCCESSFUL' if extraction_success else 'NEEDS IMPROVEMENT'} ***")

    # Save model
    print("\n" + "=" * 70 + "\nSAVING RECONSTRUCTED MODEL\n" + "=" * 70)
    if   tiniest:   model_name = "reconstructed_tiniest"
    elif tinier:    model_name = "reconstructed_tinier"
    elif makeblobs: model_name = "reconstructed_makeblobs"
    elif tiny:      model_name = "reconstructed_tiny"
    else:           model_name = "reconstructed_full"
    save_reconstructed_model(reconstructed_model, args.output_path, model_name)

    # Metrics
    metrics_path = os.path.join(args.output_path, "extraction_metrics.json")
    serializable_metrics = {}
    for k, v in layer_metrics.items():
        if k.endswith('_per_neuron'):
            continue
        if isinstance(v, dict):
            serializable_metrics[k] = {
                kk: float(vv) if isinstance(vv, (float, np.floating))
                    else int(vv) if isinstance(vv, (int, np.integer))
                    else vv
                for kk, vv in v.items()
            }
        else:
            serializable_metrics[k] = v

    all_metrics = {
        'model_type': model_type_str,
        'model_name': model_name,
        'layer_config': {str(k): list(v) for k, v in layer_config.items()},
        'layer_metrics': serializable_metrics,
        'recovery_stats': {
            'total_neurons': recovery_stats['total_neurons'],
            'recovered_neurons': recovery_stats['recovered_neurons'],
            'random_init_neurons': recovery_stats['random_init_neurons'],
            'overall_recovery_rate': recovery_stats['recovered_neurons'] / max(1, recovery_stats['total_neurons']),
            'per_layer': {str(k): v for k, v in recovery_stats['per_layer'].items()}
        },
        'true_accuracy': float(true_accuracy),
        'reconstructed_accuracy': float(recon_accuracy),
        'pre_sign_search_accuracy': float(pre_search_accuracy),
        'prediction_agreement': float(pred_agreement),
        'extraction_success': extraction_success,
        'sign_search_applied': bool(args.sign_search),
        'sign_search_results': sign_search_results,
        'refinement_applied': bool(args.refine),
        'refinement_results': refine_results,
        'from_scratch': bool(args.from_scratch),
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    print("\n" + "=" * 70 + "\nVERIFICATION COMPLETE\n" + "=" * 70)
