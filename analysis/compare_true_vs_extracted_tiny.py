"""
Side-by-side comparison of TRUE tiny (64-64-64-64-64-10 makeblobs) weights
vs the best EXTRACTED model.

Same metrics as compare_true_vs_extracted.py, adapted for the 5-hidden 64-wide arch.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

BASE = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase"
sys.path.insert(0, os.path.join(BASE, "analysis"))
from extraction_pipeline.weight_assembly import load_unsigned_weights  # type: ignore
from extraction_pipeline.architectures import TinyModel  # type: ignore

TRUE_PATH = os.path.join(BASE, "tiny_stuff/makeblobs_relu.pth")
EXT_PATH  = os.path.join(BASE, "results/reconstructed_models/reconstructed_makeblobs.pth")
SIG_PATH  = os.path.join(BASE, "signature_recovery/outputs/model_weights/Vrelu")
OUT_JSON  = os.path.join(BASE, "results/reconstructed_models/true_vs_extracted_tiny_metrics.json")

LAYER_SIZES = [64, 64, 64, 64, 64, 10]  # input, 4 hidden, output
N_HIDDEN_LAYERS = 4  # fc1..fc4 recovered; fc5 is LR fit


def load_model(path):
    m = TinyModel()
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


def get_recovered_masks():
    """Use the extraction_pipeline.weight_assembly loader so the mask matches what the built model used."""
    masks = {}
    layer_offsets = [0, 64, 128, 192]
    for lid, off in enumerate(layer_offsets):
        _, mask, _ = load_unsigned_weights(
            SIG_PATH, lid, 64, input_dim=64,
            use_random_init=False, layer_offset=off,
        )
        masks[lid] = mask
    return masks


def compare_layer(w_ext, w_true, mask):
    idx = np.where(mask)[0]
    per = []
    for i in idx:
        e, t = w_ext[i], w_true[i]
        l1 = float(np.sum(np.abs(e - t)))
        rel = float(np.linalg.norm(e - t) / max(np.linalg.norm(t), 1e-12))
        cos = float(np.dot(e, t) / max(np.linalg.norm(e) * np.linalg.norm(t), 1e-12))
        per.append({
            'neuron': int(i), 'l1': l1, 'rel_err': rel,
            'cos_sim': cos, 'abs_cos_sim': abs(cos),
            'sign_correct': cos > 0,
        })
    n = len(per)
    if n == 0:
        return {'n_recovered': 0, 'per_neuron': []}
    return {
        'n_recovered': n,
        'l1_mean': float(np.mean([p['l1'] for p in per])),
        'l1_median': float(np.median([p['l1'] for p in per])),
        'l1_max': float(np.max([p['l1'] for p in per])),
        'rel_err_mean': float(np.mean([p['rel_err'] for p in per])),
        'rel_err_median': float(np.median([p['rel_err'] for p in per])),
        'rel_err_max': float(np.max([p['rel_err'] for p in per])),
        'abs_cos_mean': float(np.mean([p['abs_cos_sim'] for p in per])),
        'cos_mean': float(np.mean([p['cos_sim'] for p in per])),
        'sign_accuracy': sum(p['sign_correct'] for p in per) / n,
        'per_neuron': per,
    }


def main():
    print("=" * 70)
    print("TRUE vs EXTRACTED — tiny 64-64-64-64-64-10 (makeblobs)")
    print("=" * 70)
    print(f"True:      {TRUE_PATH}")
    print(f"Extracted: {EXT_PATH}")

    true_m = load_model(TRUE_PATH)
    ext_m = load_model(EXT_PATH)
    masks = get_recovered_masks()

    layer_names = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']
    true_layers = [true_m.fc1, true_m.fc2, true_m.fc3, true_m.fc4, true_m.fc5]
    ext_layers  = [ext_m.fc1,  ext_m.fc2,  ext_m.fc3,  ext_m.fc4,  ext_m.fc5]

    report = {'layers': {}, 'bias_comparison': {}, 'fc5_comparison': {}}

    print("\n--- Hidden-layer signature recovery (recovered neurons only) ---")
    hdr = f"{'layer':>5} {'n_rec':>6} {'L1_med':>10} {'L1_mean':>10} {'Rel_med':>10} {'Rel_mean':>10} {'|cos|_mean':>11} {'sign_acc':>9}"
    print(hdr)
    for lid in range(N_HIDDEN_LAYERS):
        wt = true_layers[lid].weight.data.numpy()
        we = ext_layers[lid].weight.data.numpy()
        r = compare_layer(we, wt, masks[lid])
        report['layers'][f'fc{lid + 1}'] = r
        if r['n_recovered']:
            print(f"{'fc' + str(lid + 1):>5} {r['n_recovered']:>6} "
                  f"{r['l1_median']:>10.4f} {r['l1_mean']:>10.4f} "
                  f"{r['rel_err_median']:>10.4f} {r['rel_err_mean']:>10.4f} "
                  f"{r['abs_cos_mean']:>11.4f} {r['sign_accuracy']:>9.4f}")
        else:
            print(f"{'fc' + str(lid + 1):>5} {0:>6}  (no recovered)")

    print("\n--- Bias comparison (all layers) ---")
    print(f"{'layer':>5} {'L1_sum':>10} {'|delta|_med':>12} {'|delta|_max':>12}")
    for lid in range(5):
        bt = true_layers[lid].bias.data.numpy()
        be = ext_layers[lid].bias.data.numpy()
        d = np.abs(be - bt)
        row = {
            'l1_sum': float(np.sum(d)),
            'abs_median': float(np.median(d)),
            'abs_max': float(np.max(d)),
            'true_mean': float(bt.mean()),
            'ext_mean': float(be.mean()),
        }
        report['bias_comparison'][layer_names[lid]] = row
        print(f"{layer_names[lid]:>5} {row['l1_sum']:>10.4f} "
              f"{row['abs_median']:>12.4f} {row['abs_max']:>12.4f}")

    print("\n--- fc5 weight comparison (LR-fit on top of extracted features) ---")
    wt5 = true_layers[4].weight.data.numpy()
    we5 = ext_layers[4].weight.data.numpy()
    rows = []
    for i in range(wt5.shape[0]):
        e, t = we5[i], wt5[i]
        l1 = float(np.sum(np.abs(e - t)))
        rel = float(np.linalg.norm(e - t) / max(np.linalg.norm(t), 1e-12))
        cos = float(np.dot(e, t) / max(np.linalg.norm(e) * np.linalg.norm(t), 1e-12))
        rows.append({'class': i, 'l1': l1, 'rel': rel, 'cos': cos})
    report['fc5_comparison'] = {
        'per_row': rows,
        'l1_mean': float(np.mean([r['l1'] for r in rows])),
        'rel_mean': float(np.mean([r['rel'] for r in rows])),
        'abs_cos_mean': float(np.mean([abs(r['cos']) for r in rows])),
    }
    print(f"  row-wise L1 mean   : {report['fc5_comparison']['l1_mean']:.4f}")
    print(f"  row-wise rel mean  : {report['fc5_comparison']['rel_mean']:.4f}")
    print(f"  row-wise |cos| mean: {report['fc5_comparison']['abs_cos_mean']:.4f}")

    summary = {}
    tot_rec = sum(report['layers'][f'fc{i + 1}']['n_recovered'] for i in range(N_HIDDEN_LAYERS))
    if tot_rec:
        per_n = sum([report['layers'][f'fc{i + 1}'].get('per_neuron', []) for i in range(N_HIDDEN_LAYERS)], [])
        summary['total_recovered_neurons'] = tot_rec
        summary['overall_sign_accuracy'] = sum(p['sign_correct'] for p in per_n) / tot_rec
        summary['overall_abs_cos_mean'] = float(np.mean([p['abs_cos_sim'] for p in per_n]))
        summary['overall_rel_err_median'] = float(np.median([p['rel_err'] for p in per_n]))
        summary['overall_l1_median'] = float(np.median([p['l1'] for p in per_n]))
    report['summary'] = summary

    print("\n--- Overall (recovered hidden neurons) ---")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {OUT_JSON}")


if __name__ == '__main__':
    main()
