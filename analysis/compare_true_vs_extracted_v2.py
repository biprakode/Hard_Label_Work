"""
Parameterised true-vs-extracted comparison report writer.

Works for: --arch {tiniest|tinier|tiny}  x  --activation {relu|leakyrelu}.
Output: a markdown report at the path given by --output.

Metrics produced:
    * per-neuron L1, relative error, cosine similarity, sign correctness
      (only on the neurons Phase 1 actually recovered)
    * per-layer summary (mean/median over recovered neurons)
    * bias comparison per layer
    * fc5 row-wise weight comparison
    * oracle vs reconstructed accuracy and prediction agreement on X_test2
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

BASE = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/Hard_Label_Work"
sys.path.insert(0, os.path.join(BASE, "analysis"))

from extraction_pipeline.architectures import TiniestModel, TinierModel, TinyModel  # noqa: E402
from extraction_pipeline.weight_assembly import load_unsigned_weights  # noqa: E402


# ----------------------- arch / activation configuration --------------------- #
ARCH_CONFIG = {
    'tiniest': {
        'model_class':   TiniestModel,
        'layer_dims':    [(8, 8), (8, 8), (8, 8), (8, 8)],   # (n_neurons, input_dim) per hidden layer
        'fc5_shape':     (8, 8),                              # (out_dim, in_dim)
        'true_basename': 'tiniest_makeblobs_{act}',
        'ext_basename':  'reconstructed_tiniest',
        'x_test2':       'data/x_test2_tiniest_makeblobs.npy',
        'y_test2':       'data/y_test2_tiniest_makeblobs.npy',
        'pretty':        'Tiniest 8-8-8-8-8-8',
    },
    'tinier': {
        'model_class':   TinierModel,
        'layer_dims':    [(16, 32), (16, 16), (16, 16), (8, 16)],
        'fc5_shape':     (4, 8),
        'true_basename': 'tinier_makeblobs_{act}',
        'ext_basename':  'reconstructed_tinier',
        'x_test2':       'data/x_test2_tinier_makeblobs.npy',
        'y_test2':       'data/y_test2_tinier_makeblobs.npy',
        'pretty':        'Tinier 32->16->16->16->8->4',
    },
    'tiny': {
        'model_class':   TinyModel,
        'layer_dims':    [(64, 64), (64, 64), (64, 64), (64, 64)],
        'fc5_shape':     (10, 64),
        'true_basename': 'makeblobs_{act}',
        'ext_basename':  'reconstructed_makeblobs',
        'x_test2':       'data/x_test2_makeblobs.npy',
        'y_test2':       'data/y_test2_makeblobs.npy',
        'pretty':        'Tiny 64-64-64-64-64-10',
    },
}


# ---------------------------------- helpers ---------------------------------- #

def load_state(path, model_class):
    m = model_class()
    sd = torch.load(path, map_location='cpu')
    try:
        m.load_state_dict(sd)
    except RuntimeError:
        # Tolerate "output.*" naming etc. — fall back to loose match.
        rename = {'output.weight': 'fc5.weight', 'output.bias': 'fc5.bias'}
        new_sd = {rename.get(k, k): v for k, v in sd.items()}
        m.load_state_dict(new_sd, strict=False)
    m.eval()
    return m


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_recovered_masks(arch_cfg):
    sig_path = os.path.join(BASE, "signature_recovery/outputs/model_weights/Vrelu")
    masks = {}
    layer_offset = 0
    for lid, (n, in_d) in enumerate(arch_cfg['layer_dims']):
        _, mask, _ = load_unsigned_weights(
            sig_path, lid, n, input_dim=in_d,
            use_random_init=False, layer_offset=layer_offset,
        )
        masks[lid] = mask
        layer_offset += n
    return masks


def compare_layer(w_ext, w_true, b_ext, b_true, mask):
    """Per-recovered-neuron comparison."""
    idx = np.where(mask)[0]
    rows = []
    cos_pos = 0
    for i in idx:
        e, t = w_ext[i], w_true[i]
        l1 = float(np.sum(np.abs(e - t)))
        rel = float(np.linalg.norm(e - t) / max(np.linalg.norm(t), 1e-15))
        cs = cosine(e, t)
        sign_correct = cs > 0
        if sign_correct:
            cos_pos += 1
        rows.append({
            'neuron': int(i), 'placed_at': int(i),
            'l1': l1, 'rel_err': rel, 'cos': cs, 'abs_cos': abs(cs),
            'sign_correct': sign_correct,
        })
    return rows, cos_pos


def fc5_compare(w_ext, w_true, b_ext, b_true):
    out_dim = w_ext.shape[0]
    rows = []
    for i in range(out_dim):
        e, t = w_ext[i], w_true[i]
        rows.append({
            'row': i,
            'l1': float(np.sum(np.abs(e - t))),
            'rel_err': float(np.linalg.norm(e - t) / max(np.linalg.norm(t), 1e-15)),
            'cos': cosine(e, t),
            'abs_cos': abs(cosine(e, t)),
        })
    return rows


def fmt(x, prec=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


# -------------------------------- main loop --------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['tiniest', 'tinier', 'tiny'], required=True)
    p.add_argument('--activation', choices=['relu', 'leakyrelu'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ext-path', default=None, help='Override extracted .pth path')
    p.add_argument('--true-path', default=None, help='Override true .pth path')
    p.add_argument('--metrics-json', default=None, help='Path to extraction_metrics.json')
    p.add_argument('--timings', default=None, help='Path to JSON with stage wall times')
    args = p.parse_args()

    cfg = ARCH_CONFIG[args.arch]
    true_path = args.true_path or os.path.join(BASE, "tiny_stuff",
                                                cfg['true_basename'].format(act=args.activation) + ".pth")
    ext_path  = args.ext_path  or os.path.join(BASE, "results/reconstructed_models",
                                                cfg['ext_basename'] + ".pth")
    x_test2_path = os.path.join(BASE, cfg['x_test2'])
    y_test2_path = os.path.join(BASE, cfg['y_test2'])

    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True model not found: {true_path}")
    if not os.path.exists(ext_path):
        raise FileNotFoundError(f"Extracted model not found: {ext_path}")
    if not os.path.exists(x_test2_path):
        raise FileNotFoundError(f"X_test2 not found: {x_test2_path}")

    true_m = load_state(true_path, cfg['model_class'])
    ext_m  = load_state(ext_path,  cfg['model_class'])

    masks = get_recovered_masks(cfg)

    # Per-layer comparison
    layer_objs_true = [true_m.fc1, true_m.fc2, true_m.fc3, true_m.fc4]
    layer_objs_ext  = [ext_m.fc1,  ext_m.fc2,  ext_m.fc3,  ext_m.fc4]

    per_layer = {}
    per_neuron_all = {}
    layer_total_recovered = 0
    layer_total_neurons = 0

    for lid in range(4):
        n, _ = cfg['layer_dims'][lid]
        layer_total_neurons += n
        mask = masks.get(lid, np.zeros(n, dtype=bool))

        w_true = layer_objs_true[lid].weight.data.numpy()
        w_ext  = layer_objs_ext[lid].weight.data.numpy()
        b_true = layer_objs_true[lid].bias.data.numpy()
        b_ext  = layer_objs_ext[lid].bias.data.numpy()

        rows, cos_pos = compare_layer(w_ext, w_true, b_ext, b_true, mask)
        per_neuron_all[lid] = rows
        recovered_count = int(mask.sum())
        layer_total_recovered += recovered_count

        if recovered_count > 0:
            l1s = np.array([r['l1'] for r in rows])
            rels = np.array([r['rel_err'] for r in rows])
            abscoss = np.array([r['abs_cos'] for r in rows])
            sign_acc = cos_pos / recovered_count
        else:
            l1s = rels = abscoss = np.array([np.nan])
            sign_acc = np.nan

        per_layer[lid] = {
            'recovered': recovered_count,
            'total': n,
            'l1_median': float(np.median(l1s)) if recovered_count else float('nan'),
            'l1_mean':   float(np.mean(l1s))   if recovered_count else float('nan'),
            'rel_err_median': float(np.median(rels)) if recovered_count else float('nan'),
            'rel_err_mean':   float(np.mean(rels))   if recovered_count else float('nan'),
            'abs_cos_mean':   float(np.mean(abscoss)) if recovered_count else float('nan'),
            'sign_acc': sign_acc,
            'b_l1_sum':       float(np.sum(np.abs(b_ext - b_true))),
            'b_delta_median': float(np.median(np.abs(b_ext - b_true))),
            'b_delta_max':    float(np.max(np.abs(b_ext - b_true))),
        }

    # fc5 comparison (output layer)
    w_true_fc5 = true_m.fc5.weight.data.numpy()
    w_ext_fc5  = ext_m.fc5.weight.data.numpy()
    b_true_fc5 = true_m.fc5.bias.data.numpy()
    b_ext_fc5  = ext_m.fc5.bias.data.numpy()
    fc5_rows = fc5_compare(w_ext_fc5, w_true_fc5, b_ext_fc5, b_true_fc5)
    fc5_summary = {
        'l1_mean':     float(np.mean([r['l1']      for r in fc5_rows])),
        'rel_err_mean':float(np.mean([r['rel_err'] for r in fc5_rows])),
        'abs_cos_mean':float(np.mean([r['abs_cos'] for r in fc5_rows])),
        'cos_mean':    float(np.mean([r['cos']     for r in fc5_rows])),
        'b_l1_sum':    float(np.sum(np.abs(b_ext_fc5 - b_true_fc5))),
    }

    # X_test2 functional accuracy
    X2 = torch.tensor(np.load(x_test2_path), dtype=torch.float64)
    Y2 = torch.tensor(np.load(y_test2_path), dtype=torch.long)
    with torch.no_grad():
        oracle_preds = true_m(X2).argmax(dim=1)
        ext_preds    = ext_m(X2).argmax(dim=1)
        oracle_acc = (oracle_preds == Y2).float().mean().item()
        ext_acc    = (ext_preds    == Y2).float().mean().item()
        agreement  = (ext_preds == oracle_preds).float().mean().item()

    # Pull metrics from extraction_metrics.json if available
    extraction_metrics = {}
    if args.metrics_json and os.path.exists(args.metrics_json):
        with open(args.metrics_json) as f:
            extraction_metrics = json.load(f)
    timings = {}
    if args.timings and os.path.exists(args.timings):
        with open(args.timings) as f:
            timings = json.load(f)

    # --------------------------- write the markdown --------------------------- #
    lines = []
    lines.append(f"# {cfg['pretty']} ({args.activation.upper()}) — True vs Extracted")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Activation:** {'ReLU' if args.activation == 'relu' else 'Leaky ReLU(α=0.01)'}")
    lines.append(f"**Architecture:** {cfg['pretty']}")
    lines.append(f"**Extracted model:** `{ext_path}`")
    lines.append(f"**True model:** `{true_path}`")
    lines.append(f"**Functional accuracy on X_test2:** {ext_acc*100:.2f}% (oracle {oracle_acc*100:.2f}%, agreement {agreement*100:.2f}%)")
    lines.append("")

    # Per-layer summary
    lines.append("## Per-layer summary (recovered neurons only)")
    lines.append("| Layer | n_rec/n | L1 median | L1 mean | rel err median | rel err mean | |cos| mean | sign acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for lid in range(4):
        m = per_layer[lid]
        lines.append(
            f"| fc{lid+1} | {m['recovered']}/{m['total']} | "
            f"{fmt(m['l1_median'])} | {fmt(m['l1_mean'])} | "
            f"{fmt(m['rel_err_median'])} | {fmt(m['rel_err_mean'])} | "
            f"{fmt(m['abs_cos_mean'])} | "
            f"{fmt(m['sign_acc']) if not np.isnan(m['sign_acc']) else '—'} |"
        )
    # overall row — averages across layers that had any recovery
    valid = [m for m in per_layer.values() if m['recovered'] > 0]
    if valid:
        overall_abscos = float(np.mean([m['abs_cos_mean'] for m in valid]))
        overall_sign = float(np.mean([m['sign_acc'] for m in valid]))
        overall_relerr = float(np.mean([m['rel_err_mean'] for m in valid]))
    else:
        overall_abscos = overall_sign = overall_relerr = float('nan')
    lines.append(
        f"| **overall** | **{layer_total_recovered}/{layer_total_neurons}** | — | — | "
        f"— | **{fmt(overall_relerr)}** | **{fmt(overall_abscos)}** | **{fmt(overall_sign)}** |"
    )
    lines.append("")

    # Per-neuron details
    lines.append("## Per-neuron detail (recovered only)")
    lines.append("| Layer | Placed at | L1 | rel err | cos sim | sign correct? |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for lid in range(4):
        for r in per_neuron_all[lid]:
            lines.append(
                f"| fc{lid+1} | {r['placed_at']} | {fmt(r['l1'])} | {fmt(r['rel_err'])} | "
                f"{fmt(r['cos'])} | {'✓' if r['sign_correct'] else '✗'} |"
            )
    lines.append("")

    # Biases
    lines.append("## Biases (all hidden neurons per layer)")
    lines.append("| Layer | L1 sum | |Δ| median | |Δ| max |")
    lines.append("|---|---:|---:|---:|")
    for lid in range(4):
        m = per_layer[lid]
        lines.append(f"| fc{lid+1} | {fmt(m['b_l1_sum'])} | {fmt(m['b_delta_median'])} | {fmt(m['b_delta_max'])} |")
    lines.append(f"| fc5 | {fmt(fc5_summary['b_l1_sum'])} | — | — |")
    lines.append("")

    # fc5 weights
    lines.append("## fc5 weight comparison (row-wise)")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| row-wise L1 mean | {fmt(fc5_summary['l1_mean'])} |")
    lines.append(f"| row-wise rel err mean | {fmt(fc5_summary['rel_err_mean'])} |")
    lines.append(f"| row-wise \\|cos\\| mean | {fmt(fc5_summary['abs_cos_mean'])} |")
    lines.append(f"| row-wise (signed) cos mean | {fmt(fc5_summary['cos_mean'])} |")
    lines.append("")

    # Headline
    lines.append("## Headline")
    lines.append("| Property | Result |")
    lines.append("|---|---|")
    lines.append(f"| Signature direction (\\|cos\\|, recovered) | {fmt(overall_abscos)} |")
    lines.append(f"| Signature sign accuracy (recovered) | {fmt(overall_sign)} |")
    lines.append(f"| Mean rel err on recovered weights | {fmt(overall_relerr)} |")
    lines.append(f"| Reconstructed acc on X_test2 | {ext_acc*100:.2f}% |")
    lines.append(f"| Oracle acc on X_test2 | {oracle_acc*100:.2f}% |")
    lines.append(f"| Prediction agreement on X_test2 | {agreement*100:.2f}% |")
    lines.append(f"| Phase-1 recovery rate | {layer_total_recovered}/{layer_total_neurons} ({100*layer_total_recovered/max(layer_total_neurons,1):.1f}%) |")
    lines.append("")

    # Pipeline timings (if provided)
    if timings:
        lines.append("## Pipeline timings")
        lines.append("| Stage | Wall time (s) |")
        lines.append("|---|---:|")
        for k, v in timings.items():
            lines.append(f"| {k} | {fmt(v, 1)} |")
        total_t = sum(timings.values()) if timings else None
        if total_t:
            lines.append(f"| **total** | **{total_t:.1f}** |")
        lines.append("")

    # Extraction metrics JSON snapshot (if provided)
    if extraction_metrics:
        lines.append("## extraction_metrics.json snapshot")
        keep = ['true_accuracy', 'reconstructed_accuracy', 'pre_sign_search_accuracy',
                'prediction_agreement', 'extraction_success',
                'sign_search_applied', 'refinement_applied', 'from_scratch']
        lines.append("| key | value |")
        lines.append("|---|---|")
        for k in keep:
            if k in extraction_metrics:
                lines.append(f"| {k} | {extraction_metrics[k]} |")
        if 'recovery_stats' in extraction_metrics:
            rs = extraction_metrics['recovery_stats']
            lines.append(f"| total_neurons | {rs.get('total_neurons')} |")
            lines.append(f"| recovered_neurons | {rs.get('recovered_neurons')} |")
            lines.append(f"| overall_recovery_rate | {fmt(rs.get('overall_recovery_rate'))} |")
        lines.append("")

    out = "\n".join(lines)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(out)
    print(f"Wrote {args.output} ({len(out)} bytes)")

    # also dump a json sibling
    json_path = str(Path(args.output).with_suffix('.json'))
    summary_json = {
        'arch': args.arch, 'activation': args.activation,
        'extracted_path': ext_path, 'true_path': true_path,
        'oracle_acc_x_test2': oracle_acc, 'reconstructed_acc_x_test2': ext_acc,
        'prediction_agreement_x_test2': agreement,
        'per_layer': per_layer, 'fc5_summary': fc5_summary,
        'overall_abscos': overall_abscos, 'overall_sign_acc': overall_sign,
        'overall_rel_err': overall_relerr,
        'total_recovered': layer_total_recovered, 'total_neurons': layer_total_neurons,
    }
    with open(json_path, 'w') as f:
        json.dump(summary_json, f, indent=2, default=float)
    print(f"Wrote {json_path}")


if __name__ == '__main__':
    main()
