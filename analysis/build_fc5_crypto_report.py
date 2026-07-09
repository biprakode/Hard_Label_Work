"""
Build the fc5 cryptanalytic-vs-LR comparison report from the staged suite
metrics (run_fc5_crypto_suite.sh output). Reads
  Cryptanalytic_output_context/suite_<DATE>/<arch>_<act>_<variant>_metrics.json
and writes fc5_crypto_vs_lr_report.md + fc5_crypto_vs_lr_table.json there.

Variants: s1_lr, s1_crypto (Stage-1 algebraic: sig+bias+fc5, no ML),
          s2_lr (canonical), s2_crypto (Stage-2: +SA sign search + refine).
"""
import json, glob, os, sys

SUITE = sys.argv[1] if len(sys.argv) > 1 else None
if SUITE is None:
    cands = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "Cryptanalytic_output_context", "suite_*")))
    SUITE = cands[-1]
SUITE = os.path.abspath(SUITE)

MODELS = ["tiniest_relu", "tiniest_leakyrelu", "tinier_relu", "tinier_leakyrelu",
          "tiny_relu", "tiny_leakyrelu"]
VARIANTS = ["s1_lr", "s1_crypto", "s2_lr", "s2_crypto"]

# Canonical 21/5 reference (user-provided). Hidden-layer recovery + end-to-end.
CANON = {
    "tiniest_relu":      dict(rec="24/32 (75%)",  agree=98.90),
    "tiniest_leakyrelu": dict(rec="23/32 (72%)",  agree=99.20),
    "tinier_relu":       dict(rec="30/56 (54%)",  agree=100.0),
    "tinier_leakyrelu":  dict(rec="37/56 (66%)",  agree=100.0),
    "tiny_relu":         dict(rec="157/256 (61%)", agree=100.0),
    "tiny_leakyrelu":    dict(rec="230/256 (90%)", agree=100.0),
}


def load(model, variant):
    p = os.path.join(SUITE, f"{model}_{variant}_metrics.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def g(j, *path, default=None):
    for k in path:
        if not isinstance(j, dict) or k not in j:
            return default
        j = j[k]
    return j


def fc5_pre(j):   # pre-refine fc5 vs victim
    m = g(j, 'layer_metrics', 'fc5_recovered', default={}) or {}
    return m.get('sign_accuracy'), m.get('mean_abs_cosine_sim')


def fc5_post(j):  # post-refine (final) fc5 vs victim
    m = g(j, 'layer_metrics', 'fc5', default={}) or {}
    return m.get('sign_accuracy'), m.get('mean_abs_cosine_sim')


def fnum(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def rank_str(j):
    cs = g(j, 'fc5_crypto_stats') or {}
    if not cs:
        return "—"
    tag = "✓" if cs.get('full_rank') else ""
    return f"{cs.get('rank','?')}/{cs.get('rank_target','?')}{tag}"


def _ag(data, m, v):
    j = data[m][v]
    return fnum(g(j, 'prediction_agreement'), 3) if j else "—"


def write_snapshot(data):
    """Stage-1 'crypto ends here' algebraic snapshot report."""
    L = []
    L.append("# fc5 SNAPSHOT report — Stage-1 algebraic extraction (crypto vs LR)\n")
    L.append(f"Suite dir: `{os.path.relpath(SUITE)}`. **Stage 1** = signature + bias "
             "recovery + fc5, NO sign search, NO refine ('crypto ends here'). fc5 methods: "
             "**LR** distillation fit vs **crypto** (ePrint 2025/1118 §3 solve; drop-in "
             "true-prefix cheat for its `h4`, then imperfect hidden layers re-instated).\n")

    L.append("## 1. Final fc5 vs victim — last-layer extraction fidelity (the headline)\n")
    L.append("`sign` = gauge-invariant per-class sign accuracy; `|cos|` = mean |cos| of "
             "recovered vs true fc5 rows (**1.0 = exact victim output layer**). `rank` = "
             "crypto tie-system rank / target (✓ = full rank).\n")
    L.append("| model | crypto rank | **crypto fc5 sign** | **crypto \\|cos\\|** | LR fc5 sign | LR \\|cos\\| |")
    L.append("|---|---|---|---|---|---|")
    for m in MODELS:
        jc, jl = data[m]['s1_crypto'], data[m]['s1_lr']
        cs, cc = fc5_pre(jc) if jc else (None, None)
        ls, lc = fc5_pre(jl) if jl else (None, None)
        L.append(f"| {m} | {rank_str(jc)} | **{fnum(cs)}** | **{fnum(cc)}** | {fnum(ls)} | {fnum(lc)} |")
    L.append("\n*Crypto recovers the true output-layer **signs exactly** (and the full "
             "weights, |cos|=1.0, when the tie system is full-rank ✓ — the LeakyReLU "
             "dense-class models). The LR fit's fc5 is an arbitrary functional decoder that "
             "does not match the victim.*\n")

    L.append("## 2. Stage-1 end-to-end agreement (imperfect hidden layers, no ML)\n")
    L.append("| model | S1 LR | S1 crypto | canonical (Stage-2 LR, 21/5) |")
    L.append("|---|---|---|---|")
    for m in MODELS:
        L.append(f"| {m} | {_ag(data,m,'s1_lr')} | {_ag(data,m,'s1_crypto')} | "
                 f"{CANON.get(m,{}).get('agree','—')}% |")
    L.append("\n*Stage-1 crypto agreement is low because the correctly-extracted fc5 sits on "
             "imperfect recovered hidden layers with no ML repair yet; LR is higher only "
             "because it fits fc5 to whatever those imperfect features are (distillation). "
             "The ML stage (see full report) closes this.*\n")
    with open(os.path.join(SUITE, "fc5_snapshot_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def write_full(data):
    """Stage-2 post-ML full comparison report."""
    L = []
    L.append("# fc5 FULL report — Stage-2 post-ML (crypto vs LR vs canonical)\n")
    L.append(f"Suite dir: `{os.path.relpath(SUITE)}`. **Stage 2** = Stage 1 + SA-margin "
             "sign search + gradient refine (canonical run_one_model_enhanced.sh flags). "
             "S2 LR reproduces the canonical 21/5 pipeline.\n")

    L.append("## 1. End-to-end agreement (argmax vs victim)\n")
    L.append("| model | canon 21/5 | **S2 crypto** | S2 LR | S1 crypto | S1 LR |")
    L.append("|---|---|---|---|---|---|")
    for m in MODELS:
        L.append(f"| {m} | {CANON.get(m,{}).get('agree','—')}% | **{_ag(data,m,'s2_crypto')}** | "
                 f"{_ag(data,m,'s2_lr')} | {_ag(data,m,'s1_crypto')} | {_ag(data,m,'s1_lr')} |")
    L.append("\n*After the ML stage repairs the hidden layers, both fc5 methods reach the "
             "canonical agreement level; the fc5 method is not the bottleneck for functional "
             "agreement once refinement runs.*\n")

    L.append("## 2. Refine erodes the exact crypto fc5 (Stage-2 crypto, pre vs post refine)\n")
    L.append("| model | fc5 sign pre | fc5 sign post | \\|cos\\| pre | \\|cos\\| post |")
    L.append("|---|---|---|---|---|")
    for m in MODELS:
        j = data[m]['s2_crypto']
        ps, pc = fc5_pre(j) if j else (None, None)
        qs, qc = fc5_post(j) if j else (None, None)
        L.append(f"| {m} | {fnum(ps)} | {fnum(qs)} | {fnum(pc)} | {fnum(qc)} |")
    L.append("\n*Gradient refine distills fc5 against imperfect features, moving it away from "
             "the exact cryptanalytic extraction — so the last-layer fidelity is a Stage-1 / "
             "pre-refine property (freeze fc5 during refine to preserve it).*\n")

    L.append("## Takeaways\n")
    L.append("1. **Crypto is a genuine output-layer extraction** (Stage-1: exact fc5 signs, "
             "|cos|=1.0 when full-rank), whereas LR only fits a functional decoder.")
    L.append("2. **Functional agreement needs the ML stage** to repair imperfect hidden "
             "layers; a correct fc5 alone does not classify well on imperfect features.")
    L.append("3. **Refinement erodes the exact fc5**; read last-layer fidelity pre-refine.")
    with open(os.path.join(SUITE, "fc5_full_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    data = {m: {v: load(m, v) for v in VARIANTS} for m in MODELS}
    write_snapshot(data)
    write_full(data)
    with open(os.path.join(SUITE, "fc5_crypto_vs_lr_table.json"), "w") as f:
        json.dump({m: {v: (data[m][v] and {
            'agreement': g(data[m][v], 'prediction_agreement'),
            'fc5_pre_sign': fc5_pre(data[m][v])[0], 'fc5_pre_cos': fc5_pre(data[m][v])[1],
            'fc5_post_sign': fc5_post(data[m][v])[0], 'fc5_post_cos': fc5_post(data[m][v])[1],
            'crypto_rank': rank_str(data[m][v]),
        }) for v in VARIANTS} for m in MODELS}, f, indent=2)
    print(f"Wrote fc5_snapshot_report.md, fc5_full_report.md, fc5_crypto_vs_lr_table.json in {SUITE}")


if __name__ == "__main__":
    main()
