#!/usr/bin/env bash
# Pure-distillation baseline for the CIFAR-10 Leaky ReLU flagship.
#
# Mirror of the §8.7 "no-signature" arm: same architecture, same 20K query
# pool (X_test ∪ X_test2), same Fix-B refinement regularisers, same X_test3
# held-out eval — but NO cryptanalytic input. All 832 hidden rows are
# Kaiming-initialised and trainable for the entire refinement run.
#
# Mechanism: wipe the Phase 1+2 disk artifacts so reconstruct_model finds
# "0 recovered" per layer and falls back to Kaiming everywhere; then run
# Phase 3 with --refine-unfreeze so even the (would-be) recovered rows are
# trainable.
#
# Writes:
#   results/reconstructed_models/reconstructed_full_distillation.pth
#   results/reconstructed_models/extraction_metrics_distillation.json
#   paper_notes/section3/reports/cifar_leakyrelu_distillation.md
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/biprarshi/miniconda3/envs/MLenv/bin/python3}"
NOTES_DIR="$(cd "$HERE/../.." && pwd)/paper_notes/section3"
REPORTS_DIR="$NOTES_DIR/reports"
LOG="$NOTES_DIR/run_log.txt"
TAG="cifar_leakyrelu_distillation"

echo ""                                                              | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
echo "DISTILLATION BASELINE: $TAG    ($(date -u))"                   | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

t_total_start=$(date +%s)

# ---------- STEP A — preserve extraction artifacts under suffix --------------
echo "=== [A] preserving extraction outputs under _extraction suffix ===" | tee -a "$LOG"
if [ -f "$HERE/results/reconstructed_models/reconstructed_full.pth" ]; then
    cp "$HERE/results/reconstructed_models/reconstructed_full.pth" \
       "$HERE/results/reconstructed_models/reconstructed_full_extraction.pth"
fi
if [ -f "$HERE/results/reconstructed_models/extraction_metrics.json" ]; then
    cp "$HERE/results/reconstructed_models/extraction_metrics.json" \
       "$HERE/results/reconstructed_models/extraction_metrics_extraction.json"
fi

# ---------- STEP B — wipe Phase 1+2 outputs to force "0 recovered" -----------
echo "=== [B] wiping Phase 1+2 outputs so reconstruct_model sees 0 recovered ===" | tee -a "$LOG"
rm -rf "$HERE/signature_recovery/outputs/model_weights/Vrelu/layer_"*
rm -f  "$HERE/results/sign_recovery/"*
mkdir -p "$HERE/signature_recovery/outputs/model_weights/Vrelu" \
         "$HERE/results/sign_recovery"

# ---------- STEP C — ensure LEAKY_ALPHA + arch flags still match leakyrelu ----
# (Should already be set by the prior extraction run, but defend in case.)
echo "=== [C] verify leakyrelu config ===" | tee -a "$LOG"
"$PY" -c "
import sys
sys.path.insert(0,'$HERE/signature_recovery')
import utils
assert utils.LEAKY_ALPHA == 0.01, f'expected LEAKY_ALPHA=0.01, got {utils.LEAKY_ALPHA}'
assert utils.LAYER_SIZES == [3072, 256, 256, 256, 64, 10], f'wrong arch: {utils.LAYER_SIZES}'
print('  utils.LAYER_SIZES =', utils.LAYER_SIZES)
print('  utils.LEAKY_ALPHA =', utils.LEAKY_ALPHA)
"

# ---------- STEP D — Phase 3 pure distillation -------------------------------
# --refine-unfreeze: ALL hidden rows trainable (no frozen recovered rows since
#                    there are none on disk anyway).
# Same Fix-B regularisers as the extraction run for an apples-to-apples comparison.
echo "=== [D] Phase 3 pure distillation (Kaiming init everywhere, all rows trainable) ===" | tee -a "$LOG"
t_d_start=$(date +%s)
"$PY" analysis/run_extraction.py --full --refine --refine-unfreeze \
    --refine-epochs 500 \
    --refine-weight-decay 1e-4 \
    --refine-cosine-lr \
    --early-stop --patience 5 --eval-every 10 \
    --eval-on-test3 \
    --train-union-test12 \
    > "/tmp/${TAG}_phase3.log" 2>&1
grep -E "recovered|accuracy|agreement|EXTRACTION|Saved|refine\]|X_test3" "/tmp/${TAG}_phase3.log" | tail -25 | tee -a "$LOG"
t_d=$(( $(date +%s) - t_d_start ))
echo "  duration: ${t_d}s" | tee -a "$LOG"

# ---------- STEP E — archive distillation outputs under _distillation suffix --
echo "=== [E] archiving distillation outputs ===" | tee -a "$LOG"
mv "$HERE/results/reconstructed_models/reconstructed_full.pth" \
   "$HERE/results/reconstructed_models/reconstructed_full_distillation.pth"
mv "$HERE/results/reconstructed_models/extraction_metrics.json" \
   "$HERE/results/reconstructed_models/extraction_metrics_distillation.json"

# Restore extraction artifacts as the canonical ones (in case anything else
# expects reconstructed_full.pth).
if [ -f "$HERE/results/reconstructed_models/reconstructed_full_extraction.pth" ]; then
    cp "$HERE/results/reconstructed_models/reconstructed_full_extraction.pth" \
       "$HERE/results/reconstructed_models/reconstructed_full.pth"
fi
if [ -f "$HERE/results/reconstructed_models/extraction_metrics_extraction.json" ]; then
    cp "$HERE/results/reconstructed_models/extraction_metrics_extraction.json" \
       "$HERE/results/reconstructed_models/extraction_metrics.json"
fi

# ---------- STEP F — emit comparison markdown report -------------------------
echo "=== [F] write distillation comparison report ===" | tee -a "$LOG"
"$PY" - <<PY | tee -a "$LOG"
import json, os
from datetime import datetime
BASE = "$HERE"
ext_j = json.load(open(os.path.join(BASE, "results/reconstructed_models/extraction_metrics_extraction.json")))
dis_j = json.load(open(os.path.join(BASE, "results/reconstructed_models/extraction_metrics_distillation.json")))

def pct(x): return f"{x*100:.2f}%"

out = []
out.append("# CIFAR-10 3072-256-256-256-64-10 (LEAKY RELU) - Extraction vs Distillation")
out.append("")
out.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
out.append(f"**Activation:** Leaky ReLU(alpha=0.01)")
out.append(f"**Architecture:** 3072-256-256-256-64-10")
out.append(f"**Eval set:** X_test3 (held-out, seed=10000-19999 CIFAR train, never queried)")
out.append("")
out.append("## Headline")
out.append("")
out.append("| Metric (X_test3) | With signature (extraction) | No signature (distillation) | Oracle |")
out.append("|---|---:|---:|---:|")
out.append(f"| Reconstructed accuracy | {pct(ext_j.get('reconstructed_accuracy',0))} | {pct(dis_j.get('reconstructed_accuracy',0))} | {pct(ext_j.get('true_accuracy',0))} |")
out.append(f"| Prediction agreement vs oracle | {pct(ext_j.get('prediction_agreement',0))} | {pct(dis_j.get('prediction_agreement',0))} | --- |")
out.append(f"| Pre-sign-search accuracy | {pct(ext_j.get('pre_sign_search_accuracy',0))} | {pct(dis_j.get('pre_sign_search_accuracy',0))} | --- |")
out.append("")
out.append("## Extraction configuration")
out.append(f"- recovered: {ext_j.get('recovery_stats',{}).get('recovered_neurons','?')}/{ext_j.get('recovery_stats',{}).get('total_neurons','?')}")
out.append(f"- refinement applied: {ext_j.get('refinement_applied',False)}")
out.append(f"- from_scratch: {ext_j.get('from_scratch',False)}")
out.append("")
out.append("## Distillation configuration")
out.append(f"- recovered (none): {dis_j.get('recovery_stats',{}).get('recovered_neurons','?')}/{dis_j.get('recovery_stats',{}).get('total_neurons','?')}")
out.append(f"- refinement applied: {dis_j.get('refinement_applied',False)}")
out.append(f"- refine_unfreeze: {dis_j.get('refine_unfreeze','?')}")
out.append("")
out.append("## Notes")
out.append("- Both arms used the same 20K oracle query pool (X_test U X_test2).")
out.append("- Both arms used identical Phase-3 regularisers: AdamW(weight_decay=1e-4), CosineAnnealingLR, X_test3-watchdog early-stop (patience=5, eval_every=10).")
out.append("- Held-out eval on X_test3 (5K samples, never queried, never used for sign-flip decisions in either arm).")
out.append("- Distillation arm has Phase 1+2 outputs wiped from disk so reconstruct_model loads 0 recovered per layer.")

path = os.path.join("$REPORTS_DIR", "cifar_leakyrelu_distillation.md")
with open(path, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"Wrote {path}")
PY

# ---------- STEP H — improved evaluation scorecard (non-breaking) -----------
# Both arms now exist on disk, so emit the full metric suite + EQS comparison
# (spec: Evaluation_Metric_Improve/evaluation_metrics_REPORT.md). A failure here
# must NOT fail the baseline run.
echo "=== [H] improved evaluation scorecard (analysis/evaluate_extraction_quality.py) ===" | tee -a "$LOG"
"$PY" "$HERE/analysis/evaluate_extraction_quality.py" --full 2>&1 | tee -a "$LOG" || \
    echo "  (improved evaluation failed — baseline run still OK)" | tee -a "$LOG"

t_total=$(( $(date +%s) - t_total_start ))
echo "=== TOTAL wall time for $TAG: ${t_total}s ===" | tee -a "$LOG"
