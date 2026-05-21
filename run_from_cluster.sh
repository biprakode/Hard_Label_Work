#!/usr/bin/env bash
# Resume an extraction pipeline from the cluster step onwards.
# Use this AFTER find_duals has produced exp/1/duals_*.p files and you've restarted
# the machine to free memory.
#
# Usage:  ./run_from_cluster.sh <arch> <activation>
#   arch:        tiniest | tinier | tiny
#   activation:  relu    | leakyrelu
#
# Skips:
#   - the clean step (so existing exp/1 dual files are preserved)
#   - the find_duals stage (assumed already complete)
# Runs:
#   - cluster_dual_points_stream.py
#   - generate_dual_neuron.py
#   - recover_weights.py × 4 layers
#   - batched_sign_recovery.py
#   - Phase 3 reconstruct + refine
#   - compare_true_vs_extracted_v2.py
#
# Does NOT change LEAKY_ALPHA or arch flags — assumes they were configured by an
# earlier run_one_model.sh that completed find_duals.
set -euo pipefail

ARCH="${1:?arch required}"
ACT="${2:?activation required}"
TAG="${ARCH}_${ACT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/biprarshi/miniconda3/envs/MLenv/bin/python3}"
NOTES_DIR="$(cd "$HERE/../.." && pwd)/paper_notes/section3"
REPORTS_DIR="$NOTES_DIR/reports"
mkdir -p "$REPORTS_DIR"
LOG="$NOTES_DIR/run_log.txt"

echo ""                                                              | tee -a "$LOG"
echo "=================================================="           | tee -a "$LOG"
echo "RESUMING $TAG FROM CLUSTER STAGE   ($(date -u))"               | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

# Sanity check — confirm find_duals output exists
nfiles=$(ls "$HERE/signature_recovery/exp/1" 2>/dev/null | wc -l)
echo "Found $nfiles dual-point files in exp/1" | tee -a "$LOG"
if [ "$nfiles" -lt 10 ]; then
    echo "ERROR: too few dual files ($nfiles). Re-run find_duals first." | tee -a "$LOG"
    exit 1
fi

# Confirm LEAKY_ALPHA and arch flags are sane
"$PY" -c "
import sys
sys.path.insert(0,'$HERE/signature_recovery')
import utils
print('  utils.LAYER_SIZES =', utils.LAYER_SIZES)
print('  utils.LEAKY_ALPHA =', utils.LEAKY_ALPHA)
print('  utils.MODEL_PATH =', utils.MODEL_PATH)
import os
assert os.path.exists(utils.MODEL_PATH), 'oracle model file missing'
"

t_total_start=$(date +%s)

# ---------- STEP 3 — cluster ----------
echo "=== [3] cluster_dual_points_stream.py ===" | tee -a "$LOG"
t3_start=$(date +%s)
cd "$HERE/signature_recovery"
"$PY" cluster_dual_points_stream.py 2>&1 | tail -10 | tee -a "$LOG"
t3=$(( $(date +%s) - t3_start ))
echo "  duration: ${t3}s" | tee -a "$LOG"

# ---------- STEP 4 — generate per-neuron dual files ----------
echo "=== [4] generate_dual_neuron.py ===" | tee -a "$LOG"
t4_start=$(date +%s)
"$PY" generate_dual_neuron.py 2>&1 | tail -5 | tee -a "$LOG"
t4=$(( $(date +%s) - t4_start ))
echo "  duration: ${t4}s" | tee -a "$LOG"

# ---------- STEP 5 — recover_weights per layer ----------
echo "=== [5] recover_weights.py (layers 0..3) ===" | tee -a "$LOG"
t5_start=$(date +%s)
for L in 0 1 2 3; do
    "$PY" recover_weights.py "$L" > "/tmp/${TAG}_recover_${L}.log" 2>&1
    recovered=$(grep -c "Successfully extracted neuron" "/tmp/${TAG}_recover_${L}.log" || true)
    failed=$(grep -c "Failed to identify recovered neuron" "/tmp/${TAG}_recover_${L}.log" || true)
    echo "  layer $L: recovered=$recovered, failed_match=$failed" | tee -a "$LOG"
done
t5=$(( $(date +%s) - t5_start ))
echo "  duration: ${t5}s" | tee -a "$LOG"

# ---------- STEP 6 — batched sign recovery ----------
echo "=== [6] batched_sign_recovery.py ===" | tee -a "$LOG"
t6_start=$(date +%s)
cd "$HERE"
"$PY" sign_recovery/batched_sign_recovery.py > "/tmp/${TAG}_sign.log" 2>&1 || \
    echo "  WARNING: sign recovery exited with non-zero (may have partial output)" | tee -a "$LOG"
tail -10 "/tmp/${TAG}_sign.log" | tee -a "$LOG" || true
t6=$(( $(date +%s) - t6_start ))
echo "  duration: ${t6}s" | tee -a "$LOG"

# ---------- STEP 7 — Phase 3 ----------
echo "=== [7] Phase 3 (run_extraction.py --from-scratch --refine) ===" | tee -a "$LOG"
t7_start=$(date +%s)
case "$ARCH" in
    tiniest) ARCH_FLAG='--tiniest' ;;
    tinier)  ARCH_FLAG='--tinier'  ;;
    tiny)    ARCH_FLAG='--makeblobs' ;;
esac
"$PY" analysis/run_extraction.py $ARCH_FLAG --from-scratch --refine --refine-epochs 1000 \
    > "/tmp/${TAG}_phase3.log" 2>&1
grep -E "recovered|accuracy|agreement|EXTRACTION|Saved|refine\]" "/tmp/${TAG}_phase3.log" | tail -25 | tee -a "$LOG"
t7=$(( $(date +%s) - t7_start ))
echo "  duration: ${t7}s" | tee -a "$LOG"

# ---------- STEP 8 — Report ----------
echo "=== [8] compare_true_vs_extracted_v2 ===" | tee -a "$LOG"
TIMINGS_JSON="/tmp/${TAG}_timings.json"
cat > "$TIMINGS_JSON" <<EOF
{"find_duals": -1, "cluster": $t3, "generate_dual_neuron": $t4, "recover_weights": $t5, "sign_recovery": $t6, "phase3": $t7}
EOF
cp "$HERE/results/reconstructed_models/extraction_metrics.json" \
   "$REPORTS_DIR/${TAG}_extraction_metrics.json" 2>/dev/null || true
"$PY" analysis/compare_true_vs_extracted_v2.py \
    --arch "$ARCH" --activation "$ACT" \
    --output "$REPORTS_DIR/${TAG}_true_vs_extracted.md" \
    --metrics-json "$REPORTS_DIR/${TAG}_extraction_metrics.json" \
    --timings "$TIMINGS_JSON" 2>&1 | tee -a "$LOG"

t_total=$(( $(date +%s) - t_total_start ))
echo "=== TOTAL post-find_duals wall time for $TAG: ${t_total}s ===" | tee -a "$LOG"

SUMMARY=$("$PY" -c "
import json
j = json.load(open('$REPORTS_DIR/${TAG}_true_vs_extracted.json'))
print(f\"{('$TAG').ljust(18)}  recovered={j['total_recovered']}/{j['total_neurons']}  |cos|={j['overall_abscos']:.3f}  sign_acc={j['overall_sign_acc']:.3f}  ext_acc={j['reconstructed_acc_x_test2']*100:.2f}%  agree={j['prediction_agreement_x_test2']*100:.2f}%  (resumed; post-duals total=${t_total}s)\")
")
echo "$SUMMARY" | tee -a "$LOG"
