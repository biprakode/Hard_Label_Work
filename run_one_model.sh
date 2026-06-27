#!/usr/bin/env bash
# Drives a single end-to-end extraction for a (arch, activation) pair.
# Usage:  ./run_one_model.sh <arch> <activation> [DUAL_ITERS]
#   arch:        tiniest | tinier | tiny
#   activation:  relu    | leakyrelu
#   DUAL_ITERS:  defaults — tiniest=9, tinier=50, tiny=1000
#
# Side effects (all confined to Hard_Label_Work/):
#   signature_recovery/exp/                       — cleaned then written
#   signature_recovery/outputs/model_weights/...  — cleaned then written
#   sign_recovery/layer_neuron_npys/              — cleaned then written
#   results/sign_recovery/                        — cleaned then written
#   results/reconstructed_models/                 — cleaned then written
#
# Writes:
#   paper_notes/section3/reports/<tag>_true_vs_extracted.md
#   paper_notes/section3/reports/<tag>_true_vs_extracted.json
#   paper_notes/section3/run_log.txt              (appended)
set -euo pipefail

ARCH="${1:?arch required}"
ACT="${2:?activation required}"
DUAL_ITERS="${3:-}"
case "$ARCH" in
    tiniest) : "${DUAL_ITERS:=9}"  ;;
    tinier)  : "${DUAL_ITERS:=50}" ;;
    tiny)    : "${DUAL_ITERS:=1000}" ;;
    *) echo "bad arch: $ARCH"; exit 1 ;;
esac
case "$ACT" in
    relu|leakyrelu) ;;
    *) echo "bad activation: $ACT"; exit 1 ;;
esac

TAG="${ARCH}_${ACT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/biprarshi/miniconda3/envs/MLenv/bin/python3}"
NOTES_DIR="$(cd "$HERE/../.." && pwd)/paper_notes/section3"
REPORTS_DIR="$NOTES_DIR/reports"
mkdir -p "$REPORTS_DIR" "$NOTES_DIR"
LOG="$NOTES_DIR/run_log.txt"

echo ""                                      | tee -a "$LOG"
echo "=================================================="          | tee -a "$LOG"
echo "MODEL: $TAG    ($(date -u))"                                  | tee -a "$LOG"
echo "  arch=$ARCH  activation=$ACT  DUAL_ITERS=$DUAL_ITERS"        | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

t_total_start=$(date +%s)

# ---------- STEP 0 — CLEAN ----------
echo "=== [0] cleaning previous intermediate state ===" | tee -a "$LOG"
rm -rf "$HERE/signature_recovery/exp/1"
rm -f  "$HERE/signature_recovery/exp/1-cluster-"*.p
rm -rf "$HERE/signature_recovery/outputs/model_weights/Vrelu/layer_"*
rm -rf "$HERE/sign_recovery/layer_neuron_npys"
rm -f  "$HERE/results/sign_recovery/"*
rm -f  "$HERE/results/reconstructed_models/reconstructed_"*
mkdir -p "$HERE/signature_recovery/exp/1" \
         "$HERE/signature_recovery/outputs/model_weights/Vrelu" \
         "$HERE/sign_recovery/layer_neuron_npys" \
         "$HERE/results/sign_recovery" \
         "$HERE/results/reconstructed_models"

# ---------- STEP 1 — CONFIGURE (LEAKY_ALPHA + arch booleans) ----------
echo "=== [1] configure LEAKY_ALPHA + arch flags ===" | tee -a "$LOG"
ALPHA=$([ "$ACT" = "leakyrelu" ] && echo "0.01" || echo "0.0")

HERE="$HERE" ALPHA="$ALPHA" ARCH="$ARCH" "$PY" - <<'PY'
import os, re, pathlib
HERE = pathlib.Path(os.environ['HERE'])
ALPHA = os.environ['ALPHA']
ARCH = os.environ['ARCH']

# LEAKY_ALPHA in 4 files
targets = [
    (HERE/'signature_recovery/utils.py',                r'^LEAKY_ALPHA\s*=\s*\S+'),
    (HERE/'sign_recovery/sign_recovery.py',             r'^LEAKY_ALPHA\s*=\s*\S+'),
    (HERE/'sign_recovery/batched_sign_recovery.py',     r'^LEAKY_ALPHA\s*=\s*\S+'),
    (HERE/'analysis/extraction_pipeline/config.py',     r'^LEAKY_ALPHA\s*=\s*\S+'),
]
for path, pat in targets:
    txt = path.read_text()
    txt2 = re.sub(pat, f"LEAKY_ALPHA = {ALPHA}", txt, count=1, flags=re.M)
    if txt != txt2:
        path.write_text(txt2)
    new_line = [l for l in txt2.split('\n') if l.strip().startswith('LEAKY_ALPHA')][0]
    print(f"  {path.relative_to(HERE)}: {new_line.strip()}")

# Arch booleans in utils.py and batched_sign_recovery.py
arch_flags = {
    'tiniest': {'TINIEST': True,  'TINIER': False, 'TINY': True},
    'tinier':  {'TINIEST': False, 'TINIER': True,  'TINY': True},
    'tiny':    {'TINIEST': False, 'TINIER': False, 'TINY': True},
}[ARCH]
for path in [HERE/'signature_recovery/utils.py', HERE/'sign_recovery/batched_sign_recovery.py']:
    txt = path.read_text()
    for k, v in arch_flags.items():
        # match optional spaces around equals to also handle `LEAKY_ALPHA              = ...` style
        txt = re.sub(rf'^{k}\s*=\s*(True|False)\b', f'{k} = {v}', txt, count=1, flags=re.M)
    path.write_text(txt)
print(f"  arch flags set for {ARCH}: {arch_flags}")
PY

# Verify activation matches
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

# ---------- STEP 2 — find_duals ----------
echo "=== [2] find_duals x $DUAL_ITERS ===" | tee -a "$LOG"
t2_start=$(date +%s)
cd "$HERE/signature_recovery"
for i in $(seq 1 "$DUAL_ITERS"); do
    "$PY" find_duals.py > "/tmp/${TAG}_duals_${i}.log" 2>&1
    if [ $((i % 25)) -eq 0 ] || [ $i -eq $DUAL_ITERS ]; then
        files=$(ls "$HERE/signature_recovery/exp/1" 2>/dev/null | wc -l)
        echo "  iter $i/$DUAL_ITERS  files=$files" | tee -a "$LOG"
    fi
done
t2_end=$(date +%s)
t2=$((t2_end - t2_start))
echo "  duration: ${t2}s" | tee -a "$LOG"

# ---------- STEP 3 — cluster ----------
echo "=== [3] cluster_dual_points_stream.py ===" | tee -a "$LOG"
t3_start=$(date +%s)
"$PY" cluster_dual_points_stream.py 2>&1 | tail -10 | tee -a "$LOG"
t3=$(( $(date +%s) - t3_start ))
echo "  duration: ${t3}s" | tee -a "$LOG"

# ---------- STEP 4 — generate per-neuron dual files ----------
echo "=== [4] generate_dual_neuron.py ===" | tee -a "$LOG"
t4_start=$(date +%s)
"$PY" generate_dual_neuron.py 2>&1 | tail -5 | tee -a "$LOG"
t4=$(( $(date +%s) - t4_start ))
echo "  duration: ${t4}s" | tee -a "$LOG"

# ---------- STEP 5 — recover weights per layer ----------
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

# ---------- STEP 6 — sign recovery ----------
echo "=== [6] batched_sign_recovery.py ===" | tee -a "$LOG"
t6_start=$(date +%s)
cd "$HERE"
"$PY" sign_recovery/batched_sign_recovery.py > "/tmp/${TAG}_sign.log" 2>&1 || \
    echo "  WARNING: sign recovery exited with non-zero (may have partial output)" | tee -a "$LOG"
tail -10 "/tmp/${TAG}_sign.log" | tee -a "$LOG" || true
t6=$(( $(date +%s) - t6_start ))
echo "  duration: ${t6}s" | tee -a "$LOG"

# ---------- STEP 7 — Phase 3 reconstruct + refine ----------
echo "=== [7] Phase 3 (run_extraction.py --from-scratch --refine) ===" | tee -a "$LOG"
t7_start=$(date +%s)
case "$ARCH" in
    tiniest) ARCH_FLAG='--tiniest' ;;
    tinier)  ARCH_FLAG='--tinier'  ;;
    tiny)    ARCH_FLAG='--makeblobs' ;;
esac
# MetaHeuristic combinatorial sign search (SA+margin default; SIGN_METHOD=pt|greedy|tabu to override)
"$PY" analysis/run_extraction.py $ARCH_FLAG --from-scratch --refine --refine-epochs 1000 \
    --sign-search-method "${SIGN_METHOD:-sa}" --sign-search-objective "${SIGN_OBJ:-margin}" \
    > "/tmp/${TAG}_phase3.log" 2>&1
grep -E "recovered|accuracy|agreement|EXTRACTION|Saved|refine\]" "/tmp/${TAG}_phase3.log" | tail -25 | tee -a "$LOG"
t7=$(( $(date +%s) - t7_start ))
echo "  duration: ${t7}s" | tee -a "$LOG"

# ---------- STEP 8 — Generate report ----------
echo "=== [8] compare_true_vs_extracted_v2 ===" | tee -a "$LOG"
TIMINGS_JSON="/tmp/${TAG}_timings.json"
cat > "$TIMINGS_JSON" <<EOF
{"find_duals": $t2, "cluster": $t3, "generate_dual_neuron": $t4, "recover_weights": $t5, "sign_recovery": $t6, "phase3": $t7}
EOF
# Archive extraction_metrics.json under the per-model name BEFORE the next run wipes it.
cp "$HERE/results/reconstructed_models/extraction_metrics.json" \
   "$REPORTS_DIR/${TAG}_extraction_metrics.json" 2>/dev/null || true

"$PY" analysis/compare_true_vs_extracted_v2.py \
    --arch "$ARCH" --activation "$ACT" \
    --output "$REPORTS_DIR/${TAG}_true_vs_extracted.md" \
    --metrics-json "$REPORTS_DIR/${TAG}_extraction_metrics.json" \
    --timings "$TIMINGS_JSON" 2>&1 | tee -a "$LOG"

t_total=$(( $(date +%s) - t_total_start ))
echo "=== TOTAL wall time for $TAG: ${t_total}s ===" | tee -a "$LOG"

# Append one-line summary to run_log
SUMMARY=$("$PY" -c "
import json
j = json.load(open('$REPORTS_DIR/${TAG}_true_vs_extracted.json'))
print(f\"{('$TAG').ljust(18)}  recovered={j['total_recovered']}/{j['total_neurons']}  |cos|={j['overall_abscos']:.3f}  sign_acc={j['overall_sign_acc']:.3f}  ext_acc={j['reconstructed_acc_x_test2']*100:.2f}%  agree={j['prediction_agreement_x_test2']*100:.2f}%  total=${t_total}s\")
")
echo "$SUMMARY" | tee -a "$LOG"
