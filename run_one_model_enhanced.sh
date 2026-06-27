#!/usr/bin/env bash
# Drives a single end-to-end extraction for a (arch, activation) pair using the
# enhanced CIFAR Phase-3 methodology (Fix A: X_test3 honest eval + train-union,
# Fix B: AdamW+cosine+early-stop watchdog, Fix C: fc5-before-sign, restarts,
# pair-flip, cycles). Phase 1 dual search uses the batched-torch parallel finder
# (signature_recovery/torch_impl/parallel_duals.py) — same finder the CIFAR
# flagship 2026-06-04/06-05 runs used.
#
# Usage:  ./run_one_model_enhanced.sh <arch> <activation> [DUAL_ITERS]
#   arch:        tiniest | tinier | tiny | full
#   activation:  relu    | leakyrelu
#   DUAL_ITERS:  parallel-finder ROUND count (each round emits TARGET triplets:
#                tiniest=3000, tinier=2000, tiny=10000, full=10000). Defaults:
#                tiniest=6, tinier=8, tiny=20, full=80.
#
# Note: arch=full is the CIFAR-10 flagship 3072-256-256-256-64-10. Phase 1 dual
# search uses smaller batch (48) and fewer workers (5) to fit the 22 Gi RAM box;
# total wall budget is ~5-8 hours including Phase 2 sign recovery.
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
    tiniest) : "${DUAL_ITERS:=6}"  ;;
    tinier)  : "${DUAL_ITERS:=8}"  ;;
    tiny)    : "${DUAL_ITERS:=20}" ;;
    full)    : "${DUAL_ITERS:=80}" ;;
    *) echo "bad arch: $ARCH"; exit 1 ;;
esac
# Parallel-finder defaults (CIFAR flagship uses smaller batch + fewer workers
# to fit the 22 Gi RAM box; matches the 2026-06-05 flagship run profile).
case "$ARCH" in
    full)    : "${DUAL_WORKERS:=5}"; : "${DUAL_BATCH:=48}"  ;;
    *)       : "${DUAL_WORKERS:=7}"; : "${DUAL_BATCH:=256}" ;;
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

# Per-arch Fix-C tuning. The CIFAR run found restart=0 was as good as 4 on the
# 832-neuron arch (no random restart ever beat current-state). For the smaller
# arches we keep one restart as a regression guard.
case "$ARCH" in
    tiniest) SIGN_RESTARTS=1; SIGN_PAIR=8;  SIGN_CYCLES=3; REFINE_EPOCHS=300 ;;
    tinier)  SIGN_RESTARTS=1; SIGN_PAIR=8;  SIGN_CYCLES=3; REFINE_EPOCHS=500 ;;
    tiny)    SIGN_RESTARTS=2; SIGN_PAIR=8;  SIGN_CYCLES=3; REFINE_EPOCHS=500 ;;
    full)    SIGN_RESTARTS=4; SIGN_PAIR=8;  SIGN_CYCLES=3; REFINE_EPOCHS=500 ;;
esac

# MetaHeuristic / combinatorial sign search: per-layer sign optimizer.
#   greedy (legacy) | tabu | sa (default) | pt    objective: agree | margin (default)
# Default is the metaheuristic combinatorial search SA+margin (the canonical 2026-06-21
# Phase-3 sign step); set SIGN_METHOD=pt for the widest/hardest layers (CIFAR full),
# or SIGN_METHOD=greedy SIGN_OBJ=agree to reproduce legacy behaviour. See README
# "Sign-search methods (MetaHeuristic Sign Search)".
SIGN_METHOD="${SIGN_METHOD:-sa}"
SIGN_OBJ="${SIGN_OBJ:-margin}"

echo ""                                      | tee -a "$LOG"
echo "=================================================="          | tee -a "$LOG"
echo "MODEL (enhanced): $TAG    ($(date -u))"                       | tee -a "$LOG"
echo "  arch=$ARCH  activation=$ACT  DUAL_ITERS=$DUAL_ITERS"        | tee -a "$LOG"
echo "  flags: restarts=$SIGN_RESTARTS pair=$SIGN_PAIR cycles=$SIGN_CYCLES refine=$REFINE_EPOCHS sign_method=$SIGN_METHOD/$SIGN_OBJ" | tee -a "$LOG"
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

arch_flags = {
    'tiniest': {'TINIEST': True,  'TINIER': False, 'TINY': True,  'MAKEBLOBS': True},
    'tinier':  {'TINIEST': False, 'TINIER': True,  'TINY': True,  'MAKEBLOBS': True},
    'tiny':    {'TINIEST': False, 'TINIER': False, 'TINY': True,  'MAKEBLOBS': True},
    'full':    {'TINIEST': False, 'TINIER': False, 'TINY': False, 'MAKEBLOBS': False},
}[ARCH]
for path in [HERE/'signature_recovery/utils.py', HERE/'sign_recovery/batched_sign_recovery.py']:
    txt = path.read_text()
    for k, v in arch_flags.items():
        txt = re.sub(rf'^{k}\s*=\s*(True|False)\b', f'{k} = {v}', txt, count=1, flags=re.M)
    path.write_text(txt)
print(f"  arch flags set for {ARCH}: {arch_flags}")
PY

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

# ---------- STEP 2 — parallel batched dual search (torch impl) ----------
echo "=== [2] parallel_duals.py rounds=$DUAL_ITERS workers=$DUAL_WORKERS batch=$DUAL_BATCH ===" | tee -a "$LOG"
t2_start=$(date +%s)
cd "$HERE/signature_recovery"
"$PY" torch_impl/parallel_duals.py \
    --iterations "$DUAL_ITERS" \
    --workers    "$DUAL_WORKERS" \
    --batch-size "$DUAL_BATCH" \
    --impl       torch \
    > "/tmp/${TAG}_duals.log" 2>&1
grep -E "round [0-9]+ done|finished|impl=|target/round" "/tmp/${TAG}_duals.log" | tail -10 | tee -a "$LOG"
files=$(ls "$HERE/signature_recovery/exp/1" 2>/dev/null | wc -l)
echo "  pickles emitted: $files" | tee -a "$LOG"
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

# ---------- STEP 7 — Phase 3 reconstruct + refine (enhanced CIFAR methodology) ----------
echo "=== [7] Phase 3 enhanced (--eval-on-test3 --train-union-test12 + Fix B/C) ===" | tee -a "$LOG"
t7_start=$(date +%s)
case "$ARCH" in
    tiniest) ARCH_FLAG='--tiniest' ;;
    tinier)  ARCH_FLAG='--tinier'  ;;
    tiny)    ARCH_FLAG='--makeblobs' ;;
    full)    ARCH_FLAG='--full' ;;
esac
"$PY" analysis/run_extraction.py $ARCH_FLAG --from-scratch --refine \
    --refine-epochs "$REFINE_EPOCHS" \
    --refine-weight-decay 1e-4 \
    --refine-cosine-lr \
    --early-stop --patience 5 --eval-every 10 \
    --eval-on-test3 \
    --train-union-test12 \
    --sign-restarts "$SIGN_RESTARTS" \
    --sign-pair-lookahead "$SIGN_PAIR" \
    --sign-refine-cycles "$SIGN_CYCLES" \
    --sign-search-method "$SIGN_METHOD" \
    --sign-search-objective "$SIGN_OBJ" \
    > "/tmp/${TAG}_phase3.log" 2>&1
grep -E "recovered|accuracy|agreement|EXTRACTION|Saved|refine\]|cycle|fc5|pair|X_test3" "/tmp/${TAG}_phase3.log" | tail -40 | tee -a "$LOG"
t7=$(( $(date +%s) - t7_start ))
echo "  duration: ${t7}s" | tee -a "$LOG"

# ---------- STEP 8 — Generate report ----------
echo "=== [8] compare_true_vs_extracted_v2 ===" | tee -a "$LOG"
TIMINGS_JSON="/tmp/${TAG}_timings.json"
cat > "$TIMINGS_JSON" <<EOF
{"find_duals": $t2, "cluster": $t3, "generate_dual_neuron": $t4, "recover_weights": $t5, "sign_recovery": $t6, "phase3": $t7}
EOF
cp "$HERE/results/reconstructed_models/extraction_metrics.json" \
   "$REPORTS_DIR/${TAG}_extraction_metrics.json" 2>/dev/null || true

"$PY" analysis/compare_true_vs_extracted_v2.py \
    --arch "$ARCH" --activation "$ACT" \
    --output "$REPORTS_DIR/${TAG}_true_vs_extracted.md" \
    --metrics-json "$REPORTS_DIR/${TAG}_extraction_metrics.json" \
    --timings "$TIMINGS_JSON" 2>&1 | tee -a "$LOG"

# ---------- STEP 9 — Improved evaluation scorecard (LATEST WORKFLOW) ----------
# Mandatory two-arm extraction-vs-distillation scorecard (Metrics 1-5 + EQS).
# Activation is auto-detected from the extraction metrics, and the distillation
# baseline is auto-built if missing. Writes results/reports/eval_<arch>_<date>.md
# (+ a copy and JSON under Evaluation_Metric_Improve/). Non-fatal on failure.
echo "=== [9] improved evaluation scorecard (analysis/evaluate_extraction_quality.py) ===" | tee -a "$LOG"
t9_start=$(date +%s)
"$PY" analysis/evaluate_extraction_quality.py $ARCH_FLAG 2>&1 | tee -a "$LOG" \
    || echo "  (improved evaluation failed — extraction run still OK)" | tee -a "$LOG"
t9=$(( $(date +%s) - t9_start ))
echo "  duration: ${t9}s" | tee -a "$LOG"

t_total=$(( $(date +%s) - t_total_start ))
echo "=== TOTAL wall time for $TAG: ${t_total}s ===" | tee -a "$LOG"

SUMMARY=$("$PY" -c "
import json
j = json.load(open('$REPORTS_DIR/${TAG}_true_vs_extracted.json'))
tag = j.get('eval_tag', 'X_test2')
print(f\"{('$TAG').ljust(18)}  recovered={j['total_recovered']}/{j['total_neurons']}  |cos|={j['overall_abscos']:.3f}  sign_acc={j['overall_sign_acc']:.3f}  ext_acc={j['reconstructed_acc_eval']*100:.2f}%  agree={j['prediction_agreement_eval']*100:.2f}%  eval={tag}  total=${t_total}s\")
")
echo "$SUMMARY" | tee -a "$LOG"
