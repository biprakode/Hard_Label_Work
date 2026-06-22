#!/usr/bin/env bash
# Resume cifar_lr extraction after pause.
# Skips STEPs 0-5 (Phase 1 already on disk). Runs STEP 6 (sign recovery) +
# STEP 7 (Phase 3 reconstruct/refine with latest flag set) + STEP 8 (report).
#
# Pre-conditions verified before pause:
#   - 80 dual pickles in signature_recovery/exp/1/
#   - 4 cluster pickles in signature_recovery/exp/
#   - weight_recovery layer_{0,1,2,3} dirs populated (812/832 neurons)
#   - 821 per-neuron .npy files in sign_recovery/layer_neuron_npys/
#   - LEAKY_ALPHA = 0.01 in all 4 touchpoints
#   - arch flags all False (CIFAR full)

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY="${PYTHON_BIN:-/home/biprarshi/miniconda3/envs/MLenv/bin/python3}"
LOG=/tmp/cifar_lr/resume.log
mkdir -p /tmp/cifar_lr
TAG=full_leakyrelu

# Per-arch tuning (mirrors run_one_model_enhanced.sh for arch=full)
REFINE_EPOCHS=500
SIGN_RESTARTS=4
SIGN_PAIR=8
SIGN_CYCLES=3

echo "==================================================" | tee "$LOG"
echo "RESUME cifar_lr: $(date)"                          | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

# Sanity — bail if any expected Phase-1 output is missing or LEAKY_ALPHA drifted
n_duals=$(ls signature_recovery/exp/1/*.p 2>/dev/null | wc -l)
n_npys=$(ls sign_recovery/layer_neuron_npys/*.npy 2>/dev/null | wc -l)
if [ "$n_duals" -lt 80 ] || [ "$n_npys" -lt 800 ]; then
    echo "ABORT: Phase 1 outputs incomplete (duals=$n_duals npys=$n_npys). Refusing to resume." | tee -a "$LOG"
    exit 1
fi
for f in signature_recovery/utils.py sign_recovery/sign_recovery.py \
         sign_recovery/batched_sign_recovery.py analysis/extraction_pipeline/config.py; do
    alpha=$(grep -E "^LEAKY_ALPHA\s*=" "$f" | head -1 | awk -F= '{print $2}' | tr -d ' ')
    if [ "$alpha" != "0.01" ]; then
        echo "ABORT: LEAKY_ALPHA != 0.01 in $f (got $alpha)" | tee -a "$LOG"
        exit 1
    fi
done
echo "  preflight OK: duals=$n_duals, npys=$n_npys, LEAKY_ALPHA=0.01 in all 4 files" | tee -a "$LOG"

# ---------- STEP 6 — sign recovery (re-runs all layers, idempotent overwrite) ----------
echo "=== [6] batched_sign_recovery.py ===" | tee -a "$LOG"
t6_start=$(date +%s)
"$PY" sign_recovery/batched_sign_recovery.py > "/tmp/${TAG}_sign.log" 2>&1 || \
    echo "  WARNING: sign recovery exited non-zero" | tee -a "$LOG"
tail -10 "/tmp/${TAG}_sign.log" | tee -a "$LOG" || true
t6=$(( $(date +%s) - t6_start ))
echo "  duration: ${t6}s" | tee -a "$LOG"

# ---------- STEP 7 — Phase 3 enhanced (latest flag set) ----------
echo "=== [7] Phase 3 enhanced (--eval-on-test3 --train-union-test12 + Fix B/C) ===" | tee -a "$LOG"
t7_start=$(date +%s)
"$PY" analysis/run_extraction.py --full --from-scratch --refine \
    --refine-epochs "$REFINE_EPOCHS" \
    --refine-weight-decay 1e-4 \
    --refine-cosine-lr \
    --early-stop --patience 5 --eval-every 10 \
    --eval-on-test3 \
    --train-union-test12 \
    --sign-restarts "$SIGN_RESTARTS" \
    --sign-pair-lookahead "$SIGN_PAIR" \
    --sign-refine-cycles "$SIGN_CYCLES" \
    > "/tmp/${TAG}_phase3.log" 2>&1
grep -E "recovered|accuracy|agreement|EXTRACTION|Saved|refine\]|cycle|fc5|pair|X_test3" \
    "/tmp/${TAG}_phase3.log" | tail -40 | tee -a "$LOG"
t7=$(( $(date +%s) - t7_start ))
echo "  duration: ${t7}s" | tee -a "$LOG"

# ---------- STEP 8 — report ----------
echo "=== [8] emit report ===" | tee -a "$LOG"
REPORTS_DIR="$HERE/../../paper_notes/section3/reports"
mkdir -p "$REPORTS_DIR"
cp results/reconstructed_models/extraction_metrics.json \
   "$REPORTS_DIR/${TAG}_extraction_metrics.json" 2>/dev/null || true
echo "  metrics copied to $REPORTS_DIR/${TAG}_extraction_metrics.json" | tee -a "$LOG"

echo "==================================================" | tee -a "$LOG"
echo "RESUME COMPLETE: $(date)"                          | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
