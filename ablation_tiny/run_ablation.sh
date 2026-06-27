#!/usr/bin/env bash
#
# run_ablation.sh — ONE-COMMAND additive Phase-3 ablation on the 6 make_blobs victims.
#
# A reviewer fires this single script. For each victim it:
#   (1) runs the canonical end-to-end attack via run_one_model_enhanced.sh
#       (Phase 1 dual search -> Phase 2 sign recovery -> Phase 3 SA+margin
#       reconstruction + step-9 scorecard). This produces the *headline* artifacts
#       and the on-disk Phase-1/2 outputs (unsigned weights + signs).
#   (2) runs ablation_harness.py, which REUSES those Phase-1/2 artifacts and
#       re-runs Phase 3 staged at 5 cumulative checkpoints + a distillation
#       baseline, evaluating every stage on the held-out X_test3.
# Finally it aggregates all per-victim JSONs into ABLATION_RESULTS.md.
#
# The harness imports the pipeline functions read-only; no method code is modified.
#
# Usage:
#   ./run_ablation.sh                 # all 6 victims (fast-first ordering)
#   ./run_ablation.sh tiniest tinier  # subset by arch name (tiniest|tinier|tiny)
#   SKIP_DRIVER=1 ./run_ablation.sh   # reuse already-on-disk Phase-1/2 (debug only)
#
# Env:
#   PYTHON_BIN    python interpreter (default: MLenv conda python)
#   SKIP_DRIVER   if =1, do NOT re-run the enhanced driver (assumes artifacts exist)
#   SIGN_NTHREADS Phase-2 worker count (default 6; set 28 on a big box)
set -euo pipefail

ABLATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ablation_tiny lives inside the repo; HLW is its parent (fallback: sibling repo).
if [ -d "$ABLATION_DIR/../analysis" ]; then HLW="$(cd "$ABLATION_DIR/.." && pwd)";
else HLW="$(cd "$ABLATION_DIR/../Hard_Label_Work" && pwd)"; fi
PY="${PYTHON_BIN:-/home/biprarshi/miniconda3/envs/MLenv/bin/python3}"
export PYTHON_BIN="$PY"

RESULTS_DIR="$ABLATION_DIR/results_json"
LOG_DIR="$ABLATION_DIR/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Phase-2 worker-count adaptation (resource constant, NOT a method change).
# batched_sign_recovery.py hardcodes nThreads=28 (sized for a 56-thread cloud
# box). On a smaller machine that spawns 28 TF processes and swap-thrashes,
# which stalls the slower Leaky-ReLU sign recovery. We temporarily lower it to
# SIGN_NTHREADS for the duration of this run and restore the file byte-for-byte
# on exit (trap). This changes NO measured quantity — only parallelism. Set
# SIGN_NTHREADS=28 to keep the original (e.g. when running on a big box).
SIGN_NTHREADS="${SIGN_NTHREADS:-6}"
BSR="$HLW/sign_recovery/batched_sign_recovery.py"
BSR_BAK="$(mktemp)"
cp "$BSR" "$BSR_BAK"
restore_bsr() { cp "$BSR_BAK" "$BSR"; rm -f "$BSR_BAK"; echo "[cleanup] restored nThreads in batched_sign_recovery.py"; }
trap restore_bsr EXIT INT TERM
sed -i -E "s/^nThreads([[:space:]]*)=([[:space:]]*)[0-9]+/nThreads\1=\2${SIGN_NTHREADS}/" "$BSR"
echo "[setup] batched_sign_recovery nThreads -> ${SIGN_NTHREADS} (original restored on exit)"

# arch order: fast-first (tiniest, tinier) then the heavier tiny pair.
ARCHES=("$@")
if [ "${#ARCHES[@]}" -eq 0 ]; then
    ARCHES=(tiniest tinier tiny)
fi
ACTS=(relu leakyrelu)

# driver arch name -> harness arch_key ("tiny" victim == makeblobs 64x64 path)
harness_arch() {
    case "$1" in
        tiniest) echo tiniest ;;
        tinier)  echo tinier  ;;
        tiny)    echo makeblobs ;;
        *) echo "unknown arch $1" >&2; exit 1 ;;
    esac
}

echo "=========================================================================="
echo " ADDITIVE PHASE-3 ABLATION"
echo "   python : $PY"
echo "   arches : ${ARCHES[*]}   acts: ${ACTS[*]}"
echo "   driver : $([ "${SKIP_DRIVER:-0}" = 1 ] && echo SKIPPED || echo run_one_model_enhanced.sh)"
echo "=========================================================================="

for ARCH in "${ARCHES[@]}"; do
    HARCH="$(harness_arch "$ARCH")"
    for ACT in "${ACTS[@]}"; do
        TAG="${ARCH}_${ACT}"
        echo ""
        echo "######################## VICTIM: $TAG ########################"
        t0=$(date +%s)

        if [ "${SKIP_DRIVER:-0}" != "1" ]; then
            echo "--- [1/2] canonical end-to-end attack (headline + Phase-1/2 artifacts) ---"
            ( cd "$HLW" && "$HLW/run_one_model_enhanced.sh" "$ARCH" "$ACT" ) \
                > "$LOG_DIR/${TAG}_driver.log" 2>&1 \
                || { echo "  driver FAILED for $TAG — see $LOG_DIR/${TAG}_driver.log"; continue; }
            echo "    driver done ($(( $(date +%s) - t0 ))s); log: $LOG_DIR/${TAG}_driver.log"
        else
            echo "--- [1/2] SKIP_DRIVER=1: reusing on-disk Phase-1/2 artifacts ---"
        fi

        echo "--- [2/2] staged ablation harness (stages 0-4 + distillation, eval on X_test3) ---"
        ( cd "$HLW" && "$PY" "$ABLATION_DIR/ablation_harness.py" \
            --arch "$HARCH" --act "$ACT" \
            --out "$RESULTS_DIR/${TAG}.json" ) \
            2>&1 | tee "$LOG_DIR/${TAG}_harness.log" \
            || { echo "  harness FAILED for $TAG — see $LOG_DIR/${TAG}_harness.log"; continue; }
        echo "    victim $TAG complete in $(( $(date +%s) - t0 ))s"
    done
done

echo ""
echo "=========================================================================="
echo " Aggregating per-victim JSONs -> ABLATION_RESULTS.md"
echo "=========================================================================="
( cd "$HLW" && "$PY" "$ABLATION_DIR/aggregate_ablation.py" \
    --results-dir "$RESULTS_DIR" \
    --out "$ABLATION_DIR/ABLATION_RESULTS.md" )

echo ""
echo "DONE. Report: $ABLATION_DIR/ABLATION_RESULTS.md"
