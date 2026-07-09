#!/usr/bin/env bash
# Generic driver: runs both arms (ON canonical / OFF honest) for all six
# make_blobs victims for a given cheat flag, evaluating Stage-0 (Phase 1+2
# raw) via ablation_harness.py immediately after each driver run (before the
# next run's STEP 0 clean wipes the on-disk artifacts), and archiving
# artifacts + per-layer recovery counts.
#
# Usage: run_one_cheat_sweep.sh <FLAG_NAME> <cheat_report_dir_name>
# e.g.   run_one_cheat_sweep.sh HONEST_BOUNDARY_REFINE boundary_refinement
set -uo pipefail

FLAG_NAME="${1:?flag name required}"
REPORT_DIR="${2:?report dir name required}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
HLW="$(pwd)"
PY="/home/biprarshi/miniconda3/envs/MLenv/bin/python3"
OUT="$HLW/cheating_ablation/reports/$REPORT_DIR"
mkdir -p "$OUT/raw" "$OUT/logs"

# Resource-only adaptation (not a method change), same rationale/pattern as
# ablation_tiny/run_ablation.sh's SIGN_NTHREADS: batched_sign_recovery.py
# hardcodes nThreads=48 (sized for a 56-thread cloud box). On this dev box
# that spawned 48 processes and one worker hung for 200+ minutes while the
# other 47 sat idle, stalling the whole sweep -- confirmed live during this
# study's NO_SIG_CHEAT rerun. Lower it for the duration of this sweep, restore
# byte-for-byte on exit.
BSR="$HLW/sign_recovery/batched_sign_recovery.py"
BSR_BAK="$(mktemp)"
cp "$BSR" "$BSR_BAK"
restore_bsr() { cp "$BSR_BAK" "$BSR"; rm -f "$BSR_BAK"; echo "[cleanup] restored nThreads in batched_sign_recovery.py"; }
trap restore_bsr EXIT INT TERM
sed -i -E "s/^nThreads([[:space:]]*)=([[:space:]]*)[0-9]+/nThreads\1=\2 6/" "$BSR"
echo "[setup] batched_sign_recovery nThreads -> 6 (original restored on exit)"

declare -A HKEY=( [tiniest]=tiniest [tinier]=tinier [tiny]=makeblobs )

run_arm() {
    local ARCH="$1" ACT="$2" ARM="$3" FLAGVAL="$4"
    local TAG="${ARCH}_${ACT}"
    echo "=== $TAG ARM=$ARM ($FLAG_NAME=$FLAGVAL) ==="
    local t0=$(date +%s)
    env STOP_AFTER_PHASE2=1 "$FLAG_NAME=$FLAGVAL" "$HLW/run_one_model_enhanced.sh" "$ARCH" "$ACT" \
        > "$OUT/logs/${TAG}_${ARM}.log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "  driver rc=$rc  ${dt}s"
    "$PY" "$HLW/ablation_tiny/ablation_harness.py" --arch "${HKEY[$ARCH]}" --act "$ACT" \
        --out "$OUT/raw/${TAG}_${ARM}_stage0.json" \
        > "$OUT/logs/${TAG}_${ARM}_harness.log" 2>&1
    mkdir -p "$OUT/raw/${TAG}_${ARM}_artifacts"
    cp -r "$HLW/signature_recovery/outputs/model_weights/Vrelu" "$OUT/raw/${TAG}_${ARM}_artifacts/" 2>/dev/null
    cp -r "$HLW/results/sign_recovery" "$OUT/raw/${TAG}_${ARM}_artifacts/" 2>/dev/null
    grep -E "layer [0-9]+: recovered" "$OUT/logs/${TAG}_${ARM}.log" | tee "$OUT/raw/${TAG}_${ARM}_recovery_counts.txt"
    echo "  archived."
}

for pair in "tiniest relu" "tiniest leakyrelu" "tinier relu" "tinier leakyrelu" "tiny relu" "tiny leakyrelu"; do
    set -- $pair; ARCH="$1"; ACT="$2"
    run_arm "$ARCH" "$ACT" ON 0
    run_arm "$ARCH" "$ACT" OFF 1
done
echo "ALL DONE ($REPORT_DIR)"
