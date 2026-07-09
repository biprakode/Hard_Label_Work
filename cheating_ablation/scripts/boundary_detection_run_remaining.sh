#!/usr/bin/env bash
# Runs both arms (ON canonical / OFF honest HONEST_BOUNDARY_DETECT) for the
# remaining make_blobs victims, evaluating Stage-0 (Phase 1+2 raw) via
# ablation_harness.py immediately after each driver run (before the next
# run's STEP 0 clean wipes the on-disk artifacts), and archiving artifacts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
HLW="$(pwd)"
PY="/home/biprarshi/miniconda3/envs/MLenv/bin/python3"
OUT="$HLW/cheating_ablation/reports/boundary_detection"
mkdir -p "$OUT/raw" "$OUT/logs"

declare -A HKEY=( [tiniest]=tiniest [tinier]=tinier [tiny]=makeblobs )

run_arm() {
    local ARCH="$1" ACT="$2" ARM="$3" FLAGVAL="$4"
    local TAG="${ARCH}_${ACT}"
    echo "=== $TAG ARM=$ARM (HONEST_BOUNDARY_DETECT=$FLAGVAL) ==="
    local t0=$(date +%s)
    STOP_AFTER_PHASE2=1 HONEST_BOUNDARY_DETECT="$FLAGVAL" "$HLW/run_one_model_enhanced.sh" "$ARCH" "$ACT" \
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

for pair in "tinier relu" "tinier leakyrelu" "tiniest leakyrelu" "tiny relu" "tiny leakyrelu"; do
    set -- $pair; ARCH="$1"; ACT="$2"
    run_arm "$ARCH" "$ACT" ON 0
    run_arm "$ARCH" "$ACT" OFF 1
done
echo "ALL DONE"
