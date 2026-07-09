#!/usr/bin/env bash
# HONEST_SIGN_WALK sweep: 6 victims x 2 arms. Resource caps (not method
# changes -- see cheat_disable_map.md): SIGN_NEXP_CAP=500 (per-neuron
# experiment-count cap, canonical is 2000/1000), STEP6_TIMEOUT=300 (outer
# wall-clock backstop -- an individual neuron's inner walk cost is not
# strictly bounded by nExp under the honest weights, discovered during this
# study's smoke testing). Tested against still-cheating (canonical) Phase 1/2
# per the confirmed decision -- HONEST_SIGN_WALK is the only flag toggled.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
HLW="$(pwd)"
PY="/home/biprarshi/miniconda3/envs/MLenv/bin/python3"
OUT="$HLW/cheating_ablation/reports/sign_walk"
mkdir -p "$OUT/raw" "$OUT/logs"

declare -A HKEY=( [tiniest]=tiniest [tinier]=tinier [tiny]=makeblobs )

run_arm() {
    local ARCH="$1" ACT="$2" ARM="$3" FLAGVAL="$4"
    local TAG="${ARCH}_${ACT}"
    echo "=== $TAG ARM=$ARM (HONEST_SIGN_WALK=$FLAGVAL) ==="
    local t0=$(date +%s)
    STOP_AFTER_PHASE2=1 HONEST_SIGN_WALK="$FLAGVAL" SIGN_NEXP_CAP=500 STEP6_TIMEOUT=300 \
        "$HLW/run_one_model_enhanced.sh" "$ARCH" "$ACT" \
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
    grep -E "layer [0-9]+: recovered|Total neurons processed|WARNING: sign recovery" "$OUT/logs/${TAG}_${ARM}.log" | tee "$OUT/raw/${TAG}_${ARM}_summary.txt"
    # Belt-and-suspenders: catch any straggler workers before the next victim's STEP 0 clean.
    pkill -9 -f "sign_recovery/batched_sign_recovery.py" 2>/dev/null || true
    echo "  archived."
}

for pair in "tiniest relu" "tiniest leakyrelu" "tinier relu" "tinier leakyrelu" "tiny relu" "tiny leakyrelu"; do
    set -- $pair; ARCH="$1"; ACT="$2"
    run_arm "$ARCH" "$ACT" ON 0
    run_arm "$ARCH" "$ACT" OFF 1
done
echo "ALL DONE (sign_walk)"
