#!/usr/bin/env bash
# HONEST_CLUSTER sweep: 6 victims x 2 arms. Resource caps (CLUSTER_SLOW_MAX_SEEDS,
# CLUSTER_SLOW_MAX_INNER, VERBOSE_IS_CONSISTENT=0) applied to keep cluster_slow's
# O(n^2) cost tractable -- see cheat_disable_map.md for rationale. Caps apply to
# BOTH arms identically where relevant (ON/cheat_cluster ignores them; only
# OFF/cluster_slow reads them), so they don't bias the comparison.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
HLW="$(pwd)"
PY="/home/biprarshi/miniconda3/envs/MLenv/bin/python3"
OUT="$HLW/cheating_ablation/reports/neuron_clustering"
mkdir -p "$OUT/raw" "$OUT/logs"

declare -A HKEY=( [tiniest]=tiniest [tinier]=tinier [tiny]=makeblobs )

run_arm() {
    local ARCH="$1" ACT="$2" ARM="$3" FLAGVAL="$4"
    local TAG="${ARCH}_${ACT}"
    echo "=== $TAG ARM=$ARM (HONEST_CLUSTER=$FLAGVAL) ==="
    local t0=$(date +%s)
    STOP_AFTER_PHASE2=1 HONEST_CLUSTER="$FLAGVAL" \
        CLUSTER_SLOW_MAX_SEEDS=25 CLUSTER_SLOW_MAX_INNER=1500 VERBOSE_IS_CONSISTENT=0 \
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
    grep -E "layer [0-9]+: recovered" "$OUT/logs/${TAG}_${ARM}.log" | tee "$OUT/raw/${TAG}_${ARM}_recovery_counts.txt"
    echo "  archived."
}

for pair in "tiniest relu" "tiniest leakyrelu" "tinier relu" "tinier leakyrelu" "tiny relu" "tiny leakyrelu"; do
    set -- $pair; ARCH="$1"; ACT="$2"
    run_arm "$ARCH" "$ACT" ON 0
    run_arm "$ARCH" "$ACT" OFF 1
done
echo "ALL DONE (neuron_clustering)"
