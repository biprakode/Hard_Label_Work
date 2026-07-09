#!/usr/bin/env bash
# Prefix-init DEGRADATION sweep (non-confirmatory rerun of cheat #6).
#
# ON arm  = canonical full-cheat pipeline (true prefix at every layer, scaling
#           cheat on, sign-walk cheat on) -- same as every other report's ON arm.
# OFF arm = compounding chain: honest scaling (NO_SIG_CHEAT=1) AND each layer's
#           prefix built from what THIS chain already reconstructed for the
#           earlier layers (not the true model, not a static independently-
#           cheating baseline like the confirmatory experiment used). Layer 0
#           has no prefix to compound -- honest, prefix-trivial recovery, same
#           as the OFF arm of signature_scaling_rerun. Layers 1-3 are rebuilt
#           via recover_weights_compounding_chain.py. Signs are Phase-2's
#           actual statistical output (results/sign_recovery/), reused as-is
#           against each newly-recovered direction by neuron id -- NOT
#           recomputed (sign recovery cannot recover layer 0/last-layer signs
#           reliably; propagating that as-is into the chain is the point).
#
# Usage: run_prefix_init_degradation_sweep.sh
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
HLW="$(pwd)"
PY="/home/biprarshi/miniconda3/envs/MLenv/bin/python3"
OUT="$HLW/cheating_ablation/reports/prefix_init_degradation"
mkdir -p "$OUT/raw" "$OUT/logs"

# Same nThreads precedent as run_one_cheat_sweep.sh.
BSR="$HLW/sign_recovery/batched_sign_recovery.py"
BSR_BAK="$(mktemp)"
cp "$BSR" "$BSR_BAK"
restore_bsr() { cp "$BSR_BAK" "$BSR"; rm -f "$BSR_BAK"; echo "[cleanup] restored nThreads in batched_sign_recovery.py"; }
trap restore_bsr EXIT INT TERM
sed -i -E "s/^nThreads([[:space:]]*)=([[:space:]]*)[0-9]+/nThreads\1=\2 6/" "$BSR"
echo "[setup] batched_sign_recovery nThreads -> 6 (original restored on exit)"

declare -A HKEY=( [tiniest]=tiniest [tinier]=tinier [tiny]=makeblobs )

archive_stage() {
    local TAG="$1" ARM="$2"
    "$PY" "$HLW/ablation_tiny/ablation_harness.py" --arch "${HKEY[$ARCH]}" --act "$ACT" \
        --out "$OUT/raw/${TAG}_${ARM}_stage0.json" \
        > "$OUT/logs/${TAG}_${ARM}_harness.log" 2>&1
    mkdir -p "$OUT/raw/${TAG}_${ARM}_artifacts"
    cp -r "$HLW/signature_recovery/outputs/model_weights/Vrelu" "$OUT/raw/${TAG}_${ARM}_artifacts/" 2>/dev/null
    cp -r "$HLW/results/sign_recovery" "$OUT/raw/${TAG}_${ARM}_artifacts/" 2>/dev/null
}

run_on_arm() {
    local ARCH="$1" ACT="$2"
    local TAG="${ARCH}_${ACT}"
    echo "=== $TAG ARM=ON (canonical full cheat) ==="
    local t0=$(date +%s)
    env STOP_AFTER_PHASE2=1 STEP6_TIMEOUT=300 "$HLW/run_one_model_enhanced.sh" "$ARCH" "$ACT" \
        > "$OUT/logs/${TAG}_ON.log" 2>&1
    echo "  driver rc=$?  $(( $(date +%s) - t0 ))s"
    pkill -9 -f "sign_recovery/batched_sign_recovery.py" 2>/dev/null || true
    archive_stage "$TAG" "ON"
    grep -E "layer [0-9]+: recovered" "$OUT/logs/${TAG}_ON.log" | tee "$OUT/raw/${TAG}_ON_recovery_counts.txt"
    echo "  archived ON."
}

run_off_arm() {
    local ARCH="$1" ACT="$2" ARCH_KEY="$3"
    local TAG="${ARCH}_${ACT}"
    echo "=== $TAG ARM=OFF (compounding chain, NO_SIG_CHEAT=1) ==="

    echo "  -- setup pass (regenerate clusters + honest L0 + canonical Phase-2 signs) --"
    local t0=$(date +%s)
    env NO_SIG_CHEAT=1 STOP_AFTER_PHASE2=1 STEP6_TIMEOUT=300 "$HLW/run_one_model_enhanced.sh" "$ARCH" "$ACT" \
        > "$OUT/logs/${TAG}_OFF_setup.log" 2>&1
    echo "  setup rc=$?  $(( $(date +%s) - t0 ))s"
    pkill -9 -f "sign_recovery/batched_sign_recovery.py" 2>/dev/null || true
    grep -E "layer [0-9]+: recovered" "$OUT/logs/${TAG}_OFF_setup.log" | tee "$OUT/raw/${TAG}_OFF_setup_recovery_counts.txt"

    echo "  -- compounding chain (layers 1,2,3 rebuilt against reconstructed prefix) --"
    t0=$(date +%s)
    cd "$HLW/signature_recovery"
    env NO_SIG_CHEAT=1 "$PY" recover_weights_compounding_chain.py "$ARCH_KEY" \
        > "$OUT/logs/${TAG}_OFF_chain.log" 2>&1
    local rc=$?
    cd "$HLW"
    echo "  chain rc=$rc  $(( $(date +%s) - t0 ))s"
    tail -15 "$OUT/logs/${TAG}_OFF_chain.log"

    archive_stage "$TAG" "OFF"
    echo "  archived OFF."
}

# Resumable: skip an arm if already archived by a prior partial run (this
# sweep hit the documented tiniest_leakyrelu sign-recovery stall precedent --
# see cheat_disable_map.md -- killed and restarted here with STEP6_TIMEOUT).
for pair in "tiniest relu tiniest" "tiniest leakyrelu tiniest" "tinier relu tinier" "tinier leakyrelu tinier" "tiny relu makeblobs" "tiny leakyrelu makeblobs"; do
    set -- $pair; ARCH="$1"; ACT="$2"; ARCH_KEY="$3"
    TAG="${ARCH}_${ACT}"
    if [ -f "$OUT/raw/${TAG}_ON_stage0.json" ]; then
        echo "=== $TAG ARM=ON: SKIPPED (already archived) ==="
    else
        run_on_arm "$ARCH" "$ACT"
    fi
    if [ -f "$OUT/raw/${TAG}_OFF_stage0.json" ]; then
        echo "=== $TAG ARM=OFF: SKIPPED (already archived) ==="
    else
        run_off_arm "$ARCH" "$ACT" "$ARCH_KEY"
    fi
done
echo "ALL DONE (prefix_init_degradation)"
