#!/usr/bin/env bash
# ============================================================================
# fc5 cryptanalytic vs LR — staged A/B over the 6 make_blobs victims.
#
# Per victim, Phase-1/2 artifacts are generated ONCE (via run_one_model_enhanced.sh,
# which also produces the canonical Stage-2 LR arm), then reused for 3 more
# Phase-3 variants on the SAME hidden-layer recovery:
#
#   s2_lr     Stage-2 LR      = run_one_model_enhanced.sh (sign search SA+margin + refine) [canonical]
#   s1_lr     Stage-1 LR      = sig + bias + LR fc5             (no sign search, no refine)
#   s1_crypto Stage-1 crypto  = sig + bias + crypto fc5         (no sign search, no refine)
#   s2_crypto Stage-2 crypto  = crypto fc5 + SA sign search + refine
#
# crypto fc5 uses the drop-in true-prefix cheat for its h4 solve, then re-instates
# the imperfect recovered hidden layers (only fc5 kept).
#
# Metrics (extraction_metrics.json) for each variant are copied to:
#   ../Cryptanalytic_output_context/suite_<DATE>/<arch>_<act>_<variant>_metrics.json
#
# Usage: ./run_fc5_crypto_suite.sh [pairs...]     (default: all 6)
#        ./run_fc5_crypto_suite.sh "tiniest relu"     # single, for validation
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/biprarshi/miniconda3/envs/MLenv/bin/python3}"
DATE="$(date -u +%Y-%m-%d)"
OUT="$(cd "$HERE/.." && pwd)/Cryptanalytic_output_context/suite_${DATE}"
RDIR="$HERE/results/reconstructed_models"
LOG="$OUT/suite_log.txt"
mkdir -p "$OUT"

declare -A ARCH_FLAG=( [tiniest]="--tiniest" [tinier]="--tinier" [tiny]="--makeblobs" )
declare -A SIGN_RESTARTS=( [tiniest]=1 [tinier]=1 [tiny]=2 )
declare -A REFINE_EPOCHS=( [tiniest]=300 [tinier]=500 [tiny]=500 )

log(){ echo "$@" | tee -a "$LOG"; }

# canonical Stage-2 flag set (mirrors run_one_model_enhanced.sh STEP 7)
stage2_flags () {  # $1=arch
    local A="$1"
    echo "--from-scratch --refine --refine-epochs ${REFINE_EPOCHS[$A]} \
--refine-weight-decay 1e-4 --refine-cosine-lr --early-stop --patience 5 --eval-every 10 \
--eval-on-test3 --train-union-test12 --sign-restarts ${SIGN_RESTARTS[$A]} \
--sign-pair-lookahead 8 --sign-refine-cycles 3 --sign-search-method sa --sign-search-objective margin"
}

save_metrics () {  # $1=tag
    cp "$RDIR/extraction_metrics.json" "$OUT/$1_metrics.json" 2>/dev/null \
        && log "    saved $1_metrics.json" || log "    !! missing metrics for $1"
}

PAIRS=("$@")
if [ ${#PAIRS[@]} -eq 0 ]; then
    PAIRS=("tiniest relu" "tiniest leakyrelu" "tinier relu" "tinier leakyrelu" "tiny relu" "tiny leakyrelu")
fi

log "############ fc5 crypto suite ($(date -u)) -> $OUT ############"

for pair in "${PAIRS[@]}"; do
    set -- $pair; ARCH="$1"; ACT="$2"; TAG="${ARCH}_${ACT}"
    FLAG="${ARCH_FLAG[$ARCH]}"
    log ""
    log "================  VICTIM $TAG  ($(date -u))  ================"

    # ---- Phase-1/2 + canonical Stage-2 LR (regenerates artifacts) ----
    log "  [1/4] s2_lr  (run_one_model_enhanced.sh: Phase-1 + SA sign search + refine)"
    if SIGN_METHOD=sa SIGN_OBJ=margin PYTHON_BIN="$PY" "$HERE/run_one_model_enhanced.sh" "$ARCH" "$ACT" \
        > "$OUT/${TAG}_s2_lr.log" 2>&1; then
        log "    s2_lr OK"
    else
        log "    s2_lr NONZERO (see ${TAG}_s2_lr.log) — continuing"
    fi
    save_metrics "${TAG}_s2_lr"

    # ---- Stage-1 LR (reuse artifacts): sig + bias + LR fc5 ----
    log "  [2/4] s1_lr  (--skip-sign-search --fc5-method lr)"
    "$PY" -u analysis/run_extraction.py $FLAG --from-scratch --skip-sign-search \
        --fc5-method lr > "$OUT/${TAG}_s1_lr.log" 2>&1 \
        && log "    s1_lr OK" || log "    s1_lr NONZERO"
    save_metrics "${TAG}_s1_lr"

    # ---- Stage-1 crypto (reuse artifacts): sig + bias + crypto fc5 (drop-in cheat) ----
    log "  [3/4] s1_crypto  (--skip-sign-search --fc5-method cryptanalytic)"
    "$PY" -u analysis/run_extraction.py $FLAG --from-scratch --skip-sign-search \
        --fc5-method cryptanalytic > "$OUT/${TAG}_s1_crypto.log" 2>&1 \
        && log "    s1_crypto OK" || log "    s1_crypto NONZERO"
    save_metrics "${TAG}_s1_crypto"

    # ---- Stage-2 crypto (reuse artifacts): crypto fc5 + SA sign search + refine ----
    log "  [4/4] s2_crypto  (canonical stage-2 flags + --fc5-method cryptanalytic)"
    "$PY" -u analysis/run_extraction.py $FLAG $(stage2_flags "$ARCH") \
        --fc5-method cryptanalytic > "$OUT/${TAG}_s2_crypto.log" 2>&1 \
        && log "    s2_crypto OK" || log "    s2_crypto NONZERO"
    save_metrics "${TAG}_s2_crypto"

    log "================  $TAG DONE  ================"
done

log ""
log "############  SUITE COMPLETE ($(date -u))  ############"
ls -1 "$OUT" | tee -a "$LOG"
