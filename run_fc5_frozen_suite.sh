#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fc5 CRYPTO-FROZEN suite (2026-07-05)
#
# For each make_blobs victim {tiniest,tinier,tiny} x {relu,leakyrelu}:
#   [1] canonical Stage-2 LR baseline  (run_one_model_enhanced.sh) -> reconstructed_<arch>.pth (LR)
#   [2] EQS eval of the LR arm         (force a fresh distillation baseline for this victim)
#   [3] Stage-2 crypto + FROZEN fc5 @ budget 100  (--fc5-method cryptanalytic --fc5-budget-mult 100)
#       fc5 freeze is auto-on iff the tie system reaches FULL RANK (--refine-freeze-fc5 auto, default)
#   [4] EQS eval of the crypto-frozen arm (reuse the distillation baseline from [2])
#   [5] budget diminishing-returns sweep: Stage-1 crypto @ mult {50,100,200}
#
# All arms of a victim reuse the SAME on-disk Phase-1/2 artifacts regenerated in [1].
# EQS eval writes eval_<arch>.{md,json}; we copy it off with a tagged name after every run
# (relu/leaky of the same arch share the arch basename and would otherwise clobber).
# ---------------------------------------------------------------------------
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/biprarshi/miniconda3/envs/MLenv/bin/python3}"
[ -x "$PY" ] || PY=python3
DATE="$(date -u +%Y-%m-%d)"
ENH="$(cd "$HERE/.." && pwd)"
OUT="$ENH/Cryptanalytic_output_context/frozen_suite_${DATE}"
RDIR="$HERE/results/reconstructed_models"
EVALDIR="$ENH/Evaluation_Metric_Improve"
LOG="$OUT/suite_log.txt"
mkdir -p "$OUT"

declare -A ARCH_FLAG=( [tiniest]="--tiniest" [tinier]="--tinier" [tiny]="--makeblobs" )
declare -A SIGN_RESTARTS=( [tiniest]=1 [tinier]=1 [tiny]=2 )
declare -A REFINE_EPOCHS=( [tiniest]=300 [tinier]=500 [tiny]=500 )

log(){ echo "$@" | tee -a "$LOG"; }

stage2_flags () {  # $1=arch  (canonical Stage-2, mirrors run_one_model_enhanced.sh STEP 7)
    local A="$1"
    echo "--from-scratch --refine --refine-epochs ${REFINE_EPOCHS[$A]} \
--refine-weight-decay 1e-4 --refine-cosine-lr --early-stop --patience 5 --eval-every 10 \
--eval-on-test3 --train-union-test12 --sign-restarts ${SIGN_RESTARTS[$A]} \
--sign-pair-lookahead 8 --sign-refine-cycles 3 --sign-search-method sa --sign-search-objective margin"
}

save_metrics () { cp "$RDIR/extraction_metrics.json" "$OUT/$1_metrics.json" 2>/dev/null \
        && log "    saved $1_metrics.json" || log "    !! missing metrics for $1"; }

run_eqs () {  # $1=arch_flag  $2=tag  $3="force" | ""
    local FORCE=""
    [ "$3" = "force" ] && FORCE="--force-distill"
    if timeout 1800 "$PY" -u analysis/evaluate_extraction_quality.py $1 $FORCE \
            > "$OUT/$2_eval.log" 2>&1; then
        local ARCH_KEY
        ARCH_KEY=$(echo "$1" | sed 's/--//')          # tiniest|tinier|makeblobs
        cp "$EVALDIR/eval_${ARCH_KEY}.json" "$OUT/$2_eval.json" 2>/dev/null \
            && log "    saved $2_eval.json" || log "    !! missing eval json for $2 (arch=$ARCH_KEY)"
    else
        log "    !! EQS eval FAILED for $2 (see $2_eval.log)"
    fi
}

PAIRS=("$@")
if [ ${#PAIRS[@]} -eq 0 ]; then
    PAIRS=("tiniest relu" "tiniest leakyrelu" "tinier relu" "tinier leakyrelu" "tiny relu" "tiny leakyrelu")
fi

log "############ fc5 crypto-FROZEN suite ($(date -u)) -> $OUT ############"

for pair in "${PAIRS[@]}"; do
    set -- $pair; ARCH="$1"; ACT="$2"; TAG="${ARCH}_${ACT}"
    FLAG="${ARCH_FLAG[$ARCH]}"
    log ""
    log "================  VICTIM $TAG  ($(date -u))  ================"

    # [1] canonical Stage-2 LR baseline (regenerates Phase-1 artifacts + sets alpha/arch)
    log "  [1/5] s2_lr baseline (run_one_model_enhanced.sh)"
    if SIGN_METHOD=sa SIGN_OBJ=margin PYTHON_BIN="$PY" "$HERE/run_one_model_enhanced.sh" "$ARCH" "$ACT" \
        > "$OUT/${TAG}_s2_lr.log" 2>&1; then log "    s2_lr OK"; else log "    s2_lr NONZERO (continuing)"; fi
    save_metrics "${TAG}_s2_lr"
    cp "$RDIR/reconstructed_${ARCH/tiny/makeblobs}.pth" "$OUT/${TAG}_s2_lr.pth" 2>/dev/null || true

    # [2] EQS eval of LR arm — force a fresh distillation baseline for THIS victim's alpha
    log "  [2/5] EQS eval (LR arm, force fresh distillation)"
    run_eqs "$FLAG" "${TAG}_s2_lr" "force"

    # [3] Stage-2 crypto + frozen fc5 @ budget 100 (freeze auto-on iff full rank)
    log "  [3/5] s2_crypto FROZEN (--fc5-method cryptanalytic --fc5-budget-mult 100 --refine-freeze-fc5 auto)"
    "$PY" -u analysis/run_extraction.py $FLAG $(stage2_flags "$ARCH") \
        --fc5-method cryptanalytic --fc5-budget-mult 100 --refine-freeze-fc5 auto \
        > "$OUT/${TAG}_s2_crypto_frozen.log" 2>&1 \
        && log "    s2_crypto_frozen OK" || log "    s2_crypto_frozen NONZERO"
    save_metrics "${TAG}_s2_crypto_frozen"
    cp "$RDIR/reconstructed_${ARCH/tiny/makeblobs}.pth" "$OUT/${TAG}_s2_crypto_frozen.pth" 2>/dev/null || true

    # [4] EQS eval of crypto-frozen arm — reuse distillation baseline from [2]
    log "  [4/5] EQS eval (crypto-frozen arm, reuse distillation)"
    run_eqs "$FLAG" "${TAG}_s2_crypto_frozen" ""

    # [5] budget diminishing-returns sweep (Stage-1 crypto, fast; reuses artifacts)
    for M in 50 100 200; do
        log "  [5/5] budget sweep: s1_crypto @ ${M}x"
        "$PY" -u analysis/run_extraction.py $FLAG --from-scratch --skip-sign-search \
            --fc5-method cryptanalytic --fc5-budget-mult $M \
            > "$OUT/${TAG}_s1_crypto_b${M}.log" 2>&1 \
            && log "    s1_crypto b${M} OK" || log "    s1_crypto b${M} NONZERO"
        save_metrics "${TAG}_s1_crypto_b${M}"
    done

    log "================  $TAG DONE  ($(date -u))  ================"
done
log ""
log "############ SUITE DONE ($(date -u)) ############"
