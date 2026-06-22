#!/usr/bin/env bash
# ============================================================================
# Batch end-to-end extraction of the six make_blobs victims, LATEST WORKFLOW:
#   parallel dual search (torch_impl/parallel_duals.py)
#   + improved combinatorial sign search (metaheuristic: SA+margin and PT+margin)
#   + improved evaluation scorecard (evaluate_extraction_quality.py, Metrics 1-5 + EQS)
#
# Victims:  {tiniest, tinier, tiny} x {relu, leakyrelu}  = 6 models
#
# Per victim, two sign-search arms scored against the SAME Phase-1/2 artifacts
# (honest A/B, mirrors sign_search_improve/validate_eqs_ab.sh):
#   ARM A  SA+margin : full pipeline (dual search -> ... -> Phase 3 -> eval)
#   ARM B  PT+margin : Phase-3-only re-run on ARM A's on-disk duals/signs/sigs
#
# After each victim, residuals are cleaned (run_one_model_enhanced.sh STEP 0
# cleans at the start of the next victim; a final clean runs at the end).
#
# All per-arm reports collected under:
#   paper_notes/section3/reports/<DATE>/
# ============================================================================
set -uo pipefail

DATE=2026-06-21
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/biprarshi/miniconda3/envs/MLenv/bin/python3}"
NOTES_DIR="$(cd "$HERE/../.." && pwd)/paper_notes/section3"
SRC_REPORTS="$NOTES_DIR/reports"
DATED="$SRC_REPORTS/$DATE"
RESULTS_REPORTS="$HERE/results/reports"
RDIR="$HERE/results/reconstructed_models"
BATCH_LOG="$DATED/batch_run_log.txt"
mkdir -p "$DATED"

log(){ echo "$@" | tee -a "$BATCH_LOG"; }

# arch -> (ARCH_FLAG for Phase-3/eval, arch_key for eval scorecard filename,
#          Fix-C tuning identical to run_one_model_enhanced.sh)
declare -A ARCH_FLAG=( [tiniest]="--tiniest" [tinier]="--tinier" [tiny]="--makeblobs" )
declare -A ARCH_KEY=(  [tiniest]="tiniest"   [tinier]="tinier"   [tiny]="makeblobs" )
declare -A SIGN_RESTARTS=( [tiniest]=1 [tinier]=1 [tiny]=2 )
declare -A REFINE_EPOCHS=( [tiniest]=300 [tinier]=500 [tiny]=500 )

log ""
log "############################################################"
log "# make_blobs batch extraction  ($(date -u))"
log "# arms: ARM A = SA+margin (full) ; ARM B = PT+margin (Phase-3 reuse)"
log "# reports -> $DATED"
log "############################################################"

snapshot_armA () {  # $1=TAG  $2=arch_key
    local TAG="$1" KEY="$2"
    cp "$SRC_REPORTS/${TAG}_true_vs_extracted.md"   "$DATED/${TAG}_sa_margin_true_vs_extracted.md"   2>/dev/null || true
    cp "$SRC_REPORTS/${TAG}_true_vs_extracted.json" "$DATED/${TAG}_sa_margin_true_vs_extracted.json" 2>/dev/null || true
    cp "$SRC_REPORTS/${TAG}_extraction_metrics.json" "$DATED/${TAG}_sa_margin_extraction_metrics.json" 2>/dev/null || true
    cp "$RESULTS_REPORTS/eval_${KEY}_${DATE}.md"    "$DATED/${TAG}_sa_margin_eval_scorecard.md"      2>/dev/null || true
}

run_armB_pt () {    # $1=arch  $2=act  $3=TAG
    local ARCH="$1" ACT="$2" TAG="$3"
    local FLAG="${ARCH_FLAG[$ARCH]}" KEY="${ARCH_KEY[$ARCH]}"
    log "=== [$TAG] ARM B  PT+margin  (Phase-3 re-run on ARM A artifacts) ==="
    # Phase 3 only — reuse on-disk signature/sign/dual artifacts, swap optimizer to PT.
    "$PY" analysis/run_extraction.py $FLAG --from-scratch --refine \
        --refine-epochs "${REFINE_EPOCHS[$ARCH]}" \
        --refine-weight-decay 1e-4 --refine-cosine-lr \
        --early-stop --patience 5 --eval-every 10 \
        --eval-on-test3 --train-union-test12 \
        --sign-restarts "${SIGN_RESTARTS[$ARCH]}" \
        --sign-pair-lookahead 8 --sign-refine-cycles 3 \
        --sign-search-method pt --sign-search-objective margin \
        > "/tmp/${TAG}_pt_phase3.log" 2>&1 \
        && log "  ARM B phase3 OK" || log "  ARM B phase3 NONZERO (see /tmp/${TAG}_pt_phase3.log)"
    cp "$RDIR/extraction_metrics.json" "$DATED/${TAG}_pt_margin_extraction_metrics.json" 2>/dev/null || true
    # report
    "$PY" analysis/compare_true_vs_extracted_v2.py \
        --arch "$ARCH" --activation "$ACT" \
        --output "$DATED/${TAG}_pt_margin_true_vs_extracted.md" \
        --metrics-json "$DATED/${TAG}_pt_margin_extraction_metrics.json" \
        --timings "/tmp/${TAG}_timings.json" \
        >> "$BATCH_LOG" 2>&1 || log "  ARM B compare failed"
    # improved evaluation scorecard
    "$PY" analysis/evaluate_extraction_quality.py $FLAG > "/tmp/${TAG}_pt_eval.log" 2>&1 \
        && log "  ARM B eval OK" || log "  ARM B eval failed (extraction still OK)"
    cp "$RESULTS_REPORTS/eval_${KEY}_${DATE}.md" "$DATED/${TAG}_pt_margin_eval_scorecard.md" 2>/dev/null || true
}

cd "$HERE"
for pair in "tiniest relu" "tiniest leakyrelu" "tinier relu" "tinier leakyrelu" "tiny relu" "tiny leakyrelu"; do
    set -- $pair; ARCH="$1"; ACT="$2"; TAG="${ARCH}_${ACT}"
    log ""
    log "================  VICTIM: $TAG  ($(date -u))  ================"

    # ----- ARM A: full pipeline, SA+margin (cleans residuals at its STEP 0) -----
    log "=== [$TAG] ARM A  SA+margin  (full pipeline) ==="
    if SIGN_METHOD=sa SIGN_OBJ=margin "$HERE/run_one_model_enhanced.sh" "$ARCH" "$ACT"; then
        log "  ARM A OK"
    else
        log "  ARM A FAILED (rc=$?) — see paper_notes/section3/run_log.txt ; continuing"
    fi
    snapshot_armA "$TAG" "${ARCH_KEY[$ARCH]}"

    # ----- ARM B: Phase-3-only PT+margin on the same artifacts -----
    run_armB_pt "$ARCH" "$ACT" "$TAG"

    log "================  VICTIM $TAG DONE  ================"
done

# ----- final residual clean -----
log "=== final residual clean ==="
rm -rf "$HERE/signature_recovery/exp/1" 2>/dev/null || true
rm -f  "$HERE/signature_recovery/exp/1-cluster-"*.p 2>/dev/null || true
rm -rf "$HERE/signature_recovery/outputs/model_weights/Vrelu/layer_"* 2>/dev/null || true
rm -rf "$HERE/sign_recovery/layer_neuron_npys" 2>/dev/null || true

log ""
log "############  BATCH COMPLETE ($(date -u))  ############"
log "Reports in: $DATED"
ls -1 "$DATED" | tee -a "$BATCH_LOG"
