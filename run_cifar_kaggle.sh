#!/usr/bin/env bash
# ============================================================================
# CIFAR-10 flagship (full, 3072-256-256-256-64-10) end-to-end extraction,
# packaged to run on a Kaggle GPU notebook (P100 recommended). Extracts the two
# victims tiny_stuff/TinyModel_{relu,leakyrelu}.pth (= the CIFAR full arch).
#
# LATEST WORKFLOW: GPU parallel dual search (float64) + combinatorial sign search
# (PT+margin) + improved eval scorecard.
#
# KAGGLE NOTES
#   * The repo hardcodes the absolute prefix
#       /run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction
#     in ~20 files. Rather than edit them all, this script SYMLINKS that prefix to
#     wherever the repo actually lives (e.g. /kaggle/working/<repo>), so every
#     hardcoded path resolves unchanged. Kaggle runs as root, so the symlink works.
#   * GPU: the dual finder (find_duals_torch.py) now runs on CUDA in float64 when
#     a GPU is present; on a CPU box it falls back to CPU (identical results). On
#     GPU we use ONE worker with a large batch (one CUDA context); on CPU we use
#     the multi-worker profile.
#   * DISK CAP: each full triplet is 3x3072 float64 ~= 72 KB; 80 rounds x 10000
#     ~= 57 GB > the 54 GB cap. So dual search runs in CHUNKS: after each chunk we
#     cluster (CLUSTER_MERGE=1, accumulating) and DELETE the raw pickles, keeping
#     peak raw-pickle disk ~= CHUNK*10000*72KB. A du-guard aborts above 54 GB.
#
# USAGE
#   ./run_cifar_kaggle.sh                 # both victims, full run (GPU)
#   ./run_cifar_kaggle.sh relu            # one victim
#   ./run_cifar_kaggle.sh --smoke         # local CPU smoke: relu, 1 tiny round
#   ./run_cifar_kaggle.sh --smoke relu
#
# OVERRIDES (env)
#   PYTHON_BIN, SIGN_METHOD (default pt), SIGN_OBJ (default margin),
#   DUAL_ROUNDS (default 80), DUAL_CHUNK (default 10), DUAL_BATCH, DUAL_WORKERS,
#   CLUSTER_PER_NEURON_CAP (default 150), DISK_CAP_GB (default 54)
# ============================================================================
set -uo pipefail

# ---------- arg parse ----------
SMOKE=0
WHICH="both"
for a in "$@"; do
    case "$a" in
        --smoke) SMOKE=1 ;;
        relu) WHICH="relu" ;;
        leakyrelu) WHICH="leakyrelu" ;;
        both) WHICH="both" ;;
        *) echo "unknown arg: $a"; exit 1 ;;
    esac
done

# ---------- locate repo + symlink canonical Hard_Label_Work ----------
# Symlink the *canonical Hard_Label_Work path* directly to the real script dir,
# so the ~20 hardcoded absolute paths resolve regardless of where the repo was
# cloned (e.g. /kaggle/working/Hard_Label_Work with no enhanced_codebase wrapper).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # real .../Hard_Label_Work, any clone location
CANON="/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction"
CANON_HLW="$CANON/enhanced_codebase/Hard_Label_Work"
if [ "$HERE" != "$CANON_HLW" ]; then
    echo "[bootstrap] symlinking $CANON_HLW -> $HERE"
    [ -L "$CANON" ] && rm -f "$CANON"                        # drop stale prefix symlink from earlier broken runs
    mkdir -p "$CANON/enhanced_codebase" || { echo "[bootstrap] FATAL: cannot mkdir $CANON/enhanced_codebase"; exit 1; }
    rm -rf "$CANON_HLW"                                       # clear whatever squats the canonical slot (dir or symlink)
    ln -sfn "$HERE" "$CANON_HLW" || { echo "[bootstrap] FATAL: symlink failed"; exit 1; }
fi
cd "$HERE" || { echo "[bootstrap] FATAL: cd $HERE failed"; exit 1; }
echo "[bootstrap] OK: HERE=$HERE  ->  $CANON_HLW"

PY="${PYTHON_BIN:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY="/home/biprarshi/miniconda3/envs/MLenv/bin/python3"

SIGN_METHOD="${SIGN_METHOD:-pt}"          # PT+margin: §2.3.11 recommends PT for the widest CIFAR-full layers
SIGN_OBJ="${SIGN_OBJ:-margin}"
DISK_CAP_GB="${DISK_CAP_GB:-54}"
export CLUSTER_PER_NEURON_CAP="${CLUSTER_PER_NEURON_CAP:-150}"

# ---------- GPU detect ----------
HAS_GPU=$("$PY" -c "
import torch
ok=0
if torch.cuda.is_available():
    try:
        x=torch.randn(8,8,dtype=torch.float64,device='cuda'); _=(x@x).sum().item(); ok=1   # real fp64 kernel launch
    except Exception: ok=0
print(ok)
" 2>/dev/null || echo 0)
if [ "$HAS_GPU" = "1" ]; then
    GPU_NAME=$("$PY" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    : "${DUAL_WORKERS:=1}"   ; : "${DUAL_BATCH:=256}"     # one CUDA context, large batch
    echo "[gpu] CUDA available: $GPU_NAME  (workers=$DUAL_WORKERS batch=$DUAL_BATCH)"
else
    : "${DUAL_WORKERS:=5}"   ; : "${DUAL_BATCH:=48}"      # CPU profile (22Gi box)
    echo "[gpu] no CUDA — CPU fallback (workers=$DUAL_WORKERS batch=$DUAL_BATCH)"
fi
export DUAL_WORKERS DUAL_BATCH

# ---------- smoke vs full sizing ----------
if [ "$SMOKE" = "1" ]; then
    DUAL_ROUNDS=1; DUAL_CHUNK=1; SMOKE_TARGET=8; WHICH="relu"
    REFINE_EPOCHS=20; SIGN_RESTARTS=0; SIGN_PAIR=2; SIGN_CYCLES=1
    echo "[smoke] full_relu, 1 round, target=$SMOKE_TARGET, minimal Phase-3 — plumbing test only"
else
    DUAL_ROUNDS="${DUAL_ROUNDS:-80}"; DUAL_CHUNK="${DUAL_CHUNK:-10}"; SMOKE_TARGET=""
    REFINE_EPOCHS=500; SIGN_RESTARTS=4; SIGN_PAIR=8; SIGN_CYCLES=3
fi

DATE="$(date +%Y-%m-%d)"
NOTES_DIR="$(cd "$HERE/.." && pwd)/paper_notes/section3"
OUTDIR="$NOTES_DIR/reports/cifar_kaggle_${DATE}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/cifar_kaggle_run.log"
log(){ echo "$@" | tee -a "$LOG"; }

# residual dirs that count toward the disk cap
RESID=( "$HERE/signature_recovery/exp" "$HERE/signature_recovery/outputs"
        "$HERE/sign_recovery/layer_neuron_npys" "$HERE/results/sign_recovery"
        "$HERE/results/reconstructed_models" )

du_guard(){
    local used_kb cap_kb
    used_kb=$(du -sk "${RESID[@]}" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    cap_kb=$(( DISK_CAP_GB * 1024 * 1024 ))
    local used_gb=$(( used_kb / 1024 / 1024 ))
    if [ "$used_kb" -gt "$cap_kb" ]; then
        log "!!! DISK GUARD TRIPPED: residuals ${used_gb} GB > cap ${DISK_CAP_GB} GB — aborting"
        exit 3
    fi
    echo "${used_gb}"
}

clean_residuals(){
    rm -rf "$HERE/signature_recovery/exp/1" "$HERE/signature_recovery/exp/1-cluster-"*.p
    rm -rf "$HERE/signature_recovery/outputs/model_weights/Vrelu/layer_"*
    rm -rf "$HERE/sign_recovery/layer_neuron_npys"
    rm -f  "$HERE/results/sign_recovery/"* "$HERE/results/reconstructed_models/reconstructed_"*
    mkdir -p "$HERE/signature_recovery/exp/1" \
             "$HERE/signature_recovery/outputs/model_weights/Vrelu" \
             "$HERE/sign_recovery/layer_neuron_npys" \
             "$HERE/results/sign_recovery" "$HERE/results/reconstructed_models"
}

configure_arch(){    # $1 = activation
    local ACT="$1"
    local ALPHA; ALPHA=$([ "$ACT" = "leakyrelu" ] && echo "0.01" || echo "0.0")
    HERE="$HERE" ALPHA="$ALPHA" "$PY" - <<'PY'
import os, re, pathlib
HERE = pathlib.Path(os.environ['HERE']); ALPHA = os.environ['ALPHA']
targets = [
    (HERE/'signature_recovery/utils.py',            r'^LEAKY_ALPHA\s*=\s*\S+'),
    (HERE/'sign_recovery/sign_recovery.py',         r'^LEAKY_ALPHA\s*=\s*\S+'),
    (HERE/'sign_recovery/batched_sign_recovery.py', r'^LEAKY_ALPHA\s*=\s*\S+'),
    (HERE/'analysis/extraction_pipeline/config.py', r'^LEAKY_ALPHA\s*=\s*\S+'),
]
for path, pat in targets:
    txt = path.read_text()
    txt = re.sub(pat, f"LEAKY_ALPHA = {ALPHA}", txt, count=1, flags=re.M)
    path.write_text(txt)
# full arch booleans: all small-arch toggles off
arch_flags = {'TINIEST': False, 'TINIER': False, 'TINY': False, 'MAKEBLOBS': False}
for path in [HERE/'signature_recovery/utils.py', HERE/'sign_recovery/batched_sign_recovery.py']:
    txt = path.read_text()
    for k, v in arch_flags.items():
        txt = re.sub(rf'^{k}\s*=\s*(True|False)\b', f'{k} = {v}', txt, count=1, flags=re.M)
    path.write_text(txt)
print(f"  configured full/{'leakyrelu' if float(ALPHA)>0 else 'relu'} (LEAKY_ALPHA={ALPHA})")
PY
    "$PY" -c "
import sys; sys.path.insert(0,'$HERE/signature_recovery')
import utils
print('  LAYER_SIZES =', utils.LAYER_SIZES, ' MODEL_PATH =', utils.MODEL_PATH)
import os; assert os.path.exists(utils.MODEL_PATH), 'victim model missing'
"
}

# ---------- chunked dual search (cluster-merge + delete per chunk) ----------
chunked_dual_search(){   # $1 = TAG
    local TAG="$1"; local done=0 chunk_no=0
    cd "$HERE/signature_recovery"
    while [ "$done" -lt "$DUAL_ROUNDS" ]; do
        local n=$(( DUAL_ROUNDS - done )); [ "$n" -gt "$DUAL_CHUNK" ] && n=$DUAL_CHUNK
        chunk_no=$(( chunk_no + 1 ))
        local targ_arg=""; [ -n "$SMOKE_TARGET" ] && targ_arg="--target $SMOKE_TARGET"
        log "  [duals] chunk $chunk_no: $n rounds (workers=$DUAL_WORKERS batch=$DUAL_BATCH) ..."
        "$PY" torch_impl/parallel_duals.py --iterations "$n" --workers "$DUAL_WORKERS" \
              --batch-size "$DUAL_BATCH" --impl torch $targ_arg \
              >> "/tmp/${TAG}_duals.log" 2>&1 || log "  [duals] chunk $chunk_no nonzero rc"
        # cluster-merge this chunk, then delete raw pickles
        CLUSTER_MERGE=1 "$PY" cluster_dual_points_stream.py >> "/tmp/${TAG}_cluster.log" 2>&1 \
            || log "  [cluster] chunk $chunk_no nonzero rc"
        rm -f "$HERE/signature_recovery/exp/1/"*.p
        done=$(( done + n ))
        local used; used=$(du_guard)
        log "  [duals] progress ${done}/${DUAL_ROUNDS} rounds; residuals ${used} GB"
    done
    cd "$HERE"
}

extract_one(){       # $1 = activation
    local ACT="$1"; local TAG="full_${ACT}"
    log ""; log "================  CIFAR VICTIM: $TAG  ($(date -u))  ================"
    : > "/tmp/${TAG}_duals.log"; : > "/tmp/${TAG}_cluster.log"

    log "=== [0] clean ===";        clean_residuals
    log "=== [1] configure ===";    configure_arch "$ACT"
    log "=== [2] chunked GPU dual search (${DUAL_ROUNDS} rounds / chunk ${DUAL_CHUNK}) ==="
    chunked_dual_search "$TAG"

    log "=== [3] per-neuron dual files ==="
    (cd "$HERE/signature_recovery" && "$PY" generate_dual_neuron.py 2>&1 | tail -3 | tee -a "$LOG")

    log "=== [4] recover weights (layers 0..3) ==="
    for L in 0 1 2 3; do
        (cd "$HERE/signature_recovery" && "$PY" recover_weights.py "$L" > "/tmp/${TAG}_recover_${L}.log" 2>&1) || true
        local rec; rec=$(grep -c "Successfully extracted neuron" "/tmp/${TAG}_recover_${L}.log" 2>/dev/null || echo 0)
        log "  layer $L: recovered=$rec"
    done

    log "=== [5] batched sign recovery (TF, GPU-aware) ==="
    "$PY" sign_recovery/batched_sign_recovery.py > "/tmp/${TAG}_sign.log" 2>&1 \
        || log "  sign recovery nonzero rc (may have partial output)"

    log "=== [6] Phase 3 reconstruct+refine (full, ${SIGN_METHOD}/${SIGN_OBJ}) ==="
    "$PY" analysis/run_extraction.py --full --from-scratch --refine \
        --refine-epochs "$REFINE_EPOCHS" --refine-weight-decay 1e-4 --refine-cosine-lr \
        --early-stop --patience 5 --eval-every 10 --eval-on-test3 --train-union-test12 \
        --sign-restarts "$SIGN_RESTARTS" --sign-pair-lookahead "$SIGN_PAIR" \
        --sign-refine-cycles "$SIGN_CYCLES" \
        --sign-search-method "$SIGN_METHOD" --sign-search-objective "$SIGN_OBJ" \
        > "/tmp/${TAG}_phase3.log" 2>&1 || log "  Phase 3 nonzero rc (see /tmp/${TAG}_phase3.log)"
    grep -E "recovered|accuracy|agreement|EXTRACTION|Saved|X_test3" "/tmp/${TAG}_phase3.log" | tail -20 | tee -a "$LOG"

    log "=== [7] report + eval scorecard ==="
    cp "$HERE/results/reconstructed_models/extraction_metrics.json" \
       "$OUTDIR/${TAG}_extraction_metrics.json" 2>/dev/null || true
    "$PY" analysis/compare_true_vs_extracted_v2.py --arch full --activation "$ACT" \
        --output "$OUTDIR/${TAG}_true_vs_extracted.md" \
        --metrics-json "$OUTDIR/${TAG}_extraction_metrics.json" 2>&1 | tee -a "$LOG" || true
    "$PY" analysis/evaluate_extraction_quality.py --full > "/tmp/${TAG}_eval.log" 2>&1 \
        && cp "$HERE/results/reports/eval_full_${DATE}.md" "$OUTDIR/${TAG}_eval_scorecard.md" 2>/dev/null \
        || log "  eval scorecard nonzero rc (extraction still OK)"
    # keep the reconstructed model + logs for download
    cp "$HERE/results/reconstructed_models/reconstructed_full.pth" "$OUTDIR/${TAG}_reconstructed_full.pth" 2>/dev/null || true
    cp "/tmp/${TAG}_phase3.log" "$OUTDIR/${TAG}_phase3.log" 2>/dev/null || true
    log "================  $TAG DONE  ================"
}

log "############################################################"
log "# CIFAR Kaggle extraction  ($(date -u))  smoke=$SMOKE which=$WHICH"
log "# sign=${SIGN_METHOD}/${SIGN_OBJ}  rounds=${DUAL_ROUNDS} chunk=${DUAL_CHUNK}  cap=${DISK_CAP_GB}GB"
log "# output -> $OUTDIR"
log "############################################################"

case "$WHICH" in
    relu)      extract_one relu ;;
    leakyrelu) extract_one leakyrelu ;;
    both)      extract_one relu; clean_residuals; extract_one leakyrelu ;;
esac

clean_residuals   # final cleanup so nothing huge lingers in the image
log ""; log "############  CIFAR EXTRACTION COMPLETE ($(date -u))  ############"
log "Download this folder: $OUTDIR"
ls -1 "$OUTDIR" | tee -a "$LOG"
