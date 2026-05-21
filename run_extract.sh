#!/usr/bin/env bash
# Convenience runner: full hard-label extraction pipeline end to end.
#
# Usage:
#   ./run_extract.sh tiniest [DUAL_ITERS]     # 8-8-8-8-8-8 make_blobs (fast, fits 24GB)
#   ./run_extract.sh tiny    [DUAL_ITERS]     # 64x5->10 make_blobs (slow, memory-heavy)
#   ./run_extract.sh tinier  [DUAL_ITERS]     # 32->16->16->16->8->4 make_blobs
#
# DUAL_ITERS defaults: tiniest=9, tinier=50, tiny=1000.
#
# Produces:
#   results/reconstructed_models/reconstructed_<model>.pth
#   results/reconstructed_models/extraction_metrics.json
#   results/sign_recovery/...
set -euo pipefail

MODEL="${1:-tiniest}"
DUAL_ITERS="${2:-}"
case "$MODEL" in
    tiniest) : "${DUAL_ITERS:=9}"  ;;
    tinier)  : "${DUAL_ITERS:=50}" ;;
    tiny)    : "${DUAL_ITERS:=1000}" ;;
    *) echo "Usage: $0 {tiniest|tinier|tiny} [DUAL_ITERS]"; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python3}"
SR="$HERE/signature_recovery"
SIGN="$HERE/sign_recovery"
ANA="$HERE/analysis"

# ---- 1. Configure utils.py flags to match the requested model ----
# (The three TINIEST/TINIER/TINY booleans decide the architecture.)
python3 - "$MODEL" <<'PY'
import re, sys, pathlib
model = sys.argv[1]
p = pathlib.Path(__file__).resolve().parents[0] / 'signature_recovery/utils.py'
# Fall back to env variable if the heredoc isn't resolving __file__
import os
p = pathlib.Path(os.environ.get('HERE', '.')) / 'signature_recovery/utils.py'
txt = p.read_text()
flags = {
    'tiniest': {'TINIEST': True,  'TINIER': False, 'TINY': True},
    'tinier':  {'TINIEST': False, 'TINIER': True,  'TINY': True},
    'tiny':    {'TINIEST': False, 'TINIER': False, 'TINY': True},
}[model]
for k, v in flags.items():
    txt = re.sub(rf'^{k}\s*=\s*(True|False)', f'{k} = {v}', txt, count=1, flags=re.M)
p.write_text(txt)
print(f"[setup] utils.py configured: {flags}")
PY

# Sign recovery has its own TINIEST/TINIER flags
HERE="$HERE" python3 - "$MODEL" <<'PY'
import re, sys, pathlib, os
model = sys.argv[1]
p = pathlib.Path(os.environ['HERE']) / 'sign_recovery/batched_sign_recovery.py'
txt = p.read_text()
flags = {
    'tiniest': {'TINIEST': True,  'TINIER': False},
    'tinier':  {'TINIEST': False, 'TINIER': True},
    'tiny':    {'TINIEST': False, 'TINIER': False},
}[model]
for k, v in flags.items():
    txt = re.sub(rf'^{k}\s*=\s*(True|False)\b', f'{k} = {v}', txt, count=1, flags=re.M)
p.write_text(txt)
print(f"[setup] batched_sign_recovery.py configured: {flags}")
PY

export HERE

# Pre-flight
mkdir -p "$SR/outputs/model_weights/Vrelu" "$SR/exp" \
         "$SIGN/layer_neuron_npys" \
         "$HERE/results/reconstructed_models" "$HERE/results/sign_recovery"

# ---- 2. find_duals (decision-boundary walk) ----
echo "=== [1/6] find_duals x$DUAL_ITERS ==="
cd "$SR"
for i in $(seq 1 "$DUAL_ITERS"); do
    $PY find_duals.py > /tmp/extract_duals_$i.log 2>&1
    echo "  iter $i/$DUAL_ITERS  files=$(ls "$SR/exp/1" 2>/dev/null | wc -l)"
done

# ---- 3. streaming cluster ----
echo "=== [2/6] streaming cluster ==="
$PY cluster_dual_points_stream.py 2>&1 | tail -8

# ---- 4. per-neuron dual files ----
echo "=== [3/6] generate per-neuron dual files ==="
$PY generate_dual_neuron.py 2>&1 | tail -3

# ---- 5. signature weight recovery (per layer) ----
echo "=== [4/6] recover weights per layer ==="
for L in 0 1 2 3; do
    $PY recover_weights.py "$L" > /tmp/extract_recover_$L.log 2>&1
    echo "  layer $L done"
done

# ---- 6. batched sign recovery ----
echo "=== [5/6] batched sign recovery ==="
cd "$HERE"
$PY sign_recovery/batched_sign_recovery.py 2>&1 | tail -6

# ---- 7. reconstruction + refinement ----
echo "=== [6/6] reconstruct + refine ==="
case "$MODEL" in
    tiniest) FLAG='--tiniest' ;;
    tinier)  FLAG='--tinier'  ;;
    tiny)    FLAG='--makeblobs' ;;
esac
$PY analysis/run_extraction.py $FLAG --from-scratch --refine --refine-epochs 1000 \
    | tee /tmp/extract_reconstruct.log | grep -E "recovered|accuracy|agreement|EXTRACTION|Saved|refine]" | tail -40

echo "=== DONE. Model saved to results/reconstructed_models/ ==="
