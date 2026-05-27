#!/usr/bin/env bash
# Drop-in replacement for the STEP-2 find_duals shell loop in run_one_model.sh.
# Produces pickles in the SAME format and SAME directory (exp/{SEED}/) as the
# original; downstream cluster_dual_points_stream.py is unchanged.
#
# Usage:  ./run_duals_torch.sh [ITERATIONS] [WORKERS] [BATCH_SIZE] [IMPL]
#   ITERATIONS  number of pickle rounds       (default 9)
#   WORKERS     concurrent worker processes   (default = cores/2)
#   BATCH_SIZE  walks per batch, torch impl    (default 256)
#   IMPL        torch | subprocess            (default torch)
#
# Arch/activation come from signature_recovery/utils.py toggles (set by
# run_one_model.sh STEP 1), exactly like the original.
set -euo pipefail

ITERS="${1:-9}"
WORKERS="${2:-}"
BATCH="${3:-256}"
IMPL="${4:-torch}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/biprarshi/miniconda3/envs/MLenv/bin/python3}"
cd "$HERE/signature_recovery"

WARG=()
[ -n "$WORKERS" ] && WARG=(--workers "$WORKERS")

"$PY" torch_impl/parallel_duals.py \
    --iterations "$ITERS" \
    --batch-size "$BATCH" \
    --impl "$IMPL" \
    "${WARG[@]}"
