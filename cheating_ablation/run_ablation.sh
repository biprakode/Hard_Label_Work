#!/usr/bin/env bash
#
# cheating_ablation/run_ablation.sh — one-command end-to-end reproduction of
# every tested cheat's ON/OFF ablation (6 make_blobs victims x 2 arms each),
# canonical 2026-06-21 parameters held fixed for everything not under test.
#
# See REPRODUCE.md for what each cheat is, what's deliberately not run
# (Phase 3, CIFAR-10), and where results land. Not included here: the
# prefix-init confirmatory experiment (#6, locked/manual two-pass protocol,
# not a single sweep -- see reports/prefix_init_confirmatory/results_table.md).
#
# Usage: ./cheating_ablation/run_ablation.sh [cheat_name ...]
#   No args: run all 5 sweepable cheats in pipeline order.
#   Args: run only the named cheat(s), e.g.
#         ./cheating_ablation/run_ablation.sh boundary_detection sign_walk
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HLW="$(cd "$HERE/.." && pwd)"
cd "$HLW"

run_boundary_detection() {
    echo "### boundary_detection ###"
    "$HERE/run_one_cheat_sweep.sh" HONEST_BOUNDARY_DETECT boundary_detection
}
run_boundary_refinement() {
    echo "### boundary_refinement ###"
    "$HERE/run_one_cheat_sweep.sh" HONEST_BOUNDARY_REFINE boundary_refinement
}
run_neuron_clustering() {
    echo "### neuron_clustering ###"
    "$HERE/reports/neuron_clustering/run_sweep.sh"
}
run_signature_scaling_rerun() {
    echo "### signature_scaling_rerun (NO_SIG_CHEAT) ###"
    "$HERE/run_one_cheat_sweep.sh" NO_SIG_CHEAT signature_scaling_rerun
}
run_sign_walk() {
    echo "### sign_walk ###"
    "$HERE/reports/sign_walk/run_sweep.sh"
}

ALL_CHEATS=(boundary_detection boundary_refinement neuron_clustering signature_scaling_rerun sign_walk)

TARGETS=("$@")
if [ "${#TARGETS[@]}" -eq 0 ]; then
    TARGETS=("${ALL_CHEATS[@]}")
fi

for t in "${TARGETS[@]}"; do
    case "$t" in
        boundary_detection)       run_boundary_detection ;;
        boundary_refinement)      run_boundary_refinement ;;
        neuron_clustering)        run_neuron_clustering ;;
        signature_scaling_rerun)  run_signature_scaling_rerun ;;
        sign_walk)                run_sign_walk ;;
        *) echo "Unknown cheat: $t (expected one of: ${ALL_CHEATS[*]})" >&2; exit 1 ;;
    esac
done

echo ""
echo "DONE. Per-cheat results_table.md / observations.md under cheating_ablation/reports/<cheat>/"
echo "See cheating_ablation/cheat_disable_map.md for the consolidated status table."
