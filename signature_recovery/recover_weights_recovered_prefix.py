"""
Prefix-init CONFIRMATORY harness (one-off, not a permanent flag).

The paper's Section 3.4 stance is that the prefix-init cheat in
recover_weights.py's dosteal() (`transfer_weights(cheat_net_cpu, prefix)`,
building layer L's linearization prefix from the TRUE weights of layers
0..L-1) should be RETAINED, not fixed -- replacing it with a reconstructed
prefix would compound the first/last-hidden-layer sign blind spot into every
deeper layer for no honesty gain. This script exists only to measure HOW MUCH
recovery/sign-accuracy degrades if you did swap it, as evidence supporting
that non-removal decision -- not as a step toward shipping the honest
replacement. It does not modify recover_weights.py; dosteal() there is
untouched and stays exactly as before for every other cheat/run in this study.

Mechanism: dosteal() reads the module-global `cheat_net_cpu` (via
transfer_weights(cheat_net_cpu, prefix)) at call time -- Python functions look
up globals dynamically, so monkeypatching recover_weights.cheat_net_cpu to a
RECONSTRUCTED (Phase-1+2-combined, not true) model before calling the
unmodified dosteal() makes it build the prefix from recovered weights instead,
with zero duplication of dosteal()'s ~130 lines of extraction logic.

Two-pass protocol (see cheating_ablation/reports/prefix_init_confirmatory/):
  1. A canonical run_one_model_enhanced.sh pass already populated complete
     Phase-1 (unsigned, signature_recovery/outputs/model_weights/Vrelu/) and
     Phase-2 (signs, results/sign_recovery/) artifacts for ALL layers.
  2. This script rebuilds ONE target layer's output using a prefix sourced
     from those (static, already-recovered) artifacts instead of the true
     model, and reports how that layer's own recovery degrades. It does NOT
     recursively re-peel earlier layers with their own reconstructed
     prefixes -- each layer's confirmatory measurement is independent,
     against the same fixed "what Phase 1+2 already recovered" baseline.

Usage:
    python recover_weights_recovered_prefix.py <LAYER> <arch_key>
    arch_key: tiniest | tinier | makeblobs   (matches extraction_pipeline._ARCHS)

Biases: kept as TRUE biases for this first cut (biases aren't the axis under
test here; using true biases isolates the prefix-weights question from a
second, orthogonal cheat).
"""
import os
import sys
import pickle

LAYER = int(sys.argv[1])
ARCH_KEY = sys.argv[2]

_HERE = os.path.dirname(os.path.abspath(__file__))
_HLW = os.path.dirname(_HERE)
_ANALYSIS = os.path.join(_HLW, "analysis")
for p in (_ANALYSIS, _HLW):
    if p not in sys.path:
        sys.path.insert(0, p)

# recover_weights.py reads sys.argv[1] as a filename prefix at import time
# (`file = open(f"{sys.argv[1]}_weight_vectors.txt", "w")`) -- harmless
# side-effect file, keep argv consistent with its own convention.
sys.argv = [sys.argv[0], str(LAYER)]

from extraction_pipeline.workflow import _ARCHS
from extraction_pipeline.config import SIGNATURE_WEIGHTS_PATH, SIGN_RECOVERY_PATH
from extraction_pipeline.weight_assembly import reconstruct_model

model_class, true_path, layer_config, label = _ARCHS[ARCH_KEY]

print(f"[recover_weights_recovered_prefix] LAYER={LAYER} arch={ARCH_KEY} ({label})")
print("Building reconstructed prefix from on-disk Phase-1+2 artifacts "
      f"(NOT true weights): {SIGNATURE_WEIGHTS_PATH}, {SIGN_RECOVERY_PATH}")

reconstructed_model, _metrics, recovery_stats, _masks = reconstruct_model(
    SIGNATURE_WEIGHTS_PATH, SIGN_RECOVERY_PATH,
    model_class, layer_config, true_path, random_seed=42,
    copy_true_biases=True,   # biases not under test here, see module docstring
    copy_true_output=False,
)
print(f"  Reconstructed-prefix recovery: "
      f"{recovery_stats['recovered_neurons']}/{recovery_stats['total_neurons']} "
      f"neurons across earlier layers")

sys.path.insert(0, _HERE)
import recover_weights as rw

_true_cheat_net_cpu = rw.cheat_net_cpu
rw.cheat_net_cpu = reconstructed_model
print("Monkeypatched recover_weights.cheat_net_cpu -> reconstructed model "
      "(dosteal()'s transfer_weights(cheat_net_cpu, prefix) now reads this)")

# dosteal()'s extract_weights(maybe, prefix, layer) call references the
# module-level global `layer` (lowercase), which recover_weights.py's own
# `if __name__ == '__main__':` block sets before calling dosteal() normally
# (`layer = int(sys.argv[1])`). That block never runs on import, so replicate
# its one assignment here -- not a bug fix, just reproducing what the normal
# CLI entry point already does.
rw.layer = LAYER

cluster_path = os.path.join(_HERE, "exp", "1-cluster-%d.p" % LAYER)
with open(cluster_path, "rb") as f:
    cluster = pickle.load(f)

try:
    rw.dosteal(LAYER, cluster)
finally:
    rw.cheat_net_cpu = _true_cheat_net_cpu

print(f"[recover_weights_recovered_prefix] LAYER={LAYER} done. "
      f"Output written to the same signature_recovery/outputs/model_weights/Vrelu/layer_{LAYER}/ "
      "path dosteal() always uses -- caller is responsible for snapshotting "
      "the canonical (true-prefix) output before invoking this script, and "
      "restoring it afterward if subsequent layers/victims need the canonical baseline.")
