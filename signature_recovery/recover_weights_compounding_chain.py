"""
Prefix-init COMPOUNDING-chain harness (non-confirmatory rerun).

Unlike recover_weights_recovered_prefix.py's confirmatory experiment (which
swaps ONE layer's prefix against a fixed, independently-recovered baseline
and found a null result -- because that baseline was built from
scaling-cheat-ON, sign-walk-cheat-ON artifacts that are numerically
near-identical to the true weights), this script builds a genuinely
RECURSIVE chain: layer L's prefix is the layers 0..L-1 this SAME chain
already reconstructed (honest scaling, NO_SIG_CHEAT=1), not the true model
and not a static independently-cheating baseline.

Precondition (see run_chain_sweep.sh): a canonical setup pass has already run
with NO_SIG_CHEAT=1 STOP_AFTER_PHASE2=1, populating:
  - signature_recovery/exp/1-cluster-{0,1,2,3}.p   (cluster pickles, prefix-independent)
  - signature_recovery/outputs/model_weights/Vrelu/layer_0  (honest, prefix-TRIVIAL --
    layer 0 has no earlier layers, so "compounding" vs "true" prefix is moot for it)
  - results/sign_recovery/layer{1,2,3,4}_signs.npy  (Phase-2's ACTUAL statistical
    sign per neuron id, computed by the always-on sign-walk cheat against the
    TRUE model -- this script deliberately does NOT recompute these; it reuses
    them as-is against whatever new direction this chain recovers for that
    neuron id, exactly as instructed: "Use Phase-2's actual statistical sign".)

What this script does: for LAYER in (1, 2, 3), build a prefix model from the
CURRENT on-disk state (weight_assembly.reconstruct_model over layers 0..LAYER-1,
whatever this chain has written there so far -- honest magnitude x reused
sign), monkeypatch recover_weights.cheat_net_cpu to it, and call the
UNMODIFIED dosteal(LAYER, cluster) (still with NO_SIG_CHEAT=1). dosteal()
overwrites signature_recovery/outputs/.../layer_{LAYER} on disk, which is
then the input the NEXT iteration's reconstruct_model() call picks up --
this is the actual compounding: layer 2's prefix includes layer 1 as THIS
chain recovered it, not as the canonical/true-prefix run recovered it.

Biases: kept as TRUE biases (same simplification as the confirmatory
experiment -- bias recovery is separate, out-of-scope machinery here).

Usage:
    NO_SIG_CHEAT=1 python recover_weights_compounding_chain.py <arch_key>
    arch_key: tiniest | tinier | makeblobs
"""
import os
import sys
import pickle

assert os.environ.get("NO_SIG_CHEAT") == "1", (
    "recover_weights_compounding_chain.py must be run with NO_SIG_CHEAT=1 "
    "(honest scaling) -- this is the whole point of the rerun, see module docstring."
)

ARCH_KEY = sys.argv[1]

_HERE = os.path.dirname(os.path.abspath(__file__))
_HLW = os.path.dirname(_HERE)
_ANALYSIS = os.path.join(_HLW, "analysis")
for p in (_ANALYSIS, _HLW):
    if p not in sys.path:
        sys.path.insert(0, p)

# recover_weights.py reads sys.argv[1] as a filename prefix at import time
# (`file = open(f"{sys.argv[1]}_weight_vectors.txt", "w")`) -- harmless
# side-effect file, reused across all 3 layers processed by this script.
sys.argv = [sys.argv[0], "chain"]

from extraction_pipeline.workflow import _ARCHS
from extraction_pipeline.config import SIGNATURE_WEIGHTS_PATH, SIGN_RECOVERY_PATH
from extraction_pipeline.weight_assembly import reconstruct_model

model_class, true_path, layer_config, label = _ARCHS[ARCH_KEY]

sys.path.insert(0, _HERE)
import recover_weights as rw

assert rw.NO_SIG_CHEAT, "recover_weights.py did not pick up NO_SIG_CHEAT=1"

print(f"[compounding_chain] arch={ARCH_KEY} ({label})")
print(f"  signature_path={SIGNATURE_WEIGHTS_PATH}")
print(f"  sign_path={SIGN_RECOVERY_PATH} (reused as-is, not recomputed)")

_true_cheat_net_cpu = rw.cheat_net_cpu

# dosteal() only os.makedirs(exist_ok=True) a neuron's output dir and writes
# weights_unscaled.npz/metadata.json ONLY when soln is not None -- it never
# cleans stale neuron dirs from a prior run. The canonical setup pass already
# wrote layer_1/2/3 dirs (honest scaling, but TRUE prefix -- the recovery
# we're specifically trying NOT to measure here). Without wiping them first,
# any cluster where the compounding-prefix extraction FAILS would silently
# keep showing the setup pass's true-prefix success, masking exactly the
# degradation this rerun exists to detect. Layer 0 is untouched (its prefix
# is trivially empty, "compounding" doesn't apply to it).
import shutil
for _L in (1, 2, 3):
    _d = os.path.join(SIGNATURE_WEIGHTS_PATH, f"layer_{_L}")
    if os.path.isdir(_d):
        shutil.rmtree(_d)
    os.makedirs(_d, exist_ok=True)
print("[compounding_chain] wiped stale layer_1/2/3 (true-prefix) artifacts "
      "from the setup pass before starting the compounding chain")

try:
    for LAYER in (1, 2, 3):
        print(f"\n=== LAYER={LAYER}: building prefix from chain-reconstructed "
              f"layers 0..{LAYER - 1} (on-disk state as this chain left it) ===")
        reconstructed_model, _metrics, recovery_stats, _masks = reconstruct_model(
            SIGNATURE_WEIGHTS_PATH, SIGN_RECOVERY_PATH,
            model_class, layer_config, true_path, random_seed=42,
            copy_true_biases=True,   # not under test here, see module docstring
            copy_true_output=False,
        )
        print(f"  Prefix source recovery: "
              f"{recovery_stats['recovered_neurons']}/{recovery_stats['total_neurons']} "
              "neurons across ALL 4 layers reconstructed (only layers "
              f"0..{LAYER - 1} are actually used as this LAYER's prefix; "
              "later layers in this count are stale/irrelevant until their own turn)")

        rw.cheat_net_cpu = reconstructed_model
        rw.layer = LAYER

        cluster_path = os.path.join(_HERE, "exp", "1-cluster-%d.p" % LAYER)
        with open(cluster_path, "rb") as f:
            cluster = pickle.load(f)

        rw.dosteal(LAYER, cluster)
        print(f"[compounding_chain] LAYER={LAYER} done -- "
              f"signature_recovery/outputs/.../layer_{LAYER} now reflects "
              "recovery against the COMPOUNDING (chain-reconstructed) prefix, "
              "not the true model.")
finally:
    rw.cheat_net_cpu = _true_cheat_net_cpu

print("\n[compounding_chain] all layers (1,2,3) done. Layer 0 and "
      f"{SIGN_RECOVERY_PATH} are unchanged from the canonical setup pass -- "
      "layer 0 has no prefix to compound (recovery there was already honest "
      "and prefix-independent), and signs are Phase-2's actual statistical "
      "output reused by neuron id against whatever new direction was "
      "recovered here, per instruction.")
