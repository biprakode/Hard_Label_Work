# Imports
import os
import json
import numpy as np
from pathlib import Path
# Thread-cap BEFORE sign_recovery (pulls in TF/BLAS): each Pool worker runs
# sign_recovery.main() in-process, so without this every worker grabs all cores
# for TF/BLAS -> N_workers * N_cores oversubscription (was ~190 threads on a
# 12-thread box -> ~59h). Scope each worker to 1 thread; parallelism comes from
# the Pool (nThreads workers), not from intra-op threading.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "1")
import sign_recovery

# ========== Global Settings ========== #
MAKEBLOBS = True  # Use make_blobs synthetic dataset instead of CIFAR-10
TINIEST = False  # Use tiniest 8-8-8-8-8-8 model
TINIER = False  # Use tinier model with non-uniform hidden widths (32->16->16->16->8->4)
FULL = True  # Full flagship CIFAR-10 (3072->256->256->256->64->10)

# Activation toggle. Must match signature_recovery/utils.py LEAKY_ALPHA.
#   LEAKY_ALPHA = 0.0  -> plain ReLU (DEFAULT, original pipeline preserved)
#   LEAKY_ALPHA > 0    -> Leaky ReLU(alpha) — selects *_leakyrelu.keras model
LEAKY_ALPHA = 0.01
sign_recovery.LEAKY_ALPHA = LEAKY_ALPHA  # propagate to sign_recovery activation forwards
_act_suffix              = "leakyrelu" if LEAKY_ALPHA > 0 else "relu"
_TINY_STUFF              = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/Hard_Label_Work/tiny_stuff"

if TINIEST and MAKEBLOBS:
    model_name           = "tiniest_makeblobs_4hidden_float64"
    model_path           = f"{_TINY_STUFF}/tiniest_makeblobs_{_act_suffix}.keras"
    LAYER_NEURON_COUNTS  = {1: 8, 2: 8, 3: 8, 4: 8}
elif TINIER and MAKEBLOBS:
    model_name           = "tinier_makeblobs_4hidden_float64"
    model_path           = f"{_TINY_STUFF}/tinier_makeblobs_{_act_suffix}.keras"
    # Per-layer neuron counts for tinier model (32->16->16->16->8->4)
    LAYER_NEURON_COUNTS  = {1: 16, 2: 16, 3: 16, 4: 8}
elif MAKEBLOBS:
    model_name           = "makeblobs_4x64_10_float64"
    model_path           = f"{_TINY_STUFF}/makeblobs_{_act_suffix}.keras"
    LAYER_NEURON_COUNTS  = {1: 64, 2: 64, 3: 64, 4: 64}
elif FULL:
    model_name           = "cifar10_3x256_64_10_float64"
    model_path           = f"{_TINY_STUFF}/TinyModel_{_act_suffix}.keras"
    LAYER_NEURON_COUNTS  = {1: 256, 2: 256, 3: 256, 4: 64}
else:
    model_name           = "cifar10_4x64_10_float64"
    model_path           = f"{_TINY_STUFF}/TinyModel_{_act_suffix}.keras"
    LAYER_NEURON_COUNTS  = {1: 64, 2: 64, 3: 64, 4: 64}

duals_path               = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/Hard_Label_Work/sign_recovery/layer_neuron_npys"  # Path to precomputed dual points
output_path              = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/Hard_Label_Work/results/sign_recovery"  # Output path for aggregated results
LAYERIDS                 = (1, 2, 3, 4)  # layer IDs to analyze
analyzeWiggleSensitivity = 'True'  # Record the sensitivity to the wiggle at the target layer
analyzeSpeed             = 'True'  # Record the rate of change of future layer neurons
nDebug                   = 'True'  # Set to True to skip logfile
nThreads                 = 48  # Pool workers, each single-threaded (see thread-cap above). Sized for the c2d-56 box (56 threads / 224GB); each worker loads a small Keras model — RAM is not the bound here.
# ==================================== #


def run_single_neuron(args_str):
    """Wrapper to run sign recovery for a single neuron and return the result."""
    try:
        args = args_str.split(' ')
        result = sign_recovery.main(args)
        return result
    except Exception as e:
        print(f"Error processing: {e}")
        return None


def aggregate_layer_results(layer_results, layerID, num_neurons, output_dir):
    """
    Aggregate sign recovery results for a single layer.

    Args:
        layer_results: List of sign_result dicts from sign_recovery.main()
        layerID: Layer ID
        num_neurons: Total number of neurons in the layer
        output_dir: Directory to save aggregated results

    Returns:
        Dict with aggregated statistics
    """
    # Initialize arrays
    signs = np.zeros(num_neurons, dtype=np.int8)
    confidences = np.zeros(num_neurons, dtype=np.float64)
    votes = np.zeros(num_neurons, dtype=np.int32)
    processed = np.zeros(num_neurons, dtype=bool)

    # Fill in results
    for result in layer_results:
        if result is None:
            continue
        nid = result['neuronID']
        if nid < num_neurons:
            signs[nid] = result['recovered_sign']
            confidences[nid] = result['confidence']
            votes[nid] = result['votes']
            processed[nid] = True

    # Save arrays
    np.save(os.path.join(output_dir, f'layer{layerID}_signs.npy'), signs)
    np.save(os.path.join(output_dir, f'layer{layerID}_confidences.npy'), confidences)
    np.save(os.path.join(output_dir, f'layer{layerID}_votes.npy'), votes)

    # Calculate statistics
    n_processed = np.sum(processed)
    n_positive = np.sum(signs[processed] == 1)
    n_negative = np.sum(signs[processed] == -1)
    mean_confidence = np.mean(confidences[processed]) if n_processed > 0 else 0.0
    min_confidence = np.min(confidences[processed]) if n_processed > 0 else 0.0

    # Save detailed JSON
    layer_summary = {
        'layerID': layerID,
        'num_neurons': num_neurons,
        'neurons_processed': int(n_processed),
        'neurons_positive_sign': int(n_positive),
        'neurons_negative_sign': int(n_negative),
        'mean_confidence': float(mean_confidence),
        'min_confidence': float(min_confidence),
        'signs': signs.tolist(),
        'confidences': confidences.tolist(),
        'votes': votes.tolist(),
        'neurons_missing': [i for i in range(num_neurons) if not processed[i]],
    }

    with open(os.path.join(output_dir, f'layer{layerID}_summary.json'), 'w') as f:
        json.dump(layer_summary, f, indent=2)

    print(f"\n=== Layer {layerID} Summary ===")
    print(f"Neurons processed: {n_processed}/{num_neurons}")
    print(f"Positive signs: {n_positive}, Negative signs: {n_negative}")
    print(f"Mean confidence: {mean_confidence:.4f}, Min confidence: {min_confidence:.4f}")
    print(f"Results saved to: {output_dir}/layer{layerID}_*.npy")

    return layer_summary


def main():
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Store all results for final aggregation
    all_layer_summaries = {}

    # Set up multiprocessing if needed
    if nThreads > 1:
        import multiprocessing
        pool = multiprocessing.Pool(nThreads)

    for layerID in LAYERIDS:
        print(f"\n{'='*60}")
        print(f"Processing Layer {layerID}")
        print(f"{'='*60}")

        # Get neuron count for this layer from config
        num_neurons_in_layer = LAYER_NEURON_COUNTS.get(layerID, 0)
        if num_neurons_in_layer == 0:
            print(f"Warning: No neuron count configured for layer {layerID}, skipping")
            continue

        NEURONIDS = range(num_neurons_in_layer)

        # Parameter adjustments for each layer.
        # IMPORTANT: `perfect_control_along_decision_boundary` caps dOFF at 3*dON
        # (see sign_recovery.py:397-398). Under that cap dOFF >= dON always, so
        # votes get pushed to +1 regardless of the true sign. For layers whose
        # future-toggle signal is sparse (layer 1 and the last hidden layer)
        # this produces 100%-confidence but structurally biased +1 outputs.
        # Using `along_decision_boundary` for ALL layers removes the cap and
        # lets dOFF/dON votes reflect the true sign asymmetry.
        # Reduced caps for tinier/larger architectures. The original
        # 1000/10000 nExpMin/nExp values stalled tiniest at layer-2 neuron-7
        # for >30 minutes on a degenerate boundary walk. With α=0.01 the
        # dON/dOFF asymmetry is strong enough that 200/2000 is plenty for
        # confident recovery, and bounding nExp prevents pathological hangs.
        if layerID == 1:
            nExpMin = 200
            nExp = 2_000
            choose_dx = 'along_decision_boundary'
        elif layerID in (2, 3):
            nExpMin = 200
            nExp = 2_000
            choose_dx = 'along_decision_boundary'
        elif layerID == 4:
            nExpMin = 100
            nExp = 1_000
            choose_dx = 'along_decision_boundary'
        else:
            print(f"Warning: Skipping layer {layerID} (not configured)")
            continue

        # Build argument strings for each neuron
        args_list = []
        for neuronID in NEURONIDS:
            filepath_load_x0 = f"{duals_path}/layer{layerID}_neuron{neuronID}.npy"

            # Check if dual points file exists
            if not os.path.exists(filepath_load_x0):
                print(f"Warning: Dual points file not found: {filepath_load_x0}")
                continue

            args_str = (
                f"--model {model_path} "
                f"--layerID {layerID} "
                f"--neuronID {neuronID} "
                f"--nExp {nExp} "
                f"--analyzeWiggleSensitivity {analyzeWiggleSensitivity} "
                f"--analyzeSpeed {analyzeSpeed} "
                f"--handlePrevLayerToggles True "
                f"--nToggles 1 "
                f"--nDebug {nDebug} "
                f"--filepath_load_x0 {filepath_load_x0} "
                f"--nExpMin {nExpMin} "
                f"--choose_dx {choose_dx}"
            )
            args_list.append(args_str)

        # Run sign recovery for all neurons
        layer_results = []

        if nThreads > 1:
            # Parallel execution
            async_results = [pool.apply_async(run_single_neuron, (args,)) for args in args_list]
            for async_result in async_results:
                try:
                    result = async_result.get(timeout=3600)  # 1 hour timeout per neuron
                    layer_results.append(result)
                except Exception as e:
                    print(f"Error getting result: {e}")
                    layer_results.append(None)
        else:
            # Sequential execution
            for args_str in args_list:
                result = run_single_neuron(args_str)
                layer_results.append(result)

        # Aggregate results for this layer
        layer_summary = aggregate_layer_results(layer_results, layerID, num_neurons_in_layer, output_path)
        all_layer_summaries[layerID] = layer_summary

    # Close pool if using multiprocessing
    if nThreads > 1:
        pool.close()
        pool.join()

    # Save complete model summary
    model_summary = {
        'model_name': model_name,
        'model_path': model_path,
        'layer_neuron_counts': LAYER_NEURON_COUNTS,
        'layers': all_layer_summaries,
        'total_neurons': sum(s['num_neurons'] for s in all_layer_summaries.values()),
        'total_processed': sum(s['neurons_processed'] for s in all_layer_summaries.values()),
    }

    with open(os.path.join(output_path, 'model_sign_recovery_summary.json'), 'w') as f:
        json.dump(model_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SIGN RECOVERY COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"Total neurons processed: {model_summary['total_processed']}/{model_summary['total_neurons']}")

    # Print per-layer sign arrays for quick reference
    print("\n=== Recovered Signs per Layer ===")
    for lid, summary in all_layer_summaries.items():
        signs_str = ''.join(['+' if s == 1 else '-' if s == -1 else '?' for s in summary['signs']])
        print(f"Layer {lid}: [{signs_str}]")

    return model_summary


if __name__ == '__main__':
    main()
