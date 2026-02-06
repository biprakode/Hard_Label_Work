"""
Aggregate sign recovery results into summary tables.
Reads from both individual neuron results and aggregated layer results.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# ========== Configuration ========== #
# Path to sign recovery results (individual neuron results)
NEURON_RESULTS_BASE = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/sign_recovery/results"
# Path to aggregated sign recovery results
AGGREGATED_RESULTS_PATH = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/results/sign_recovery"
# Output path for tables
OUTPUT_PATH = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/results/tables"
# Model name pattern to search for
TINIER = True
if TINIER:
    MODEL_NAME = "tinier_makeblobs_relu"
    LAYER_CONFIG = {1: 16, 2: 16, 3: 16, 4: 8}  # layerID -> num_neurons
else:
    MODEL_NAME = "TinyModel_relu"
    LAYER_CONFIG = {1: 64, 2: 64, 3: 64, 4: 64}  # layerID -> num_neurons
# ================================== #


def get_confidence_alpha(votes_p, votes_m):
    """
    Calculate the confidence level (alpha) using Hoeffding's inequality.
    Lower alpha = higher confidence that the majority vote is correct.
    """
    n = votes_p + votes_m
    if n == 0:
        return 1.0  # No data, no confidence
    p_observed = max(votes_m, votes_p) / n
    epsilon = p_observed - 0.5
    alpha = np.exp(-2 * epsilon**2 * n)
    return alpha


def analyze_neuron_df(df):
    """
    Analyze a single neuron's DataFrame to extract sign recovery metrics.
    """
    if df.empty:
        return None

    # Calculate cumulative votes
    df["votes+"] = df["Vote dOFF>dON"].cumsum()
    df["votes-"] = (~df["Vote dOFF>dON"]).cumsum()

    votes_p = int(df["votes+"].values[-1])
    votes_m = int(df["votes-"].values[-1])
    n_experiments = len(df)

    # Calculate confidence
    alpha = get_confidence_alpha(votes_p, votes_m)
    confidence = 1.0 - alpha

    # Determine sign: +1 if dOFF > dON more often (votes+ > votes-)
    recovered_sign = 1 if votes_p >= votes_m else -1

    # Calculate ratio metrics
    mean_dOFF = df["dOFF"].mean() if "dOFF" in df.columns else np.nan
    mean_dON = df["dON"].mean() if "dON" in df.columns else np.nan
    ratio = mean_dOFF / mean_dON if mean_dON > 0 else np.nan

    return {
        "votes+": votes_p,
        "votes-": votes_m,
        "n_experiments": n_experiments,
        "alpha": alpha,
        "confidence": confidence,
        "recovered_sign": recovered_sign,
        "mean_dOFF": mean_dOFF,
        "mean_dON": mean_dON,
        "dOFF/dON_ratio": ratio,
    }


def load_neuron_results_from_pkl(base_path, model_name, layer_config):
    """
    Load individual neuron results from pickle files.
    """
    all_rows = []

    for layerID, n_neurons in layer_config.items():
        for neuronID in range(n_neurons):
            # Try multiple path patterns
            path_patterns = [
                f"{base_path}/model_{model_name}/layerID_{layerID}/neuronID_{neuronID}/df.pkl",
                f"{base_path}/{model_name}/layerID_{layerID}/neuronID_{neuronID}/df.pkl",
            ]

            df = None
            for path in path_patterns:
                if os.path.exists(path):
                    try:
                        df = pd.read_pickle(path)
                        break
                    except Exception as e:
                        print(f"Error loading {path}: {e}")

            if df is None:
                continue

            # Analyze the neuron's results
            res = analyze_neuron_df(df)
            if res is None:
                continue

            res["layerID"] = layerID
            res["neuronID"] = neuronID
            all_rows.append(res)

    return pd.DataFrame(all_rows)


def load_aggregated_results(aggregated_path):
    """
    Load results from the aggregated sign recovery output (from batched_sign_recovery.py).
    """
    all_rows = []

    # Load model summary if it exists
    summary_path = os.path.join(aggregated_path, "model_sign_recovery_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            model_summary = json.load(f)

        for layer_id_str, layer_data in model_summary.get('layers', {}).items():
            layer_id = int(layer_id_str)
            signs = layer_data.get('signs', [])
            confidences = layer_data.get('confidences', [])
            votes = layer_data.get('votes', [])

            for neuron_id, (sign, conf, vote) in enumerate(zip(signs, confidences, votes)):
                if sign != 0:  # Only include processed neurons
                    all_rows.append({
                        'layerID': layer_id,
                        'neuronID': neuron_id,
                        'recovered_sign': sign,
                        'confidence': conf,
                        'votes': vote,
                    })

    # Also try loading individual layer summaries
    for layer_file in Path(aggregated_path).glob("layer*_summary.json"):
        try:
            with open(layer_file, 'r') as f:
                layer_data = json.load(f)
            layer_id = layer_data.get('layerID')
            if layer_id is None:
                continue
            # Data already loaded from model summary
        except Exception as e:
            print(f"Error loading {layer_file}: {e}")

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def create_summary_tables(df, output_path):
    """
    Create various summary tables from the results DataFrame.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 1. Full results table
    if not df.empty:
        full_path = os.path.join(output_path, "sign_recovery_full.csv")
        df.to_csv(full_path, index=False)
        print(f"Saved full results to: {full_path}")

    # 2. Per-layer summary
    if 'layerID' in df.columns:
        layer_summary = df.groupby('layerID').agg({
            'neuronID': 'count',
            'recovered_sign': lambda x: (x == 1).sum(),
            'confidence': ['mean', 'min', 'max'],
        }).round(4)
        layer_summary.columns = ['n_neurons', 'positive_signs', 'mean_confidence', 'min_confidence', 'max_confidence']
        layer_summary['negative_signs'] = layer_summary['n_neurons'] - layer_summary['positive_signs']

        layer_path = os.path.join(output_path, "sign_recovery_by_layer.csv")
        layer_summary.to_csv(layer_path)
        print(f"Saved layer summary to: {layer_path}")

        # Print to console
        print("\n" + "="*60)
        print("SIGN RECOVERY SUMMARY BY LAYER")
        print("="*60)
        print(layer_summary.to_string())

    # 3. Signs array per layer (for easy integration)
    if 'layerID' in df.columns and 'recovered_sign' in df.columns:
        signs_by_layer = {}
        for layer_id in df['layerID'].unique():
            layer_df = df[df['layerID'] == layer_id].sort_values('neuronID')
            signs = layer_df.set_index('neuronID')['recovered_sign'].to_dict()
            max_neuron = layer_df['neuronID'].max()
            signs_array = [signs.get(i, 0) for i in range(max_neuron + 1)]
            signs_by_layer[layer_id] = signs_array

            # Save as numpy array
            np_path = os.path.join(output_path, f"layer{layer_id}_signs.npy")
            np.save(np_path, np.array(signs_array, dtype=np.int8))

        # Save combined signs as JSON
        signs_json_path = os.path.join(output_path, "all_layer_signs.json")
        with open(signs_json_path, 'w') as f:
            json.dump({str(k): v for k, v in signs_by_layer.items()}, f, indent=2)
        print(f"Saved signs arrays to: {signs_json_path}")

        # Print visual representation
        print("\n" + "="*60)
        print("RECOVERED SIGNS (+ = positive, - = negative)")
        print("="*60)
        for layer_id, signs in sorted(signs_by_layer.items()):
            signs_str = ''.join(['+' if s == 1 else '-' if s == -1 else '?' for s in signs])
            print(f"Layer {layer_id}: [{signs_str}]")

    # 4. Low confidence neurons (for review)
    if 'confidence' in df.columns:
        low_conf = df[df['confidence'] < 0.95].sort_values('confidence')
        if not low_conf.empty:
            low_conf_path = os.path.join(output_path, "low_confidence_neurons.csv")
            low_conf.to_csv(low_conf_path, index=False)
            print(f"\nWarning: {len(low_conf)} neurons with confidence < 95%")
            print(f"See: {low_conf_path}")


def main():
    print("="*60)
    print("SIGN RECOVERY TABLE AGGREGATION")
    print("="*60)

    # Try loading from aggregated results first (faster)
    print("\nLoading aggregated results...")
    df_aggregated = load_aggregated_results(AGGREGATED_RESULTS_PATH)

    # Also try loading from individual pickle files
    print("Loading individual neuron results...")
    df_individual = load_neuron_results_from_pkl(NEURON_RESULTS_BASE, MODEL_NAME, LAYER_CONFIG)

    # Combine results (prefer aggregated if available)
    if not df_aggregated.empty:
        print(f"Found {len(df_aggregated)} neurons from aggregated results")
        df = df_aggregated
    elif not df_individual.empty:
        print(f"Found {len(df_individual)} neurons from individual results")
        df = df_individual
    else:
        print("No results found! Check paths:")
        print(f"  Aggregated: {AGGREGATED_RESULTS_PATH}")
        print(f"  Individual: {NEURON_RESULTS_BASE}")
        return

    # Create summary tables
    create_summary_tables(df, OUTPUT_PATH)

    print("\n" + "="*60)
    print("TABLE AGGREGATION COMPLETE")
    print("="*60)
    print(f"Output directory: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
