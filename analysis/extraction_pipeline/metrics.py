"""
Three-tier per-neuron weight metrics + accuracy testing.

Three-tier metrics:
    SIGN      - sign(cosine_sim) per neuron  -> sign accuracy
    MAGNITUDE - relative error after sign-aligning  -> magnitude accuracy
    COMBINED  - relative error without alignment  -> overall accuracy
"""

import numpy as np
import torch


def compute_weight_metrics_v2(extracted_weights, true_weights):
    """Three-tier metrics separating sign from magnitude analysis."""
    if extracted_weights is None or true_weights is None:
        return None
    if extracted_weights.shape != true_weights.shape:
        print(f"Shape mismatch: extracted {extracted_weights.shape} vs true {true_weights.shape}")
        return None

    n_neurons = extracted_weights.shape[0]
    per_neuron = []
    correct_signs = 0
    magnitude_errors = []
    combined_errors = []

    for i in range(n_neurons):
        ext = extracted_weights[i]
        true = true_weights[i]
        ext_norm = np.linalg.norm(ext)
        true_norm = np.linalg.norm(true)

        if ext_norm < 1e-12 or true_norm < 1e-12:
            per_neuron.append({
                'neuron': i, 'cosine_sim': 0.0, 'sign_correct': False,
                'magnitude_rel_error': 1.0, 'combined_rel_error': 1.0
            })
            combined_errors.append(1.0)
            magnitude_errors.append(1.0)
            continue

        cos_sim = np.dot(ext, true) / (ext_norm * true_norm)
        sign_correct = cos_sim > 0

        aligned_ext = ext if cos_sim > 0 else -ext
        mag_rel_error = np.linalg.norm(aligned_ext - true) / true_norm
        comb_rel_error = np.linalg.norm(ext - true) / true_norm

        if sign_correct:
            correct_signs += 1

        magnitude_errors.append(mag_rel_error)
        combined_errors.append(comb_rel_error)

        per_neuron.append({
            'neuron': i,
            'cosine_sim': float(cos_sim),
            'sign_correct': bool(sign_correct),
            'magnitude_rel_error': float(mag_rel_error),
            'combined_rel_error': float(comb_rel_error),
        })

    return {
        'sign_accuracy':            correct_signs / n_neurons if n_neurons > 0 else 0,
        'magnitude_mean_rel_error': float(np.mean(magnitude_errors))   if magnitude_errors else 1.0,
        'magnitude_median_rel_error': float(np.median(magnitude_errors)) if magnitude_errors else 1.0,
        'combined_mean_rel_error':  float(np.mean(combined_errors))    if combined_errors  else 1.0,
        'combined_median_rel_error': float(np.median(combined_errors))  if combined_errors  else 1.0,
        'mean_abs_cosine_sim':      float(np.mean([abs(p['cosine_sim']) for p in per_neuron])),
        'n_neurons': n_neurons,
        'per_neuron': per_neuron,
    }


def test_model_accuracy(model, X_test, Y_test, model_name="Model"):
    """Top-1 accuracy on a (X_test, Y_test) tensor pair."""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = outputs.argmax(dim=1)
        correct = (predictions == Y_test).sum().item()
        accuracy = correct / len(Y_test)

    print(f"{model_name} Accuracy: {accuracy:.4f} ({correct}/{len(Y_test)})")
    return accuracy
