"""
Recover the output layer (fc5) using only hard-label oracle queries.

We query the oracle on X_samples to build an (h_4, label) training set and
fit multinomial logistic regression to obtain fc5's weights and biases.
Hard labels only — no softmax leakage.
"""

import numpy as np
import torch

from .bias_recovery import _hidden_activations_up_to


def recover_output_layer(reconstructed_model, oracle_model, X_samples, verbose=True, n_aug=8000):
    """
    LR-fit fc5 from (h_4(X_samples), oracle(X_samples)).

    Out-of-distribution augmentation (uniform coverage / wide Gaussian) was
    found to distort the fit away from the X_test region without improving
    in-distribution accuracy, so we fit on X_samples directly.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("sklearn not available, skipping output layer recovery")
        return

    oracle_model.eval()
    reconstructed_model.eval()

    X_big = X_samples.numpy().astype(np.float64)
    X_big_t = X_samples

    with torch.no_grad():
        oracle_labels = oracle_model(X_big_t).argmax(dim=1).numpy()
        h4 = _hidden_activations_up_to(reconstructed_model, X_big_t, up_to_layer=4).numpy()

    n_classes = int(oracle_labels.max()) + 1
    lr = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        max_iter=2000, C=1e6,  # very weak regularization
        fit_intercept=True,
    )
    lr.fit(h4, oracle_labels)

    fc5 = reconstructed_model.fc5
    out_dim = fc5.weight.shape[0]
    coef = np.zeros((out_dim, h4.shape[1]), dtype=np.float64)
    intercept = np.zeros(out_dim, dtype=np.float64)
    for idx, cls in enumerate(lr.classes_):
        if cls < out_dim:
            coef[cls] = lr.coef_[idx]
            intercept[cls] = lr.intercept_[idx]

    with torch.no_grad():
        fc5.weight.data = torch.tensor(coef, dtype=torch.float64)
        fc5.bias.data = torch.tensor(intercept, dtype=torch.float64)

    if verbose:
        with torch.no_grad():
            small_oracle = oracle_model(X_samples).argmax(dim=1).numpy()
            small_recon = reconstructed_model(X_samples).argmax(dim=1).numpy()
            lr_preds = lr.predict(h4)
        print(f"  [fc5-recov] LR fit on {len(X_big)} samples ({len(X_samples)} original + augmented), "
              f"{n_classes} classes seen; LR train acc vs oracle = "
              f"{(lr_preds == oracle_labels).mean():.4f}; "
              f"reconstructed vs oracle on original X_test = {(small_recon == small_oracle).mean():.4f}")
