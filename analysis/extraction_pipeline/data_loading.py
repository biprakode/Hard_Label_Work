"""
Test data loaders + ground-truth model loader.

Two test sets are surfaced:
    * X_test  — used for Phase-3 oracle training (sign search, fc5 LR, refinement)
    * X_test2 — fresh eval-only set (seed=99, same scaler) used for final scoring
"""

import os
import numpy as np
import torch

from .config import (
    X_TEST_PATH,
    X_TEST_MAKEBLOBS_PATH,  Y_TEST_MAKEBLOBS_PATH,
    X_TEST_TINIER_PATH,     Y_TEST_TINIER_PATH,
    X_TEST_TINIEST_PATH,    Y_TEST_TINIEST_PATH,
    X_TEST2_TINIEST_PATH,   Y_TEST2_TINIEST_PATH,
    X_TEST2_MAKEBLOBS_PATH, Y_TEST2_MAKEBLOBS_PATH,
)


def load_ground_truth_model(model_path, model_class, device='cpu'):
    """Load the ground-truth model from a .pth file (handles legacy key names)."""
    model = model_class().to(device)
    state_dict = torch.load(model_path, map_location=device)

    rename_maps = [
        # TinyModel naming
        {
            "fc1.weight": "fc1.weight", "fc1.bias": "fc1.bias",
            "fc2.weight": "fc2.weight", "fc2.bias": "fc2.bias",
            "fc3.weight": "fc3.weight", "fc3.bias": "fc3.bias",
            "fc4.weight": "fc4.weight", "fc4.bias": "fc4.bias",
            "output.weight": "fc5.weight", "output.bias": "fc5.bias",
        },
        # Alternative naming with hidden_layer
        {
            "hidden_layer1.weight": "fc1.weight", "hidden_layer1.bias": "fc1.bias",
            "hidden_layer2.weight": "fc2.weight", "hidden_layer2.bias": "fc2.bias",
            "hidden_layer3.weight": "fc3.weight", "hidden_layer3.bias": "fc3.bias",
            "hidden_layer4.weight": "fc4.weight", "hidden_layer4.bias": "fc4.bias",
            "output.weight": "fc5.weight", "output.bias": "fc5.bias",
        },
    ]

    loaded = False
    try:
        model.load_state_dict(state_dict)
        loaded = True
    except RuntimeError:
        pass

    if not loaded:
        for rename_map in rename_maps:
            try:
                new_state_dict = {}
                for old_key, new_key in rename_map.items():
                    if old_key in state_dict:
                        new_state_dict[new_key] = state_dict[old_key]
                if new_state_dict:
                    model.load_state_dict(new_state_dict, strict=False)
                    loaded = True
                    break
            except Exception:
                continue

    if not loaded:
        print(f"Warning: Could not load model from {model_path}")

    return model


def load_test_data(tiny=True, makeblobs=False, tinier=False, tiniest=False):
    """Phase-3 training set."""
    if tiniest:
        if os.path.exists(X_TEST_TINIEST_PATH):
            x_test = np.load(X_TEST_TINIEST_PATH).astype(np.float64)
            y_test = (np.load(Y_TEST_TINIEST_PATH)
                      if os.path.exists(Y_TEST_TINIEST_PATH)
                      else np.zeros(len(x_test), dtype=np.int64))
        else:
            print(f"Tiniest test data not found at {X_TEST_TINIEST_PATH}")
            return None, None
        return torch.tensor(x_test, dtype=torch.float64), torch.tensor(y_test, dtype=torch.long)

    if tinier:
        if os.path.exists(X_TEST_TINIER_PATH):
            x_test = np.load(X_TEST_TINIER_PATH).astype(np.float64)
            y_test = (np.load(Y_TEST_TINIER_PATH)
                      if os.path.exists(Y_TEST_TINIER_PATH)
                      else np.zeros(len(x_test), dtype=np.int64))
        else:
            print(f"Tinier test data not found at {X_TEST_TINIER_PATH}")
            return None, None
        return torch.tensor(x_test, dtype=torch.float64), torch.tensor(y_test, dtype=torch.long)

    if makeblobs:
        if os.path.exists(X_TEST_MAKEBLOBS_PATH):
            x_test = np.load(X_TEST_MAKEBLOBS_PATH).astype(np.float64)
            y_test = (np.load(Y_TEST_MAKEBLOBS_PATH)
                      if os.path.exists(Y_TEST_MAKEBLOBS_PATH)
                      else np.zeros(len(x_test), dtype=np.int64))
        else:
            print(f"Makeblobs test data not found at {X_TEST_MAKEBLOBS_PATH}")
            return None, None
        return torch.tensor(x_test, dtype=torch.float64), torch.tensor(y_test, dtype=torch.long)

    # CIFAR-10 path
    if not os.path.exists(X_TEST_PATH):
        try:
            import tensorflow as tf
            (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            np.save(X_TEST_PATH, x_test)
        except Exception as e:
            print(f"Could not load test data: {e}")
            return None, None

    x_test = np.load(X_TEST_PATH)

    if tiny:
        if len(x_test.shape) == 4 and x_test.shape[-1] == 3:
            x_test = x_test.mean(axis=-1)
        if x_test.shape[1] > 8:
            x_test = x_test[:, ::4, ::4]
        x_test = x_test.reshape(-1, 64)
    else:
        x_test = x_test.reshape(-1, 3072)

    x_test = x_test.astype(np.float64)
    x_test = x_test / 255.0 * 2 - 1

    try:
        import tensorflow as tf
        (_, _), (_, y_test) = tf.keras.datasets.cifar10.load_data()
        y_test = y_test.squeeze()
    except Exception:
        y_test = np.zeros(len(x_test), dtype=np.int64)

    return torch.tensor(x_test, dtype=torch.float64), torch.tensor(y_test, dtype=torch.long)


def load_test2_data(tiny=True, makeblobs=False, tinier=False, tiniest=False):
    """Fresh eval-only set (seed=99, same scaler). Returns (X_test2, Y_test2)."""
    if tiniest:
        if os.path.exists(X_TEST2_TINIEST_PATH):
            x = np.load(X_TEST2_TINIEST_PATH).astype(np.float64)
            y = (np.load(Y_TEST2_TINIEST_PATH)
                 if os.path.exists(Y_TEST2_TINIEST_PATH)
                 else np.zeros(len(x), dtype=np.int64))
            return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.long)
    if makeblobs:
        if os.path.exists(X_TEST2_MAKEBLOBS_PATH):
            x = np.load(X_TEST2_MAKEBLOBS_PATH).astype(np.float64)
            y = (np.load(Y_TEST2_MAKEBLOBS_PATH)
                 if os.path.exists(Y_TEST2_MAKEBLOBS_PATH)
                 else np.zeros(len(x), dtype=np.int64))
            return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.long)
    return None, None
