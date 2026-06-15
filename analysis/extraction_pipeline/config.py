"""
Paths, activation toggle, and the _act helper used by every architecture
defined in `architectures.py`.

The single source of truth for activation choice is `LEAKY_ALPHA`:
    LEAKY_ALPHA = 0.0   ->  plain ReLU (original pipeline preserved exactly)
    LEAKY_ALPHA  > 0    ->  Leaky ReLU(alpha)

This must be kept in sync with `signature_recovery/utils.py::LEAKY_ALPHA`
and `sign_recovery/sign_recovery.py::LEAKY_ALPHA`.
"""

import os
import torch.nn.functional as F


# ---------------------------------------------------------------- base paths --
BASE_DIR = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/Hard_Label_Work"


# ----------------------------------------------------------- activation toggle --
LEAKY_ALPHA = 0.01


def _act(x):
    """ReLU mode (LEAKY_ALPHA == 0): F.relu. Leaky mode: F.leaky_relu(x, alpha)."""
    if LEAKY_ALPHA > 0:
        return F.leaky_relu(x, negative_slope=LEAKY_ALPHA)
    return F.relu(x)


_act_suffix = "leakyrelu" if LEAKY_ALPHA > 0 else "relu"


# -------------------------------------------------------- recovery I/O paths --
SIGNATURE_WEIGHTS_PATH = os.path.join(BASE_DIR, "signature_recovery/outputs/model_weights/Vrelu")
SIGN_RECOVERY_PATH     = os.path.join(BASE_DIR, "results/sign_recovery")


# --------------------------------------------------------- ground-truth models --
TINY_MODEL_PTH      = os.path.join(BASE_DIR, f"tiny_stuff/TinyModel_{_act_suffix}.pth")
TINY_MODEL_KERAS    = os.path.join(BASE_DIR, f"tiny_stuff/TinyModel_{_act_suffix}.keras")
MAKEBLOBS_MODEL_PTH = os.path.join(BASE_DIR, f"tiny_stuff/makeblobs_{_act_suffix}.pth")
TINIER_MODEL_PTH    = os.path.join(BASE_DIR, f"tiny_stuff/tinier_makeblobs_{_act_suffix}.pth")
TINIEST_MODEL_PTH   = os.path.join(BASE_DIR, f"tiny_stuff/tiniest_makeblobs_{_act_suffix}.pth")
# Flagship CIFAR victim: same file utils.MODEL_PATH resolves to (suffix-keyed).
FULL_MODEL_PTH      = os.path.join(BASE_DIR, f"tiny_stuff/TinyModel_{_act_suffix}.pth")


# --------------------------------------------------------------- test data paths --
# X_test: used for Phase-3 oracle training (sign search, fc5 LR, refinement)
X_TEST_PATH                = os.path.join(BASE_DIR, "data/x_test.npy")
Y_TEST_PATH                = os.path.join(BASE_DIR, "data/y_test.npy")        # CIFAR test labels
# CIFAR fresh eval-only set (held-out train slice; no Phase-3 training overlap)
X_TEST2_CIFAR_PATH         = os.path.join(BASE_DIR, "data/x_test2_cifar.npy")
Y_TEST2_CIFAR_PATH         = os.path.join(BASE_DIR, "data/y_test2_cifar.npy")
# X_test3: second disjoint CIFAR train slice (10000-19999). When the enhanced
# Phase 3 pipeline promotes X_test2 into the query/training tier, X_test3 is
# the held-out eval set + early-stop watchdog (so the headline metric is
# computed on data that was NEVER queried or used to early-stop, only scored).
X_TEST3_CIFAR_PATH         = os.path.join(BASE_DIR, "data/x_test3_cifar.npy")
Y_TEST3_CIFAR_PATH         = os.path.join(BASE_DIR, "data/y_test3_cifar.npy")
X_TEST_MAKEBLOBS_PATH      = os.path.join(BASE_DIR, "data/x_test_makeblobs.npy")
Y_TEST_MAKEBLOBS_PATH      = os.path.join(BASE_DIR, "data/y_test_makeblobs.npy")
X_TEST_TINIER_PATH         = os.path.join(BASE_DIR, "data/x_test_tinier_makeblobs.npy")
Y_TEST_TINIER_PATH         = os.path.join(BASE_DIR, "data/y_test_tinier_makeblobs.npy")
X_TEST_TINIEST_PATH        = os.path.join(BASE_DIR, "data/x_test_tiniest_makeblobs.npy")
Y_TEST_TINIEST_PATH        = os.path.join(BASE_DIR, "data/y_test_tiniest_makeblobs.npy")

# X_test2: fresh eval-only set (seed=99, same scaler) — no Phase-3 training overlap
X_TEST2_TINIEST_PATH       = os.path.join(BASE_DIR, "data/x_test2_tiniest_makeblobs.npy")
Y_TEST2_TINIEST_PATH       = os.path.join(BASE_DIR, "data/y_test2_tiniest_makeblobs.npy")
X_TEST2_TINIER_PATH        = os.path.join(BASE_DIR, "data/x_test2_tinier_makeblobs.npy")
Y_TEST2_TINIER_PATH        = os.path.join(BASE_DIR, "data/y_test2_tinier_makeblobs.npy")
X_TEST2_MAKEBLOBS_PATH     = os.path.join(BASE_DIR, "data/x_test2_makeblobs.npy")
Y_TEST2_MAKEBLOBS_PATH     = os.path.join(BASE_DIR, "data/y_test2_makeblobs.npy")

# X_test3: third disjoint draw (seed=123 for make_blobs, CIFAR train slice
# 10000-19999 for the flagship). Held-out eval + early-stop watchdog. Never
# queried, never used for fc5 LR or sign-flip decisions other than
# restart-selection (see sign_search).
X_TEST3_MAKEBLOBS_PATH     = os.path.join(BASE_DIR, "data/x_test3_makeblobs.npy")
Y_TEST3_MAKEBLOBS_PATH     = os.path.join(BASE_DIR, "data/y_test3_makeblobs.npy")
X_TEST3_TINIER_PATH        = os.path.join(BASE_DIR, "data/x_test3_tinier_makeblobs.npy")
Y_TEST3_TINIER_PATH        = os.path.join(BASE_DIR, "data/y_test3_tinier_makeblobs.npy")
X_TEST3_TINIEST_PATH       = os.path.join(BASE_DIR, "data/x_test3_tiniest_makeblobs.npy")
Y_TEST3_TINIEST_PATH       = os.path.join(BASE_DIR, "data/y_test3_tiniest_makeblobs.npy")


# --------------------------------------------------------------- aux paths --
DUAL_POINTS_DIR = os.path.join(BASE_DIR, "sign_recovery/layer_neuron_npys")
OUTPUT_PATH     = os.path.join(BASE_DIR, "results/reconstructed_models")
