"""
Emit X_test3 for the three make_blobs arches (makeblobs / tinier / tiniest).

Translates the CIFAR-flagship X_test3 honest-eval contract to the make_blobs
arches: a third disjoint draw, fit against the original train-side scaler so the
distribution matches X_test and X_test2 exactly. Models are never touched —
only the data files emitted.

  data/x_test3_makeblobs.npy + y_test3_makeblobs.npy           (5000 rows)
  data/x_test3_tinier_makeblobs.npy + y_test3_tinier_makeblobs.npy  (5000 rows)
  data/x_test3_tiniest_makeblobs.npy + y_test3_tiniest_makeblobs.npy (1000 rows)

Each draw is generated with seed=123 (X_test=42, X_test2=99, X_test3=123 — all
disjoint). The scaler + centers are reproduced from the original train pipeline
deterministically (same seed=42, same train/test split), then applied to the new
draw.
"""
import os
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 42
TEST3_SEED = 123


def emit(arch_name, n_samples, n_features, n_classes, n_test, cluster_std, n_test3):
    np.random.seed(SEED)

    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_classes,
        cluster_std=cluster_std, random_state=SEED, shuffle=True,
    )
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=n_test, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X_train)

    _Xf, _yf, centers = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_classes,
        cluster_std=cluster_std, random_state=SEED, shuffle=True,
        return_centers=True,
    )

    X3, y3 = make_blobs(
        n_samples=n_test3, centers=centers, cluster_std=cluster_std,
        random_state=TEST3_SEED, shuffle=True,
    )
    X3 = (np.clip(scaler.transform(X3), -3, 3) / 3.0).astype(np.float64)
    y3 = y3.astype(np.int64)

    x_path = os.path.join(DATA_DIR, f"x_test3_{arch_name}.npy")
    y_path = os.path.join(DATA_DIR, f"y_test3_{arch_name}.npy")
    np.save(x_path, X3)
    np.save(y_path, y3)
    print(f"  {arch_name}: X3 {X3.shape} dtype={X3.dtype} -> {x_path}")
    print(f"  {arch_name}: y3 {y3.shape} class-counts={np.bincount(y3)}")


if __name__ == "__main__":
    print("emitting X_test3 for make_blobs arches (seed=123)")
    print("-" * 60)
    emit("makeblobs",         n_samples=15000, n_features=64, n_classes=10, n_test=10000, cluster_std=3.0, n_test3=5000)
    emit("tinier_makeblobs",  n_samples=15000, n_features=32, n_classes=4,  n_test=10000, cluster_std=2.0, n_test3=5000)
    emit("tiniest_makeblobs", n_samples=12000, n_features=8,  n_classes=8,  n_test=2000,  cluster_std=1.6, n_test3=1000)
    print("-" * 60)
    print("done.")
