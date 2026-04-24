"""
Evaluate the reconstructed tiniest model on the original sklearn make_blobs task.

Loads:
  - Oracle: tiny_stuff/tiniest_makeblobs_relu.pth (ground-truth trained model)
  - Reconstructed: results/reconstructed_models/reconstructed_tiniest.pth
  - Test split: data/x_test_tiniest_makeblobs.npy, data/y_test_tiniest_makeblobs.npy
  - Regenerates make_blobs with the same seed for a full-dataset eval (train + test)

Reports:
  - Test accuracy vs TRUE labels
  - Train accuracy vs TRUE labels
  - Full-dataset accuracy vs TRUE labels
  - Prediction agreement vs oracle model
  - Per-class breakdown
  - Confusion matrix
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase"
ORACLE_PATH = os.path.join(BASE, "tiny_stuff/tiniest_makeblobs_relu.pth")
RECON_PATH = os.path.join(BASE, "results/reconstructed_models/reconstructed_tiniest.pth")
X_TEST_PATH = os.path.join(BASE, "data/x_test_tiniest_makeblobs.npy")
Y_TEST_PATH = os.path.join(BASE, "data/y_test_tiniest_makeblobs.npy")

SEED = 42
N_SAMPLES = 12000
N_FEATURES = 8
N_CLASSES = 8
N_TEST = 2000
CLUSTER_STD = 1.6


class TiniestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 8)
        self.fc5 = nn.Linear(8, 8)
        self.double()

    def forward(self, x):
        x = x.view(-1, 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def load_model(path):
    model = TiniestModel()
    sd = torch.load(path, map_location='cpu')
    model.load_state_dict(sd)
    model.eval()
    return model


def regenerate_full_dataset():
    X, y = make_blobs(
        n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CLASSES,
        cluster_std=CLUSTER_STD, random_state=SEED, shuffle=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=N_TEST, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.clip(X_train, -3, 3) / 3.0
    X_test = np.clip(X_test, -3, 3) / 3.0
    return (
        X_train.astype(np.float64), y_train.astype(np.int64),
        X_test.astype(np.float64), y_test.astype(np.int64),
    )


def predict(model, X):
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float64)
        return model(xt).argmax(dim=1).numpy()


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_accuracy(y_true, y_pred, n_classes):
    out = []
    for c in range(n_classes):
        mask = y_true == c
        n = int(mask.sum())
        acc = float((y_pred[mask] == c).mean()) if n > 0 else 0.0
        out.append({'class': c, 'count': n, 'accuracy': acc})
    return out


def main():
    print("=" * 70)
    print("Reconstructed Model Evaluation on sklearn make_blobs")
    print("=" * 70)

    oracle = load_model(ORACLE_PATH)
    recon = load_model(RECON_PATH)
    print(f"Loaded oracle:        {ORACLE_PATH}")
    print(f"Loaded reconstructed: {RECON_PATH}")

    print("\nRegenerating make_blobs with original seed/config...")
    X_train, y_train, X_test, y_test = regenerate_full_dataset()
    X_full = np.concatenate([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)

    X_test_saved = np.load(X_TEST_PATH)
    y_test_saved = np.load(Y_TEST_PATH)
    match = np.allclose(X_test, X_test_saved) and np.array_equal(y_test, y_test_saved)
    print(f"Regenerated test matches saved .npy: {match}")

    sets = {
        'train': (X_train, y_train),
        'test':  (X_test, y_test),
        'full':  (X_full, y_full),
    }

    results = {}
    for name, (X, y) in sets.items():
        oracle_pred = predict(oracle, X)
        recon_pred = predict(recon, X)
        oracle_acc = float((oracle_pred == y).mean())
        recon_acc = float((recon_pred == y).mean())
        agreement = float((recon_pred == oracle_pred).mean())
        results[name] = {
            'n': len(y),
            'oracle_acc_vs_true': oracle_acc,
            'reconstructed_acc_vs_true': recon_acc,
            'agreement_vs_oracle': agreement,
            'gap_to_oracle': oracle_acc - recon_acc,
        }
        print(f"\n[{name.upper()}]  n={len(y)}")
        print(f"  oracle acc (vs true)          = {oracle_acc:.4f}")
        print(f"  reconstructed acc (vs true)   = {recon_acc:.4f}")
        print(f"  reconstructed agreement (vs oracle) = {agreement:.4f}")
        print(f"  gap (oracle - reconstructed)  = {oracle_acc - recon_acc:+.4f}")

    X, y = sets['test']
    recon_pred = predict(recon, X)
    oracle_pred = predict(oracle, X)
    cm_recon = confusion(y, recon_pred, N_CLASSES)
    cm_oracle = confusion(y, oracle_pred, N_CLASSES)
    pc_recon = per_class_accuracy(y, recon_pred, N_CLASSES)
    pc_oracle = per_class_accuracy(y, oracle_pred, N_CLASSES)

    print("\n[TEST] per-class accuracy:")
    print(f"{'class':>6} {'n':>6} {'oracle':>10} {'recon':>10} {'gap':>10}")
    for oc, rc in zip(pc_oracle, pc_recon):
        print(f"{oc['class']:>6d} {oc['count']:>6d} "
              f"{oc['accuracy']:>10.4f} {rc['accuracy']:>10.4f} "
              f"{oc['accuracy'] - rc['accuracy']:>+10.4f}")

    print("\n[TEST] confusion matrix (reconstructed, rows=true, cols=pred):")
    header = "     " + " ".join([f"{c:>4d}" for c in range(N_CLASSES)])
    print(header)
    for c in range(N_CLASSES):
        print(f"{c:>3d}: " + " ".join([f"{v:>4d}" for v in cm_recon[c]]))

    disagree_mask = recon_pred != oracle_pred
    n_disagree = int(disagree_mask.sum())
    disagree_true_acc = float((recon_pred[disagree_mask] == y[disagree_mask]).mean()) if n_disagree else 0.0
    oracle_on_disagree = float((oracle_pred[disagree_mask] == y[disagree_mask]).mean()) if n_disagree else 0.0
    print(f"\n[TEST] disagreement analysis:")
    print(f"  n samples where recon != oracle: {n_disagree}/{len(y)}")
    print(f"    on these, reconstructed acc vs true = {disagree_true_acc:.4f}")
    print(f"    on these, oracle acc vs true        = {oracle_on_disagree:.4f}")

    out_path = os.path.join(BASE, "results/reconstructed_models/makeblobs_eval.json")
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'seed': SEED, 'n_samples': N_SAMPLES, 'n_features': N_FEATURES,
                'n_classes': N_CLASSES, 'n_test': N_TEST, 'cluster_std': CLUSTER_STD,
            },
            'results': results,
            'per_class_test': {'oracle': pc_oracle, 'reconstructed': pc_recon},
            'confusion_matrices_test': {
                'oracle': cm_oracle.tolist(),
                'reconstructed': cm_recon.tolist(),
            },
            'disagreement_test': {
                'n_disagree': n_disagree,
                'recon_acc_on_disagreement': disagree_true_acc,
                'oracle_acc_on_disagreement': oracle_on_disagree,
            },
        }, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == '__main__':
    main()
