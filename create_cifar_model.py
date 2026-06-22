"""
Train the FLAGSHIP CIFAR-10 victim:  3072 -> 256 -> 256 -> 256 -> 64 -> 10.

This is the same architecture the vanilla codebase extracts
(`cifar10_3x256_64_10_float64.keras`). We train BOTH activation variants
(ReLU and LeakyReLU(0.01)) so the hard-label extraction can be run on each.

Why a plain Linear+activation MLP (no BatchNorm / dropout): the extraction
attack assumes the victim is a pure piecewise-linear function f(x) = fc5 . act .
fc4 . act ... fc1, so the architecture must contain only affine layers and the
activation. An MLP on raw CIFAR pixels tops out around ~50% test accuracy; that
is the expected flagship ceiling and is irrelevant to whether signature/sign
recovery succeeds (the attack works on the function, not the accuracy).

Preprocessing convention (kept consistent across the whole pipeline):
  * data/x_test.npy is stored RAW uint8, shape (N, 3072), channel order = the
    CIFAR python-pickle order (R,G,B planes).
  * The model input is  x/255 * 2 - 1  -> [-1, 1].
    - signature_recovery/utils.py  (else branch)      applies /255*2-1
    - analysis/extraction_pipeline/data_loading.py    applies /255*2-1
  Training here uses the identical transform so the victim sees [-1,1] inputs.

Outputs (all under Hard_Label_Work/):
  tiny_stuff/TinyModel_relu.pth        tiny_stuff/TinyModel_relu.keras
  tiny_stuff/TinyModel_leakyrelu.pth   tiny_stuff/TinyModel_leakyrelu.keras
  tiny_stuff/TinyModel_<act>_alpha.txt
  data/x_test.npy   data/y_test.npy            (CIFAR test set, raw uint8)
  data/x_test2_cifar.npy  data/y_test2_cifar.npy (held-out CIFAR train slice
                                                  40000-49999, raw uint8)
  data/x_test3_cifar.npy  data/y_test3_cifar.npy (disjoint CIFAR train slice
                                                  10000-19999, raw uint8 — used
                                                  as held-out eval + early-stop
                                                  watchdog by the enhanced
                                                  Phase 3 pipeline)
"""
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(os.cpu_count() or 8)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/Hard_Label_Work"
OUT_DIR = os.path.join(BASE, "tiny_stuff")
DATA_DIR = os.path.join(BASE, "data")
CIFAR_DIR = os.path.expanduser("~/.keras/datasets/cifar-10-batches-py-target/cifar-10-batches-py")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

LAYER_SIZES = [3072, 256, 256, 256, 64, 10]
N_EPOCHS = 60
BS = 256


# --------------------------------------------------------------- data loading --
def _load_batch(fname):
    with open(os.path.join(CIFAR_DIR, fname), "rb") as fh:
        e = pickle.load(fh, encoding="bytes")
    return e[b"data"], np.array(e[b"labels"], dtype=np.int64)


def load_cifar():
    xs, ys = [], []
    for i in range(1, 6):
        x, y = _load_batch(f"data_batch_{i}")
        xs.append(x); ys.append(y)
    x_train = np.concatenate(xs).astype(np.uint8)        # (50000, 3072)
    y_train = np.concatenate(ys)
    x_test, y_test = _load_batch("test_batch")           # (10000, 3072)
    return x_train, y_train, x_test.astype(np.uint8), y_test


def pre(x_uint8):
    """raw uint8 -> [-1, 1] float64 (the victim's input space)."""
    return x_uint8.astype(np.float64) / 255.0 * 2.0 - 1.0


# ------------------------------------------------------------------- the model --
class CIFAR10Net(nn.Module):
    """Mirrors signature_recovery.utils.CIFAR10Net / extraction FullModel."""
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(LAYER_SIZES[0], LAYER_SIZES[1])
        self.fc2 = nn.Linear(LAYER_SIZES[1], LAYER_SIZES[2])
        self.fc3 = nn.Linear(LAYER_SIZES[2], LAYER_SIZES[3])
        self.fc4 = nn.Linear(LAYER_SIZES[3], LAYER_SIZES[4])
        self.fc5 = nn.Linear(LAYER_SIZES[4], LAYER_SIZES[5])

    def act(self, x):
        return F.leaky_relu(x, self.alpha) if self.alpha > 0 else F.relu(x)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        return self.fc5(x)


def train_one(alpha, X_tr, y_tr, X_te, y_te):
    tag = "leakyrelu" if alpha > 0 else "relu"
    print(f"\n{'='*64}\nTraining CIFAR flagship [{ '->'.join(map(str,LAYER_SIZES)) }]  act={tag}\n{'='*64}")
    torch.manual_seed(SEED)
    net = CIFAR10Net(alpha)             # float32 for training speed
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xe = torch.tensor(X_te, dtype=torch.float32)
    ye = torch.tensor(y_te, dtype=torch.long)

    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(N_EPOCHS):
        net.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            b = idx[i:i + BS]
            opt.zero_grad()
            loss = crit(net(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            net.eval()
            with torch.no_grad():
                tr_acc = (net(Xt).argmax(1) == yt).float().mean().item()
                te_acc = (net(Xe).argmax(1) == ye).float().mean().item()
            print(f"  epoch {ep+1:3d}  train_acc {tr_acc:.4f}  test_acc {te_acc:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)")

    net.eval()
    with torch.no_grad():
        final_te = (net(Xe).argmax(1) == ye).float().mean().item()
    print(f"  FINAL test_acc {final_te:.4f}")

    # cast to float64 for the (float64) extraction pipeline
    net = net.double()

    pth = os.path.join(OUT_DIR, f"TinyModel_{tag}.pth")
    torch.save(net.state_dict(), pth)         # keys fc1..fc5 — load_converted_model leaves them as-is
    print("  saved", pth)

    _save_keras(net, alpha, tag, X_te)
    with open(os.path.join(OUT_DIR, f"TinyModel_{tag}_alpha.txt"), "w") as f:
        f.write(f"{alpha}\n")
    return net, final_te


def _save_keras(net, alpha, tag, X_te):
    """Mirror the torch weights into a Keras model (Dense + explicit activation
    layers) for the sign-recovery phase, which loads a .keras oracle."""
    import tensorflow as tf
    from tensorflow import keras
    net = net.cpu().double()
    act_layer = (lambda n: keras.layers.LeakyReLU(alpha=alpha, dtype="float64", name=n)) if alpha > 0 \
        else (lambda n: keras.layers.ReLU(dtype="float64", name=n))
    km = keras.Sequential([
        keras.layers.Input(shape=(LAYER_SIZES[0],), dtype="float64"),
        keras.layers.Dense(LAYER_SIZES[1], dtype="float64", name="dense_1"), act_layer("act_1"),
        keras.layers.Dense(LAYER_SIZES[2], dtype="float64", name="dense_2"), act_layer("act_2"),
        keras.layers.Dense(LAYER_SIZES[3], dtype="float64", name="dense_3"), act_layer("act_3"),
        keras.layers.Dense(LAYER_SIZES[4], dtype="float64", name="dense_4"), act_layer("act_4"),
        keras.layers.Dense(LAYER_SIZES[5], activation=None, dtype="float64", name="dense_5"),
    ])
    fcs = [net.fc1, net.fc2, net.fc3, net.fc4, net.fc5]
    dense_layers = [l for l in km.layers if isinstance(l, keras.layers.Dense)]
    for fc, dl in zip(fcs, dense_layers):
        dl.set_weights([fc.weight.detach().numpy().T, fc.bias.detach().numpy()])
    # equivalence check on [-1,1] inputs
    xin = pre(X_te[:16])
    with torch.no_grad():
        pt = net(torch.tensor(xin, dtype=torch.float64)).numpy()
    kk = km.predict(xin, verbose=0)
    print(f"  PT<>Keras max diff: {np.abs(pt - kk).max():.2e}")
    kpath = os.path.join(OUT_DIR, f"TinyModel_{tag}.keras")
    km.save(kpath)
    print("  saved", kpath)


def main():
    x_train, y_train, x_test, y_test = load_cifar()
    print(f"CIFAR train {x_train.shape}  test {x_test.shape}")

    # Hold out the last 10k of TRAIN as a fresh eval set (X_test2): never used by
    # Phase-3 (which trains on the CIFAR test set), so it measures generalization.
    x_test2 = x_train[-10000:].copy()
    y_test2 = y_train[-10000:].copy()
    # X_test3 = a second disjoint train slice (10000-19999). Used by the enhanced
    # Phase 3 pipeline as the held-out eval + early-stop watchdog when X_test2
    # is promoted into the query/training tier alongside X_test.
    x_test3 = x_train[10000:20000].copy()
    y_test3 = y_train[10000:20000].copy()
    # Build training set excluding BOTH held-out slices so X_test3 is honest.
    keep_mask = np.ones(len(x_train), dtype=bool)
    keep_mask[10000:20000] = False
    keep_mask[40000:50000] = False
    x_tr = x_train[keep_mask]
    y_tr = y_train[keep_mask]
    assert len(x_tr) == 30000, f"expected 30000 training samples after dropping both holdouts, got {len(x_tr)}"

    X_tr = pre(x_tr)        # [-1,1] for training
    X_te = pre(x_test)

    for alpha in (0.0, 0.01):
        train_one(alpha, X_tr, y_tr, X_te, y_test)

    # ---- data files (raw uint8; pipeline applies /255*2-1) ----
    np.save(os.path.join(DATA_DIR, "x_test.npy"), x_test.astype(np.uint8))
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test.astype(np.int64))
    np.save(os.path.join(DATA_DIR, "x_test2_cifar.npy"), x_test2.astype(np.uint8))
    np.save(os.path.join(DATA_DIR, "y_test2_cifar.npy"), y_test2.astype(np.int64))
    np.save(os.path.join(DATA_DIR, "x_test3_cifar.npy"), x_test3.astype(np.uint8))
    np.save(os.path.join(DATA_DIR, "y_test3_cifar.npy"), y_test3.astype(np.int64))
    print("\nSaved data: x_test.npy, y_test.npy, "
          "x_test2_cifar.npy, y_test2_cifar.npy, "
          "x_test3_cifar.npy, y_test3_cifar.npy")
    print("Done.")


if __name__ == "__main__":
    main()
