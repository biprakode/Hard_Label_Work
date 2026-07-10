"""
Tinier model with plain ReLU activations.
Architecture: 32 -> 16 -> 16 -> 16 -> 8 -> 4 (input, 4 hidden, output).
Identical training settings as create_tinier_makeblobs_leakyrelu.py so X_test
is byte-identical between the two variants.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)

N_SAMPLES = 15000
N_FEATURES = 32
N_CLASSES = 4
N_TEST = 10000
LAYER_SIZES = [16, 16, 16, 8, 4]

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "tiny_stuff")
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print(f"TINIER ReLU: {N_FEATURES} -> {' -> '.join(map(str,LAYER_SIZES))}")
print("=" * 60)

X, y = make_blobs(
    n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CLASSES,
    cluster_std=2.0, random_state=SEED, shuffle=True,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=N_TEST, random_state=SEED, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = (np.clip(X_train, -3, 3) / 3.0).astype(np.float64)
X_test = (np.clip(X_test, -3, 3) / 3.0).astype(np.float64)
print(f"Train {X_train.shape}  Test {X_test.shape}")


class TinierReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, LAYER_SIZES[0])
        self.fc2 = nn.Linear(LAYER_SIZES[0], LAYER_SIZES[1])
        self.fc3 = nn.Linear(LAYER_SIZES[1], LAYER_SIZES[2])
        self.fc4 = nn.Linear(LAYER_SIZES[2], LAYER_SIZES[3])
        self.fc5 = nn.Linear(LAYER_SIZES[3], LAYER_SIZES[4])
        self.double()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


device = torch.device("cpu")
net = TinierReLU().to(device)
Xt = torch.tensor(X_train, dtype=torch.float64, device=device)
yt = torch.tensor(y_train, dtype=torch.long, device=device)
Xe = torch.tensor(X_test, dtype=torch.float64, device=device)
ye = torch.tensor(y_test, dtype=torch.long, device=device)

opt = optim.Adam(net.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
BS = 64
N_EPOCHS = 100
for ep in range(N_EPOCHS):
    net.train()
    idx = torch.randperm(len(Xt))
    total = 0
    correct = 0
    tl = 0.0
    for i in range(0, len(Xt), BS):
        b = idx[i:i + BS]
        bx, by = Xt[b], yt[b]
        opt.zero_grad()
        out = net(bx)
        loss = crit(out, by)
        loss.backward()
        opt.step()
        tl += loss.item()
        total += by.size(0)
        correct += (out.argmax(1) == by).sum().item()
    if (ep + 1) % 10 == 0:
        net.eval()
        with torch.no_grad():
            tacc = (net(Xe).argmax(1) == ye).float().mean().item()
        print(f"Epoch {ep+1:3d}  train_acc {correct/total:.4f}  test_acc {tacc:.4f}  loss {tl:.4f}")

net.eval()
with torch.no_grad():
    final_acc = (net(Xe).argmax(1) == ye).float().mean().item()
print(f"Final test_acc {final_acc:.4f}")

pth_path = os.path.join(OUT_DIR, "tinier_makeblobs_relu.pth")
torch.save(net.state_dict(), pth_path)
print("Saved", pth_path)

# Keras model: plain ReLU baked into Dense layers
keras_model = keras.Sequential([
    keras.layers.Dense(LAYER_SIZES[0], activation='relu', input_shape=(N_FEATURES,), dtype='float64', name='dense_1'),
    keras.layers.Dense(LAYER_SIZES[1], activation='relu', dtype='float64', name='dense_2'),
    keras.layers.Dense(LAYER_SIZES[2], activation='relu', dtype='float64', name='dense_3'),
    keras.layers.Dense(LAYER_SIZES[3], activation='relu', dtype='float64', name='dense_4'),
    keras.layers.Dense(LAYER_SIZES[4], activation=None, dtype='float64', name='dense_5'),
])
net.cpu()
pw = {
    'fc1': (net.fc1.weight.detach().numpy().T, net.fc1.bias.detach().numpy()),
    'fc2': (net.fc2.weight.detach().numpy().T, net.fc2.bias.detach().numpy()),
    'fc3': (net.fc3.weight.detach().numpy().T, net.fc3.bias.detach().numpy()),
    'fc4': (net.fc4.weight.detach().numpy().T, net.fc4.bias.detach().numpy()),
    'fc5': (net.fc5.weight.detach().numpy().T, net.fc5.bias.detach().numpy()),
}
for fc_key, idx in zip(['fc1', 'fc2', 'fc3', 'fc4', 'fc5'], [0, 1, 2, 3, 4]):
    keras_model.layers[idx].set_weights(list(pw[fc_key]))

diff = np.abs(net(torch.tensor(X_test[:10], dtype=torch.float64)).detach().numpy() - keras_model.predict(X_test[:10], verbose=0)).max()
print(f"PT<>Keras max diff {diff:.2e}")
keras_path = os.path.join(OUT_DIR, "tinier_makeblobs_relu.keras")
keras_model.save(keras_path)
print("Saved", keras_path)

# Test data — same make_blobs call as the leaky version, so X_test stays byte-identical
np.save(os.path.join(DATA_DIR, "x_test_tinier_makeblobs.npy"), X_test)
np.save(os.path.join(DATA_DIR, "y_test_tinier_makeblobs.npy"), y_test)
print("Saved test data")

# X_test2: seed=99, same centers, same scaler
X_full, y_full, centers = make_blobs(
    n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CLASSES,
    cluster_std=2.0, random_state=SEED, shuffle=True, return_centers=True,
)
N_TEST2 = 5000
X2, y2 = make_blobs(
    n_samples=N_TEST2, centers=centers, cluster_std=2.0, random_state=99, shuffle=True,
)
X2 = (np.clip(scaler.transform(X2), -3, 3) / 3.0).astype(np.float64)
y2 = y2.astype(np.int64)
np.save(os.path.join(DATA_DIR, "x_test2_tinier_makeblobs.npy"), X2)
np.save(os.path.join(DATA_DIR, "y_test2_tinier_makeblobs.npy"), y2)
print(f"Saved X_test2 (seed=99, n={N_TEST2})")

with torch.no_grad():
    acc2 = (net(torch.tensor(X2, dtype=torch.float64)).argmax(1).numpy() == y2).mean()
print(f"Oracle acc on X_test2: {acc2:.4f}")
print(f"\nDone. Tinier ReLU test_acc={final_acc:.4f}, X_test2_acc={acc2:.4f}")
