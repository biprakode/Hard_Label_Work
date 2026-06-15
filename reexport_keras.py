"""
Rebuild the Keras .keras victims from the trained .pth files, CORRECTLY.

Findings (Keras 3.10 / TF 2.18):
  * model.predict() on float64 input is buggy here (gives wrong outputs) — but a
    DIRECT call model(tf.constant(x)) is exact. sign_recovery uses direct calls
    (intermediateLayerModel(x)), so we build + verify with direct calls only.
  * The earlier in-trainer export produced a model that was wrong even on direct
    call. The robust pattern (verified to 1e-13) is: build with input_shape on
    the first layer, force-build the layer variables with one dummy call, THEN
    set_weights, then verify with a direct call.
"""
import os
import numpy as np
import torch
import tensorflow as tf
import keras
keras.backend.set_floatx("float64")

BASE = "/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/Hard_Label_Work"
OUT = os.path.join(BASE, "tiny_stuff")
LAYER_SIZES = [3072, 256, 256, 256, 64, 10]


def torch_forward(sd, x, alpha):
    w = [sd[f"fc{i}.weight"].numpy() for i in range(1, 6)]
    b = [sd[f"fc{i}.bias"].numpy() for i in range(1, 6)]
    h = x
    for i in range(4):
        z = h @ w[i].T + b[i]
        h = np.where(z >= 0, z, alpha * z)
    return h @ w[4].T + b[4]


def build_keras(sd, alpha):
    def act(name):
        return keras.layers.LeakyReLU(negative_slope=alpha, dtype="float64", name=name) if alpha > 0 \
            else keras.layers.ReLU(dtype="float64", name=name)
    m = keras.Sequential()
    m.add(keras.layers.Dense(LAYER_SIZES[1], activation=None, dtype="float64",
                             input_shape=(LAYER_SIZES[0],), name="dense_1"))
    m.add(act("act_1"))
    for li in range(2, 5):
        m.add(keras.layers.Dense(LAYER_SIZES[li], activation=None, dtype="float64", name=f"dense_{li}"))
        m.add(act(f"act_{li}"))
    m.add(keras.layers.Dense(LAYER_SIZES[5], activation=None, dtype="float64", name="dense_5"))

    # force-build all layer variables before set_weights
    _ = m(tf.zeros((1, LAYER_SIZES[0]), dtype=tf.float64))

    dense = [l for l in m.layers if isinstance(l, keras.layers.Dense)]
    for i, dl in enumerate(dense, 1):
        W = sd[f"fc{i}.weight"].numpy().T.astype(np.float64)   # (in, out)
        bvec = sd[f"fc{i}.bias"].numpy().astype(np.float64)
        dl.set_weights([W, bvec])
    return m


def direct_call(km, x):
    return km(tf.constant(x, dtype=tf.float64)).numpy()


def main():
    rng = np.random.RandomState(0)
    x = (rng.rand(32, LAYER_SIZES[0]) * 2 - 1).astype(np.float64)
    for tag, alpha in (("relu", 0.0), ("leakyrelu", 0.01)):
        sd = torch.load(os.path.join(OUT, f"TinyModel_{tag}.pth"), map_location="cpu")
        km = build_keras(sd, alpha)
        ref = torch_forward(sd, x, alpha)
        diff = np.abs(ref - direct_call(km, x)).max()
        print(f"[{tag}] build-time (direct call) torch vs keras: {diff:.2e}")
        assert diff < 1e-9, f"{tag}: build mismatch {diff:.3e}"

        kpath = os.path.join(OUT, f"TinyModel_{tag}.keras")
        km.save(kpath)
        km2 = keras.models.load_model(kpath, compile=False)
        diff2 = np.abs(ref - direct_call(km2, x)).max()
        print(f"[{tag}] reloaded (direct call) torch vs keras: {diff2:.2e}  -> {kpath}")
        assert diff2 < 1e-9, f"{tag}: reloaded mismatch {diff2:.3e}"
    print("OK: both .keras victims match their .pth within 1e-9 (direct call)")


if __name__ == "__main__":
    main()
