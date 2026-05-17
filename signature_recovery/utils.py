import collections
import time
import os
import pickle
import torch
import torch.nn as nn
import random
import numpy as np
import sys
import scipy.linalg
import functools
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = True
USE_GRADIENT = True

LAYERS = 5
TINIEST = True  # Use tiniest 8-8-8-8-8-8 model (input 8, 4 hidden of 8, output 8)
TINY = True
TINIER = False  # Use tinier model with non-uniform hidden widths (32->16->16->16->8->4)
MAKEBLOBS = True  # Use make_blobs synthetic dataset instead of CIFAR-10

# Activation toggle.
#   LEAKY_ALPHA = 0.0  -> plain ReLU (DEFAULT, identical to the original pipeline)
#   LEAKY_ALPHA > 0    -> Leaky ReLU(alpha), e.g. 0.01
# Single source of truth read by all forward passes and by model-path resolution
# across signature_recovery, sign_recovery, and analysis/test_extraction4.
LEAKY_ALPHA = 0.01


def act(x):
    """Activation function controlled by LEAKY_ALPHA.
    LEAKY_ALPHA == 0  -> torch.relu (byte-identical to nn.ReLU()).
    LEAKY_ALPHA  > 0  -> F.leaky_relu(x, alpha)."""
    if LEAKY_ALPHA > 0:
        return torch.nn.functional.leaky_relu(x, negative_slope=LEAKY_ALPHA)
    return torch.relu(x)


def act_np(x):
    """NumPy mirror of act() for sign_recovery / whitebox forward sims."""
    if LEAKY_ALPHA > 0:
        return np.where(x >= 0, x, LEAKY_ALPHA * x)
    return np.maximum(0.0, x)


def cell_slope_mask(x):
    """Per-cell linear slope of the activation, evaluated at x.
    Returns 1 where x>=0 (ON cell) and LEAKY_ALPHA where x<0 (OFF cell).
    When LEAKY_ALPHA == 0 this reduces to (x>=0).double(), the original ReLU mask.
    Used for prefix linearisation in recover_weights.relu_around.
    """
    on = (x >= 0).to(torch.float64)
    return on + LEAKY_ALPHA * (1.0 - on)

# LAYER_SIZES: full list [input_dim, hidden1, hidden2, ..., output_dim]
# This is the single source of truth for all dimension-dependent code.
if TINIEST:
    LAYER_SIZES = [8, 8, 8, 8, 8, 8]
elif TINIER:
    LAYER_SIZES = [32, 16, 16, 16, 8, 4]
elif TINY:
    LAYER_SIZES = [64, 64, 64, 64, 64, 10]
else:
    LAYER_SIZES = [32*32*3, 256, 256, 256, 64, 10]

IDIM = LAYER_SIZES[0]
# DIM = max hidden layer width (for backward compat with code that uses DIM for padding)
DIM = max(LAYER_SIZES[1:-1])
SHRINK = LAYER_SIZES[-2]

# Compute layer boundaries for flattened neuron indexing:
# layer_boundaries[i] = cumulative sum of hidden dims up to layer i
# e.g. for [32,16,16,16,8,4]: boundaries = [0, 16, 32, 48, 56]
LAYER_BOUNDARIES = [0]
for dim in LAYER_SIZES[1:-1]:
    LAYER_BOUNDARIES.append(LAYER_BOUNDARIES[-1] + dim)
TOTAL_HIDDEN_NEURONS = LAYER_BOUNDARIES[-1]

SEED = 1 if len(sys.argv) < 3 else int(sys.argv[1])

# Load test data based on dataset type
if TINIEST and MAKEBLOBS:
    x_test = np.load("/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/data/x_test_tiniest_makeblobs.npy")
    x_test = np.array(x_test, dtype=np.float64)
elif TINIER and MAKEBLOBS:
    x_test = np.load("/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/data/x_test_tinier_makeblobs.npy")
    x_test = np.array(x_test, dtype=np.float64)
elif MAKEBLOBS:
    x_test = np.load("/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/data/x_test_makeblobs.npy")
    x_test = np.array(x_test, dtype=np.float64)
else:
    # Load CIFAR-10 data with preprocessing
    x_test = np.load("/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/data/x_test.npy")
    if TINY:
        x_test = x_test.mean(1)[:, ::4, ::4]
    x_test = x_test.reshape((-1, IDIM))
    x_test = (x_test*2-1)
    x_test = np.array(x_test, dtype=np.float64)

random.seed(SEED)
np.random.seed(SEED)


class MathIsHard(Exception):
    pass

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.fc1 = nn.Linear(LAYER_SIZES[0], LAYER_SIZES[1])
        self.fc2 = nn.Linear(LAYER_SIZES[1], LAYER_SIZES[2])
        self.fc3 = nn.Linear(LAYER_SIZES[2], LAYER_SIZES[3])
        self.fc4 = nn.Linear(LAYER_SIZES[3], LAYER_SIZES[4])
        self.fc5 = nn.Linear(LAYER_SIZES[4], LAYER_SIZES[5])
        self.relu = nn.ReLU()
        self.double()

    @torch.no_grad
    def forward(self, x):
        x = x.view(-1, IDIM)
        x = act(self.fc1(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        x = act(self.fc4(x))
        x = self.fc5(x)
        return x

    def forward_grad(self, x):
        x = x.view(-1, IDIM)
        x = act(self.fc1(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        x = act(self.fc4(x))
        x = self.fc5(x)
        return x

    @torch.no_grad
    def cheat(self, x):
        # Returns *pre-activation* outputs of each hidden layer, padded to max width with 1s.
        # cheat(x)>0 detects cell membership (sign of preact) for both ReLU and Leaky ReLU.
        o = []
        max_dim = DIM  # pad all layers to max hidden width for uniform stacking

        def collect(x):
            if x.size(-1) < max_dim:
                padding = max_dim - x.size(-1)
                xx = torch.nn.functional.pad(x, (0, padding))
                xx[:, -padding:] = 1
                o.append(xx)
            else:
                o.append(x)
            return act(x)

        x = x.view(-1, IDIM)
        x = collect(self.fc1(x))
        x = collect(self.fc2(x))
        x = collect(self.fc3(x))
        x = collect(self.fc4(x))
        return torch.stack(o)

def load_converted_model(path, model, device):
    state_dict = torch.load(path, map_location=device)

    rename_map = {
        "hidden_layer1": "fc1",
        "hidden_layer2": "fc2",
        "hidden_layer3": "fc3",
        "hidden_layer4": "fc4",
        "output": "fc5",
    }

    new_state_dict = {}
    for k, v in state_dict.items():
        for old, new in rename_map.items():
            if k.startswith(old):
                k = k.replace(old, new)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model

BASE_DIR = os.path.dirname(os.path.abspath("/run/media/biprarshi/COMMON/files/AI/hard-label-dnn-extraction/enhanced_codebase/signature_recovery/"))  # directory of utils.py

# Select model based on dataset type and activation toggle.
# (Pre-existing bug fix: paths previously read /enhanced_codebase/... which is not
#  rooted in the filesystem; switched to BASE_DIR-relative.)
_act_suffix = "leakyrelu" if LEAKY_ALPHA > 0 else "relu"
if TINIEST and MAKEBLOBS:
    MODEL_PATH = os.path.join(BASE_DIR, f"tiny_stuff/tiniest_makeblobs_{_act_suffix}.pth")
elif TINIER and MAKEBLOBS:
    MODEL_PATH = os.path.join(BASE_DIR, f"tiny_stuff/tinier_makeblobs_{_act_suffix}.pth")
elif MAKEBLOBS:
    MODEL_PATH = os.path.join(BASE_DIR, f"tiny_stuff/makeblobs_{_act_suffix}.pth")
else:
    MODEL_PATH = os.path.join(BASE_DIR, f"tiny_stuff/TinyModel_{_act_suffix}.pth")

cheat_net_cuda = CIFAR10Net().to(device).double()
cheat_net_cuda = load_converted_model(MODEL_PATH, cheat_net_cuda, device)
cheat_net_cuda.double().cpu()

cheat_net_cpu = CIFAR10Net()
cheat_net_cpu = load_converted_model(MODEL_PATH, cheat_net_cpu, device)
cheat_net_cpu.double()

cheat_solution = [x.cpu().detach().numpy() for x in cheat_net_cpu.parameters()][::2]

def model(x):
    assert len(x.shape) == 1
    return (cheat_net_cpu(torch.tensor(x).double()).numpy().argmax(1)).astype(np.int32)[0]

def bmodel(x):
    return (cheat_net_cuda(x).argmax(1)).to(torch.int32)

def cheat(x):
    if not DEBUG: raise
    return cheat_net_cpu.cheat(torch.tensor(x).double()).numpy()

def cheat_cuda(x):
    if not DEBUG: raise
    return cheat_net_cuda.cheat(x)

def gap(x):
    if not DEBUG: raise
    out = cheat_net_cpu(torch.tensor(x).double()).numpy()
    max_idx = np.argmax(out, 1)
    top = out[np.arange(len(out)), max_idx]
    out[np.arange(len(out)), max_idx] = -100
    return top - np.max(out, 1)

def gapt(x, grad=False):
    if not DEBUG: raise
    if grad:
        out = cheat_net_cuda.forward_grad(x)
    else:
        out = cheat_net_cuda(x)

    max_idx = torch.argmax(out, 1)
    top = out[torch.arange(len(out)), max_idx]
    out[torch.arange(len(out)), max_idx] = -100
    return top - torch.max(out, 1).values

def cheat_num_flips(a,b):
    if not DEBUG: raise
    return np.sum((cheat(a)>0) != (cheat(b)>0))

def cheat_neuron_diff(a,b):
    if not DEBUG: raise
    return np.where((cheat(a)>0).flatten() != (cheat(b)>0).flatten())[0]

def cheat_neuron_diff_cuda(a,b):
    if not DEBUG: raise
    ab = torch.tensor(np.stack([a,b])).double().cpu()
    out = cheat_cuda(ab)>0
    a = out[:,0,:]
    b = out[:,1,:]
    return torch.where(a.flatten() != b.flatten())[0].cpu().numpy()

def cheat_neuron_diff_cuda_2(a,b):
    if not DEBUG: raise
    ab = torch.tensor(np.stack([a,b])).double().cpu()
    out = cheat_cuda(ab)>0
    a = out[:,0,:]
    b = out[:,1,:]
    return torch.where(a.flatten() != b.flatten())[0].cpu().numpy(), np.array(a.flatten().cpu().numpy(), dtype=np.uint8)


def neuron_flat_to_layer(flat_idx):
    """Convert a flat neuron index to (layer_idx, local_neuron_idx).
    Uses LAYER_BOUNDARIES computed from LAYER_SIZES.
    """
    for i in range(len(LAYER_BOUNDARIES) - 1):
        if LAYER_BOUNDARIES[i] <= flat_idx < LAYER_BOUNDARIES[i + 1]:
            return i, flat_idx - LAYER_BOUNDARIES[i]
    raise ValueError(f"Invalid flat neuron index: {flat_idx} (total hidden neurons: {TOTAL_HIDDEN_NEURONS})")


# Find the decision boundary, given two inputs:
#    zero must have model(zero) = 0
#    one  must have model(one)  = 1
# returns the midpoint that's almost 0/1 at the same time
def find_decision_boundary(zero=None, one=None, tensor=False):
    if zero is None and one is None:
        points = {}
        while len(points) < 2:
            if not TINY and not TINIER and not TINIEST:
                maybe = random.sample(range(len(x_test)), 10)
                maybe = x_test[maybe]
            else:
                maybe = np.random.normal(size=(10, IDIM))
            maybe = torch.tensor(maybe).cpu().double()
            outs = bmodel(maybe)
            for out, point in zip(outs, maybe):
                points[out.item()] = point
        zero, one = list(points.values())[:2]

    model_zero = bmodel(zero)
    last = 1e9
    while torch.sum(torch.abs(zero - one)) > 1e-16 and torch.sum(torch.abs(zero - one)) < last:
        last = torch.sum(torch.abs(zero - one))
        mid = (zero+one)/2
        if bmodel(mid) == model_zero:
            zero = mid
        else:
            one = mid


    if tensor:
        return zero
    return zero.cpu().numpy()

def find_decision_boundary_batched(zero, one):
    zero = torch.tensor(zero).double().cpu()
    one = torch.tensor(one).double().cpu()
    last = torch.tensor(1e9).cpu()

    orig_label = bmodel(zero)[0]

    while True:
        s = torch.sum(torch.abs(zero - one), dim=1)
        if not torch.any((s > 1e-14) | (s < last)).item():
            break

        last = s
        mid = (zero + one) / 2

        idx = bmodel(mid)

        zero_mask = (idx == orig_label)
        one_mask = (idx != orig_label)

        zero[zero_mask] = mid[zero_mask]
        one[one_mask] = mid[one_mask]

    return zero

# Compute the gradient direction of the decision boundary
# More correctly, this function returns a parallel direction
# to the hyperplane, so we can walk along it.
def get_gradient_dir(x, cache={}, debug=False, step_size=1e-7, dimensions=None):
    if len(cache) > 1e6:
        cache.clear()
    if tuple(x) in cache:
        return cache[tuple(x)]

    original = model(x)

    if dimensions is None:
        dimensions = range(IDIM)

    ratios = []
    for i in dimensions:
        if debug:
            print('iter', i)
        xp = np.array(x)


        if debug:
            print('start', gap(xp))
            sig = np.sign(cheat(x).flatten())

        xp[0] += step_size
        if model(xp) != original:
            xp[0] -= step_size*2

        if debug:
            print('   ', gap(xp))
            print('sigchange', np.sum(sig!= np.sign(cheat(xp).flatten())))

        for step in 10**np.arange(-7, 0, .33):
            xp2 = np.array(xp)
            xp2[i] += step
            if model(xp2) == original:
                xp2[i] -= 2*step
            if model(xp2) != original:
                break
        else:
            assert False

        if debug:
            print('   ', gap(xp2))

        boundary = find_decision_boundary(xp, xp2)

        ratio = (xp[i]-boundary[i])/(step_size)
        ratios.append(ratio)

    # So far we have the gradient direction, now let's make is
    # so that we can go parallel
    ratios = np.array(ratios)

    cache[tuple(x)] = ratios

    return ratios

def vectorized_right_boundary_search(left, orig_label, dimensions):
    device = 'cuda'
    batch_size = IDIM

    # Initialize the batch with copies of the left boundary
    batch = left.repeat(batch_size, 1)

    # Create a mask to keep track of dimensions that haven't found the boundary yet
    active_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    active_mask[dimensions] = 1

    for step in 10**np.arange(-7, 0, .33):
        # Try positive step
        pos_batch = batch.clone()
        pos_batch[active_mask] += torch.eye(IDIM, device=device)[active_mask] * step

        pos_results = bmodel(pos_batch)

        # Update batch and mask for positive steps that crossed the boundary
        pos_crossed = (pos_results != orig_label) & active_mask
        batch[pos_crossed] = pos_batch[pos_crossed]
        active_mask[pos_crossed] = False

        if not active_mask.any():
            break

        # Try negative step for remaining active dimensions
        neg_batch = batch.clone()
        neg_batch[active_mask] -= torch.eye(IDIM, device=device)[active_mask] * step

        neg_results = bmodel(neg_batch)

        # Update batch and mask for negative steps that crossed the boundary
        neg_crossed = (neg_results != orig_label) & active_mask

        batch[neg_crossed] = neg_batch[neg_crossed]
        active_mask[neg_crossed] = False

        if not active_mask.any():
            break

    if active_mask.any():
        raise MathIsHard("Boundary not found for all dimensions")

    return left.repeat(IDIM, 1)[dimensions, :], batch[dimensions, :]

def get_gradient_dir_fast(x, cache={}, debug=False, step_size=1e-7, dimensions=None):
    if len(cache) > 1e6:
        cache.clear()
    if tuple(x) in cache:
        return cache[tuple(x)]

    if dimensions is None:
        dimensions = range(IDIM)

    # 1. init
    # find the left and right sides
    leftright = []

    left = np.array(x)
    left = torch.tensor(left, dtype=torch.float64, device='cuda')
    original = bmodel(left)

    left[0] += step_size
    if bmodel(left) != original:
        left[0] -= step_size*2

    xp, xp2 = vectorized_right_boundary_search(left, original, dimensions)

    ratios = []

    boundary = find_decision_boundary_batched(xp, xp2)

    ratios = ((xp-boundary)/(step_size))[torch.arange(len(dimensions)), dimensions].cpu().numpy()

    # So far we have the gradient direction, now let's make is
    # so that we can go parallel
    ratios = np.array(ratios)

    cache[tuple(x)] = ratios

    return ratios

def get_normal(x, step_size=1e-6):
    if USE_GRADIENT:
        x = torch.tensor(x, requires_grad=True)
        out = gapt(x.cpu(), grad=True)
        out[0].backward()
        real = np.random.normal(0, 1) * x.grad.cpu().numpy()
        real = norm(real)
        return real
    else:
        try:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size)
        except MathIsHard:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size/10)
        fnormal = norm(fnormal)

        return fnormal

def get_normal_t(x, step_size=1e-6):
    if USE_GRADIENT:
        x = torch.tensor(x, requires_grad=True)
        out = gapt(x.cpu(), grad=True)
        out[0].backward()
        real = torch.tensor(np.random.normal(0, 1)) * x.grad
        real = normt(real)
        return real
    else:
        try:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size)
        except MathIsHard:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size/10)
        fnormal = norm(fnormal)

        return fnormal

def norm(x):
    return x / np.sum(x**2)**.5

def normt(x):
    return x / torch.sum(x**2)**.5
