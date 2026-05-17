"""
PyTorch model classes for every supported victim architecture.

Activation is routed through `config._act` so a single LEAKY_ALPHA toggle
re-keys ReLU vs Leaky-ReLU(alpha) for all four classes simultaneously.
"""

import torch.nn as nn

from .config import _act


class TinyModel(nn.Module):
    """5-layer tiny model (64x64 hidden)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 10)
        self.double()

    def forward(self, x):
        x = x.view(-1, 64)
        x = _act(self.fc1(x))
        x = _act(self.fc2(x))
        x = _act(self.fc3(x))
        x = _act(self.fc4(x))
        x = self.fc5(x)
        return x


class TinierModel(nn.Module):
    """5-layer tinier model with non-uniform hidden widths (32->16->16->16->8->4)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.double()

    def forward(self, x):
        x = x.view(-1, 32)
        x = _act(self.fc1(x))
        x = _act(self.fc2(x))
        x = _act(self.fc3(x))
        x = _act(self.fc4(x))
        x = self.fc5(x)
        return x


class TiniestModel(nn.Module):
    """Tiniest 8-8-8-8-8-8 make_blobs model."""
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
        x = _act(self.fc1(x))
        x = _act(self.fc2(x))
        x = _act(self.fc3(x))
        x = _act(self.fc4(x))
        x = self.fc5(x)
        return x


class FullModel(nn.Module):
    """Full CIFAR-10 model (3072 -> 256 -> 256 -> 256 -> 64 -> 10)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 10)
        self.double()

    def forward(self, x):
        x = x.view(-1, 3072)
        x = _act(self.fc1(x))
        x = _act(self.fc2(x))
        x = _act(self.fc3(x))
        x = _act(self.fc4(x))
        x = self.fc5(x)
        return x
