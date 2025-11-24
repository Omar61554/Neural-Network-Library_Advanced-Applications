# --------------------------
# FILE: lib/__init__.py
# --------------------------
"""
Minimal neural-network package `lib` (Keras-like API).
Expose high-level Model, layers, activations, losses, optimizers.
"""
from .network import Model
from .layers import Linear, Flatten
from .activations import Tanh, Sigmoid, ReLU, Softmax
from .losses import MSELoss, CrossEntropy
from .optimizer import SGD


__all__ = [
'Model', 'Linear', 'Flatten', 'Tanh', 'Sigmoid', 'ReLU','Softmax', 'MSELoss', 'CrossEntropy', 'SGD'
]