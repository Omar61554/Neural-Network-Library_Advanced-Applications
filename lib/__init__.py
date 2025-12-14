# =============================================================================
#  File:        lib/__init__.py
#  Author:      Omar Khaled
#  Created:     5/12/2025
#
#  Description:
#      Minimal neural-network package `lib` (Keras-like API).
#      
#
#  Contents:
#  Network:
#		- Model: Neural network model class.
# 		- Linear: Fully connected layer.
# 		- Flatten: Flattens input.
#   Activations:
# 		- Tanh: Hyperbolic tangent activation.
# 		- Sigmoid: Sigmoid activation.
# 		- ReLU: Rectified Linear Unit activation.
# 		- Softmax: Softmax activation.
#   Losses:
# 		- MSELoss: Mean Squared Error loss.
# 		- CrossEntropy: Cross-Entropy loss.
#   Optimizers:
# 		- SGD: Stochastic Gradient Descent optimizer.
#
#  Notes:
#     - Provides high-level API for building, training, and evaluating neural networks.
#
# =============================================================================

from .network import Model
from .layers import Dense, Flatten
from .activations import Tanh, Sigmoid, ReLU, Softmax
from .losses import MSELoss, CrossEntropy
from .optimizer import SGD



__all__ = [
'Model', 'Dense', 'Flatten', 'Tanh', 'Sigmoid', 'ReLU','Softmax', 'MSELoss', 'CrossEntropy', 'SGD'
]