# =============================================================================
#  File:        lib/activations.py
#  Author:      Omar Khaled
#  Created:     5/12/2025
#
#  Description:
#      Activation function implementations for the neural network library.
#      
#
#  Contents:
#		- Activation: Base activation class.
# 		- Tanh: Hyperbolic tangent activation.
# 		- Sigmoid: Sigmoid activation.
# 		- ReLU: Rectified Linear Unit activation.
# 		- Softmax: Softmax activation. 
#
#  Notes:
#     - All activations inherit from the base Activation class.
#
# =============================================================================

import numpy as np


class Activation:
	"""Base class for all activation functions."""
	def forward(self, x):
		raise NotImplementedError
	def backward(self, grad):
		raise NotImplementedError


class Tanh(Activation):
	"""Hyperbolic tangent activation function.
		Tanh squashes input values to the range [-1, 1].
	"""
	def __init__(self):
		self._out = None
	def forward(self, x):
		self._out = np.tanh(x)
		return self._out
	def backward(self, grad):
		return grad * (1 - self._out**2) # derivative of  tanh= 1 - tanh^2(x)
	def parameters(self):
		return []
	def zero_grad(self):
		return []


class Sigmoid(Activation):
	"""Sigmoid activation function.
		Sigmoid squashes input values to the range [0, 1].
		g(x) = 1 / (1 + exp(-x))
	"""
	def __init__(self):
		self._out = None
	def forward(self, x):
		self._out = 1 / (1 + np.exp(-x))
		return self._out
	def backward(self, grad):
		return grad * (self._out * (1 - self._out)) # derivative of sigmoid = sigmoid(x) * (1 - sigmoid(x))
	def parameters(self):
		return []
	def zero_grad(self):
		return []


class ReLU(Activation):
	"""Rectified Linear Unit activation function.
		ReLU outputs the input directly if it is positive; otherwise, it outputs zero.
		g(x) = max(0, x)
	"""
	def __init__(self):
		self._mask = None
	def forward(self, x):
		self._mask = (x > 0).astype(float)
		return x * self._mask
	def backward(self, grad):
		return grad * self._mask # derivative of ReLU is 1 for x>0 else 0
	def parameters(self):
		return []
	def zero_grad(self):
		return []
	
class Softmax(Activation):
	"""Softmax activation function.
		Softmax converts logits into probabilities that sum to 1.
		Typically used in the output layer for multi-class classification.
		g(x_i) = exp(x_i) / sum_j exp(x_j)
		g'(x_i) = g(x_i) * (1 - g(x_i)) for i=j
		          -g(x_i) * g(x_j) for i!=j
	"""
	def __init__(self):
		self._out = None

	def forward(self, x):
		exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
		self._out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
		return self._out

	def backward(self, grad):
		batch_size = grad.shape[0]
		dx = np.zeros_like(grad)
		for i in range(batch_size):
			y = self._out[i].reshape(-1, 1)
			jacobian = np.diagflat(y) - np.dot(y, y.T)
			dx[i] = np.dot(jacobian, grad[i])
		return dx
	def parameters(self):
		return []
	def zero_grad(self):
		return []