# --------------------------
# FILE: lib/activations.py
# --------------------------
"""Activation layers (stateful for backward)."""
import numpy as np


class Activation:
	def forward(self, x):
		raise NotImplementedError
	def backward(self, grad):
		raise NotImplementedError


class Tanh(Activation):
	def __init__(self):
		self._out = None
	def forward(self, x):
		self._out = np.tanh(x)
		return self._out
	def backward(self, grad):
		return grad * (1 - self._out**2)
	def parameters(self):
		return []
	def zero_grad(self):
		return []


class Sigmoid(Activation):
	def __init__(self):
		self._out = None
	def forward(self, x):
		self._out = 1 / (1 + np.exp(-x))
		return self._out
	def backward(self, grad):
		return grad * (self._out * (1 - self._out))
	def parameters(self):
		return []
	def zero_grad(self):
		return []


class ReLU(Activation):
	def __init__(self):
		self._mask = None
	def forward(self, x):
		self._mask = (x > 0).astype(float)
		return x * self._mask
	def backward(self, grad):
		return grad * self._mask
	def parameters(self):
		return []
	def zero_grad(self):
		return []
	
class Softmax(Activation):
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