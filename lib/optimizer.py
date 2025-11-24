# --------------------------
# FILE: lib/optimizers.py
# --------------------------
"""Simple optimizers. Only update in-place references."""
import numpy as np


class Optimizer:
	def __init__(self, params):
		# params: list of (param_array, grad_array)
		self.params = list(params)

	def step(self):
		raise NotImplementedError


class SGD(Optimizer):
	def __init__(self, params, lr=0.01, momentum=0.0):
		super().__init__(params)
		self.lr = lr
		self.momentum = momentum
		# create velocity arrays matching params
		self.velocities = [np.zeros_like(p) for p, g in self.params]

	def step(self):
		for i, (p, g) in enumerate(self.params):
			if g is None:
				continue
			v = self.velocities[i]
			v[:] = self.momentum * v - self.lr * g
			p += v