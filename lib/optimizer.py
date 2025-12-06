# =============================================================================
#  File:        lib/optimizer.py
#  Author:      Omar Khaled
#  Created:     5/12/2025
#
#  Description:
#       Optimizer implementations for the neural network library.
#      
#
#  Contents:
#		- Optimizer: Base optimizer class.
# 		- SGD: Stochastic Gradient Descent optimizer.
#
#  Notes:
#     - All optimizers inherit from the base Optimizer class.
# =============================================================================
import numpy as np


class Optimizer:
	"""Base class for all optimizers."""
	def __init__(self, params):
		# params: list of (param_array, grad_array)
		self.params = list(params)

	def step(self):
		raise NotImplementedError


class SGD(Optimizer):
	"""Stochastic Gradient Descent optimizer with optional momentum.
		Updates parameters using the formula:
		v = momentum * v - lr * grad
		param += v
	"""
	def __init__(self, params, lr=0.01, momentum=0.0):
		"""Initializes the SGD optimizer.

		Parameters
		----------
		params : list of tuples
			List of (parameter array, gradient array) to optimize.
		lr : float, optional
			Learning rate (default is 0.01).
		momentum : float, optional
			Momentum factor (default is 0.0, no momentum) 
			"Adds a fraction of the previous update vector to the current one, smoothing updates and speeding convergence."
		"""	
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