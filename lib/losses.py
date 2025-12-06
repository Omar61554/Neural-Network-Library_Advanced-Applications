# =============================================================================
#  File:        lib/losses.py
#  Author:      Omar Khaled
#  Created:     5/12/2025
#
#  Description:
#       Loss function implementations for the neural network library.
#      
#
#  Contents:
#		- Loss: Base loss class.
# 		- MSELoss: Mean Squared Error loss.
# 		- CrossEntropy: Cross-Entropy loss.
#
#  Notes:
#     - All losses inherit from the base Loss class.
# =============================================================================

import numpy as np


class Loss:
	"""Base class for all loss functions."""
	def forward(self, pred, target):
		raise NotImplementedError

	def backward(self):
		raise NotImplementedError


class MSELoss(Loss):
	"""Mean Squared Error loss function.
		Calculates the average squared difference between predictions and targets.
		g(x, y) = mean((x - y)^2)
		g'(x, y) = 2 * (x - y) / n
	"""
	def __init__(self):
		self._pred = None
		self._target = None

	def forward(self, pred, target):
		self._pred = pred
		self._target = target
		return np.mean((pred - target) ** 2)

	def backward(self):
		n = self._pred.shape[0]
		return 2 * (self._pred - self._target) / n


class CrossEntropy(Loss):
	"""Cross-Entropy loss function.
		Calculates the cross-entropy loss between predicted probabilities and true labels.
		For binary classification, it expects probabilities from a Sigmoid activation.
		For multi-class classification, it expects probabilities from a Softmax activation.
		g(p, t) = -mean(t * log(p))  (categorical)
		g'(p, t) = (p - t) / n
	"""
	def __init__(self):
		self._pred = None
		self._target = None

	def forward(self, pred, target):
		# pred: probabilities from softmax or sigmoid
		eps = 1e-12
		self._pred = np.clip(pred, eps, 1 - eps)
		self._target = target
		if self._pred.shape == self._target.shape and self._pred.shape[1] == 1:
			# binary cross-entropy
			return -np.mean(self._target * np.log(self._pred) + (1 - self._target) * np.log(1 - self._pred))
		else:
			# categorical cross-entropy expects one-hot targets
			return -np.mean(np.sum(self._target * np.log(self._pred), axis=1))

	def backward(self):
		n = self._pred.shape[0]
		return (self._pred - self._target) / n
