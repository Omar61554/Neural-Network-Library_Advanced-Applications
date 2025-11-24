# --------------------------
# FILE: lib/network.py
# --------------------------
"""Keras-like Model API: add, compile, fit, predict."""
import numpy as np
from .utils import to_batches


class Model:
	def __init__(self):
		self.layers = []
		self._built = False
		self._loss = None
		self._optimizer = None

	def add(self, layer):
		self.layers.append(layer)

	def _forward(self, x):
		out = x
		for layer in self.layers:
			out = layer.forward(out)
		return out

	def _backward(self, grad):
		g = grad
		for layer in reversed(self.layers):
			g = layer.backward(g)

	def parameters(self):
		ps = []
		for layer in self.layers:
			if hasattr(layer, "parameters"):
				ps.extend(layer.parameters())
		return ps

	def compile(self, optimizer, loss):
		self._optimizer = optimizer(self.parameters()) if callable(optimizer) else optimizer
		self._loss = loss if not isinstance(loss, str) else __import__('lib').losses.__dict__[loss]()

	def fit(self, X, y, epochs=100, batch_size=4, verbose=1):
		history = {'loss': []}
		for epoch in range(1, epochs+1):
			epoch_loss = 0.0
			batches = list(to_batches(X, y, batch_size=batch_size, shuffle=True))
			for xb, yb in batches:
				preds = self._forward(xb)
				loss_val = self._loss.forward(preds, yb)
				epoch_loss += loss_val * xb.shape[0]
				grad = self._loss.backward()
				self._backward(grad)
				# optimizer step
				self._optimizer.step()
				# zero grads
				for layer in self.layers:
					if hasattr(layer, "zero_grad"):
						layer.zero_grad()
			epoch_loss /= X.shape[0]
			history['loss'].append(epoch_loss)
			if verbose and (epoch == 1 or epoch % max(1, epochs//10) == 0):
				print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f}")
		return history

	def predict(self, X):
		return self._forward(X)