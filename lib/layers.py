# --------------------------
# FILE: lib/layers.py
# --------------------------
"""Layer implementations."""
import numpy as np
from .utils import xavier_init


class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        """
        Returns list of tuples (param, grad) for all parameters.
        """
        return []

    def zero_grad(self):
        """
        Resets gradients to zero.
        """
        pass

class Linear(Layer):
	def __init__(self, out_features, use_bias=True, init='xavier'):
		super().__init__()
		self.out_features = out_features
		self.use_bias = use_bias
		self.init = init
		self.built = False


	def build(self, input_shape):
		in_features = input_shape[1]
		if self.init == 'xavier':
			self.W = xavier_init(in_features, self.out_features)
		elif self.init == 'he':
			from .utils import he_init
			self.W = he_init(in_features, self.out_features)
		else:
			self.W = np.random.randn(in_features, self.out_features) * 0.01
		self.b = np.zeros((1, self.out_features)) if self.use_bias else None
		self.dW = np.zeros_like(self.W)
		self.db = np.zeros_like(self.b) if self.use_bias else None
		self._x = None
		self.built = True


	def forward(self, x):
		if not self.built:
			self.build(x.shape)
		self._x = x
		out = x.dot(self.W)
		if self.use_bias:
			out = out + self.b
		return out


	def backward(self, grad):
		# grad: (batch, out_features)
		batch_size = self._x.shape[0]
		self.dW[:] = self._x.T.dot(grad) / batch_size
		if self.use_bias:
			self.db[:] = np.mean(grad, axis=0, keepdims=True)
		return grad.dot(self.W.T)


	def parameters(self):
		params = [(self.W, self.dW)]
		if self.use_bias:
			params.append((self.b, self.db))
		return params


	def zero_grad(self):
		self.dW.fill(0)
		if self.use_bias:
			self.db.fill(0)


class Flatten(Layer):
	def __init__(self):
		super().__init__()
		self._orig_shape = None

	def forward(self, x):
		self._orig_shape = x.shape
		return x.reshape(x.shape[0], -1)

	def backward(self, grad):
		return grad.reshape(self._orig_shape)