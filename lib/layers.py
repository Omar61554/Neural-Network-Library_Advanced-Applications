# =============================================================================
#  File:        lib/layers.py
#  Author:      Omar Khaled
#  Created:     5/12/2025
#
#  Description:
# 	Layer implementations for the neural network library.
#   Includes fully connected (Linear) and Flatten layers.
#      
#
#  Contents:
#		- Layer: Base layer class.
# 		- Linear: Fully connected layer.
# 		- Flatten: Flattens input.    
#
#  Notes:
#     - All layers inherit from the base Layer class.
#
# =============================================================================

import numpy as np
from .utils import xavier_init, he_init, to_batches 


class Layer:
    """
    Base class for all neural network layers.

    This class defines the essential interface that every layer in the
    network must implement, including forward and backward passes and
    parameter/gradient management.
    """

    def __init__(self):
        """
        Initializes the base Layer object.

        Subclasses may override this to define trainable parameters or
        internal states.
        """
        pass

    def forward(self, x):
        """
        Performs the forward pass of the layer.

        Parameters
        ----------
        x : numpy.ndarray
            Input tensor to the layer.

        Returns
        -------
        numpy.ndarray
            Output tensor after applying the layer's transformation.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad):
        """
        Performs the backward pass of the layer.

        Parameters
        ----------
        grad : numpy.ndarray
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to the input of this layer.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Backward method not implemented.")

    def parameters(self):
        """
        Returns the trainable parameters of the layer.

        Returns
        -------
        list of tuples
            A list containing `(param, grad)` tuples for every
            trainable parameter in the layer.
            Layers without trainable parameters return an empty list.
        """
        return []

    def zero_grad(self):
        """
        Resets gradients of all trainable parameters to zero.

        Notes
        -----
        This is typically called at the beginning of each optimization step.
        Layers without trainable parameters may ignore this method.
        """
        pass


class Dense(Layer):
    """
    Fully-connected (dense) layer.

    Computes:
        y = xW + b

    where:
    - x is the input of shape (batch_size, in_features)
    - W is the weight matrix of shape (in_features, out_features)
    - b is the optional bias vector of shape (1, out_features)
    """

    def __init__(self, out_features, use_bias=True, init="xavier"):
        """
        Initializes a Linear layer (weights created later in `build`).

        Parameters
        ----------
        out_features : int
            Dimensionality of the output features.
        use_bias : bool, optional
            Whether to include a bias term (default is True).
        init : str, optional
            Weight initialization method. One of:
            ["xavier", "he", "normal"].
        """
        super().__init__()
        self.out_features = out_features
        self.use_bias = use_bias
        self.init = init
        self.built = False   # set False, build when input is known

    def build(self, input_shape):
        """
        Creates layer parameters based on the input shape.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor (batch_size, in_features).
        """
        in_features = input_shape[1]

        # Initialize weights
        if self.init == "xavier":
            self.W = xavier_init(in_features, self.out_features)
        elif self.init == "he":
            from .utils import he_init
            self.W = he_init(in_features, self.out_features)
        else:
            # Default normal initialization
            self.W = np.random.randn(in_features, self.out_features) * 0.01

        # Initialize bias if needed
        self.b = np.zeros((1, self.out_features)) if self.use_bias else None

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if self.use_bias else None

        # Cache for backward
        self._x = None

        self.built = True

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        np.ndarray
            Output of shape (batch_size, out_features).
        """
        if not self.built:
            self.build(x.shape)

        self._x = x

        out = x.dot(self.W)
        if self.use_bias:
            out = out + self.b

        return out

    def backward(self, grad):
        """
        Performs backward pass and computes gradients.

        Parameters
        ----------
        grad : np.ndarray
            Gradient of the loss w.r.t. output
            shape (batch_size, out_features)

        Returns
        -------
        np.ndarray
            Gradient of the loss w.r.t. input,
            shape (batch_size, in_features)
        """
        batch_size = self._x.shape[0]

        # Compute gradients
        self.dW[:] = self._x.T.dot(grad) / batch_size # Scale by batch size
        if self.use_bias:
            self.db[:] = np.mean(grad, axis=0, keepdims=True) # Average over batch

        # Gradient w.r.t. input
        return grad.dot(self.W.T)

    def parameters(self):
        """
        Returns all trainable parameters and their gradients.

        Returns
        -------
        list of tuple
            [(W, dW), (b, db if use_bias)]
        """
        params = [(self.W, self.dW)]
        if self.use_bias:
            params.append((self.b, self.db))
        return params

    def zero_grad(self):
        """
        Resets all gradients to zero.
        """
        self.dW.fill(0)
        if self.use_bias:
            self.db.fill(0)


class Flatten(Layer):
	"""
	Flattens the input tensor except for the batch dimension.

	For example, an input of shape (batch_size, d1, d2, d3)
	will be flattened to (batch_size, d1*d2*d3).
    used before fully connected layers to convert multi-dimensional data may be used for autoencoding and GAN.
	"""
	def __init__(self):
		
		super().__init__()
		self._orig_shape = None
	def forward(self, x):
		self._orig_shape = x.shape
		return x.reshape(x.shape[0], -1)

	def backward(self, grad):
		return grad.reshape(self._orig_shape)