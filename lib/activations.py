import numpy as np
from .layers import Layer

class Activation(Layer):
    """
    Base class for activations, inheriting from Layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        self.output = self._function(self.input)
        return self.output

    def backward(self, output_gradient):
        # Element-wise multiplication: dL/dY * f'(X)
        return np.multiply(output_gradient, self._derivative(self.input))

    def _function(self, x):
        raise NotImplementedError

    def _derivative(self, x):
        raise NotImplementedError

class ReLU(Activation):
    def _function(self, x):
        return np.maximum(0, x)

    def _derivative(self, x):
        return np.where(x > 0, 1, 0)

class Sigmoid(Activation):
    def _function(self, x):
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _derivative(self, x):
        s = self._function(x)
        return s * (1 - s)

class Tanh(Activation):
    def _function(self, x):
        return np.tanh(x)

    def _derivative(self, x):
        t = np.tanh(x)
        return 1 - t**2

class Softmax(Layer):
    """
    Softmax is unique because it depends on all inputs for each output.
    It doesn't fit perfectly into the element-wise Activation pattern above.
    """
    def forward(self, input_data):
        self.input = input_data
        # Shift values for numerical stability
        exps = np.exp(self.input - np.max(self.input, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        # We need to compute the gradient of the loss w.r.t input (Z)
        # dL/dZ = dL/dY * dY/dZ
        # For Softmax, this is complex to vectorize efficiently without huge Jacobian matrices for batches.
        # However, a simplified efficient implementation for batch processing:
        
        n = np.size(self.output, 0) # Batch size
        input_gradient = np.empty_like(output_gradient)
        
        for i in range(n):
            # Reshape to (output_dim, 1)
            out_vector = self.output[i].reshape(-1, 1)
            grad_vector = output_gradient[i].reshape(-1, 1)
            
            # Jacobian matrix for single sample: S * (diag(1) - S^T)
            # or: diag(S) - S * S^T
            jacobian = np.diagflat(out_vector) - np.dot(out_vector, out_vector.T)
            
            # dL/dX = Jacobian . dL/dY
            input_gradient[i] = np.dot(jacobian, grad_vector).reshape(-1)
            
        return input_gradient