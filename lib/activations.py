import numpy as np
from .layers import Layer

class Activation(Layer):
    """
    Base class for activations. 
    Implements forward and backward methods common to all activations.
    """
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """
        Applies the activation function to the input data.
        """
        self.input = input_data
        self.output = self._activation_function(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Computes the gradient of the loss with respect to the input of the activation.
        """
        # We ignore learning_rate because activations have no learnable parameters
        return np.multiply(output_gradient, self._derivative_function(self.input))

    def _activation_function(self, x):
        raise NotImplementedError

    def _derivative_function(self, x):
        raise NotImplementedError


class Sigmoid(Activation):
    """
    Sigmoid Activation: 1 / (1 + e^-x)
    Range: (0, 1)
    """
    def _activation_function(self, x):
        # Clip x to prevent overflow in exp()
        # Values < -500 become 0.0, > 500 become 1.0
        clipped_x = np.clip(x, -500, 500) 
        return 1 / (1 + np.exp(-clipped_x))

    def _derivative_function(self, x):
        # Derivative of Sigmoid is: f(x) * (1 - f(x))
        s = self._activation_function(x)
        return s * (1 - s)


class Tanh(Activation):
    """
    Hyperbolic Tangent: (e^x - e^-x) / (e^x + e^-x)
    Range: (-1, 1)
    """
    def _activation_function(self, x):
        return np.tanh(x)

    def _derivative_function(self, x):
        # Derivative of Tanh is: 1 - tanh(x)^2
        return 1 - np.tanh(x) ** 2


class ReLU(Activation):
    """
    Rectified Linear Unit: max(0, x)
    Range: [0, inf)
    """
    def _activation_function(self, x):
        return np.maximum(0, x)

    def _derivative_function(self, x):
        # Derivative is 1 if x > 0, else 0
        return (x > 0).astype(float)


class Softmax(Layer):
    """
    Softmax: e^x_i / sum(e^x_j)
    Range: (0, 1), sum = 1
    Used for: Output layer in Multi-class classification.
    
    NOTE: Softmax is unique because the output of one neuron depends on ALL inputs 
    (due to the sum in denominator). It strictly inherits from Layer, not Activation,
    because its backward pass is more complex (Jacobian matrix) than element-wise multiplication.
    """
    def forward(self, input_data):
        self.input = input_data
        # Shift x by subtracting max to prevent overflow (numerical stability)
        # exp(x - c) / sum(exp(x - c)) == exp(x) / sum(exp(x))
        tmp = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = tmp / np.sum(tmp, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Softmax derivative is complex. 
        # If used with CrossEntropy, terms cancel out beautifully.
        # But as a standalone layer, we must implement the vector calculus version.
        
        # We calculate: y * (grad - sum(grad * y))
        
        n = np.size(self.output, axis=0) # Batch size
        input_gradient = np.zeros(output_gradient.shape)
        
        # We iterate over the batch because the derivative is a matrix (Jacobian) per sample
        # This loop can be vectorized, but loop is clearer for understanding:
        for i in range(n):
            # Reshape single sample to (1, -1)
            output_i = self.output[i].reshape(-1, 1)
            grad_i = output_gradient[i].reshape(-1, 1)
            
            # Jacobian matrix for softmax: diag(y) - y * y.T
            jacobian_matrix = np.diagflat(output_i) - np.dot(output_i, output_i.T)
            
            # Multiply Jacobian by output gradient
            input_gradient[i] = np.dot(jacobian_matrix, grad_i).reshape(-1)
            
        return input_gradient