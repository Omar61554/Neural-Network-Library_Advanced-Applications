import numpy as np

class Layer:
    """
    Abstract base class for all layers.
    """
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """
        Computes the output of the layer given the input.
        """
        raise NotImplementedError

    def backward(self, output_gradient):
        """
        Computes the gradient with respect to the input.
        Updates parameters if the layer has any.
        """
        raise NotImplementedError

class Dense(Layer):
    """
    Fully connected layer.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        # Initialize weights using Xavier/Glorot initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros((1, output_size))
        
        # Gradients
        self.dweights = None
        self.dbiases = None

    def forward(self, input_data):
        self.input = input_data
        # Z = X . W + B
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient):
        # Gradient w.r.t weights: X^T . dL/dY
        self.dweights = np.dot(self.input.T, output_gradient)
        
        # Gradient w.r.t biases: sum(dL/dY) across batch dimension
        self.dbiases = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Gradient w.r.t input: dL/dY . W^T
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        return input_gradient