class SGD:
    """
    Stochastic Gradient Descent Optimizer.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Updates the weights and biases of all Dense layers.
        """
        for layer in layers:
            # Only update layers that have weights (e.g., Dense)
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.dweights
                layer.biases  -= self.learning_rate * layer.dbiases