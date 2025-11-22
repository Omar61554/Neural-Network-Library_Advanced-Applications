import numpy as np
from .layers import Layer
from .losses import mse, mse_prime
from .optimizer import SGD

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.optimizer = None

    def add(self, layer):
        """
        Adds a layer to the network.
        """
        if not isinstance(layer, Layer):
            raise TypeError("Added object must be a Layer instance")
        self.layers.append(layer)

    def use_loss(self, loss_func, loss_prime_func):
        """
        Sets the loss function and its derivative.
        """
        self.loss = loss_func
        self.loss_prime = loss_prime_func

    def config_optimizer(self, optimizer):
        """
        Sets the optimizer.
        """
        self.optimizer = optimizer

    def forward(self, input_data):
        """
        Performs the forward pass through all layers.
        """
        # Forward pass through all layers
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, loss_gradient):
        """
        Performs the backward pass through all layers in reverse.
        """
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def train(self, x_train, y_train, epochs, learning_rate=0.01, verbose=True):
        """
        Main training loop.
        """
        # Initialize optimizer if not already set, or update its LR
        if self.optimizer is None:
            self.optimizer = SGD(learning_rate)
        else:
            self.optimizer.learning_rate = learning_rate

        # Default loss if not set (Project requirement asks for MSE)
        if self.loss is None:
            self.use_loss(mse, mse_prime)

        for epoch in range(epochs):
            loss_display = 0
            
            # Forward Pass
            output = self.forward(x_train)
            
            # Calculate Loss
            loss_display = self.loss(y_train, output)
            
            # Backward Pass
            # 1. Calculate initial gradient from loss function
            grad = self.loss_prime(y_train, output)
            
            # 2. Propagate gradient backward
            self.backward(grad)
            
            # 3. Update parameters
            self.optimizer.step(self.layers)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_display:.6f}")