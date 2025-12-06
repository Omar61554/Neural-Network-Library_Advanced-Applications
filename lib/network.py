# =============================================================================
#  File:        lib/network.py
#  Author:      Omar Khaled
#  Created:     5/12/2025
#
#  Description:
#       Neural network model class for the neural network library.
#      
#
#  Contents:
#		- Model: Neural network model class.
#
#  Notes:
#     - The Model class manages layers, training, and prediction.
# =============================================================================

import numpy as np
from .utils import to_batches
from lib import losses


class Model:
    """
    Minimal sequential neural network model.

    Layers are applied in the order they are added.

    Example
    -------
    >>> model = Model()
    >>> model.add(Linear(4))
    >>> model.add(Tanh())
    >>> model.add(Linear(1))
    >>> model.compile(optimizer=lambda p: SGD(p, lr=0.1),
                      loss=MSELoss())
    """

    def __init__(self):
        self.layers = []
        self._built = False
        self._loss = None
        self._optimizer = None

    def add(self, layer):
        """Appends a layer to the model."""
        self.layers.append(layer)

    def _forward(self, x):
        """Internal forward pass through all layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out) # pass output to next layer
        return out

    def _backward(self, grad):
        """Internal backward pass (reverse order)."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad) # pass gradient to previous layer

    def parameters(self):
        """
        Collects all trainable parameters from every layer.

        Returns
        -------
        list of tuple
            List of (param, grad) pairs.
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def compile(self, optimizer, loss):
        """
        Configures the model for training.

        Parameters
        ----------
        optimizer : callable or Optimizer
            If callable → optimizer(parameters) is invoked.
        loss : Loss or str
            Loss function instance or name of loss.
        """
        self._optimizer = (
            optimizer(self.parameters()) if callable(optimizer) else optimizer
        )

        if isinstance(loss, str):
            self._loss = losses.__dict__[loss]()
        else:
            self._loss = loss

    def fit(self, X, y, epochs=100, batch_size=4, verbose=1):
        """
        Trains the model.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features).
        y : np.ndarray
            Ground-truth labels, shape compatible with output layer.
        epochs : int, optional
            Number of training epochs.
        batch_size : int, optional
            Size of mini-batches.
        verbose : int, optional
            Whether to print progress.

        Returns
        -------
        dict
            Training history with recorded losses.
        """
        history = {"loss": []}

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0

            for xb, yb in to_batches(X, y, batch_size=batch_size, shuffle=True): 
                preds = self._forward(xb)
                loss_val = self._loss.forward(preds, yb)
                epoch_loss += loss_val * xb.shape[0] 

                grad = self._loss.backward()
                self._backward(grad)

                self._optimizer.step()

                # Reset gradients after update
                for layer in self.layers:
                    if hasattr(layer, "zero_grad"):
                        layer.zero_grad()

            epoch_loss /= X.shape[0]
            history["loss"].append(epoch_loss)

            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
                print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f}")

        return history

    def predict(self, X):
        """Runs inference on given input."""
        return self._forward(X)
