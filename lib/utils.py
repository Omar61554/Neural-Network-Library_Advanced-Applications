# =============================================================================
#  File:        lib/utils.py
#  Author:      Omar Khaled
#  Created:     5/12/2025
#
#  Description:
#      Collection of utility functions used across the neural network library.
#      Includes weight initialization methods and data preprocessing helpers.
#
#  Contents:
#      - xavier_init: Xavier/Glorot weight initialization.
#      - he_init: He/Kaiming weight initialization.
#      - to_batches: Split datasets into mini-batches.
#
#  Notes:
#      - All initialization functions return NumPy arrays suitable for use
#        in fully connected neural network layers.
#
# =============================================================================

import numpy as np

def xavier_init(in_dim: int, out_dim: int) -> np.ndarray:
    """
    Initializes a weight matrix using Xavier (Glorot) initialization.

    This method is designed to keep the variance of activations and gradients
    consistent across layers, improving training stability—especially for 
    networks using activation functions such as Tanh or Sigmoid.

    The function uses the uniform Xavier initialization:

        limit = sqrt(6 / (in_dim + out_dim))

    and samples weights from the range [-limit, limit].

    Parameters
    ----------
    in_dim : int
        Number of input units (fan-in) of the layer.
    out_dim : int
        Number of output units (fan-out) of the layer.

    Returns
    -------
    np.ndarray
        The initialized weight matrix of shape (in_dim, out_dim).

    Notes
    -----
    - Use this initializer for activations such as **tanh** or **sigmoid**.
    - For ReLU-like activations, consider **He initialization** instead.
    - The classical Xavier initialization uses:
        - Normal: Var = 2 / (fan_in + fan_out)
        - Uniform: Limit = sqrt(6 / (fan_in + fan_out))

    Examples
    --------
    >>> W = xavier_init(3, 5)
    >>> W.shape
    (3, 5)
    """
    limit = np.sqrt(2.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(in_dim, out_dim))


def he_init(in_dim, out_dim) -> np.ndarray:
    """ 
    Initializes a weight matrix using Kaiming He initialization.
    This method is particularly effective for layers using ReLU or its variants,
    as it helps maintain variance of activations through the layers.

    Parameters
    ----------
    in_dim : int
        Number of input units (fan-in) of the layer.
    out_dim : int
        Number of output units (fan-out) of the layer.
    Returns
    -------
    np.ndarray
        The initialized weight matrix of shape (in_dim, out_dim).
    """
    std = np.sqrt(2.0 / in_dim)
    return np.random.randn(in_dim, out_dim) * std


def to_batches(X, y=None, batch_size=32, shuffle=True) :
    """
    Generator that yields batches of data from X (and y if provided) may be used for autoencoding and GAN.
    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    y : np.ndarray, optional
        Target labels of shape (n_samples, ...) (default is None).
    batch_size : int, optional
        Number of samples per batch (default is 32).
    shuffle : bool, optional
        Whether to shuffle the data before batching (default is True) to improve training.

    Yields 
    ------
    If y is provided:
        Tuple (X_batch, y_batch) for each batch.
    If y is None:
        X_batch for each batch.
    """ 
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        batch_idx = idx[i:i+batch_size]
        if y is None:
            yield X[batch_idx]
        else:
            yield X[batch_idx], y[batch_idx]