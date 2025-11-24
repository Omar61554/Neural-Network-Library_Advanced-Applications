# --------------------------
# FILE: lib/utils.py
# --------------------------
"""Utility helpers: weight init and batching"""
import numpy as np


def xavier_init(in_dim, out_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(in_dim, out_dim))


def he_init(in_dim, out_dim):
    std = np.sqrt(2.0 / in_dim)
    return np.random.randn(in_dim, out_dim) * std


def to_batches(X, y=None, batch_size=32, shuffle=True):
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