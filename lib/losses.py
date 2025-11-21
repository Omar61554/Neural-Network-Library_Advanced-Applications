import numpy as np

def mse(y_true, y_pred):
    """
    Mean Squared Error (MSE) Loss.
    """
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """
    Derivative of MSE Loss w.r.t y_pred.
    """
    return 2 * (y_pred - y_true) / np.size(y_true)