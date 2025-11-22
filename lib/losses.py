import numpy as np

class MSE:
    """
    Mean Squared Error Loss.
    Formula: L = (1/N) * sum((y_true - y_pred)^2)
    """
    def compute(self, y_true, y_pred):
        """
        Returns the scalar loss (mean over all samples).
        """
        return np.mean(np.power(y_true - y_pred, 2))

    def gradient(self, y_true, y_pred):
        """
        Returns the derivative of MSE w.r.t y_pred.
        Formula: dL/dY_pred = 2/N * (y_pred - y_true)
        """
        samples = np.size(y_true)
        return 2 * (y_pred - y_true) / samples