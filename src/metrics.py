import numpy as np


class Metrics:
    """
    Class containing evaluation metrics implemented from scratch.
    """

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Compute mean squared error."""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def r2_score(y_true, y_pred):
        """Compute the coefficient of determination, R²."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - ss_res / ss_tot
