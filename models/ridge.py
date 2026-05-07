import numpy as np
class RidgeRegression:
    """
    Ridge Regression from scratch using the closed-form solution.

    Objective used here:
        mean squared error + lambda * ||w||^2

    The intercept is not regularized.
    """

    def __init__(self, l2_penalty=1.0):
        self.l2_penalty = l2_penalty

    def fit(self, X, y):
        n_samples = X.shape[0]
        X_bias = np.column_stack((np.ones(n_samples), X))

        identity = np.eye(X_bias.shape[1])
        identity[0, 0] = 0  # do not regularize intercept

        regularized_matrix = X_bias.T @ X_bias + n_samples * self.l2_penalty * identity
        self.theta = np.linalg.pinv(regularized_matrix) @ X_bias.T @ y

        return self

    def predict(self, X):
        X_bias = np.column_stack((np.ones(X.shape[0]), X))
        return X_bias @ self.theta
