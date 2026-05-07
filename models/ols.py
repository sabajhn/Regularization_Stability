import numpy as np

class LinearRegression:
    """Ordinary Least Squares using the closed-form pseudo-inverse solution."""

    def fit(self, X, y):
        X_bias = np.column_stack((np.ones(X.shape[0]), X))
        self.theta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
        return self

    def predict(self, X):
        X_bias = np.column_stack((np.ones(X.shape[0]), X))
        return X_bias @ self.theta