import numpy as np


class LassoRegression:
    """
    Lasso Regression from scratch using coordinate descent.

    Objective used here:
        1/(2n) * ||y - Xw - b||^2 + lambda * ||w||_1

    The intercept is not regularized.
    This model is used only for the L1 vs L2 extension.
    """

    def __init__(self, l1_penalty=1.0, max_iter=500, tol=1e-6):
        self.l1_penalty = l1_penalty
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def _soft_threshold(self, value, penalty):
        if value > penalty:
            return value - penalty
        if value < -penalty:
            return value + penalty
        return 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()

        X_centered = X - self.x_mean_
        y_centered = y - self.y_mean_

        gram = (X_centered.T @ X_centered) / n_samples
        xy = (X_centered.T @ y_centered) / n_samples

        diagonal = np.diag(gram).copy()
        diagonal[diagonal == 0] = 1.0

        weights = np.zeros(n_features)

        for _ in range(self.max_iter):
            old_weights = weights.copy()

            for j in range(n_features):
                rho = xy[j] - gram[j, :] @ weights + gram[j, j] * weights[j]
                weights[j] = self._soft_threshold(rho, self.l1_penalty) / diagonal[j]

            max_change = np.max(np.abs(weights - old_weights))
            if max_change < self.tol:
                break

        self.coef_ = weights
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
