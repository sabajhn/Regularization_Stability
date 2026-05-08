import numpy as np


class Preprocessor:
    """
    Class containing preprocessing methods used by the experiments.
    """

    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=42):
        """Simple train-test split implemented from scratch."""

        rng = np.random.default_rng(random_state)

        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        test_count = int(n_samples * test_size)

        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def standardize_train_test(X_train, X_test):
        """
        Standardize features using only the training set.
        This avoids data leakage from the test set.
        """

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        std[std == 0] = 1

        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std

        return X_train_scaled, X_test_scaled
