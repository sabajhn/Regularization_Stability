import numpy as np
from sklearn.datasets import load_diabetes


class DataUtils:
    """
    Utility class for loading and generating datasets.

    All dataset-related helper methods are kept inside this class so the
    project does not rely on standalone utility functions.
    """

    @staticmethod
    def load_diabetes_dataset():
        """
        Load the Diabetes dataset from scikit-learn.

        This is the real-world regression dataset used in the project.
        The target is a quantitative measure of disease progression one year
        after baseline. The input matrix has 10 numeric baseline variables:
        age, sex, bmi, bp, tc, ldl, hdl, tch, ltg, and glu.

        The original scikit-learn dataset is already mean-centered and scaled,
        but the project still applies train-only standardization before modeling
        to keep the preprocessing pipeline consistent for all datasets.
        """

        data = load_diabetes()

        X = data.data.astype(float)
        y = data.target.astype(float)

        feature_names = list(data.feature_names)
        renamed_features = {
            "s1": "tc",
            "s2": "ldl",
            "s3": "hdl",
            "s4": "tch",
            "s5": "ltg",
            "s6": "glu",
        }
        feature_names = [renamed_features.get(name, name) for name in feature_names]

        return X, y, feature_names

    @staticmethod
    def make_synthetic_regression(
        n_samples=250,
        n_features=20,
        noise=5.0,
        random_state=42
    ):
        """
        Create a synthetic linear regression dataset.

        X is sampled from a normal distribution.
        y is generated from a linear function plus Gaussian noise.
        """

        rng = np.random.default_rng(random_state)

        X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
        true_weights = rng.normal(loc=0.0, scale=1.0, size=n_features)
        noise_term = rng.normal(loc=0.0, scale=noise, size=n_samples)

        y = X @ true_weights + noise_term

        return X, y
