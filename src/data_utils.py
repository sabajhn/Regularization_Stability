import numpy as np
import pandas as pd


ENERGY_COLUMNS = [
    "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"
]


def load_energy_efficiency_dataset(file_path, target="Y1"):
    """
    Load the Energy Efficiency dataset.

    X1-X8 are the input features.
    Y1 is Heating Load.
    Y2 is Cooling Load.
    """

    data = pd.read_excel(file_path)

    if data.shape[1] >= 10:
        data = data.iloc[:, :10]
        data.columns = ENERGY_COLUMNS

    X = data[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values

    if target not in ["Y1", "Y2"]:
        raise ValueError("target must be either 'Y1' or 'Y2'")

    y = data[target].values

    return X, y


def make_synthetic_regression(n_samples=250, n_features=20, noise=5.0, random_state=42):
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