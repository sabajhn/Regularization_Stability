import os
import numpy as np

from config import (
    RANDOM_STATE,
    LAMBDA_VALUES,
    FIGURE_FOLDER,
    SAVE_FIGURES,
    SHOW_FIGURES,
    REAL_DATA_PATH,
    REAL_TARGET,
)

from src.data_utils import load_energy_efficiency_dataset, make_synthetic_regression
from src.experiments import run_experiment, run_dataset_size_experiment
from src.plotting import Plotter


def main():
    np.random.seed(RANDOM_STATE)

    plotter = Plotter(
        figure_folder=FIGURE_FOLDER,
        save_figures=SAVE_FIGURES,
        show_figures=SHOW_FIGURES
    )

    # -----------------------------------------------------
    # 1. Real-world regression dataset
    # -----------------------------------------------------
    if not os.path.exists(REAL_DATA_PATH):
        raise FileNotFoundError(
            f"Could not find the real dataset at: {REAL_DATA_PATH}\n"
            "Put ENB2012_data.xlsx inside the data/ folder."
        )

    X_real, y_real = load_energy_efficiency_dataset(
        REAL_DATA_PATH,
        target=REAL_TARGET
    )

    real_results = run_experiment(
        X_real,
        y_real,
        dataset_name="Energy Efficiency Dataset",
        lambda_values=LAMBDA_VALUES,
        random_state=RANDOM_STATE,
        plotter=plotter
    )

    # -----------------------------------------------------
    # 2. Synthetic regression dataset
    # -----------------------------------------------------
    X_syn, y_syn = make_synthetic_regression(
        n_samples=250,
        n_features=20,
        noise=5.0,
        random_state=RANDOM_STATE
    )

    synthetic_results = run_experiment(
        X_syn,
        y_syn,
        dataset_name="Synthetic Regression Dataset",
        lambda_values=LAMBDA_VALUES,
        random_state=RANDOM_STATE,
        plotter=plotter
    )

    # -----------------------------------------------------
    # 3. Dataset size experiment
    # -----------------------------------------------------
    X_size, y_size = make_synthetic_regression(
        n_samples=400,
        n_features=20,
        noise=5.0,
        random_state=123
    )

    size_results = run_dataset_size_experiment(
        X_size,
        y_size,
        lambda_fixed=1.0,
        sample_sizes=[60, 100, 150, 220, 300, 400],
        random_state=RANDOM_STATE,
        plotter=plotter
    )

    print("\nExperiment completed.")
    print("Lower prediction/loss change means higher empirical stability.")
    print("Figures were saved in the figures folder if SAVE_FIGURES=True.")

    return real_results, synthetic_results, size_results


if __name__ == "__main__":
    main()