import numpy as np

from config import (
    RANDOM_STATE,
    LAMBDA_VALUES,
    FIGURE_FOLDER,
    SAVE_FIGURES,
    SHOW_FIGURES,
    RUN_L1_L2_EXTENSION,
    LASSO_MAX_ITER,
    LASSO_TOL,
    COMPUTE_LASSO_STABILITY,
)

from src.data_utils import load_diabetes_dataset, make_synthetic_regression
from src.experiments import ExperimentRunner
from src.plotting import Plotter


def main():
    np.random.seed(RANDOM_STATE)

    plotter = Plotter(
        figure_folder=FIGURE_FOLDER,
        save_figures=SAVE_FIGURES,
        show_figures=SHOW_FIGURES
    )
    experiment = ExperimentRunner(
        lambda_values=LAMBDA_VALUES,
        random_state=RANDOM_STATE,
        plotter=plotter,
        run_l1_l2_extension=RUN_L1_L2_EXTENSION,
        lasso_max_iter=LASSO_MAX_ITER,
        lasso_tol=LASSO_TOL,
        compute_lasso_stability=COMPUTE_LASSO_STABILITY
    )

    # -----------------------------------------------------
    # 1. Real-world regression dataset: Diabetes
    # -----------------------------------------------------
    X_real, y_real, feature_names = load_diabetes_dataset()

    print("Real-world dataset: Diabetes")
    print(f"Samples: {X_real.shape[0]}")
    print(f"Features: {X_real.shape[1]}")
    print("Feature names:", feature_names)
    print("Target: quantitative disease progression after one year")

    real_results = experiment.run_experiment(
        X_real,
        y_real,
        dataset_name="Diabetes Dataset",
        feature_names=feature_names,
        ridge_plot_feature_index=2
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

    synthetic_results = experiment.run_experiment(
        X_syn,
        y_syn,
        dataset_name="Synthetic Regression Dataset",
        ridge_plot_feature_index=0
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

    size_results = experiment.run_dataset_size_experiment(
        X_size,
        y_size,
        lambda_fixed=1.0,
        sample_sizes=[50, 80, 120, 180, 240, 320],
        n_repeats=10,
        test_size=0.2
    )

    print("\nExperiment completed.")
    print("Lower prediction/loss change means higher empirical stability.")
    print("Figures were saved in the figures folder if SAVE_FIGURES=True.")

    return real_results, synthetic_results, size_results


if __name__ == "__main__":
    main()
