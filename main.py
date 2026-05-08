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

from src.data_utils import DataUtils
from src.experiments import ExperimentRunner
from src.plotting import Plotter


class ProjectRunner:
    """
    Main project runner.

    This class creates the data utility, plotter, and experiment runner,
    then runs all experiments from one place.
    """

    def __init__(self):
        np.random.seed(RANDOM_STATE)

        self.data_utils = DataUtils()
        self.plotter = Plotter(
            figure_folder=FIGURE_FOLDER,
            save_figures=SAVE_FIGURES,
            show_figures=SHOW_FIGURES
        )
        self.experiment = ExperimentRunner(
            lambda_values=LAMBDA_VALUES,
            random_state=RANDOM_STATE,
            plotter=self.plotter,
            run_l1_l2_extension=RUN_L1_L2_EXTENSION,
            lasso_max_iter=LASSO_MAX_ITER,
            lasso_tol=LASSO_TOL,
            compute_lasso_stability=COMPUTE_LASSO_STABILITY
        )

    def _run_real_dataset_experiment(self):
        """Run the main experiment on the real Diabetes dataset."""

        X_real, y_real, feature_names = self.data_utils.load_diabetes_dataset()

        print("Real-world dataset: Diabetes")
        print(f"Samples: {X_real.shape[0]}")
        print(f"Features: {X_real.shape[1]}")
        print("Feature names:", feature_names)
        print("Target: quantitative disease progression after one year")

        return self.experiment.run_experiment(
            X_real,
            y_real,
            dataset_name="Diabetes Dataset",
            feature_names=feature_names,
            ridge_plot_feature_index=2
        )

    def _run_synthetic_dataset_experiment(self):
        """Run the main experiment on the synthetic regression dataset."""

        X_syn, y_syn = self.data_utils.make_synthetic_regression(
            n_samples=250,
            n_features=20,
            noise=5.0,
            random_state=RANDOM_STATE
        )

        return self.experiment.run_experiment(
            X_syn,
            y_syn,
            dataset_name="Synthetic Regression Dataset",
            ridge_plot_feature_index=0
        )

    def _run_dataset_size_experiment(self):
        """Run the dataset-size stability experiment."""

        X_size, y_size = self.data_utils.make_synthetic_regression(
            n_samples=400,
            n_features=20,
            noise=5.0,
            random_state=123
        )

        return self.experiment.run_dataset_size_experiment(
            X_size,
            y_size,
            lambda_fixed=1.0,
            sample_sizes=[50, 80, 120, 180, 240, 320],
            n_repeats=10,
            test_size=0.2
        )

    def run(self):
        """Run all project experiments."""

        real_results = self._run_real_dataset_experiment()
        synthetic_results = self._run_synthetic_dataset_experiment()
        size_results = self._run_dataset_size_experiment()

        print("\nExperiment completed.")
        print("Lower prediction/loss change means higher empirical stability.")
        print("Figures were saved in the figures folder if SAVE_FIGURES=True.")

        return real_results, synthetic_results, size_results


if __name__ == "__main__":
    ProjectRunner().run()
