import numpy as np

from models.ols import LinearRegression
from models.ridge import RidgeRegression
from models.lasso import LassoRegression

from src.metrics import Metrics
from src.preprocessing import Preprocessor
from src.stability import StabilityEstimator
from src.printing import ResultPrinter


class ExperimentRunner:
    """
    Runs the main regularization/stability experiments.

    This class keeps the experiment logic together instead of using
    standalone experiment functions.
    """

    def __init__(
        self,
        lambda_values,
        random_state=42,
        plotter=None,
        run_l1_l2_extension=True,
        lasso_max_iter=1000,
        lasso_tol=1e-6,
        compute_lasso_stability=True
    ):
        self.lambda_values = lambda_values
        self.random_state = random_state
        self.plotter = plotter
        self.run_l1_l2_extension = run_l1_l2_extension
        self.lasso_max_iter = lasso_max_iter
        self.lasso_tol = lasso_tol
        self.compute_lasso_stability = compute_lasso_stability

        self.metrics = Metrics()
        self.preprocessor = Preprocessor()
        self.stability_estimator = StabilityEstimator()
        self.printer = ResultPrinter()

    def _get_model_coefficients(self, model):
        """Return only feature coefficients, not the intercept."""

        if hasattr(model, "theta"):
            return model.theta[1:]

        if hasattr(model, "coef_"):
            return model.coef_

        return np.array([])

    def _coefficient_summary(self, model):
        """Measure model complexity using coefficient norm and non-zero coefficients."""

        coefficients = self._get_model_coefficients(model)

        return {
            "coef_l2_norm": float(np.sqrt(np.sum(coefficients ** 2))),
            "nonzero_coefs": int(np.sum(np.abs(coefficients) > 1e-8)),
        }

    def _evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Return training and test metrics for a fitted model."""

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        summary = self._coefficient_summary(model)

        return {
            "train_mse": self.metrics.mean_squared_error(y_train, train_pred),
            "test_mse": self.metrics.mean_squared_error(y_test, test_pred),
            "train_r2": self.metrics.r2_score(y_train, train_pred),
            "test_r2": self.metrics.r2_score(y_test, test_pred),
            "coef_l2_norm": summary["coef_l2_norm"],
            "nonzero_coefs": summary["nonzero_coefs"],
        }

    def _add_result_row(
        self,
        results,
        dataset_name,
        model_name,
        lam,
        metrics,
        pred_stability,
        loss_stability
    ):
        """Add one model result to the results list."""

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "lambda": lam,
            "train_mse": metrics["train_mse"],
            "test_mse": metrics["test_mse"],
            "train_r2": metrics["train_r2"],
            "test_r2": metrics["test_r2"],
            "coef_l2_norm": metrics["coef_l2_norm"],
            "nonzero_coefs": metrics["nonzero_coefs"],
            "prediction_change": pred_stability,
            "loss_change": loss_stability,
        })

    def _run_ols(self, results, dataset_name, X_train, y_train, X_test, y_test):
        """Fit and evaluate the OLS baseline."""

        ols_model = LinearRegression()
        ols_model.fit(X_train, y_train)

        ols_metrics = self._evaluate_model(
            ols_model,
            X_train,
            y_train,
            X_test,
            y_test
        )

        ols_pred_stability, ols_loss_stability = self.stability_estimator.estimate(
            LinearRegression,
            {},
            X_train,
            y_train,
            X_test,
            y_test
        )

        self._add_result_row(
            results,
            dataset_name,
            "OLS",
            0.0,
            ols_metrics,
            ols_pred_stability,
            ols_loss_stability
        )

    def _run_ridge(self, results, dataset_name, X_train, y_train, X_test, y_test):
        """Fit and evaluate Ridge for all lambda values."""

        for lam in self.lambda_values:
            ridge_model = RidgeRegression(l2_penalty=lam)
            ridge_model.fit(X_train, y_train)

            ridge_metrics = self._evaluate_model(
                ridge_model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            ridge_pred_stability, ridge_loss_stability = self.stability_estimator.estimate(
                RidgeRegression,
                {"l2_penalty": lam},
                X_train,
                y_train,
                X_test,
                y_test
            )

            self._add_result_row(
                results,
                dataset_name,
                "Ridge",
                lam,
                ridge_metrics,
                ridge_pred_stability,
                ridge_loss_stability
            )

    def _run_lasso(self, results, dataset_name, X_train, y_train, X_test, y_test):
        """Fit and evaluate Lasso for all lambda values."""

        if not self.run_l1_l2_extension:
            return

        for lam in self.lambda_values:
            lasso_model = LassoRegression(
                l1_penalty=lam,
                max_iter=self.lasso_max_iter,
                tol=self.lasso_tol
            )
            lasso_model.fit(X_train, y_train)

            lasso_metrics = self._evaluate_model(
                lasso_model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            if self.compute_lasso_stability:
                lasso_pred_stability, lasso_loss_stability = self.stability_estimator.estimate(
                    LassoRegression,
                    {
                        "l1_penalty": lam,
                        "max_iter": self.lasso_max_iter,
                        "tol": self.lasso_tol,
                    },
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )
            else:
                lasso_pred_stability = None
                lasso_loss_stability = None

            self._add_result_row(
                results,
                dataset_name,
                "Lasso",
                lam,
                lasso_metrics,
                lasso_pred_stability,
                lasso_loss_stability
            )

    def _make_standard_plots(
        self,
        results,
        dataset_name,
        X_train,
        y_train,
        feature_names=None,
        ridge_plot_feature_index=0
    ):
        """Create all plots for the main experiment."""

        if self.plotter is None:
            return

        self.plotter.plot_error_curves(results, dataset_name)
        self.plotter.plot_stability_curves(results, dataset_name)
        self.plotter.plot_stability_vs_test_error(results, dataset_name)
        self.plotter.plot_model_complexity(results, dataset_name)

    def run_experiment(
        self,
        X,
        y,
        dataset_name,
        feature_names=None,
        ridge_plot_feature_index=0
    ):
        """
        Run OLS, Ridge, and optional Lasso experiments on one dataset.

        The main models use all features.
        """

        print("\n" + "=" * 70)
        print(f"Dataset: {dataset_name}")
        print("=" * 70)

        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state
        )

        X_train, X_test = self.preprocessor.standardize_train_test(X_train, X_test)

        results = []

        self._run_ols(results, dataset_name, X_train, y_train, X_test, y_test)
        self._run_ridge(results, dataset_name, X_train, y_train, X_test, y_test)
        self._run_lasso(results, dataset_name, X_train, y_train, X_test, y_test)

        self.printer.print_results_table(results)

        self._make_standard_plots(
            results,
            dataset_name,
            X_train,
            y_train,
            feature_names=feature_names,
            ridge_plot_feature_index=ridge_plot_feature_index
        )

        return results

    def run_dataset_size_experiment(
        self,
        X,
        y,
        lambda_fixed=1.0,
        sample_sizes=None,
        n_repeats=10,
        test_size=0.2
    ):
        """
        Study the effect of dataset size on Ridge stability.

        This version uses:
        - one fixed test set
        - repeated random training subsets
        - average test MSE and average stability values
        """

        print("\n" + "=" * 70)
        print("Dataset size experiment using Ridge Regression")
        print("=" * 70)

        X_train_pool, X_test_fixed, y_train_pool, y_test_fixed = self.preprocessor.train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state
        )

        if sample_sizes is None:
            sample_sizes = [50, 80, 120, 180, 240, 320]

        rng = np.random.default_rng(self.random_state)
        pool_indices = np.arange(X_train_pool.shape[0])

        results = []

        for sample_size in sample_sizes:
            if sample_size > X_train_pool.shape[0]:
                continue

            mse_values = []
            pred_stability_values = []
            loss_stability_values = []

            for repeat in range(n_repeats):
                chosen_indices = rng.choice(
                    pool_indices,
                    size=sample_size,
                    replace=False
                )

                X_train_raw = X_train_pool[chosen_indices]
                y_train = y_train_pool[chosen_indices]

                X_train, X_test = self.preprocessor.standardize_train_test(
                    X_train_raw,
                    X_test_fixed
                )

                model = RidgeRegression(l2_penalty=lambda_fixed)
                model.fit(X_train, y_train)

                metrics = self._evaluate_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test_fixed
                )

                pred_stability, loss_stability = self.stability_estimator.estimate(
                    RidgeRegression,
                    {"l2_penalty": lambda_fixed},
                    X_train,
                    y_train,
                    X_test,
                    y_test_fixed
                )

                mse_values.append(metrics["test_mse"])
                pred_stability_values.append(pred_stability)
                loss_stability_values.append(loss_stability)

            results.append({
                "sample_size": sample_size,
                "train_size": sample_size,
                "test_size": len(y_test_fixed),
                "test_mse": float(np.mean(mse_values)),
                "test_mse_std": float(np.std(mse_values)),
                "prediction_change": float(np.mean(pred_stability_values)),
                "prediction_change_std": float(np.std(pred_stability_values)),
                "loss_change": float(np.mean(loss_stability_values)),
                "loss_change_std": float(np.std(loss_stability_values)),
            })

        self.printer.print_size_results_table(results)

        if self.plotter is not None:
            self.plotter.plot_dataset_size_effect(results, lambda_fixed)

        return results
