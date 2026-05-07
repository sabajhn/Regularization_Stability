import os
import matplotlib.pyplot as plt


class Plotter:
    """
    Plotting class for the stability project.

    This class is responsible only for figures.
    It does not train models, compute metrics, or estimate stability.
    """

    def __init__(self, figure_folder="figures", save_figures=True, show_figures=True):
        self.figure_folder = figure_folder
        self.save_figures = save_figures
        self.show_figures = show_figures

    def _safe_name(self, name):
        return name.lower().replace(" ", "_").replace("/", "_")

    def _save_or_show(self, filename):
        if self.save_figures:
            os.makedirs(self.figure_folder, exist_ok=True)
            path = os.path.join(self.figure_folder, filename)
            plt.savefig(path, dpi=300, bbox_inches="tight")

        if self.show_figures:
            plt.show()
        else:
            plt.close()

    def _get_ridge_and_ols(self, results):
        ridge_rows = [row for row in results if row["model"] == "Ridge"]
        ridge_rows = sorted(ridge_rows, key=lambda row: row["lambda"])

        ols_rows = [row for row in results if row["model"] == "OLS"]
        ols_row = ols_rows[0] if len(ols_rows) > 0 else None

        return ridge_rows, ols_row

    def plot_error_curves(self, results, dataset_name):
        """Plot training and test MSE for different lambda values."""

        ridge_rows, ols_row = self._get_ridge_and_ols(results)

        lambdas = [row["lambda"] for row in ridge_rows]
        train_mse = [row["train_mse"] for row in ridge_rows]
        test_mse = [row["test_mse"] for row in ridge_rows]

        plt.figure(figsize=(8, 5))

        plt.plot(lambdas, train_mse, marker="o", label="Ridge Train MSE")
        plt.plot(lambdas, test_mse, marker="o", label="Ridge Test MSE")

        if ols_row is not None:
            plt.axhline(ols_row["train_mse"], linestyle="--", label="OLS Train MSE")
            plt.axhline(ols_row["test_mse"], linestyle="--", label="OLS Test MSE")

        plt.xscale("log")
        plt.xlabel("Regularization strength (lambda)")
        plt.ylabel("Mean Squared Error")
        plt.title(f"Train/Test Error vs Regularization - {dataset_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        filename = f"{self._safe_name(dataset_name)}_error_curves.png"
        self._save_or_show(filename)

    def plot_prediction_stability(self, results, dataset_name):
        """Plot prediction-based stability for different lambda values."""

        ridge_rows, ols_row = self._get_ridge_and_ols(results)

        lambdas = [row["lambda"] for row in ridge_rows]
        pred_changes = [row["prediction_change"] for row in ridge_rows]

        plt.figure(figsize=(8, 5))

        plt.plot(lambdas, pred_changes, marker="o", label="Ridge prediction change")

        if ols_row is not None:
            plt.axhline(
                ols_row["prediction_change"],
                linestyle="--",
                label="OLS prediction"
            )

        plt.xscale("log")
        plt.xlabel("Regularization strength (lambda)")
        plt.ylabel("Average prediction change")
        plt.title(f"Prediction Stability vs Regularization - {dataset_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        filename = f"{self._safe_name(dataset_name)}_prediction_stability.png"
        self._save_or_show(filename)

    def plot_loss_stability(self, results, dataset_name):
        """Plot loss-based stability for different lambda values."""

        ridge_rows, ols_row = self._get_ridge_and_ols(results)

        lambdas = [row["lambda"] for row in ridge_rows]
        loss_changes = [row["loss_change"] for row in ridge_rows]

        plt.figure(figsize=(8, 5))

        plt.plot(lambdas, loss_changes, marker="o", label="Ridge loss change")

        if ols_row is not None:
            plt.axhline(
                ols_row["loss_change"],
                linestyle="--",
                label="OLS loss"
            )

        plt.xscale("log")
        plt.xlabel("Regularization strength (lambda)")
        plt.ylabel("Average loss change")
        plt.title(f"Loss Stability vs Regularization - {dataset_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        filename = f"{self._safe_name(dataset_name)}_loss_stability.png"
        self._save_or_show(filename)

    def plot_stability_curves(self, results, dataset_name):
        """
        Create both stability plots.

        This wrapper makes the experiment code shorter.
        """

        self.plot_prediction_stability(results, dataset_name)
        self.plot_loss_stability(results, dataset_name)

    def plot_stability_vs_test_error(self, results, dataset_name):
        """Plot the relationship between stability and test error."""

        ridge_rows, ols_row = self._get_ridge_and_ols(results)

        pred_changes = [row["prediction_change"] for row in ridge_rows]
        test_mse = [row["test_mse"] for row in ridge_rows]
        lambdas = [row["lambda"] for row in ridge_rows]

        plt.figure(figsize=(8, 5))

        plt.scatter(pred_changes, test_mse, label="Ridge models")

        if ols_row is not None:
            plt.scatter(
                [ols_row["prediction_change"]],
                [ols_row["test_mse"]],
                marker="x",
                s=90,
                label="OLS"
            )

        for x_value, y_value, lam in zip(pred_changes, test_mse, lambdas):
            plt.annotate(
                f"lambda={lam}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

        plt.xlabel("Average prediction change")
        plt.ylabel("Test MSE")
        plt.title(f"Stability and Test Error - {dataset_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        filename = f"{self._safe_name(dataset_name)}_stability_vs_test_error.png"
        self._save_or_show(filename)

    def plot_dataset_size_effect(self, results, lambda_fixed):
        """Plot how dataset size affects stability and test error."""

        sample_sizes = [row["sample_size"] for row in results]
        pred_changes = [row["prediction_change"] for row in results]
        test_mse = [row["test_mse"] for row in results]

        plt.figure(figsize=(8, 5))

        plt.plot(sample_sizes, pred_changes, marker="o")
        plt.xlabel("Total sample size before train/test split")
        plt.ylabel("Average prediction change")
        plt.title(f"Dataset Size Effect on Stability (Ridge, lambda={lambda_fixed})")
        plt.grid(True, alpha=0.3)

        self._save_or_show("dataset_size_effect_stability.png")

        plt.figure(figsize=(8, 5))

        plt.plot(sample_sizes, test_mse, marker="o")
        plt.xlabel("Total sample size before train/test split")
        plt.ylabel("Test MSE")
        plt.title(f"Dataset Size Effect on Test Error (Ridge, lambda={lambda_fixed})")
        plt.grid(True, alpha=0.3)

        self._save_or_show("dataset_size_effect_test_error.png")