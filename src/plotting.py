import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Shared theme constants ────────────────────────────────────────────────────
_C = {
    "ridge":      "#1244B0",   # blue
    "lasso":      "#9128D3",   # purple
    "ols":        "#D97706",   # amber
    "bg":         "#F9FAFB",   # near-white panel
    "grid":       "#E5E7EB",   # light grey grid
    "text":       "#1F2937",   # dark grey text
    "arrow":      "#9CA3AF",   # mid-grey trajectory arrows
    "good":       "#059669",   # green for "ideal" annotations
}

_RIDGE_MARKER = "o"
_LASSO_MARKER = "s"
_OLS_MARKER   = "D"


def _apply_theme(ax):
    """Apply the shared background / grid style to an Axes object."""
    ax.set_facecolor(_C["bg"])
    ax.grid(True, color=_C["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(_C["grid"])


def _new_fig():
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    _apply_theme(ax)
    return fig, ax


def _ols_hline(ax, value, label):
    """Draw a horizontal OLS baseline dashed line."""
    if value is not None:
        ax.axhline(value, linestyle="--", color=_C["ols"],
                   linewidth=1.4, alpha=0.85, label=label, zorder=3)


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
            plt.savefig(path, dpi=180, bbox_inches="tight")

        if self.show_figures:
            plt.show()
        else:
            plt.close()

    def _get_rows(self, results, model_name):
        rows = [row for row in results if row["model"] == model_name]
        return sorted(rows, key=lambda row: row["lambda"])

    def _get_ols_row(self, results):
        ols_rows = [row for row in results if row["model"] == "OLS"]
        return ols_rows[0] if ols_rows else None

    # ── Error curves ──────────────────────────────────────────────────────────

    def plot_error_curves(self, results, dataset_name):
        """Plot training and test MSE for different lambda values."""

        ridge_rows = self._get_rows(results, "Ridge")
        lasso_rows = self._get_rows(results, "Lasso")
        ols_row    = self._get_ols_row(results)

        fig, ax = _new_fig()

        if ridge_rows:
            lambdas   = [r["lambda"]    for r in ridge_rows]
            train_mse = [r["train_mse"] for r in ridge_rows]
            test_mse  = [r["test_mse"]  for r in ridge_rows]
            ax.plot(lambdas, train_mse, marker=_RIDGE_MARKER, color=_C["ridge"],
                    linestyle="--", linewidth=1.6, alpha=0.7, label="Ridge Train MSE", zorder=4)
            ax.plot(lambdas, test_mse,  marker=_RIDGE_MARKER, color=_C["ridge"],
                    linewidth=2.0, label="Ridge Test MSE", zorder=4)

        if lasso_rows:
            lambdas  = [r["lambda"]   for r in lasso_rows]
            test_mse = [r["test_mse"] for r in lasso_rows]
            ax.plot(lambdas, test_mse, marker=_LASSO_MARKER, color=_C["lasso"],
                    linewidth=2.0, label="Lasso Test MSE", zorder=4)

        if ols_row is not None:
            _ols_hline(ax, ols_row["train_mse"], "OLS Train MSE")
            _ols_hline(ax, ols_row["test_mse"],  "OLS Test MSE")

        ax.set_xscale("log")
        ax.set_xlabel("Regularization strength (λ)", fontsize=12)
        ax.set_ylabel("Mean Squared Error", fontsize=12)
        # ax.set_title(f"Train / Test Error vs Regularization\n{dataset_name}",
                    #  fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=9, framealpha=0.9, edgecolor=_C["grid"])

        plt.tight_layout()
        self._save_or_show(f"{self._safe_name(dataset_name)}_error_curves.png")

    # ── Prediction stability ──────────────────────────────────────────────────

    def plot_prediction_stability(self, results, dataset_name):
        """Plot prediction-based stability for different lambda values."""

        ridge_rows = self._get_rows(results, "Ridge")
        lasso_rows = [r for r in self._get_rows(results, "Lasso")
                      if r["prediction_change"] is not None]
        ols_row    = self._get_ols_row(results)

        fig, ax = _new_fig()

        if ridge_rows:
            lambdas      = [r["lambda"]           for r in ridge_rows]
            pred_changes = [r["prediction_change"] for r in ridge_rows]
            ax.plot(lambdas, pred_changes, marker=_RIDGE_MARKER, color=_C["ridge"],
                    linewidth=2.0, label="Ridge prediction change", zorder=4)

        if lasso_rows:
            lambdas      = [r["lambda"]           for r in lasso_rows]
            pred_changes = [r["prediction_change"] for r in lasso_rows]
            ax.plot(lambdas, pred_changes, marker=_LASSO_MARKER, color=_C["lasso"],
                    linewidth=2.0, label="Lasso prediction change", zorder=4)

        if ols_row is not None:
            _ols_hline(ax, ols_row["prediction_change"], "OLS prediction change")

        ax.set_xscale("log")
        ax.set_xlabel("Regularization strength (λ)", fontsize=12)
        ax.set_ylabel("Average prediction change", fontsize=12)
        # ax.set_title(f"Prediction Stability vs Regularization\n{dataset_name}",
        #              fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=9, framealpha=0.9, edgecolor=_C["grid"])

        plt.tight_layout()
        self._save_or_show(f"{self._safe_name(dataset_name)}_prediction_stability.png")

    # ── Loss stability ────────────────────────────────────────────────────────

    def plot_loss_stability(self, results, dataset_name):
        """Plot loss-based stability for different lambda values."""

        ridge_rows = self._get_rows(results, "Ridge")
        lasso_rows = [r for r in self._get_rows(results, "Lasso")
                      if r["loss_change"] is not None]
        ols_row    = self._get_ols_row(results)

        fig, ax = _new_fig()

        if ridge_rows:
            lambdas      = [r["lambda"]      for r in ridge_rows]
            loss_changes = [r["loss_change"] for r in ridge_rows]
            ax.plot(lambdas, loss_changes, marker=_RIDGE_MARKER, color=_C["ridge"],
                    linewidth=2.0, label="Ridge loss change", zorder=4)

        if lasso_rows:
            lambdas      = [r["lambda"]      for r in lasso_rows]
            loss_changes = [r["loss_change"] for r in lasso_rows]
            ax.plot(lambdas, loss_changes, marker=_LASSO_MARKER, color=_C["lasso"],
                    linewidth=2.0, label="Lasso loss change", zorder=4)

        if ols_row is not None:
            _ols_hline(ax, ols_row["loss_change"], "OLS loss change")

        ax.set_xscale("log")
        ax.set_xlabel("Regularization strength (λ)", fontsize=12)
        ax.set_ylabel("Average loss change", fontsize=12)
        # ax.set_title(f"Loss Stability vs Regularization\n{dataset_name}",
        #              fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=9, framealpha=0.9, edgecolor=_C["grid"])

        plt.tight_layout()
        self._save_or_show(f"{self._safe_name(dataset_name)}_loss_stability.png")

    def plot_stability_curves(self, results, dataset_name):
        """Create both stability plots."""
        self.plot_prediction_stability(results, dataset_name)
        self.plot_loss_stability(results, dataset_name)

    # ── Stability vs test error (improved scatter) ────────────────────────────

    def plot_stability_vs_test_error(self, results, dataset_name):
        """Plot the relationship between stability and test error."""

        ridge_rows = [r for r in self._get_rows(results, "Ridge")
                      if r["prediction_change"] is not None]
        lasso_rows = [r for r in self._get_rows(results, "Lasso")
                      if r["prediction_change"] is not None]
        ols_row    = self._get_ols_row(results)

        fig, ax = _new_fig()

        # ── Ridge: colour-coded scatter + trajectory ──────────────────────────
        if ridge_rows:
            xs      = [r["prediction_change"] for r in ridge_rows]
            ys      = [r["test_mse"]          for r in ridge_rows]
            lambdas = [r["lambda"]            for r in ridge_rows]

            log_lam = np.log10([max(l, 1e-16) for l in lambdas])
            norm    = plt.Normalize(log_lam.min(), log_lam.max())
            cmap    = plt.cm.plasma_r

            # dashed trajectory
            ax.plot(xs, ys, color=_C["arrow"], linewidth=1.2,
                    linestyle="--", alpha=0.45, zorder=1)

            # direction arrows along path
            for i in range(len(xs) - 1):
                dx, dy = xs[i+1] - xs[i], ys[i+1] - ys[i]
                if abs(dx) + abs(dy) < 1:
                    continue
                ax.annotate("",
                    xy=(xs[i] + dx*0.55, ys[i] + dy*0.55),
                    xytext=(xs[i] + dx*0.45, ys[i] + dy*0.45),
                    arrowprops=dict(arrowstyle="-|>", color=_C["arrow"],
                                    lw=1.0, mutation_scale=10),
                    zorder=2)

            sc = ax.scatter(xs, ys, c=log_lam, cmap=cmap, norm=norm,
                            s=90, zorder=5, edgecolors="white", linewidths=0.8,
                            label="Ridge models")

            # colorbar
            cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
            cb.set_label("log₁₀(λ)", fontsize=10)

            # annotate key lambda values
            _LABEL_LAMS = {1e-15, 0.001, 1, 5, 30, 100}
            for x, y, lam in zip(xs, ys, lambdas):
                if lam not in _LABEL_LAMS:
                    continue
                text = f"λ={lam:.0e}" if lam < 0.01 else f"λ={lam:g}"
                if lam == 5:
                    text += "\n(most stable)"
                ax.annotate(text,
                    xy=(x, y), xytext=(6, 8), textcoords="offset points",
                    fontsize=8, color=_C["text"],
                    arrowprops=dict(arrowstyle="-", color=_C["arrow"], lw=0.7),
                    zorder=7)

        # ── Lasso scatter ─────────────────────────────────────────────────────
        if lasso_rows:
            xs      = [r["prediction_change"] for r in lasso_rows]
            ys      = [r["test_mse"]          for r in lasso_rows]
            lambdas = [r["lambda"]            for r in lasso_rows]
            ax.scatter(xs, ys, marker=_LASSO_MARKER, color=_C["lasso"],
                       s=80, zorder=5, edgecolors="white", linewidths=0.8,
                       label="Lasso models")
            for x, y, lam in zip(xs, ys, lambdas):
                ax.annotate(f"λ={lam:g}",
                    xy=(x, y), xytext=(6, -14), textcoords="offset points",
                    fontsize=7.5, color=_C["lasso"], zorder=7)

        # ── OLS marker ────────────────────────────────────────────────────────
        if ols_row is not None and ols_row["prediction_change"] is not None:
            ax.scatter([ols_row["prediction_change"]], [ols_row["test_mse"]],
                       marker=_OLS_MARKER, s=110, color=_C["ols"],
                       edgecolors="white", linewidths=0.8, zorder=6, label="OLS")
            ax.annotate("OLS",
                xy=(ols_row["prediction_change"], ols_row["test_mse"]),
                xytext=(8, -16), textcoords="offset points",
                fontsize=8.5, color=_C["ols"], fontweight="bold", zorder=7)

        # ── ideal-corner annotation ───────────────────────────────────────────
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.annotate("← ideal: stable & accurate",
            xy=(xlim[0] + (xlim[1]-xlim[0])*0.02,
                ylim[0] + (ylim[1]-ylim[0])*0.04),
            fontsize=8, color=_C["good"], style="italic")

        # ── legend ────────────────────────────────────────────────────────────
        legend_elements = [
            Line2D([0],[0], marker=_RIDGE_MARKER, color="w",
                   markerfacecolor=plt.cm.plasma_r(0.1), markersize=9,
                   label="Ridge (low λ)"),
            Line2D([0],[0], marker=_RIDGE_MARKER, color="w",
                   markerfacecolor=plt.cm.plasma_r(0.9), markersize=9,
                   label="Ridge (high λ)"),
            Line2D([0],[0], marker=_OLS_MARKER, color="w",
                   markerfacecolor=_C["ols"], markersize=9, label="OLS"),
            Line2D([0],[0], linestyle="--", color=_C["arrow"],
                   alpha=0.6, label="λ trajectory"),
        ]
        if lasso_rows:
            legend_elements.insert(2,
                Line2D([0],[0], marker=_LASSO_MARKER, color="w",
                       markerfacecolor=_C["lasso"], markersize=9,
                       label="Lasso models"))
        ax.legend(handles=legend_elements, loc="upper left",
                  fontsize=9, framealpha=0.9, edgecolor=_C["grid"])

        ax.set_xlabel("Avg Prediction Change  (↓ more stable)", fontsize=12)
        ax.set_ylabel("Test MSE  (↓ better accuracy)", fontsize=12)
        # ax.set_title(f"Stability vs. Test Error\n{dataset_name}",
        #              fontsize=13, fontweight="bold", pad=12)

        plt.tight_layout()
        self._save_or_show(f"{self._safe_name(dataset_name)}_stability_vs_test_error.png")

    # ── Model complexity ──────────────────────────────────────────────────────

    def plot_model_complexity(self, results, dataset_name):
        """Plot coefficient shrinkage as a simple model-complexity measure."""

        ridge_rows = self._get_rows(results, "Ridge")
        lasso_rows = self._get_rows(results, "Lasso")
        ols_row    = self._get_ols_row(results)

        fig, ax = _new_fig()

        if ridge_rows:
            lambdas    = [r["lambda"]       for r in ridge_rows]
            coef_norms = [r["coef_l2_norm"] for r in ridge_rows]
            ax.plot(lambdas, coef_norms, marker=_RIDGE_MARKER, color=_C["ridge"],
                    linewidth=2.0, label="Ridge coefficient norm", zorder=4)

        if lasso_rows:
            lambdas    = [r["lambda"]       for r in lasso_rows]
            coef_norms = [r["coef_l2_norm"] for r in lasso_rows]
            ax.plot(lambdas, coef_norms, marker=_LASSO_MARKER, color=_C["lasso"],
                    linewidth=2.0, label="Lasso coefficient norm", zorder=4)

        if ols_row is not None:
            _ols_hline(ax, ols_row["coef_l2_norm"], "OLS coefficient norm")

        ax.set_xscale("log")
        ax.set_xlabel("Regularization strength (λ)", fontsize=12)
        ax.set_ylabel("L2 norm of coefficients", fontsize=12)
        # ax.set_title(f"Model Complexity vs Regularization\n{dataset_name}",
        #              fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=9, framealpha=0.9, edgecolor=_C["grid"])

        plt.tight_layout()
        self._save_or_show(f"{self._safe_name(dataset_name)}_model_complexity.png")

    # ── Dataset size effect ───────────────────────────────────────────────────

    def plot_dataset_size_effect(self, results, lambda_fixed):
        """Plot how dataset size affects stability and test error."""

        sample_sizes = [r["sample_size"]       for r in results]
        pred_changes = [r["prediction_change"] for r in results]
        test_mse     = [r["test_mse"]          for r in results]

        # ── Stability plot ────────────────────────────────────────────────────
        fig, ax = _new_fig()
        ax.errorbar(
            sample_sizes, pred_changes,
            # yerr=pred_changes_std,
            marker=_RIDGE_MARKER, color=_C["ridge"],
            linewidth=2.0, capsize=4,
            ecolor=_C["ridge"], elinewidth=1.2,
        )
        ax.set_xlabel("Training sample size", fontsize=12)
        ax.set_ylabel("Average prediction change", fontsize=12)
        plt.tight_layout()
        self._save_or_show("dataset_size_effect_stability.png")

        # ── Test error plot ───────────────────────────────────────────────────
        fig, ax = _new_fig()
        ax.errorbar(
            sample_sizes, test_mse,
            # yerr=test_mse_std,
            marker=_RIDGE_MARKER, color=_C["ridge"],
            linewidth=2.0, capsize=4,
            ecolor=_C["ridge"], elinewidth=1.2,
        )
        ax.set_xlabel("Training sample size", fontsize=12)
        ax.set_ylabel("Average Test MSE", fontsize=12)
        plt.tight_layout()
        self._save_or_show("dataset_size_effect_test_error.png")
