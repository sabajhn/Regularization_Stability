import numpy as np

from models.ols import LinearRegression
from models.ridge import RidgeRegression

from src.metrics import mean_squared_error_scratch, r2_score_scratch
from src.preprocessing import train_test_split_scratch, standardize_train_test
from src.stability import estimate_stability
from src.printing import print_results_table, print_size_results_table


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Return training and test metrics for a fitted model."""

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "train_mse": mean_squared_error_scratch(y_train, train_pred),
        "test_mse": mean_squared_error_scratch(y_test, test_pred),
        "train_r2": r2_score_scratch(y_train, train_pred),
        "test_r2": r2_score_scratch(y_test, test_pred),
    }


def run_experiment(X, y, dataset_name, lambda_values, random_state=42, plotter=None):
    """
    Run OLS and Ridge experiments on one dataset.

    This function is used for both the real and synthetic datasets.
    """

    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name}")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split_scratch(
        X,
        y,
        test_size=0.2,
        random_state=random_state
    )

    X_train, X_test = standardize_train_test(X_train, X_test)

    results = []

    # -------------------------
    # OLS baseline
    # -------------------------
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)

    ols_metrics = evaluate_model(ols_model, X_train, y_train, X_test, y_test)

    ols_pred_stability, ols_loss_stability = estimate_stability(
        LinearRegression,
        {},
        X_train,
        y_train,
        X_test,
        y_test
    )

    results.append({
        "dataset": dataset_name,
        "model": "OLS",
        "lambda": 0.0,
        "train_mse": ols_metrics["train_mse"],
        "test_mse": ols_metrics["test_mse"],
        "train_r2": ols_metrics["train_r2"],
        "test_r2": ols_metrics["test_r2"],
        "prediction_change": ols_pred_stability,
        "loss_change": ols_loss_stability,
    })

    # -------------------------
    # Ridge for different lambdas
    # -------------------------
    for lam in lambda_values:
        ridge_model = RidgeRegression(l2_penalty=lam)
        ridge_model.fit(X_train, y_train)

        ridge_metrics = evaluate_model(ridge_model, X_train, y_train, X_test, y_test)

        ridge_pred_stability, ridge_loss_stability = estimate_stability(
            RidgeRegression,
            {"l2_penalty": lam},
            X_train,
            y_train,
            X_test,
            y_test
        )

        results.append({
            "dataset": dataset_name,
            "model": "Ridge",
            "lambda": lam,
            "train_mse": ridge_metrics["train_mse"],
            "test_mse": ridge_metrics["test_mse"],
            "train_r2": ridge_metrics["train_r2"],
            "test_r2": ridge_metrics["test_r2"],
            "prediction_change": ridge_pred_stability,
            "loss_change": ridge_loss_stability,
        })

    print_results_table(results)

    if plotter is not None:
        plotter.plot_error_curves(results, dataset_name)
        plotter.plot_stability_curves(results, dataset_name)
        plotter.plot_stability_vs_test_error(results, dataset_name)

    return results


def run_dataset_size_experiment(
    X,
    y,
    lambda_fixed=1.0,
    sample_sizes=None,
    random_state=42,
    plotter=None
):
    """
    Study the effect of dataset size on Ridge stability.

    A larger training set is usually expected to make the algorithm more stable.
    """

    if sample_sizes is None:
        sample_sizes = [60, 100, 150, 220, 300]

    rng = np.random.default_rng(random_state)
    all_indices = np.arange(X.shape[0])
    results = []

    print("\n" + "=" * 70)
    print("Dataset size experiment using Ridge Regression")
    print("=" * 70)

    for sample_size in sample_sizes:
        if sample_size > X.shape[0]:
            continue

        chosen_indices = rng.choice(all_indices, size=sample_size, replace=False)
        X_subset = X[chosen_indices]
        y_subset = y[chosen_indices]

        X_train, X_test, y_train, y_test = train_test_split_scratch(
            X_subset,
            y_subset,
            test_size=0.2,
            random_state=random_state
        )

        X_train, X_test = standardize_train_test(X_train, X_test)

        model = RidgeRegression(l2_penalty=lambda_fixed)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

        pred_stability, loss_stability = estimate_stability(
            RidgeRegression,
            {"l2_penalty": lambda_fixed},
            X_train,
            y_train,
            X_test,
            y_test
        )

        results.append({
            "sample_size": sample_size,
            "train_size": len(X_train),
            "test_mse": metrics["test_mse"],
            "prediction_change": pred_stability,
            "loss_change": loss_stability,
        })

    print_size_results_table(results)

    if plotter is not None:
        plotter.plot_dataset_size_effect(results, lambda_fixed)

    return results