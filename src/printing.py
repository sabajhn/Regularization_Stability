def print_results_table(results):
    header = (
        f"{'Model':<8} {'Lambda':>10} {'Train MSE':>12} {'Test MSE':>12} "
        f"{'Train R2':>10} {'Test R2':>10} {'Pred Change':>14} {'Loss Change':>14}"
    )

    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['model']:<8} "
            f"{row['lambda']:>10.4g} "
            f"{row['train_mse']:>12.4f} "
            f"{row['test_mse']:>12.4f} "
            f"{row['train_r2']:>10.4f} "
            f"{row['test_r2']:>10.4f} "
            f"{row['prediction_change']:>14.6f} "
            f"{row['loss_change']:>14.6f}"
        )

    ridge_rows = [row for row in results if row["model"] == "Ridge"]

    if len(ridge_rows) > 0:
        best_test = min(ridge_rows, key=lambda row: row["test_mse"])
        best_stability = min(ridge_rows, key=lambda row: row["prediction_change"])

        print("\nBest Ridge by test MSE:")
        print(f"lambda = {best_test['lambda']}, test MSE = {best_test['test_mse']:.4f}")

        print("Most stable Ridge by prediction change:")
        print(
            f"lambda = {best_stability['lambda']}, "
            f"prediction change = {best_stability['prediction_change']:.6f}"
        )


def print_size_results_table(results):
    header = (
        f"{'Sample Size':>12} {'Train Size':>12} {'Test MSE':>12} "
        f"{'Pred Change':>14} {'Loss Change':>14}"
    )

    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['sample_size']:>12} "
            f"{row['train_size']:>12} "
            f"{row['test_mse']:>12.4f} "
            f"{row['prediction_change']:>14.6f} "
            f"{row['loss_change']:>14.6f}"
        )