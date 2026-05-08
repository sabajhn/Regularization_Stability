def _format_number(value, decimals=4):
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _format_float_width(value, width, decimals=4):
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def print_results_table(results):
    header = (
        f"{'Model':<8} {'Lambda':>10} {'Train MSE':>12} {'Test MSE':>12} "
        f"{'Train R2':>10} {'Test R2':>10} {'Coef Norm':>12} "
        f"{'Nonzero':>8} {'Pred Change':>14} {'Loss Change':>14}"
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
            f"{row['coef_l2_norm']:>12.4f} "
            f"{row['nonzero_coefs']:>8} "
            f"{_format_float_width(row['prediction_change'], 14, 6)} "
            f"{_format_float_width(row['loss_change'], 14, 6)}"
        )

    for model_name in ["Ridge", "Lasso"]:
        model_rows = [row for row in results if row["model"] == model_name]

        if len(model_rows) > 0:
            best_test = min(model_rows, key=lambda row: row["test_mse"])
            rows_with_stability = [row for row in model_rows if row["prediction_change"] is not None]

            print(f"\nBest {model_name} by test MSE:")
            print(f"lambda = {best_test['lambda']}, test MSE = {best_test['test_mse']:.4f}")

            if rows_with_stability:
                best_stability = min(rows_with_stability, key=lambda row: row["prediction_change"])
                print(f"Most stable {model_name} by prediction change:")
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
