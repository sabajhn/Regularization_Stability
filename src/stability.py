import numpy as np


def estimate_stability(model_class, model_params, X_train, y_train, X_test, y_test):
    """
    Estimate algorithmic stability empirically.

    Steps:
    1. Train the model on the full training set.
    2. Remove one training point.
    3. Retrain the same model.
    4. Compare predictions on the fixed test set.

    Smaller values mean the algorithm is more stable.
    """

    full_model = model_class(**model_params)
    full_model.fit(X_train, y_train)

    full_predictions = full_model.predict(X_test)
    full_losses = (y_test - full_predictions) ** 2

    prediction_changes = []
    loss_changes = []

    for i in range(len(X_train)):
        X_minus_i = np.delete(X_train, i, axis=0)
        y_minus_i = np.delete(y_train, i, axis=0)

        loo_model = model_class(**model_params)
        loo_model.fit(X_minus_i, y_minus_i)

        loo_predictions = loo_model.predict(X_test)
        loo_losses = (y_test - loo_predictions) ** 2

        avg_prediction_change = np.mean(np.abs(full_predictions - loo_predictions))
        avg_loss_change = np.mean(np.abs(full_losses - loo_losses))

        prediction_changes.append(avg_prediction_change)
        loss_changes.append(avg_loss_change)

    return np.mean(prediction_changes), np.mean(loss_changes)