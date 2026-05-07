import numpy as np


def mean_squared_error_scratch(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score_scratch(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - ss_res / ss_tot