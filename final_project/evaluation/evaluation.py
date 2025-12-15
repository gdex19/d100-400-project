import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    weights: pd.Series,
) -> pd.Series:
    """
    Evaluate weighted predictive performance under Gamma deviance.

    Parameters
    ----------
    y_true : pd.Series
        True responder.
    y_pred : pd.Series
        Model predictions.
    weights : pd.Series
        Observation weights.

    Returns
    -------
    pd.Series
        Bias, Gamma deviance, MAE, and RMSE.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    weights_arr = np.asarray(weights)

    true_mean = np.sum(y_true_arr * weights_arr) / np.sum(weights_arr)
    pred_mean = np.sum(y_pred_arr * weights_arr) / np.sum(weights_arr)
    bias = pred_mean - true_mean

    deviance = mean_tweedie_deviance(
        y_true_arr,
        y_pred_arr,
        sample_weight=weights_arr,
        power=2,  # 2 for gamma
    )

    mae = np.sum(np.abs(y_true_arr - y_pred_arr) * weights_arr) / np.sum(
        weights_arr
    )
    rmse = np.sqrt(
        np.sum((y_true_arr - y_pred_arr) ** 2 * weights_arr)
        / np.sum(weights_arr)
    )

    result: pd.Series = pd.Series(
        {
            "bias": bias,
            "gamma_deviance": deviance,
            "mae": mae,
            "rmse": rmse,
        }
    )

    return result
