import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.metrics import mean_tweedie_deviance
from sklearn.pipeline import Pipeline
from final_project.data import read_data, split_data
from final_project.preprocessing import NUM_FEATURES, CAT_FEATURES, RESPONDER
from final_project.modeling import EVENT_WEIGHT, load_models


def get_models_and_val_data(
    dataset_name: str,
) -> Tuple[Pipeline, Pipeline, pd.DataFrame, pd.Series,]:
    """
    Load data  + fit models, and return validation set.

    Parameters
    ----------
    dataset_name : str
        Dataset to read.

    Returns
    -------
    glm : Pipeline
        Fitted GLM pipeline.
    lgbm : Pipeline
        Fitted LGBM pipeline.
    X_val : pd.DataFrame
        Val features.
    y_val : pd.Series
        Val responder.
    """
    df = read_data(dataset_name)

    df_train, _, df_val = split_data(df, [0.6, 0.0, 0.4])

    X_train = df_train[NUM_FEATURES + CAT_FEATURES].copy()
    y_train = df_train[RESPONDER]

    sample_weight_train = np.where(
        X_train["event_code"] != "NONE", EVENT_WEIGHT, 1
    )

    X_val = df_val[NUM_FEATURES + CAT_FEATURES + ["date"]].copy()
    y_val = df_val[RESPONDER]

    glm, lgbm = load_models(X_train, y_train, sample_weight_train)

    return glm, lgbm, X_val, y_val


def get_pred_summary(
    glm: Pipeline,
    lgbm: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """
    Generate prediction summary DataFrame for validation set.

    Parameters
    ----------
    glm : Pipeline
        Fitted GLM pipeline.
    lgbm : Pipeline
        Fitted LGBM pipeline.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation responder.

    Returns
    -------
    pd.DataFrame
    """
    glm_y_pred = glm.predict(X_val)
    lgbm_y_pred = lgbm.predict(X_val)
    date = X_val["date"]
    X_val = X_val[NUM_FEATURES + CAT_FEATURES].copy()

    df_pred = pd.DataFrame(
        {
            "y_true": y_val,
            "glm_y_pred": glm_y_pred,
            "lgbm_y_pred": lgbm_y_pred,
            "baseline_y_pred": X_val["past_50m_span_ewm_vol"],
            "weight": 1,
            "date": date,
            "time_of_day": X_val["time_of_day"],
        }
    )

    return df_pred


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
