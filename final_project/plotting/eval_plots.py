import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def plot_pred_vs_true(
    df_pred: pd.DataFrame,
    model: str,
    log: bool = False,
) -> Figure | SubFigure | None:
    """
    Plot predicted vs true values regplot w/ r2.

    Parameters
    ----------
    df_pred : pd.DataFrame
        DataFrame containing y_true and model predictions.
    model : str
        Either "glm" or "lgbm".
    log : bool, default False
        Whether to use log-log scale.

    Returns
    -------
    Figure | SubFigure | None
    """
    if model not in {"glm", "lgbm", "baseline"}:
        raise ValueError("model must be 'glm' or 'lgbm' or 'baseline'")

    y_pred_col = f"{model}_y_pred"
    df_plot = df_pred.copy()

    if log:
        x = np.log(df_plot["y_true"].to_numpy())
        y = np.log(df_plot[y_pred_col].to_numpy())
    else:
        x = df_plot["y_true"].to_numpy()
        y = df_plot[y_pred_col].to_numpy()

    X = x.reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    r2 = float(lr.score(X, y))

    x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_line = lr.predict(x_line)

    x_raw = df_plot["y_true"].to_numpy()
    y_raw = df_plot[y_pred_col].to_numpy()

    fig, ax = plt.subplots()
    ax.scatter(x_raw, y_raw, alpha=0.4)

    if log:
        ax.plot(np.exp(x_line.ravel()), np.exp(y_line), linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.plot(x_line.ravel(), y_line, linewidth=2)

    title = (
        f"{model.upper()} true vs predicted future_30m_vol | "
        f"log={log} | $R^2$={r2:.3f}"
    )
    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    return fig


def plot_day_predictions(
    df_pred: pd.DataFrame,
    date: str,
) -> Figure | SubFigure | None:
    """
    Plot both models' predictions, and baseline, for a single day.

    Parameters
    ----------
    df_pred : pd.DataFrame
        DataFrame that contains at least: date, time_of_day, and predictions.
    date : str
        Date to plot (YYYY-MM-DD).

    Returns
    -------
    Figure | SubFigure | None
    """
    glm_col = "glm_y_pred"
    lgbm_col = "lgbm_y_pred"
    baseline_col = "baseline_y_pred"

    df_day = df_pred[df_pred["date"] == date].copy()
    if df_day.empty:
        return None

    # Keep time in order (string times sort fine as HH:MM)
    df_day = df_day.sort_values("time_of_day")

    fig, ax = plt.subplots()
    ax.plot(df_day["time_of_day"], df_day[glm_col], label="GLM")
    ax.plot(df_day["time_of_day"], df_day[lgbm_col], label="LGBM")
    ax.plot(
        df_day["time_of_day"],
        df_day[baseline_col],
        label="baseline (past 50m ewm vol)",
    )
    ax.plot(df_day["time_of_day"], df_day["y_true"], label="true")

    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"Predictions + True Values on {date}")
    ax.set_xlabel("time_of_day")
    ax.set_ylabel("predicted future_30m_vol")
    ax.legend()

    for label in ax.get_xticklabels()[::2]:
        label.set_visible(False)

    fig.tight_layout()
    return fig
