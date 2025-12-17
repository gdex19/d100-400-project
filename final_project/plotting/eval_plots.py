import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import dalex as dx
from final_project.evaluation import evaluate_predictions


def plot_feature_relevance(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """
    Plot feature relevance for a fitted model.

    Parameters
    ----------
    model: Pipeline
        Fitted model.
    X: pd.DataFrame
        Validation features.
    y: pd.Series
        Validation responder.

    Returns
    -------
    None
    """
    explainer = dx.Explainer(
        model=model, data=X, y=y, model_type="regression", verbose=False
    )

    relevance = explainer.model_parts()
    relevance.plot()


def plot_pdps(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 5,
) -> None:
    """
    Plot PDPs for given model.

    Parameters
    ----------
    model: Pipeline
        Fitted model.
    X: pd.DataFrame
        Validation features.
    y: pd.Series
        Validation responder.
    n_top: int
        Number of top features to plot.

    Returns
    -------
    None
    """
    explainer = dx.Explainer(
        model=model, data=X, y=y, model_type="regression", verbose=False
    )

    rel = explainer.model_parts()

    imp = rel.result
    if imp is None:
        raise ValueError("dalex model_parts() returned no result")

    imp = imp[
        (imp["variable"] != "_baseline_") & (imp["variable"].isin(X.columns))
    ]
    score_col = imp.select_dtypes("number").columns[0]
    top = (
        imp.sort_values(score_col, ascending=False)
        .head(n_top)["variable"]
        .tolist()
    )

    num_set = set(X.select_dtypes(include="number").columns)
    num_feats = [f for f in top if f in num_set]
    cat_feats = [f for f in top if f not in num_set]

    prof_num = prof_cat = None
    if num_feats:
        prof_num = explainer.model_profile(
            variables=num_feats, variable_type="numerical"
        )
        prof_num.plot()
        plt.show()
    if cat_feats:
        prof_cat = explainer.model_profile(
            variables=cat_feats, variable_type="categorical"
        )
        prof_cat.plot()
        plt.show()


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


def plot_model_metrics(
    df_pred_raw: pd.DataFrame,
    df_pred_clip: pd.DataFrame,
) -> Figure | SubFigure | None:
    """
    Compare baseline, GLM, and LGBM on raw vs clipped data
    metrics on one bar chart.

    Note: this is essentially unedited ChatGPT code, as I couldn't get this
    working myself.

    Parameters
    ----------
    df_pred_raw : pd.DataFrame
        Predictions from raw data.
    df_pred_clip : pd.DataFrame
        Predictions from clipped data.
    """
    cols = {
        "Baseline": "baseline_y_pred",
        "GLM": "glm_y_pred",
        "LGBM": "lgbm_y_pred",
    }
    models = list(cols.keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(models)]

    raw = {
        m: evaluate_predictions(
            df_pred_raw["y_true"], df_pred_raw[c], df_pred_raw["weight"]
        )
        for m, c in cols.items()
    }
    clip = {
        m: evaluate_predictions(
            df_pred_clip["y_true"], df_pred_clip[c], df_pred_clip["weight"]
        )
        for m, c in cols.items()
    }

    metrics = list(next(iter(raw.values())).index)
    x = range(len(metrics))
    w = 0.12

    fig, ax = plt.subplots(figsize=(10, 5))

    for j, (m, color) in enumerate(zip(models, colors)):
        raw_vals = raw[m].reindex(metrics).to_numpy()
        clip_vals = clip[m].reindex(metrics).to_numpy()

        # positions: per model we have two bars (raw, clipped)
        raw_offset = (j * 2 + 0 - (len(models) * 2 - 1) / 2) * w
        clip_offset = (j * 2 + 1 - (len(models) * 2 - 1) / 2) * w

        ax.bar(
            [i + raw_offset for i in x],
            raw_vals,
            width=w,
            color=color,
            alpha=1.0,
        )
        ax.bar(
            [i + clip_offset for i in x],
            clip_vals,
            width=w,
            color=color,
            alpha=0.45,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_title(
        "Model comparison across metrics (lighter = Winsorized features)"
    )
    ax.set_ylabel("Metric value")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        handles=[Patch(facecolor=c, label=m) for m, c in zip(models, colors)],
        title="",
    )

    fig.tight_layout()
    return fig
