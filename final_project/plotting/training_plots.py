import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from matplotlib.figure import Figure
import numpy as np


def plot_lasso_paths_numeric(
    coef_path: pd.DataFrame,
    max_features: Optional[int] = None,
) -> Figure:
    """
    Plot LASSO coefficient paths for numeric features.

    Parameters
    ----------
    coef_path : pd.DataFrame
        Coefficient paths indexed by alpha, as returned by
        lasso path function.

    max_features : int, optional
        If provided, plot only the top `max_features` features
        ranked by absolute coefficient size at the smallest alpha.


    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    # Keep numeric features only
    coef_num = coef_path.loc[:, coef_path.columns.str.startswith("num__")]

    # Select features at smallest alpha
    alpha_min = coef_num.index.min()
    importance = coef_num.loc[alpha_min].abs().sort_values(ascending=False)

    if max_features is not None:
        keep = importance.head(max_features).index
        coef_num = coef_num[keep]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, coef_num.shape[1]))

    for color, col in zip(colors, coef_num.columns):
        ax.plot(
            coef_num.index,
            coef_num[col],
            label=col.replace("num__", ""),
            color=color,
        )

    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Alpha (L1 regularization strength)")
    ax.set_ylabel("Coefficient")
    ax.set_title("LASSO coefficient paths (numeric features only)")
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False,
    )

    return fig
