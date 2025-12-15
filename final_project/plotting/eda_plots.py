import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure, SubFigure
import numpy as np

MARKET_TIMES_ET = {
    "Asia Open": "20:00",
    "Europe Open": "03:00",
    "US Pre-Mkt": "04:00",
    "US Open": "09:30",
    "US Close": "16:00",
}  # Not always right due to daylight savings, but aligns most of the time.


def plot_hourly_averages(df: pd.DataFrame) -> Figure | SubFigure | None:
    """
    Plot mean forward 30-minute volatility by half-hour time of day.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    Returns
    -------
    Figure | SubFigure | None
        The matplotlib figure for the plot.
    """
    df = df.copy()

    half_hours = df[(df["minute"] == 30) | (df["minute"] == 0)].copy()
    half_hours = half_hours.sort_values(by="time_of_day", ascending=True)

    ax = sns.lineplot(
        data=half_hours,
        x="time_of_day",
        y="future_30m_vol",
        estimator="mean",
        errorbar="sd",
    )

    ax.tick_params(axis="x", rotation=45)

    for label in ax.get_xticklabels()[::2]:
        label.set_visible(False)

    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Average 30m Forward Volatility")
    ax.set_title("30m Volatility Means Throughout Trading Day")

    # Add market open/closes
    times = list(half_hours["time_of_day"].unique())

    for time_label, t in MARKET_TIMES_ET.items():
        if t in times:
            x = times.index(t)
            ax.axvline(x=x, linestyle="--", alpha=0.6)
            ax.text(
                x,
                ax.get_ylim()[1],
                time_label,
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=9,
                alpha=0.8,
            )

    return ax.get_figure()


def plot_day(
    df: pd.DataFrame, date: str, col: str, freq: int
) -> Figure | SubFigure | None:
    """
    Plot a single day's time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    date : str
        Date to plot (YYYY-MM-DD).
    col : str
        Column to plot.
    freq : int
        Subsampling frequency.

    Returns
    -------
    Figure | SubFigure | None
        The matplotlib figure for the plot.
    """
    df_day = df[df["date"] == date].iloc[::freq].copy()
    ax = sns.lineplot(data=df_day, x="time_of_day", y=col)

    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{col} on {date}")

    for label in ax.get_xticklabels()[::2]:
        label.set_visible(False)

    return ax.get_figure()


def calculate_ewm_mse(df: pd.DataFrame, span: int) -> float:
    """
    Calculate squared return ewm vol, use as a baseline predictor.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price data.
    span: int
        Span of ewm.

    Returns
    -------
    float
        MSE of ewm as baseline predictor.
    """
    df = df.copy()
    df["pred"] = np.sqrt(df["past_1m_sq_ret"].ewm(span=span).mean()) * np.sqrt(
        365 * 24 * 60
    )

    df = df[df["pred"].notna() & df["future_30m_vol"].notna()]

    mse = ((df["future_30m_vol"] - df["pred"]) ** 2).mean()

    return mse


def calculate_equal_mse(df: pd.DataFrame, span: int) -> float:
    """
    Calculate squared return mean vol, use as a baseline predictor.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price data.
    span: int
        Span of ewm.

    Returns
    -------
    float
        MSE of ewm as baseline predictor.
    """
    df = df.copy()
    df["pred"] = np.sqrt(
        df["past_1m_sq_ret"].rolling(window=span).mean()
    ) * np.sqrt(365 * 24 * 60)

    df = df[df["pred"].notna() & df["future_30m_vol"].notna()]

    mse = ((df["future_30m_vol"] - df["pred"]) ** 2).mean()

    return mse


def plot_mses(df: pd.DataFrame, max_span: int) -> Figure | SubFigure | None:
    """
    Graph mses of various backward-looking vols to predict future vol.

    Parameters
    ----------
    df : pd.DataFrame
        Bitcoin price data.
    max_span: int
        Max span of ewm.

    Returns
    -------
    Figure | SubFigure | None
        The matplotlib figure for the plot.
    """
    spans = np.arange(1, max_span + 1)

    ewm_mses = []
    equal_mses = []

    for span in spans:
        ewm_mses.append(calculate_ewm_mse(df, int(span)))
        equal_mses.append(calculate_equal_mse(df, int(span)))

    # Best spans
    best_ewm_span = spans[int(np.argmin(ewm_mses))]
    best_equal_span = spans[int(np.argmin(equal_mses))]

    ax = sns.lineplot(x=spans, y=ewm_mses, label="EWM volatility")
    sns.lineplot(
        x=spans, y=equal_mses, label="Equal-weight rolling volatility", ax=ax
    )

    ax.axvline(
        best_ewm_span,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Best EWM span = {best_ewm_span}",
    )
    ax.axvline(
        best_equal_span,
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label=f"Best rolling span = {best_equal_span}",
    )

    ax.set_xlabel("Lookback window (minutes)")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Baseline volatility predictors for 30m forward volatility")
    ax.legend()

    return ax.get_figure()
