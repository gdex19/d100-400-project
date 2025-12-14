import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure, SubFigure


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
    df["minute"] = df["open_time"].str[14:16].astype(int)
    df["hour"] = df["open_time"].str[10:13].astype(int)
    half_hours = df[(df["minute"] == 30) | (df["minute"] == 0)].copy()

    half_hours["time"] = (
        half_hours["hour"].astype(str).str.zfill(2)
        + ":"
        + half_hours["minute"].astype(str).str.zfill(2)
    )

    half_hours = half_hours.sort_values(by="time", ascending=True)

    ax = sns.lineplot(
        data=half_hours,
        x="time",
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
    df_day = df[df["open_time"].str[0:10] == date].iloc[::freq]
    ax = sns.lineplot(data=df_day, x="open_time", y=col)

    ax.tick_params(axis="x", rotation=45)

    for label in ax.get_xticklabels()[::120]:
        label.set_visible(False)

    return ax.get_figure()
