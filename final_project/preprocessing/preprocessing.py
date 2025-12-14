import pandas as pd
import numpy as np
import pandas_market_calendars as tcal
from pathlib import Path

DATA_RELEASE_FP = (
    Path(__file__).parent.parent.parent / "data" / "data_releases.csv"
)

RESPONDER = "future_30m_vol"

NUM_FEATURES = [
    # Returns
    "past_1m_ret",
    "past_5m_ret",
    "past_30m_ret",
    "past_60m_ret",
    "past_120m_ret",
    # Squared returns
    "past_1m_sq_ret",
    "past_5m_sq_ret",
    "past_30m_sq_ret",
    "past_60m_sq_ret",
    "past_120m_sq_ret",
    # Past realized vols
    "past_50m_vol",
    "past_1440m_vol",
    "past_10080m_vol",
    "past_43200m_vol",
]

CAT_FEATURES = ["event_code", "time_of_day", "is_trading_day"]


def add_dates_and_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dates and times for grouping and graphing.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with price data.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["minute"] = df["open_time"].str[14:16].astype(int)
    df["hour"] = df["open_time"].str[10:13].astype(int)
    df["date"] = df["open_time"].str[:10]
    df["time_of_day"] = (
        df["hour"]
        .astype(str)
        .str.zfill(2)
        .str.cat(df["minute"].astype(str).str.zfill(2), sep=":")
    )
    return df


def add_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trading day flag.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with price data.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    nyse = tcal.get_calendar("NYSE")
    start_date = df["open_time"].str[:10].min()
    end_date = df["open_time"].str[:10].max()
    sched = nyse.schedule(start_date=start_date, end_date=end_date)

    df["is_trading_day"] = (
        df["open_time"].str[:10].isin(sched.index.astype(str))
    )

    return df


def compute_future_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward-looking 30 minute realized volatility,
    based on 1 minute returns.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with Bitcoin price data.

    Returns
    -------
    pd.DataFrame
    """
    df = df.sort_values(by="open_time", ascending=True).copy()
    df["future_30m_vol"] = np.sqrt(
        df["past_1m_sq_ret"].rolling(window=30).mean().shift(-30)
    ) * np.sqrt(365 * 24 * 60)

    return df


def compute_past_volatility(
    df: pd.DataFrame, windows: list[int]
) -> pd.DataFrame:
    """
    Compute past window minute realized volatility,
    based on 1 minute returns.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with Bitcoin price data.

    windows: list[int]
        List of windows to compute vol for, e.g. [5, 10, 30] minutes
    Returns
    -------
    pd.DataFrame
    """
    df = df.sort_values(by="open_time", ascending=True).copy()
    for window in windows:
        df[f"past_{window}m_vol"] = np.sqrt(
            df["past_1m_sq_ret"].rolling(window=window).mean()
        ) * np.sqrt(365 * 24 * 60)

    return df


def compute_returns(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Compute past returns for each window in the list.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with Bitcoin price data.
    windows: list[int]
        List of windows to compute returns for, e.g. [5, 10, 30] minutes

    Returns
    -------
    pd.DataFrame
        New dataframe with returns calculated.
    """
    df = df.copy()
    for window in windows:
        df[f"past_{window}m_ret"] = df["open"] / df["open"].shift(window) - 1

    return df


def compute_squared_returns(
    df: pd.DataFrame, windows: list[int]
) -> pd.DataFrame:
    """
    Compute squared past returns for each window in the list.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with Bitcoin price data.
    windows: list[int]
        List of windows to compute returns for, e.g. [5, 10, 30] minutes

    Returns
    -------
    pd.DataFrame
        New dataframe with returns calculated.
    """
    df = df.copy()
    for window in windows:
        df[f"past_{window}m_sq_ret"] = (
            df["open"] / df["open"].shift(window) - 1
        ) ** 2

    return df


def compute_past_ewm_vols(df: pd.DataFrame, spans: list[int]) -> pd.DataFrame:
    """
    Compute past ewm vols.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with Bitcoin price data.
    spans: list[int]
        List of spans to compute ewms for, e.g. [5, 10, 30] minutes

    Returns
    -------
    pd.DataFrame
        New dataframe with ewms calculated.
    """
    df = df.copy()
    for span in spans:
        df[f"past_{span}_span_ewm_vol"] = np.sqrt(
            df["past_1m_sq_ret"].ewm(span=span).mean()
        ) * np.sqrt(365 * 24 * 60)

    return df


def add_data_releases(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add CPI, jobs, and fed releases.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with Bitcoin price data.

    Returns
    -------
    pd.DataFrame
        New dataframe with ewms calculated.
    """
    releases = pd.read_csv(DATA_RELEASE_FP)
    releases["open_time"] = releases["release_time"].astype(str)
    releases["event_code"] = releases["event_type"].map(
        {
            "Federal Reserve Rate Decision": "FED",
            "Nonfarm Payrolls": "JOB",
            "Consumer Price Index": "CPI",
        }
    )
    data = releases[["open_time", "event_code"]]
    df = pd.merge(left=df, right=data, on="open_time", how="left")
    df["event_code"] = df["event_code"].fillna("NONE")

    return df


def add_features_responder(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Raw Bitcoin price data.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with Bitcoin price data.

    Returns
    -------
    pd.DataFrame
        New dataframe with clipped features.
    """
    df = add_dates_and_times(df_raw)
    df = add_trading_days(df)
    df = compute_returns(df, [1, 5, 30, 60, 120])
    df = compute_squared_returns(df, [1, 5, 30, 60, 120])
    df = compute_squared_returns(df, [1, 5, 30, 60, 120])
    df = compute_future_volatility(df)

    # Add 50m, 1d, 7d, 30d past vol ewms
    df = compute_past_volatility(df, [50, 24 * 60, 24 * 60 * 7, 24 * 60 * 30])

    df = add_data_releases(df)

    return df


def winsorize_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip predictors for linear modeling, using thresholds
    from first half of 2024.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with features and responder.

    Returns
    -------
    pd.DataFrame
        New dataframe with all features and responders.
    """
    df = df.copy()

    # Training window for threshold estimation
    train_mask = df["date"] < "2024-07-01"
    df_train = df.loc[train_mask]

    low_q = 0.0005
    high_q = 0.9995

    returns = [
        c
        for c in NUM_FEATURES
        if c.endswith("_ret") and not c.endswith("_sq_ret")
    ]
    not_returns = [c for c in NUM_FEATURES if c not in returns]

    # Clip
    for col in returns:
        lo = df_train[col].quantile(low_q)
        hi = df_train[col].quantile(high_q)
        df[col] = df[col].clip(lower=lo, upper=hi)

    for col in not_returns:
        hi = df_train[col].quantile(high_q)
        df[col] = df[col].clip(upper=hi)

    return df


def write_data(df_final: pd.DataFrame, name: str) -> None:
    fp = Path(__file__).parent.parent.parent / "data" / f"{name}.pq"
    df_final.to_parquet(fp)
