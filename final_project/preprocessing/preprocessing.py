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
    "past_30m_quote_volume",
    "past_30m_trades",
]

CAT_FEATURES = [
    "event_code",
    "time_of_day",
    "is_us_trading_day",
    "is_eu_trading_day",
    "is_uk_trading_day",
    "is_jp_trading_day",
    "is_cn_trading_day",
    "is_hk_trading_day",
    "is_us_dst",
    "is_eu_dst",
]


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
    Add trading-day features for
    US, UK, Europe, and Asia markets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with open_time column.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    dates = df["date"]

    calendars = {
        "us": "NYSE",
        "uk": "LSE",
        "eu": "XETR",
        "jp": "JPX",
        "cn": "SSE",
        "hk": "HKEX",
    }

    for key, cal in calendars.items():
        cal_obj = tcal.get_calendar(cal)
        sched = cal_obj.schedule(
            start_date=min(dates),
            end_date=max(dates),
        )
        df[f"is_{key}_trading_day"] = dates.isin(sched.index.astype(str))

    return df


def add_dst(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daylight savings flag for
    US and Europe

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with open_time column.

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()
    dates = df["date"]

    # --- DST flags ---
    us_dst = ((dates >= "2024-03-10") & (dates <= "2024-11-03")) | (
        (dates >= "2025-03-09") & (dates <= "2025-11-02")
    )

    eu_dst = ((dates >= "2024-03-31") & (dates <= "2024-10-27")) | (
        (dates >= "2025-03-30") & (dates <= "2025-10-26")
    )

    df["is_us_dst"] = us_dst
    df["is_eu_dst"] = eu_dst

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


def add_lagged_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add last 30 minutes cumulative number of trades and quote volume.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with Bitcoin price data.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["past_30m_trades"] = df["number_of_trades"].rolling(30).sum().shift(1)
    df["past_30m_quote_volume"] = (
        df["quote_asset_volume"].rolling(30).sum().shift(1)
    )

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
    df = add_dst(df)
    df = compute_returns(df, [1, 5, 30, 60, 120])
    df = compute_squared_returns(df, [1, 5, 30, 60, 120])
    df = compute_future_volatility(df)
    df = add_lagged_metadata(df)

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
    """
    Write prepared data to disk as parquet

    Parameters
    ----------
    df_final: pd.DataFrame
        DataFrame with features and responder.

    Returns
    -------
    None
    """
    fp = Path(__file__).parent.parent.parent / "data" / f"{name}.pq"
    df_final.to_parquet(fp)
