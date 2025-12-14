import pandas as pd
import numpy as np


def fill_no_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in rows in trading hours without trades.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with price data.

    Returns
    -------
    pd.DataFrame
    """
    df = df.sort_values(by="open_time", ascending=True).copy()
    # TODO!!!!
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward-looking 30 minute realized volatility,
    based on 1 minute returns. Also add minutely returns.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with Bitcoin price data.

    Returns
    -------
    pd.DataFrame
    """
    df = df.sort_values(by="open_time", ascending=True).copy()
    df["past_1m_ret"] = df["open"].pct_change()
    df["past_1m_sq_ret"] = df["past_1m_ret"] ** 2
    df["future_30m_vol"] = np.sqrt(
        df["past_1m_sq_ret"].rolling(window=30).mean().shift(-30)
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
