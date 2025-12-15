import pandas as pd
import numpy as np
from typing import Tuple


def split_data(
    df: pd.DataFrame, train_val_test_split: list[float]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test/val. Only retain every 30 minutes to
    have independent responders.

    Parameters
    ----------
    df: pd.DataFrame
        Prepared bitcoin dataframe.
    train_test_val_split: list[float]
        Percentage of data in each split. Must sum to 1.

    Returns
    -------
    pd.DataFrame
    """
    if sum(train_val_test_split) != 1:
        raise Exception("Invalid distribution, must sum to 1")
    percents = np.cumsum(train_val_test_split)

    df = df.sort_values(by="open_time", ascending=True)
    df_ind = (
        df[(df["minute"] == 0) | (df["minute"] == 30)]
        .reset_index(drop=True)
        .dropna(axis=0)
    )

    len = df_ind.shape[0]

    test_idx = int(percents[0] * len) + 1
    val_idx = int(percents[1] * len) + 1

    df_train = df_ind.iloc[:test_idx]
    df_test = df_ind.iloc[test_idx:val_idx]
    df_val = df_ind.iloc[val_idx:]

    return (df_train, df_test, df_val)
