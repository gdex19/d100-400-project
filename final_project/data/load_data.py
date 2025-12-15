import pandas as pd
import click
import requests
from datetime import datetime
from datetime import timezone
from pathlib import Path
import typing

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]
POLYGON_URL = "https://api.polygon.io/v2/aggs/ticker"


def unix_time_from_date(date: str) -> int:
    """
    Convert string date to unix time.

    Parameters
    ----------
    date: str
        date, as 'YYYY/MM/DD'
    Returns
    -------
    int
        Unix time in ms.
    """
    date_time = f"{date} 00:00:00"
    dt = datetime.strptime(date_time, "%Y/%m/%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )

    time_ms = int(dt.timestamp() * 1000)
    return time_ms


def query_binance_ticks(symbol: str, start_date: str) -> pd.DataFrame:
    """
    Query historical data from Binance for given symbol.

    Parameters
    ----------
    symbol : str
        Symbol for security of interest.
    start_date : str
        Start date, as 'YYYY/MM/DD'
    Returns
    -------
    pd.DataFrame
    """
    start_time = unix_time_from_date(start_date)
    limit = 1000

    all_rows = []
    while True:
        params: dict[str, typing.Any] = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": start_time,
            "limit": limit,
        }

        response = requests.get(BINANCE_URL, params=params)
        if response.status_code != 200:
            raise Exception(f"Binance query failed: {response.text}")

        batch = response.json()
        if not batch:
            break

        all_rows.extend(batch)

        last_close = batch[-1][6]
        start_time = last_close + 1

        if len(batch) < limit:
            break

    df = pd.DataFrame(all_rows, columns=BINANCE_KLINE_COLS)

    # Enforce correct types
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Convert times to lexographically sorted strings
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["open_time"] = df["open_time"].dt.tz_convert("America/New_York")
    df["close_time"] = df["close_time"].dt.tz_convert("America/New_York")
    df["open_time"] = df["open_time"].dt.strftime("%Y-%m-%d %H:%M:%S ET")
    df["close_time"] = df["close_time"].dt.strftime("%Y-%m-%d %H:%M:%S ET")

    return df


def query_polygon_ticks(
    symbol: str,
    start_date: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Query historical 1-minute data from Polygon for a given symbol.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g., "SPY", "QQQ", "VIX".
    api_key : str
        Polygon API key.
    start_date : str
        Start date in 'YYYY/MM/DD'.

    Returns
    -------
    pd.DataFrame
        Query results from start date until now.
    """
    end_date = datetime.now(timezone.utc).strftime("%Y/%m/%d")

    start_clean = start_date.replace("/", "-")
    end_clean = end_date.replace("/", "-")

    url = (
        f"{POLYGON_URL}/{symbol}/range/1/minute/" f"{start_clean}/{end_clean}"
    )

    params: dict[str, str] = {"apiKey": api_key}

    all_results = []

    # Loop gets paginated data
    while True:
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code != 200:
            raise Exception(f"Polygon query failed: {data}")

        results = data.get("results", [])
        all_results.extend(results)

        cursor = data.get("next_url")
        if not cursor:
            break

        url = cursor

    df = pd.DataFrame(all_results)

    df = df.rename(
        columns={
            "t": "open_time",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "num_trades",
        }
    )

    # Convert timestamp to lexicographic string
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["open_time"] = df["open_time"].dt.tz_convert("America/New_York")
    df["open_time"] = df["open_time"].dt.strftime("%Y-%m-%d %H:%M:%S ET")

    numeric_cols = ["open", "high", "low", "close", "volume", "vwap"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def read_data(symbol: str) -> pd.DataFrame:
    """
    Read the parquet into a pandas DataFrame.

    Parameters
    ----------
    symbol: str
        Parquet of interest.

    Returns
    -------
    pd.DataFrame
    """
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data"
    fp = data_path / f"{symbol}.pq"

    if not fp.is_file():
        raise Exception("Data not found")

    return pd.read_parquet(fp)


def load_data(start_date: str, api_key: str, polygon: bool) -> None:
    """
    Download and save historical minute-level data for BTC (Binance),
    SPY (Polygon), and QQQ (Polygon), starting from a specified date.

    Parameters
    ----------
    start_date : str
        The start date for all data queries, formatted as "YYYY/MM/DD"

    api_key : str
        Polygon API key. Required for querying SPY and QQQ data via Polygon.

    polygon : bool
        True if querying polygon data too.
    """

    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    binance_df = query_binance_ticks("BTCUSDT", start_date)
    binance_path = data_path / "btc.pq"
    binance_df.to_parquet(binance_path)

    if polygon:
        spy_df = query_polygon_ticks("SPY", start_date, api_key)
        spy_path = data_path / "spy.pq"
        spy_df.to_parquet(spy_path)

        qqq_df = query_polygon_ticks("QQQ", start_date, api_key)
        qqq_path = data_path / "qqq.pq"
        qqq_df.to_parquet(qqq_path)

    print(f"Data has been downloaded to {data_path}")


@click.command()  # type: ignore
@click.option(  # type: ignore
    "--start_date",
    required=True,
    help='Start date, in format "YYYY/MM/DD"',
)
@click.option("--api_key", help="Polygon API Key")  # type: ignore
@click.option(
    "--polygon", is_flag=True, help="Include Polygon data"
)  # type: ignore
def main(start_date: str, api_key: str, polygon: bool) -> None:
    if polygon and api_key is None:
        raise Exception("Requested polygon data without API key")
    load_data(start_date, api_key, polygon)


if __name__ == "__main__":
    main()
