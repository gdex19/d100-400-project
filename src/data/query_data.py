import pandas as pd
import click
import requests
from datetime import datetime
from datetime import timezone
from pathlib import Path

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
    params: dict[str, str] = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": str(start_time),
    }

    response = requests.get(BINANCE_URL, params=params)

    # Make sure response is valid
    if response.status_code != 200:
        raise Exception("Binance query failed")

    df = pd.DataFrame(response.json(), columns=BINANCE_KLINE_COLS)

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
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df["open_time"] = df["open_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["close_time"] = df["close_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


def query_polygon_ticks(
    symbol: str,
    start_date: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Query historical 1-minute OHLCV data from Polygon for a given symbol.

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
        Minute-resolution OHLCV data with lexographically sortable timestamps.
    """
    end_date = datetime.now(timezone.utc).strftime("%Y/%m/%d")

    start_clean = start_date.replace("/", "-")
    end_clean = end_date.replace("/", "-")

    url = (
        f"{POLYGON_URL}/{symbol}/range/1/minute/" f"{start_clean}/{end_clean}"
    )

    params: dict[str, str] = {"apiKey": api_key}

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Polygon query failed: {response.text}")

    data = response.json()

    if "results" not in data:
        raise Exception("Bad request to Polygon")

    df = pd.DataFrame(data["results"])

    # Rename columns
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

    # Convert timestamp to datetime, sorted, lexicographic string
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["open_time"] = df["open_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Enforce dtype consistency
    numeric_cols = ["open", "high", "low", "close", "volume", "vwap"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def download_data(start_date: str, api_key: str) -> None:
    """
    Download and save historical minute-level data for BTC (Binance),
    SPY (Polygon), and QQQ (Polygon), starting from a specified date.

    Parameters
    ----------
    start_date : str
        The start date for all data queries, formatted as "YYYY/MM/DD"

    api_key : str
        Polygon API key. Required for querying SPY and QQQ data via Polygon.
    """

    root_path = Path(__file__).parent.parent
    data_path = root_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    binance_df = query_binance_ticks("BTCUSDT", start_date)
    binance_path = data_path / "btc.pq"
    binance_df.to_parquet(binance_path)

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
@click.option(
    "--api-key", "-k", required=True, help="Polygon API Key"
)  # type: ignore
def main(start_date: str, api_key: str) -> None:
    download_data(start_date, api_key)


if __name__ == "__main__":
    main()
