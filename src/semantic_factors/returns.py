"""Getting the price history a regression needs.

The package does not sell price data and does not pretend to. If you have returns, pass them
to ``regress(returns=...)`` as a wide frame and this module is never touched. If you do not,
it will pull adjusted closes from EODHD, which is where the ``CTVA.US`` ticker convention comes
from in the first place.
"""
import concurrent.futures as futures
import datetime as dt
import os
from typing import Dict, List, Optional, Sequence

import polars as pl
import requests

from semantic_factors import logs
from semantic_factors.errors import CredentialError, ReturnsUnavailableError

#: EODHD end-of-day price endpoint.
EODHD_URL = "https://eodhd.com/api/eod"

#: Environment variables consulted, in order, when no key is passed.
EODHD_KEY_VARIABLES = ("EODHD_API_KEY", "EODHD_API_TOKEN", "SEMANTIC_FACTORS_EODHD_KEY")

#: Concurrent downloads.
DOWNLOAD_WORKERS = 8


def download_returns(
    tickers: Sequence[str],
    start: dt.date,
    stop: dt.date,
    api_key: Optional[str] = None
) -> pl.DataFrame:
    """
    Download adjusted closes and convert them to simple daily returns.

    :param tickers: Symbols in EODHD form, such as "CTVA.US".
    :param start: First date wanted. One extra day is fetched so the first return is real.
    :param stop: Last date wanted.
    :param api_key: EODHD key, or None to read the environment.
    :return: Wide frame with a "date" column and one column per ticker.
    """
    if not tickers:
        raise ReturnsUnavailableError("at least one ticker is required")
    key = resolve_eodhd_key(api_key=api_key)
    logs.info(message=f"downloading closes for {len(tickers)} tickers from EODHD")
    frames: List[pl.DataFrame] = []
    failed: List[str] = []
    with requests.Session() as session:
        with futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            pending = {
                pool.submit(
                    download_closes,
                    session=session,
                    ticker=ticker,
                    start=start - dt.timedelta(days=7),
                    stop=stop,
                    api_key=key
                ): ticker
                for ticker in tickers
            }
            for task in futures.as_completed(pending):
                ticker = pending[task]
                frame = task.result()
                if frame is None or frame.is_empty():
                    failed.append(ticker)
                else:
                    frames.append(frame)
    if failed:
        logs.warn(message=f"no price history for {', '.join(sorted(failed))}")
    if not frames:
        raise ReturnsUnavailableError(
            f"EODHD returned no price history for any of {', '.join(tickers)}"
        )
    return to_returns(frames=frames)


def download_closes(
    session: requests.Session,
    ticker: str,
    start: dt.date,
    stop: dt.date,
    api_key: str
) -> Optional[pl.DataFrame]:
    """
    Download one ticker's adjusted closes.

    :param session: Reused HTTP session.
    :param ticker: Symbol in EODHD form.
    :param start: First date to request.
    :param stop: Last date to request.
    :param api_key: EODHD key.
    :return: Two-column frame of date and close, or None when nothing came back.
    """
    response = session.get(
        url=f"{EODHD_URL}/{ticker}",
        params={
            "api_token": api_key,
            "from": start.isoformat(),
            "to": stop.isoformat(),
            "period": "d",
            "fmt": "json"
        },
        timeout=(10, 120)
    )
    if response.status_code == 401 or response.status_code == 403:
        raise CredentialError("EODHD rejected the API key")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        return None
    return pl.DataFrame(
        data={
            "date": [dt.date.fromisoformat(row["date"]) for row in rows],
            ticker: [float(row["adjusted_close"]) for row in rows]
        }
    )


def to_returns(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """
    Join per-ticker close frames and difference them into simple daily returns.

    :param frames: One two-column frame per ticker, each with a date column.
    :return: Wide frame with a "date" column and one return column per ticker.
    """
    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.join(other=frame, on="date", how="full", coalesce=True)
    joined = joined.sort(by="date")
    columns = [name for name in joined.columns if name != "date"]
    return joined.select(
        pl.col("date"),
        *[(pl.col(name) / pl.col(name).shift(1) - 1.0).alias(name) for name in columns]
    ).slice(offset=1)


def returns_from_prices(prices: pl.DataFrame, date_column: str = "date") -> pl.DataFrame:
    """
    Convert a caller's own wide price frame into simple daily returns.

    :param prices: Wide frame with a date column and one price column per ticker.
    :param date_column: Name of the date column.
    :return: Wide frame with a "date" column and one return column per ticker.
    """
    columns = [name for name in prices.columns if name != date_column]
    if not columns:
        raise ReturnsUnavailableError("the price frame has no ticker columns")
    return prices.sort(by=date_column).select(
        pl.col(date_column).alias("date"),
        *[(pl.col(name) / pl.col(name).shift(1) - 1.0).alias(name) for name in columns]
    ).slice(offset=1)


def resolve_eodhd_key(api_key: Optional[str] = None) -> str:
    """
    Find the EODHD key, explicit argument first, then the environment.

    :param api_key: Explicit key, or None.
    :return: The key.
    """
    if api_key:
        return api_key.strip()
    for variable in EODHD_KEY_VARIABLES:
        value = os.environ.get(variable, "").strip()
        if value:
            return value
    raise CredentialError(
        "regress() needs price history. Either pass returns=<wide frame of daily returns> or "
        f"supply an EODHD key with eodhd_api_key=... or one of {', '.join(EODHD_KEY_VARIABLES)}."
    )


def describe_coverage(returns: pl.DataFrame) -> Dict[str, object]:
    """
    Summarise a returns frame, for diagnostics.

    :param returns: Wide returns frame.
    :return: Mapping of ticker to the number of observations it carries.
    """
    return {
        name: int(returns[name].drop_nulls().len())
        for name in returns.columns
        if name != "date"
    }
