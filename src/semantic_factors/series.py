"""Turning daily sufficient statistics into a factor time series.

Every day contributes four numbers: how much weight the matching events carried (``raw``), how
many events were eligible at all (``volume``), the weighted sum of their pole scores, and the
weight that sum was taken over. Everything a caller reads is derived from those.

The one decision that matters here is stabilisation. NOSIBLE World grew from a few hundred
events a day in 2015 to several thousand today, so a raw count rises whether or not the world
changed. Dividing by the same day's event count removes that, but it also divides out
genuine days when everything is about one story. Dividing by a trailing average of the event
count removes the growth and keeps the spike, which is what ``stabilize=True`` does and what
every published index of this kind does.
"""
import datetime as dt
from typing import Dict, List, Optional, Sequence

import polars as pl

#: Length of the trailing window used to stabilise against corpus growth, in days.
STABILISE_WINDOW_DAYS = 365

#: Days that must be present in the window before a stabilised value is emitted. Below this the
#: baseline is too noisy to divide by, so the series starts with nulls rather than nonsense.
STABILISE_MIN_DAYS = 90

#: Columns every daily frame carries, in order.
DAILY_COLUMNS = (
    "date",
    "value",
    "polarity",
    "n_events",
    "n_eligible",
    "raw",
    "volume",
    "baseline",
    "share",
    "polarity_num",
    "polarity_weight",
)


def daily_frame(
    days: Sequence[Dict[str, object]],
    start: dt.date,
    stop: dt.date,
    stabilize: bool = True,
    window_days: int = STABILISE_WINDOW_DAYS,
    min_days: int = STABILISE_MIN_DAYS
) -> pl.DataFrame:
    """
    Assemble the daily factor series from per-day sufficient statistics.

    Days with no eligible events at all are dropped rather than zero-filled: the corpus has
    genuine gaps, and a zero would be read as "nothing happened" when it means "nothing was
    collected". Days that were covered but produced no match are kept, with value zero.

    :param days: One mapping per day with date, raw, volume, n_events, polarity_num and
        polarity_weight keys.
    :param start: First date the caller asked for.
    :param stop: Last date the caller asked for.
    :param stabilize: Divide by a trailing baseline instead of the same day's volume.
    :param window_days: Length of the trailing baseline window.
    :param min_days: Minimum observations before a stabilised value is emitted.
    :return: Daily frame with the columns named in DAILY_COLUMNS.
    """
    rows = [row for row in days if start <= parse_date(value=row["date"]) <= stop]
    if not rows:
        return empty_daily()
    frame = (
        pl.DataFrame(
            data={
                "date": [parse_date(value=row["date"]) for row in rows],
                "raw": [number(value=row.get("raw")) for row in rows],
                "volume": [number(value=row.get("volume")) for row in rows],
                "n_events": [int(number(value=row.get("n_events"))) for row in rows],
                "n_eligible": [
                    int(number(value=row.get("n_eligible") or row.get("volume"))) for row in rows
                ],
                "polarity_num": [number(value=row.get("polarity_num")) for row in rows],
                "polarity_weight": [number(value=row.get("polarity_weight")) for row in rows]
            }
        )
        .filter(pl.col("volume") > 0)
        .sort(by="date")
    )
    if frame.is_empty():
        return empty_daily()

    frame = frame.with_columns(
        pl.col("volume")
        .rolling_mean_by(by="date", window_size=f"{window_days}d", min_samples=min_days)
        .alias("baseline")
    )
    denominator = pl.col("baseline") if stabilize else pl.col("volume")
    return frame.with_columns(
        (pl.col("raw") / pl.col("volume")).alias("share"),
        (pl.col("raw") / denominator).alias("value"),
        pl.when(pl.col("polarity_weight") > 0)
        .then(pl.col("polarity_num") / pl.col("polarity_weight"))
        .otherwise(None)
        .alias("polarity")
    ).select(DAILY_COLUMNS)


def empty_daily() -> pl.DataFrame:
    """
    Build the empty daily frame, so a factor with no matches still has a usable schema.

    :return: Zero-row frame with the DAILY_COLUMNS schema.
    """
    return pl.DataFrame(
        schema={
            "date": pl.Date,
            "value": pl.Float64,
            "polarity": pl.Float64,
            "n_events": pl.Int64,
            "n_eligible": pl.Int64,
            "raw": pl.Float64,
            "volume": pl.Float64,
            "baseline": pl.Float64,
            "share": pl.Float64,
            "polarity_num": pl.Float64,
            "polarity_weight": pl.Float64,
        }
    )


def resample(daily: pl.DataFrame, every: str = "1mo") -> pl.DataFrame:
    """
    Aggregate a daily factor series to a coarser calendar frequency.

    Sums are re-summed and the ratios rebuilt from those sums, rather than averaging daily
    ratios: a month is one long day, not a mean of short ones, and averaging ratios silently
    over-weights quiet days.

    :param daily: Frame from daily_frame.
    :param every: Polars duration string, such as "1w", "1mo" or "1q".
    :return: One row per period, keyed by the period start in a "period" column.
    """
    if daily.is_empty():
        return daily.rename({"date": "period"})
    if "polarity_num" not in daily.columns or "polarity_weight" not in daily.columns:
        # Backward compatibility for frames created before daily sufficient statistics were
        # retained. Library-generated frames always take the exact branch below.
        daily = daily.with_columns(
            (pl.col("polarity") * pl.col("raw")).alias("polarity_num"),
            pl.when(pl.col("polarity").is_not_null())
            .then(pl.col("raw"))
            .otherwise(0.0)
            .alias("polarity_weight"),
        )
    grouped = (
        daily.group_by(pl.col("date").dt.truncate(every=every).alias("period"))
        .agg(
            pl.col("raw").sum(),
            pl.col("volume").sum(),
            pl.col("baseline").mean(),
            pl.col("n_events").sum(),
            pl.col("n_eligible").sum(),
            pl.col("polarity_num").sum(),
            pl.col("polarity_weight").sum(),
            pl.col("value").mean().alias("value_mean")
        )
        .sort(by="period")
    )
    return grouped.with_columns(
        (pl.col("raw") / pl.col("volume")).alias("share"),
        pl.col("value_mean").alias("value"),
        pl.when(pl.col("polarity_weight") > 0)
        .then(pl.col("polarity_num") / pl.col("polarity_weight"))
        .otherwise(None)
        .alias("polarity")
    ).select(
        "period",
        "value",
        "polarity",
        "n_events",
        "n_eligible",
        "raw",
        "volume",
        "baseline",
        "share",
        "polarity_num",
        "polarity_weight",
    )


def standardise(daily: pl.DataFrame, column: str = "value") -> pl.DataFrame:
    """
    Add a full-sample z-score of one column, for comparing factors on one pair of axes.

    :param daily: Frame from daily_frame.
    :param column: Column to standardise.
    :return: The frame with a "<column>_z" column added.
    """
    if daily.is_empty():
        return daily.with_columns(pl.lit(None, dtype=pl.Float64).alias(f"{column}_z"))
    mean = number(value=daily[column].mean())
    deviation = number(value=daily[column].std())
    # A constant series has a standard deviation of floating-point dust, not zero, and dividing
    # by dust turns rounding error into a z-score of order one.
    scale = deviation if abs(deviation) > 1e-12 else 1.0
    return daily.with_columns(((pl.col(column) - mean) / scale).alias(f"{column}_z"))


def rolling_percentile(
    daily: pl.DataFrame,
    column: str = "value",
    window_days: int = STABILISE_WINDOW_DAYS * 3
) -> pl.DataFrame:
    """
    Add the trailing percentile rank of one column, the form a risk desk reads a level in.

    :param daily: Frame from daily_frame.
    :param column: Column to rank.
    :param window_days: Length of the trailing window, in days.
    :return: The frame with a "<column>_pct" column in [0, 1].
    """
    if daily.is_empty():
        return daily.with_columns(pl.lit(None, dtype=pl.Float64).alias(f"{column}_pct"))
    values: List[Optional[float]] = daily[column].to_list()
    dates: List[dt.date] = daily["date"].to_list()
    ranks: List[Optional[float]] = []
    window: List[float] = []
    cutoff_index = 0
    for index, (today, value) in enumerate(zip(dates, values, strict=True)):
        while dates[cutoff_index] < today - dt.timedelta(days=window_days):
            cutoff_index += 1
        window = [item for item in values[cutoff_index:index + 1] if item is not None]
        if value is None or len(window) < STABILISE_MIN_DAYS:
            ranks.append(None)
        else:
            level = float(value)
            ranks.append(sum(1 for item in window if item <= level) / len(window))
    return daily.with_columns(pl.Series(name=f"{column}_pct", values=ranks, dtype=pl.Float64))


def number(value: object) -> float:
    """
    Coerce one field of a day record to a float, treating null and nonsense as zero.

    :param value: Field value from a day record.
    :return: Finite float.
    """
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"cannot read a number from {value!r}")


def parse_date(value: object) -> dt.date:
    """
    Coerce a date field to a date, accepting the ISO strings the API returns.

    :param value: A date, a datetime, or a "YYYY-MM-DD" string.
    :return: The date.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value[:10])
    raise TypeError(f"cannot read a date from {value!r}")
