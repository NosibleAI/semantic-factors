"""Measuring which companies are exposed to a factor.

The question a quant asks of a news factor is not "does it look interesting" but "who moves
when it moves". That is a regression of returns on the factor, and three details decide whether
the answer means anything.

**Changes, not levels.** A news index is persistent; a return is not. Regressing a return on a
level mostly measures the level's trend, so ``difference=True`` is the default and regresses on
the day-on-day change.

**Exposure, not prediction.** ``horizon=0``, the default, regresses the same day's return on
the same day's change. That is a loading: how much this name moves when the factor moves, which
is what a risk model wants. Setting ``horizon`` to one or more shifts the return forward
instead and asks a different, much harder question — whether the factor predicts anything.
Do not read a predictive claim into a contemporaneous beta.

**Overlapping windows.** A horizon beyond one day makes consecutive observations share returns,
which makes ordinary standard errors far too small. The t-statistics here use Newey-West with a
lag matched to the horizon.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from semantic_factors.errors import ConfigurationError

#: Observations a rolling window needs before a point-in-time beta is reported.
MIN_WINDOW_OBSERVATIONS = 60

#: Columns the per-ticker statistics frame carries.
STAT_COLUMNS = (
    "ticker",
    "beta",
    "t_stat",
    "r_squared",
    "observations",
    "correlation",
    "beta_latest",
    "beta_latest_date"
)


def regress_factor(
    daily: pl.DataFrame,
    returns: pl.DataFrame,
    tickers: Sequence[str],
    column: str = "value",
    horizon: int = 0,
    window: int = 252,
    min_observations: int = 120,
    difference: bool = True
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Regress every ticker's forward return on the factor and report its exposure.

    :param daily: Daily factor frame from SemanticFactor.compute.
    :param returns: Wide frame with a "date" column and one daily return column per ticker.
    :param tickers: Tickers to regress, in the order they should be reported.
    :param column: Factor column to regress against.
    :param horizon: 0 for a contemporaneous exposure, or the forward return horizon in trading
        days for a predictive fit.
    :param window: Length of the rolling window used for point-in-time loadings.
    :param min_observations: Observations a ticker needs before it is reported at all.
    :param difference: Regress on the change in the factor rather than its level.
    :return: Tuple of the per-ticker statistics frame and the long point-in-time loadings frame.
    """
    if horizon < 0:
        raise ConfigurationError(f"horizon cannot be negative, got {horizon}")
    if column not in daily.columns:
        raise ConfigurationError(f"the daily frame has no column named {column!r}")
    aligned = align(
        daily=daily,
        returns=returns,
        column=column,
        difference=difference
    )
    available = [name for name in tickers if name in aligned.columns]
    if not available:
        raise ConfigurationError(
            f"none of {', '.join(tickers)} appear in the returns frame, which has "
            f"{', '.join(name for name in returns.columns if name != 'date')}"
        )
    stats: List[dict] = []
    loadings: List[pl.DataFrame] = []
    for ticker in available:
        pair = aligned.select("date", "factor", pl.col(ticker).alias("ret")).drop_nulls()
        pair = forward_returns(
            frame=pair,
            horizon=horizon
        )
        if pair.height < min_observations:
            continue
        stats.append(
            full_sample_statistics(
                ticker=ticker,
                frame=pair,
                horizon=horizon
            )
        )
        rolling = rolling_betas(
            ticker=ticker,
            frame=pair,
            window=window
        )
        if not rolling.is_empty():
            loadings.append(rolling)

    if not stats:
        raise ConfigurationError(
            f"no ticker had at least {min_observations} overlapping observations with the factor"
        )
    loading_frame = (
        pl.concat(items=loadings).sort(by=["date", "ticker"])
        if loadings
        else pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8, "beta": pl.Float64})
    )
    stats_frame = attach_latest(
        stats=pl.DataFrame(data=stats),
        loadings=loading_frame
    )
    return stats_frame.sort(by="beta", descending=True), loading_frame


def align(
    daily: pl.DataFrame,
    returns: pl.DataFrame,
    column: str,
    difference: bool
) -> pl.DataFrame:
    """
    Join the factor onto the return calendar, carrying it forward across non-trading days.

    The factor exists every calendar day and returns only on trading days, so each trading day
    takes the most recent factor value at or before it.

    :param daily: Daily factor frame.
    :param returns: Wide returns frame.
    :param column: Factor column to use.
    :param difference: Difference the factor before joining.
    :return: Frame with date, factor and one column per ticker.
    """
    factor = daily.select("date", pl.col(column).alias("factor")).drop_nulls().sort(by="date")
    if difference:
        factor = factor.with_columns(pl.col("factor").diff().alias("factor")).drop_nulls()
    calendar = returns.sort(by="date")
    return calendar.join_asof(other=factor, on="date", strategy="backward")


def forward_returns(frame: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """
    Attach the return the regression explains, contemporaneous or forward.

    :param frame: Frame with date, factor and ret columns, sorted by date.
    :param horizon: 0 for the same day's return, or the forward horizon in trading days.
    :return: Frame with a "target" column, rows without a full horizon dropped.
    """
    if horizon == 0:
        return frame.with_columns(pl.col("ret").alias("target")).drop_nulls()
    if horizon == 1:
        return frame.with_columns(pl.col("ret").shift(-1).alias("target")).drop_nulls()
    compounded = (
        (pl.col("ret") + 1.0).log().rolling_sum(window_size=horizon).shift(-horizon).exp() - 1.0
    )
    return frame.with_columns(compounded.alias("target")).drop_nulls()


def full_sample_statistics(
    ticker: str,
    frame: pl.DataFrame,
    horizon: int
) -> dict:
    """
    Fit one ticker over the whole sample.

    :param ticker: Symbol being fitted.
    :param frame: Frame with factor and target columns.
    :param horizon: Return horizon, which sets the Newey-West lag.
    :return: Mapping of statistic name to value.
    """
    factor = frame["factor"].to_numpy().astype(np.float64)
    target = frame["target"].to_numpy().astype(np.float64)
    beta, intercept, residuals = fit(factor=factor, target=target)
    standard_error = newey_west_error(
        factor=factor,
        residuals=residuals,
        lags=max(0, horizon - 1)
    )
    explained = 1.0 - residuals.var() / target.var() if target.var() > 0 else 0.0
    correlation = float(np.corrcoef(factor, target)[0, 1]) if factor.std() > 0 else float("nan")
    return {
        "ticker": ticker,
        "beta": float(beta),
        "t_stat": float(beta / standard_error) if standard_error > 0 else float("nan"),
        "r_squared": float(explained),
        "observations": int(frame.height),
        "correlation": correlation,
        "intercept": float(intercept)
    }


def rolling_betas(
    ticker: str,
    frame: pl.DataFrame,
    window: int
) -> pl.DataFrame:
    """
    Fit the same regression in a trailing window ending on each date.

    :param ticker: Symbol being fitted.
    :param frame: Frame with date, factor and target columns.
    :param window: Window length in observations.
    :return: Long frame of date, ticker and beta.
    """
    factor = frame["factor"].to_numpy().astype(np.float64)
    target = frame["target"].to_numpy().astype(np.float64)
    dates = frame["date"].to_list()
    span = max(MIN_WINDOW_OBSERVATIONS, min(window, len(factor)))
    if len(factor) < span:
        return pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8, "beta": pl.Float64})
    betas: List[float] = []
    for end in range(span, len(factor) + 1):
        slice_factor = factor[end - span:end]
        slice_target = target[end - span:end]
        beta, _, _ = fit(factor=slice_factor, target=slice_target)
        betas.append(float(beta))
    return pl.DataFrame(
        data={
            "date": dates[span - 1:],
            "ticker": [ticker] * len(betas),
            "beta": betas
        }
    )


def fit(factor: np.ndarray, target: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Fit target on a constant and the factor by ordinary least squares.

    :param factor: Explanatory series.
    :param target: Dependent series.
    :return: Tuple of slope, intercept and residuals.
    """
    design = np.column_stack([np.ones_like(factor), factor])
    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    intercept, beta = float(solution[0]), float(solution[1])
    return beta, intercept, target - design @ solution


def newey_west_error(
    factor: np.ndarray,
    residuals: np.ndarray,
    lags: int
) -> float:
    """
    Compute a Newey-West standard error for the slope.

    Overlapping forward returns make consecutive residuals correlated. Ignoring that inflates
    every t-statistic, sometimes by a factor of three, which is how a news factor comes to look
    significant when it is not.

    :param factor: Explanatory series.
    :param residuals: Regression residuals.
    :param lags: Number of autocorrelation lags to correct for.
    :return: Standard error of the slope.
    """
    centred = factor - factor.mean()
    denominator = float(centred @ centred)
    if denominator <= 0:
        return float("nan")
    scores = centred * residuals
    variance = float(scores @ scores)
    for lag in range(1, min(lags, len(scores) - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(scores[lag:] @ scores[:-lag])
        variance += 2.0 * weight * covariance
    if variance <= 0:
        return float("nan")
    return float(np.sqrt(variance) / denominator)


def attach_latest(stats: pl.DataFrame, loadings: pl.DataFrame) -> pl.DataFrame:
    """
    Add each ticker's most recent point-in-time beta to its full-sample row.

    :param stats: Per-ticker statistics.
    :param loadings: Long frame of point-in-time betas.
    :return: The statistics frame with beta_latest and beta_latest_date columns.
    """
    if loadings.is_empty():
        return stats.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("beta_latest"),
            pl.lit(None, dtype=pl.Date).alias("beta_latest_date")
        ).select(STAT_COLUMNS)
    latest = (
        loadings.sort(by="date")
        .group_by("ticker")
        .agg(
            pl.col("beta").last().alias("beta_latest"),
            pl.col("date").last().alias("beta_latest_date")
        )
    )
    return stats.join(other=latest, on="ticker", how="left").select(STAT_COLUMNS)


def beta_matrix(loadings: pl.DataFrame) -> Optional[pl.DataFrame]:
    """
    Pivot the point-in-time loadings into one column per ticker.

    :param loadings: Long frame of date, ticker and beta.
    :return: Wide frame, or None when there are no loadings.
    """
    if loadings.is_empty():
        return None
    return loadings.pivot(on="ticker", index="date", values="beta").sort(by="date")
