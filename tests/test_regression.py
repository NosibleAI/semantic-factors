"""Measuring which companies are exposed to a factor."""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from semantic_factors import regression, series
from semantic_factors.errors import ConfigurationError

DAYS = 900
START = dt.date(2020, 1, 1)


def synthetic(beta: float, noise: float = 0.01, seed: int = 7) -> tuple:
    """
    Build a factor and a return series with a known exposure between them.

    :param beta: True sensitivity of the return to the change in the factor.
    :param noise: Standard deviation of the idiosyncratic return.
    :param seed: Random seed.
    :return: Tuple of the daily factor frame and the wide returns frame.
    """
    rng = np.random.default_rng(seed)
    dates = [START + dt.timedelta(days=step) for step in range(DAYS)]
    level = np.cumsum(rng.normal(scale=0.01, size=DAYS)) + 1.0
    change = np.diff(level, prepend=level[0])
    returns = beta * change + rng.normal(scale=noise, size=DAYS)
    daily = pl.DataFrame(
        data={
            "date": dates,
            "value": level,
            "polarity": [None] * DAYS,
            "n_events": [1] * DAYS,
            "n_eligible": [1] * DAYS,
            "raw": level,
            "volume": [1.0] * DAYS,
            "baseline": [1.0] * DAYS,
            "share": level
        }
    ).select(series.DAILY_COLUMNS)
    return daily, pl.DataFrame(data={"date": dates, "ACME.US": returns})


def test_a_known_exposure_is_recovered() -> None:
    """
    Require the regression to recover a planted beta.

    :return: None.
    """
    daily, returns = synthetic(beta=2.5)

    stats, _ = regression.regress_factor(
        daily=daily,
        returns=returns,
        tickers=["ACME.US"]
    )

    assert stats["beta"][0] == pytest.approx(2.5, abs=0.25)
    assert abs(stats["t_stat"][0]) > 3.0


def test_an_unrelated_ticker_is_not_significant() -> None:
    """
    Require pure noise to fail the significance test the longs and shorts filter on.

    :return: None.
    """
    daily, returns = synthetic(beta=0.0, noise=0.02)

    stats, _ = regression.regress_factor(
        daily=daily,
        returns=returns,
        tickers=["ACME.US"]
    )

    assert abs(stats["t_stat"][0]) < 2.5


def test_overlapping_horizons_widen_the_standard_error() -> None:
    """
    Require the Newey-West correction to bite when forward returns overlap.

    Ignoring the overlap is how a news factor comes to look significant when it is not, so this
    is the test that protects the headline claim.

    :return: None.
    """
    factor = np.random.default_rng(3).normal(size=500)
    residuals = np.repeat(np.random.default_rng(4).normal(size=100), 5)

    plain = regression.newey_west_error(factor=factor, residuals=residuals, lags=0)
    corrected = regression.newey_west_error(factor=factor, residuals=residuals, lags=9)

    assert corrected > plain


def test_point_in_time_loadings_are_produced_per_date() -> None:
    """
    Require a rolling beta per date, which is what makes the exposure point-in-time.

    :return: None.
    """
    daily, returns = synthetic(beta=1.5)

    _, loadings = regression.regress_factor(
        daily=daily,
        returns=returns,
        tickers=["ACME.US"],
        window=252
    )

    assert loadings.height > 500
    assert set(loadings.columns) == {"date", "ticker", "beta"}
    assert loadings["beta"].median() == pytest.approx(1.5, abs=0.4)


def test_regressing_on_the_level_differs_from_the_change() -> None:
    """
    Require difference=False to be a real alternative rather than a no-op.

    :return: None.
    """
    daily, returns = synthetic(beta=2.0)

    changed, _ = regression.regress_factor(
        daily=daily, returns=returns, tickers=["ACME.US"], difference=True
    )
    levelled, _ = regression.regress_factor(
        daily=daily, returns=returns, tickers=["ACME.US"], difference=False
    )

    assert changed["beta"][0] != levelled["beta"][0]
    assert abs(changed["beta"][0] - 2.0) < abs(levelled["beta"][0] - 2.0)


def test_a_ticker_absent_from_the_returns_is_reported() -> None:
    """
    Require a missing ticker to produce a message naming what is available.

    :return: None.
    """
    daily, returns = synthetic(beta=1.0)

    with pytest.raises(ConfigurationError) as error:
        regression.regress_factor(daily=daily, returns=returns, tickers=["NOPE.US"])

    assert "ACME.US" in str(error.value)


def test_a_negative_horizon_is_rejected() -> None:
    """
    Require the horizon to be zero or forward, never backwards into the future.

    :return: None.
    """
    daily, returns = synthetic(beta=1.0)

    with pytest.raises(ConfigurationError):
        regression.regress_factor(daily=daily, returns=returns, tickers=["ACME.US"], horizon=-1)


def test_the_factor_is_carried_forward_onto_non_trading_days() -> None:
    """
    Require a weekend to inherit Friday's factor rather than dropping the Monday return.

    :return: None.
    """
    dates = [START + dt.timedelta(days=step) for step in range(10)]
    daily = pl.DataFrame(
        data={
            "date": dates,
            "value": [float(step) for step in range(10)],
            "polarity": [None] * 10,
            "n_events": [1] * 10,
            "n_eligible": [1] * 10,
            "raw": [1.0] * 10,
            "volume": [1.0] * 10,
            "baseline": [1.0] * 10,
            "share": [1.0] * 10
        }
    ).select(series.DAILY_COLUMNS)
    trading = pl.DataFrame(
        data={"date": [dates[0], dates[3], dates[7]], "ACME.US": [0.01, -0.02, 0.03]}
    )

    aligned = regression.align(daily=daily, returns=trading, column="value", difference=False)

    assert aligned["factor"].to_list() == [0.0, 3.0, 7.0]
