"""Choosing the relevance floor."""
import numpy as np
import pytest

from semantic_factors import floors
from semantic_factors.errors import ConfigurationError

SAMPLE = np.linspace(start=0.0, stop=1.0, num=10_001).astype(np.float32)


def test_a_quantile_string_is_recognised() -> None:
    """
    Require the quantile form to be told apart from an absolute cosine.

    :return: None.
    """
    assert floors.is_quantile(floor="p99.9")
    assert not floors.is_quantile(floor=0.35)


def test_auto_resolves_to_the_default() -> None:
    """
    Require "auto" to mean the documented default rather than something surprising.

    :return: None.
    """
    assert floors.parse_quantile(floor="auto") == floors.parse_quantile(floor=floors.DEFAULT_FLOOR)


def test_a_quantile_becomes_the_matching_cosine() -> None:
    """
    Require p99.9 of a uniform sample to land at 0.999.

    :return: None.
    """
    result = floors.resolve(floor="p99.9", sample=SAMPLE)

    assert abs(result - 0.999) < 1e-3


def test_an_absolute_floor_passes_straight_through() -> None:
    """
    Require a cosine to be used as given, with no calibration.

    :return: None.
    """
    assert floors.resolve(floor=0.42, sample=np.zeros(shape=0, dtype=np.float32)) == 0.42


def test_a_quantile_without_a_sample_is_an_error() -> None:
    """
    Require an empty calibration sample to say so, rather than silently admitting everything.

    :return: None.
    """
    with pytest.raises(ConfigurationError):
        floors.resolve(floor="p99", sample=np.zeros(shape=0, dtype=np.float32))


def test_a_malformed_floor_is_rejected() -> None:
    """
    Require nonsense to be caught at definition time.

    :return: None.
    """
    for bad in ("p", "99.9", "p101", "p0", 0.0, 1.0, -0.2):
        with pytest.raises(ConfigurationError):
            floors.validate(floor=bad)


def test_the_quantile_table_reports_yield_after_filters() -> None:
    """
    Require the table to say how many events per day survive the filters at each floor.

    That is the number a caller is actually choosing between, and it is not the same as the
    share of the corpus, because the filters cut independently of relevance.

    :return: None.
    """
    eligible = SAMPLE[SAMPLE > 0.5]

    table = floors.quantile_table(sample=SAMPLE, eligible=eligible, days=10)
    row = next(item for item in table if item["quantile"] == "p99.9")

    assert row["corpus_matches_per_day"] == pytest.approx(1.0, abs=0.05)
    assert row["matches_per_day"] == pytest.approx(1.0, abs=0.2)
    assert table[0]["matches_per_day"] > table[-1]["matches_per_day"]


def test_calibration_dates_span_the_window() -> None:
    """
    Require the sample to reach both ends, since the corpus in 2015 was nothing like today's.

    :return: None.
    """
    dates = list(range(1000))

    chosen = floors.calibration_dates(dates=dates, target_days=10)

    assert chosen[0] == 0
    assert chosen[-1] == 999
    assert len(chosen) == 10


def test_calibration_takes_everything_when_the_window_is_short() -> None:
    """
    Require a short window to be sampled in full rather than thinned.

    :return: None.
    """
    assert floors.calibration_dates(dates=[1, 2, 3], target_days=10) == [1, 2, 3]


def test_a_high_quantile_resolves_from_only_the_retrieved_tail() -> None:
    """Require p99 to use the 100th highest score out of an exact 10,000-event universe."""
    tail = np.linspace(start=1.0, stop=0.5, num=500, dtype=np.float32)

    result = floors.resolve_from_tail(floor="p99", tail_scores=tail, eligible_events=10_000)

    assert result == pytest.approx(np.sort(tail)[::-1][99])


def test_tail_calibration_refuses_a_rank_it_did_not_retrieve() -> None:
    """Require a low quantile to fail rather than extrapolate below the observed tail."""
    with pytest.raises(ConfigurationError):
        floors.resolve_from_tail(
            floor="p90",
            tail_scores=np.ones(shape=100, dtype=np.float32),
            eligible_events=10_000
        )
