"""The factor object: definition, validation, and the reduction to days."""
import datetime as dt

import numpy as np
import pytest

from semantic_factors import factor as factor_module
from semantic_factors.errors import ConfigurationError, NotComputedError
from semantic_factors.factor import SemanticFactor

ANCHORS = ["Locust swarms destroy crops across the region, threatening the food supply."]


def test_a_factor_reports_its_shape() -> None:
    """
    Require the representation to say what the factor is and whether it has run.

    :return: None.
    """
    result = repr(SemanticFactor(anchors=ANCHORS, name="pests"))

    assert "pests" in result
    assert "uncomputed" in result


def test_anchors_are_cleaned_and_deduplicated() -> None:
    """
    Require whitespace to be normalised and repeats dropped, since both distort the geometry.

    :return: None.
    """
    result = SemanticFactor(anchors=["  a   b  ", "a b", "c d"]).anchors

    assert result == ["a b", "c d"]


def test_a_bare_string_of_anchors_is_rejected() -> None:
    """
    Require a single string to be caught, rather than embedding it one character at a time.

    :return: None.
    """
    with pytest.raises(ConfigurationError):
        SemanticFactor(anchors="one sentence")


def test_a_single_pole_is_rejected() -> None:
    """
    Require poles to come in pairs, because the measure is the difference between them.

    :return: None.
    """
    with pytest.raises(ConfigurationError):
        SemanticFactor(anchors=ANCHORS, pole_hi=["worse"])


def test_an_unknown_aggregation_is_rejected_at_definition_time() -> None:
    """
    Require a bad aggregation to fail before any network call.

    :return: None.
    """
    with pytest.raises(ConfigurationError):
        SemanticFactor(anchors=ANCHORS, aggregate="median")


def test_quantile_floors_reject_aggregate_tails_that_retrieval_cannot_represent() -> None:
    """Require mean/softmax quantile calibration to fail instead of looking exact."""
    with pytest.raises(ConfigurationError, match="quantile floors require"):
        SemanticFactor(anchors=ANCHORS, aggregate="mean", floor="p99.9")

    subject = SemanticFactor(anchors=ANCHORS, aggregate="mean", floor=0.35)
    assert subject.aggregate == "mean"


def test_reading_results_before_computing_explains_why() -> None:
    """
    Require an unhelpful AttributeError to be replaced by a message that says what to do.

    :return: None.
    """
    subject = SemanticFactor(anchors=ANCHORS)

    with pytest.raises(NotComputedError):
        subject.values
    with pytest.raises(NotComputedError):
        subject.longs


def test_a_nested_event_record_is_flattened() -> None:
    """
    Require the weighting fields to be lifted out of the nested World record.

    :return: None.
    """
    result = factor_module.flatten_event(
        event={
            "event_id": "abc",
            "_search_date": "2024-03-05",
            "event": {"title": "Locusts", "country": "Kenya", "language": "en"},
            "coverage": {"total_coverage": 40, "total_netlocs": 12},
            "signals": {"sentiment": "negative", "materiality_score": 0.8}
        }
    )

    assert result["date"] == "2024-03-05"
    assert result["total_netlocs"] == 12
    assert result["sentiment"] == "negative"


def test_records_fold_into_one_row_per_day() -> None:
    """
    Require the per-event scores to reduce to daily sufficient statistics.

    :return: None.
    """
    records = [
        {"event_id": "a", "date": "2024-03-05", "total_netlocs": 10.0, "title": "", "country": ""},
        {"event_id": "b", "date": "2024-03-05", "total_netlocs": 30.0, "title": "", "country": ""},
        {"event_id": "c", "date": "2024-03-06", "total_netlocs": 5.0, "title": "", "country": ""}
    ]

    days, exemplars = factor_module.fold_records_into_days(
        records=records,
        relevance=np.asarray(a=[0.5, 0.6, 0.4], dtype=np.float32),
        polarity=np.asarray(a=[1.0, -1.0, 0.5], dtype=np.float32),
        weights="netlocs",
        volume={"2024-03-05": 100, "2024-03-06": 80, "2024-03-07": 90},
        top_events=1
    )
    first = next(day for day in days if day["date"] == "2024-03-05")

    assert first["raw"] == 10.0
    assert first["n_events"] == 2
    assert first["polarity_num"] == pytest.approx(10.0 - 30.0)
    assert next(day for day in days if day["date"] == "2024-03-07")["n_events"] == 0
    assert exemplars.height == 2


def test_only_significant_exposures_reach_the_longs_and_shorts() -> None:
    """
    Require the sign filter to also demand significance, so noise stays out of a trade list.

    :return: None.
    """
    import polars as pl

    stats = pl.DataFrame(
        data={
            "ticker": ["A", "B", "C", "D"],
            "beta": [2.0, 0.5, -3.0, -0.2],
            "t_stat": [4.0, 0.3, -5.0, -0.4]
        }
    )

    assert factor_module.significant_side(stats=stats, positive=True)["ticker"].to_list() == ["A"]
    assert factor_module.significant_side(stats=stats, positive=False)["ticker"].to_list() == ["C"]


def test_the_default_window_stops_before_today() -> None:
    """
    Require today to be excluded, because the live pipeline is still rewriting it.

    :return: None.
    """
    _, stop = factor_module.resolve_window(start_from=None, stop_at=None)

    assert stop < dt.datetime.now(tz=dt.timezone.utc).date()


def test_a_backwards_window_is_rejected() -> None:
    """
    Require an inverted date range to fail immediately.

    :return: None.
    """
    with pytest.raises(ConfigurationError):
        factor_module.resolve_window(start_from="2024-01-01", stop_at="2023-01-01")


def test_residualisation_is_refused_instead_of_silently_skipped() -> None:
    """
    Require unsupported geometry to fail before credentials or network access are touched.

    :return: None.
    """
    subject = SemanticFactor(anchors=ANCHORS, floor=0.35)

    with pytest.raises(ConfigurationError) as error:
        subject.compute(
            residualize=True,
            start_from="2024-01-01",
            stop_at="2024-01-02"
        )

    assert "silently" in str(error.value)


def test_poles_can_be_removed_from_the_numerator_without_changing_diagnostics() -> None:
    """Require polarity_weighted=False to preserve the direction-blind base weight."""
    records = [
        {"event_id": "a", "date": "2024-03-05", "total_netlocs": 10.0},
        {"event_id": "b", "date": "2024-03-05", "total_netlocs": 30.0}
    ]
    days, _ = factor_module.fold_records_into_days(
        records=records,
        relevance=np.asarray(a=[0.5, 0.6], dtype=np.float32),
        polarity=np.asarray(a=[1.0, -1.0], dtype=np.float32),
        weights="netlocs",
        volume={"2024-03-05": 100.0},
        eligible_counts={"2024-03-05": 50},
        polarity_weighted=False,
        top_events=0
    )

    assert days[0]["raw"] == 40.0
    assert days[0]["n_eligible"] == 50
    assert days[0]["polarity_num"] == pytest.approx(-20.0)


@pytest.mark.parametrize(
    ("scheme", "expected"),
    [
        ("equal", 7.0),
        ("relevance", 7.0),
        ("coverage", 100.0),
        ("netlocs", 50.0),
        ("materiality", 3.5),
        ("sentiment", 4.0),
    ]
)
def test_each_named_weight_uses_its_matching_exact_denominator(scheme, expected) -> None:
    """Require numerator and denominator units to stay aligned for every public scheme."""
    eligible = {
        "count": 7,
        "total_coverage": 100.0,
        "total_netlocs": 50.0,
        "total_materiality": 3.5,
        "total_sentiment_weight": 4.0,
    }

    assert factor_module.denominator_weight(eligible=eligible, weights=scheme) == expected
