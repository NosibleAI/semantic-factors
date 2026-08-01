"""How much each event counts."""
import numpy as np
import pytest

from semantic_factors import weights

RECORDS = [
    {"total_coverage": 100.0, "total_netlocs": 20.0, "materiality_score": 0.9, "sentiment": "negative"},
    {"total_coverage": 10.0, "total_netlocs": 5.0, "materiality_score": 0.1, "sentiment": "positive"},
    {"total_coverage": None, "total_netlocs": 0.0, "materiality_score": None, "sentiment": "neutral"}
]
RELEVANCE = np.asarray(a=[0.5, 0.4, -0.1], dtype=np.float32)


def test_equal_weighting_counts_events() -> None:
    """
    Require the equal scheme to give every matching event the same weight.

    :return: None.
    """
    result = weights.event_weights(records=RECORDS, scheme="equal", relevance=RELEVANCE)

    assert result.tolist() == [1.0, 1.0, 1.0]


def test_coverage_and_netlocs_read_their_fields() -> None:
    """
    Require the coverage schemes to read the coverage block of the event record.

    :return: None.
    """
    coverage = weights.event_weights(records=RECORDS, scheme="coverage", relevance=RELEVANCE)
    netlocs = weights.event_weights(records=RECORDS, scheme="netlocs", relevance=RELEVANCE)

    assert coverage.tolist() == [100.0, 10.0, 0.0]
    assert netlocs.tolist() == [20.0, 5.0, 0.0]


def test_sentiment_weighting_favours_negative_events() -> None:
    """
    Require the documented sentiment mapping, since it is a choice rather than a measurement.

    :return: None.
    """
    result = weights.event_weights(records=RECORDS, scheme="sentiment", relevance=RELEVANCE)

    assert result.tolist() == [1.0, 0.0, 0.5]


def test_relevance_weighting_cannot_go_negative() -> None:
    """
    Require a negative cosine to contribute nothing rather than subtracting from the day.

    :return: None.
    """
    result = weights.event_weights(records=RECORDS, scheme="relevance", relevance=RELEVANCE)

    assert result.tolist() == [0.5, pytest.approx(0.4, abs=1e-6), 0.0]


def test_a_missing_field_becomes_zero_not_nan() -> None:
    """
    Require a null coverage value to drop out of the sum instead of poisoning it.

    :return: None.
    """
    result = weights.event_weights(records=RECORDS, scheme="materiality", relevance=RELEVANCE)

    assert np.isfinite(result).all()
    assert result.tolist() == [0.9, pytest.approx(0.1, abs=1e-6), 0.0]


def test_a_callable_scheme_is_accepted() -> None:
    """
    Require an arbitrary weighting rule to be usable without changing the package.

    :return: None.
    """
    result = weights.event_weights(
        records=RECORDS,
        scheme=lambda records, relevance: [float(record["total_netlocs"] or 0.0) ** 0.5
                                           for record in records],
        relevance=RELEVANCE
    )

    assert result[0] == pytest.approx(20.0 ** 0.5)


def test_a_callable_returning_the_wrong_length_is_rejected() -> None:
    """
    Require a mis-sized weight vector to fail loudly rather than broadcast.

    :return: None.
    """
    with pytest.raises(ValueError):
        weights.event_weights(
            records=RECORDS,
            scheme=lambda records, relevance: [1.0],
            relevance=RELEVANCE
        )


def test_an_unknown_scheme_names_the_valid_ones() -> None:
    """
    Require a bad scheme name to list the alternatives.

    :return: None.
    """
    with pytest.raises(ValueError) as error:
        weights.event_weights(records=RECORDS, scheme="volume", relevance=RELEVANCE)

    assert "netlocs" in str(error.value)


def test_columnar_weighting_matches_record_weighting() -> None:
    """
    Require the server's columnar path to agree with the client's record path exactly.

    These are two implementations of one definition, and a factor built either way has to come
    out the same.

    :return: None.
    """
    columns = {
        "total_coverage": np.asarray(a=[100.0, 10.0, np.nan]),
        "total_netlocs": np.asarray(a=[20.0, 5.0, 0.0]),
        "materiality_score": np.asarray(a=[0.9, 0.1, np.nan]),
        "sentiment": np.asarray(a=["negative", "positive", "neutral"], dtype=object)
    }

    for scheme in ("equal", "coverage", "netlocs", "materiality", "sentiment", "relevance"):
        columnar = weights.column_weights(columns=columns, scheme=scheme, relevance=RELEVANCE)
        record = weights.event_weights(records=RECORDS, scheme=scheme, relevance=RELEVANCE)

        assert np.allclose(columnar, record), scheme


def test_required_fields_lists_only_what_is_read() -> None:
    """
    Require the field advertisement to be accurate, since the server materialises exactly it.

    :return: None.
    """
    assert weights.required_fields(scheme="equal") == ()
    assert weights.required_fields(scheme="netlocs") == ("total_netlocs",)
