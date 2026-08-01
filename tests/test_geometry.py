"""The vector maths a factor is made of."""
import numpy as np
import pytest

from semantic_factors import geometry


def unit(values: list) -> np.ndarray:
    """
    Build a row-normalised matrix from a list of rows.

    :param values: Rows of numbers.
    :return: Unit-length float32 matrix.
    """
    return geometry.l2(matrix=np.asarray(a=values, dtype=np.float32))


def test_l2_makes_every_row_unit_length() -> None:
    """
    Require normalisation so a dot product is a cosine.

    :return: None.
    """
    result = geometry.l2(matrix=np.asarray(a=[[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_l2_leaves_a_zero_row_alone() -> None:
    """
    Require a zero row to survive rather than becoming nan.

    :return: None.
    """
    result = geometry.l2(matrix=np.zeros(shape=(1, 4), dtype=np.float32))

    assert np.isfinite(result).all()


def test_removing_a_supplied_direction_recentres_the_family() -> None:
    """
    Require residualisation against a direction to leave vectors orthogonal to it.

    :return: None.
    """
    direction = np.asarray(a=[1.0, 0.0, 0.0], dtype=np.float32)
    vectors = unit([[1.0, 0.2, 0.0], [1.0, 0.0, 0.2]])

    result = geometry.residualize(vectors=vectors, direction=direction * 0.9)

    assert result.shape == vectors.shape
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_a_tiny_family_is_not_residualised_against_its_own_mean() -> None:
    """
    Require a pair to survive, because subtracting their mean leaves two exact opposites.

    :return: None.
    """
    vectors = unit([[1.0, 0.1, 0.0], [1.0, 0.0, 0.1]])

    result = geometry.residualize(vectors=vectors)

    assert np.allclose(result, vectors, atol=1e-6)
    assert float(result[0] @ result[1]) > 0.9


def test_a_large_family_is_residualised_against_its_own_mean() -> None:
    """
    Require a family of at least MIN_RESIDUALIZE_MEMBERS to lose its shared direction.

    :return: None.
    """
    vectors = unit([[1.0, 0.1, 0.0], [1.0, 0.0, 0.1], [1.0, 0.1, 0.1], [1.0, 0.2, 0.0]])

    result = geometry.residualize(vectors=vectors)

    assert abs(float(result.mean(axis=0) @ np.asarray(a=[1.0, 0.0, 0.0]))) < 0.2


def test_max_aggregation_takes_the_best_anchor() -> None:
    """
    Require max to ask whether any single anchor describes the event.

    :return: None.
    """
    scores = np.asarray(a=[[0.1, 0.6], [0.4, 0.2]], dtype=np.float32)

    assert np.allclose(geometry.relevance(scores=scores, aggregate="max"), [0.6, 0.4])


def test_mean_aggregation_averages_the_anchors() -> None:
    """
    Require mean to ask whether the event sits among all of them.

    :return: None.
    """
    scores = np.asarray(a=[[0.1, 0.5]], dtype=np.float32)

    assert np.allclose(geometry.relevance(scores=scores, aggregate="mean"), [0.3])


def test_softmax_aggregation_lies_between_mean_and_max() -> None:
    """
    Require softmax to interpolate rather than pick a side.

    :return: None.
    """
    scores = np.asarray(a=[[0.10, 0.30, 0.32]], dtype=np.float32)

    soft = float(geometry.relevance(scores=scores, aggregate="softmax")[0])

    assert float(scores.mean()) < soft <= float(scores.max())


def test_an_unknown_aggregation_is_rejected() -> None:
    """
    Require a bad aggregation name to fail loudly.

    :return: None.
    """
    with pytest.raises(ValueError):
        geometry.relevance(scores=np.zeros(shape=(1, 2), dtype=np.float32), aggregate="median")


def test_polarity_is_signed_and_bounded() -> None:
    """
    Require an event nearer the high pole to score positive, and everything to stay in (-1, 1).

    :return: None.
    """
    high = unit([[1.0, 0.0]])
    low = unit([[0.0, 1.0]])
    events = unit([[1.0, 0.1], [0.1, 1.0], [1.0, 1.0]])

    result = geometry.polarity(events=events, pole_hi=high, pole_lo=low, tanh_scale=0.5)

    assert result[0] > 0
    assert result[1] < 0
    assert abs(float(result[2])) < 1e-6
    assert np.all(np.abs(result) < 1.0)


def test_polarity_rejects_a_non_positive_scale() -> None:
    """
    Require a zero or negative tanh scale to fail rather than divide by zero.

    :return: None.
    """
    with pytest.raises(ValueError):
        geometry.polarity(
            events=unit([[1.0, 0.0]]),
            pole_hi=unit([[1.0, 0.0]]),
            pole_lo=unit([[0.0, 1.0]]),
            tanh_scale=0.0
        )


def test_a_stored_vector_round_trips_from_its_text_form() -> None:
    """
    Require the comma-separated form World stores to decode to a unit vector.

    :return: None.
    """
    result = geometry.decode_vector_text(text="0.3,0.4,0.0", dims=3)

    assert np.allclose(result, [0.6, 0.8, 0.0], atol=1e-6)


def test_decoding_checks_the_dimension() -> None:
    """
    Require a truncated vector to be caught rather than silently reshaped.

    :return: None.
    """
    with pytest.raises(ValueError):
        geometry.decode_vector_text(text="0.3,0.4", dims=3)
