"""Poles: matched pairs, and the weight that lets them move the indicator.

The published Trade Policy Uncertainty replication is the reference these tests protect. Its
whole method is three anchors and one matched pair, where the pair replaces the paper's "AND an
uncertainty word" rule by turning each event's framing into a 0-to-1 multiplier on its weight.
If the multiplier stops reaching the numerator, the library silently stops computing that index
and starts computing a direction-blind count of trade coverage instead.
"""
import numpy as np
import pytest

from semantic_factors import geometry
from semantic_factors.errors import ConfigurationError
from semantic_factors.factor import SemanticFactor, validate_poles

ANCHORS = ["A government imposes tariffs on goods imported from other countries."]


def unit(values: list) -> np.ndarray:
    """
    Build a row-normalised matrix from a list of rows.

    :param values: Rows of numbers.
    :return: Unit-length float32 matrix.
    """
    return geometry.l2(matrix=np.asarray(a=values, dtype=np.float32))


def test_the_uncertainty_weight_spans_zero_to_one() -> None:
    """
    Require the published mapping (1 + polarity) / 2, which is what tilts the numerator.

    :return: None.
    """
    result = geometry.uncertainty_weight(polarity_scores=np.asarray(a=[-1.0, 0.0, 1.0]))

    assert result.tolist() == [0.0, 0.5, 1.0]


def test_the_default_tanh_scale_matches_the_published_recipe() -> None:
    """
    Require the shipped default to be the value the replication used.

    A narrower scale drives nearly every event to +/-1 and turns a graded axis into a sign bit,
    which reads like a clean result and is not one.

    :return: None.
    """
    assert geometry.DEFAULT_TANH_SCALE == 0.1


def test_polarity_is_graded_rather_than_saturated_at_the_default_scale() -> None:
    """
    Require realistic cosine gaps to land inside the axis, not pinned to its ends.

    :return: None.
    """
    high = unit([[1.0, 0.0, 0.0]])
    low = unit([[0.0, 1.0, 0.0]])
    events = unit([[1.0, 0.90, 0.0], [1.0, 0.97, 0.0], [0.97, 1.0, 0.0]])

    result = geometry.paired_polarity(events=events, pairs=[(high, low)])

    assert np.all(np.abs(result) < 0.999)
    assert result[0] > result[1] > result[2]


def test_each_event_reads_its_polarity_off_the_pair_that_describes_it() -> None:
    """
    Require per-event best-pair selection across several matched pairs.

    Resolution is always written as a topic's own concrete act, so a settlement in one topic has
    to be read against that topic's own pair. Pooling every pair together loses that, and the
    settlement reads as merely unlike the other topics' escalation language.

    :return: None.
    """
    trade = (unit([[1.0, 0.0, 0.0, 0.0]]), unit([[0.0, 1.0, 0.0, 0.0]]))
    monetary = (unit([[0.0, 0.0, 1.0, 0.0]]), unit([[0.0, 0.0, 0.0, 1.0]]))
    settled_trade = unit([[0.1, 1.0, 0.0, 0.0]])

    both = geometry.paired_polarity(events=settled_trade, pairs=[trade, monetary])
    trade_only = geometry.paired_polarity(events=settled_trade, pairs=[trade])

    assert both[0] < 0
    assert both[0] == pytest.approx(trade_only[0], abs=1e-6)


def test_an_unrelated_pair_does_not_capture_the_event() -> None:
    """
    Require the pair with the weaker match to be ignored entirely.

    :return: None.
    """
    strong = (unit([[1.0, 0.0, 0.0, 0.0]]), unit([[0.0, 1.0, 0.0, 0.0]]))
    weak = (unit([[0.0, 0.0, 0.0, 1.0]]), unit([[0.0, 0.0, 1.0, 0.0]]))
    event = unit([[1.0, 0.05, 0.0, 0.02]])

    with_weak = geometry.paired_polarity(events=event, pairs=[strong, weak])
    without = geometry.paired_polarity(events=event, pairs=[strong])

    assert with_weak[0] == pytest.approx(without[0], abs=1e-6)


def test_polarity_needs_at_least_one_pair() -> None:
    """
    Require an empty pair list to fail rather than return zeros that look like neutrality.

    :return: None.
    """
    with pytest.raises(ValueError):
        geometry.paired_polarity(events=unit([[1.0, 0.0]]), pairs=[])


def test_the_single_pair_shorthand_is_the_same_as_one_pair() -> None:
    """
    Require pole_hi/pole_lo to be exactly poles=[(hi, lo)], so existing code keeps working.

    :return: None.
    """
    shorthand = SemanticFactor(anchors=ANCHORS, pole_hi=["worse"], pole_lo=["better"])
    explicit = SemanticFactor(anchors=ANCHORS, poles=[(["worse"], ["better"])])

    assert shorthand.poles == explicit.poles == [(["worse"], ["better"])]


def test_several_pairs_are_kept_in_order() -> None:
    """
    Require a multi-topic factor to keep each pair distinct.

    :return: None.
    """
    factor = SemanticFactor(
        anchors=ANCHORS,
        poles=[(["tariffs threatened"], ["tariffs in force"]),
               (["rate decision pending"], ["rate decision made"])]
    )

    assert len(factor.poles) == 2
    assert factor.poles[1][0] == ["rate decision pending"]


def test_mixing_the_two_declarations_is_rejected() -> None:
    """
    Require one way or the other, so there is never a question of which won.

    :return: None.
    """
    with pytest.raises(ConfigurationError):
        SemanticFactor(anchors=ANCHORS, poles=[(["a"], ["b"])], pole_hi=["c"], pole_lo=["d"])


def test_a_malformed_pair_names_its_position() -> None:
    """
    Require a pair that is not (high, low) to say which one is wrong.

    :return: None.
    """
    with pytest.raises(ConfigurationError) as error:
        validate_poles(poles=[(["a"], ["b"]), (["c"],)], pole_hi=None, pole_lo=None)

    assert "pair 1" in str(error.value)


def test_an_unsigned_factor_has_no_pairs() -> None:
    """
    Require a factor with no poles to be plainly unsigned rather than carrying an empty pair.

    :return: None.
    """
    assert SemanticFactor(anchors=ANCHORS).poles == []


def test_the_numerator_arithmetic_matches_the_published_formula() -> None:
    """
    Require breadth * w_unc, summed, to be what a day contributes.

    This is the arithmetic the whole index rests on: an event framed as unresolved contributes
    its full breadth, one framed as settled contributes almost none, a neutral one half.

    :return: None.
    """
    breadth = np.asarray(a=[100.0, 100.0, 100.0])
    polarity = np.asarray(a=[1.0, 0.0, -1.0])

    contribution = breadth * geometry.uncertainty_weight(polarity_scores=polarity)

    assert contribution.tolist() == [100.0, 50.0, 0.0]
    assert contribution.sum() == 150.0
    assert breadth.sum() == 300.0


def test_score_projection_matches_vector_based_pair_selection() -> None:
    """Require the lean API's pole cosines to reproduce the full-vector geometry exactly."""
    events = unit([[1.0, 0.2, 0.1], [0.1, 0.2, 1.0]])
    first = (unit([[1.0, 0.0, 0.0]]), unit([[0.0, 1.0, 0.0]]))
    second = (unit([[0.0, 0.0, 1.0]]), unit([[0.0, 1.0, 0.0]]))
    score_matrix = geometry.cosines(
        events=events,
        queries=np.concatenate([first[0], first[1], second[0], second[1]], axis=0)
    )

    projected = geometry.paired_polarity_from_scores(
        scores=score_matrix,
        pairs=[(slice(0, 1), slice(1, 2)), (slice(2, 3), slice(3, 4))]
    )
    vector_based = geometry.paired_polarity(events=events, pairs=[first, second])

    assert np.allclose(projected, vector_based)
