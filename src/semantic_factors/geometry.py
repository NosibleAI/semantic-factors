"""The vector maths a factor is made of.

Four operations, in the order a factor uses them:

1. ``l2`` puts every vector on the unit sphere so a dot product is a cosine.
2. ``residualize`` subtracts a family's mean direction. Anchors written to describe one idea
   share a large common component — the phrasing, the register, the topic frame — and that
   component is the same for every event, so it inflates every score without separating
   anything. Removing it leaves the part of each anchor that actually discriminates.
3. ``relevance`` collapses one event's cosine against every anchor into a single number.
4. ``polarity`` places a relevant event on the axis between two poles.

Nothing here touches the network, and every function is pure, which is why the tests for the
scoring rules run offline.
"""
from typing import Optional, Sequence, Tuple

import numpy as np

#: Divisor applied to the raw pole gap before tanh. This is the value the published Trade Policy
#: Uncertainty replication uses. It is deliberately wide: a narrower scale drives almost every
#: event to +/-1 and turns a graded axis into a sign bit.
DEFAULT_TANH_SCALE = 0.1

#: How per-anchor cosines are collapsed into one relevance score.
AGGREGATIONS = ("max", "mean", "softmax")

#: Temperature of the softmax aggregation, in cosine units.
SOFTMAX_TEMPERATURE = 0.02

#: Statements a family needs before residualising it means anything. Subtracting the mean of two
#: vectors leaves two exact opposites, which describes nothing; with three it is still mostly
#: noise. Below this threshold residualisation is skipped rather than applied destructively.
MIN_RESIDUALIZE_MEMBERS = 4


def l2(matrix: np.ndarray) -> np.ndarray:
    """
    Scale every row to unit length, leaving zero rows alone rather than dividing by zero.

    :param matrix: Two-dimensional array of row vectors.
    :return: Row-normalised float32 copy.
    """
    values = np.asarray(a=matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {values.shape}")
    norm = np.linalg.norm(x=values, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (values / norm).astype(np.float32)


def residualize(
    vectors: np.ndarray,
    direction: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Remove a shared direction from a set of vectors and put them back on the unit sphere.

    With a direction supplied — normally the corpus mean, the direction every news event points
    in a little — this is the operation ``residualize=True`` performs, applied identically to
    events and to queries so cosines stay comparable. Removing it roughly triples the gap
    between a relevant event and a typical one.

    With no direction, the family's own mean is removed instead. Families smaller than
    MIN_RESIDUALIZE_MEMBERS are returned untouched, because subtracting the mean of two vectors
    leaves two exact opposites.

    :param vectors: Vectors to residualise, one per row.
    :param direction: Direction to remove, or None to remove the family's own mean.
    :return: Residualised, renormalised vectors of the same shape.
    """
    values = np.asarray(a=vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {values.shape}")
    if direction is not None:
        return l2(matrix=values - np.asarray(a=direction, dtype=np.float32).reshape(1, -1))
    if values.shape[0] < MIN_RESIDUALIZE_MEMBERS:
        return l2(matrix=values)
    return l2(matrix=values - values.mean(axis=0))


def cosines(events: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """
    Score every event against every query vector.

    Both sides are assumed unit length, which the loaders guarantee, so this is one matrix
    multiply and no per-row division.

    :param events: Unit-length event vectors, one per row.
    :param queries: Unit-length query vectors, one per row.
    :return: Array of shape (events, queries) holding cosine similarities.
    """
    return np.asarray(a=events, dtype=np.float32) @ np.asarray(a=queries, dtype=np.float32).T


def relevance(scores: np.ndarray, aggregate: str = "max") -> np.ndarray:
    """
    Collapse each event's per-anchor cosines into one relevance score.

    ``max`` asks whether any single anchor describes the event, which is what a list of
    alternative phrasings means. ``mean`` asks whether the event sits in the middle of all of
    them, which suits anchors written as facets of one idea. ``softmax`` interpolates.

    :param scores: Array of shape (events, anchors) from cosines().
    :param aggregate: One of the names in AGGREGATIONS.
    :return: One relevance score per event.
    """
    if aggregate not in AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {AGGREGATIONS}, got {aggregate!r}")
    values = np.asarray(a=scores, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D score matrix, got shape {values.shape}")
    if aggregate == "max":
        return values.max(axis=1)
    if aggregate == "mean":
        return values.mean(axis=1)
    shifted = (values - values.max(axis=1, keepdims=True)) / SOFTMAX_TEMPERATURE
    weights = np.exp(shifted)
    return (values * weights).sum(axis=1) / weights.sum(axis=1)


def paired_polarity(
    events: np.ndarray,
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    tanh_scale: float = DEFAULT_TANH_SCALE
) -> np.ndarray:
    """
    Place each event on the signed axis of whichever pole pair describes it best.

    A pole pair is one way of saying the same thing twice, once as the high reading and once as
    the low one. Where a factor spans several topics, one generic pair is lopsided: uncertainty
    shares a vocabulary across topics but resolution is always written as the topic's own
    concrete act. So each topic gets its own matched pair.

    A pair's score is the better of its two framings, and an event reads its polarity off the
    pair that scores highest. That is what makes a settlement in one topic register as settled
    rather than as merely unlike the other topics' escalation language.

    :param events: Unit-length event vectors, one per row.
    :param pairs: Sequence of (high, low) unit-length vector matrices, one entry per pair.
    :param tanh_scale: Divisor applied to the cosine gap before tanh.
    :return: One signed score per event, in (-1, +1).
    """
    if tanh_scale <= 0:
        raise ValueError(f"tanh_scale must be positive, got {tanh_scale}")
    if not pairs:
        raise ValueError("at least one pole pair is required")
    best_gap = np.zeros(shape=events.shape[0], dtype=np.float32)
    best_score = np.full(shape=events.shape[0], fill_value=-np.inf, dtype=np.float32)
    for high, low in pairs:
        towards_high = cosines(events=events, queries=high).max(axis=1)
        towards_low = cosines(events=events, queries=low).max(axis=1)
        score = np.maximum(towards_high, towards_low)
        chosen = score > best_score
        best_gap = np.where(chosen, towards_high - towards_low, best_gap)
        best_score = np.where(chosen, score, best_score)
    return np.tanh(best_gap / tanh_scale).astype(np.float32)


def paired_polarity_from_scores(
    scores: np.ndarray,
    pairs: Sequence[Tuple[slice, slice]],
    tanh_scale: float = DEFAULT_TANH_SCALE
) -> np.ndarray:
    """
    Compute matched-pair polarity from already-scored pole vectors.

    :param scores: Exact event-by-pole-vector cosine matrix returned by World search.
    :param pairs: High and low column slices for each matched pair.
    :param tanh_scale: Divisor applied to the selected pair's cosine gap.
    :return: One signed score per event, in (-1, +1).
    """
    values = np.asarray(a=scores, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D score matrix, got shape {values.shape}")
    if tanh_scale <= 0:
        raise ValueError(f"tanh_scale must be positive, got {tanh_scale}")
    if not pairs:
        raise ValueError("at least one pole pair is required")
    best_gap = np.zeros(shape=values.shape[0], dtype=np.float32)
    best_score = np.full(shape=values.shape[0], fill_value=-np.inf, dtype=np.float32)
    for high_slice, low_slice in pairs:
        high = values[:, high_slice]
        low = values[:, low_slice]
        if high.shape[1] == 0 or low.shape[1] == 0:
            raise ValueError("pole slices must each select at least one score column")
        towards_high = high.max(axis=1)
        towards_low = low.max(axis=1)
        score = np.maximum(towards_high, towards_low)
        chosen = score > best_score
        best_gap = np.where(chosen, towards_high - towards_low, best_gap)
        best_score = np.where(chosen, score, best_score)
    return np.tanh(best_gap / tanh_scale).astype(np.float32)


def uncertainty_weight(polarity_scores: np.ndarray) -> np.ndarray:
    """
    Map a signed pole score onto the 0-to-1 multiplier the index numerator uses.

    This is what makes the poles change the indicator rather than sit beside it. An event at the
    high pole contributes its full weight, an event at the low pole contributes almost none, and
    a neutral event contributes half.

    :param polarity_scores: Signed scores in (-1, +1).
    :return: Multipliers in (0, 1).
    """
    return ((1.0 + np.asarray(a=polarity_scores, dtype=np.float64)) / 2.0)


def polarity(
    events: np.ndarray,
    pole_hi: np.ndarray,
    pole_lo: np.ndarray,
    tanh_scale: float = DEFAULT_TANH_SCALE,
    aggregate: str = "mean"
) -> np.ndarray:
    """
    Place each event on the signed axis running from the low pole to the high pole.

    The two poles are scored separately and subtracted, so the shared component of the pole
    language cancels and only the direction survives. tanh bounds the result to (-1, +1) and
    keeps one unusually strong event from dominating a day.

    :param events: Unit-length event vectors, one per row.
    :param pole_hi: Unit-length vectors for statements that should read as high.
    :param pole_lo: Unit-length vectors for statements that should read as low.
    :param tanh_scale: Divisor applied to the cosine gap before tanh.
    :param aggregate: How each pole's members are collapsed, one of AGGREGATIONS.
    :return: One signed score per event, in (-1, +1).
    """
    if tanh_scale <= 0:
        raise ValueError(f"tanh_scale must be positive, got {tanh_scale}")
    high = relevance(
        scores=cosines(events=events, queries=pole_hi),
        aggregate=aggregate
    )
    low = relevance(
        scores=cosines(events=events, queries=pole_lo),
        aggregate=aggregate
    )
    return np.tanh((high - low) / tanh_scale).astype(np.float32)


def decode_vector_text(text: str, dims: Optional[int] = None) -> np.ndarray:
    """
    Parse the comma-separated float text a World event carries in its oai_vector field.

    :param text: Comma-separated decimal string.
    :param dims: Expected length, or None to accept whatever is present.
    :return: Unit-length float32 vector.
    """
    vector = np.asarray(a=text.split(","), dtype=np.float32)
    if dims is not None and vector.shape[0] != dims:
        raise ValueError(f"expected {dims} dimensions, parsed {vector.shape[0]}")
    return l2(matrix=vector.reshape(1, -1))[0]
