"""How much each event counts.

A factor is a weighted count of the events that matched, and the weight decides what the
factor is measuring. Counting events equally measures how often something is written about.
Weighting by distinct sources measures how widely it travelled. Weighting by relevance
measures how squarely it hit the definition.

There is no universally correct answer, which is why the choice is the caller's. The public
World projection supports the named schemes below and returns the matching exact denominator.
"""
from typing import Callable, Dict, Mapping, Sequence, Union

import numpy as np

#: Weighting schemes, with the question each one answers.
WEIGHT_SCHEMES: Dict[str, str] = {
    "equal": "Every matching event counts once: how often is this written about?",
    "coverage": "Articles published (total_coverage): how much column space did it take?",
    "netlocs": "Distinct publishers (total_netlocs): how widely did it travel?",
    "materiality": "Materiality score in [0, 1]: how much did it matter?",
    "sentiment": "Negativity of the event: 1.0 negative, 0.5 neutral, 0.0 positive.",
    "relevance": "The event's own relevance score: how squarely did it hit the definition?"
}

#: Weight assigned to each sentiment label by the "sentiment" scheme.
SENTIMENT_WEIGHTS = {"negative": 1.0, "neutral": 0.5, "positive": 0.0}

#: Field each scheme reads out of an event record.
SCHEME_FIELDS = {
    "coverage": "total_coverage",
    "netlocs": "total_netlocs",
    "materiality": "materiality_score",
    "sentiment": "sentiment"
}

WeightCallable = Callable[[Sequence[Mapping[str, object]], np.ndarray], Sequence[float]]
WeightSpec = Union[str, WeightCallable]


def event_weights(
    records: Sequence[Mapping[str, object]],
    scheme: WeightSpec,
    relevance: np.ndarray
) -> np.ndarray:
    """
    Turn a batch of event records into one non-negative weight each.

    :param records: Event records carrying the flat fields named in SCHEME_FIELDS.
    :param scheme: A name from WEIGHT_SCHEMES, or a callable taking the records and the
        relevance array and returning one weight per event.
    :param relevance: Relevance score per event, in record order.
    :return: Float64 weights, one per record, clipped at zero.
    """
    if callable(scheme):
        weights = np.asarray(a=scheme(records, relevance), dtype=np.float64)
        if weights.shape != (len(records),):
            raise ValueError(
                f"weight callable returned shape {weights.shape}, expected ({len(records)},)"
            )
        return np.clip(a=np.nan_to_num(x=weights, nan=0.0), a_min=0.0, a_max=None)
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(
            f"weights must be a callable or one of {sorted(WEIGHT_SCHEMES)}, got {scheme!r}"
        )
    if scheme == "equal":
        return np.ones(shape=len(records), dtype=np.float64)
    if scheme == "relevance":
        return np.clip(
            a=np.asarray(a=relevance, dtype=np.float64),
            a_min=0.0,
            a_max=None
        )
    if scheme == "sentiment":
        return np.asarray(
            a=[SENTIMENT_WEIGHTS.get(str(record.get("sentiment")), 0.5) for record in records],
            dtype=np.float64
        )
    field = SCHEME_FIELDS[str(scheme)]
    values = np.asarray(
        a=[numeric(value=record.get(field)) for record in records],
        dtype=np.float64
    )
    return np.clip(a=values, a_min=0.0, a_max=None)


def column_weights(
    columns: Mapping[str, np.ndarray],
    scheme: str,
    relevance: np.ndarray
) -> np.ndarray:
    """
    Apply a named scheme to columnar event data, for callers holding arrays rather than dicts.

    This path serves columnar/offline callers that would otherwise build one dictionary per
    event just to apply a named weighting scheme.

    :param columns: Field name to array, aligned with relevance.
    :param scheme: A name from WEIGHT_SCHEMES.
    :param relevance: Relevance score per event.
    :return: Float64 weights, one per event, clipped at zero.
    """
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"weights must be one of {sorted(WEIGHT_SCHEMES)}, got {scheme!r}")
    if scheme == "equal":
        return np.ones(shape=relevance.shape[0], dtype=np.float64)
    if scheme == "relevance":
        return np.clip(a=relevance.astype(np.float64), a_min=0.0, a_max=None)
    if scheme == "sentiment":
        labels = columns["sentiment"]
        weights = np.full(shape=labels.shape[0], fill_value=SENTIMENT_WEIGHTS["neutral"])
        for label, value in SENTIMENT_WEIGHTS.items():
            weights[labels == label] = value
        return weights
    values = np.nan_to_num(
        x=columns[SCHEME_FIELDS[scheme]].astype(np.float64),
        nan=0.0
    )
    return np.clip(a=values, a_min=0.0, a_max=None)


def required_fields(scheme: WeightSpec) -> Sequence[str]:
    """
    Report which event fields a scheme reads, so a caller can request only those.

    :param scheme: A name from WEIGHT_SCHEMES, or a callable.
    :return: Field names the scheme needs, empty when it needs none.
    """
    if callable(scheme) or scheme in ("equal", "relevance"):
        return ()
    return (SCHEME_FIELDS[str(scheme)],)


def numeric(value: object) -> float:
    """
    Coerce one field value to a float, treating anything unusable as zero.

    :param value: Field value from an event record.
    :return: Finite float.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if np.isfinite(result) else 0.0
    return 0.0
