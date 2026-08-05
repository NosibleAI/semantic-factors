"""Choosing the relevance floor.

The floor is the one number in a factor definition nobody can guess. It is a cosine, and what
counts as a high cosine depends on the embedding model, on whether residualisation is on, and
on how tightly the anchors are written. A floor of 0.35 admits the top tenth of a percent of
events for one factor and nothing at all for another.

So the floor is normally expressed as a quantile of the eligible corpus. ``"p99.9"`` means "the
most relevant one event in a thousand". That is a statement a person can reason about, and it
survives changing residualisation, rewriting the anchors, or the corpus growing tenfold.

The quantile is resolved **once**, on a sample spanning the whole window, and then applied as a
fixed cosine to every day. Resolving it per day would force a constant share of events through
every day and flatten the factor into a straight line.

It is taken over the **whole corpus, before filters**. The floor says how relevant an event has
to be, which is a property of the definition; the filters say whose events count, which is a
property of the universe. Calibrating the floor on the filtered pool would tangle the two, so
that narrowing the universe silently changed what the factor was about.
"""
import re
from typing import List, Optional, Sequence, Union

import numpy as np

from semantic_factors.errors import ConfigurationError

#: Floor used when the caller does not choose one.
DEFAULT_FLOOR = "p99.9"

#: Lowest absolute cosine floor accepted by the public package.
MIN_ABSOLUTE_FLOOR = 0.30

#: Quantiles reported by calibration, so a caller can see the trade-off before committing.
REPORTED_QUANTILES = (0.99, 0.995, 0.999, 0.9995, 0.9999)

#: Days a calibration sample will read at most, however few events they hold.
CALIBRATION_MAX_DAYS = 24

QUANTILE_PATTERN = re.compile(r"^p(\d{1,2}(?:\.\d+)?)$")

FloorSpec = Union[float, int, str]


def is_quantile(floor: FloorSpec) -> bool:
    """
    Report whether a floor specification has to be resolved against the corpus.

    :param floor: Absolute cosine, or a quantile string such as "p99.9".
    :return: True when the floor is a quantile.
    """
    return isinstance(floor, str)


def parse_quantile(floor: str) -> float:
    """
    Turn a quantile string into a fraction in (0, 1).

    :param floor: Quantile string such as "p99.9", or "auto" for the default.
    :return: The quantile as a fraction.
    """
    text = DEFAULT_FLOOR if floor.strip().lower() == "auto" else floor.strip().lower()
    match = QUANTILE_PATTERN.match(text)
    if not match:
        raise ConfigurationError(
            f"floor must be a cosine between 0 and 1, or a quantile such as 'p99.9', got {floor!r}"
        )
    percent = float(match.group(1))
    if not 0.0 < percent < 100.0:
        raise ConfigurationError(f"floor quantile must lie strictly between p0 and p100, got {floor!r}")
    return percent / 100.0


def validate(floor: FloorSpec) -> FloorSpec:
    """
    Check a floor specification without resolving it.

    :param floor: Absolute cosine, or a quantile string.
    :return: The floor, unchanged, once it is known to be usable.
    """
    if is_quantile(floor=floor):
        parse_quantile(floor=str(floor))
        return floor
    value = float(floor)
    if not 0.0 < value < 1.0:
        raise ConfigurationError(f"an absolute floor must lie in (0, 1), got {value}")
    if value < MIN_ABSOLUTE_FLOOR:
        raise ConfigurationError(
            f"an absolute floor must be at least {MIN_ABSOLUTE_FLOOR:.2f}, got {value:.4f}"
        )
    return value


def resolve(floor: FloorSpec, sample: np.ndarray) -> float:
    """
    Turn a floor specification into the cosine the scan will actually use.

    :param floor: Absolute cosine, or a quantile string.
    :param sample: Relevance scores of a representative sample of eligible events.
    :return: The absolute cosine floor.
    """
    if not is_quantile(floor=floor):
        return float(floor)
    if sample.size == 0:
        raise ConfigurationError(
            "a quantile floor needs a calibration sample, and no eligible events were found. "
            "Loosen the filters or give an absolute cosine floor."
        )
    return float(np.quantile(a=sample, q=parse_quantile(floor=str(floor))))


def quantile_table(
    sample: np.ndarray,
    eligible: Optional[np.ndarray] = None,
    days: int = 0
) -> List[dict]:
    """
    Describe a relevance distribution, so a caller can pick a floor with their eyes open.

    The quantiles are taken over the whole corpus, but the yield is reported after filters,
    because those are the two things a caller has to weigh: how relevant is relevant enough,
    and how many events per day survive once the universe is narrowed.

    :param sample: Relevance scores of a representative sample of every event.
    :param eligible: Relevance scores of the subset that passed the filters, or None.
    :param days: Days the sample spans, used to express the yield as a daily rate.
    :return: One record per reported quantile.
    """
    if sample.size == 0:
        return []
    rows: List[dict] = []
    for quantile in REPORTED_QUANTILES:
        cosine = float(np.quantile(a=sample, q=quantile))
        record = {
            "quantile": f"p{quantile * 100:g}",
            "cosine": round(cosine, 4),
            "share_of_corpus": round(float(1.0 - quantile), 6),
            "corpus_matches_per_day": round(float(sample.size * (1.0 - quantile) / days), 3)
            if days
            else None
        }
        if eligible is not None and days:
            record["matches_per_day"] = round(float((eligible >= cosine).sum()) / days, 3)
        rows.append(record)
    return rows


def calibration_dates(dates: Sequence, target_days: Optional[int] = None) -> List:
    """
    Pick an evenly spaced subset of dates to calibrate on.

    Even spacing matters: sampling the first hundred days would calibrate on 2015, when the
    corpus was a fraction of its present size and its composition was different.

    :param dates: Every date in the window, in order.
    :param target_days: How many to keep, or None for CALIBRATION_MAX_DAYS.
    :return: The chosen dates, in order.
    """
    wanted = min(target_days or CALIBRATION_MAX_DAYS, len(dates))
    if wanted <= 0:
        return []
    if len(dates) <= wanted:
        return list(dates)
    positions = np.linspace(start=0, stop=len(dates) - 1, num=wanted).round().astype(int)
    return [dates[position] for position in sorted(set(positions.tolist()))]


def resolve_from_tail(floor: FloorSpec, tail_scores: np.ndarray, eligible_events: int) -> float:
    """
    Resolve a high quantile from a descending sample of the corpus tail.

    :param floor: Absolute cosine or quantile specification.
    :param tail_scores: Retrieved high scores; values below the desired rank are unnecessary.
    :param eligible_events: Exact number of events represented by the calibration dates.
    :return: Absolute cosine floor.
    """
    if not is_quantile(floor=floor):
        return float(floor)
    if eligible_events <= 0:
        raise ConfigurationError("quantile calibration found no eligible events")
    scores = np.sort(np.asarray(a=tail_scores, dtype=np.float64))[::-1]
    tail_count = (1.0 - parse_quantile(floor=str(floor))) * eligible_events
    rank = max(1, int(np.ceil(tail_count - 1e-9)))
    if scores.size < rank:
        raise ConfigurationError(
            f"calibration needs the top {rank:,} scores but retrieval returned only "
            f"{scores.size:,}; increase the per-date candidate limit or use a higher quantile"
        )
    return float(scores[rank - 1])


def tail_quantile_table(
    tail_scores: np.ndarray,
    eligible_events: int,
    days: int
) -> List[dict]:
    """Describe the high-quantile tail that the retrieved max-score sample resolves by rank."""
    rows: List[dict] = []
    for quantile in REPORTED_QUANTILES:
        label = f"p{quantile * 100:g}"
        try:
            cosine = resolve_from_tail(
                floor=label,
                tail_scores=tail_scores,
                eligible_events=eligible_events
            )
        except ConfigurationError:
            continue
        rows.append(
            {
                "quantile": label,
                "cosine": round(cosine, 4),
                "share_of_corpus": round(float(1.0 - quantile), 6),
                "corpus_matches_per_day": round(
                    float(eligible_events * (1.0 - quantile) / days),
                    3
                ) if days else None
            }
        )
    return rows
