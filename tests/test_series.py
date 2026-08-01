"""Turning daily sufficient statistics into a factor time series."""
import datetime as dt

from semantic_factors import series

START = dt.date(2020, 1, 1)


def days(count: int, raw: float = 1.0, volume: float = 100.0) -> list:
    """
    Build a run of identical day records.

    :param count: How many consecutive days to build.
    :param raw: Weighted mass of matching events each day.
    :param volume: Eligible weight each day.
    :return: List of day records.
    """
    return [
        {
            "date": (START + dt.timedelta(days=step)).isoformat(),
            "raw": raw,
            "volume": volume,
            "n_events": 1,
            "n_eligible": int(volume),
            "polarity_num": 0.0,
            "polarity_weight": 0.0
        }
        for step in range(count)
    ]


def test_share_is_the_same_day_ratio() -> None:
    """
    Require share to divide by the same day's volume.

    :return: None.
    """
    frame = series.daily_frame(
        days=days(count=10, raw=5.0, volume=100.0),
        start=START,
        stop=START + dt.timedelta(days=9),
        stabilize=False
    )

    assert frame["share"].to_list() == [0.05] * 10


def test_stabilising_waits_for_enough_history() -> None:
    """
    Require the stabilised series to start null until the trailing baseline is trustworthy.

    A baseline built from three days is noise, and dividing by it manufactures spikes.

    :return: None.
    """
    frame = series.daily_frame(
        days=days(count=200),
        start=START,
        stop=START + dt.timedelta(days=199),
        stabilize=True
    )
    values = frame["value"].to_list()

    assert values[: series.STABILISE_MIN_DAYS - 1] == [None] * (series.STABILISE_MIN_DAYS - 1)
    assert values[series.STABILISE_MIN_DAYS - 1] is not None


def test_stabilising_removes_a_growing_corpus() -> None:
    """
    Require a factor whose share is constant to stay flat while the corpus grows tenfold.

    This is the whole reason stabilisation exists: without it, a series that measured nothing
    would still trend upwards simply because NOSIBLE World collects more every year.

    :return: None.
    """
    growing = [
        {
            "date": (START + dt.timedelta(days=step)).isoformat(),
            "raw": 0.05 * (100.0 + step * 5.0),
            "volume": 100.0 + step * 5.0,
            "n_events": 1,
            "n_eligible": 1,
            "polarity_num": 0.0,
            "polarity_weight": 0.0
        }
        for step in range(400)
    ]
    frame = series.daily_frame(
        days=growing,
        start=START,
        stop=START + dt.timedelta(days=399),
        stabilize=True
    )
    values = [value for value in frame["value"].to_list() if value is not None]

    assert max(values) / min(values) < 2.0


def test_a_day_with_no_coverage_is_dropped_not_zeroed() -> None:
    """
    Require a day the corpus never collected to leave a gap rather than read as a quiet day.

    :return: None.
    """
    records = days(count=3)
    records[1]["volume"] = 0.0
    frame = series.daily_frame(
        days=records,
        start=START,
        stop=START + dt.timedelta(days=2),
        stabilize=False
    )

    assert frame.height == 2
    assert START + dt.timedelta(days=1) not in frame["date"].to_list()


def test_a_covered_day_with_no_match_is_kept_at_zero() -> None:
    """
    Require a day where nothing happened to be a real zero observation.

    :return: None.
    """
    records = days(count=3)
    records[1]["raw"] = 0.0
    records[1]["n_events"] = 0
    frame = series.daily_frame(
        days=records,
        start=START,
        stop=START + dt.timedelta(days=2),
        stabilize=False
    )

    assert frame.height == 3
    assert frame["value"].to_list()[1] == 0.0


def test_polarity_is_the_weighted_mean_of_the_events_that_had_one() -> None:
    """
    Require polarity to average over its own weight, not over every matching event.

    :return: None.
    """
    records = days(count=1)
    records[0]["polarity_num"] = 1.5
    records[0]["polarity_weight"] = 3.0
    frame = series.daily_frame(
        days=records,
        start=START,
        stop=START,
        stabilize=False
    )

    assert frame["polarity"].to_list() == [0.5]


def test_days_outside_the_window_are_discarded() -> None:
    """
    Require a response that overshoots the requested window to be trimmed.

    :return: None.
    """
    frame = series.daily_frame(
        days=days(count=10),
        start=START + dt.timedelta(days=2),
        stop=START + dt.timedelta(days=4),
        stabilize=False
    )

    assert frame.height == 3


def test_resampling_rebuilds_ratios_from_sums() -> None:
    """
    Require a monthly share to come from summed numerators and denominators.

    Averaging daily ratios would weight a quiet Sunday the same as a busy Tuesday.

    :return: None.
    """
    records = days(count=31, raw=1.0, volume=10.0)
    records[0]["raw"] = 100.0
    records[0]["volume"] = 100.0
    frame = series.daily_frame(
        days=records,
        start=START,
        stop=START + dt.timedelta(days=30),
        stabilize=False
    )

    monthly = series.resample(daily=frame, every="1mo")

    assert monthly.height == 1
    assert abs(monthly["share"][0] - 130.0 / 400.0) < 1e-9


def test_an_empty_result_still_has_the_right_schema() -> None:
    """
    Require a factor that matched nothing to return a usable empty frame.

    :return: None.
    """
    frame = series.daily_frame(days=[], start=START, stop=START, stabilize=True)

    assert frame.is_empty()
    assert list(frame.columns) == list(series.DAILY_COLUMNS)


def test_standardising_centres_the_series() -> None:
    """
    Require the z-score column to have zero mean.

    :return: None.
    """
    records = days(count=10)
    for step, record in enumerate(records):
        record["raw"] = float(step + 1)
    frame = series.standardise(
        daily=series.daily_frame(
            days=records,
            start=START,
            stop=START + dt.timedelta(days=9),
            stabilize=False
        )
    )

    assert abs(float(frame["value_z"].mean())) < 1e-9
