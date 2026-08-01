"""Chart construction, which is where a series is most easily misread."""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from semantic_factors import series

plotting = pytest.importorskip(modname="semantic_factors.plotting")

START = dt.date(2020, 1, 1)


def test_packaged_brand_fonts_are_registered_without_fallback() -> None:
    """Require an installed chart to use both NOSIBLE brand families."""
    for family in ("Inter", "Orbitron"):
        path = plotting.matplotlib.font_manager.findfont(
            plotting.matplotlib.font_manager.FontProperties(family=family),
            fallback_to_default=False
        )
        assert family.lower() in path.lower()


def daily_from(values: list) -> pl.DataFrame:
    """
    Build a daily frame carrying the given values.

    :param values: One value per consecutive day.
    :return: Daily frame.
    """
    days = [
        {
            "date": (START + dt.timedelta(days=step)).isoformat(),
            "raw": float(value),
            "volume": 100.0,
            "n_events": 1 if value else 0,
            "n_eligible": 100,
            "polarity_num": 0.0,
            "polarity_weight": 0.0
        }
        for step, value in enumerate(values)
    ]
    return series.daily_frame(
        days=days,
        start=START,
        stop=START + dt.timedelta(days=len(values) - 1),
        stabilize=False
    )


def test_the_three_means_nest_by_the_power_mean_inequality() -> None:
    """
    Require geometric <= arithmetic <= quadratic on every day that carries all three.

    They are the same window summarised three ways, so any day that breaks the ordering is a
    window or sign bug rather than a fact about the data.

    :return: None.
    """
    rng = np.random.default_rng(11)
    daily = daily_from(values=list(rng.gamma(shape=0.4, scale=9.0, size=400)))

    frame = plotting.prepare(
        daily=daily,
        column="value",
        trends=["geometric", "arithmetic", "quadratic"]
    )
    low, middle, high = (frame[kind].to_numpy() for kind in
                         ("geometric", "arithmetic", "quadratic"))
    complete = np.isfinite(low) & np.isfinite(middle) & np.isfinite(high)

    assert complete.sum() > 250
    assert np.all(low[complete] <= middle[complete] + 1e-9)
    assert np.all(middle[complete] <= high[complete] + 1e-9)


def test_the_quadratic_mean_registers_a_spike_a_plain_average_dilutes() -> None:
    """
    Require the quadratic mean to sit above the arithmetic one when a window has a loud day.

    This is the whole reason the published indicators use it.

    :return: None.
    """
    steady = daily_from(values=[10.0] * 120)
    spiky = daily_from(values=[10.0] * 119 + [400.0])

    calm = plotting.prepare(daily=steady, column="value", trends=["arithmetic", "quadratic"])
    loud = plotting.prepare(daily=spiky, column="value", trends=["arithmetic", "quadratic"])

    assert calm["quadratic"][-1] == pytest.approx(calm["arithmetic"][-1], rel=1e-9)
    assert loud["quadratic"][-1] > loud["arithmetic"][-1] * 1.5


def test_the_geometric_mean_survives_a_window_full_of_zeros() -> None:
    """
    Require a sparse factor to still get a geometric line rather than one flat on the floor.

    The textbook geometric mean is zero if any day in the window is zero, which for a factor
    matching a handful of events a day is nearly every window.

    :return: None.
    """
    daily = daily_from(values=([0.0] * 9 + [50.0]) * 20)

    frame = plotting.prepare(daily=daily, column="value", trends=["geometric"])
    drawn = frame["geometric"].to_numpy()

    assert np.isfinite(drawn).sum() > 150
    assert np.nanmax(drawn) > 0.0


def test_the_axis_is_scaled_from_the_trend_not_the_tallest_spike() -> None:
    """
    Require the trend to fill the stated share of the frame.

    One loud day setting the ceiling squashes a decade of trend into the bottom of the plot.

    :return: None.
    """
    ceiling = plotting.headroom(
        raw=np.asarray(a=[1.0] * 999 + [500.0]),
        trend=np.asarray(a=[1.0] * 1000)
    )

    assert ceiling == pytest.approx(1.0 / plotting.TREND_HEADROOM)


def test_a_degenerate_trend_falls_back_to_the_daily_peak() -> None:
    """
    Require a window too short to have a trend to still produce a usable axis.

    :return: None.
    """
    ceiling = plotting.headroom(
        raw=np.asarray(a=[2.0, 3.0]),
        trend=np.asarray(a=[np.nan, np.nan])
    )

    assert ceiling == 3.0


def test_labels_are_spread_far_enough_apart_to_read() -> None:
    """
    Require same-row labels to keep the minimum gap even when the episodes are days apart.

    Placing each label at its own date is what made them overlap; the leaders carry the
    difference instead.

    :return: None.
    """
    daily = daily_from(values=[10.0] * 800)
    figure = plotting.plot_factor(
        daily=daily,
        name="probe",
        episodes={
            (START + dt.timedelta(days=step)).isoformat(): f"episode {step}"
            for step in (100, 104, 108, 112, 116, 120)
        }
    )
    positions = [
        text.get_position()[0]
        for text in figure.texts
        if text.get_text().startswith("EPISODE")
    ]

    assert len(positions) == 6
    for row in (positions[0::2], positions[1::2]):
        for earlier, later in zip(sorted(row)[:-1], sorted(row)[1:], strict=True):
            assert later - earlier >= plotting.LABEL_MIN_GAP - 1e-9


def test_peaks_are_labelled_once_per_episode() -> None:
    """
    Require nearby peaks to share one label, so an episode does not collect several.

    :return: None.
    """
    values = [1.0] * 900
    for step in (300, 302, 305, 700):
        values[step] = 50.0
    frame = plotting.prepare(daily=daily_from(values=values), column="value", trends=["quadratic"])

    marks = plotting.peak_marks(frame=frame, events=None)
    dates = [date for date, _, _ in marks]

    assert len(dates) == len(set(dates))
    for earlier in dates:
        for later in dates:
            if earlier != later:
                assert abs((earlier - later).days) >= plotting.ANNOTATION_SPACING_DAYS


def test_a_long_label_is_trimmed_at_a_word_boundary() -> None:
    """
    Require truncation not to cut a word in half.

    :return: None.
    """
    result = plotting.shorten(text="Unprecedented " * 12)

    assert len(result) <= plotting.ANNOTATION_WIDTH
    assert not result.endswith("Unprecedente")


def test_a_short_label_is_left_alone() -> None:
    """
    Require a label that fits to survive untouched.

    :return: None.
    """
    assert plotting.shorten(text="Locusts hit Kenya") == "Locusts hit Kenya"


def test_an_unknown_trend_is_rejected() -> None:
    """
    Require a mistyped trend name to fail rather than silently draw nothing.

    :return: None.
    """
    with pytest.raises(ValueError):
        plotting.plot_factor(daily=daily_from(values=[1.0] * 50), trends=["median"])


def test_plotting_an_empty_factor_explains_itself() -> None:
    """
    Require a factor with no days to say so rather than draw an empty canvas.

    :return: None.
    """
    with pytest.raises(ValueError):
        plotting.plot_factor(daily=series.empty_daily(), name="nothing", annotate=False)
