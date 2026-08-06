"""Chart construction, which is where a series is most easily misread."""
import datetime as dt
import warnings

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


def test_inter_fonts_cover_every_printable_ascii_glyph() -> None:
    """Catch truncated or accidentally substituted font binaries before a wheel ships."""
    required = set(range(32, 127))
    for filename in ("Inter-Medium.ttf", "Inter-Regular.ttf"):
        path = plotting.os.path.join(plotting.FONT_DIR, filename)
        available = set(plotting.matplotlib.ft2font.FT2Font(path).get_charmap())
        assert not required - available


def test_representative_chart_renders_without_missing_glyphs() -> None:
    """Exercise the actual canvas with the letters, numbers, and punctuation charts use."""
    daily = daily_from(values=[float(index % 17) for index in range(180)])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r"Glyph .* missing from font")
        figure = plotting.plot_factor(
            daily=daily,
            name="Supply Risk 2026: Africa / Europe (P99.9)",
            episodes={"2020-03-15": "Factories close; prices rise 42%"},
        )
        figure.canvas.draw()


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


def test_means_keyword_selects_a_single_geometric_trend_by_default() -> None:
    """Expose a readable single-mean plot while retaining explicit choices."""
    assert plotting.DEFAULT_TRENDS == ("geometric",)
    assert plotting.TREND_COLOURS["geometric"] == plotting.NEO_GREEN

    daily = daily_from(values=[float(index % 17) for index in range(180)])
    frame = plotting.prepare(daily=daily, column="value", trends=plotting.DEFAULT_TRENDS)
    assert frame.columns == ["date", "raw_scaled", "geometric"]

    with pytest.raises(ValueError, match="either trends or means"):
        plotting.plot_factor(
            daily=daily,
            trends=["geometric"],
            means=["geometric"],
            annotate=False,
        )


def test_default_curve_matches_the_public_tanh_30d_geometric_7d_recipe() -> None:
    """Keep package plots numerically identical to the reviewed website curve."""
    daily = daily_from(values=[float((index * 11) % 37) for index in range(90)])
    frame = plotting.prepare(daily=daily, column="value", trends=plotting.DEFAULT_TRENDS)

    native = daily["value"].to_numpy().astype(np.float64)
    rebased = native / float(native.mean()) * plotting.REBASE_MEAN
    lower, upper = np.quantile(rebased, [0.01, 0.99])
    unit = np.clip((rebased - lower) / (upper - lower), 0.0, 1.0)
    half = np.tanh(plotting.PLOT_SQUASH_K / 2.0)
    squashed = (
        np.tanh(plotting.PLOT_SQUASH_K * (unit - 0.5)) + half
    ) / (2.0 * half)
    squashed = squashed / float(squashed.mean()) * plotting.REBASE_MEAN

    inner = np.full(shape=squashed.shape, fill_value=np.nan)
    for index in range(4, len(squashed)):
        window = squashed[max(0, index - 29):index + 1]
        inner[index] = np.expm1(np.log1p(window).mean())
    expected = np.full(shape=squashed.shape, fill_value=np.nan)
    for index in range(4, len(squashed)):
        window = inner[max(0, index - 6):index + 1]
        expected[index] = np.nanmean(window)

    assert np.allclose(frame["raw_scaled"].to_numpy(), squashed)
    assert np.allclose(frame["geometric"].to_numpy(), expected, equal_nan=True)

    figure = plotting.plot_factor(daily=daily, annotate=False)
    green_lines = [
        line for line in figure.axes[0].lines if line.get_color() == plotting.NEO_GREEN
    ]
    assert len(green_lines) == 1
    assert any(
        text.get_text() == "7D AVG OF 30D GEOMETRIC" for text in figure.texts
    )


def test_the_quadratic_mean_registers_a_spike_a_plain_average_dilutes() -> None:
    """
    Require the quadratic mean to sit above the arithmetic one when a window has a loud day.

    This is the whole reason the published indicators use it.

    :return: None.
    """
    steady = daily_from(values=[10.0] * 120)
    spiky = daily_from(values=list(np.linspace(1.0, 20.0, 119)) + [400.0])

    calm = plotting.prepare(daily=steady, column="value", trends=["arithmetic", "quadratic"])
    loud = plotting.prepare(daily=spiky, column="value", trends=["arithmetic", "quadratic"])

    assert calm["quadratic"][-1] == pytest.approx(calm["arithmetic"][-1], rel=1e-9)
    assert loud["quadratic"][-1] > loud["arithmetic"][-1]


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


def test_annotation_leaders_clip_raw_spikes_to_the_visible_axis() -> None:
    """Require off-scale raw spikes and out-of-range dates to terminate at the chart edge."""
    figure = plotting.plt.figure()
    axis = figure.add_axes([0.1, 0.1, 0.8, 0.6])
    left = float(plotting.matplotlib.dates.date2num(START))
    right = float(plotting.matplotlib.dates.date2num(START + dt.timedelta(days=10)))
    axis.set_xlim(left=left, right=right)
    axis.set_ylim(bottom=0.0, top=10.0)

    assert plotting.visible_target(
        axis=axis, date=START - dt.timedelta(days=1), value=25.0
    ) == pytest.approx((left, 10.0))
    assert plotting.visible_target(
        axis=axis, date=START + dt.timedelta(days=11), value=-5.0
    ) == pytest.approx((right, 0.0))
    assert plotting.visible_y(axis=axis, value=float("nan")) == pytest.approx(0.0)
    plotting.plt.close(figure)


def test_auto_annotations_require_event_headlines() -> None:
    """Require automatic annotations to have event data to explain."""
    daily = daily_from(values=[10.0] * 180)
    figure = plotting.plot_factor(daily=daily, annotate=True, events=None)
    assert not any(text.get_text() == START.isoformat() for text in figure.texts)
    plotting.plt.close(figure)


def test_auto_annotations_show_the_event_headline() -> None:
    """Require an exemplar title to be rendered beneath its date."""
    daily = daily_from(values=[10.0] * 180)
    events = pl.DataFrame({
        "date": [START],
        "weight": [1.0],
        "title": ["Factories close; prices rise"],
    })
    figure = plotting.plot_factor(daily=daily, annotate=True, events=events)
    assert any(text.get_text() == "FACTORIES CLOSE; PRICES RISE" for text in figure.texts)
    plotting.plt.close(figure)


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


def test_signed_polarity_plot_preserves_negative_values_and_symmetric_axis() -> None:
    """Never clip, square, or mean-rebase the low pole out of a polarity chart."""
    values = np.linspace(-0.9, 0.7, 120)
    daily = daily_from(values=[1.0] * len(values)).with_columns(
        pl.Series(name="polarity", values=values)
    )

    prepared = plotting.prepare(
        daily=daily,
        column="polarity",
        trends=["arithmetic"],
        signed=True,
    )
    figure = plotting.plot_factor(
        daily=daily,
        name="signed probe",
        annotate=False,
        column="polarity",
        signed=True,
    )
    low, high = figure.axes[0].get_ylim()

    assert prepared["raw_scaled"].min() == pytest.approx(-0.9)
    assert prepared["raw_scaled"].max() == pytest.approx(0.7)
    assert low < -0.9
    assert high > 0.7
    assert low == pytest.approx(-high)
