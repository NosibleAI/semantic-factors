"""Drawing a factor in the NOSIBLE risk-indicator style.

Matplotlib is an optional dependency, so this module is imported lazily by
SemanticFactor.plot() and never at package import.

The layout is transcribed from the NOSIBLE World Risk Indicators deck rather than invented.
Three of its decisions are load-bearing and none are obvious:

**The default is the public-site curve.** Unsigned values receive the reviewed tanh display
squash, then a 30-day geometric mean and a 7-day rolling average. The chart draws that single
green curve by default; arithmetic and quadratic comparisons remain explicit options.

Three trends can be drawn together. For non-negative data they always nest, geometric below
arithmetic below quadratic by the power-mean inequality, so the gap between them is itself a
reading: wide means the window was carried by a few loud days, tight means it was steady.

**Labels are spread evenly across a band above the plot, not placed at their own dates.** Their
leader lines slant down to a marker on the series. Decoupling label position from data position
is the only arrangement in which labels cannot collide, however close together the episodes are.

**The axis is scaled so the trend peak fills TREND_HEADROOM of the height**, and the raw daily
spikes clip above it. Scaling to the tallest daily spike instead lets one crisis set the ceiling
and squashes a decade of trend into the bottom centimetre of the frame.
"""
import datetime as dt
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import polars as pl

matplotlib.use(backend="Agg", force=False)

import matplotlib.dates
import matplotlib.font_manager
import matplotlib.pyplot as plt

#: Brand faces shipped inside the package, so a chart looks the same on a customer's machine
#: as it does here. Missing package data is a broken installation and must not silently change
#: the chart's typography.
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_FILES = ("Inter-Medium.ttf", "Inter-Regular.ttf", "Orbitron-Regular.ttf")

missing_fonts = [
    font_file
    for font_file in FONT_FILES
    if not os.path.isfile(os.path.join(FONT_DIR, font_file))
]
if missing_fonts:
    raise RuntimeError(
        "semantic-factors is missing its packaged chart fonts: " + ", ".join(missing_fonts)
    )
for font_file in FONT_FILES:
    matplotlib.font_manager.fontManager.addfont(os.path.join(FONT_DIR, font_file))

#: Page canvas, matching the deck's 960x540 points.
FIGURE_SIZE = (13.3333, 7.5)
FIGURE_DPI = 144

OBSIDIAN = "#0C0C0E"
NEO_GREEN = "#08D188"
SIGNAL_BLUE = "#4DA8DA"
AMBER = "#F5A623"
TITLE_WHITE = "#F5F7F8"
BODY_GREY = "#B8BDC3"
LABEL_WHITE = "#E1E4E7"
DATE_GREY = "#9CA2A8"
MUTED_GREY = "#777D84"
REGIME_GREY = "#1E2327"
RAW_GREY = "#5A6067"

FONT_DISPLAY = ["Orbitron", "DejaVu Sans", "sans-serif"]
FONT_BODY = ["Inter", "Arial", "DejaVu Sans", "sans-serif"]

#: Axes rectangle as a fraction of the figure, leaving the band above free for labels.
AXES_RECT = (0.058, 0.085, 0.914, 0.594)

#: Where the two label rows sit, as figure fractions: (date baseline, label baseline).
LABEL_ROWS = ((0.794, 0.770), (0.741, 0.717))

#: Horizontal span the labels are distributed across, as figure fractions.
LABEL_SPAN = (0.10, 0.89)

#: Space two labels on the same row must keep between them, as a figure fraction.
LABEL_MIN_GAP = 0.155

#: Calendar window of the inner trend, matching the public website plots.
TREND_DAYS = 30

#: Calendar window used to smooth the inner trend for display.
DISPLAY_TREND_DAYS = 7

#: Observations needed before the inner trend is available.
TREND_MIN_OBSERVATIONS = 5

#: Observations needed before the outer display trend is available.
DISPLAY_TREND_MIN_OBSERVATIONS = 1

#: Bounds and steepness of the visual cosine-like display squash.
PLOT_SQUASH_LOW = 0.25
PLOT_SQUASH_HIGH = 0.50
PLOT_SQUASH_K = 6.0

#: The means that can be drawn, in the order they are layered, with their brand colours.
TREND_COLOURS = {
    "geometric": NEO_GREEN,
    "arithmetic": SIGNAL_BLUE,
    "quadratic": AMBER,
}

#: Means drawn when the caller does not choose.
DEFAULT_TRENDS = ("geometric",)

#: Peaks labelled when the caller names no episodes.
ANNOTATION_COUNT = 10

#: Days that must separate two auto-picked peaks, so one episode does not collect several.
ANNOTATION_SPACING_DAYS = 120

#: Characters a label is truncated to, keeping the two-row annotation band readable.
ANNOTATION_WIDTH = 34

#: Mean the displayed series is rebased to, so differently scaled factors read alike.
REBASE_MEAN = 100.0

#: Share of the plot height the trend peak is scaled to fill.
TREND_HEADROOM = 0.72


def plot_factor(
    daily: pl.DataFrame,
    events: Optional[pl.DataFrame] = None,
    name: str = "factor",
    annotate: bool = True,
    path: Optional[str] = None,
    episodes: Optional[Dict[str, str]] = None,
    subtitle: str = "",
    slug: str = "",
    regimes: Optional[Sequence[Tuple[str, str]]] = None,
    column: str = "value",
    trends: Optional[Sequence[str]] = None,
    signed: bool = False,
    means: Optional[Sequence[str]] = None,
) -> Any:
    """
    Draw a factor as a risk-indicator page.

    :param daily: Daily frame from SemanticFactor.compute.
    :param events: Exemplar events per day, used for labels when no episodes are named.
    :param name: Title, in sentence case.
    :param annotate: Mark the episodes, or the largest peaks when none are named.
    :param path: File to write, or None to leave the figure unsaved.
    :param episodes: Dates you want marked, mapped to their labels, keyed "YYYY-MM-DD" or
        "YYYY-MM". Naming them beats letting the chart guess: you know which episode a peak
        belongs to and the loudest headline of that day often does not say it.
    :param subtitle: One line under the title saying what the factor is for.
    :param slug: Machine name shown in the footer.
    :param regimes: Date ranges to shade, as (start, stop) ISO pairs.
    :param column: Column of the daily frame to draw.
    :param trends: Backward-compatible name for which trailing means to draw.
    :param signed: Preserve a zero-centred signed scale, used for pole polarity.
    :param means: Which trailing means to draw, alone or together; see TREND_COLOURS.
    :return: The matplotlib figure.
    """
    if trends is not None and means is not None:
        raise ValueError("pass either trends or means, not both")
    requested = means if means is not None else trends
    requested = tuple(requested) if requested is not None else DEFAULT_TRENDS
    unknown = [name for name in requested if name not in TREND_COLOURS]
    if unknown:
        raise ValueError(
            f"unknown trend {unknown[0]!r}; expected one of {sorted(TREND_COLOURS)}"
        )
    chosen = list(dict.fromkeys(requested))
    if signed:
        chosen = ["arithmetic"]
    if not chosen:
        raise ValueError(f"trends must name at least one of {sorted(TREND_COLOURS)}")
    frame = prepare(daily=daily, column=column, trends=chosen, signed=signed)
    figure = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor=OBSIDIAN)
    axis = figure.add_axes(AXES_RECT)
    axis.set_facecolor(OBSIDIAN)

    dates: List[dt.date] = frame["date"].to_list()
    stamps = matplotlib.dates.date2num(dates)
    raw = frame["raw_scaled"].to_numpy()
    ceiling = (
        signed_headroom(raw=raw, trend=frame[chosen[-1]].to_numpy())
        if signed
        else headroom(raw=raw, trend=frame[chosen[-1]].to_numpy())
    )

    draw_regimes(axis=axis, regimes=regimes)
    axis.vlines(stamps, 0.0, raw, colors=RAW_GREY, linewidth=0.45, alpha=0.55)
    for drawn in chosen:
        axis.plot(
            stamps,
            frame[drawn].to_numpy(),
            color=TREND_COLOURS[drawn],
            linewidth=1.7 if drawn == "geometric" else 1.1,
            solid_joinstyle="round"
        )
    axis.set_ylim(bottom=-ceiling if signed else 0.0, top=ceiling)
    axis.set_xlim(left=float(stamps[0]), right=float(stamps[-1]))

    style_axes(axis=axis, ceiling=ceiling, dates=dates, signed=signed)
    header(figure=figure, name=name, subtitle=subtitle, regimes=regimes, trends=chosen)
    footer(figure=figure, slug=slug or name.lower().replace(" ", "_"))
    if annotate and not signed:
        draw_labels(
            figure=figure,
            axis=axis,
            frame=frame,
            events=events,
            episodes=episodes,
            trend_column=chosen[-1]
        )
    if path:
        figure.savefig(fname=path, dpi=FIGURE_DPI, facecolor=OBSIDIAN)
    return figure


def prepare(
    daily: pl.DataFrame,
    column: str,
    trends: Sequence[str],
    signed: bool = False,
) -> pl.DataFrame:
    """
    Rebase the series and attach each requested trailing mean.

    :param daily: Daily frame.
    :param column: Column to draw.
    :param trends: Names of the means to compute.
    :param signed: Keep signed values on their natural zero-centred scale.
    :return: Frame of date, raw_scaled, and one column per requested trend.
    """
    if daily.is_empty():
        raise ValueError("there is nothing to plot: the factor matched no days")
    frame = daily.drop_nulls(subset=[column]).sort(by="date")
    if frame.is_empty():
        raise ValueError(f"there is nothing to plot: every {column} is null")
    values = frame[column].to_numpy().astype(np.float64)
    average = float(np.nanmean(values))
    scaled = values if signed else (values / average * REBASE_MEAN if average else values)
    scaled = squash_plot_values(values=scaled, signed=signed)
    dates = frame["date"].to_list()
    return frame.select("date").with_columns(
        pl.Series(name="raw_scaled", values=scaled),
        *[
            pl.Series(name=kind, values=trailing_mean(values=scaled, dates=dates, kind=kind))
            for kind in trends
        ]
    )


def trailing_mean(values: np.ndarray, dates: Sequence[dt.date], kind: str) -> np.ndarray:
    """
    Compute one trailing mean of a daily series over a calendar window.

    Each mean is calculated over 30 calendar days, then averaged over seven calendar days for
    display. The geometric mean is taken on log1p and inverted with expm1. The textbook form is
    zero whenever any day in the window is zero, and for a sparse news factor that would draw a
    line flat on the floor.

    :param values: Daily values, already rebased.
    :param dates: The dates those values fall on, ascending.
    :param kind: One of the names in TREND_COLOURS.
    :return: The trailing mean, nan until the window is populated.
    """
    if kind == "quadratic":
        prepared = np.square(values)
    elif kind == "geometric":
        prepared = np.log1p(np.abs(values))
    elif kind == "arithmetic":
        prepared = values
    else:
        raise ValueError(f"unknown trend {kind!r}; expected one of {sorted(TREND_COLOURS)}")

    inner = rolling_mean(
        values=prepared,
        dates=dates,
        window_days=TREND_DAYS,
        minimum=TREND_MIN_OBSERVATIONS,
    )
    if kind == "quadratic":
        inner = np.sqrt(inner)
    elif kind == "geometric":
        inner = np.expm1(inner)
    return rolling_mean(
        values=inner,
        dates=dates,
        window_days=DISPLAY_TREND_DAYS,
        minimum=DISPLAY_TREND_MIN_OBSERVATIONS,
    )


def rolling_mean(
    values: np.ndarray,
    dates: Sequence[dt.date],
    window_days: int,
    minimum: int,
) -> np.ndarray:
    """Calculate a calendar-window arithmetic mean over finite observations."""
    output = np.full(shape=values.shape, fill_value=np.nan, dtype=np.float64)
    start = 0
    for index, today in enumerate(dates):
        while start < index and (today - dates[start]).days >= window_days:
            start += 1
        window = values[start:index + 1]
        finite = window[np.isfinite(window)]
        if finite.size >= minimum:
            output[index] = float(finite.mean())
    return output


def squash_plot_values(values: np.ndarray, signed: bool = False) -> np.ndarray:
    """Apply the public website's bounded tanh display transform.

    The native daily factor remains unchanged. For plotting, the first and ninety-ninth
    percentiles are mapped to the reviewed 0.25--0.50 cosine interval, then squashed with
    ``tanh(k * (x - 0.5))`` at ``k=6`` and rebased to a mean of 100.
    """
    if signed or not values.size:
        return values
    finite = values[np.isfinite(values)]
    if not finite.size:
        return values
    lower, upper = np.quantile(finite, [0.01, 0.99])
    if upper <= lower:
        return np.full_like(values, REBASE_MEAN)
    proxy = PLOT_SQUASH_LOW + (PLOT_SQUASH_HIGH - PLOT_SQUASH_LOW) * np.clip(
        (np.nan_to_num(values, nan=0.0) - lower) / (upper - lower),
        0.0,
        1.0,
    )
    unit = np.clip(
        (proxy - PLOT_SQUASH_LOW) / (PLOT_SQUASH_HIGH - PLOT_SQUASH_LOW),
        0.0,
        1.0,
    )
    half = np.tanh(PLOT_SQUASH_K / 2.0)
    squashed = (np.tanh(PLOT_SQUASH_K * (unit - 0.5)) + half) / (2.0 * half)
    squashed[~np.isfinite(values)] = np.nan
    average = float(np.nanmean(squashed))
    return squashed / average * REBASE_MEAN if average else squashed


def headroom(raw: np.ndarray, trend: np.ndarray) -> float:
    """
    Scale the axis so the trend peak fills TREND_HEADROOM of it.

    :param raw: Rebased daily values, used only when there is no trend to scale from.
    :param trend: The quadratic-mean trend.
    :return: Upper y-limit.
    """
    peak = float(np.nanmax(trend)) if np.isfinite(trend).any() else 0.0
    if peak > 0:
        return peak / TREND_HEADROOM
    return float(np.nanmax(raw)) or 1.0


def signed_headroom(raw: np.ndarray, trend: np.ndarray) -> float:
    """Return a symmetric limit that preserves both poles around zero."""
    finite_trend = np.abs(trend[np.isfinite(trend)])
    peak = float(finite_trend.max()) if finite_trend.size else 0.0
    if peak > 0:
        return peak / TREND_HEADROOM
    finite_raw = np.abs(raw[np.isfinite(raw)])
    return float(finite_raw.max()) if finite_raw.size and finite_raw.max() > 0 else 1.0


def draw_regimes(axis: Any, regimes: Optional[Sequence[Tuple[str, str]]]) -> None:
    """
    Shade the date ranges the caller flagged as elevated regimes.

    :param axis: Axes to draw on.
    :param regimes: Date ranges as (start, stop) ISO pairs, or None.
    :return: None.
    """
    for start, stop in regimes or ():
        axis.axvspan(
            matplotlib.dates.date2num(dt.date.fromisoformat(start)),
            matplotlib.dates.date2num(dt.date.fromisoformat(stop)),
            color=REGIME_GREY,
            linewidth=0,
            zorder=0
        )


def style_axes(
    axis: Any,
    ceiling: float,
    dates: Sequence[dt.date],
    signed: bool = False,
) -> None:
    """
    Apply the deck's axis treatment: no spines, five ticks, faint year rules.

    :param axis: Axes to style.
    :param ceiling: Top of the axis.
    :param dates: The dates plotted, for the year rules.
    :param signed: Draw symmetric negative-to-positive tick labels.
    :return: None.
    """
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(axis="both", length=0, pad=6)
    ticks = (
        [ceiling * step / 2 for step in range(-2, 3)]
        if signed
        else [ceiling * step / 4 for step in range(5)]
    )
    axis.set_yticks(ticks)
    axis.set_yticklabels(
        [f"{tick:#.4g}" for tick in ticks],
        fontsize=7.5,
        fontfamily=FONT_BODY,
        color=BODY_GREY
    )
    years = [dt.date(year, 1, 1) for year in range(dates[0].year + 1, dates[-1].year + 1)]
    stamps = [float(matplotlib.dates.date2num(year)) for year in years]
    axis.set_xticks(stamps)
    axis.set_xticklabels(
        [str(year.year) for year in years],
        fontsize=8.2,
        fontfamily=FONT_BODY,
        fontweight="bold",
        color=BODY_GREY
    )
    for stamp in stamps:
        axis.axvline(x=stamp, color=REGIME_GREY, linewidth=0.6, alpha=0.9, zorder=0)


def header(
    figure: Any,
    name: str,
    subtitle: str,
    regimes: Optional[Sequence[Tuple[str, str]]],
    trends: Sequence[str]
) -> None:
    """
    Draw the title block and the legend.

    :param figure: Figure to draw on.
    :param name: Title, in sentence case.
    :param subtitle: One line under the title.
    :param regimes: Whether a regime swatch belongs in the legend.
    :param trends: The trends drawn, for the legend.
    :return: None.
    """
    figure.patches.append(
        plt.Rectangle(
            xy=(0.036, 0.945),
            width=0.006,
            height=0.011,
            transform=figure.transFigure,
            facecolor=NEO_GREEN,
            edgecolor="none"
        )
    )
    figure.text(
        0.036, 0.855, name,
        fontsize=26, fontfamily=FONT_DISPLAY, color=TITLE_WHITE, va="bottom"
    )
    if subtitle:
        figure.text(
            0.036, 0.831, subtitle.upper(),
            fontsize=6.8, fontfamily=FONT_BODY, color=BODY_GREY, va="top"
        )
    legend(figure=figure, regimes=regimes, trends=trends)


def legend(
    figure: Any,
    regimes: Optional[Sequence[Tuple[str, str]]],
    trends: Sequence[str]
) -> None:
    """
    Draw the right-aligned legend strip.

    The default label names both stages of the public plotting recipe. Optional comparison
    curves use the same 30-day inner and 7-day outer windows.

    :param figure: Figure to draw on.
    :param regimes: Whether a regime swatch is included.
    :param trends: The trends drawn.
    :return: None.
    """
    entries = [("RAW DAILY", RAW_GREY, "line")]
    for kind in trends:
        entries.append(
            (
                f"{DISPLAY_TREND_DAYS}D AVG OF {TREND_DAYS}D {kind.upper()}",
                TREND_COLOURS[kind],
                "line",
            )
        )
    if regimes:
        entries.append(("REGIME", REGIME_GREY, "patch"))
    cursor = 0.972
    for text, colour, kind in reversed(entries):
        figure.text(
            cursor, 0.845, text,
            fontsize=7.1, fontfamily=FONT_BODY, color=BODY_GREY, ha="right", va="center"
        )
        mark = cursor - 0.0043 * len(text) - 0.028
        if kind == "line":
            figure.add_artist(
                plt.Line2D(
                    xdata=[mark, mark + 0.017], ydata=[0.845, 0.845],
                    transform=figure.transFigure, color=colour,
                    linewidth=0.8 if colour == RAW_GREY else 2.0
                )
            )
        else:
            figure.patches.append(
                plt.Rectangle(
                    xy=(mark, 0.839), width=0.017, height=0.012,
                    transform=figure.transFigure, facecolor=colour, edgecolor="none"
                )
            )
        cursor = mark - 0.020


def footer(figure: Any, slug: str) -> None:
    """
    Draw the machine name in the footer.

    :param figure: Figure to draw on.
    :param slug: Machine name.
    :return: None.
    """
    figure.text(
        0.036, 0.022, slug.upper(),
        fontsize=6.0, fontfamily=FONT_BODY, color=MUTED_GREY, va="center"
    )


def draw_labels(
    figure: Any,
    axis: Any,
    frame: pl.DataFrame,
    events: Optional[pl.DataFrame],
    episodes: Optional[Dict[str, str]],
    trend_column: str
) -> None:
    """
    Place the labels in the band above the plot and run leaders down to the series.

    Label positions are spread evenly across the width regardless of when the episodes happened,
    and the leaders slant to reach them. Two episodes a week apart get labels a tenth of the page
    apart, which is the only way a fixed band holds ten of them.

    :param figure: Figure to draw on.
    :param axis: Axes the series is drawn on.
    :param frame: Frame of date, raw_scaled and trend.
    :param events: Exemplar events per day, or None.
    :param episodes: Named dates to mark, or None to pick peaks.
    :param trend_column: The trend the leaders should land on.
    :return: None.
    """
    # A date without an event headline is not an explanation.  Explicit episode labels are
    # still valid, but automatic annotations must be backed by the event exemplars returned by
    # the factor computation.
    if not episodes and (
        events is None or events.is_empty() or "title" not in events.columns
    ):
        return
    marks = (
        episode_marks(episodes=episodes, frame=frame, trend_column=trend_column)
        if episodes
        else peak_marks(frame=frame, events=events)
    )
    if not marks:
        return
    positions = label_positions(
        figure=figure,
        axis=axis,
        marks=marks
    )
    for index, (date, value, text) in enumerate(marks):
        position = positions[index]
        date_y, label_y = LABEL_ROWS[index % len(LABEL_ROWS)]
        figure.text(
            position, date_y, date.isoformat(),
            fontsize=5.6, fontfamily=FONT_BODY, color=DATE_GREY, ha="center", va="center"
        )
        figure.text(
            position, label_y, shorten(text=text).upper(),
            fontsize=6.2, fontfamily=FONT_BODY, fontweight="bold",
            color=LABEL_WHITE, ha="center", va="center"
        )
        leader(
            figure=figure,
            axis=axis,
            start=(position, label_y - 0.017),
            target=(date, value)
        )
        axis.plot(
            [visible_target(axis=axis, date=date, value=value)[0]],
            [visible_target(axis=axis, date=date, value=value)[1]],
            marker="o", markersize=2.6, markerfacecolor="none",
            markeredgecolor=BODY_GREY, markeredgewidth=0.6, zorder=5
        )


def label_positions(
    figure: Any,
    axis: Any,
    marks: Sequence[Tuple[dt.date, float, str]]
) -> List[float]:
    """
    Choose where each label sits along the top band.

    Start each label above its own date, so the leader is as short as it can be, then push
    same-row neighbours apart until none overlap. Spreading them evenly instead is simpler but
    drags a label for a 2025 episode halfway across a decade of chart.

    :param figure: Figure the labels are drawn on.
    :param axis: Axes the series is drawn on.
    :param marks: The (date, value, label) triples, in date order.
    :return: One figure-fraction x per mark, in the same order.
    """
    wanted = [
        float(
            figure.transFigure.inverted().transform(
                axis.transData.transform(
                    (visible_target(axis=axis, date=date, value=axis.get_ylim()[0])[0], axis.get_ylim()[0])
                )
            )[0]
        )
        for date, _, _ in marks
    ]
    positions = list(wanted)
    for row in range(len(LABEL_ROWS)):
        members = list(range(row, len(marks), len(LABEL_ROWS)))
        for slot, index in enumerate(members):
            lowest = LABEL_SPAN[0] + slot * LABEL_MIN_GAP
            positions[index] = max(positions[index], lowest)
        for slot in range(len(members) - 2, -1, -1):
            index, later = members[slot], members[slot + 1]
            highest = min(LABEL_SPAN[1], positions[later] - LABEL_MIN_GAP)
            positions[index] = min(positions[index], highest)
        for index in members:
            positions[index] = min(max(positions[index], LABEL_SPAN[0]), LABEL_SPAN[1])
    return positions


def leader(
    figure: Any,
    axis: Any,
    start: Tuple[float, float],
    target: Tuple[dt.date, float]
) -> None:
    """
    Run a leader from a label down to its point on the series.

    Two segments: a diagonal that carries the line across to the episode's date, then a vertical
    drop onto the marker. The vertical tail is what tells a reader which date the label belongs
    to, since the label itself is nowhere near it.

    :param figure: Figure to draw on.
    :param axis: Axes the series is drawn on.
    :param start: Label anchor, in figure fractions.
    :param target: The point on the series, in data coordinates.
    :return: None.
    """
    visible_date, visible_value = visible_target(
        axis=axis,
        date=target[0],
        value=target[1],
    )
    point = (visible_date, visible_value)
    corner = figure.transFigure.inverted().transform(axis.transData.transform(point))
    elbow_y = start[1] - (start[1] - corner[1]) * 0.45
    figure.add_artist(
        plt.Line2D(
            xdata=[start[0], corner[0], corner[0]],
            ydata=[start[1], elbow_y, corner[1]],
            transform=figure.transFigure,
            color=MUTED_GREY,
            linewidth=0.55,
            solid_joinstyle="miter"
        )
    )


def visible_y(axis: Any, value: float) -> float:
    """Clamp a plotted value to the visible y-range of an axis."""
    lower, upper = sorted(axis.get_ylim())
    finite = float(value) if np.isfinite(value) else lower
    return float(np.clip(finite, lower, upper))


def visible_target(axis: Any, date: dt.date, value: float) -> Tuple[float, float]:
    """Return a data point clipped to the visible rectangle of an axis."""
    left, right = sorted(axis.get_xlim())
    stamp = float(matplotlib.dates.date2num(date))
    return float(np.clip(stamp, left, right)), visible_y(axis=axis, value=value)


def episode_marks(
    episodes: Dict[str, str],
    frame: pl.DataFrame,
    trend_column: str
) -> List[Tuple[dt.date, float, str]]:
    """
    Resolve named episodes onto the series, so each leader ends on a real point.

    :param episodes: Dates mapped to labels, keyed "YYYY-MM-DD" or "YYYY-MM".
    :param frame: Frame of date, raw_scaled and the drawn trends.
    :param trend_column: The trend the leaders should land on.
    :return: List of (date, value, label) triples in date order.
    """
    lookup = dict(zip(frame["date"].to_list(), frame[trend_column].to_list(), strict=True))
    dates = sorted(lookup)
    marks: List[Tuple[dt.date, float, str]] = []
    for key, label in episodes.items():
        stamp = dt.date.fromisoformat(key if len(key) > 7 else f"{key}-01")
        nearest = min(dates, key=lambda item: abs((item - stamp).days))
        value = lookup[nearest]
        marks.append((stamp, 0.0 if value is None or np.isnan(value) else float(value), label))
    return sorted(marks)


def peak_marks(
    frame: pl.DataFrame,
    events: Optional[pl.DataFrame]
) -> List[Tuple[dt.date, float, str]]:
    """
    Pick the largest well-separated peaks and label them from the exemplar events.

    :param frame: Frame of date, raw_scaled and the drawn trends.
    :param events: Exemplar events per day, or None.
    :return: List of (date, value, label) triples in date order.
    """
    titles: Dict[dt.date, str] = {}
    if events is not None and not events.is_empty():
        best = events.sort(by=["date", "weight"], descending=[False, True]).group_by(
            "date", maintain_order=True
        ).head(1)
        titles = dict(zip(best["date"].to_list(), best["title"].to_list(), strict=True))
    ranked = frame.drop_nulls(subset=["raw_scaled"]).sort(by="raw_scaled", descending=True)
    chosen: List[Tuple[dt.date, float, str]] = []
    for date, value in zip(ranked["date"].to_list(), ranked["raw_scaled"].to_list(), strict=True):
        if len(chosen) >= ANNOTATION_COUNT:
            break
        if any(abs((date - taken).days) < ANNOTATION_SPACING_DAYS for taken, _, _ in chosen):
            continue
        title = titles.get(date)
        if title:
            chosen.append((date, float(value), str(title)))
    return sorted(chosen)


def shorten(text: str) -> str:
    """
    Trim a label to the annotation width, at a word boundary where possible.

    :param text: Label text.
    :return: Shortened text.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) <= ANNOTATION_WIDTH:
        return cleaned
    return cleaned[:ANNOTATION_WIDTH].rsplit(" ", 1)[0]
