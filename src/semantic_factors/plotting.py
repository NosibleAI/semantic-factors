"""Drawing a factor in the NOSIBLE risk-indicator style.

Matplotlib is an optional dependency, so this module is imported lazily by
SemanticFactor.plot() and never at package import.

The layout is transcribed from the NOSIBLE World Risk Indicators deck rather than invented.
Three of its decisions are load-bearing and none are obvious:

**The trend is a quadratic mean, not an average.** A root-mean-square over the window weights
loud days by their square, so an episode lifts the line and a quiet fortnight does not drag it
back as fast. A plain moving average of a spiky news series understates every episode it is
supposed to describe.

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

#: Window of the trailing means, in days.
TREND_DAYS = 90

#: The trends that can be drawn, in the order they are layered, with their colours. Quadratic
#: is drawn last and in the brand green because it is the one the published indicators use.
TREND_COLOURS = {
    "geometric": AMBER,
    "arithmetic": SIGNAL_BLUE,
    "quadratic": NEO_GREEN
}

#: Trends drawn when the caller does not choose.
DEFAULT_TRENDS = ("arithmetic", "geometric", "quadratic")

#: Days in the window that must carry a value before the trend is drawn.
TREND_MIN_OBSERVATIONS = 10

#: Peaks labelled when the caller names no episodes.
ANNOTATION_COUNT = 10

#: Days that must separate two auto-picked peaks, so one episode does not collect several.
ANNOTATION_SPACING_DAYS = 120

#: Characters a label is truncated to.
ANNOTATION_WIDTH = 46

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
    trends: Sequence[str] = DEFAULT_TRENDS
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
    :param trends: Which trailing means to draw; see TREND_COLOURS.
    :return: The matplotlib figure.
    """
    chosen = [name for name in TREND_COLOURS if name in trends]
    if not chosen:
        raise ValueError(f"trends must name at least one of {sorted(TREND_COLOURS)}")
    frame = prepare(daily=daily, column=column, trends=chosen)
    figure = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor=OBSIDIAN)
    axis = figure.add_axes(AXES_RECT)
    axis.set_facecolor(OBSIDIAN)

    dates: List[dt.date] = frame["date"].to_list()
    stamps = matplotlib.dates.date2num(dates)
    raw = frame["raw_scaled"].to_numpy()
    ceiling = headroom(raw=raw, trend=frame[chosen[-1]].to_numpy())

    draw_regimes(axis=axis, regimes=regimes)
    axis.vlines(stamps, 0.0, raw, colors=RAW_GREY, linewidth=0.45, alpha=0.55)
    for drawn in chosen:
        axis.plot(
            stamps,
            frame[drawn].to_numpy(),
            color=TREND_COLOURS[drawn],
            linewidth=1.7 if drawn == "quadratic" else 1.1,
            solid_joinstyle="round"
        )
    axis.set_ylim(bottom=0.0, top=ceiling)
    axis.set_xlim(left=float(stamps[0]), right=float(stamps[-1]))

    style_axes(axis=axis, ceiling=ceiling, dates=dates)
    header(figure=figure, name=name, subtitle=subtitle, regimes=regimes, trends=chosen)
    footer(figure=figure, slug=slug or name.lower().replace(" ", "_"))
    if annotate:
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


def prepare(daily: pl.DataFrame, column: str, trends: Sequence[str]) -> pl.DataFrame:
    """
    Rebase the series and attach each requested trailing mean.

    :param daily: Daily frame.
    :param column: Column to draw.
    :param trends: Names of the means to compute.
    :return: Frame of date, raw_scaled, and one column per requested trend.
    """
    if daily.is_empty():
        raise ValueError("there is nothing to plot: the factor matched no days")
    frame = daily.drop_nulls(subset=[column]).sort(by="date")
    if frame.is_empty():
        raise ValueError(f"there is nothing to plot: every {column} is null")
    values = frame[column].to_numpy().astype(np.float64)
    average = float(np.nanmean(values))
    scaled = values / average * REBASE_MEAN if average else values
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

    The quadratic mean is the one the published indicators use. A plain average understates
    every episode in a spiky series: one day at twenty times the run rate moves a ninety-day
    mean by a fifth of its own size, where squaring before averaging lets it register.

    The geometric mean is taken on log1p and inverted with expm1. The textbook form is zero
    whenever any day in the window is zero, and for a factor matching a handful of events a day
    that is nearly every window, so the textbook form would draw a line flat on the floor. This
    variant tends to the arithmetic mean for small values and stays below it for large ones,
    which is the behaviour the comparison is for.

    :param values: Daily values, already rebased.
    :param dates: The dates those values fall on, ascending.
    :param kind: One of the names in TREND_COLOURS.
    :return: The trailing mean, nan until the window is populated.
    """
    if kind == "quadratic":
        prepared = np.square(values)
    elif kind == "geometric":
        prepared = np.log1p(np.clip(a=values, a_min=0.0, a_max=None))
    elif kind == "arithmetic":
        prepared = values
    else:
        raise ValueError(f"unknown trend {kind!r}; expected one of {sorted(TREND_COLOURS)}")

    output = np.full(shape=values.shape, fill_value=np.nan)
    start = 0
    for index, today in enumerate(dates):
        while (today - dates[start]).days >= TREND_DAYS:
            start += 1
        window = prepared[start:index + 1]
        if window.size < TREND_MIN_OBSERVATIONS:
            continue
        average = float(window.mean())
        if kind == "quadratic":
            output[index] = float(np.sqrt(average))
        elif kind == "geometric":
            output[index] = float(np.expm1(average))
        else:
            output[index] = average
    return output


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


def style_axes(axis: Any, ceiling: float, dates: Sequence[dt.date]) -> None:
    """
    Apply the deck's axis treatment: no spines, five ticks, faint year rules.

    :param axis: Axes to style.
    :param ceiling: Top of the axis.
    :param dates: The dates plotted, for the year rules.
    :return: None.
    """
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(axis="both", length=0, pad=6)
    axis.set_yticks([ceiling * step / 4 for step in range(5)])
    axis.set_yticklabels(
        [f"{ceiling * step / 4:#.4g}" for step in range(5)],
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

    The window length is named once, on the first trend, rather than repeated: three entries
    each opening with "90-DAY" run the strip into the title.

    :param figure: Figure to draw on.
    :param regimes: Whether a regime swatch is included.
    :param trends: The trends drawn.
    :return: None.
    """
    entries = [("RAW DAILY", RAW_GREY, "line")]
    for position, kind in enumerate(trends):
        prefix = f"{TREND_DAYS}-DAY " if position == 0 else ""
        entries.append((f"{prefix}{kind.upper()}", TREND_COLOURS[kind], "line"))
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
            [float(matplotlib.dates.date2num(date))], [value],
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
                axis.transData.transform((matplotlib.dates.date2num(date), 0.0))
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
    point = (matplotlib.dates.date2num(target[0]), target[1])
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
        chosen.append((date, float(value), titles.get(date, date.isoformat())))
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
