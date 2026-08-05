"""The object a quant actually uses.

A factor is three plain-English inputs and nothing else:

* **anchors** say what the factor is about. Write them as full sentences describing the kind of
  event you want to count. They are compared against every event's meaning, not its wording, so
  they find the story in any language and any phrasing.
* **poles** say which direction is up. Two short statements, one for the reading you call high
  and one for the reading you call low, turn a count of events into a signed measure of state.
* **filters** say which events are even eligible, in the vocabulary of the World event record.

Everything after that is arithmetic, and every arithmetic choice is a keyword argument on
compute() rather than something hidden in the library.
"""
import datetime as dt
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from semantic_factors import client as world_client
from semantic_factors import embeddings, floors, geometry, logs, series
from semantic_factors import filters as filter_rules
from semantic_factors import weights as weighting
from semantic_factors.errors import ConfigurationError, NotComputedError

#: Earliest date NOSIBLE World covers.
WORLD_FIRST_DATE = dt.date(2010, 1, 1)

#: Fields the grouped World projection returns for each retained event.
LITE_FIELDS = ("total_coverage", "total_netlocs", "coverage_concentration_index")


def validate_response_windows(
    responses: Sequence[Mapping[str, Any]],
    start: dt.date,
    stop: dt.date
) -> List[Mapping[str, Any]]:
    """Require grouped-search chunks to cover the requested window without gaps or overlap."""
    cursor = start
    windows: List[Mapping[str, Any]] = []
    for response in responses:
        window = response.get("date_window")
        if not isinstance(window, Mapping):
            raise ConfigurationError("World grouped response is missing its date window")
        try:
            window_start = dt.date.fromisoformat(str(window["from"]))
            window_stop = dt.date.fromisoformat(str(window["to"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ConfigurationError("World grouped response has an invalid date window") from error
        if window_start != cursor:
            raise ConfigurationError(
                "World grouped responses contain a date gap or overlap: "
                f"expected {cursor.isoformat()}, got {window_start.isoformat()}"
            )
        if window_stop < window_start or window_stop > stop:
            raise ConfigurationError("World grouped response has an invalid effective window")
        windows.append(window)
        cursor = window_stop + dt.timedelta(days=1)
    if cursor != stop + dt.timedelta(days=1):
        raise ConfigurationError(
            "World grouped responses do not cover the requested window through "
            f"{stop.isoformat()}"
        )
    return windows


class SemanticFactor:
    """One factor definition, and the series it produces once computed."""

    def __init__(
        self,
        anchors: Sequence[str],
        poles: Optional[Sequence[Tuple[Sequence[str], Sequence[str]]]] = None,
        pole_hi: Optional[Sequence[str]] = None,
        pole_lo: Optional[Sequence[str]] = None,
        filters: filter_rules.FilterInput = None,
        name: str = "factor",
        floor: floors.FloorSpec = floors.DEFAULT_FLOOR,
        aggregate: str = "max",
        tanh_scale: float = geometry.DEFAULT_TANH_SCALE
    ) -> None:
        """
        Define a factor.

        :param anchors: Sentences describing the events the factor counts.
        :param poles: Matched (high, low) statement pairs. Each pair says the same thing twice,
            once as the reading that should push the factor up and once as the reading that
            should push it down. Give one pair per topic the factor spans: resolution is always
            written as a topic's own concrete act, so one generic pair across many topics is
            lopsided.
        :param pole_hi: The high statements of a single pair, as a shorthand for
            ``poles=[(pole_hi, pole_lo)]``.
        :param pole_lo: The low statements of that single pair.
        :param filters: Event attributes that make an event eligible, as a mapping or pairs.
        :param name: Label used in plots, exports and diagnostics.
        :param floor: How relevant an event must be to count. Either a quantile of the eligible
            corpus, such as "p99.9" for the most relevant one event in a thousand, or an
            absolute cosine. Prefer the quantile: a cosine that is selective for one factor
            admits everything for another. Call calibrate() to see the trade-off.
        :param aggregate: How per-anchor cosines are collapsed; see geometry.AGGREGATIONS.
        :param tanh_scale: Divisor applied to the pole gap before tanh.
        :return: None.
        """
        self.anchors = validate_texts(texts=anchors, label="anchors")
        self.poles = validate_poles(
            poles=poles,
            pole_hi=pole_hi,
            pole_lo=pole_lo
        )
        if aggregate not in geometry.AGGREGATIONS:
            raise ConfigurationError(
                f"aggregate must be one of {geometry.AGGREGATIONS}, got {aggregate!r}"
            )
        self.filters = filter_rules.normalise_filters(filters=filters)
        self.name = name
        self.floor = floors.validate(floor=floor)
        if floors.is_quantile(floor=self.floor) and aggregate != "max":
            raise ConfigurationError(
                "quantile floors require aggregate='max'. Date-local retrieval can recover the "
                "high-score max tail from each anchor, but it cannot guarantee the mean or "
                "softmax tail. Use an explicit cosine floor with those aggregations."
            )
        self.aggregate = aggregate
        self.tanh_scale = float(tanh_scale)

        self.daily: Optional[pl.DataFrame] = None
        self.events: Optional[pl.DataFrame] = None
        self.diagnostics: Dict[str, Any] = {}
        self.loadings: Optional[pl.DataFrame] = None
        self.stats: Optional[pl.DataFrame] = None

    def __repr__(self) -> str:
        """
        Summarise the definition and whether it has been computed.

        :return: One-line representation.
        """
        state = "uncomputed" if self.daily is None else f"{self.daily.height} days"
        shape = f"{len(self.poles)} pole pairs" if self.poles else "unsigned"
        return (
            f"SemanticFactor(name={self.name!r}, anchors={len(self.anchors)}, "
            f"{shape}, floor={self.floor}, {state})"
        )

    def compute(
        self,
        weights: str = "netlocs",
        residualize: bool = False,
        stabilize: bool = True,
        polarity_weighted: bool = True,
        start_from: Optional[str] = None,
        stop_at: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        embed_base_url: str = embeddings.EMBED_BASE_URL,
        cache_dir: Optional[str] = None,
        top_events: int = 5,
        per_date_limit: int = world_client.DEFAULT_PER_DATE_LIMIT
    ) -> pl.DataFrame:
        """
        Build the daily factor series.

        :param weights: Named weighting scheme from weights.WEIGHT_SCHEMES. Callable weights are
            unavailable because grouped retrieval cannot return their exact denominator.
        :param residualize: Reserved for a future World API projection. Passing True fails
            explicitly; this release never silently changes the requested geometry.
        :param stabilize: Divide by a trailing average of daily event volume rather than the
            same day's, so corpus growth does not become a trend.
        :param polarity_weighted: Let the poles move the factor. Each event's weight is
            multiplied by (1 + polarity) / 2, so an event at the high pole counts fully, one at
            the low pole counts for almost nothing, and a neutral one counts half. This is what
            makes the poles part of the indicator rather than a second series beside it. Turn it
            off for a direction-blind count of relevant coverage. Ignored without poles.
        :param start_from: First date, as "YYYY-MM-DD". Defaults to the start of World.
        :param stop_at: Last date, as "YYYY-MM-DD". Defaults to yesterday.
        :param api_key: NOSIBLE World API key with archive access.
        :param base_url: World deployment to build against, or None to read NOSIBLE_WORLD_URL.
        :param embed_api_key: OpenAI-compatible embedding key used to embed anchors and poles.
        :param embed_base_url: OpenAI-compatible embedding API root. Use
            ``https://openrouter.ai/api/v1`` for OpenRouter.
        :param cache_dir: Cache root, or None for the default.
        :param top_events: Exemplar events to keep per period, for inspection and annotation.
        :param per_date_limit: Candidates retained per anchor and date; the measured default is
            1,000 and the World API accepts at most 10,000.
        :return: The daily frame, also available afterwards as .daily.
        """
        if residualize:
            raise ConfigurationError(
                "residualize=True is not supported by the public grouped-search contract. "
                "This release refuses the request rather than silently scoring different geometry."
            )
        if not isinstance(weights, str) or weights not in weighting.WEIGHT_SCHEMES:
            raise ConfigurationError(
                "World retrieval supports the named weighting schemes "
                f"{sorted(weighting.WEIGHT_SCHEMES)}; custom callables cannot provide a correct "
                "eligible-universe denominator."
            )
        if per_date_limit <= 0 or per_date_limit > 10_000:
            raise ConfigurationError("per_date_limit must be between 1 and 10,000")
        start, stop = resolve_window(
            start_from=start_from,
            stop_at=stop_at
        )
        started_at = dt.datetime.now(tz=dt.timezone.utc)
        with world_client.WorldClient(
            api_key=api_key,
            base_url=base_url,
            cache_dir=cache_dir
        ) as client:
            logs.step(message=f"building {self.name} from {start} to {stop}")
            days, exemplars, meta = self.compute_from_world(
                client=client,
                start=start,
                stop=stop,
                weights=weights,
                polarity_weighted=polarity_weighted,
                embed_api_key=embed_api_key,
                embed_base_url=embed_base_url,
                cache_dir=cache_dir,
                top_events=top_events,
                per_date_limit=per_date_limit
            )

        self.daily = series.daily_frame(
            days=days,
            start=start,
            stop=stop,
            stabilize=stabilize
        )
        self.events = exemplars
        self.diagnostics = {
            "name": self.name,
            "engine": "world_grouped_search",
            "start": start.isoformat(),
            "stop": stop.isoformat(),
            "anchors": len(self.anchors),
            "signed": bool(self.poles),
            "pole_pairs": len(self.poles),
            "polarity_weighted": polarity_weighted,
            "floor": self.floor,
            "aggregate": self.aggregate,
            "weights": weights if isinstance(weights, str) else "callable",
            "residualize": residualize,
            "stabilize": stabilize,
            "filters": dict(self.filters),
            "days": self.daily.height,
            "matched_events": int(self.daily["n_events"].sum()) if self.daily.height else 0,
            "computed_at": started_at.isoformat(),
            "elapsed_seconds": round(
                (dt.datetime.now(tz=dt.timezone.utc) - started_at).total_seconds(), 1
            ),
            **meta
        }
        logs.done(
            message=(
                f"{self.name}: {self.diagnostics['days']} days, "
                f"{self.diagnostics['matched_events']:,} matching events"
            )
        )
        return self.daily

    def prepare_vectors(
        self,
        embed_api_key: Optional[str],
        embed_base_url: str,
        cache_dir: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[slice, slice]]]:
        """Embed the complete definition once and record the pole score-column layout."""
        texts = list(self.anchors)
        pole_texts: List[str] = []
        pole_slices: List[Tuple[slice, slice]] = []
        cursor = 0
        for high, low in self.poles:
            high_slice = slice(cursor, cursor + len(high))
            pole_texts.extend(high)
            cursor += len(high)
            low_slice = slice(cursor, cursor + len(low))
            pole_texts.extend(low)
            cursor += len(low)
            pole_slices.append((high_slice, low_slice))
        matrix = embeddings.embed_texts(
            texts=texts + pole_texts,
            api_key=embed_api_key,
            base_url=embed_base_url,
            cache_dir=cache_dir
        )
        anchors = matrix[:len(texts)]
        pole_vectors = matrix[len(texts):]
        return anchors, pole_vectors, pole_slices

    def calibration_sample(
        self,
        client: world_client.WorldClient,
        anchors: np.ndarray,
        start: dt.date,
        stop: dt.date,
        per_date_limit: int
    ) -> Tuple[np.ndarray, int, int]:
        """Retrieve the high-score tail from evenly spaced, unfiltered archive dates."""
        span = (stop - start).days + 1
        all_dates = [start + dt.timedelta(days=offset) for offset in range(span)]
        chosen = floors.calibration_dates(dates=all_dates)
        scores: List[float] = []
        eligible_events = 0
        represented_days = 0
        for index, date in enumerate(chosen, start=1):
            responses = client.grouped_search(
                vectors=anchors,
                start=date,
                stop=date,
                filters={},
                min_score=-1.0,
                aggregate=self.aggregate,
                per_date_limit=per_date_limit
            )
            response = responses[0]
            scores.extend(float(event["score"]) for event in response.get("events") or [])
            groups = response.get("groups") or []
            represented_days += len(groups)
            eligible_events += sum(
                int((group.get("eligible") or {}).get("count") or 0)
                for group in groups
            )
            if index == len(chosen) or index % 8 == 0:
                logs.info(message=f"calibrated {index}/{len(chosen)} sampled dates")
        return np.asarray(a=scores, dtype=np.float32), eligible_events, represented_days

    def calibrate(
        self,
        start_from: Optional[str] = None,
        stop_at: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        embed_base_url: str = embeddings.EMBED_BASE_URL,
        cache_dir: Optional[str] = None,
        per_date_limit: int = world_client.DEFAULT_PER_DATE_LIMIT
    ) -> pl.DataFrame:
        """
        Estimate the definition's high relevance quantiles on evenly spaced archive dates.

        :return: One row per resolvable high quantile.
        """
        start, stop = resolve_window(start_from=start_from, stop_at=stop_at)
        anchors, _, _ = self.prepare_vectors(
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            cache_dir=cache_dir
        )
        with world_client.WorldClient(
            api_key=api_key,
            base_url=base_url,
            cache_dir=cache_dir
        ) as client:
            scores, eligible_events, sampled_days = self.calibration_sample(
                client=client,
                anchors=anchors,
                start=start,
                stop=stop,
                per_date_limit=per_date_limit
            )
        rows = floors.tail_quantile_table(
            tail_scores=scores,
            eligible_events=eligible_events,
            days=sampled_days
        )
        logs.info(
            message=(
                f"{self.name}: calibrated {eligible_events:,} events across "
                f"{sampled_days} dates"
            )
        )
        return pl.DataFrame(data=rows)

    def compute_from_world(
        self,
        client: world_client.WorldClient,
        start: dt.date,
        stop: dt.date,
        weights: str,
        polarity_weighted: bool,
        embed_api_key: Optional[str],
        embed_base_url: str,
        cache_dir: Optional[str],
        top_events: int,
        per_date_limit: int
    ) -> Tuple[List[Dict[str, Any]], Optional[pl.DataFrame], Dict[str, Any]]:
        """Build a factor from the generic date-grouped World search projection."""
        anchors, pole_vectors, pole_slices = self.prepare_vectors(
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            cache_dir=cache_dir
        )
        calibration_events: Optional[int] = None
        calibration_days: Optional[int] = None
        relevance_quantiles: Optional[List[dict]] = None
        if floors.is_quantile(floor=self.floor):
            tail, calibration_events, calibration_days = self.calibration_sample(
                client=client,
                anchors=anchors,
                start=start,
                stop=stop,
                per_date_limit=per_date_limit
            )
            resolved_floor = floors.resolve_from_tail(
                floor=self.floor,
                tail_scores=tail,
                eligible_events=calibration_events
            )
            relevance_quantiles = floors.tail_quantile_table(
                tail_scores=tail,
                eligible_events=calibration_events,
                days=calibration_days
            )
        else:
            resolved_floor = float(self.floor)
        logs.info(message=f"relevance floor {self.floor!r} resolves to {resolved_floor:.4f}")

        responses = client.grouped_search(
            vectors=anchors,
            score_vectors=pole_vectors,
            start=start,
            stop=stop,
            filters=self.filters,
            min_score=resolved_floor,
            aggregate=self.aggregate,
            per_date_limit=per_date_limit
        )
        api_events = [
            event
            for response in responses
            for event in response.get("events") or []
        ]
        records = [flatten_event(event=event) for event in api_events]
        relevance = np.asarray(
            a=[float(event.get("score") or 0.0) for event in api_events],
            dtype=np.float32
        )
        polarity: Optional[np.ndarray] = None
        if pole_slices and api_events:
            pole_scores = np.asarray(
                a=[event.get("score_vector_scores") or [] for event in api_events],
                dtype=np.float32
            )
            if pole_scores.shape != (len(api_events), pole_vectors.shape[0]):
                raise ConfigurationError(
                    "World returned a malformed pole-score projection: expected "
                    f"({len(api_events)}, {pole_vectors.shape[0]}), got {pole_scores.shape}"
                )
            polarity = geometry.paired_polarity_from_scores(
                scores=pole_scores,
                pairs=pole_slices,
                tanh_scale=self.tanh_scale
            )

        volume: Dict[str, float] = {}
        eligible_counts: Dict[str, int] = {}
        for response in responses:
            for group in response.get("groups") or []:
                date = str(group.get("date") or "")
                eligible = group.get("eligible") or {}
                volume[date] = denominator_weight(eligible=eligible, weights=weights)
                eligible_counts[date] = int(eligible.get("count") or 0)
        days, exemplars = fold_records_into_days(
            records=records,
            relevance=relevance,
            polarity=polarity,
            weights=weights,
            volume=volume,
            eligible_counts=eligible_counts,
            polarity_weighted=polarity_weighted,
            top_events=top_events
        )
        logs.warn(
            message=(
                "date-grouped retrieval uses approximate per-day HNSW candidates; "
                "diagnostics record the candidate budget so the build is auditable"
            )
        )
        date_windows = validate_response_windows(
            responses=responses,
            start=start,
            stop=stop
        )
        return days, exemplars, {
            "resolved_floor": resolved_floor,
            "exhaustive": False,
            "retrieval": "date_local_hnsw",
            "per_date_limit": per_date_limit,
            "world_requests": len(responses),
            "effective_start": (
                str(date_windows[0].get("from")) if date_windows else start.isoformat()
            ),
            "effective_stop": (
                str(date_windows[-1].get("to")) if date_windows else stop.isoformat()
            ),
            "archive_cutoff": (
                str(date_windows[-1].get("archive_cutoff")) if date_windows else None
            ),
            "candidate_events": sum(
                int(response.get("candidate_events") or 0) for response in responses
            ),
            "volume_basis": "weight",
            "calibration_events": calibration_events,
            "calibration_days": calibration_days,
            "relevance_quantiles": relevance_quantiles
        }
    def monthly(self) -> pl.DataFrame:
        """
        Aggregate the computed daily series to calendar months.

        :return: One row per month.
        """
        return series.resample(daily=self.require_daily(), every="1mo")

    def weekly(self) -> pl.DataFrame:
        """
        Aggregate the computed daily series to calendar weeks.

        :return: One row per week.
        """
        return series.resample(daily=self.require_daily(), every="1w")

    def plot(
        self,
        annotate: bool = True,
        path: Optional[str] = None,
        show_polarity: bool = False,
        title: Optional[str] = None,
        episodes: Optional[Dict[str, str]] = None,
        subtitle: str = "",
        regimes: Optional[Sequence[Tuple[str, str]]] = None
    ) -> Any:
        """
        Draw the factor in the NOSIBLE house style.

        :param annotate: Label the largest peaks with the event that drove them.
        :param path: File to write, or None to return the figure without saving.
        :param show_polarity: Draw the signed pole series instead of the index. Normally
            redundant, because the poles already multiply into the index; use it to inspect the
            axis itself.
        :param title: Figure title, or None to use the factor name.
        :param episodes: Dates you want marked, mapped to their labels, keyed "YYYY-MM-DD" or
            "YYYY-MM". Without them the chart marks its own largest peaks.
        :param subtitle: One line under the title saying what the factor is for.
        :param regimes: Date ranges to shade, as (start, stop) ISO pairs.
        :return: The matplotlib figure.
        """
        from semantic_factors import plotting

        return plotting.plot_factor(
            daily=self.require_daily(),
            events=self.events,
            name=title or self.name.replace("_", " ").title(),
            annotate=annotate,
            path=path,
            episodes=episodes,
            subtitle=subtitle,
            slug=self.name,
            regimes=regimes,
            column="polarity" if show_polarity else "value"
        )

    def regress(
        self,
        tickers: Sequence[str],
        returns: Optional[pl.DataFrame] = None,
        horizon: int = 0,
        window: int = 252,
        min_observations: int = 120,
        column: str = "value",
        difference: bool = True,
        eodhd_api_key: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Regress each ticker's returns on the factor and report its exposure.

        :param tickers: Symbols in EODHD form, such as "CTVA.US".
        :param returns: Wide frame of daily returns with a "date" column and one column per
            ticker, or None to download closes from EODHD.
        :param horizon: 0 measures a contemporaneous exposure, which is what a loading is.
            One or more shifts the return forward and asks whether the factor predicts.
        :param window: Length of the rolling window used for point-in-time loadings.
        :param min_observations: Observations required before a beta is reported.
        :param column: Factor column to regress against.
        :param difference: Regress on the change in the factor rather than its level, which is
            what you want unless the level is already stationary.
        :param eodhd_api_key: EODHD key, when returns are being downloaded.
        :return: One row per ticker with its full-sample exposure, also stored as .stats.
        """
        from semantic_factors import regression
        from semantic_factors import returns as return_data

        daily = self.require_daily()
        if returns is None:
            returns = return_data.download_returns(
                tickers=tickers,
                start=series.parse_date(value=daily["date"].min()),
                stop=series.parse_date(value=daily["date"].max()),
                api_key=eodhd_api_key
            )
        self.stats, self.loadings = regression.regress_factor(
            daily=daily,
            returns=returns,
            tickers=list(tickers),
            column=column,
            horizon=horizon,
            window=window,
            min_observations=min_observations,
            difference=difference
        )
        return self.stats

    @property
    def longs(self) -> pl.DataFrame:
        """
        Tickers whose returns rise with the factor, most exposed first.

        :return: The significantly positive rows of .stats.
        """
        return significant_side(stats=self.require_stats(), positive=True)

    @property
    def shorts(self) -> pl.DataFrame:
        """
        Tickers whose returns fall as the factor rises, most exposed first.

        :return: The significantly negative rows of .stats.
        """
        return significant_side(stats=self.require_stats(), positive=False)

    @property
    def values(self) -> pl.Series:
        """
        The computed factor level, as a bare series.

        :return: The "value" column of the daily frame.
        """
        return self.require_daily()["value"]

    def to_csv(self, path: str) -> str:
        """
        Write the daily series to CSV.

        :param path: Destination path.
        :return: The path written.
        """
        self.require_daily().write_csv(file=path)
        return path

    def to_parquet(self, path: str) -> str:
        """
        Write the daily series to Parquet.

        :param path: Destination path.
        :return: The path written.
        """
        self.require_daily().write_parquet(file=path)
        return path

    def require_daily(self) -> pl.DataFrame:
        """
        Return the daily series, or explain that compute() has not run.

        :return: The daily frame.
        """
        if self.daily is None:
            raise NotComputedError("call compute() before reading the factor")
        return self.daily

    def require_stats(self) -> pl.DataFrame:
        """
        Return the regression statistics, or explain that regress() has not run.

        :return: The per-ticker statistics frame.
        """
        if self.stats is None:
            raise NotComputedError("call regress() before reading exposures")
        return self.stats


def fold_records_into_days(
    records: Sequence[Dict[str, Any]],
    relevance: np.ndarray,
    polarity: Optional[np.ndarray],
    weights: weighting.WeightSpec,
    volume: Mapping[str, float],
    eligible_counts: Optional[Mapping[str, int]] = None,
    polarity_weighted: bool = True,
    top_events: int = 5
) -> Tuple[List[Dict[str, Any]], Optional[pl.DataFrame]]:
    """
    Reduce scored events to one record per day, plus a handful of exemplars.

    :param records: Flattened event records that cleared the floor.
    :param relevance: Relevance score per record.
    :param polarity: Signed pole score per record, or None for an unsigned factor.
    :param weights: Weighting scheme name or callable.
    :param volume: Eligible-universe denominator weight per ISO date.
    :param eligible_counts: Exact eligible event counts, or None to derive them from volume.
    :param polarity_weighted: Whether poles tilt each matched event's numerator contribution.
    :param top_events: Exemplars to keep per day.
    :return: Tuple of day records and the exemplar frame.
    """
    event_weight = weighting.event_weights(
        records=records,
        scheme=weights,
        relevance=relevance
    )
    contribution = event_weight.copy()
    if polarity is not None and polarity_weighted:
        contribution *= geometry.uncertainty_weight(polarity_scores=polarity)
    counts = eligible_counts or {date: int(value) for date, value in volume.items()}
    per_day: Dict[str, Dict[str, float]] = {
        date: {"raw": 0.0, "n_events": 0, "polarity_num": 0.0, "polarity_weight": 0.0}
        for date in volume
    }
    for index, record in enumerate(records):
        bucket = per_day.setdefault(
            record["date"],
            {"raw": 0.0, "n_events": 0, "polarity_num": 0.0, "polarity_weight": 0.0}
        )
        bucket["raw"] += float(contribution[index])
        bucket["n_events"] += 1
        if polarity is not None:
            bucket["polarity_num"] += float(event_weight[index]) * float(polarity[index])
            bucket["polarity_weight"] += float(event_weight[index])

    days = [
        {
            "date": date,
            "raw": bucket["raw"],
            "volume": float(volume.get(date, 0)),
            "n_events": int(bucket["n_events"]),
            "n_eligible": int(counts.get(date, 0)),
            "polarity_num": bucket["polarity_num"],
            "polarity_weight": bucket["polarity_weight"]
        }
        for date, bucket in sorted(per_day.items())
    ]
    return days, top_event_frame(
        records=records,
        relevance=relevance,
        polarity=polarity,
        event_weight=contribution,
        top_events=top_events
    )


def top_event_frame(
    records: Sequence[Dict[str, Any]],
    relevance: np.ndarray,
    polarity: Optional[np.ndarray],
    event_weight: np.ndarray,
    top_events: int
) -> Optional[pl.DataFrame]:
    """
    Keep the heaviest contributors per day, so a peak can be explained rather than trusted.

    :param records: Flattened event records that cleared the floor.
    :param relevance: Relevance score per record.
    :param polarity: Signed pole score per record, or None.
    :param event_weight: Weight per record.
    :param top_events: Exemplars to keep per day.
    :return: Long frame of exemplars, or None when none were requested.
    """
    if top_events <= 0 or not records:
        return None
    frame = pl.DataFrame(
        data={
            "date": [series.parse_date(value=record["date"]) for record in records],
            "event_id": [record["event_id"] for record in records],
            "title": [record.get("title") or "" for record in records],
            "country": [record.get("country") or "" for record in records],
            "score": relevance.astype(np.float64),
            "weight": event_weight,
            "polarity": (
                polarity.astype(np.float64)
                if polarity is not None
                else np.full(shape=len(records), fill_value=np.nan)
            )
        }
    )
    return (
        frame.sort(by=["date", "weight"], descending=[False, True])
        .group_by("date", maintain_order=True)
        .head(top_events)
    )


def flatten_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested World event record into the fields the weighting schemes read.

    :param event: Lite or full event record.
    :return: Flat mapping with date, event_id, title, country, coverage and signal fields.
    """
    if "date" in event and "score" in event:
        return {
            "event_id": str(event.get("event_id") or ""),
            "date": str(event.get("date") or ""),
            "title": str(event.get("title") or ""),
            "country": str(event.get("country") or ""),
            "language": str(event.get("language") or ""),
            "sentiment": str(event.get("sentiment") or "neutral"),
            "materiality_score": event.get("materiality_score") or 0.0,
            "total_coverage": event.get("total_coverage") or 0.0,
            "total_netlocs": event.get("total_netlocs") or 0.0,
            "coverage_concentration_index": event.get("coverage_concentration_index") or 0.0
        }
    body = event.get("event") or {}
    coverage = event.get("coverage") or {}
    signals = event.get("signals") or {}
    flat: Dict[str, Any] = {
        "event_id": str(event.get("event_id") or ""),
        "date": str(event.get("_search_date") or body.get("date") or ""),
        "title": body.get("title") or "",
        "country": body.get("country") or "",
        "language": body.get("language") or "",
        "sentiment": signals.get("sentiment") or "neutral",
        "materiality_score": signals.get("materiality_score") or 0.0
    }
    for field in LITE_FIELDS:
        flat[field] = coverage.get(field) or 0.0
    return flat


def denominator_weight(eligible: Mapping[str, Any], weights: str) -> float:
    """Select the exact eligible-universe total matching one built-in numerator weight."""
    field = {
        "equal": "count",
        "relevance": "count",
        "coverage": "total_coverage",
        "netlocs": "total_netlocs",
        "materiality": "total_materiality",
        "sentiment": "total_sentiment_weight"
    }.get(weights)
    if field is None:
        raise ConfigurationError(f"unknown weighting scheme {weights!r}")
    value = eligible.get(field)
    return float(value) if isinstance(value, (int, float)) else 0.0


def significant_side(stats: pl.DataFrame, positive: bool) -> pl.DataFrame:
    """
    Select the statistically significant exposures on one side of zero.

    :param stats: Per-ticker statistics from regress().
    :param positive: True for positive exposures, False for negative.
    :return: The matching rows, largest absolute exposure first.
    """
    condition = pl.col("beta") > 0 if positive else pl.col("beta") < 0
    return (
        stats.filter(condition & (pl.col("t_stat").abs() >= 1.96))
        .sort(by="beta", descending=positive)
    )


def validate_poles(
    poles: Optional[Sequence[Tuple[Sequence[str], Sequence[str]]]],
    pole_hi: Optional[Sequence[str]],
    pole_lo: Optional[Sequence[str]]
) -> List[Tuple[List[str], List[str]]]:
    """
    Resolve the two ways of declaring poles into one canonical list of matched pairs.

    :param poles: Explicit (high, low) pairs, or None.
    :param pole_hi: High statements of a single pair, or None.
    :param pole_lo: Low statements of a single pair, or None.
    :return: List of (high, low) pairs, empty for an unsigned factor.
    """
    if poles and (pole_hi or pole_lo):
        raise ConfigurationError(
            "give either poles=[(high, low), ...] or the single-pair pole_hi/pole_lo, not both"
        )
    if poles:
        pairs = []
        for index, pair in enumerate(poles):
            if len(pair) != 2:
                raise ConfigurationError(
                    f"pole pair {index} has {len(pair)} entries; each pair is (high, low)"
                )
            pairs.append(
                (
                    validate_texts(texts=pair[0], label=f"poles[{index}] high"),
                    validate_texts(texts=pair[1], label=f"poles[{index}] low")
                )
            )
        return pairs
    if bool(pole_hi) != bool(pole_lo):
        raise ConfigurationError(
            "poles come in pairs: give both pole_hi and pole_lo, or neither"
        )
    if not pole_hi:
        return []
    return [
        (
            validate_texts(texts=pole_hi, label="pole_hi"),
            validate_texts(texts=pole_lo, label="pole_lo")
        )
    ]


def validate_texts(texts: Optional[Sequence[str]], label: str) -> List[str]:
    """
    Check that a list of anchor or pole statements is usable, and return it cleaned.

    :param texts: The statements.
    :param label: Argument name, for the error message.
    :return: Stripped, de-duplicated statements in their original order.
    """
    if not texts:
        raise ConfigurationError(f"{label} must contain at least one sentence")
    if isinstance(texts, str):
        raise ConfigurationError(f"{label} takes a list of sentences, not a single string")
    cleaned: List[str] = []
    for item in texts:
        if not isinstance(item, str) or not item.strip():
            raise ConfigurationError(f"every entry in {label} must be a non-empty string")
        value = " ".join(item.split())
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


def resolve_window(
    start_from: Optional[str],
    stop_at: Optional[str]
) -> Tuple[dt.date, dt.date]:
    """
    Resolve the requested date window, defaulting to all of World up to yesterday.

    Today is excluded because the live pipeline reprocesses the most recent days, so a value
    computed for today changes under you.

    :param start_from: First date as "YYYY-MM-DD", or None.
    :param stop_at: Last date as "YYYY-MM-DD", or None.
    :return: Tuple of inclusive start and stop dates.
    """
    start = dt.date.fromisoformat(start_from) if start_from else WORLD_FIRST_DATE
    default_stop = dt.datetime.now(tz=dt.timezone.utc).date() - dt.timedelta(days=1)
    stop = dt.date.fromisoformat(stop_at) if stop_at else default_stop
    if stop < start:
        raise ConfigurationError(f"stop_at {stop} precedes start_from {start}")
    return start, stop
