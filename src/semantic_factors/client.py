"""Authenticated access to NOSIBLE World's grouped semantic search.

The package has one data path: caller-supplied vectors are searched independently in every
date slice, exact scores are returned for the retained candidates, and the response includes
the exact eligible-universe totals needed by each built-in weighting scheme. Factor arithmetic
stays in this package; the World API remains a generic retrieval API.
"""
import datetime as dt
import email.utils
import hashlib
import json
import math
import os
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests

from semantic_factors import cache, logs
from semantic_factors.errors import AccessDeniedError, CredentialError, WorldApiError

WORLD_BASE_URL = "https://nosible.world"
WORLD_KEY_VARIABLES = ("NOSIBLE_API_KEY", "SEMANTIC_FACTORS_API_KEY", "NOSIBLE_WORLD_API_KEY")
WORLD_URL_VARIABLE = "NOSIBLE_WORLD_URL"
SEARCH_PATH = "/api/search"
SEARCH_SCHEMA_PATH = "/api/search/schema"
SEARCH_SCHEMA = "nosible_world_search_schema_v1"
SEARCH_SCHEMA_FACTOR_COLUMNS_REVISION = 4
GROUPED_RESPONSE_SCHEMA = "nosible_world_factor_search_v1"
GROUPED_RESPONSE_REVISION = 3
FACTOR_COLUMNS_RESPONSE_SCHEMA = "nosible_world_factor_columns_v1"
FACTOR_COLUMNS_RESPONSE_REVISION = 1
NORMALIZED_FACTOR_COLUMNS_MARKER = "_semantic_factors_factor_rows_v1"
FACTOR_ROWS_INCLUDE = ("factor",)
FACTOR_COLUMNS_INCLUDE = ("factor", "factor_columns_v1")
FACTOR_COLUMNS_ADVERTISEMENT = {
    "include": list(FACTOR_COLUMNS_INCLUDE),
    "revision": FACTOR_COLUMNS_RESPONSE_REVISION,
    "score_range": [-1, 1],
    "score_vector_width": "request.score_vectors.length",
    "score_vector_order": "request order",
}
FACTOR_EVENT_FIELDS = (
    "date",
    "event_id",
    "title",
    "country",
    "score",
    "score_vector_scores",
    "total_coverage",
    "total_netlocs",
    "coverage_concentration_index",
    "materiality_score",
    "sentiment",
)
RETRYABLE_503_ERRORS = frozenset(
    {
        "grouped_search_busy",
        "api_key_changed",
        "api_key_auth_not_configured",
        "api_key_auth_unavailable",
        "api_key_metering_unavailable",
        "archive_date_gap",
        "fast_metadata_stale",
        "fast_metadata_unavailable",
    }
)

# The Rust route refuses wider requests. Keeping the same constant here makes a long build
# resumable in small immutable cache entries rather than failing after one enormous response.
GROUPED_MAX_DAYS = 120
DEFAULT_PER_DATE_LIMIT = 1_000
MAX_PER_DATE_LIMIT = 10_000
MAX_GROUPED_VECTORS = 64
WORLD_VECTOR_DIMENSIONS = 3_072
GROUPED_AGGREGATES = frozenset({"max", "mean", "softmax"})
MAX_CANDIDATE_RETRIEVALS = 2_000_000
MAX_SCORE_DOT_PRODUCTS = 50_000_000

#: World serializes the grouped-search floor to four decimal places in the response envelope.
GROUPED_MIN_SCORE_ECHO_TOLERANCE = 5e-5
RETRY_STATUSES = (408, 425, 429, 500, 502, 503, 504)
MAX_ATTEMPTS = 5
DEFAULT_TIMEOUT = (10.0, 300.0)
MAX_RETRY_DELAY_SECONDS = 60.0


class WorldClient:
    """Session-scoped access to one NOSIBLE World deployment."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Tuple[float, float] = DEFAULT_TIMEOUT,
        cache_dir: Optional[str] = None
    ) -> None:
        """
        Open an authenticated session.

        :param api_key: NOSIBLE World key, or None to read the environment.
        :param base_url: Deployment root, or None for NOSIBLE_WORLD_URL and the public service.
        :param timeout: Connect and read timeouts in seconds.
        :param cache_dir: Cache root, or None for the default.
        :return: None.
        """
        self.base_url = resolve_base_url(base_url=base_url)
        self.api_key = resolve_world_key(api_key=api_key)
        if not self.api_key:
            raise CredentialError(
                "semantic factor builds require a NOSIBLE World API key. Pass api_key=... or "
                f"set one of {', '.join(WORLD_KEY_VARIABLES)}."
            )
        self.timeout = timeout
        self.cache_dir = cache.resolve_cache_dir(cache_dir=cache_dir)
        # Cache entries must never cross API-key identities: two keys can have
        # different archive entitlements even against the same deployment.
        self.cache_scope = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()
        self._factor_columns_supported: Optional[bool] = None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": "semantic-factors/0.1"
            }
        )

    def __enter__(self) -> "WorldClient":
        """Return this client from a context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the HTTP session on context-manager exit."""
        self.close()

    def close(self) -> None:
        """Release the HTTP session."""
        self.session.close()

    def _factor_projection_include(self) -> List[str]:
        """Return the safest factor projection supported by this exact server session."""
        if self._factor_columns_supported is None:
            try:
                schema = self.request(method="GET", path=SEARCH_SCHEMA_PATH)
            except WorldApiError as error:
                # Projection discovery is an optimisation gate. A legacy grouped request is
                # still valid when an old deployment lacks the catalog or discovery is briefly
                # unavailable, so do not turn an additive rollout into a factor-build outage.
                logs.info(
                    message=(
                        "World factor-column capability discovery failed; using row projection "
                        f"status={error.status}"
                    )
                )
                self._factor_columns_supported = False
            else:
                self._factor_columns_supported = search_schema_supports_factor_columns(
                    response=schema
                )
        return list(
            FACTOR_COLUMNS_INCLUDE if self._factor_columns_supported else FACTOR_ROWS_INCLUDE
        )

    def grouped_search(
        self,
        vectors: np.ndarray,
        start: dt.date,
        stop: dt.date,
        filters: Dict[str, Any],
        min_score: float,
        aggregate: str,
        score_vectors: Optional[np.ndarray] = None,
        per_date_limit: int = DEFAULT_PER_DATE_LIMIT
    ) -> List[Dict[str, Any]]:
        """Return legacy row-shaped grouped chunks through the stable public client contract."""
        return list(
            self.grouped_search_iter(
                vectors=vectors,
                start=start,
                stop=stop,
                filters=filters,
                min_score=min_score,
                aggregate=aggregate,
                score_vectors=score_vectors,
                per_date_limit=per_date_limit,
            )
        )

    def _factor_grouped_search(
        self,
        vectors: np.ndarray,
        start: dt.date,
        stop: dt.date,
        filters: Dict[str, Any],
        min_score: float,
        aggregate: str,
        score_vectors: Optional[np.ndarray] = None,
        per_date_limit: int = DEFAULT_PER_DATE_LIMIT
    ) -> List[Dict[str, Any]]:
        """Return all chunks through the private compact-capable factor data path."""
        return list(
            self._factor_grouped_search_iter(
                vectors=vectors,
                start=start,
                stop=stop,
                filters=filters,
                min_score=min_score,
                aggregate=aggregate,
                score_vectors=score_vectors,
                per_date_limit=per_date_limit,
            )
        )

    def grouped_search_iter(
        self,
        vectors: np.ndarray,
        start: dt.date,
        stop: dt.date,
        filters: Dict[str, Any],
        min_score: float,
        aggregate: str,
        score_vectors: Optional[np.ndarray] = None,
        per_date_limit: int = DEFAULT_PER_DATE_LIMIT
    ) -> Iterator[Dict[str, Any]]:
        """Yield legacy row-shaped chunks through the stable public client contract."""
        return self._grouped_search_iter(
            vectors=vectors,
            start=start,
            stop=stop,
            filters=filters,
            min_score=min_score,
            aggregate=aggregate,
            score_vectors=score_vectors,
            per_date_limit=per_date_limit,
            allow_factor_columns=False,
        )

    def _factor_grouped_search_iter(
        self,
        vectors: np.ndarray,
        start: dt.date,
        stop: dt.date,
        filters: Dict[str, Any],
        min_score: float,
        aggregate: str,
        score_vectors: Optional[np.ndarray] = None,
        per_date_limit: int = DEFAULT_PER_DATE_LIMIT
    ) -> Iterator[Dict[str, Any]]:
        """Yield grouped chunks through the private compact-capable factor data path."""
        return self._grouped_search_iter(
            vectors=vectors,
            start=start,
            stop=stop,
            filters=filters,
            min_score=min_score,
            aggregate=aggregate,
            score_vectors=score_vectors,
            per_date_limit=per_date_limit,
            allow_factor_columns=True,
        )

    def _grouped_search_iter(
        self,
        vectors: np.ndarray,
        start: dt.date,
        stop: dt.date,
        filters: Dict[str, Any],
        min_score: float,
        aggregate: str,
        score_vectors: Optional[np.ndarray],
        per_date_limit: int,
        allow_factor_columns: bool,
    ) -> Iterator[Dict[str, Any]]:
        """
        Search a long window as consecutive date-grouped requests.

        :param vectors: Unit query vectors, one per anchor.
        :param start: First date, inclusive.
        :param stop: Last date, inclusive.
        :param filters: Normalised World filters.
        :param min_score: Inclusive exact aggregate-cosine floor.
        :param aggregate: ``max``, ``mean`` or ``softmax``.
        :param score_vectors: Extra vectors scored on candidates without widening retrieval.
        :param per_date_limit: HNSW candidates retained per anchor and date.
        :return: Decoded responses yielded one date chunk at a time, in chronological order.
        """
        vector_rows = grouped_matrix_json(
            matrix=vectors,
            field="vectors",
            required=True,
        )
        assert vector_rows is not None
        score_vector_rows = grouped_matrix_json(
            matrix=score_vectors,
            field="score_vectors",
            required=False,
        )
        if not isinstance(aggregate, str) or aggregate not in GROUPED_AGGREGATES:
            raise ValueError(
                f"aggregate must be one of {sorted(GROUPED_AGGREGATES)}, got {aggregate!r}"
            )
        try:
            min_score_value = float(min_score)
        except (OverflowError, TypeError, ValueError):
            min_score_value = math.nan
        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float, np.integer, np.floating))
            or not math.isfinite(min_score_value)
            or not -1.0 <= min_score_value <= 1.0
        ):
            raise ValueError("min_score must be a finite number between -1 and 1")
        if (
            isinstance(per_date_limit, bool)
            or not isinstance(per_date_limit, (int, np.integer))
            or not 1 <= int(per_date_limit) <= MAX_PER_DATE_LIMIT
        ):
            raise ValueError(
                f"per_date_limit must be an integer between 1 and {MAX_PER_DATE_LIMIT:,}"
            )
        vector_count = len(vector_rows)
        score_vector_count = 0 if score_vector_rows is None else len(score_vector_rows)
        retrievals_per_day = vector_count * int(per_date_limit)
        scores_per_day = retrievals_per_day * (vector_count + score_vector_count)
        chunk_days = min(
            GROUPED_MAX_DAYS,
            MAX_CANDIDATE_RETRIEVALS // max(1, retrievals_per_day),
            MAX_SCORE_DOT_PRODUCTS // max(1, scores_per_day)
        )
        if chunk_days <= 0:
            raise ValueError(
                "the vector count and per_date_limit exceed the World grouped-search work cap "
                "for even one date"
            )
        windows = day_chunks(start=start, stop=stop, days=chunk_days)
        return self._grouped_search_chunks(
            vector_rows=vector_rows,
            score_vector_rows=score_vector_rows,
            windows=windows,
            filters=filters,
            min_score=min_score_value,
            aggregate=aggregate,
            per_date_limit=int(per_date_limit),
            allow_factor_columns=allow_factor_columns,
        )

    def _grouped_search_chunks(
        self,
        vector_rows: List[List[float]],
        score_vector_rows: Optional[List[List[float]]],
        windows: List[Tuple[dt.date, dt.date]],
        filters: Dict[str, Any],
        min_score: float,
        aggregate: str,
        per_date_limit: int,
        allow_factor_columns: bool,
    ) -> Iterator[Dict[str, Any]]:
        """Yield already-validated grouped windows without deferring validation to iteration."""
        for index, (chunk_start, chunk_stop) in enumerate(windows, start=1):
            chunk_responses, reached_cutoff = self._grouped_window_with_cutoff(
                vector_rows=vector_rows,
                score_vector_rows=score_vector_rows,
                chunk_start=chunk_start,
                chunk_stop=chunk_stop,
                filters=filters,
                min_score=min_score,
                aggregate=aggregate,
                per_date_limit=per_date_limit,
                allow_factor_columns=allow_factor_columns,
            )
            yield from chunk_responses
            if index == 1 or index % 10 == 0 or index == len(windows):
                logs.info(message=f"retrieved date chunk {index}/{len(windows)}")
            if reached_cutoff:
                logs.info(message="reached the World archive cutoff; skipping later date chunks")
                break

    def _grouped_window(
        self,
        vector_rows: List[List[float]],
        score_vector_rows: Optional[List[List[float]]],
        chunk_start: dt.date,
        chunk_stop: dt.date,
        filters: Dict[str, Any],
        min_score: float,
        aggregate: str,
        per_date_limit: int,
        allow_factor_columns: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch one window, recursively splitting only an explicit oversized response."""
        responses, _ = self._grouped_window_with_cutoff(
            vector_rows=vector_rows,
            score_vector_rows=score_vector_rows,
            chunk_start=chunk_start,
            chunk_stop=chunk_stop,
            filters=filters,
            min_score=min_score,
            aggregate=aggregate,
            per_date_limit=per_date_limit,
            allow_factor_columns=allow_factor_columns,
        )
        return responses

    def _grouped_window_with_cutoff(
        self,
        vector_rows: List[List[float]],
        score_vector_rows: Optional[List[List[float]]],
        chunk_start: dt.date,
        chunk_stop: dt.date,
        filters: Dict[str, Any],
        min_score: float,
        aggregate: str,
        per_date_limit: int,
        allow_factor_columns: bool = False,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Fetch one window and retain whether this exact request was archive-clamped."""
        body: Dict[str, Any] = {
            "vectors": vector_rows,
            "vector_aggregate": aggregate,
            "group_by": "date",
            "per_date_limit": int(per_date_limit),
            "min_score": float(min_score),
            "date": {
                "from": chunk_start.isoformat(),
                "to": chunk_stop.isoformat(),
                "include_live": False
            },
            "include": (
                self._factor_projection_include()
                if allow_factor_columns
                else list(FACTOR_ROWS_INCLUDE)
            )
        }
        if score_vector_rows is not None:
            body["score_vectors"] = score_vector_rows
        if filters:
            body["filters"] = filters
        try:
            response = self.cached_search(body=body)
            return [response], grouped_response_is_archive_clamped(
                response=response,
                request=body,
            )
        except WorldApiError as error:
            if not grouped_response_too_large(error=error):
                raise
            if chunk_start == chunk_stop:
                raise WorldApiError(
                    "World grouped response is too large for a one-day window; "
                    "reduce vector count or per_date_limit",
                    status=error.status,
                    body=error.body,
                    retry_after=error.retry_after,
                ) from error
            span = (chunk_stop - chunk_start).days
            midpoint = chunk_start + dt.timedelta(days=span // 2)
            right_start = midpoint + dt.timedelta(days=1)
            logs.info(
                message=(
                    f"splitting grouped date window {chunk_start} to {chunk_stop} "
                    f"into {chunk_start} to {midpoint} and {right_start} to {chunk_stop}"
                )
            )
            left_responses, left_reached_cutoff = self._grouped_window_with_cutoff(
                vector_rows=vector_rows,
                score_vector_rows=score_vector_rows,
                chunk_start=chunk_start,
                chunk_stop=midpoint,
                filters=filters,
                min_score=min_score,
                aggregate=aggregate,
                per_date_limit=per_date_limit,
                allow_factor_columns=allow_factor_columns,
            )
            if left_reached_cutoff:
                return left_responses, True
            right_responses, right_reached_cutoff = self._grouped_window_with_cutoff(
                vector_rows=vector_rows,
                score_vector_rows=score_vector_rows,
                chunk_start=right_start,
                chunk_stop=chunk_stop,
                filters=filters,
                min_score=min_score,
                aggregate=aggregate,
                per_date_limit=per_date_limit,
                allow_factor_columns=allow_factor_columns,
            )
            return left_responses + right_responses, right_reached_cutoff

    def cached_search(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Return an immutable grouped-search response from cache or the World API."""
        expects_columns = request_uses_factor_columns(request=body)
        namespace = "world-search-factor-columns-v1" if expects_columns else "world-search"
        cache_revision = "factor-columns-v1-adapter-1" if expects_columns else 3
        key = cache.digest(
            payload={
                "base_url": self.base_url,
                "authorization_scope": self.cache_scope,
                "request": body,
                "revision": cache_revision,
            }
        )
        path = cache.json_cache_path(
            cache_dir=self.cache_dir,
            namespace=namespace,
            key=key
        )
        cached = cache.read_json(path=path)
        if isinstance(cached, dict):
            try:
                cached_events = _validate_grouped_response(response=cached, request=body)
            except WorldApiError:
                pass
            else:
                cache_projection_matches = (
                    cached.get("schema") == FACTOR_COLUMNS_RESPONSE_SCHEMA
                    if expects_columns
                    else cached.get("schema") == GROUPED_RESPONSE_SCHEMA
                )
                if cache_projection_matches and not grouped_response_is_archive_clamped(
                    response=cached, request=body
                ):
                    return _normalise_compact_response(
                        response=cached,
                        events=cached_events,
                    )
        started = time.perf_counter()
        try:
            response = self.request(method="POST", path=SEARCH_PATH, payload=body)
        except Exception as error:
            logs.info(
                message=(
                    f"POST {SEARCH_PATH} failed elapsed={time.perf_counter() - started:.3f}s "
                    f"status={getattr(error, 'status', None)}"
                )
            )
            raise
        logs.info(
            message=(
                f"POST {SEARCH_PATH} completed elapsed={time.perf_counter() - started:.3f}s "
                f"from={body['date']['from']} to={body['date']['to']}"
            )
        )
        response_events = _validate_grouped_response(response=response, request=body)
        response_is_columns = response.get("schema") == FACTOR_COLUMNS_RESPONSE_SCHEMA
        if expects_columns and not response_is_columns:
            # A mixed-version deployment may advertise the feature through one replica and
            # answer the grouped request through another. Keep this valid legacy response for
            # the caller, but disable the opt-in for later chunks and never persist the fallback
            # under the compact-capability cache namespace.
            self._factor_columns_supported = False
        # A response clamped at today's archive edge is not immutable: the same request will
        # cover more dates as World advances. Never freeze that moving boundary in the cache.
        if (
            (not expects_columns or response_is_columns)
            and not grouped_response_is_archive_clamped(response=response, request=body)
        ):
            cache.write_json(path=path, payload=response)
        return _normalise_compact_response(response=response, events=response_events)

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Issue one request with bounded, ``Retry-After``-aware retries."""
        url = f"{self.base_url}{path}"
        last_status: Optional[int] = None
        last_body: Optional[str] = None
        last_retry_after: Optional[str] = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=payload,
                    timeout=self.timeout
                )
            except requests.RequestException as error:
                if attempt == MAX_ATTEMPTS - 1:
                    raise WorldApiError(f"{method} {path} failed: {error}") from error
                delay = min(
                    MAX_RETRY_DELAY_SECONDS,
                    1.5 * (attempt + 1) + random.uniform(0.0, 0.5)
                )
                time.sleep(delay)
                continue
            last_status = response.status_code
            last_body = response.text[:400]
            last_retry_after = response.headers.get("Retry-After")
            if response.status_code == 401:
                raise CredentialError("NOSIBLE World rejected the API key")
            if response.status_code == 403:
                raise AccessDeniedError(
                    "the API key does not grant date-grouped factor access",
                    status=403,
                    body=response.text[:400],
                    retry_after=last_retry_after
                )
            if (
                response.status_code in RETRY_STATUSES
                and retryable_response(response=response)
                and attempt < MAX_ATTEMPTS - 1
            ):
                delay = retry_delay(response=response, attempt=attempt)
                logs.info(
                    message=(
                        f"{method} {path} returned {response.status_code}; "
                        f"retrying in {delay:.1f}s"
                    )
                )
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                raise WorldApiError(
                    f"{method} {path} returned {response.status_code}",
                    status=response.status_code,
                    body=response.text[:400],
                    retry_after=last_retry_after
                )
            try:
                decoded = response.json()
            except ValueError as error:
                raise WorldApiError(f"{method} {path} returned invalid JSON") from error
            if not isinstance(decoded, dict):
                raise WorldApiError(f"{method} {path} returned a non-object JSON document")
            return decoded
        raise WorldApiError(
            f"{method} {path} exhausted its retries",
            status=last_status,
            body=last_body,
            retry_after=last_retry_after
        )


def search_schema_supports_factor_columns(response: Dict[str, Any]) -> bool:
    """Recognise only the exact, versioned compact-projection advertisement."""
    revision = response.get("revision")
    if (
        response.get("schema") != SEARCH_SCHEMA
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < SEARCH_SCHEMA_FACTOR_COLUMNS_REVISION
    ):
        return False
    grouped = response.get("date_grouped_semantic")
    if not isinstance(grouped, dict):
        return False
    projections = grouped.get("response_projections")
    if not isinstance(projections, dict):
        return False
    advertisement = projections.get(FACTOR_COLUMNS_RESPONSE_SCHEMA)
    if not isinstance(advertisement, dict):
        return False
    advertised_revision = advertisement.get("revision")
    return (
        isinstance(advertised_revision, int)
        and not isinstance(advertised_revision, bool)
        and _strict_json_equal(advertisement, FACTOR_COLUMNS_ADVERTISEMENT)
    )


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare a capability document without Python's bool-as-int equality loophole."""
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        try:
            actual_number = float(actual)
            expected_number = float(expected)
        except (OverflowError, TypeError, ValueError):
            return False
        return math.isfinite(actual_number) and actual_number == expected_number
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_strict_json_equal(actual[key], value) for key, value in expected.items())
        )
    return type(actual) is type(expected) and actual == expected


def request_uses_factor_columns(request: Dict[str, Any]) -> bool:
    """Return whether a grouped request explicitly opts into the compact projection."""
    include = request.get("include")
    return isinstance(include, list) and include == list(FACTOR_COLUMNS_INCLUDE)


def _decode_grouped_events(
    response: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Decode either response projection into ordered event rows without losing precision."""
    schema = response.get("schema")
    payload = response.get("events")
    if schema == GROUPED_RESPONSE_SCHEMA:
        if not isinstance(payload, list):
            raise WorldApiError("World grouped response has a malformed row event projection")
        if not all(isinstance(event, dict) for event in payload):
            raise WorldApiError("World grouped response contains a malformed event")
        return payload
    if schema != FACTOR_COLUMNS_RESPONSE_SCHEMA or not isinstance(payload, dict):
        raise WorldApiError("World grouped response has a malformed column event projection")
    length = payload.get("length")
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise WorldApiError("World grouped response has an invalid column event length")
    columns: Dict[str, List[Any]] = {}
    for field in FACTOR_EVENT_FIELDS:
        values = payload.get(field)
        if not isinstance(values, list) or len(values) != length:
            raise WorldApiError(
                f"World grouped response has a malformed {field} event column"
            )
        columns[field] = values
    return [
        {field: columns[field][index] for field in FACTOR_EVENT_FIELDS}
        for index in range(length)
    ]


def factor_event_rows(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Project either wire format into the exact ordered rows used by factor arithmetic."""
    payload = response.get("events")
    if response.get(NORMALIZED_FACTOR_COLUMNS_MARKER) is True:
        if (
            response.get("schema") != FACTOR_COLUMNS_RESPONSE_SCHEMA
            or not isinstance(payload, list)
            or not all(isinstance(event, dict) for event in payload)
        ):
            raise WorldApiError("World compact factor rows lost their internal normalisation")
        return payload
    decoded = _decode_grouped_events(response=response)
    if response.get("schema") == FACTOR_COLUMNS_RESPONSE_SCHEMA:
        return decoded
    return [
        {field: event[field] for field in FACTOR_EVENT_FIELDS}
        for event in decoded
    ]


def _normalise_compact_response(
    response: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach already-validated factor rows only to the private compact response path."""
    if response.get("schema") != FACTOR_COLUMNS_RESPONSE_SCHEMA:
        return response
    normalised = dict(response)
    normalised["events"] = events
    normalised[NORMALIZED_FACTOR_COLUMNS_MARKER] = True
    return normalised


def validate_grouped_response(response: Dict[str, Any], request: Dict[str, Any]) -> None:
    """Public validation helper retaining its historical no-return-value contract."""
    _validate_grouped_response(response=response, request=request)


def _validate_grouped_response(
    response: Dict[str, Any],
    request: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Reject ordinary, stale, or incomplete responses to a grouped search request."""
    schema = response.get("schema")
    if schema == GROUPED_RESPONSE_SCHEMA:
        expected_revision = GROUPED_RESPONSE_REVISION
    elif schema == FACTOR_COLUMNS_RESPONSE_SCHEMA and request_uses_factor_columns(request=request):
        expected_revision = FACTOR_COLUMNS_RESPONSE_REVISION
    else:
        raise WorldApiError("World returned a non-grouped-search response")
    response_revision = response.get("revision")
    if (
        not isinstance(response_revision, int)
        or isinstance(response_revision, bool)
        or response_revision != expected_revision
    ):
        raise WorldApiError("World returned an unsupported grouped-search revision")
    if response.get("group_by") != "date":
        raise WorldApiError("World returned a grouped response without date grouping")

    window = response.get("date_window")
    requested = request.get("date")
    if not isinstance(window, dict) or not isinstance(requested, dict):
        raise WorldApiError("World grouped response is missing its date window")
    try:
        effective_start = dt.date.fromisoformat(str(window["from"]))
        effective_stop = dt.date.fromisoformat(str(window["to"]))
        requested_start = dt.date.fromisoformat(str(requested["from"]))
        requested_stop = dt.date.fromisoformat(str(requested["to"]))
    except (KeyError, TypeError, ValueError) as error:
        raise WorldApiError("World grouped response has an invalid date window") from error
    if effective_start < requested_start or effective_stop > requested_stop:
        raise WorldApiError("World grouped response escaped the requested date window")
    if effective_stop < effective_start:
        raise WorldApiError("World grouped response has an inverted date window")
    try:
        archive_cutoff = dt.date.fromisoformat(str(window["archive_cutoff"]))
    except (KeyError, TypeError, ValueError) as error:
        raise WorldApiError("World grouped response has an invalid archive cutoff") from error
    if effective_stop > archive_cutoff:
        raise WorldApiError("World grouped response extends beyond its archive cutoff")

    groups = response.get("groups")
    if not isinstance(groups, list):
        raise WorldApiError("World grouped response is missing date groups")
    group_dates = [str(group.get("date") or "") for group in groups if isinstance(group, dict)]
    expected_dates = [
        (effective_start + dt.timedelta(days=offset)).isoformat()
        for offset in range((effective_stop - effective_start).days + 1)
    ]
    if len(group_dates) != len(groups) or len(set(group_dates)) != len(group_dates):
        raise WorldApiError("World grouped response contains duplicate or malformed dates")
    if group_dates != expected_dates:
        raise WorldApiError(
            "World grouped response does not cover every effective date in chronological order"
        )

    if response.get("vector_aggregate") != request.get("vector_aggregate"):
        raise WorldApiError("World grouped response changed the requested vector aggregate")
    echoed_limit = response.get("per_date_limit")
    requested_limit = request.get("per_date_limit")
    if (
        not isinstance(echoed_limit, int)
        or isinstance(echoed_limit, bool)
        or not isinstance(requested_limit, int)
        or isinstance(requested_limit, bool)
        or echoed_limit != requested_limit
    ):
        raise WorldApiError("World grouped response changed the requested per-date limit")
    try:
        echoed_floor = _require_finite(response["min_score"], "relevance floor")
        requested_floor = _require_finite(request["min_score"], "requested relevance floor")
    except KeyError as error:
        raise WorldApiError("World grouped response is missing its relevance floor") from error
    if not math.isclose(
        echoed_floor,
        requested_floor,
        rel_tol=0.0,
        abs_tol=GROUPED_MIN_SCORE_ECHO_TOLERANCE,
    ):
        raise WorldApiError("World grouped response changed the requested relevance floor")
    if window.get("include_live") is not requested.get("include_live"):
        raise WorldApiError("World grouped response changed the requested live-data setting")

    events = _decode_grouped_events(response=response)
    total = response.get("total")
    response_limit = response.get("response_event_limit")
    if not _is_count(total) or total != len(events):
        raise WorldApiError("World grouped response has an inconsistent event total")
    if not _is_count(response_limit) or response_limit < total:
        raise WorldApiError("World grouped response has an invalid event limit")

    vector_count = len(request.get("vectors") or [])
    score_vector_count = len(request.get("score_vectors") or [])
    event_dates: Dict[str, int] = {date: 0 for date in expected_dates}
    event_ids = set()
    prior_date = ""
    prior_score: Optional[float] = None
    for event in events:
        if not isinstance(event, dict):
            raise WorldApiError("World grouped response contains a malformed event")
        date = event.get("date")
        event_id = event.get("event_id")
        if (
            not isinstance(date, str)
            or date not in event_dates
            or not isinstance(event_id, str)
            or not event_id
        ):
            raise WorldApiError("World grouped response contains an invalid event identity")
        for field in ("title", "country"):
            if not isinstance(event.get(field), str):
                raise WorldApiError(
                    f"World grouped response contains an invalid event {field}"
                )
        if event_id in event_ids:
            raise WorldApiError("World grouped response contains a duplicate event")
        event_ids.add(event_id)
        event_dates[date] += 1
        event_score = _require_score(value=event.get("score"), label="event score")
        if event_score < requested_floor:
            raise WorldApiError("World grouped response contains an event below the requested floor")
        if schema == GROUPED_RESPONSE_SCHEMA:
            _require_score_row(
                value=event.get("vector_scores"),
                width=vector_count,
                label="retrieval-vector scores",
            )
            vector_scores = event["vector_scores"]
            expected_score: Optional[float] = None
            if request.get("vector_aggregate") == "max":
                expected_score = max(vector_scores)
            elif request.get("vector_aggregate") == "mean":
                expected_score = sum(vector_scores) / len(vector_scores)
            if expected_score is not None and not math.isclose(
                event_score, expected_score, rel_tol=0.0, abs_tol=1e-12
            ):
                raise WorldApiError(
                    "World grouped response event score contradicts its vector scores"
                )
        _require_score_row(
            value=event.get("score_vector_scores"),
            width=score_vector_count,
            label="pole-vector scores",
        )
        for field in ("total_coverage", "total_netlocs"):
            if _require_finite(value=event.get(field), label=f"event {field}") < 0:
                raise WorldApiError(f"World grouped response has a negative event {field}")
        for field in ("coverage_concentration_index", "materiality_score"):
            value = _require_finite(value=event.get(field), label=f"event {field}")
            if not 0.0 <= value <= 1.0:
                raise WorldApiError(f"World grouped response has an out-of-range event {field}")
        sentiment = event.get("sentiment")
        if not isinstance(sentiment, str) or sentiment not in {
            "negative",
            "neutral",
            "positive",
        }:
            raise WorldApiError("World grouped response contains an invalid event sentiment")
        if date < prior_date or (date == prior_date and prior_score is not None and event_score > prior_score):
            raise WorldApiError("World grouped response events are not in canonical order")
        prior_date = str(date)
        prior_score = event_score

    matched_total = 0
    candidate_total = 0
    retrieval_hit_total = 0
    eligible_total = 0
    truncated_total = 0
    for group in groups:
        date = str(group["date"])
        for field in ("matched_count", "candidate_count", "candidate_retrieval_hits"):
            if not _is_count(group.get(field)):
                raise WorldApiError(f"World grouped response has an invalid group {field}")
        if group["matched_count"] != event_dates[date]:
            raise WorldApiError("World grouped response group counts do not match its events")
        if group["matched_count"] > group["candidate_count"]:
            raise WorldApiError("World grouped response matches exceed its candidates")
        if not isinstance(group.get("truncated"), bool):
            raise WorldApiError("World grouped response has an invalid truncation marker")
        eligible = group.get("eligible")
        if not isinstance(eligible, dict) or not _is_count(eligible.get("count")):
            raise WorldApiError("World grouped response is missing an eligible denominator")
        for field in (
            "total_coverage",
            "total_netlocs",
            "total_materiality",
            "total_sentiment_weight",
        ):
            value = _require_finite(value=eligible.get(field), label=f"eligible {field}")
            if value < 0:
                raise WorldApiError(f"World grouped response has a negative eligible {field}")
        if group["candidate_count"] > eligible["count"]:
            raise WorldApiError("World grouped response candidates exceed its eligible universe")
        matched_total += group["matched_count"]
        candidate_total += group["candidate_count"]
        retrieval_hit_total += group["candidate_retrieval_hits"]
        eligible_total += eligible["count"]
        truncated_total += int(group["truncated"])

    stats = response.get("stats")
    if not isinstance(stats, dict):
        raise WorldApiError("World grouped response is missing retrieval statistics")
    expected_stats = {
        "dates_requested": len(groups),
        "dates_scanned": len(groups),
        "eligible_events": eligible_total,
        "candidate_retrieval_hits": retrieval_hit_total,
        "candidate_events": candidate_total,
        "matched_events": matched_total,
        "truncated_dates": truncated_total,
    }
    for field, expected in expected_stats.items():
        if not _is_count(stats.get(field)) or stats[field] != expected:
            raise WorldApiError(f"World grouped response has inconsistent {field}")
    if stats.get("truncated") is not bool(truncated_total):
        raise WorldApiError("World grouped response has inconsistent truncation statistics")
    if not _is_count(response.get("candidate_events")) or response["candidate_events"] != candidate_total:
        raise WorldApiError("World grouped response has inconsistent candidate totals")
    timing = response.get("timing_ms")
    if not isinstance(timing, dict) or _require_finite(timing.get("total"), "timing total") < 0:
        raise WorldApiError("World grouped response has invalid timing metadata")
    return events


def _is_count(value: Any) -> bool:
    """Return whether a value is a non-negative JSON integer rather than a bool."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _require_finite(value: Any, label: str) -> float:
    """Return one finite response number or reject the response."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise WorldApiError(f"World grouped response has an invalid {label}")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise WorldApiError(f"World grouped response has an invalid {label}") from error
    if not math.isfinite(number):
        raise WorldApiError(f"World grouped response has a non-finite {label}")
    return number


def _require_score(value: Any, label: str) -> float:
    """Require one exact cosine score."""
    number = _require_finite(value=value, label=label)
    if not -1.0 <= number <= 1.0:
        raise WorldApiError(f"World grouped response has an out-of-range {label}")
    return number


def _require_score_row(value: Any, width: int, label: str) -> None:
    """Require an exact-width finite cosine-score row."""
    if not isinstance(value, list) or len(value) != width:
        raise WorldApiError(f"World grouped response has malformed {label}")
    for score in value:
        _require_score(value=score, label=label)


def grouped_response_is_archive_clamped(
    response: Dict[str, Any],
    request: Dict[str, Any],
) -> bool:
    """Return whether a grouped response ended before the requested stop at the moving cutoff."""
    window = response.get("date_window")
    requested = request.get("date")
    if not isinstance(window, dict) or not isinstance(requested, dict):
        return False
    try:
        effective_stop = dt.date.fromisoformat(str(window["to"]))
        requested_stop = dt.date.fromisoformat(str(requested["to"]))
        archive_cutoff = dt.date.fromisoformat(str(window["archive_cutoff"]))
    except (KeyError, TypeError, ValueError):
        return False
    return effective_stop == archive_cutoff and effective_stop < requested_stop


def retryable_response(response: requests.Response) -> bool:
    """Retry transient responses but fail immediately on explicit grouped-search errors."""
    if response.status_code != 503:
        return True
    try:
        payload = response.json()
    except ValueError:
        return True
    if not isinstance(payload, dict):
        return True
    error = payload.get("error")
    return not error or error in RETRYABLE_503_ERRORS


def grouped_response_too_large(error: WorldApiError) -> bool:
    """Recognize the exact 422 that must be handled by deterministic date splitting."""
    if error.status != 422 or not error.body:
        return False
    try:
        payload = json.loads(error.body)
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("code") == "grouped_response_too_large" or payload.get("error") == "grouped_response_too_large"


def resolve_base_url(base_url: Optional[str] = None) -> str:
    """Resolve a deployment root without a trailing slash."""
    chosen = base_url or os.environ.get(WORLD_URL_VARIABLE, "").strip() or WORLD_BASE_URL
    return chosen.rstrip("/")


def resolve_world_key(api_key: Optional[str] = None) -> str:
    """Resolve the NOSIBLE key from the explicit value and supported environment variables."""
    if api_key:
        return api_key.strip()
    for variable in WORLD_KEY_VARIABLES:
        value = os.environ.get(variable, "").strip()
        if value:
            return value
    return ""


def day_chunks(start: dt.date, stop: dt.date, days: int) -> List[Tuple[dt.date, dt.date]]:
    """Tile an inclusive window into consecutive chunks of at most ``days`` calendar days."""
    if stop < start:
        raise ValueError(f"stop {stop} precedes start {start}")
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    windows: List[Tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= stop:
        edge = min(stop, cursor + dt.timedelta(days=days - 1))
        windows.append((cursor, edge))
        cursor = edge + dt.timedelta(days=1)
    return windows


def matrix_json(matrix: np.ndarray) -> List[List[float]]:
    """Validate a finite two-dimensional matrix and convert it to JSON-native rows."""
    values = np.asarray(a=matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError(f"expected a non-empty finite 2-D matrix, got shape {values.shape}")
    # Nine significant decimal digits are sufficient to round-trip every binary32 value
    # exactly, while avoiding Python's much longer binary64 representation in request JSON.
    return [[float(format(float(value), ".9g")) for value in row] for row in values]


def grouped_matrix_json(
    matrix: Optional[np.ndarray],
    field: str,
    required: bool,
) -> Optional[List[List[float]]]:
    """Validate a grouped-search matrix against World's count and dimension contract."""
    if matrix is None:
        if required:
            raise ValueError(f"{field} must contain at least one vector")
        return None
    try:
        values = np.asarray(a=matrix, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a rectangular numeric matrix") from error
    if not required and values.size == 0:
        return None
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"{field} must be a non-empty 2-D matrix, got shape {values.shape}")
    if values.shape[0] > MAX_GROUPED_VECTORS:
        raise ValueError(
            f"{field} contains {values.shape[0]} vectors, above the maximum {MAX_GROUPED_VECTORS}"
        )
    if values.shape[1] != WORLD_VECTOR_DIMENSIONS:
        raise ValueError(
            f"{field} vectors must have {WORLD_VECTOR_DIMENSIONS:,} dimensions, "
            f"got {values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{field} contains a non-finite value")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError(f"{field} contains a zero vector")
    return matrix_json(matrix=values)


def retry_delay(response: requests.Response, attempt: int) -> float:
    """Read ``Retry-After`` as seconds or an HTTP date, falling back to linear backoff."""
    raw = response.headers.get("Retry-After", "").strip()
    delay: Optional[float] = None
    if raw:
        try:
            delay = float(raw)
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(raw)
                if parsed is not None:
                    delay = parsed.timestamp() - time.time()
            except (OverflowError, TypeError, ValueError):
                delay = None
    base = delay if delay is not None and delay > 0 else 2.0 * (attempt + 1)
    # Jitter even authoritative Retry-After guidance so concurrent workers do
    # not wake on the same scheduler tick and recreate the overload.
    chosen = base + random.uniform(0.0, 0.5)
    return min(MAX_RETRY_DELAY_SECONDS, max(0.1, chosen))
