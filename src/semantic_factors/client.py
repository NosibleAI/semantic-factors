"""Authenticated access to NOSIBLE World's grouped semantic search.

The package has one data path: caller-supplied vectors are searched independently in every
date slice, exact scores are returned for the retained candidates, and the response includes
the exact eligible-universe totals needed by each built-in weighting scheme. Factor arithmetic
stays in this package; the World API remains a generic retrieval API.
"""
import datetime as dt
import email.utils
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from semantic_factors import cache, logs
from semantic_factors.errors import AccessDeniedError, CredentialError, WorldApiError

WORLD_BASE_URL = "https://nosible.world"
WORLD_KEY_VARIABLES = ("NOSIBLE_API_KEY", "SEMANTIC_FACTORS_API_KEY", "NOSIBLE_WORLD_API_KEY")
WORLD_URL_VARIABLE = "NOSIBLE_WORLD_URL"
SEARCH_PATH = "/api/search"
GROUPED_RESPONSE_SCHEMA = "nosible_world_factor_search_v1"
GROUPED_RESPONSE_REVISION = 3

# The Rust route refuses wider requests. Keeping the same constant here makes a long build
# resumable in small immutable cache entries rather than failing after one enormous response.
GROUPED_MAX_DAYS = 120
DEFAULT_PER_DATE_LIMIT = 1_000
MAX_CANDIDATE_RETRIEVALS = 2_000_000
MAX_SCORE_DOT_PRODUCTS = 50_000_000
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
        :return: One decoded response per date chunk, in chronological order.
        """
        vector_rows = matrix_json(matrix=vectors)
        score_vector_rows = (
            None
            if score_vectors is None or np.asarray(a=score_vectors).size == 0
            else matrix_json(matrix=score_vectors)
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
        responses: List[Dict[str, Any]] = []
        windows = day_chunks(start=start, stop=stop, days=chunk_days)
        for index, (chunk_start, chunk_stop) in enumerate(windows, start=1):
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
                "include": ["factor"]
            }
            if score_vector_rows is not None:
                body["score_vectors"] = score_vector_rows
            if filters:
                body["filters"] = filters
            responses.append(self.cached_search(body=body))
            if index == 1 or index % 10 == 0 or index == len(windows):
                logs.info(message=f"retrieved date chunk {index}/{len(windows)}")
        return responses

    def cached_search(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Return an immutable grouped-search response from cache or the World API."""
        key = cache.digest(payload={"base_url": self.base_url, "request": body, "revision": 3})
        path = cache.json_cache_path(
            cache_dir=self.cache_dir,
            namespace="world-search",
            key=key
        )
        cached = cache.read_json(path=path)
        if isinstance(cached, dict):
            try:
                validate_grouped_response(response=cached, request=body)
            except WorldApiError:
                pass
            else:
                return cached
        response = self.request(method="POST", path=SEARCH_PATH, payload=body)
        validate_grouped_response(response=response, request=body)
        cache.write_json(path=path, payload=response)
        return response

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Issue one request with bounded, ``Retry-After``-aware retries."""
        url = f"{self.base_url}{path}"
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
            if response.status_code == 401:
                raise CredentialError("NOSIBLE World rejected the API key")
            if response.status_code == 403:
                raise AccessDeniedError(
                    "the API key does not grant date-grouped factor access",
                    status=403,
                    body=response.text[:400]
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
                    body=response.text[:400]
                )
            try:
                decoded = response.json()
            except ValueError as error:
                raise WorldApiError(f"{method} {path} returned invalid JSON") from error
            if not isinstance(decoded, dict):
                raise WorldApiError(f"{method} {path} returned a non-object JSON document")
            return decoded
        raise WorldApiError(f"{method} {path} exhausted its retries")


def validate_grouped_response(response: Dict[str, Any], request: Dict[str, Any]) -> None:
    """Reject ordinary, stale, or incomplete responses to a grouped search request."""
    if response.get("schema") != GROUPED_RESPONSE_SCHEMA:
        raise WorldApiError("World returned a non-grouped-search response")
    if response.get("revision") != GROUPED_RESPONSE_REVISION:
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
    if set(group_dates) != set(expected_dates):
        raise WorldApiError("World grouped response does not cover every effective date")


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
    return not error or error == "grouped_search_busy"


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
    return [[float(value) for value in row] for row in values]


def retry_delay(response: requests.Response, attempt: int) -> float:
    """Read ``Retry-After`` as seconds or an HTTP date, falling back to linear backoff."""
    raw = response.headers.get("Retry-After", "").strip()
    delay: Optional[float] = None
    if raw:
        try:
            delay = float(raw)
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(raw)
            if parsed is not None:
                delay = parsed.timestamp() - time.time()
    chosen = delay if delay is not None and delay > 0 else 2.0 * (attempt + 1)
    return min(MAX_RETRY_DELAY_SECONDS, max(0.1, chosen))
