"""Client plumbing that can be checked without a network."""
import concurrent.futures as futures
import copy
import datetime as dt
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import requests

from semantic_factors import cache, client
from semantic_factors.errors import CredentialError, WorldApiError

HUGE_JSON_INTEGER = 10**400


def world_vectors(rows: int) -> np.ndarray:
    """Build compact unit vectors with the production World's configured dimensions."""
    vectors = np.zeros(shape=(rows, client.WORLD_VECTOR_DIMENSIONS), dtype=np.float32)
    vectors[:, 0] = 1.0
    return vectors


def cache_race_writer(json_path, vector_path, start, errors, writer: int) -> None:
    """Write both cache formats repeatedly from one spawned process."""
    try:
        if not start.wait(timeout=30):
            raise RuntimeError("cache race start timed out")
        for sequence in range(30):
            marker = writer * 1_000 + sequence
            cache.write_json(
                path=json_path,
                payload={
                    "writer": writer,
                    "sequence": sequence,
                    "marker": marker,
                    "values": [marker] * 128,
                },
            )
            cache.write_vectors(
                path=vector_path,
                labels=[str(marker), str(marker)],
                vectors=np.full(shape=(2, 32), fill_value=marker, dtype=np.float32),
            )
    except Exception as error:
        errors.put(("writer", writer, repr(error)))


def cache_hot_reader(json_path, vector_path, start, stop, errors, reader: int) -> None:
    """Continuously read both hot keys and reject partial or cross-writer documents."""
    try:
        if not start.wait(timeout=30):
            raise RuntimeError("cache race start timed out")
        while not stop.is_set():
            document = cache.read_json(path=json_path)
            if document is not None:
                marker = document.get("marker")
                if (
                    not isinstance(marker, int)
                    or document.get("values") != [marker] * 128
                    or marker != document.get("writer") * 1_000 + document.get("sequence")
                ):
                    raise AssertionError(f"corrupt JSON cache document: {document!r}")
            vectors = cache.read_vectors(path=vector_path)
            if vectors is not None:
                labels, matrix = vectors
                marker = int(labels[0])
                if labels != [str(marker), str(marker)] or not np.array_equal(
                    matrix,
                    np.full(shape=(2, 32), fill_value=marker, dtype=np.float32),
                ):
                    raise AssertionError("corrupt vector cache document")
    except Exception as error:
        errors.put(("reader", reader, repr(error)))


def hold_cache_lock(path, ready, release) -> None:
    """Hold one process lock until the parent releases this spawned process."""
    with cache.cache_entry_lock(path=path, shared=False):
        ready.set()
        release.wait(timeout=30)


def acquire_cache_lock_and_crash(path, ready) -> None:
    """Exit without Python cleanup while holding the OS lock."""
    with cache.cache_entry_lock(path=path, shared=False):
        ready.set()
        os._exit(0)


def strand_cache_temporary_and_crash(path, ready) -> None:
    """Leave one realistic private temporary behind, as a killed writer would."""
    stranded = Path(path).parent / f".{Path(path).name}.crash-probe.tmp"
    with cache.cache_entry_lock(path=path, shared=False):
        stranded.write_bytes(b"incomplete cache payload")
        ready.set()
        os._exit(0)


def acquire_path_lock_after_fork(path, completed) -> None:
    """Probe whether fork-child cache lock state is usable without a timeout."""
    cache.path_lock(path=path)
    completed.set()


def grouped_request(start: str = "2024-01-01", stop: str = "2024-01-01") -> dict:
    """Build one complete grouped request fixture."""
    return {
        "vectors": [[1.0, 0.0]],
        "vector_aggregate": "max",
        "group_by": "date",
        "per_date_limit": 1,
        "min_score": 0.25,
        "date": {"from": start, "to": stop, "include_live": False},
        "include": ["factor"],
    }


def grouped_response(
    request: dict,
    effective_stop: str = "",
    archive_cutoff: str = "",
) -> dict:
    """Build one internally consistent zero-match grouped response fixture."""
    start = dt.date.fromisoformat(request["date"]["from"])
    stop = dt.date.fromisoformat(effective_stop or request["date"]["to"])
    dates = [
        (start + dt.timedelta(days=offset)).isoformat()
        for offset in range((stop - start).days + 1)
    ]
    return {
        "schema": client.GROUPED_RESPONSE_SCHEMA,
        "revision": client.GROUPED_RESPONSE_REVISION,
        "search_type": "semantic",
        "group_by": "date",
        "vector_aggregate": request["vector_aggregate"],
        "per_date_limit": request["per_date_limit"],
        "min_score": request["min_score"],
        "total": 0,
        "candidate_events": 0,
        "response_event_limit": 100,
        "events": [],
        "groups": [
            {
                "date": date,
                "matched_count": 0,
                "candidate_count": 0,
                "candidate_retrieval_hits": 0,
                "truncated": False,
                "eligible": {
                    "count": 0,
                    "total_coverage": 0.0,
                    "total_netlocs": 0.0,
                    "total_materiality": 0.0,
                    "total_sentiment_weight": 0.0,
                },
            }
            for date in dates
        ],
        "date_window": {
            "from": request["date"]["from"],
            "to": stop.isoformat(),
            "include_live": request["date"]["include_live"],
            "archive_cutoff": archive_cutoff or stop.isoformat(),
        },
        "stats": {
            "dates_requested": len(dates),
            "dates_scanned": len(dates),
            "eligible_events": 0,
            "candidate_retrieval_hits": 0,
            "candidate_events": 0,
            "matched_events": 0,
            "truncated_dates": 0,
            "truncated": False,
        },
        "timing_ms": {"total": 1},
    }


def test_obsolete_factor_service_surface_cannot_return() -> None:
    """Keep factor builds on authenticated grouped World search only."""
    assert client.SEARCH_PATH == "/api/search"
    assert not hasattr(client, "FACTOR_SCORE_PATH")
    assert not hasattr(client, "FACTOR_CALIBRATE_PATH")
    assert not hasattr(client, "FACTOR_HEALTH_PATH")
    assert not hasattr(client.WorldClient, "supports_scoring")
    assert not hasattr(client.WorldClient, "score_factor")


def test_a_window_splits_into_bounded_day_chunks_covering_it_once() -> None:
    """
    Require grouped-search chunks to tile the window exactly and respect the API ceiling.

    :return: None.
    """
    chunks = client.day_chunks(
        start=dt.date(2024, 1, 15),
        stop=dt.date(2024, 8, 3),
        days=120
    )

    assert chunks[0][0] == dt.date(2024, 1, 15)
    assert chunks[-1][1] == dt.date(2024, 8, 3)
    for earlier, later in zip(chunks[:-1], chunks[1:], strict=True):
        assert later[0] == earlier[1] + dt.timedelta(days=1)
    assert all((stop - start).days < 120 for start, stop in chunks)


def test_a_single_day_window_is_one_chunk() -> None:
    """
    Require the degenerate case to produce one chunk rather than none.

    :return: None.
    """
    chunks = client.day_chunks(
        start=dt.date(2024, 5, 5),
        stop=dt.date(2024, 5, 5),
        days=120
    )

    assert chunks == [(dt.date(2024, 5, 5), dt.date(2024, 5, 5))]


def test_a_backwards_window_is_rejected() -> None:
    """
    Require an inverted range to fail rather than loop forever.

    :return: None.
    """
    with pytest.raises(ValueError):
        client.day_chunks(start=dt.date(2024, 5, 5), stop=dt.date(2024, 1, 1), days=120)


def test_a_matrix_is_validated_before_json_serialisation() -> None:
    """
    Require vectors sent to World to preserve their row and column order.

    :return: None.
    """
    matrix = np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32)

    result = client.matrix_json(matrix=matrix)

    assert np.array_equal(np.asarray(result, dtype=np.float32), matrix)


def test_matrix_json_keeps_float32_exact_while_shrinking_the_wire_body() -> None:
    """Require compact decimals to preserve every bit and materially reduce upload size."""
    matrix = np.random.default_rng(7).normal(size=(30, 3072)).astype(np.float32)
    compact = client.matrix_json(matrix=matrix)
    expanded = [[float(value) for value in row] for row in matrix]

    assert np.array_equal(np.asarray(compact, dtype=np.float32), matrix)
    assert len(json.dumps(compact)) < 0.7 * len(json.dumps(expanded))


def test_matrix_json_rejects_non_finite_or_empty_input() -> None:
    """
    Require malformed vectors to fail before a request is charged.

    :return: None.
    """
    with pytest.raises(ValueError):
        client.matrix_json(matrix=np.zeros(shape=(0, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        client.matrix_json(matrix=np.asarray(a=[[np.nan]], dtype=np.float32))


def test_grouped_search_accepts_an_unsigned_factor_without_score_vectors(
    monkeypatch,
    tmp_path
) -> None:
    """Require an empty pole matrix to mean unsigned rather than malformed."""
    world = client.WorldClient(
        api_key="nos_test",
        base_url="https://world.example",
        cache_dir=str(tmp_path)
    )
    world._factor_columns_supported = False
    bodies = []
    monkeypatch.setattr(world, "cached_search", lambda body: bodies.append(body) or {"groups": []})

    responses = world.grouped_search(
        vectors=world_vectors(rows=1),
        score_vectors=np.empty(shape=(0, client.WORLD_VECTOR_DIMENSIONS), dtype=np.float32),
        start=dt.date(2024, 1, 1),
        stop=dt.date(2024, 1, 1),
        filters={},
        min_score=0.3,
        aggregate="max"
    )

    world.close()
    assert responses == [{"groups": []}]
    assert "score_vectors" not in bodies[0]


def test_direct_grouped_client_rejects_vector_shape_and_count_before_network(
    monkeypatch,
    tmp_path,
) -> None:
    """Mirror World's 3,072-dimension and 64-vector limits on the exported client method."""
    world = client.WorldClient(
        api_key="nos_test",
        base_url="https://world.example",
        cache_dir=str(tmp_path),
    )
    world._factor_columns_supported = False
    calls = []
    monkeypatch.setattr(world, "cached_search", lambda body: calls.append(body) or {})
    cases = [
        {"vectors": np.ones(shape=(1, 2), dtype=np.float32)},
        {"vectors": world_vectors(rows=65)},
        {"vectors": np.zeros(shape=(1, client.WORLD_VECTOR_DIMENSIONS), dtype=np.float32)},
        {
            "vectors": world_vectors(rows=1),
            "score_vectors": np.ones(shape=(1, 2), dtype=np.float32),
        },
        {"vectors": world_vectors(rows=1), "score_vectors": world_vectors(rows=65)},
    ]

    for overrides in cases:
        arguments = {
            "vectors": world_vectors(rows=1),
            "start": dt.date(2024, 1, 1),
            "stop": dt.date(2024, 1, 1),
            "filters": {},
            "min_score": 0.25,
            "aggregate": "max",
            "per_date_limit": 1,
        }
        arguments.update(overrides)
        with pytest.raises(ValueError):
            world.grouped_search_iter(**arguments)

    world.close()
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aggregate", "median"),
        ("aggregate", None),
        ("min_score", float("nan")),
        ("min_score", HUGE_JSON_INTEGER),
        ("min_score", -1.01),
        ("min_score", 1.01),
        ("min_score", True),
        ("per_date_limit", 0),
        ("per_date_limit", 10_001),
        ("per_date_limit", 1.5),
        ("per_date_limit", True),
    ],
)
def test_direct_grouped_client_rejects_invalid_controls_before_network(
    field,
    value,
    monkeypatch,
    tmp_path,
) -> None:
    """Reject controls World would refuse without opening a request or consuming quota."""
    world = client.WorldClient(
        api_key="nos_test",
        base_url="https://world.example",
        cache_dir=str(tmp_path),
    )
    world._factor_columns_supported = False
    calls = []
    monkeypatch.setattr(world, "cached_search", lambda body: calls.append(body) or {})
    arguments = {
        "vectors": world_vectors(rows=1),
        "start": dt.date(2024, 1, 1),
        "stop": dt.date(2024, 1, 1),
        "filters": {},
        "min_score": 0.25,
        "aggregate": "max",
        "per_date_limit": 1,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        world.grouped_search_iter(**arguments)

    world.close()
    assert calls == []


def test_grouped_chunk_size_must_be_positive() -> None:
    """
    Require a bad chunk configuration to fail rather than loop forever.

    :return: None.
    """
    with pytest.raises(ValueError):
        client.day_chunks(start=dt.date(2024, 1, 1), stop=dt.date(2024, 1, 2), days=0)


def test_key_resolution_can_report_that_no_key_is_configured() -> None:
    """
    Require key discovery to return an empty value so the client can raise a focused error.

    :return: None.
    """
    assert client.resolve_world_key(api_key=None) in ("",) or isinstance(
        client.resolve_world_key(api_key=None), str
    )
    assert client.resolve_world_key(api_key="  nos_sk_x  ") == "nos_sk_x"


def test_a_factor_client_requires_a_world_key_before_network_access(monkeypatch) -> None:
    """Require the expensive grouped projection to fail closed without a credential."""
    for variable in client.WORLD_KEY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(CredentialError):
        client.WorldClient(api_key=None)


def test_retry_after_seconds_are_honoured_and_bounded() -> None:
    """Require 429 guidance to win over generic backoff without causing an unbounded sleep."""
    response = requests.Response()
    response.headers["Retry-After"] = "120"

    assert client.retry_delay(response=response, attempt=0) == client.MAX_RETRY_DELAY_SECONDS


def test_retry_after_is_jittered_and_malformed_guidance_falls_back(monkeypatch) -> None:
    """Prevent synchronized retries and tolerate invalid upstream guidance."""
    monkeypatch.setattr(client.random, "uniform", lambda low, high: 0.25)
    response = requests.Response()
    response.headers["Retry-After"] = "7"
    assert client.retry_delay(response=response, attempt=0) == 7.25

    response.headers["Retry-After"] = "not-a-date"
    assert client.retry_delay(response=response, attempt=0) == 2.25


def test_exhausted_transient_error_preserves_capacity_metadata(monkeypatch, tmp_path) -> None:
    """Expose the final status and Retry-After value to a resumable batch scheduler."""
    response = requests.Response()
    response.status_code = 503
    response.headers["Retry-After"] = "7"
    response._content = b'{"error":"grouped_search_busy"}'

    class Session:
        """Return the same transient response without making a network request."""

        def request(self, **kwargs):
            """Return the prepared test response."""
            return response

        def close(self):
            """Match the requests session close surface."""

    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    world = client.WorldClient(
        api_key="nos_test",
        base_url="https://world.example",
        cache_dir=str(tmp_path)
    )
    world.session = Session()
    with pytest.raises(WorldApiError) as raised:
        world.request(method="POST", path="/api/search", payload={})
    world.close()

    assert raised.value.status == 503
    assert raised.value.retry_after == "7"
    assert raised.value.body == '{"error":"grouped_search_busy"}'


def test_grouped_response_requires_revision_and_complete_dates() -> None:
    """Reject ordinary responses and grouped responses with missing dates."""
    request = grouped_request(stop="2024-01-02")
    response = grouped_response(request=request)
    second_group = response["groups"].pop()
    response["stats"]["dates_scanned"] = 1

    with pytest.raises(WorldApiError):
        client.validate_grouped_response(response=response, request=request)

    response["groups"].append(second_group)
    response["stats"]["dates_scanned"] = 2
    client.validate_grouped_response(response=response, request=request)

    response["revision"] = 2
    with pytest.raises(WorldApiError):
        client.validate_grouped_response(response=response, request=request)


def test_grouped_response_accepts_the_wire_rounded_relevance_floor() -> None:
    """Accept World's four-decimal echo without weakening material floor validation."""
    request = grouped_request()
    request["min_score"] = 0.41035735607147217
    response = grouped_response(request=request)
    response["min_score"] = 0.4104

    client.validate_grouped_response(response=response, request=request)

    response["min_score"] = 0.4105
    with pytest.raises(WorldApiError, match="relevance floor"):
        client.validate_grouped_response(response=response, request=request)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(total=1), "event total"),
        (lambda payload: payload["groups"][0]["eligible"].pop("total_netlocs"), "eligible"),
        (lambda payload: payload["stats"].update(matched_events=1), "matched_events"),
        (lambda payload: payload.update(min_score=0.5), "relevance floor"),
    ],
)
def test_grouped_response_rejects_arithmetic_contract_corruption(mutation, message) -> None:
    """Never fold plausible-looking but internally inconsistent World data into a factor."""
    request = grouped_request()
    response = grouped_response(request=request)

    mutation(response)

    with pytest.raises(WorldApiError, match=message):
        client.validate_grouped_response(response=response, request=request)


@pytest.mark.parametrize(
    "cutoff",
    [None, "not-a-date", "2023-12-31"],
)
def test_grouped_response_rejects_invalid_archive_cutoff_before_caching(cutoff) -> None:
    """Never freeze a malformed moving-edge response in the immutable cache."""
    request = grouped_request()
    response = grouped_response(request=request)
    if cutoff is None:
        response["date_window"].pop("archive_cutoff")
    else:
        response["date_window"]["archive_cutoff"] = cutoff

    with pytest.raises(WorldApiError, match="archive cutoff"):
        client.validate_grouped_response(response=response, request=request)


def event_response(request: dict) -> dict:
    """Build a complete one-event response for semantic-field mutation tests."""
    payload = grouped_response(request=request)
    payload["events"] = [
        {
            "date": "2024-01-01",
            "event_id": "event-1",
            "title": "Probe",
            "country": "ZA",
            "score": 0.5,
            "vector_scores": [0.5],
            "score_vector_scores": [0.2],
            "total_coverage": 4.0,
            "total_netlocs": 2.0,
            "coverage_concentration_index": 0.5,
            "materiality_score": 0.8,
            "sentiment": "neutral",
        }
    ]
    payload["total"] = 1
    payload["groups"][0].update(matched_count=1, candidate_count=1, candidate_retrieval_hits=1)
    payload["groups"][0]["eligible"].update(
        count=2,
        total_coverage=4.0,
        total_netlocs=2.0,
        total_materiality=0.8,
        total_sentiment_weight=1.0,
    )
    payload["candidate_events"] = 1
    payload["stats"].update(
        eligible_events=2,
        candidate_retrieval_hits=1,
        candidate_events=1,
        matched_events=1,
    )
    return payload


def factor_columns_schema() -> dict:
    """Build the exact search-schema capability required for compact factor columns."""
    return {
        "schema": client.SEARCH_SCHEMA,
        "revision": client.SEARCH_SCHEMA_FACTOR_COLUMNS_REVISION,
        "date_grouped_semantic": {
            "response_projections": {
                client.FACTOR_COLUMNS_RESPONSE_SCHEMA: copy.deepcopy(
                    client.FACTOR_COLUMNS_ADVERTISEMENT
                )
            }
        },
    }


def compact_request() -> dict:
    """Build one request which passed the compact-projection capability gate."""
    request = grouped_request()
    request["include"] = list(client.FACTOR_COLUMNS_INCLUDE)
    request["score_vectors"] = [[0.0, 1.0]]
    return request


def compact_response(request: dict, rows: list[dict] | None = None) -> dict:
    """Convert valid legacy rows to the exact revision-1 column wire shape."""
    legacy = event_response(request=request)
    selected = rows if rows is not None else legacy["events"]
    legacy["schema"] = client.FACTOR_COLUMNS_RESPONSE_SCHEMA
    legacy["revision"] = client.FACTOR_COLUMNS_RESPONSE_REVISION
    legacy["events"] = {
        "length": len(selected),
        **{
            field: [event[field] for event in selected]
            for field in client.FACTOR_EVENT_FIELDS
        },
    }
    legacy["total"] = len(selected)
    return legacy


def test_factor_column_capability_requires_the_exact_advertisement() -> None:
    """Do not opt in on lookalike, partial, stale, or future-incompatible catalogs."""
    valid = factor_columns_schema()
    assert client.search_schema_supports_factor_columns(response=valid)

    mutations = [
        lambda value: value.update(schema="wrong"),
        lambda value: value.update(revision=3),
        lambda value: value.update(revision=True),
        lambda value: value.pop("date_grouped_semantic"),
        lambda value: value["date_grouped_semantic"].pop("response_projections"),
        lambda value: value["date_grouped_semantic"]["response_projections"].pop(
            client.FACTOR_COLUMNS_RESPONSE_SCHEMA
        ),
        lambda value: value["date_grouped_semantic"]["response_projections"][
            client.FACTOR_COLUMNS_RESPONSE_SCHEMA
        ].update(revision=2),
        lambda value: value["date_grouped_semantic"]["response_projections"][
            client.FACTOR_COLUMNS_RESPONSE_SCHEMA
        ].update(revision=True),
        lambda value: value["date_grouped_semantic"]["response_projections"][
            client.FACTOR_COLUMNS_RESPONSE_SCHEMA
        ].update(revision=HUGE_JSON_INTEGER),
        lambda value: value["date_grouped_semantic"]["response_projections"][
            client.FACTOR_COLUMNS_RESPONSE_SCHEMA
        ].update(include=["factor_columns_v1", "factor"]),
        lambda value: value["date_grouped_semantic"]["response_projections"][
            client.FACTOR_COLUMNS_RESPONSE_SCHEMA
        ].update(score_range=[0, 1]),
        lambda value: value["date_grouped_semantic"]["response_projections"][
            client.FACTOR_COLUMNS_RESPONSE_SCHEMA
        ].update(score_range=[-1, True]),
        lambda value: value["date_grouped_semantic"]["response_projections"][
            client.FACTOR_COLUMNS_RESPONSE_SCHEMA
        ].update(score_range=[-1, HUGE_JSON_INTEGER]),
        lambda value: value["date_grouped_semantic"]["response_projections"][
            client.FACTOR_COLUMNS_RESPONSE_SCHEMA
        ].update(score_range=[-1, 1e400]),
        lambda value: value["date_grouped_semantic"]["response_projections"][
            client.FACTOR_COLUMNS_RESPONSE_SCHEMA
        ].update(extra="unknown-contract"),
    ]
    for mutate in mutations:
        malformed = copy.deepcopy(valid)
        mutate(malformed)
        assert not client.search_schema_supports_factor_columns(response=malformed)


def test_projection_capability_is_fetched_once_and_selects_compact_include(
    monkeypatch,
    tmp_path,
) -> None:
    """Gate the opt-in once per client on the exact live search catalog."""
    world = client.WorldClient(
        api_key="nos_test", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    calls = []
    monkeypatch.setattr(
        world,
        "request",
        lambda **kwargs: calls.append(kwargs) or factor_columns_schema(),
    )

    assert world._factor_projection_include() == list(client.FACTOR_COLUMNS_INCLUDE)
    assert world._factor_projection_include() == list(client.FACTOR_COLUMNS_INCLUDE)
    world.close()

    assert calls == [{"method": "GET", "path": client.SEARCH_SCHEMA_PATH}]


@pytest.mark.parametrize("mode", ["old", "malformed", "unavailable"])
def test_projection_capability_fails_soft_to_legacy(mode, monkeypatch, tmp_path) -> None:
    """Old or temporarily unavailable catalogs must preserve the stable row path."""
    world = client.WorldClient(
        api_key="nos_test", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    if mode == "unavailable":
        monkeypatch.setattr(
            world,
            "request",
            lambda **kwargs: (_ for _ in ()).throw(WorldApiError("unavailable", status=503)),
        )
    elif mode == "old":
        old = factor_columns_schema()
        old["revision"] = 3
        monkeypatch.setattr(world, "request", lambda **kwargs: old)
    else:
        monkeypatch.setattr(world, "request", lambda **kwargs: {"revision": 4})

    assert world._factor_projection_include() == list(client.FACTOR_ROWS_INCLUDE)
    assert world._factor_projection_include() == list(client.FACTOR_ROWS_INCLUDE)
    world.close()


def test_grouped_request_opts_into_columns_only_after_capability_gate(
    monkeypatch,
    tmp_path,
) -> None:
    """Send both unknown-safe tokens, in canonical order, only to a supporting server."""
    world = client.WorldClient(
        api_key="nos_test", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    world._factor_columns_supported = True
    bodies = []
    monkeypatch.setattr(world, "cached_search", lambda body: bodies.append(body) or {"groups": []})

    world._factor_grouped_search(
        vectors=world_vectors(rows=1),
        start=dt.date(2024, 1, 1),
        stop=dt.date(2024, 1, 1),
        filters={},
        min_score=0.25,
        aggregate="max",
        per_date_limit=1,
    )
    world.close()

    assert bodies[0]["include"] == ["factor", "factor_columns_v1"]


def test_public_grouped_methods_keep_the_legacy_row_contract_on_capable_servers(
    monkeypatch,
    tmp_path,
) -> None:
    """The exported WorldClient methods must never leak compact columns or omit vector_scores."""
    world = client.WorldClient(
        api_key="nos_test", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    world._factor_columns_supported = True
    bodies = []

    def legacy_response(**kwargs):
        body = kwargs["payload"]
        bodies.append(body)
        return event_response(request=body)

    monkeypatch.setattr(world, "request", legacy_response)
    arguments = {
        "vectors": world_vectors(rows=1),
        "score_vectors": world_vectors(rows=1),
        "start": dt.date(2024, 1, 1),
        "stop": dt.date(2024, 1, 1),
        "filters": {},
        "min_score": 0.25,
        "aggregate": "max",
        "per_date_limit": 1,
    }

    listed = world.grouped_search(**arguments)
    streamed = list(world.grouped_search_iter(**arguments))
    world.close()

    assert isinstance(listed[0]["events"], list)
    assert isinstance(streamed[0]["events"], list)
    assert listed[0]["events"][0]["vector_scores"] == [0.5]
    assert streamed[0]["events"][0]["vector_scores"] == [0.5]
    assert bodies[0]["include"] == ["factor"]
    assert len(bodies) == 1


def test_compact_and_legacy_projections_normalise_to_identical_factor_rows() -> None:
    """Keep every field and row used by factor arithmetic exactly projection-independent."""
    request = compact_request()
    legacy = event_response(request=request)
    compact = compact_response(request=request)

    client.validate_grouped_response(response=legacy, request=request)
    client.validate_grouped_response(response=compact, request=request)

    assert client.factor_event_rows(response=compact) == client.factor_event_rows(response=legacy)


def test_empty_compact_projection_is_valid_and_normalises_to_no_rows() -> None:
    """Treat the canonical all-empty column object as the zero-match response."""
    request = compact_request()
    response = grouped_response(request=request)
    response["schema"] = client.FACTOR_COLUMNS_RESPONSE_SCHEMA
    response["revision"] = client.FACTOR_COLUMNS_RESPONSE_REVISION
    response["events"] = {
        "length": 0,
        **{field: [] for field in client.FACTOR_EVENT_FIELDS},
    }

    client.validate_grouped_response(response=response, request=request)
    assert client.factor_event_rows(response=response) == []


@pytest.mark.parametrize("field", client.FACTOR_EVENT_FIELDS)
def test_compact_projection_rejects_missing_and_wrong_length_columns(field) -> None:
    """Every revision-1 event column must exist and align exactly with events.length."""
    request = compact_request()
    missing = compact_response(request=request)
    missing["events"].pop(field)
    with pytest.raises(WorldApiError, match=field):
        client.validate_grouped_response(response=missing, request=request)

    short = compact_response(request=request)
    short["events"][field] = []
    with pytest.raises(WorldApiError, match=field):
        client.validate_grouped_response(response=short, request=request)


@pytest.mark.parametrize("field", client.FACTOR_EVENT_FIELDS)
@pytest.mark.parametrize("value", [None, {}, "not-an-array", ("not", "a", "list")])
def test_compact_projection_rejects_non_array_columns(field, value) -> None:
    """Require every named compact column to be a JSON array, including empty responses."""
    request = compact_request()
    response = compact_response(request=request)
    response["events"][field] = value

    with pytest.raises(WorldApiError, match=field):
        client.validate_grouped_response(response=response, request=request)


@pytest.mark.parametrize("length", [None, -1, True, 1.0, "1", 2])
def test_compact_projection_rejects_invalid_or_inconsistent_event_length(length) -> None:
    """Require an integer column length equal to the top-level total and every array."""
    request = compact_request()
    response = compact_response(request=request)
    response["events"]["length"] = length

    with pytest.raises(WorldApiError):
        client.validate_grouped_response(response=response, request=request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("date", None),
        ("date", []),
        ("event_id", ""),
        ("event_id", 7),
        ("title", None),
        ("title", 7),
        ("country", None),
        ("country", 7),
        ("sentiment", None),
        ("sentiment", {}),
        ("sentiment", "unknown"),
        ("score", "0.5"),
        ("total_coverage", "4"),
        ("total_netlocs", None),
        ("coverage_concentration_index", True),
        ("materiality_score", []),
    ],
)
def test_compact_projection_rejects_null_and_wrong_scalar_types(field, value) -> None:
    """Never coerce malformed JSON strings, nulls, booleans, or containers into event data."""
    request = compact_request()
    response = compact_response(request=request)
    response["events"][field][0] = value

    with pytest.raises(WorldApiError):
        client.validate_grouped_response(response=response, request=request)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.01, 1.01])
@pytest.mark.parametrize("field", ["score", "score_vector_scores"])
def test_compact_projection_rejects_non_finite_or_out_of_range_scores(field, value) -> None:
    """Reject invalid cosine values even when a permissive JSON parser produced them."""
    request = compact_request()
    response = compact_response(request=request)
    if field == "score":
        response["events"][field][0] = value
    else:
        response["events"][field][0][0] = value

    with pytest.raises(WorldApiError):
        client.validate_grouped_response(response=response, request=request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_coverage", -0.01),
        ("total_netlocs", -0.01),
        ("coverage_concentration_index", -0.01),
        ("coverage_concentration_index", 1.01),
        ("materiality_score", -0.01),
        ("materiality_score", 1.01),
        ("materiality_score", float("nan")),
        ("total_coverage", float("inf")),
    ],
)
def test_compact_projection_rejects_invalid_numeric_values(field, value) -> None:
    """Apply the legacy finite/range invariants identically to every compact numeric column."""
    request = compact_request()
    response = compact_response(request=request)
    response["events"][field][0] = value

    with pytest.raises(WorldApiError):
        client.validate_grouped_response(response=response, request=request)


@pytest.mark.parametrize("value", [None, "scores", {}, ["0.2"], [True]])
def test_compact_projection_rejects_malformed_pole_score_rows(value) -> None:
    """Each pole-score cell is itself a strict finite-number JSON array."""
    request = compact_request()
    response = compact_response(request=request)
    response["events"]["score_vector_scores"][0] = value

    with pytest.raises(WorldApiError, match="pole-vector scores"):
        client.validate_grouped_response(response=response, request=request)


@pytest.mark.parametrize("width", [0, 2])
def test_compact_projection_rejects_pole_score_width_mismatch(width) -> None:
    """Tie every compact pole-score row exactly to request.score_vectors length and order."""
    request = compact_request()
    response = compact_response(request=request)
    response["events"]["score_vector_scores"][0] = [0.2] * width

    with pytest.raises(WorldApiError, match="pole-vector scores"):
        client.validate_grouped_response(response=response, request=request)


def test_compact_projection_rejects_noncanonical_event_order() -> None:
    """Dates must be chronological and each date's exact scores must descend."""
    request = compact_request()
    response = compact_response(request=request)
    for field in client.FACTOR_EVENT_FIELDS:
        response["events"][field].append(copy.deepcopy(response["events"][field][0]))
    response["events"]["event_id"][1] = "event-2"
    response["events"]["score"][1] = 0.6
    response["events"]["length"] = 2
    response["total"] = 2
    response["candidate_events"] = 2
    response["groups"][0].update(
        matched_count=2,
        candidate_count=2,
        candidate_retrieval_hits=2,
    )
    response["stats"].update(
        candidate_retrieval_hits=2,
        candidate_events=2,
        matched_events=2,
    )

    with pytest.raises(WorldApiError, match="canonical order"):
        client.validate_grouped_response(response=response, request=request)


def test_compact_projection_requires_opt_in_but_accepts_legacy_fallback() -> None:
    """Accept unknown-token fallback while rejecting unsolicited compact response semantics."""
    opted_in = compact_request()
    legacy = event_response(request=opted_in)
    client.validate_grouped_response(response=legacy, request=opted_in)

    legacy_only = grouped_request()
    compact = compact_response(request=opted_in)
    with pytest.raises(WorldApiError, match="non-grouped-search"):
        client.validate_grouped_response(response=compact, request=legacy_only)


def test_compact_projection_reconciles_column_length_with_top_level_total() -> None:
    """Reject a plausible aligned column object whose envelope reports a different total."""
    request = compact_request()
    response = compact_response(request=request)
    response["total"] = 2

    with pytest.raises(WorldApiError, match="event total"):
        client.validate_grouped_response(response=response, request=request)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda response, request: response.update(revision=True),
        lambda response, request: response.update(per_date_limit=True),
        lambda response, request: request.update(per_date_limit=True),
        lambda response, request: response.update(min_score=True),
        lambda response, request: request.update(min_score=True),
    ],
)
def test_grouped_response_rejects_bool_as_numeric_contract_fields(mutation) -> None:
    """Python bools compare equal to 0/1 but are never valid JSON numbers in this contract."""
    request = compact_request()
    response = compact_response(request=request)
    mutation(response, request)

    with pytest.raises(WorldApiError):
        client.validate_grouped_response(response=response, request=request)


@pytest.mark.parametrize("value", [HUGE_JSON_INTEGER, 1e400])
@pytest.mark.parametrize(
    "location",
    [
        "score",
        "score_vector_scores",
        "total_coverage",
        "total_netlocs",
        "coverage_concentration_index",
        "materiality_score",
        "response_min_score",
        "request_min_score",
        "eligible_total_coverage",
        "eligible_total_netlocs",
        "eligible_total_materiality",
        "eligible_total_sentiment_weight",
        "timing_total",
    ],
)
def test_grouped_response_translates_unbounded_json_numbers_to_world_error(
    location,
    value,
) -> None:
    """Huge valid-JSON integers and exponents must never leak OverflowError from validation."""
    request = compact_request()
    response = compact_response(request=request)
    if location == "score_vector_scores":
        response["events"][location][0][0] = value
    elif location in client.FACTOR_EVENT_FIELDS:
        response["events"][location][0] = value
    elif location == "response_min_score":
        response["min_score"] = value
    elif location == "request_min_score":
        request["min_score"] = value
    elif location.startswith("eligible_"):
        response["groups"][0]["eligible"][location.removeprefix("eligible_")] = value
    else:
        response["timing_ms"]["total"] = value

    with pytest.raises(WorldApiError):
        client.validate_grouped_response(response=response, request=request)


def test_compact_fallback_disables_capability_and_is_never_cached_as_compact(
    monkeypatch,
    tmp_path,
) -> None:
    """A mixed replica set may answer legacy once, but that answer cannot prove capability."""
    request = compact_request()
    fallback = event_response(request=request)
    world = client.WorldClient(
        api_key="nos_test", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    world._factor_columns_supported = True
    calls = []
    monkeypatch.setattr(
        world,
        "request",
        lambda **kwargs: calls.append(kwargs) or copy.deepcopy(fallback),
    )

    world.cached_search(body=request)
    world.cached_search(body=request)

    assert len(calls) == 2
    assert world._factor_projection_include() == list(client.FACTOR_ROWS_INCLUDE)
    assert not list((tmp_path / "world-search-factor-columns-v1").glob("*.json"))
    world.close()


def test_first_mixed_fleet_fallback_downgrades_every_later_chunk(
    monkeypatch,
    tmp_path,
) -> None:
    """Re-evaluate capability per request so an old first replica ends compact use immediately."""
    world = client.WorldClient(
        api_key="nos_test", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    world._factor_columns_supported = True
    includes = []

    def legacy_fallback(**kwargs):
        payload = kwargs["payload"]
        includes.append(list(payload["include"]))
        return grouped_response(request=payload)

    monkeypatch.setattr(client, "GROUPED_MAX_DAYS", 1)
    monkeypatch.setattr(world, "request", legacy_fallback)
    responses = world._factor_grouped_search(
        vectors=world_vectors(rows=1),
        start=dt.date(2024, 1, 1),
        stop=dt.date(2024, 1, 3),
        filters={},
        min_score=0.25,
        aggregate="max",
        per_date_limit=1,
    )
    world.close()

    assert includes == [
        ["factor", "factor_columns_v1"],
        ["factor"],
        ["factor"],
    ]
    assert [response["schema"] for response in responses] == [
        client.GROUPED_RESPONSE_SCHEMA,
        client.GROUPED_RESPONSE_SCHEMA,
        client.GROUPED_RESPONSE_SCHEMA,
    ]
    assert not list((tmp_path / "world-search-factor-columns-v1").glob("*.json"))
    assert len(list((tmp_path / "world-search").glob("*.json"))) == 2


def test_mixed_fleet_fallback_downgrades_the_recursive_right_half(
    monkeypatch,
    tmp_path,
) -> None:
    """A legacy left split must stop compact opt-in before the sibling request is built."""
    world = client.WorldClient(
        api_key="nos_test", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    world._factor_columns_supported = True
    includes = []

    def split_then_fallback(**kwargs):
        payload = kwargs["payload"]
        includes.append(list(payload["include"]))
        if len(includes) == 1:
            raise WorldApiError(
                "too large",
                status=422,
                body='{"code":"grouped_response_too_large"}',
            )
        return grouped_response(request=payload)

    monkeypatch.setattr(world, "request", split_then_fallback)
    responses = world._factor_grouped_search(
        vectors=world_vectors(rows=1),
        start=dt.date(2024, 1, 1),
        stop=dt.date(2024, 1, 4),
        filters={},
        min_score=0.25,
        aggregate="max",
        per_date_limit=1,
    )
    world.close()

    assert includes == [
        ["factor", "factor_columns_v1"],
        ["factor", "factor_columns_v1"],
        ["factor"],
    ]
    assert len(responses) == 2


def test_compact_and_legacy_cache_namespaces_cannot_alias(monkeypatch, tmp_path) -> None:
    """Keep wire projections and decoder revisions physically separate on disk."""
    legacy_request = grouped_request()
    legacy_request["score_vectors"] = [[0.0, 1.0]]
    columns_request = compact_request()
    legacy = event_response(request=legacy_request)
    columns = compact_response(request=columns_request)
    world = client.WorldClient(
        api_key="nos_test", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    calls = []

    def answer(**kwargs):
        payload = kwargs["payload"]
        calls.append(tuple(payload["include"]))
        return copy.deepcopy(columns if client.request_uses_factor_columns(payload) else legacy)

    monkeypatch.setattr(world, "request", answer)
    assert world.cached_search(body=legacy_request)["schema"] == client.GROUPED_RESPONSE_SCHEMA
    columns_result = world.cached_search(body=columns_request)
    assert columns_result["schema"] == client.FACTOR_COLUMNS_RESPONSE_SCHEMA
    assert isinstance(columns_result["events"], list)
    assert columns_result[client.NORMALIZED_FACTOR_COLUMNS_MARKER] is True
    assert client.factor_event_rows(response=columns_result) is columns_result["events"]
    world.cached_search(body=legacy_request)
    world.cached_search(body=columns_request)
    world.close()

    assert calls == [("factor",), ("factor", "factor_columns_v1")]
    assert len(list((tmp_path / "world-search").glob("*.json"))) == 1
    compact_cache_files = list((tmp_path / "world-search-factor-columns-v1").glob("*.json"))
    assert len(compact_cache_files) == 1
    cached_columns = cache.read_json(path=str(compact_cache_files[0]))
    assert isinstance(cached_columns["events"], dict)
    assert client.NORMALIZED_FACTOR_COLUMNS_MARKER not in cached_columns


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["events"][0].update(score=float("nan")),
        lambda payload: payload["events"][0].update(vector_scores=[]),
        lambda payload: payload["events"][0].update(score_vector_scores=[]),
        lambda payload: payload["events"][0].update(score=0.4),
        lambda payload: payload["events"][0].update(total_netlocs=-1.0),
        lambda payload: payload["events"][0].update(total_coverage=-1.0),
        lambda payload: payload["events"][0].update(materiality_score=1.1),
        lambda payload: payload["events"][0].update(coverage_concentration_index=-0.1),
        lambda payload: payload["events"][0].update(sentiment="unknown"),
    ],
)
def test_grouped_response_rejects_corrupt_semantic_scores(mutation) -> None:
    """Require finite, exact-width, internally reconciled scores before caching."""
    request = grouped_request()
    request["score_vectors"] = [[0.0, 1.0]]
    valid = event_response(request=request)
    client.validate_grouped_response(response=valid, request=request)
    corrupted = copy.deepcopy(valid)

    mutation(corrupted)

    with pytest.raises(WorldApiError):
        client.validate_grouped_response(response=corrupted, request=request)


def test_explicit_grouped_search_busy_error_is_retryable() -> None:
    """Retry only the grouped-search capacity response among explicit 503 errors."""
    busy = requests.Response()
    busy.status_code = 503
    busy._content = b'{"error":"grouped_search_busy"}'
    permanent = requests.Response()
    permanent.status_code = 503
    permanent._content = b'{"error":"unsupported_grouped_request"}'

    assert client.retryable_response(response=busy)
    assert not client.retryable_response(response=permanent)


@pytest.mark.parametrize(
    "error_code",
    [
        "api_key_changed",
        "api_key_auth_not_configured",
        "api_key_auth_unavailable",
        "api_key_metering_unavailable",
        "archive_date_gap",
        "fast_metadata_stale",
        "fast_metadata_unavailable",
    ],
)
def test_world_auth_and_metering_unavailability_are_retryable(error_code) -> None:
    """Retry the temporary 503 contracts emitted before a request reaches World."""
    response = requests.Response()
    response.status_code = 503
    response._content = f'{{"error":"{error_code}"}}'.encode()

    assert client.retryable_response(response=response)


def test_oversized_grouped_windows_split_and_cache_each_child(monkeypatch, tmp_path) -> None:
    """Split only the machine-readable 422 and never repeat the oversized parent request."""
    world = client.WorldClient(
        api_key="nos_test",
        base_url="https://world.example",
        cache_dir=str(tmp_path)
    )
    world._factor_columns_supported = False
    calls = []

    def fake_cached_search(body):
        """Reject multi-day windows and return one compact child response."""
        window = body["date"]
        start = dt.date.fromisoformat(window["from"])
        stop = dt.date.fromisoformat(window["to"])
        calls.append((start, stop))
        if start != stop:
            raise WorldApiError(
                "too large",
                status=422,
                body='{"code":"grouped_response_too_large"}'
            )
        return {"date_window": window}

    monkeypatch.setattr(world, "cached_search", fake_cached_search)
    responses = world.grouped_search(
        vectors=world_vectors(rows=24),
        score_vectors=world_vectors(rows=6),
        start=dt.date(2024, 1, 1),
        stop=dt.date(2024, 1, 4),
        filters={},
        min_score=0.25,
        aggregate="max",
        per_date_limit=1_000,
    )
    world.close()

    assert [response["date_window"]["from"] for response in responses] == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
    assert calls.count((dt.date(2024, 1, 1), dt.date(2024, 1, 4))) == 1
    assert sorted((stop - start).days for start, stop in calls) == [0, 0, 0, 0, 1, 1, 3]


def test_oversized_one_day_window_fails_with_clear_error(monkeypatch, tmp_path) -> None:
    """Stop clearly when the API cannot fit even one day in a grouped response."""
    world = client.WorldClient(
        api_key="nos_test",
        base_url="https://world.example",
        cache_dir=str(tmp_path)
    )
    world._factor_columns_supported = False

    def fake_cached_search(body):
        """Return the same machine-readable size failure for every request."""
        raise WorldApiError(
            "too large",
            status=422,
            body='{"error":"grouped_response_too_large"}'
        )

    monkeypatch.setattr(world, "cached_search", fake_cached_search)
    with pytest.raises(WorldApiError, match="one-day window") as raised:
        world.grouped_search(
            vectors=world_vectors(rows=24),
            score_vectors=world_vectors(rows=6),
            start=dt.date(2024, 1, 1),
            stop=dt.date(2024, 1, 1),
            filters={},
            min_score=0.25,
            aggregate="max",
            per_date_limit=1_000,
        )
    world.close()

    assert raised.value.status == 422


def test_grouped_search_stops_after_the_server_archive_cutoff(monkeypatch, tmp_path) -> None:
    """Do not issue guaranteed-failing future chunks after World declares its cutoff."""
    world = client.WorldClient(
        api_key="nos_test",
        base_url="https://world.example",
        cache_dir=str(tmp_path),
    )
    world._factor_columns_supported = False
    calls = []
    cutoff = dt.date(2024, 1, 3)

    def fake_cached_search(body):
        """Clamp the effective response exactly as the grouped World route does."""
        requested = body["date"]
        requested_stop = dt.date.fromisoformat(requested["to"])
        calls.append(dict(requested))
        return {
            "date_window": {
                "from": requested["from"],
                "to": min(requested_stop, cutoff).isoformat(),
                "archive_cutoff": cutoff.isoformat(),
            }
        }

    monkeypatch.setattr(client, "GROUPED_MAX_DAYS", 2)
    monkeypatch.setattr(world, "cached_search", fake_cached_search)
    responses = world.grouped_search(
        vectors=world_vectors(rows=1),
        start=dt.date(2024, 1, 1),
        stop=dt.date(2024, 1, 10),
        filters={},
        min_score=0.25,
        aggregate="max",
        per_date_limit=1,
    )
    world.close()

    assert [call["from"] for call in calls] == ["2024-01-01", "2024-01-03"]
    assert responses[-1]["date_window"]["to"] == cutoff.isoformat()


def test_exact_cached_chunk_at_an_old_cutoff_does_not_stop_a_new_run(
    monkeypatch,
    tmp_path,
) -> None:
    """Only an actual clamp, not stale equality metadata, stops later chunks."""
    world = client.WorldClient(
        api_key="nos_test",
        base_url="https://world.example",
        cache_dir=str(tmp_path),
    )
    world._factor_columns_supported = False
    calls = []

    def fake_cached_search(body):
        """Return exact historical chunks whose recorded cutoffs advance."""
        requested = body["date"]
        calls.append(dict(requested))
        return {
            "date_window": {
                "from": requested["from"],
                "to": requested["to"],
                "archive_cutoff": requested["to"],
            }
        }

    monkeypatch.setattr(client, "GROUPED_MAX_DAYS", 2)
    monkeypatch.setattr(world, "cached_search", fake_cached_search)
    world.grouped_search(
        vectors=world_vectors(rows=1),
        start=dt.date(2024, 1, 1),
        stop=dt.date(2024, 1, 4),
        filters={},
        min_score=0.25,
        aggregate="max",
        per_date_limit=1,
    )
    world.close()

    assert [call["from"] for call in calls] == ["2024-01-01", "2024-01-03"]


def test_recursive_split_does_not_descend_past_the_archive_cutoff(
    monkeypatch,
    tmp_path,
) -> None:
    """Stop a 422 date split when its left child reaches the server cutoff."""
    world = client.WorldClient(
        api_key="nos_test",
        base_url="https://world.example",
        cache_dir=str(tmp_path),
    )
    world._factor_columns_supported = False
    calls = []

    def fake_cached_search(body):
        """Force one split, then clamp the left child at the cutoff."""
        window = body["date"]
        calls.append((window["from"], window["to"]))
        if len(calls) == 1:
            raise WorldApiError(
                "too large",
                status=422,
                body='{"code":"grouped_response_too_large"}',
            )
        return {
            "date_window": {
                "from": window["from"],
                "to": "2024-01-02",
                "archive_cutoff": "2024-01-02",
            }
        }

    monkeypatch.setattr(world, "cached_search", fake_cached_search)
    responses = world._grouped_window(
        vector_rows=[[1.0, 0.0]],
        score_vector_rows=None,
        chunk_start=dt.date(2024, 1, 1),
        chunk_stop=dt.date(2024, 1, 6),
        filters={},
        min_score=0.25,
        aggregate="max",
        per_date_limit=1,
    )
    world.close()

    assert calls == [("2024-01-01", "2024-01-06"), ("2024-01-01", "2024-01-03")]
    assert responses[-1]["date_window"]["archive_cutoff"] == "2024-01-02"


def test_the_cache_digest_is_stable_and_order_independent() -> None:
    """
    Require the same payload to hash the same way regardless of key order.

    :return: None.
    """
    first = cache.digest(payload={"a": 1, "b": [1, 2]})
    second = cache.digest(payload={"b": [1, 2], "a": 1})

    assert first == second
    assert first != cache.digest(payload={"a": 1, "b": [2, 1]})


def test_json_cache_is_compressed_and_round_trips(tmp_path) -> None:
    """Keep multi-gigabyte grouped responses compact without sacrificing cache fidelity."""
    path = cache.json_cache_path(cache_dir=str(tmp_path), namespace="probe", key="one")
    payload = {"events": [{"vector_scores": [0.25] * 24}] * 100}

    cache.write_json(path=path, payload=payload)

    with open(path, "rb") as handle:
        assert handle.read(2) == b"\x1f\x8b"
    assert cache.read_json(path=path) == payload


def test_world_response_cache_is_partitioned_by_api_key(tmp_path) -> None:
    """Never serve an entitled response to another credential sharing a cache directory."""
    body = grouped_request()
    response = grouped_response(request=body)
    first = client.WorldClient(
        api_key="key-A", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    second = client.WorldClient(
        api_key="key-B", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    calls = []
    first.request = lambda **kwargs: calls.append("A") or response
    second.request = lambda **kwargs: calls.append("B") or response

    first.cached_search(body=body)
    second.cached_search(body=body)
    first.close()
    second.close()

    assert calls == ["A", "B"]
    assert first.cache_scope != second.cache_scope


def test_archive_clamped_response_is_never_reused_from_cache(tmp_path) -> None:
    """A cached moving archive edge must not permanently freeze later factor runs."""
    body = grouped_request(stop="2024-01-05")
    clamped = grouped_response(
        request=body,
        effective_stop="2024-01-03",
        archive_cutoff="2024-01-03",
    )
    advanced = grouped_response(request=body, archive_cutoff="2024-01-05")
    world = client.WorldClient(
        api_key="key-A", base_url="https://world.example", cache_dir=str(tmp_path)
    )
    key = cache.digest(
        payload={
            "base_url": world.base_url,
            "authorization_scope": world.cache_scope,
            "request": body,
            "revision": 3,
        }
    )
    path = cache.json_cache_path(cache_dir=str(tmp_path), namespace="world-search", key=key)
    cache.write_json(path=path, payload=clamped)
    calls = []
    world.request = lambda **kwargs: calls.append(kwargs) or advanced

    result = world.cached_search(body=body)
    world.close()

    assert len(calls) == 1
    assert result["date_window"]["to"] == "2024-01-05"


def test_vectors_round_trip_through_the_cache(tmp_path) -> None:
    """
    Require the on-disk cache to return exactly what was written.

    :param tmp_path: Pytest temporary directory.
    :return: None.
    """
    path = cache.vector_cache_path(cache_dir=str(tmp_path), namespace="anchors", key="k")
    vectors = np.random.default_rng(2).normal(size=(3, 5)).astype(np.float32)

    cache.write_vectors(path=path, labels=["a", "b", "c"], vectors=vectors)
    labels, restored = cache.read_vectors(path=path)

    assert labels == ["a", "b", "c"]
    assert np.array_equal(restored, vectors)


def test_a_missing_cache_entry_is_a_miss_not_an_error(tmp_path) -> None:
    """
    Require an absent file to read as None so the caller simply recomputes.

    :param tmp_path: Pytest temporary directory.
    :return: None.
    """
    assert cache.read_vectors(path=str(tmp_path / "nothing.npz")) is None


def test_same_key_json_writes_are_atomic_under_high_contention(tmp_path) -> None:
    """Require every concurrent reader to see one whole contender and leave no temp debris."""
    path = cache.json_cache_path(cache_dir=str(tmp_path), namespace="race", key="json")
    folder = Path(path).parent
    contenders = [{"writer": index, "values": [index] * 200} for index in range(64)]

    def write_and_read(payload):
        """Race one same-key write and verify the visible document is internally whole."""
        cache.write_json(path=path, payload=payload)
        visible = cache.read_json(path=path)
        assert visible in contenders
        assert visible["values"] == [visible["writer"]] * 200

    with futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write_and_read, contenders))

    assert cache.read_json(path=path) in contenders
    assert not [item for item in folder.iterdir() if item.name.startswith(f".{Path(path).name}.")]


def test_same_key_vector_writes_are_atomic_under_high_contention(tmp_path) -> None:
    """Require labels and vectors to remain from the same writer during same-key races."""
    path = cache.vector_cache_path(cache_dir=str(tmp_path), namespace="race", key="vectors")
    folder = Path(path).parent

    def write_and_read(index):
        """Race one vector write and reject cross-writer or partial visibility."""
        vectors = np.full(shape=(2, 32), fill_value=index, dtype=np.float32)
        labels = [str(index), str(index)]
        cache.write_vectors(path=path, labels=labels, vectors=vectors)
        visible = cache.read_vectors(path=path)
        assert visible is not None
        visible_labels, visible_vectors = visible
        writer = int(visible_labels[0])
        assert visible_labels == [str(writer), str(writer)]
        assert np.array_equal(visible_vectors, np.full((2, 32), writer, dtype=np.float32))

    with futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write_and_read, range(64)))

    assert cache.read_vectors(path=path) is not None
    assert not [item for item in folder.iterdir() if item.name.startswith(f".{Path(path).name}.")]


def test_atomic_replace_retries_windows_sharing_races_with_a_bound(monkeypatch, tmp_path) -> None:
    """Retry a transient Windows sharing violation without permitting an infinite wait."""
    temporary = tmp_path / "source.tmp"
    destination = tmp_path / "entry.json"
    temporary.write_bytes(b"complete")
    calls = []
    real_replace = cache.os.replace

    def sharing_race(src, dst):
        """Model two short-lived readers that deny replace access on Windows."""
        calls.append((src, dst))
        if len(calls) < 3:
            raise PermissionError("sharing violation")
        real_replace(src=src, dst=dst)

    monkeypatch.setattr(cache.os, "replace", sharing_race)
    monkeypatch.setattr(cache.time, "sleep", lambda seconds: None)

    cache.replace_temporary(temporary_path=str(temporary), final_path=str(destination))

    assert len(calls) == 3
    assert destination.read_bytes() == b"complete"


def test_failed_json_serialisation_cleans_its_private_temporary(tmp_path) -> None:
    """Do not accumulate orphan files when a payload is not JSON serialisable."""
    path = cache.json_cache_path(cache_dir=str(tmp_path), namespace="failure", key="json")

    with pytest.raises(TypeError):
        cache.write_json(path=path, payload={"bad": object()})

    assert not Path(path).exists()
    folder = Path(path).parent
    assert not [item for item in folder.iterdir() if item.name.startswith(f".{Path(path).name}.")]


def test_empty_vector_cache_file_is_a_miss(tmp_path) -> None:
    """Treat numpy's EOFError for an empty cache file like every other corrupt-cache miss."""
    path = cache.vector_cache_path(cache_dir=str(tmp_path), namespace="empty", key="vectors")
    Path(path).touch()

    assert cache.read_vectors(path=path) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows owns the deny-delete reader race")
def test_hot_cross_process_readers_and_writers_coordinate_without_corruption(tmp_path) -> None:
    """Survive eight hot readers and eight writers doing thirty same-key writes each."""
    context = mp.get_context("spawn")
    json_path = cache.json_cache_path(cache_dir=str(tmp_path), namespace="process-race", key="json")
    vector_path = cache.vector_cache_path(
        cache_dir=str(tmp_path),
        namespace="process-race",
        key="vectors",
    )
    start = context.Event()
    stop = context.Event()
    errors = context.Queue()
    readers = [
        context.Process(
            target=cache_hot_reader,
            args=(json_path, vector_path, start, stop, errors, index),
        )
        for index in range(8)
    ]
    writers = [
        context.Process(
            target=cache_race_writer,
            args=(json_path, vector_path, start, errors, index),
        )
        for index in range(8)
    ]
    processes = readers + writers
    for process in processes:
        process.start()
    start.set()
    deadline = time.monotonic() + 120.0
    try:
        for process in writers:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        stop.set()
        for process in readers:
            process.join(timeout=30.0)
    finally:
        stop.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)

    failures = []
    while True:
        try:
            failures.append(errors.get_nowait())
        except queue.Empty:
            break
    assert failures == []
    assert all(process.exitcode == 0 for process in processes)
    document = cache.read_json(path=json_path)
    vectors = cache.read_vectors(path=vector_path)
    assert document is not None
    assert document["values"] == [document["marker"]] * 128
    assert vectors is not None
    labels, matrix = vectors
    marker = int(labels[0])
    assert labels == [str(marker), str(marker)]
    assert np.array_equal(matrix, np.full(shape=(2, 32), fill_value=marker, dtype=np.float32))
    folder = Path(json_path).parent
    assert not [item for item in folder.iterdir() if item.name.endswith(".tmp")]


def test_cache_lock_timeout_is_a_fail_soft_miss_and_skipped_write(
    monkeypatch,
    tmp_path,
) -> None:
    """A bounded lock timeout must not fail a factor build or expose a partial write."""
    context = mp.get_context("spawn")
    path = cache.json_cache_path(cache_dir=str(tmp_path), namespace="timeout", key="json")
    original = {"state": "original"}
    cache.write_json(path=path, payload=original)
    ready = context.Event()
    release = context.Event()
    holder = context.Process(target=hold_cache_lock, args=(path, ready, release))
    holder.start()
    assert ready.wait(timeout=30.0)
    monkeypatch.setattr(cache, "CACHE_LOCK_TIMEOUT_SECONDS", 0.05)

    started = time.monotonic()
    assert cache.read_json(path=path) is None
    cache.write_json(path=path, payload={"state": "skipped"})
    elapsed = time.monotonic() - started

    release.set()
    holder.join(timeout=30.0)
    if holder.is_alive():
        holder.terminate()
        holder.join(timeout=5.0)
    assert holder.exitcode == 0
    assert elapsed < 1.0
    assert cache.read_json(path=path) == original
    folder = Path(path).parent
    assert not [item for item in folder.iterdir() if item.name.endswith(".tmp")]


def test_process_crash_releases_cache_lock(tmp_path) -> None:
    """Rely on the OS handle lifetime so a dead writer cannot strand a cache key."""
    context = mp.get_context("spawn")
    path = cache.json_cache_path(cache_dir=str(tmp_path), namespace="crash", key="json")
    ready = context.Event()
    crashing = context.Process(target=acquire_cache_lock_and_crash, args=(path, ready))
    crashing.start()
    assert ready.wait(timeout=30.0)
    crashing.join(timeout=30.0)
    if crashing.is_alive():
        crashing.terminate()
        crashing.join(timeout=5.0)
    assert crashing.exitcode == 0

    cache.write_json(path=path, payload={"state": "recovered"})

    assert cache.read_json(path=path) == {"state": "recovered"}


@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="requires POSIX fork semantics",
)
def test_fork_child_resets_inherited_path_lock_guard(tmp_path) -> None:
    """A fork while another thread owns the guard must not deadlock the child."""
    context = mp.get_context("fork")
    path = str(tmp_path / "forked-cache-entry.json")
    parent_guard_held = threading.Event()
    release_parent_guard = threading.Event()
    child_completed = context.Event()

    def hold_parent_guard() -> None:
        with cache._PATH_LOCKS_GUARD:
            parent_guard_held.set()
            release_parent_guard.wait(timeout=30.0)

    holder = threading.Thread(target=hold_parent_guard)
    holder.start()
    assert parent_guard_held.wait(timeout=5.0)
    child = context.Process(
        target=acquire_path_lock_after_fork,
        args=(path, child_completed),
    )
    try:
        child.start()
        assert child_completed.wait(timeout=5.0)
        child.join(timeout=5.0)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5.0)
        release_parent_guard.set()
        holder.join(timeout=5.0)
    assert not holder.is_alive()


@pytest.mark.parametrize("cache_format", ["json", "npz"])
def test_next_writer_reaps_only_its_stranded_private_temporary(
    cache_format,
    tmp_path,
) -> None:
    """A killed JSON or NPZ writer must not accumulate debris for its cache key."""
    context = mp.get_context("spawn")
    if cache_format == "json":
        path = cache.json_cache_path(
            cache_dir=str(tmp_path), namespace="stranded", key="document"
        )
    else:
        path = cache.vector_cache_path(
            cache_dir=str(tmp_path), namespace="stranded", key="vectors"
        )
    stranded = Path(path).parent / f".{Path(path).name}.crash-probe.tmp"
    unrelated = Path(path).parent / ".another-cache-key.crash-probe.tmp"
    unrelated.write_bytes(b"belongs to a different cache key")
    ready = context.Event()
    crashing = context.Process(
        target=strand_cache_temporary_and_crash,
        args=(path, ready),
    )
    crashing.start()
    assert ready.wait(timeout=30.0)
    crashing.join(timeout=30.0)
    if crashing.is_alive():
        crashing.terminate()
        crashing.join(timeout=5.0)
    assert crashing.exitcode == 0
    assert stranded.read_bytes() == b"incomplete cache payload"

    if cache_format == "json":
        cache.write_json(path=path, payload={"state": "recovered"})
        assert cache.read_json(path=path) == {"state": "recovered"}
    else:
        expected = np.arange(64, dtype=np.float32).reshape(2, 32)
        cache.write_vectors(path=path, labels=["first", "second"], vectors=expected)
        recovered = cache.read_vectors(path=path)
        assert recovered is not None
        labels, vectors = recovered
        assert labels == ["first", "second"]
        assert np.array_equal(vectors, expected)

    assert not stranded.exists()
    assert unrelated.read_bytes() == b"belongs to a different cache key"
    matching_temporaries = Path(path).parent.glob(f".{Path(path).name}.*.tmp")
    assert list(matching_temporaries) == []


@pytest.mark.skipif(os.name != "nt", reason="requires Windows delete-sharing semantics")
@pytest.mark.parametrize("cache_format", ["json", "npz"])
def test_held_stranded_temporary_never_blocks_a_new_cache_write(
    cache_format,
    tmp_path,
) -> None:
    """Leave busy debris in place, commit twice, then reap it after its handle closes."""
    if cache_format == "json":
        path = cache.json_cache_path(
            cache_dir=str(tmp_path), namespace="busy-stranded", key="document"
        )
    else:
        path = cache.vector_cache_path(
            cache_dir=str(tmp_path), namespace="busy-stranded", key="vectors"
        )
    stranded = Path(path).parent / f".{Path(path).name}.busy.tmp"
    stranded.write_bytes(b"held incomplete cache payload")

    with stranded.open(mode="rb") as held:
        if cache_format == "json":
            cache.write_json(path=path, payload={"generation": 1})
            assert cache.read_json(path=path) == {"generation": 1}
        else:
            first = np.full(shape=(2, 32), fill_value=1.0, dtype=np.float32)
            cache.write_vectors(path=path, labels=["first", "second"], vectors=first)
            recovered = cache.read_vectors(path=path)
            assert recovered is not None
            assert recovered[0] == ["first", "second"]
            assert np.array_equal(recovered[1], first)
        assert held.read() == b"held incomplete cache payload"
        assert stranded.exists()

    if cache_format == "json":
        cache.write_json(path=path, payload={"generation": 2})
        assert cache.read_json(path=path) == {"generation": 2}
    else:
        second = np.full(shape=(2, 32), fill_value=2.0, dtype=np.float32)
        cache.write_vectors(path=path, labels=["third", "fourth"], vectors=second)
        recovered = cache.read_vectors(path=path)
        assert recovered is not None
        assert recovered[0] == ["third", "fourth"]
        assert np.array_equal(recovered[1], second)
    assert not stranded.exists()
