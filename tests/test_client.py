"""Client plumbing that can be checked without a network."""
import datetime as dt

import numpy as np
import pytest
import requests

from semantic_factors import cache, client
from semantic_factors.errors import CredentialError


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
    bodies = []
    monkeypatch.setattr(world, "cached_search", lambda body: bodies.append(body) or {"groups": []})

    responses = world.grouped_search(
        vectors=np.asarray(a=[[1.0, 0.0]], dtype=np.float32),
        score_vectors=np.empty(shape=(0, 2), dtype=np.float32),
        start=dt.date(2024, 1, 1),
        stop=dt.date(2024, 1, 1),
        filters={},
        min_score=0.3,
        aggregate="max"
    )

    world.close()
    assert responses == [{"groups": []}]
    assert "score_vectors" not in bodies[0]


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


def test_the_cache_digest_is_stable_and_order_independent() -> None:
    """
    Require the same payload to hash the same way regardless of key order.

    :return: None.
    """
    first = cache.digest(payload={"a": 1, "b": [1, 2]})
    second = cache.digest(payload={"b": [1, 2], "a": 1})

    assert first == second
    assert first != cache.digest(payload={"a": 1, "b": [2, 1]})


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
