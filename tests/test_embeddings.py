"""Embedding response validation that never needs a provider request."""
import numpy as np
import pytest
import requests

from semantic_factors import embeddings
from semantic_factors.errors import EmbeddingError


def test_provider_rows_are_restored_to_input_order() -> None:
    """Require explicit provider indices to win over response order."""
    result = embeddings.parse_embeddings(
        payload={
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]}
            ]
        },
        expected=2,
        dims=2
    )

    assert np.array_equal(result, np.asarray(a=[[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_openrouter_key_selects_openrouter_endpoint() -> None:
    """Require an OpenRouter key to work without a provider-specific URL setting."""
    assert embeddings.resolve_embed_base_url(api_key="sk-or-example") == (
        "https://openrouter.ai/api/v1"
    )


def test_explicit_embedding_endpoint_wins_over_key_detection() -> None:
    """Require an explicit compatible endpoint to remain authoritative."""
    assert embeddings.resolve_embed_base_url(
        base_url="https://embedding.example/v1/",
        api_key="sk-or-example"
    ) == "https://embedding.example/v1"


def response(status: int, payload: str = "{}", retry_after: str = "") -> requests.Response:
    """Build a requests response without network access."""
    result = requests.Response()
    result.status_code = status
    result._content = payload.encode("utf-8")
    result.url = "https://embedding.example/v1/embeddings"
    if retry_after:
        result.headers["Retry-After"] = retry_after
    return result


def test_openrouter_default_model_is_provider_qualified(monkeypatch, tmp_path) -> None:
    """Exercise the uncached automatic path and assert the exact provider payload."""
    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, **kwargs):
            calls.append(kwargs)
            return response(200, '{"data":[{"index":0,"embedding":[1.0,0.0]}]}')

    monkeypatch.setattr(embeddings.requests, "Session", Session)
    result = embeddings.embed_texts(
        texts=["probe"],
        api_key="sk-or-example",
        dims=2,
        cache_dir=str(tmp_path),
    )

    assert result.shape == (1, 2)
    assert calls[0]["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert calls[0]["json"]["model"] == "openai/text-embedding-3-large"


def test_deterministic_embedding_4xx_is_not_retried() -> None:
    """Fail paid bad requests immediately instead of repeating them five times."""
    calls = []

    class Session:
        def post(self, **kwargs):
            calls.append(kwargs)
            return response(400, '{"error":"bad model"}')

    with pytest.raises(EmbeddingError):
        embeddings.embed_batch(
            session=Session(),
            texts=["probe"],
            api_key="key",
            base_url="https://embedding.example/v1",
            model="bad-model",
            dims=2,
        )

    assert len(calls) == 1


def test_embedding_rate_limit_honours_retry_after(monkeypatch) -> None:
    """Use bounded provider guidance before retrying a transient rate limit."""
    sleeps = []
    replies = [
        response(429, retry_after="7"),
        response(200, '{"data":[{"index":0,"embedding":[1.0,0.0]}]}'),
    ]

    class Session:
        def post(self, **kwargs):
            return replies.pop(0)

    monkeypatch.setattr(embeddings.random, "uniform", lambda low, high: 0.0)
    monkeypatch.setattr(embeddings.time, "sleep", sleeps.append)
    embeddings.embed_batch(
        session=Session(),
        texts=["probe"],
        api_key="key",
        base_url="https://embedding.example/v1",
        model="model",
        dims=2,
    )

    assert sleeps == [7.0]


def test_embedding_rate_limit_accepts_http_date_retry_after(monkeypatch) -> None:
    """Support both standard Retry-After representations without unbounded sleeps."""
    reply = response(429, retry_after="Wed, 05 Aug 2026 12:00:07 GMT")
    monkeypatch.setattr(embeddings.time, "time", lambda: 1785931200.0)
    monkeypatch.setattr(embeddings.random, "uniform", lambda low, high: 0.0)

    assert embeddings.embedding_retry_delay(response=reply, attempt=0) == 7.0


@pytest.mark.parametrize(
    "rows",
    [
        ["not an object"],
        [{"index": 2, "embedding": [1.0, 0.0]}],
        [{"index": True, "embedding": [1.0, 0.0]}],
        [{"index": 0}],
        [{"index": 0, "embedding": [float("nan"), 0.0]}]
    ]
)
def test_malformed_provider_rows_are_rejected(rows: list) -> None:
    """Require bad indices and vectors to stay out of the content cache."""
    with pytest.raises(EmbeddingError):
        embeddings.parse_embeddings(payload={"data": rows}, expected=1, dims=2)
