"""Turning anchor text into vectors.

NOSIBLE World embeds every event with ``text-embedding-3-large`` at 3,072 dimensions, so anchor
and pole text must use the same model. The resulting vectors are sent to the generic grouped
search contract; the embedding provider never sees NOSIBLE World events. Any OpenAI-compatible
embeddings endpoint works: pass ``base_url`` for Azure, OpenRouter, or a proxy.
"""
import email.utils
import os
import random
import time
from typing import List, Optional

import numpy as np
import requests

from semantic_factors import cache, geometry, logs
from semantic_factors.errors import CredentialError, EmbeddingError

#: The model World is built with. Changing it makes the scores incomparable with events.
EMBED_MODEL = "text-embedding-3-large"

#: Dimension World stores. The model supports truncation, but the events do not.
EMBED_DIMS = 3072

#: Default OpenAI-compatible endpoint for non-OpenRouter keys.
EMBED_BASE_URL = "https://api.openai.com/v1"

#: OpenRouter's OpenAI-compatible endpoint.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: OpenRouter requires the provider-qualified form of World's embedding model.
OPENROUTER_EMBED_MODEL = "openai/text-embedding-3-large"

#: Inputs per request. Large batches keep one response open for minutes; this stays under that.
EMBED_BATCH = 32

#: Environment variables consulted, in order, when no key is passed.
EMBED_KEY_VARIABLES = (
    "SEMANTIC_FACTORS_EMBED_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OR_API_KEY"
)

MAX_ATTEMPTS = 5


def embed_texts(
    texts: List[str],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = EMBED_MODEL,
    dims: int = EMBED_DIMS,
    cache_dir: Optional[str] = None
) -> np.ndarray:
    """
    Embed a list of strings and return unit-length vectors in input order.

    The whole list is cached under one content digest, so a repeated build with the same
    anchors makes no request at all and costs nothing.

    :param texts: Strings to embed. Order is preserved and significant to the cache key.
    :param api_key: Provider key, or None to read the environment.
    :param base_url: OpenAI-compatible API root, without a trailing slash.
    :param model: Embedding model name as the provider spells it.
    :param dims: Requested dimension, which must match what World stores.
    :param cache_dir: Cache root, or None for the default.
    :return: Unit-length float32 matrix with one row per input string.
    """
    if not texts:
        raise EmbeddingError("at least one string is required")
    resolved_base_url = resolve_embed_base_url(base_url=base_url, api_key=api_key)
    resolved_model = resolve_embed_model(model=model, base_url=resolved_base_url)
    root = cache.resolve_cache_dir(cache_dir=cache_dir)
    path = cache.vector_cache_path(
        cache_dir=root,
        namespace="anchors",
        key=cache.digest(
            payload={
                "base_url": resolved_base_url,
                "model": resolved_model,
                "dims": dims,
                "texts": list(texts)
            }
        )
    )
    cached = cache.read_vectors(path=path)
    if cached is not None and cached[0] == list(texts):
        return cached[1]

    key = resolve_embed_key(api_key=api_key)
    logs.info(message=f"embedding {len(texts)} strings with {resolved_model}")
    vectors = np.zeros(shape=(len(texts), dims), dtype=np.float32)
    with requests.Session() as session:
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            vectors[start:start + len(batch)] = embed_batch(
                session=session,
                texts=batch,
                api_key=key,
                base_url=resolved_base_url,
                model=resolved_model,
                dims=dims
            )
    vectors = geometry.l2(matrix=vectors)
    cache.write_vectors(
        path=path,
        labels=list(texts),
        vectors=vectors
    )
    return vectors


def embed_batch(
    session: requests.Session,
    texts: List[str],
    api_key: str,
    base_url: str,
    model: str,
    dims: int
) -> np.ndarray:
    """
    Embed one batch, retrying transport failures, rate limits, and server errors.

    :param session: Reused HTTP session with normal certificate verification.
    :param texts: Non-empty batch of strings.
    :param api_key: Provider key.
    :param base_url: OpenAI-compatible API root.
    :param model: Embedding model name.
    :param dims: Requested dimension.
    :return: Float32 matrix in input order.
    """
    payload = {"model": model, "input": texts, "dimensions": dims}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = session.post(
                url=f"{base_url.rstrip('/')}/embeddings",
                json=payload,
                headers=headers,
                timeout=(10, 120)
            )
        except requests.RequestException as error:
            if attempt == MAX_ATTEMPTS - 1:
                raise EmbeddingError(f"embedding request failed: {error}") from error
            logs.info(message=f"embedding retry {attempt + 1}: {error}")
            time.sleep(embedding_retry_delay(response=None, attempt=attempt))
            continue

        if response.status_code in (401, 403):
            raise CredentialError(
                f"the embedding provider rejected the key ({response.status_code})"
            )
        if response.status_code in (408, 425, 429) or response.status_code >= 500:
            transient_error = EmbeddingError(f"provider returned {response.status_code}")
            if attempt == MAX_ATTEMPTS - 1:
                raise transient_error
            logs.info(message=f"embedding retry {attempt + 1}: {transient_error}")
            time.sleep(embedding_retry_delay(response=response, attempt=attempt))
            continue
        try:
            response.raise_for_status()
        except requests.RequestException as error:
            raise EmbeddingError(f"embedding request failed: {error}") from error
        try:
            decoded = response.json()
        except ValueError as error:
            raise EmbeddingError("embedding provider returned invalid JSON") from error
        return parse_embeddings(payload=decoded, expected=len(texts), dims=dims)
    raise EmbeddingError("embedding request exhausted its retries")


def embedding_retry_delay(
    response: Optional[requests.Response],
    attempt: int,
) -> float:
    """Use bounded provider guidance plus jitter so concurrent workers spread out."""
    base = 1.5 * (attempt + 1)
    if response is not None:
        raw = response.headers.get("Retry-After", "").strip()
        try:
            advised = float(raw)
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(raw)
                advised = parsed.timestamp() - time.time() if parsed is not None else 0.0
            except (OverflowError, TypeError, ValueError):
                advised = 0.0
        if advised > 0:
            base = min(60.0, advised)
    return min(60.0, base + random.uniform(0.0, 0.5))


def parse_embeddings(payload: dict, expected: int, dims: int) -> np.ndarray:
    """
    Validate an embeddings response before it is allowed into the cache.

    :param payload: Decoded JSON body.
    :param expected: Number of inputs the request carried.
    :param dims: Required vector length.
    :return: Finite, non-zero float32 matrix in input order.
    """
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) != expected:
        raise EmbeddingError(f"expected {expected} embeddings, received {len(rows or [])}")
    if not all(isinstance(row, dict) for row in rows):
        raise EmbeddingError("embedding response data must contain objects")
    indices = [row.get("index") for row in rows]
    if any(index is not None for index in indices):
        if not all(isinstance(index, int) and not isinstance(index, bool) for index in indices):
            raise EmbeddingError("embedding response indices are incomplete or invalid")
        if sorted(indices) != list(range(expected)):
            raise EmbeddingError("embedding response indices do not match the requested inputs")
        rows = sorted(rows, key=lambda row: row["index"])
    try:
        vectors = np.asarray(a=[row["embedding"] for row in rows], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise EmbeddingError(f"embedding response contains an invalid vector: {error}") from error
    if vectors.shape != (expected, dims):
        raise EmbeddingError(f"expected shape ({expected}, {dims}), received {vectors.shape}")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(x=vectors, axis=1) == 0):
        raise EmbeddingError("embedding response contains a zero or non-finite vector")
    return vectors


def resolve_embed_key(api_key: Optional[str] = None) -> str:
    """
    Find the embedding provider key, explicit argument first, then the environment.

    :param api_key: Explicit key, or None.
    :return: The key.
    """
    if api_key:
        return api_key
    for variable in EMBED_KEY_VARIABLES:
        value = os.environ.get(variable, "").strip()
        if value:
            return value
    raise CredentialError(
        "an embedding key is required to embed factor anchors and poles. "
        f"Pass embed_api_key=... or set one of {', '.join(EMBED_KEY_VARIABLES)}. "
        "The key is sent only to the configured embedding provider."
    )


def resolve_embed_base_url(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """Resolve the embedding endpoint, selecting OpenRouter for its key format."""
    chosen = base_url or os.environ.get("SEMANTIC_FACTORS_EMBED_BASE_URL", "").strip()
    if chosen:
        return chosen.rstrip("/")
    candidate = api_key
    if not candidate:
        for variable in EMBED_KEY_VARIABLES:
            value = os.environ.get(variable, "").strip()
            if value:
                candidate = value
                break
    if candidate and candidate.startswith("sk-or-"):
        return OPENROUTER_BASE_URL
    return EMBED_BASE_URL


def resolve_embed_model(model: str, base_url: str) -> str:
    """Qualify only the default model when the selected provider is OpenRouter."""
    if base_url.rstrip("/") == OPENROUTER_BASE_URL and model == EMBED_MODEL:
        return OPENROUTER_EMBED_MODEL
    return model
