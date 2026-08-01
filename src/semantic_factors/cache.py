"""Content-addressed on-disk cache.

Two things are expensive enough to be worth keeping between runs: definition embeddings, which
cost money, and grouped World responses, which consume archive quota. Both are immutable given
their key, so the cache is content addressed and never needs invalidating.

Nothing here uses pickle. Vectors are stored as npz, metadata as JSON, and both are written
through a temporary file so an interrupted run cannot leave a half-written cache entry behind.
"""
import hashlib
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_CACHE_ENV = "SEMANTIC_FACTORS_CACHE"
DEFAULT_CACHE_DIRNAME = ".semantic-factors"


def resolve_cache_dir(cache_dir: Optional[str] = None) -> str:
    """
    Resolve the cache root, creating it if necessary.

    Precedence is the explicit argument, then SEMANTIC_FACTORS_CACHE, then a directory in the
    user's home. A caller that wants no caching passes an empty string and gets a per-process
    temporary directory instead.

    :param cache_dir: Explicit cache root, or None to fall back to the environment and home.
    :return: Absolute path to an existing directory.
    """
    if cache_dir == "":
        root = os.path.join(tempfile.gettempdir(), f"{DEFAULT_CACHE_DIRNAME}-{os.getpid()}")
    else:
        root = (
            cache_dir
            or os.environ.get(DEFAULT_CACHE_ENV)
            or os.path.join(os.path.expanduser("~"), DEFAULT_CACHE_DIRNAME)
        )
    root = os.path.abspath(root)
    os.makedirs(name=root, exist_ok=True)
    return root


def digest(payload: Dict[str, Any]) -> str:
    """
    Hash a JSON-serialisable payload into a short, stable cache key.

    :param payload: Any JSON-serialisable description of what is being cached.
    :return: First 20 hexadecimal characters of the SHA-256 of the canonical encoding.
    """
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode(encoding="utf-8")).hexdigest()[:20]


def vector_cache_path(
    cache_dir: str,
    namespace: str,
    key: str
) -> str:
    """
    Resolve the npz path for one cache entry, creating its namespace directory.

    :param cache_dir: Cache root from resolve_cache_dir.
    :param namespace: Sub-directory grouping related entries, such as "anchors" or "events".
    :param key: Content digest.
    :return: Absolute npz path.
    """
    folder = os.path.join(cache_dir, namespace)
    os.makedirs(name=folder, exist_ok=True)
    return os.path.join(folder, f"{key}.npz")


def json_cache_path(cache_dir: str, namespace: str, key: str) -> str:
    """
    Resolve the JSON path for one cache entry, creating its namespace directory.

    :param cache_dir: Cache root from resolve_cache_dir.
    :param namespace: Sub-directory grouping related entries.
    :param key: Content digest.
    :return: Absolute JSON path.
    """
    folder = os.path.join(cache_dir, namespace)
    os.makedirs(name=folder, exist_ok=True)
    return os.path.join(folder, f"{key}.json")


def read_vectors(path: str) -> Optional[Tuple[List[str], np.ndarray]]:
    """
    Read a cached label/vector pair, treating any damage as a miss.

    :param path: npz path from vector_cache_path.
    :return: Tuple of labels and their float32 matrix, or None when absent or unreadable.
    """
    if not os.path.exists(path):
        return None
    try:
        with np.load(file=path, allow_pickle=False) as cached:
            labels = [str(item) for item in cached["labels"].tolist()]
            vectors = cached["vectors"].astype(np.float32, copy=False)
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if vectors.ndim != 2 or vectors.shape[0] != len(labels) or not np.isfinite(vectors).all():
        return None
    return labels, vectors


def write_vectors(
    path: str,
    labels: List[str],
    vectors: np.ndarray
) -> None:
    """
    Atomically write a label/vector pair.

    :param path: Final npz path.
    :param labels: One label per row, in row order.
    :param vectors: Float32 matrix with one row per label.
    :return: None.
    """
    if vectors.shape[0] != len(labels):
        raise ValueError(f"{vectors.shape[0]} vectors for {len(labels)} labels")
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(name=folder, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".npz", dir=folder, delete=False) as handle:
        temporary_path = handle.name
    try:
        np.savez(
            file=temporary_path,
            labels=np.asarray(a=labels, dtype=np.str_),
            vectors=np.asarray(a=vectors, dtype=np.float32)
        )
        replace_temporary(
            temporary_path=temporary_path,
            final_path=path
        )
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(path=temporary_path)
        raise


def read_json(path: str) -> Optional[Any]:
    """
    Read a cached JSON document, treating any damage as a miss.

    :param path: JSON path.
    :return: Decoded document, or None when absent or unreadable.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def write_json(path: str, payload: Any) -> None:
    """
    Atomically write a JSON document.

    :param path: Final JSON path.
    :param payload: JSON-serialisable document.
    :return: None.
    """
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(name=folder, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=folder,
        delete=False,
        encoding="utf-8"
    ) as handle:
        temporary_path = handle.name
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    replace_temporary(
        temporary_path=temporary_path,
        final_path=path
    )


def replace_temporary(temporary_path: str, final_path: str) -> None:
    """
    Move a completed temporary file into place.

    numpy appends .npz when the caller's name lacks the suffix, so the written file is not
    always the name handed to it.

    :param temporary_path: Path the writer was given.
    :param final_path: Destination path.
    :return: None.
    """
    written = temporary_path if os.path.exists(temporary_path) else f"{temporary_path}.npz"
    os.replace(src=written, dst=final_path)
