"""Content-addressed on-disk cache.

Two things are expensive enough to be worth keeping between runs: definition embeddings, which
cost money, and grouped World responses, which consume archive quota. Both are immutable given
their key, so the cache is content addressed and never needs invalidating.

Nothing here uses pickle. Vectors are stored as npz, metadata as JSON, and both are written
through a temporary file so an interrupted run cannot leave a half-written cache entry behind.
"""
import gzip
import hashlib
import io
import json
import os
import tempfile
import threading
import time
import weakref
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

DEFAULT_CACHE_ENV = "SEMANTIC_FACTORS_CACHE"
DEFAULT_CACHE_DIRNAME = ".semantic-factors"
ATOMIC_REPLACE_ATTEMPTS = 8
ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.005
CACHE_LOCK_TIMEOUT_SECONDS = 30.0
CACHE_LOCK_POLL_SECONDS = 0.005
_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()


def reset_path_locks_after_fork() -> None:
    """Discard thread locks inherited in an unknowable state by a forked child."""
    global _PATH_LOCKS_GUARD, _PATH_LOCKS
    _PATH_LOCKS_GUARD = threading.Lock()
    _PATH_LOCKS = weakref.WeakValueDictionary()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=reset_path_locks_after_fork)


class CacheLockTimeout(RuntimeError):
    """A best-effort cache operation could not acquire its bounded per-key lock."""


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
        with cache_entry_lock(path=path, shared=True):
            if not os.path.exists(path):
                return None
            try:
                with np.load(file=path, allow_pickle=False) as cached:
                    labels = [str(item) for item in cached["labels"].tolist()]
                    vectors = cached["vectors"].astype(np.float32, copy=False)
            except (EOFError, OSError, KeyError, TypeError, ValueError):
                return None
    except CacheLockTimeout:
        # Cache reads are an optimisation. A busy key is a miss, never a failed factor build.
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
    try:
        with cache_entry_lock(path=path, shared=False):
            reap_stranded_temporaries(path=path)
            temporary_path = ""
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{os.path.basename(path)}.",
                    suffix=".tmp",
                    dir=folder,
                    delete=False,
                ) as handle:
                    temporary_path = handle.name
                    np.savez(
                        file=handle,
                        labels=np.asarray(a=labels, dtype=np.str_),
                        vectors=np.asarray(a=vectors, dtype=np.float32)
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                replace_temporary(
                    temporary_path=temporary_path,
                    final_path=path
                )
            finally:
                cleanup_temporary(path=temporary_path)
    except CacheLockTimeout:
        # Another process can populate this immutable key. Skipping a cache write is safe.
        return


def read_json(path: str) -> Optional[Any]:
    """
    Read a cached JSON document, treating any damage as a miss.

    :param path: JSON path.
    :return: Decoded document, or None when absent or unreadable.
    """
    if not os.path.exists(path):
        return None
    try:
        with cache_entry_lock(path=path, shared=True):
            if not os.path.exists(path):
                return None
            try:
                with open(path, "rb") as probe:
                    compressed = probe.read(2) == b"\x1f\x8b"
                if compressed:
                    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
                        return json.load(handle)
                with open(path, mode="rt", encoding="utf-8") as handle:
                    return json.load(handle)
            except (EOFError, OSError, ValueError):
                return None
    except CacheLockTimeout:
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
    try:
        with cache_entry_lock(path=path, shared=False):
            reap_stranded_temporaries(path=path)
            temporary_path = ""
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{os.path.basename(path)}.",
                    suffix=".tmp",
                    dir=folder,
                    delete=False,
                ) as handle:
                    temporary_path = handle.name
                    with gzip.GzipFile(
                        fileobj=handle,
                        mode="wb",
                        compresslevel=6,
                        mtime=0,
                    ) as compressed:
                        with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                            json.dump(
                                payload,
                                text,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                    handle.flush()
                    os.fsync(handle.fileno())
                replace_temporary(
                    temporary_path=temporary_path,
                    final_path=path
                )
            finally:
                cleanup_temporary(path=temporary_path)
    except CacheLockTimeout:
        return


def path_lock(path: str) -> Any:
    """Return a bounded-lifetime in-process lock for one canonical cache path."""
    canonical = os.path.normcase(os.path.abspath(path))
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(canonical)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[canonical] = lock
        return lock


@contextmanager
def cache_entry_lock(path: str, shared: bool) -> Iterator[None]:
    """Acquire one key's thread lock then its crash-released OS sidecar lock."""
    deadline = time.monotonic() + CACHE_LOCK_TIMEOUT_SECONDS
    thread_lock = path_lock(path=path)
    remaining = max(0.0, deadline - time.monotonic())
    if not thread_lock.acquire(timeout=remaining):
        raise CacheLockTimeout(f"timed out acquiring in-process cache lock for {path}")
    try:
        with process_file_lock(path=path, shared=shared, deadline=deadline):
            yield
    finally:
        thread_lock.release()


@contextmanager
def process_file_lock(path: str, shared: bool, deadline: float) -> Iterator[None]:
    """Hold a per-key OS lock; closing or crashing releases it on Windows and POSIX."""
    lock_path = f"{os.path.abspath(path)}.lock"
    with open(lock_path, mode="a+b") as handle:
        if os.name == "nt":
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
        acquired = False
        while not acquired:
            acquired = try_process_file_lock(handle=handle, shared=shared)
            if acquired:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise CacheLockTimeout(f"timed out acquiring cross-process cache lock for {path}")
            time.sleep(min(CACHE_LOCK_POLL_SECONDS, remaining))
        try:
            yield
        finally:
            release_process_file_lock(handle=handle)


def try_process_file_lock(handle: Any, shared: bool) -> bool:
    """Attempt one non-blocking platform lock acquisition."""
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX  # type: ignore[attr-defined]
            fcntl.flock(  # type: ignore[attr-defined]
                handle.fileno(),
                operation | fcntl.LOCK_NB,  # type: ignore[attr-defined]
            )
    except OSError:
        return False
    return True


def release_process_file_lock(handle: Any) -> None:
    """Release a held platform lock without masking an error from the protected operation."""
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
    except OSError:
        # Closing the handle immediately afterwards is the authoritative crash-safe release.
        pass


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
    delay = ATOMIC_REPLACE_INITIAL_DELAY_SECONDS
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(src=written, dst=final_path)
            return
        except OSError as error:
            if attempt == ATOMIC_REPLACE_ATTEMPTS - 1 or not retryable_replace_error(error=error):
                raise
            time.sleep(delay)
            delay *= 2.0


def retryable_replace_error(error: OSError) -> bool:
    """Recognise Windows sharing/permission races that can clear after a reader closes."""
    return isinstance(error, PermissionError) or getattr(error, "winerror", None) in {5, 32, 33}


def cleanup_temporary(path: str) -> None:
    """Remove a private temporary after either replacement or a failed serialisation."""
    if not path:
        return
    try:
        os.unlink(path=path)
    except FileNotFoundError:
        pass


def reap_stranded_temporaries(path: str) -> None:
    """Best-effort removal of this key's private files under its exclusive cache lock."""
    folder = os.path.dirname(os.path.abspath(path))
    prefix = f".{os.path.basename(path)}."
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if (
                        entry.name.startswith(prefix)
                        and entry.name.endswith(".tmp")
                        and entry.is_file(follow_symlinks=False)
                    ):
                        cleanup_temporary(path=entry.path)
                except OSError:
                    # A stale handle or transient filesystem fault cannot fail a cache write.
                    continue
    except OSError:
        # Cache hygiene is never more important than producing the new complete entry.
        return
