"""Hugging Face model download service wrapper."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar
from unittest.mock import patch


def _apply_hf_env_defaults() -> None:
    raw_use_xet = os.environ.get("LTX_HF_USE_XET", "").strip().lower()
    if raw_use_xet in {"1", "true", "yes", "on"}:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    elif raw_use_xet in {"0", "false", "no", "off"}:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    else:
        # Large desktop model downloads are more reliable over plain HTTP than Xet.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if os.environ.get("HF_HUB_DISABLE_XET") == "1":
            os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


_apply_hf_env_defaults()

from huggingface_hub import file_download, hf_hub_download, snapshot_download  # type: ignore[reportUnknownVariableType]
from tqdm.auto import tqdm as tqdm_auto  # type: ignore[reportUnknownVariableType]

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        logger.warning("Ignoring invalid integer environment override %s=%r", name, raw_value)
        return default


def _configure_hf_backend() -> None:
    _apply_hf_env_defaults()


def _run_with_retries(label: str, operation: Callable[[], T]) -> T:
    max_attempts = _env_int("LTX_HF_DOWNLOAD_RETRIES", 4)
    backoff_seconds = _env_int("LTX_HF_DOWNLOAD_RETRY_BACKOFF_SECONDS", 3)

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            delay = backoff_seconds * attempt
            logger.warning(
                "Retrying %s after download failure (%s/%s): %s",
                label,
                attempt,
                max_attempts,
                exc,
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error


def _make_progress_tqdm_class(callback: Callable[[int], None]) -> type:
    """Return a tqdm subclass that reports aggregated progress via *callback*.

    Used for both single-file and snapshot downloads.  Snapshot downloads
    spawn one tqdm instance per file; all instances share mutable state so
    the callback reports total progress across every file in the download.
    """
    lock = Lock()
    shared: dict[str, int] = {"downloaded": 0}

    class _ProgressTqdm(tqdm_auto):  # type: ignore[reportUntypedBaseClass]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)  # type: ignore[reportUnknownMemberType]

        def update(self, n: float | int | None = 1) -> bool | None:  # type: ignore[reportIncompatibleMethodOverride]
            result = super().update(n)
            if n is not None:
                with lock:
                    shared["downloaded"] += int(n)
                callback(shared["downloaded"])
            return result

    return _ProgressTqdm


@contextlib.contextmanager
def _patch_download_progress(callback: Callable[[int], None]) -> Iterator[None]:
    """Temporarily monkey-patch ``huggingface_hub.file_download.http_get``
    and ``xet_get`` to inject a custom tqdm bar that forwards progress to
    *callback*.

    ``hf_hub_download`` does not expose a ``tqdm_class`` parameter (unlike
    ``snapshot_download``), but its internal ``http_get`` and ``xet_get``
    both accept a private ``_tqdm_bar`` kwarg.  We wrap them to inject our
    own bar when the caller hasn't already provided one.

    See ``test_http_get_accepts_tqdm_bar`` — if that test breaks after a
    huggingface_hub upgrade, this patch needs to be revisited.
    """
    tqdm_cls = _make_progress_tqdm_class(callback)
    original_http_get: Callable[..., Any] = file_download.http_get  # type: ignore[reportUnknownMemberType]

    def _wrapped_http_get(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("_tqdm_bar") is None:
            kwargs["_tqdm_bar"] = tqdm_cls(disable=True)
        return original_http_get(*args, **kwargs)

    xet_get_fn: Callable[..., Any] | None = getattr(file_download, "xet_get", None)

    def _wrapped_xet_get(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("_tqdm_bar") is None:
            kwargs["_tqdm_bar"] = tqdm_cls(disable=True)
        assert xet_get_fn is not None
        return xet_get_fn(*args, **kwargs)

    with patch.object(file_download, "http_get", _wrapped_http_get):
        if xet_get_fn is not None:
            with patch.object(file_download, "xet_get", _wrapped_xet_get):
                yield
        else:
            yield


class HuggingFaceDownloader:
    """Wraps huggingface_hub download functions."""

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        _configure_hf_backend()
        ctx = _patch_download_progress(on_progress) if on_progress is not None else contextlib.nullcontext()
        with ctx:
            path = _run_with_retries(
                f"{repo_id}/{filename}",
                lambda: hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    etag_timeout=float(_env_int("LTX_HF_ETAG_TIMEOUT_SECONDS", 30)),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                ),
            )
        return Path(path)

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        _configure_hf_backend()
        ctx = _patch_download_progress(on_progress) if on_progress is not None else contextlib.nullcontext()
        with ctx:
            path = _run_with_retries(
                repo_id,
                lambda: snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    etag_timeout=float(_env_int("LTX_HF_ETAG_TIMEOUT_SECONDS", 30)),
                    local_dir_use_symlinks=False,
                    max_workers=_env_int("LTX_HF_SNAPSHOT_MAX_WORKERS", 2),
                    resume_download=True,
                ),
            )
        return Path(path)
