"""Hugging Face model download service wrapper."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        logger.warning("Ignoring invalid integer environment override %s=%r", name, raw_value)
        return default


def _find_hf_cli() -> str:
    executable_dir = Path(sys.executable).resolve().parent
    candidates = [
        executable_dir / "hf",
        executable_dir / "hf.exe",
        executable_dir / "huggingface-cli",
        executable_dir / "huggingface-cli.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    for name in ("hf", "huggingface-cli"):
        resolved = shutil.which(name)
        if resolved:
            return resolved

    raise RuntimeError("Could not find the Hugging Face CLI (`hf`).")


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
        return total
    return 0


def _hf_command_env() -> dict[str, str]:
    _apply_hf_env_defaults()
    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return env


def _run_hf_download(
    *,
    command_args: list[str],
    progress_path: Path,
    on_progress: Callable[[int], None] | None,
    label: str,
) -> str:
    max_attempts = _env_int("LTX_HF_DOWNLOAD_RETRIES", 6)
    backoff_seconds = _env_int("LTX_HF_DOWNLOAD_RETRY_BACKOFF_SECONDS", 3)
    poll_interval = float(_env_int("LTX_HF_PROGRESS_POLL_SECONDS", 1))
    hf_cli = _find_hf_cli()

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        command = [hf_cli, "download", *command_args]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_hf_command_env(),
        )

        last_reported = -1
        while process.poll() is None:
            if on_progress is not None:
                current_size = _path_size(progress_path)
                if current_size != last_reported:
                    on_progress(current_size)
                    last_reported = current_size
            time.sleep(poll_interval)

        output, _ = process.communicate()

        if on_progress is not None:
            final_size = _path_size(progress_path)
            if final_size != last_reported:
                on_progress(final_size)

        if process.returncode == 0:
            return output.strip()

        last_error = RuntimeError(output.strip() or f"`hf download` failed for {label} with exit code {process.returncode}")
        if attempt >= max_attempts:
            break

        delay = backoff_seconds * attempt
        logger.warning(
            "Retrying %s after hf CLI download failure (%s/%s): %s",
            label,
            attempt,
            max_attempts,
            last_error,
        )
        time.sleep(delay)

    assert last_error is not None
    raise last_error


class HuggingFaceDownloader:
    """Wraps the supported `hf download` CLI for model downloads."""

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        local_dir_path = Path(local_dir)
        target_path = local_dir_path / Path(filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        _run_hf_download(
            command_args=[
                repo_id,
                filename,
                "--local-dir",
                str(local_dir_path),
                "--quiet",
                "--max-workers",
                str(_env_int("LTX_HF_MAX_WORKERS", 1)),
            ],
            progress_path=target_path,
            on_progress=on_progress,
            label=f"{repo_id}/{filename}",
        )
        return target_path

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        local_dir_path = Path(local_dir)
        local_dir_path.mkdir(parents=True, exist_ok=True)

        _run_hf_download(
            command_args=[
                repo_id,
                "--local-dir",
                str(local_dir_path),
                "--quiet",
                "--max-workers",
                str(_env_int("LTX_HF_SNAPSHOT_MAX_WORKERS", 2)),
            ],
            progress_path=local_dir_path,
            on_progress=on_progress,
            label=repo_id,
        )
        return local_dir_path
