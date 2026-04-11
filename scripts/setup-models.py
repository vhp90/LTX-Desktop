#!/usr/bin/env python3
"""Desktop-native model setup helper.

Downloads the LTX Desktop runtime models into the same layout the backend
expects. This replaces the old ComfyUI-oriented downloader with a config-driven
desktop setup flow.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse, unquote, parse_qsl, urlencode, urlunparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime_config.model_download_specs import (  # noqa: E402
    MODEL_FILE_ORDER,
    MODEL_SETUP_CONFIG_PATH,
    load_model_setup_config,
    resolve_downloading_dir,
    resolve_downloading_path,
    resolve_model_path,
)
from services.model_downloader.hugging_face_downloader import HuggingFaceDownloader  # noqa: E402
from state.app_state_types import ModelFileType  # noqa: E402


def _get_civitai_token() -> str:
    for env_name in ("CIVITAI_TOKEN", "CIVITAI_API_TOKEN", "LIGHTNING_CIVITAI_TOKEN"):
        token = os.environ.get(env_name, "").strip()
        if token:
            return token
    return ""


def _read_yaml(path: Path) -> dict[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return payload


def _parse_model_types(raw_values: list[str]) -> list[ModelFileType]:
    valid = set(MODEL_FILE_ORDER)
    parsed: list[ModelFileType] = []
    for value in raw_values:
        model_type = value.strip()
        if model_type not in valid:
            raise ValueError(f"Unknown model type: {value}")
        parsed.append(model_type)  # type: ignore[arg-type]
    return parsed


def _path_ready(path: Path, is_folder: bool) -> bool:
    if is_folder:
        return path.exists() and any(path.iterdir()) if path.exists() else False
    return path.exists()


def _download_external_loras(config_path: Path, target_dir: Path) -> list[str]:
    payload = _read_yaml(config_path)
    raw_items = payload.get("external_loras", [])
    if not isinstance(raw_items, list):
        raise ValueError(f"external_loras must be a list in {config_path}")
    return _download_url_items(raw_items, target_dir, label="external LoRA")


def _configure_hf_download_env() -> None:
    raw_use_xet = os.environ.get("LTX_HF_USE_XET", "").strip().lower()
    if raw_use_xet in {"1", "true", "yes", "on"}:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        return

    if raw_use_xet in {"0", "false", "no", "off"}:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        return

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _make_cli_progress_callback(label: str) -> Callable[[int], None]:
    started_at = time.monotonic()
    last_report_at = started_at
    last_reported_bytes = -1

    def on_progress(downloaded: int) -> None:
        nonlocal last_report_at, last_reported_bytes
        now = time.monotonic()
        if downloaded == last_reported_bytes:
            return
        if last_reported_bytes >= 0 and now - last_report_at < 5.0:
            return
        elapsed = max(now - started_at, 0.001)
        speed = downloaded / elapsed
        print(f"progress {label}: {_format_bytes(downloaded)} downloaded at {_format_bytes(int(speed))}/s")
        last_report_at = now
        last_reported_bytes = downloaded

    return on_progress


def _detect_source(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if "huggingface.co" in host:
        return "huggingface"
    if "civitai.com" in host:
        return "civitai"
    return "direct"


def _parse_hf_url(url: str) -> tuple[str, str, str]:
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) < 5:
        raise ValueError(f"Cannot parse Hugging Face URL: {url}")
    repo_id = f"{parts[0]}/{parts[1]}"
    filepath = "/".join(parts[4:])
    filename = parts[-1]
    return repo_id, filepath, filename


def _download_hf_url(url: str, target_dir: Path, downloader: HuggingFaceDownloader) -> Path:
    _configure_hf_download_env()
    repo_id, filepath, filename = _parse_hf_url(url)
    progress_cb = _make_cli_progress_callback(filename)
    downloaded = downloader.download_file(
        repo_id=repo_id,
        filename=filepath,
        local_dir=str(target_dir),
        on_progress=progress_cb,
    )
    destination = target_dir / filename
    if downloaded.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        downloaded.replace(destination)
        parent = downloaded.parent
        while parent != target_dir and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return destination


def _download_http_url(url: str, target_dir: Path, *, on_progress: Callable[[int], None] | None = None) -> Path:
    civitai_token = _get_civitai_token()
    parsed = urlparse(url)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if "civitai.com" in (parsed.netloc or "").lower() and civitai_token and not any(k == "token" for k, _ in query_pairs):
        query_pairs.append(("token", civitai_token))
        url = urlunparse(parsed._replace(query=urlencode(query_pairs)))
        parsed = urlparse(url)

    filename = unquote(Path(parsed.path).name) or "downloaded_asset"
    destination = target_dir / filename
    request = Request(url, headers={"User-Agent": "LTX-Desktop-Setup/1.0"})
    downloaded = 0
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:  # noqa: S310
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if on_progress is not None:
                on_progress(downloaded)
    return destination


def _download_url_items(raw_items: list[object], target_dir: Path, *, label: str) -> list[str]:
    downloader = HuggingFaceDownloader()
    target_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for raw_item in raw_items:
        if isinstance(raw_item, str):
            url = raw_item
        elif isinstance(raw_item, dict) and isinstance(raw_item.get("url"), str):
            url = raw_item["url"]
        else:
            raise ValueError(f"Invalid {label} entry: {raw_item!r}")

        filename = unquote(Path(urlparse(url).path).name) or "downloaded_asset"
        destination = target_dir / filename
        if destination.exists():
            print(f"skip {label}: {destination.name}")
            continue

        print(f"download {label}: {destination.name}")
        progress_cb = _make_cli_progress_callback(destination.name)
        source = _detect_source(url)
        try:
            if source == "huggingface":
                _download_hf_url(url, target_dir, downloader)
            else:
                _download_http_url(url, target_dir, on_progress=progress_cb)
            print(f"done {label}: {destination.name}")
        except HTTPError as exc:
            failure = f"{label} {url} failed with HTTP {exc.code}"
            if exc.code == 401 and source == "civitai":
                failure += " (set CIVITAI_TOKEN, CIVITAI_API_TOKEN, or LIGHTNING_CIVITAI_TOKEN to download protected CivitAI assets)"
            print(f"error {failure}")
            failures.append(failure)
        except Exception as exc:
            failure = f"{label} {url} failed: {exc}"
            print(f"error {failure}")
            failures.append(failure)

    return failures


def _download_external_assets(config_path: Path, target_root: Path) -> list[str]:
    payload = _read_yaml(config_path)
    raw_assets = payload.get("external_assets", {})
    if not isinstance(raw_assets, dict):
        raise ValueError(f"external_assets must be a mapping in {config_path}")
    failures: list[str] = []

    for folder_name, raw_items in raw_assets.items():
        if not isinstance(folder_name, str):
            raise ValueError(f"Invalid external_assets key in {config_path}: {folder_name!r}")
        if not isinstance(raw_items, list):
            raise ValueError(f"external_assets.{folder_name} must be a list in {config_path}")
        failures.extend(_download_url_items(raw_items, target_root / folder_name, label=f"external asset [{folder_name}]"))

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Download desktop runtime models from config/model_setup.yaml")
    parser.add_argument(
        "--config",
        type=Path,
        default=MODEL_SETUP_CONFIG_PATH,
        help=f"Model setup config path (default: {MODEL_SETUP_CONFIG_PATH})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / ".ltx-data" / "models",
        help="Target models directory (default: ./.ltx-data/models)",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Download all configured desktop model types, not just required ones.",
    )
    parser.add_argument(
        "--model-type",
        action="append",
        default=[],
        help="Specific model type to download. Repeatable.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print resolved model specs and exit.",
    )
    parser.add_argument(
        "--include-external-loras",
        action="store_true",
        help="Download external LoRAs preserved from the prior workflow config.",
    )
    parser.add_argument(
        "--include-external-assets",
        action="store_true",
        help="Download auxiliary assets preserved from the prior Comfy workflow config.",
    )
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    args = parser.parse_args(argv)

    specs, required = load_model_setup_config(args.config)
    models_dir = args.models_dir.resolve()
    downloader = HuggingFaceDownloader()

    if args.model_type:
        selected_types = _parse_model_types(args.model_type)
    elif args.include_optional:
        selected_types = list(MODEL_FILE_ORDER)
    else:
        selected_types = [model_type for model_type in MODEL_FILE_ORDER if model_type in required]

    if args.list:
        print(f"Config: {args.config}")
        print(f"Models dir: {models_dir}")
        for model_type in selected_types:
            spec = specs[model_type]
            final_path = resolve_model_path(models_dir, specs, model_type)
            state = "ready" if _path_ready(final_path, spec.is_folder) else "missing"
            print(f"{model_type}: {spec.repo_id} -> {spec.relative_path} [{state}]")
        if args.include_external_loras:
            print(f"external_loras: {ROOT / '.ltx-data' / 'loras' / 'external'}")
        if args.include_external_assets:
            print(f"external_assets: {ROOT / '.ltx-data' / 'comfy-imports'}")
        return 0

    print(f"Using config: {args.config}")
    print(f"Target models dir: {models_dir}")
    print(f"Selected model types: {', '.join(selected_types)}")
    _configure_hf_download_env()

    for model_type in selected_types:
        spec = specs[model_type]
        final_path = resolve_model_path(models_dir, specs, model_type)
        if _path_ready(final_path, spec.is_folder):
            print(f"skip {model_type}: {final_path}")
            continue

        print(f"download {model_type}: {spec.repo_id} -> {spec.relative_path}")
        resolve_downloading_dir(models_dir).mkdir(parents=True, exist_ok=True)
        progress_cb = _make_cli_progress_callback(model_type)
        if spec.is_folder:
            local_dir = resolve_downloading_path(models_dir, specs, model_type)
            downloader.download_snapshot(
                repo_id=spec.repo_id,
                local_dir=str(local_dir),
                on_progress=progress_cb,
            )
            if final_path.exists():
                if final_path.is_dir():
                    shutil.rmtree(final_path)
                else:
                    final_path.unlink()
            local_dir.rename(final_path)
        else:
            local_dir = resolve_downloading_path(models_dir, specs, model_type)
            downloaded = downloader.download_file(
                repo_id=spec.repo_id,
                filename=spec.name,
                local_dir=str(local_dir),
                on_progress=progress_cb,
            )
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                final_path.unlink()
            downloaded.rename(final_path)
        print(f"done {model_type}: {final_path}")

    failures: list[str] = []
    if args.include_external_loras:
        failures.extend(_download_external_loras(args.config, ROOT / ".ltx-data" / "loras" / "external"))
    if args.include_external_assets:
        failures.extend(_download_external_assets(args.config, ROOT / ".ltx-data" / "comfy-imports"))

    if failures:
        print("Model setup completed with some download failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Model setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
