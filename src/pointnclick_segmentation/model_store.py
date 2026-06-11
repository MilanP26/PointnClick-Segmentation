from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


DEFAULT_MODEL_FILENAME = "pointnclick_model.pt"


def default_app_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / "PointnClick"
    return Path.home() / ".pointnclick"


def default_model_dir() -> Path:
    return default_app_dir() / "models"


def default_config_path() -> Path:
    return default_app_dir() / "bridge_config.json"


def filename_from_url(url: str, fallback: str = DEFAULT_MODEL_FILENAME) -> str:
    parsed = urlparse(url)
    name = unquote(Path(parsed.path).name)
    if not name or name in {".", "/"}:
        return fallback
    return name


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    url: str,
    destination: str | Path | None = None,
    expected_sha256: str | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> Path:
    if not url.strip():
        raise ValueError("Model URL is required")

    if destination is None:
        destination = default_model_dir() / filename_from_url(url)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".download")

    request = Request(url, headers={"User-Agent": "PointnClickBridge/0.1"})
    with urlopen(request, timeout=120) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        received = 0
        with temp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                if progress_callback is not None:
                    progress_callback(received, total)

    if expected_sha256:
        actual = sha256_file(temp_path)
        if actual.lower() != expected_sha256.strip().lower():
            temp_path.unlink(missing_ok=True)
            raise ValueError(
                "Downloaded model SHA256 did not match. "
                f"Expected {expected_sha256}, got {actual}."
            )

    temp_path.replace(destination)
    return destination
