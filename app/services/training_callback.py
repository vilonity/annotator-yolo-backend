"""HTTP callbacks from training worker → Bun API."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:
    import requests
except ImportError:  # pragma: no cover - requests is in requirements but guard for tooling
    requests = None  # type: ignore[assignment]


CALLBACK_HEADER = "X-Annotator-Internal-Token"
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 5.0


def load_callback_settings(config_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Read callback url + secret from a job's config.json. Returns (url, secret)."""
    if not config_path.exists():
        return None, None
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None, None
    return data.get("external_callback_url"), data.get("external_callback_secret")


def _backoff_seconds(attempt: int) -> float:
    return min(2 ** attempt, 8.0)


def post_progress(
    base_url: Optional[str],
    secret: Optional[str],
    payload: dict[str, Any],
    *,
    retries: int = DEFAULT_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """POST /progress to Bun. Returns True on 2xx, False otherwise."""
    if not base_url or not secret or requests is None:
        return False

    url = f"{base_url.rstrip('/')}/progress"
    headers = {CALLBACK_HEADER: secret, "Content-Type": "application/json"}

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if 200 <= response.status_code < 300:
                return True
            print(
                f"[training-callback] progress POST failed: {response.status_code} {response.text[:200]}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[training-callback] progress POST attempt {attempt + 1} error: {exc}", flush=True)
        time.sleep(_backoff_seconds(attempt))

    if last_error:
        print(f"[training-callback] progress POST gave up after retries: {last_error}", flush=True)
    return False


def post_progress_async(
    base_url: Optional[str],
    secret: Optional[str],
    payload: dict[str, Any],
) -> None:
    """Fire-and-forget post_progress on a background thread.

    Why: post_progress uses blocking `requests.post` with retries/backoff.
    Calling it from an async FastAPI handler (notably cancel_job) stalled the
    event loop for minutes when the JS backend was slow or briefly
    unreachable, leaving the UI stuck at "cancellation requested".
    """
    if not base_url or not secret or requests is None:
        return
    threading.Thread(
        target=post_progress,
        args=(base_url, secret, payload),
        daemon=True,
    ).start()


def post_log_chunk(
    base_url: Optional[str],
    secret: Optional[str],
    chunk: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """POST /logs to Bun (best-effort, no retry)."""
    if not base_url or not secret or not chunk or requests is None:
        return False
    url = f"{base_url.rstrip('/')}/logs"
    headers = {CALLBACK_HEADER: secret, "Content-Type": "text/plain; charset=utf-8"}
    try:
        response = requests.post(url, data=chunk.encode("utf-8"), headers=headers, timeout=timeout)
        return 200 <= response.status_code < 300
    except Exception as exc:  # noqa: BLE001
        print(f"[training-callback] log POST error: {exc}", flush=True)
        return False


def put_artifact(
    base_url: Optional[str],
    secret: Optional[str],
    filename: str,
    file_path: Path,
    *,
    retries: int = DEFAULT_RETRIES,
    timeout: float = 600.0,
) -> bool:
    """PUT /artifacts/{filename} to Bun with raw file bytes."""
    if not base_url or not secret or requests is None:
        return False
    if not file_path.exists():
        print(f"[training-callback] artifact missing: {file_path}", flush=True)
        return False

    url = f"{base_url.rstrip('/')}/artifacts/{filename}"
    headers = {CALLBACK_HEADER: secret, "Content-Type": "application/octet-stream"}

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with file_path.open("rb") as fh:
                response = requests.put(url, data=fh, headers=headers, timeout=timeout)
            if 200 <= response.status_code < 300:
                return True
            print(
                f"[training-callback] artifact PUT {filename} failed: {response.status_code} {response.text[:200]}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[training-callback] artifact PUT {filename} attempt {attempt + 1} error: {exc}", flush=True)
        time.sleep(_backoff_seconds(attempt))

    if last_error:
        print(f"[training-callback] artifact PUT {filename} gave up: {last_error}", flush=True)
    return False
