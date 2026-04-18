"""RunPod-backed training provider (S3 volume flavor).

The backend-yolo process does NOT need to be publicly reachable. All data
exchange with the remote GPU worker goes through the RunPod NetworkVolume,
which the worker mounts at /runpod-volume and which backend-yolo accesses
via RunPod's S3-compatible API (the volume id is the bucket name).

Layout on volume:

  jobs/{job_id}/config.json        — job config (written by backend-yolo)
  jobs/{job_id}/dataset_hash.txt   — pointer to cached dataset
  jobs/{job_id}/logs.txt           — streamed stdout/stderr (append-only)
  jobs/{job_id}/progress.jsonl     — one JSON line per finished epoch
  jobs/{job_id}/status.json        — final {status, error?} marker
  jobs/{job_id}/weights.pt         — best weights (uploaded by worker)
  jobs/{job_id}/artifacts/*.png    — result plots
  datasets/{sha256}.zip            — cached dataset, reused across jobs
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import threading
import time
import zipfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException

from app.config import (
    RUNPOD_API_KEY,
    RUNPOD_ENABLED,
    RUNPOD_NETWORK_VOLUME_ID,
    RUNPOD_NETWORK_VOLUME_NAME,
    RUNPOD_S3_ACCESS_KEY_ID,
    RUNPOD_S3_ENDPOINT_URL,
    RUNPOD_S3_SECRET_ACCESS_KEY,
    RUNPOD_VOLUME_SIZE_GB,
    TRAIN_JOBS_DIR,
    YOLO_MODELS_DIR,
)

logger = logging.getLogger(__name__)

_BRIDGE_POLL_INTERVAL_SECONDS = 1.0
_PROGRESS_TAIL_BYTES = 16 * 1024  # enough for a progress.jsonl tail
_ARTIFACT_FILES = ("results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png", "results.csv")

from app.services import runpod_catalog, training_callback

_cloud_jobs: dict[str, "_CloudJob"] = {}
_cloud_jobs_lock = threading.Lock()


class _CloudJob:
    __slots__ = ("job_id", "endpoint_job", "watcher_task", "bridge_task", "cancelled", "dataset_hash")

    def __init__(self, *, job_id: str, endpoint_job: "_EndpointJobHandle", dataset_hash: str) -> None:
        self.job_id = job_id
        self.endpoint_job = endpoint_job
        self.watcher_task: Optional[asyncio.Task[Any]] = None
        self.bridge_task: Optional[asyncio.Task[Any]] = None
        self.cancelled = False
        self.dataset_hash = dataset_hash


class _EndpointJobHandle:
    """Adapter around runpod-flash's decorated-function call.

    Why: runpod-flash's `@Endpoint(...)` decorator returns a plain async
    callable that awaits the remote job to completion internally. The rest
    of this module was written expecting a job-handle API (`.run()`,
    `.wait()`, `.cancel()`, `.id`, `.error`). This adapter bridges the two
    by wrapping the call in an asyncio.Task that external code can await or
    cancel.
    """

    __slots__ = ("_task", "id", "error", "output")

    def __init__(self, coro: Any) -> None:
        self._task = asyncio.create_task(coro)
        self.id: Optional[str] = None
        self.error: Optional[str] = None
        self.output: Any = None

    async def wait(self) -> None:
        try:
            self.output = await asyncio.shield(self._task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)

    async def cancel(self) -> None:
        if not self._task.done():
            self._task.cancel()


def diagnose_runpod_availability() -> list[str]:
    """Return list of missing pieces — empty list means ready to go."""
    missing: list[str] = []
    if not RUNPOD_ENABLED:
        missing.append("env RUNPOD_ENABLED is not true")
    if not RUNPOD_API_KEY:
        missing.append("env RUNPOD_API_KEY is empty")
    if not RUNPOD_S3_ENDPOINT_URL:
        missing.append("env RUNPOD_S3_ENDPOINT_URL is empty")
    if not RUNPOD_S3_ACCESS_KEY_ID:
        missing.append("env RUNPOD_S3_ACCESS_KEY_ID is empty")
    if not RUNPOD_S3_SECRET_ACCESS_KEY:
        missing.append("env RUNPOD_S3_SECRET_ACCESS_KEY is empty")
    if not RUNPOD_NETWORK_VOLUME_ID:
        missing.append("env RUNPOD_NETWORK_VOLUME_ID is empty")
    try:
        import runpod_flash  # noqa: F401
    except ImportError:
        missing.append("python package 'runpod-flash' is not installed")
    try:
        import boto3  # noqa: F401
    except ImportError:
        missing.append("python package 'boto3' is not installed")
    return missing


def is_runpod_available() -> bool:
    return not diagnose_runpod_availability()


def log_runpod_readiness() -> None:
    missing = diagnose_runpod_availability()
    if not missing:
        logger.info("[runpod] ready — cloud training enabled")
        return
    logger.warning("[runpod] cloud training DISABLED — missing: %s", "; ".join(missing))


def list_runpod_devices() -> list[dict[str, Any]]:
    if not is_runpod_available():
        return []
    return [{**entry, "provider": "runpod"} for entry in runpod_catalog.build_runpod_device_options()]


def is_runpod_device(device: Optional[str]) -> bool:
    return bool(device) and device.startswith("runpod:")


def resolve_runpod_device_name(device: Optional[str]) -> Optional[str]:
    if not device:
        return None
    for entry in runpod_catalog.build_runpod_device_options():
        if entry["id"] == device:
            return entry["name"]
    return None


def _resolve_gpu_groups(device: Optional[str]) -> list[Any]:
    """Map our synthetic device ids into runpod-flash GpuGroup enums."""
    from runpod_flash import GpuGroup

    mapping: dict[str, list[Any]] = {
        "runpod:ada_24": [GpuGroup.ADA_24],
        "runpod:ampere_80": [GpuGroup.AMPERE_80],
        "runpod:auto": [GpuGroup.ADA_24, GpuGroup.AMPERE_80],
    }
    return mapping.get(device or "runpod:auto", [GpuGroup.ADA_24, GpuGroup.AMPERE_80])


# ──────────────────────────────────────────────────────────────────────────
# S3 volume client — backend-yolo side only.
# ──────────────────────────────────────────────────────────────────────────


def _region_from_endpoint(endpoint_url: Optional[str]) -> str:
    """Extract the datacenter slug from a RunPod S3 endpoint URL.

    Why: boto3 requires a region_name for SigV4 signing. RunPod's S3 proxy
    validates that the signed region matches the datacenter in the hostname
    (e.g. `s3api-us-il-1.runpod.io` → region `us-il-1`). A mismatch makes the
    proxy answer 500 Internal Server Error, which looks like a server bug.
    """
    if not endpoint_url:
        return "us-east-1"
    try:
        from urllib.parse import urlparse

        host = urlparse(endpoint_url).hostname or ""
    except Exception:
        return "us-east-1"
    prefix = "s3api-"
    if host.startswith(prefix):
        tail = host[len(prefix) :]
        slug = tail.split(".", 1)[0]
        if slug:
            return slug
    return "us-east-1"


def _is_missing_object_error(exc: Any) -> bool:
    """Recognize the various codes RunPod's S3 returns for a missing object.

    Standard S3 uses `NoSuchKey`/`404`/`NotFound`. RunPod's S3 proxy instead
    returns `InvalidArgument` with the message `object not found` for GET on a
    missing key — we must treat that as a miss, not a hard error, or the
    bridge poll log-spams every 1.5s while waiting for the worker to write
    status.json.
    """
    err = getattr(exc, "response", {}).get("Error", {}) if hasattr(exc, "response") else {}
    code = err.get("Code") or ""
    message = (err.get("Message") or "").lower()
    if code in {"404", "NoSuchKey", "NotFound"}:
        return True
    if code == "InvalidArgument" and "not found" in message:
        return True
    return False


class _VolumeClient:
    """Thin wrapper over boto3 for RunPod S3-compatible NetworkVolume access."""

    def __init__(self) -> None:
        import boto3
        from botocore.config import Config as BotoConfig

        self._bucket = RUNPOD_NETWORK_VOLUME_ID
        self._client = boto3.client(
            "s3",
            endpoint_url=RUNPOD_S3_ENDPOINT_URL,
            aws_access_key_id=RUNPOD_S3_ACCESS_KEY_ID,
            aws_secret_access_key=RUNPOD_S3_SECRET_ACCESS_KEY,
            region_name=_region_from_endpoint(RUNPOD_S3_ENDPOINT_URL),
            config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
        )

    def head(self, key: str) -> Optional[dict[str, Any]]:
        from botocore.exceptions import ClientError

        try:
            return self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _is_missing_object_error(exc):
                return None
            code = exc.response.get("Error", {}).get("Code")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            message = exc.response.get("Error", {}).get("Message") or str(exc)
            logger.error(
                "[runpod] head_object bucket=%s key=%s failed http=%s code=%s message=%s",
                self._bucket,
                key,
                status,
                code,
                message,
            )
            raise

    def exists(self, key: str) -> bool:
        return self.head(key) is not None

    def get_size(self, key: str) -> Optional[int]:
        meta = self.head(key)
        if meta is None:
            return None
        return int(meta.get("ContentLength", 0))

    def get_bytes(self, key: str, *, offset: int = 0, end: Optional[int] = None) -> bytes:
        """Range GET. Returns b"" if offset is past the end."""
        from botocore.exceptions import ClientError

        range_header = None
        if offset or end is not None:
            tail = "" if end is None else str(end)
            range_header = f"bytes={offset}-{tail}"
        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if range_header:
            kwargs["Range"] = range_header
        try:
            response = self._client.get_object(**kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"InvalidRange", "416"}:
                return b""
            if _is_missing_object_error(exc):
                return b""
            raise
        return response["Body"].read()

    def put_file(self, key: str, path: Path) -> None:
        self._client.upload_file(str(path), self._bucket, key)

    def put_bytes(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)

    def download(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._bucket, key, str(path))

    def delete_prefix(self, prefix: str) -> None:
        # RunPod's S3 proxy returns 307 Temporary Redirect for the bulk
        # DeleteObjects call (which is a POST /?delete), and boto3 does not
        # follow 307 redirects on DeleteObjects. Falling back to per-object
        # delete_object (singular) sidesteps the problem — those go through
        # as normal DELETE /<key> requests that the proxy handles.
        from botocore.exceptions import ClientError

        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        if not keys:
            return
        try:
            for chunk_start in range(0, len(keys), 1000):
                chunk = keys[chunk_start : chunk_start + 1000]
                self._client.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": [{"Key": k} for k in chunk]},
                )
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = exc.response.get("Error", {}).get("Code") or ""
            if status != 307 and code != "TemporaryRedirect":
                raise
            for key in keys:
                try:
                    self._client.delete_object(Bucket=self._bucket, Key=key)
                except ClientError:
                    logger.exception("delete_object fallback failed for key=%s", key)


_volume_client_singleton: Optional[_VolumeClient] = None


def _volume() -> _VolumeClient:
    global _volume_client_singleton
    if _volume_client_singleton is None:
        _volume_client_singleton = _VolumeClient()
    return _volume_client_singleton


# ──────────────────────────────────────────────────────────────────────────
# File helpers (backend-yolo local state).
# ──────────────────────────────────────────────────────────────────────────


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_metadata(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def _write_metadata(job_dir: Path, metadata: dict[str, Any]) -> None:
    with (job_dir / "metadata.json").open("w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=True, indent=2)


def _zip_dataset_dir(dataset_dir: Path, target_zip: Path) -> None:
    if target_zip.exists():
        target_zip.unlink()
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in dataset_dir.rglob("*"):
            if item.is_dir():
                continue
            archive.write(item, item.relative_to(dataset_dir).as_posix())


# ──────────────────────────────────────────────────────────────────────────
# Public API — called from training_service.
# ──────────────────────────────────────────────────────────────────────────


async def start_cloud_job(job_id: str, job_dir: Path, config: dict[str, Any]) -> None:
    """Upload dataset + config to volume, invoke remote training, start bridge."""
    if not is_runpod_available():
        raise RuntimeError(
            "RunPod is not available — set RUNPOD_ENABLED=true, RUNPOD_API_KEY, "
            "RUNPOD_S3_* and RUNPOD_NETWORK_VOLUME_ID, and install runpod-flash + boto3."
        )

    dataset_zip = job_dir / "dataset.zip"
    if not dataset_zip.exists():
        _zip_dataset_dir(job_dir / "dataset", dataset_zip)

    loop = asyncio.get_running_loop()
    dataset_hash = await loop.run_in_executor(None, _sha256_file, dataset_zip)

    # Upload dataset to cached path (dedup by hash) so repeat trainings reuse it.
    def _stage_volume() -> None:
        vol = _volume()
        cache_key = f"datasets/{dataset_hash}.zip"
        if not vol.exists(cache_key):
            vol.put_file(cache_key, dataset_zip)
        vol.put_bytes(f"jobs/{job_id}/dataset_hash.txt", dataset_hash.encode("utf-8"), content_type="text/plain")
        vol.put_bytes(
            f"jobs/{job_id}/config.json",
            json.dumps(config, ensure_ascii=True).encode("utf-8"),
            content_type="application/json",
        )
        # Ensure previous artifacts from a re-used job id are cleaned.
        vol.delete_prefix(f"jobs/{job_id}/logs.txt")
        vol.delete_prefix(f"jobs/{job_id}/progress.jsonl")
        vol.delete_prefix(f"jobs/{job_id}/status.json")
        vol.delete_prefix(f"jobs/{job_id}/weights.pt")
        vol.delete_prefix(f"jobs/{job_id}/artifacts/")

    try:
        await loop.run_in_executor(None, _stage_volume)
    except Exception as exc:
        from botocore.exceptions import BotoCoreError, ClientError

        if isinstance(exc, ClientError):
            err = exc.response.get("Error", {})
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = err.get("Code") or "UnknownCode"
            message = err.get("Message") or str(exc)
            detail = (
                f"RunPod S3 upload failed ({code}, HTTP {status}): {message}. "
                f"Endpoint={RUNPOD_S3_ENDPOINT_URL} volume={RUNPOD_NETWORK_VOLUME_ID}. "
                "Verify that RUNPOD_S3_ENDPOINT_URL points to the datacenter that actually hosts "
                "the network volume, and that the S3 access keys are valid."
            )
            logger.exception("[runpod] staging failed: %s", detail)
            raise HTTPException(status_code=502, detail=detail) from exc
        if isinstance(exc, BotoCoreError):
            logger.exception("[runpod] staging failed (botocore)")
            raise HTTPException(status_code=502, detail=f"RunPod S3 client error: {exc}") from exc
        logger.exception("[runpod] staging failed")
        raise

    endpoint_call = _build_endpoint(config.get("device"))

    metadata = _read_metadata(job_dir)
    metadata["provider"] = "runpod"
    metadata["runpod_endpoint_id"] = "annotator-yolo-train"
    metadata["dataset_hash"] = dataset_hash
    _write_metadata(job_dir, metadata)

    endpoint_job = _EndpointJobHandle(endpoint_call({"job_id": job_id}))

    cloud_job = _CloudJob(job_id=job_id, endpoint_job=endpoint_job, dataset_hash=dataset_hash)
    with _cloud_jobs_lock:
        _cloud_jobs[job_id] = cloud_job

    metadata = _read_metadata(job_dir)
    metadata["runpod_job_id"] = None
    _write_metadata(job_dir, metadata)

    callback_url, callback_secret = training_callback.load_callback_settings(job_dir / "config.json")
    training_callback.post_progress_async(
        callback_url,
        callback_secret,
        {
            "status": "running",
            "started_at": _now_iso(),
            "device_name": metadata.get("device_name"),
            "runpod_endpoint_id": metadata.get("runpod_endpoint_id"),
            "runpod_job_id": metadata.get("runpod_job_id"),
        },
    )

    cloud_job.bridge_task = asyncio.create_task(_bridge_task(job_id, job_dir))
    cloud_job.watcher_task = asyncio.create_task(_watch_cloud_job(job_id, job_dir, endpoint_job))


async def cancel_cloud_job(job_id: str) -> None:
    with _cloud_jobs_lock:
        cloud_job = _cloud_jobs.get(job_id)
    if cloud_job is None:
        return
    cloud_job.cancelled = True
    try:
        await cloud_job.endpoint_job.cancel()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to cancel RunPod endpoint job %s", job_id)
    for task in (cloud_job.watcher_task, cloud_job.bridge_task):
        if task and not task.done():
            task.cancel()


async def _watch_cloud_job(job_id: str, job_dir: Path, endpoint_job: Any) -> None:
    """Wait for the Flash endpoint job to end, then backstop finalization."""
    try:
        await endpoint_job.wait()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("RunPod job %s watcher crashed", job_id)
        _mark_failed(job_dir, f"RunPod watcher crashed: {exc}")
        _cancel_bridge_task(job_id)
        return

    # Give the bridge one final cycle to pull the last logs + status.
    await asyncio.sleep(_BRIDGE_POLL_INTERVAL_SECONDS * 2)

    metadata = _read_metadata(job_dir)
    if metadata.get("status") in {"completed", "failed", "cancelled"}:
        return

    error = getattr(endpoint_job, "error", None)
    if error:
        _mark_failed(job_dir, str(error))
        _cancel_bridge_task(job_id)
        return

    _mark_failed(job_dir, "Remote worker finished without uploading a status marker")
    _cancel_bridge_task(job_id)


def _cancel_bridge_task(job_id: str) -> None:
    """Stop the bridge poll loop when the watcher backstops a failed job.

    The bridge only exits on its own when it sees a terminal status.json on
    S3. If the worker crashes without uploading one, the watcher's _mark_failed
    won't make the bridge stop — it would poll S3 forever.
    """
    with _cloud_jobs_lock:
        cloud_job = _cloud_jobs.get(job_id)
    if cloud_job and cloud_job.bridge_task and not cloud_job.bridge_task.done():
        cloud_job.bridge_task.cancel()


async def _bridge_task(job_id: str, job_dir: Path) -> None:
    """Poll the NetworkVolume via S3 and mirror worker output into the job dir."""
    loop = asyncio.get_running_loop()

    log_offset = 0
    last_progress_mtime: Optional[Any] = None
    last_resources_mtime: Optional[Any] = None
    finalized = False

    callback_url, callback_secret = training_callback.load_callback_settings(job_dir / "config.json")

    logs_path = job_dir / "logs.txt"
    if not logs_path.exists():
        logs_path.touch()

    while not finalized:
        try:
            await asyncio.sleep(_BRIDGE_POLL_INTERVAL_SECONDS)

            def _poll() -> dict[str, Any]:
                vol = _volume()
                result: dict[str, Any] = {}

                # Logs: incremental range fetch.
                size = vol.get_size(f"jobs/{job_id}/logs.txt")
                if size is not None and size > log_offset:
                    chunk = vol.get_bytes(f"jobs/{job_id}/logs.txt", offset=log_offset)
                    if chunk:
                        result["log_chunk"] = chunk
                        result["new_log_offset"] = log_offset + len(chunk)

                # Progress: read tail.
                progress_head = vol.head(f"jobs/{job_id}/progress.jsonl")
                if progress_head is not None:
                    progress_size = int(progress_head.get("ContentLength", 0))
                    progress_mtime = progress_head.get("LastModified")
                    if progress_size > 0 and progress_mtime != last_progress_mtime:
                        tail_offset = max(0, progress_size - _PROGRESS_TAIL_BYTES)
                        tail = vol.get_bytes(f"jobs/{job_id}/progress.jsonl", offset=tail_offset)
                        result["progress_tail"] = tail
                        result["progress_mtime"] = progress_mtime

                # Resources: read tail (last JSON line per snapshot).
                resources_head = vol.head(f"jobs/{job_id}/resources.jsonl")
                if resources_head is not None:
                    resources_size = int(resources_head.get("ContentLength", 0))
                    resources_mtime = resources_head.get("LastModified")
                    if resources_size > 0 and resources_mtime != last_resources_mtime:
                        tail_offset = max(0, resources_size - _PROGRESS_TAIL_BYTES)
                        tail = vol.get_bytes(f"jobs/{job_id}/resources.jsonl", offset=tail_offset)
                        result["resources_tail"] = tail
                        result["resources_mtime"] = resources_mtime

                # Status marker.
                status_bytes = vol.get_bytes(f"jobs/{job_id}/status.json")
                if status_bytes:
                    try:
                        result["status"] = json.loads(status_bytes.decode("utf-8"))
                    except json.JSONDecodeError:
                        pass

                return result

            snapshot = await loop.run_in_executor(None, _poll)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("RunPod bridge poll failed for job %s", job_id)
            continue

        log_chunk = snapshot.get("log_chunk")
        if log_chunk:
            with logs_path.open("ab") as logs_file:
                logs_file.write(log_chunk)
            log_offset = snapshot["new_log_offset"]
            if callback_url and callback_secret:
                try:
                    text = log_chunk.decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    text = ""
                if text:
                    await loop.run_in_executor(
                        None,
                        training_callback.post_log_chunk,
                        callback_url,
                        callback_secret,
                        text,
                    )

        progress_tail = snapshot.get("progress_tail")
        if progress_tail is not None:
            last_progress_mtime = snapshot.get("progress_mtime")
            progress_payload = _apply_progress_tail(job_dir, progress_tail)
            if progress_payload and callback_url and callback_secret:
                training_callback.post_progress_async(
                    callback_url,
                    callback_secret,
                    {"status": "running", **progress_payload},
                )

        resources_tail = snapshot.get("resources_tail")
        if resources_tail is not None:
            last_resources_mtime = snapshot.get("resources_mtime")
            resources_sample = _parse_last_jsonl(resources_tail)
            if resources_sample and callback_url and callback_secret:
                training_callback.post_progress_async(
                    callback_url,
                    callback_secret,
                    {"status": "running", "resources": resources_sample},
                )

        status = snapshot.get("status")
        # The worker writes status.json TWICE: once with "running" right before
        # model.train() (line 1185), and again with "completed"/"failed" after
        # training. Only the latter is a terminal marker — finalizing on
        # "running" would race the worker's training, time-out polling weights,
        # and clean up S3 before the worker has a chance to upload logs or
        # progress, leaving local logs.txt empty and metrics null.
        if status is not None and status.get("status") in {"completed", "failed", "cancelled"}:
            try:
                await loop.run_in_executor(None, _finalize_from_volume, job_id, job_dir, status)
            except Exception as exc:  # noqa: BLE001
                logger.exception("RunPod bridge finalize failed for job %s", job_id)
                _mark_failed(job_dir, f"Finalize failed: {exc}")
            finalized = True


def _parse_last_jsonl(tail: bytes) -> Optional[dict[str, Any]]:
    try:
        text = tail.decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return None
    last_line: Optional[dict[str, Any]] = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            last_line = parsed
    return last_line


def _apply_progress_tail(job_dir: Path, tail: bytes) -> Optional[dict[str, Any]]:
    try:
        text = tail.decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return None
    last_line: Optional[dict[str, Any]] = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last_line = json.loads(line)
        except json.JSONDecodeError:
            continue
    if last_line is None:
        return None

    metadata = _read_metadata(job_dir)
    payload: dict[str, Any] = {}
    if isinstance(last_line.get("current_epoch"), int):
        metadata["current_epoch"] = last_line["current_epoch"]
        payload["current_epoch"] = last_line["current_epoch"]
    if isinstance(last_line.get("total_epochs"), int):
        metadata["total_epochs"] = last_line["total_epochs"]
        payload["total_epochs"] = last_line["total_epochs"]
    if isinstance(last_line.get("metrics"), dict):
        metadata["metrics"] = last_line["metrics"]
        payload["metrics"] = last_line["metrics"]
    _write_metadata(job_dir, metadata)
    return payload or None


def _finalize_from_volume(job_id: str, job_dir: Path, status_payload: dict[str, Any]) -> None:
    vol = _volume()
    status = status_payload.get("status")
    callback_url, callback_secret = training_callback.load_callback_settings(job_dir / "config.json")

    if status == "failed":
        error = str(status_payload.get("error") or "Remote training failed")
        _mark_failed(job_dir, error)
        training_callback.post_progress(
            callback_url,
            callback_secret,
            {"status": "failed", "completed_at": _now_iso(), "error": error},
        )
        _cleanup_volume_job(job_id)
        return

    if status == "cancelled":
        metadata = _read_metadata(job_dir)
        if metadata.get("status") not in {"completed", "failed"}:
            metadata["status"] = "cancelled"
            metadata["completed_at"] = _now_iso()
            metadata["error"] = metadata.get("error") or "Training job was cancelled"
            _write_metadata(job_dir, metadata)
        training_callback.post_progress(
            callback_url,
            callback_secret,
            {
                "status": "cancelled",
                "completed_at": metadata.get("completed_at") or _now_iso(),
                "error": metadata.get("error"),
            },
        )
        _cleanup_volume_job(job_id)
        return

    # status == "completed"
    metadata = _read_metadata(job_dir)

    total = metadata.get("total_epochs") or metadata.get("epochs")
    if isinstance(total, int) and total > 0:
        metadata["current_epoch"] = total

    runs_weights_dir = job_dir / "runs" / "train" / "weights"
    runs_weights_dir.mkdir(parents=True, exist_ok=True)
    best_pt = runs_weights_dir / "best.pt"

    # The worker PUTs weights.pt via the S3 API directly before writing
    # status.json, but RunPod's S3 proxy has a visibility lag between PUT and
    # HEAD/LIST — observed in the wild at 20-40s even though the object is
    # already stored. Poll generously (~90s) before giving up, otherwise a
    # perfectly successful training gets marked failed on a visibility race.
    weights_key = f"jobs/{job_id}/weights.pt"
    weights_downloaded = False
    import time as _t

    max_attempts = 45
    for attempt in range(max_attempts):
        if vol.exists(weights_key):
            try:
                vol.download(weights_key, best_pt)
                weights_downloaded = True
                break
            except Exception:  # noqa: BLE001
                logger.exception("RunPod weights download attempt %d failed for %s", attempt + 1, job_id)
        if attempt + 1 < max_attempts:
            _t.sleep(2)
    if not weights_downloaded:
        listing: list[str] = []
        try:
            paginator = _volume()._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=_volume()._bucket, Prefix=f"jobs/{job_id}/"):
                for obj in page.get("Contents", []):
                    listing.append(f"{obj['Key']} ({obj.get('Size', '?')} bytes)")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to list jobs/%s/ prefix after weights timeout", job_id)
        logger.error(
            "RunPod weights.pt missing after %ds for job %s. jobs/%s/ contents: %s",
            max_attempts * 2,
            job_id,
            job_id,
            listing or "<empty>",
        )
        _mark_failed(
            job_dir,
            f"Remote training reported status=completed but weights.pt never appeared on the "
            f"NetworkVolume (via the S3 API) after {max_attempts * 2}s of polling. "
            f"Objects currently listed under jobs/{job_id}/: {listing or '<empty>'}. "
            "If the listing is empty the worker's S3 PUT never succeeded — check worker logs "
            "for '[runpod] S3 upload_file ... failed:' lines. If the listing shows weights.pt "
            "then RunPod's S3 proxy has a visibility lag longer than our poll window.",
        )
        return

    # Artifacts — optional.
    runs_train_dir = job_dir / "runs" / "train"
    runs_train_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name in _ARTIFACT_FILES:
        if vol.exists(f"jobs/{job_id}/artifacts/{artifact_name}"):
            try:
                vol.download(f"jobs/{job_id}/artifacts/{artifact_name}", runs_train_dir / artifact_name)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to download artifact %s for %s", artifact_name, job_id)

    metrics = status_payload.get("metrics") or {}
    with (job_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=True, indent=2)

    model_name = metadata.get("output_model_name") or metadata.get("produced_model_name")
    if model_name:
        model_dir = YOLO_MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, model_dir / "weights.pt")

        classes = metadata.get("classes") or []
        with (model_dir / "classes.json").open("w", encoding="utf-8") as classes_file:
            json.dump(classes, classes_file, ensure_ascii=True, indent=2)

        with (model_dir / "training-metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(
                {
                    "job_id": metadata.get("id"),
                    "project_name": metadata.get("project_name"),
                    "user_id": metadata.get("user_id"),
                    "provider": "runpod",
                    "base_model": metadata.get("base_model"),
                    "output_model_name": model_name,
                    "epochs": metadata.get("epochs"),
                    "imgsz": metadata.get("imgsz"),
                    "batch": metadata.get("batch"),
                    "device": metadata.get("device"),
                    "split": metadata.get("split"),
                    "classes": classes,
                },
                metadata_file,
                ensure_ascii=True,
                indent=2,
            )

    artifacts = {
        "best_weights_path": str(best_pt),
        "results_csv_path": str(runs_train_dir / "results.csv"),
    }
    with (job_dir / "artifacts.json").open("w", encoding="utf-8") as artifacts_file:
        json.dump(artifacts, artifacts_file, ensure_ascii=True, indent=2)

    metadata["status"] = "completed"
    metadata["produced_model_name"] = model_name
    metadata["completed_at"] = _now_iso()
    metadata["metrics"] = metrics or metadata.get("metrics")
    metadata["artifacts"] = artifacts
    metadata["error"] = None
    _write_metadata(job_dir, metadata)

    if callback_url and callback_secret:
        total_epochs = metadata.get("total_epochs") or metadata.get("epochs")
        training_callback.post_progress(
            callback_url,
            callback_secret,
            {
                "status": "completed",
                "completed_at": metadata["completed_at"],
                "produced_model_name": model_name,
                "current_epoch": total_epochs,
                "total_epochs": total_epochs,
                "metrics": metadata["metrics"],
                "error": None,
            },
        )
        training_callback.put_artifact(callback_url, callback_secret, "weights.pt", best_pt)
        results_csv = runs_train_dir / "results.csv"
        if results_csv.exists():
            training_callback.put_artifact(callback_url, callback_secret, "results.csv", results_csv)
        for artifact_name in _ARTIFACT_FILES:
            if artifact_name in {"results.csv"}:
                continue
            artifact_path = runs_train_dir / artifact_name
            if artifact_path.exists():
                training_callback.put_artifact(callback_url, callback_secret, artifact_name, artifact_path)

    _cleanup_volume_job(job_id)


def _cleanup_volume_job(job_id: str) -> None:
    """Remove per-job state on the volume. Cached datasets stay intact."""
    try:
        _volume().delete_prefix(f"jobs/{job_id}/")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to clean up RunPod volume for job %s", job_id)


def _mark_failed(job_dir: Path, error: str) -> None:
    metadata = _read_metadata(job_dir)
    if metadata.get("status") in {"completed", "failed", "cancelled"}:
        return
    metadata["status"] = "failed"
    metadata["error"] = error
    metadata["completed_at"] = _now_iso()
    _write_metadata(job_dir, metadata)
    with (job_dir / "error.json").open("w", encoding="utf-8") as error_file:
        json.dump({"error": error}, error_file, ensure_ascii=True, indent=2)
    callback_url, callback_secret = training_callback.load_callback_settings(job_dir / "config.json")
    training_callback.post_progress_async(
        callback_url,
        callback_secret,
        {"status": "failed", "completed_at": metadata["completed_at"], "error": error},
    )


def _build_endpoint(device: Optional[str]) -> Any:
    """Return a callable that dispatches train_on_runpod to a flash Endpoint.

    runpod-flash's `@Endpoint(...)` decorator returns a plain async callable —
    invoking it submits the payload to the remote worker and awaits the final
    result. The caller wraps this in `_EndpointJobHandle` to get a cancellable
    handle with the interface the rest of the module expects.

    The NetworkVolume is pinned by id (not by name): backend-yolo uploads the
    dataset/config to `RUNPOD_NETWORK_VOLUME_ID` via the S3 API, so the worker
    MUST mount that exact volume. Passing `name=` instead triggers a lookup by
    name + default datacenter (EU_RO_1), which creates a fresh empty volume in
    the wrong region — workers mount that and can't find config.json.
    """
    from runpod_flash import Endpoint, NetworkVolume

    gpu_groups = _resolve_gpu_groups(device)
    datacenter = _datacenter_from_endpoint(RUNPOD_S3_ENDPOINT_URL)

    volume_kwargs: dict[str, Any] = {"id": RUNPOD_NETWORK_VOLUME_ID, "size": RUNPOD_VOLUME_SIZE_GB}
    if datacenter is not None:
        # NetworkVolume defaults dataCenterId to EU_RO_1; if we leave it alone
        # while pinning the endpoint to a different DC, runpod-flash's own
        # cross-validation rejects the config with a ValueError at __call__.
        volume_kwargs["datacenter"] = datacenter

    # The worker re-uploads outputs via the same S3 API rather than relying on
    # filesystem→S3 propagation, which on RunPod is only eventually consistent
    # and can lag for minutes. The credentials travel in the worker env.
    worker_env = {
        "RUNPOD_S3_ENDPOINT_URL": RUNPOD_S3_ENDPOINT_URL or "",
        "RUNPOD_S3_ACCESS_KEY_ID": RUNPOD_S3_ACCESS_KEY_ID or "",
        "RUNPOD_S3_SECRET_ACCESS_KEY": RUNPOD_S3_SECRET_ACCESS_KEY or "",
        "RUNPOD_NETWORK_VOLUME_ID": RUNPOD_NETWORK_VOLUME_ID or "",
        "RUNPOD_S3_REGION": _region_from_endpoint(RUNPOD_S3_ENDPOINT_URL),
    }

    endpoint_kwargs: dict[str, Any] = {
        "name": "annotator-yolo-train",
        "gpu": gpu_groups,
        "workers": (0, 5),
        "idle_timeout": 120,
        "dependencies": ["ultralytics>=8.4.37", "torch", "torchvision", "pyyaml", "boto3"],
        "flashboot": True,
        "execution_timeout_ms": 0,
        "volume": NetworkVolume(**volume_kwargs),
        "env": worker_env,
    }
    if datacenter is not None:
        endpoint_kwargs["datacenter"] = datacenter

    return Endpoint(**endpoint_kwargs)(train_on_runpod)


def _datacenter_from_endpoint(endpoint_url: Optional[str]) -> Optional[str]:
    """Map the S3 endpoint URL to the matching RunPod DataCenter enum value.

    `s3api-us-il-1.runpod.io` → `US-IL-1`. Returned in the canonical UPPER-DASH
    form that `runpod_flash.DataCenter.from_string` accepts. Workers must be
    pinned to the same datacenter as the NetworkVolume or they can't mount it.
    """
    region = _region_from_endpoint(endpoint_url)
    if region == "us-east-1":
        return None
    return region.upper()


# ──────────────────────────────────────────────────────────────────────────
# Remote worker — runs on a RunPod GPU. Reads/writes files on the mounted
# NetworkVolume. No HTTP calls outbound.
# ──────────────────────────────────────────────────────────────────────────


async def train_on_runpod(payload):
    # runpod-flash ships this function *source* to the remote worker and
    # execs it in a fresh namespace. That means:
    #   - module-level imports do NOT carry over
    #   - `from __future__ import annotations` does NOT apply
    #   - signature annotations are evaluated at `def` time, before the body
    #     runs, so any typing names (Any/Optional/...) on the signature line
    #     will NameError. Keep the signature annotation-free.
    # Body-only annotations are fine as long as the names are imported below.
    import io
    import json as _json
    import logging as _logging
    import os as _os
    import shutil as _shutil
    import sys as _sys
    import threading as _threading
    import time as _time
    import traceback as _traceback
    import zipfile as _zipfile
    from datetime import datetime as _datetime, UTC as _UTC
    from pathlib import Path as _Path
    from typing import Any, Optional

    from ultralytics import YOLO

    # Module-level helpers don't exist in the exec'd namespace on the worker —
    # define any we need inline (same reason we import typing names here).
    def _safe_float(value):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    job_id = payload["job_id"]

    def _probe_mount(root):
        """Print mount contents so we can see where RunPod placed the volume."""
        print(f"[probe] volume_root={root} exists={root.exists()}", flush=True)
        if root.exists():
            try:
                entries = sorted(p.name for p in root.iterdir())
                print(f"[probe] {root} contents: {entries}", flush=True)
            except Exception as probe_exc:  # noqa: BLE001
                print(f"[probe] iterdir({root}) failed: {probe_exc}", flush=True)
        parents = [_Path("/runpod-volume"), _Path("/workspace"), _Path("/")]
        for candidate in parents:
            try:
                if candidate.exists():
                    names = sorted(p.name for p in candidate.iterdir())
                    print(f"[probe] {candidate} contents (first 20): {names[:20]}", flush=True)
            except Exception:  # noqa: BLE001
                pass
        for env_name in ("RUNPOD_NETWORK_VOLUME_PATH", "RUNPOD_VOLUME_PATH", "NETWORK_VOLUME_PATH"):
            value = _os.environ.get(env_name)
            if value:
                print(f"[probe] env {env_name}={value}", flush=True)

    volume_root = _Path("/runpod-volume")
    _probe_mount(volume_root)

    job_vol = volume_root / "jobs" / job_id
    datasets_vol = volume_root / "datasets"

    logs_path = job_vol / "logs.txt"
    progress_path = job_vol / "progress.jsonl"
    status_path = job_vol / "status.json"
    config_path = job_vol / "config.json"

    job_vol.mkdir(parents=True, exist_ok=True)
    logs_path.touch()

    # S3 client on the WORKER side. The backend polls via S3, and RunPod's S3
    # view of the mounted volume is only eventually consistent with filesystem
    # writes (can lag minutes). Uploading outputs through the same S3 API
    # guarantees the backend can see them right after _write_status finishes.
    s3_endpoint = _os.environ.get("RUNPOD_S3_ENDPOINT_URL") or ""
    s3_bucket = _os.environ.get("RUNPOD_NETWORK_VOLUME_ID") or ""
    s3_region = _os.environ.get("RUNPOD_S3_REGION") or "us-east-1"
    s3_access_key = _os.environ.get("RUNPOD_S3_ACCESS_KEY_ID") or ""
    s3_secret_key = _os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY") or ""
    s3_client = None
    if s3_endpoint and s3_bucket and s3_access_key and s3_secret_key:
        try:
            import boto3 as _boto3
            from botocore.config import Config as _BotoConfig

            s3_client = _boto3.client(
                "s3",
                endpoint_url=s3_endpoint,
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key,
                region_name=s3_region,
                config=_BotoConfig(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
            )
            print(f"[runpod] S3 uploader ready endpoint={s3_endpoint} bucket={s3_bucket}", flush=True)
        except Exception as s3_init_exc:  # noqa: BLE001
            print(f"[runpod] S3 uploader init failed: {s3_init_exc}", flush=True)
            s3_client = None
    else:
        print("[runpod] S3 creds absent — outputs will only be written to filesystem", flush=True)

    def _s3_put_bytes(key, data, content_type="application/octet-stream"):
        if s3_client is None:
            return False
        try:
            s3_client.put_object(Bucket=s3_bucket, Key=key, Body=data, ContentType=content_type)
            return True
        except Exception as put_exc:  # noqa: BLE001
            print(f"[runpod] S3 PUT {key} failed: {put_exc}", flush=True)
            return False

    def _s3_put_file(key, path, content_type="application/octet-stream"):
        if s3_client is None:
            return False
        try:
            s3_client.upload_file(str(path), s3_bucket, key, ExtraArgs={"ContentType": content_type})
            return True
        except Exception as put_exc:  # noqa: BLE001
            print(f"[runpod] S3 upload_file {key} failed: {put_exc}", flush=True)
            return False

    work_root = _Path("/tmp") / f"annotator-{job_id}"
    dataset_dir = work_root / "dataset"
    runs_dir = work_root / "runs"
    work_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    def _write_status(status: str, **extra: Any) -> None:
        payload_out = {"status": status, **extra}
        blob = _json.dumps(payload_out, ensure_ascii=True).encode("utf-8")
        tmp = status_path.with_suffix(".json.tmp")
        with tmp.open("wb") as target:
            target.write(blob)
        tmp.replace(status_path)
        _s3_put_bytes(f"jobs/{job_id}/status.json", blob, content_type="application/json")

    log_lock = _threading.Lock()

    class _TeeStream(io.TextIOBase):
        """Tees `print` / `warnings.showwarning` / dynamically-resolved
        `sys.stdout`/`sys.stderr` writes into both the RunPod container stdout
        and logs.txt so the backend (which reads via S3) can see them.
        Does NOT catch output from logging handlers that cached the original
        sys.stderr at construction — those are handled separately below by
        installing a FileHandler directly on the ultralytics logger.
        """

        def __init__(self, origin: Any) -> None:
            self._origin = origin
            self._buffer = ""

        def write(self, data: str) -> int:
            try:
                self._origin.write(data)
                self._origin.flush()
            except Exception:  # noqa: BLE001
                pass
            self._buffer += data
            if "\n" in self._buffer or len(self._buffer) > 4096:
                try:
                    with log_lock:
                        with logs_path.open("ab") as logs_file:
                            logs_file.write(self._buffer.encode("utf-8", errors="replace"))
                except Exception:  # noqa: BLE001
                    pass
                self._buffer = ""
            return len(data)

        def flush(self) -> None:
            if self._buffer:
                try:
                    with log_lock:
                        with logs_path.open("ab") as logs_file:
                            logs_file.write(self._buffer.encode("utf-8", errors="replace"))
                except Exception:  # noqa: BLE001
                    pass
                self._buffer = ""
            try:
                self._origin.flush()
            except Exception:  # noqa: BLE001
                pass

    original_stdout, original_stderr = _sys.stdout, _sys.stderr
    _sys.stdout = _TeeStream(original_stdout)
    _sys.stderr = _TeeStream(original_stderr)

    # Ultralytics creates its module-level LOGGER at import time, which caches
    # the original sys.stderr in its StreamHandler(s). The rebind above does
    # nothing for that handler — it holds the old stderr object directly. So
    # per-epoch tables, scanning banners, and any LOGGER.info(...) call
    # bypass TeeStream and never reach logs.txt.
    #
    # Worse, on a reused flash worker the runpod-flash runtime can replace and
    # then CLOSE stderr between calls, leaving ultralytics' cached StreamHandler
    # pointing at a closed file. Every LOGGER.info call then raises
    # `ValueError: I/O operation on closed file`, which Python's logging
    # catches and dumps as `--- Logging error ---` + traceback into the
    # current sys.stderr — flooding logs.txt with many copies of the same
    # stack trace.
    #
    # Fix: REMOVE any existing StreamHandler on the ultralytics loggers. Two
    # reasons: (a) their .stream is usually closed on a reused worker and
    # raises on every emit, flooding logs with `--- Logging error ---`
    # tracebacks; (b) if we simply repointed those streams to our sys.stderr
    # (TeeStream), LOGGER messages would be written twice — once via
    # StreamHandler→TeeStream→logs.txt and once via our FileHandler→logs.txt.
    # The FileHandler alone is sufficient for logs.txt capture. We keep the
    # root logger's handlers intact so unrelated libraries can still log to
    # stderr if they want.
    for _lg_name in ("ultralytics", "ultralytics.utils", "py.warnings"):
        _lg = _logging.getLogger(_lg_name)
        for _handler in list(_lg.handlers):
            if isinstance(_handler, _logging.FileHandler):
                continue
            if isinstance(_handler, _logging.StreamHandler):
                try:
                    _lg.removeHandler(_handler)
                except Exception:  # noqa: BLE001
                    pass
    # Repair any remaining root StreamHandlers that point to closed streams,
    # otherwise unrelated `logging.warning(...)` calls from other libraries
    # would still emit `--- Logging error ---` tracebacks.
    _root = _logging.getLogger("")
    for _handler in list(_root.handlers):
        if isinstance(_handler, _logging.FileHandler):
            continue
        if isinstance(_handler, _logging.StreamHandler):
            _stream = getattr(_handler, "stream", None)
            if _stream is None or bool(getattr(_stream, "closed", False)):
                try:
                    _handler.stream = _sys.stderr
                except Exception:  # noqa: BLE001
                    try:
                        _root.removeHandler(_handler)
                    except Exception:  # noqa: BLE001
                        pass

    logger_file_handler: Optional[Any] = None
    try:
        logger_file_handler = _logging.FileHandler(str(logs_path), mode="a", encoding="utf-8")
        logger_file_handler.setLevel(_logging.DEBUG)
        logger_file_handler.setFormatter(_logging.Formatter("%(message)s"))
        for _lg_name in ("", "ultralytics", "ultralytics.utils"):
            _lg = _logging.getLogger(_lg_name)
            _lg.addHandler(logger_file_handler)
            if _lg.level == _logging.NOTSET or _lg.level > _logging.INFO:
                _lg.setLevel(_logging.INFO)
        _logging.captureWarnings(True)
    except Exception:  # noqa: BLE001
        logger_file_handler = None

    # RunPod's filesystem→S3 view is eventually consistent and can lag minutes.
    # Short runs finish and get their job prefix cleaned up by the backend before
    # the tail of logs.txt propagates, so the backend sees logs truncated at
    # whatever the sync managed to flush. Upload the file via S3 directly on a
    # background tick — same pattern weights.pt / progress.jsonl already use.
    logs_upload_stop = _threading.Event()

    def _upload_logs_now() -> bool:
        with log_lock:
            if not logs_path.exists():
                return False
        try:
            return _s3_put_file(f"jobs/{job_id}/logs.txt", logs_path, content_type="text/plain")
        except Exception:  # noqa: BLE001
            return False

    def _logs_uploader() -> None:
        last_size = -1
        last_mtime = -1.0
        while not logs_upload_stop.wait(2.0):
            try:
                if not logs_path.exists():
                    continue
                stat = logs_path.stat()
                if stat.st_size == last_size and stat.st_mtime == last_mtime:
                    continue
                if _upload_logs_now():
                    last_size = stat.st_size
                    last_mtime = stat.st_mtime
            except Exception:  # noqa: BLE001
                pass

    logs_uploader_thread = _threading.Thread(target=_logs_uploader, daemon=True)
    logs_uploader_thread.start()

    # Resource sampler: writes a resources.jsonl line every 5s and uploads
    # it to S3 so the backend-yolo bridge can forward snapshots to the JS
    # backend. Inlined (not imported) because runpod-flash ships this
    # function source and execs it in a fresh namespace.
    import psutil as _psutil
    try:
        _psutil.cpu_percent(interval=None)  # prime
    except Exception:  # noqa: BLE001
        pass

    resources_path = job_vol / "resources.jsonl"
    resources_stop = _threading.Event()
    _BYTES_PER_GB = 1024.0 ** 3

    def _sample_resources():
        sample = {
            "vram_used_gb": None,
            "vram_total_gb": None,
            "vram_used_pct": None,
            "ram_used_gb": None,
            "ram_total_gb": None,
            "ram_used_pct": None,
            "cpu_pct": None,
            "disk_used_gb": None,
            "disk_total_gb": None,
            "disk_used_pct": None,
            "sampled_at": _datetime.now(_UTC).isoformat(),
        }
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _dev = _torch.cuda.current_device()
                _free, _total = _torch.cuda.mem_get_info(_dev)
                _used = max(0, _total - _free)
                sample["vram_used_gb"] = round(_used / _BYTES_PER_GB, 2)
                sample["vram_total_gb"] = round(_total / _BYTES_PER_GB, 2)
                if _total > 0:
                    sample["vram_used_pct"] = round(_used / _total * 100.0, 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            _vm = _psutil.virtual_memory()
            sample["ram_used_gb"] = round(_vm.used / _BYTES_PER_GB, 2)
            sample["ram_total_gb"] = round(_vm.total / _BYTES_PER_GB, 2)
            sample["ram_used_pct"] = round(_vm.percent, 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            sample["cpu_pct"] = round(_psutil.cpu_percent(interval=None), 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            _du = _psutil.disk_usage("/")
            sample["disk_used_gb"] = round(_du.used / _BYTES_PER_GB, 2)
            sample["disk_total_gb"] = round(_du.total / _BYTES_PER_GB, 2)
            sample["disk_used_pct"] = round(_du.percent, 1)
        except Exception:  # noqa: BLE001
            pass
        return sample

    def _resources_sampler():
        while not resources_stop.wait(5.0):
            try:
                sample = _sample_resources()
                with resources_path.open("a", encoding="utf-8") as _fh:
                    _fh.write(_json.dumps(sample, ensure_ascii=True) + "\n")
                _s3_put_file(
                    f"jobs/{job_id}/resources.jsonl", resources_path, content_type="application/x-ndjson"
                )
            except Exception as _sample_exc:  # noqa: BLE001
                _log_direct_msg = f"[resources] sampler error: {_sample_exc}"
                try:
                    with log_lock:
                        with logs_path.open("ab") as _f:
                            _f.write((_log_direct_msg + "\n").encode("utf-8", errors="replace"))
                except Exception:  # noqa: BLE001
                    pass

    resources_sampler_thread = _threading.Thread(target=_resources_sampler, daemon=True)
    resources_sampler_thread.start()

    try:
        if not config_path.exists():
            raise RuntimeError("config.json not found on volume — backend-yolo did not stage the job")
        with config_path.open("r", encoding="utf-8") as config_file:
            config = _json.load(config_file)

        dataset_hash_path = job_vol / "dataset_hash.txt"
        if not dataset_hash_path.exists():
            raise RuntimeError("dataset_hash.txt not found on volume")
        dataset_hash = dataset_hash_path.read_text(encoding="utf-8").strip()

        cached_zip = datasets_vol / f"{dataset_hash}.zip"
        if not cached_zip.exists():
            raise RuntimeError(f"Cached dataset {cached_zip} missing on volume")

        print(f"[runpod] extracting dataset {dataset_hash}", flush=True)
        if dataset_dir.exists():
            _shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with _zipfile.ZipFile(cached_zip, "r") as archive:
            archive.extractall(dataset_dir)

        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists():
            children = [entry for entry in dataset_dir.iterdir() if entry.is_dir()]
            if len(children) == 1 and (children[0] / "data.yaml").exists():
                nested = children[0]
                for entry in nested.iterdir():
                    _shutil.move(str(entry), str(dataset_dir / entry.name))
                nested.rmdir()
        if not data_yaml.exists():
            raise RuntimeError("data.yaml missing in dataset zip")

        try:
            import yaml as _yaml

            with data_yaml.open("r", encoding="utf-8") as yf:
                data_cfg = _yaml.safe_load(yf) or {}
            data_cfg["path"] = str(dataset_dir.resolve())
            with data_yaml.open("w", encoding="utf-8") as yf:
                _yaml.safe_dump(data_cfg, yf, sort_keys=False, allow_unicode=True)
        except Exception:  # noqa: BLE001
            pass

        _write_status("running", started_at=_time.time())

        total_epochs = int(config.get("epochs") or 50)

        def _log_direct(msg: str) -> None:
            # Bypass sys.stdout/sys.stderr entirely so diagnostic lines land
            # in logs.txt even when Python-level streams are in a bad state
            # (closed, replaced by the flash runtime, etc.).
            try:
                with log_lock:
                    with logs_path.open("ab") as _logs_file:
                        _logs_file.write((msg + "\n").encode("utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                pass

        def _emit_progress(trainer: Any, hook: str) -> None:
            # Logged so we can tell from logs.txt whether ultralytics is
            # actually firing the callback — if this line never appears, the
            # issue is hook registration, not progress.jsonl uploading.
            try:
                epoch = int(getattr(trainer, "epoch", 0)) + 1
            except Exception:  # noqa: BLE001
                epoch = 0
            _log_direct(f"[progress] {hook} epoch={epoch}/{total_epochs}")
            try:
                metrics = getattr(trainer, "metrics", {}) or {}
                line = _json.dumps(
                    {
                        "current_epoch": epoch,
                        "total_epochs": total_epochs,
                        "metrics": {
                            "precision": _safe_float(metrics.get("metrics/precision(B)")),
                            "recall": _safe_float(metrics.get("metrics/recall(B)")),
                            "map50": _safe_float(metrics.get("metrics/mAP50(B)")),
                            "map": _safe_float(metrics.get("metrics/mAP50-95(B)")),
                        },
                    },
                    ensure_ascii=True,
                )
                with progress_path.open("a", encoding="utf-8") as progress_file:
                    progress_file.write(line + "\n")
                if not _s3_put_file(
                    f"jobs/{job_id}/progress.jsonl", progress_path, content_type="application/x-ndjson"
                ):
                    _log_direct(f"[progress] S3 upload of progress.jsonl failed at epoch {epoch}")
            except Exception as progress_exc:  # noqa: BLE001
                _log_direct(f"[progress] callback error at epoch {epoch}: {progress_exc}")

        def _on_train_epoch_end(trainer: Any) -> None:
            _emit_progress(trainer, "on_train_epoch_end")

        def _on_fit_epoch_end(trainer: Any) -> None:
            _emit_progress(trainer, "on_fit_epoch_end")

        def _on_train_start(trainer: Any) -> None:
            # Smoke test — if this line never appears, none of the callbacks
            # we registered on the model were carried through to the trainer.
            _log_direct("[progress] on_train_start fired")

        base = config["base_model"]
        print(f"[runpod] loading base model {base}", flush=True)
        model = YOLO(base)
        # Register hooks — different ultralytics versions fire them in
        # slightly different orders; taking the latest valid one is fine
        # because the backend's _apply_progress_tail only uses the last line.
        model.add_callback("on_train_start", _on_train_start)
        model.add_callback("on_train_epoch_end", _on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        try:
            for _event_name in ("on_train_start", "on_train_epoch_end", "on_fit_epoch_end"):
                _count = len((model.callbacks or {}).get(_event_name, []))
                _log_direct(f"[runpod] model.callbacks[{_event_name}]={_count}")
        except Exception as _cb_exc:  # noqa: BLE001
            _log_direct(f"[runpod] callback introspection failed: {_cb_exc}")

        train_args: dict[str, Any] = {
            "data": str(data_yaml),
            "project": str(runs_dir),
            "name": "train",
            "exist_ok": True,
            "epochs": total_epochs,
            "imgsz": int(config.get("imgsz") or 640),
            "batch": int(config.get("batch") or 16),
            "workers": int(config.get("workers") or 2),
            "device": 0,
            "verbose": True,
        }

        _runpod_reserved = {"data", "project", "name", "exist_ok", "epochs", "imgsz", "batch", "workers", "device", "verbose"}
        _hyperparams = config.get("hyperparams") or {}
        if isinstance(_hyperparams, dict) and _hyperparams:
            for _k, _v in _hyperparams.items():
                if _k in _runpod_reserved or _v is None:
                    continue
                train_args[_k] = _v

        print(f"[runpod] train args: {_json.dumps(train_args, default=str)}", flush=True)
        results = model.train(**train_args)

        save_dir = _Path(getattr(results, "save_dir", ""))
        best_pt = save_dir / "weights" / "best.pt" if save_dir else None
        if not best_pt or not best_pt.exists():
            raise RuntimeError("best.pt not produced by ultralytics")

        # Copy weights + artifacts to volume so backend-yolo can fetch them.
        # We also upload via S3 directly because RunPod's filesystem→S3 view is
        # only eventually consistent and can lag for minutes — the backend
        # gives up after ~16s of retries, so a single missed S3 PUT here used
        # to surface as a confusing "weights.pt never appeared" job failure.
        weights_dst = job_vol / "weights.pt"
        _shutil.copy2(best_pt, weights_dst)
        if s3_client is None:
            raise RuntimeError(
                "RunPod S3 credentials missing on the worker — weights.pt cannot be uploaded. "
                "Set RUNPOD_S3_ENDPOINT_URL / RUNPOD_S3_ACCESS_KEY_ID / RUNPOD_S3_SECRET_ACCESS_KEY / "
                "RUNPOD_NETWORK_VOLUME_ID on the endpoint."
            )
        weights_uploaded = False
        last_put_error: Optional[str] = None
        for weights_attempt in range(5):
            if _s3_put_file(
                f"jobs/{job_id}/weights.pt", weights_dst, content_type="application/octet-stream"
            ):
                weights_uploaded = True
                break
            last_put_error = f"attempt {weights_attempt + 1}/5 failed"
            _time.sleep(2)
        if not weights_uploaded:
            raise RuntimeError(
                "Failed to upload weights.pt to the RunPod NetworkVolume via S3 after 5 attempts"
                + (f" ({last_put_error})" if last_put_error else "")
                + ". Check the preceding '[runpod] S3 upload_file ... failed:' lines for the underlying error."
            )
        print(
            f"[runpod] wrote weights.pt ({weights_dst.stat().st_size} bytes) s3_upload=ok",
            flush=True,
        )
        artifacts_vol = job_vol / "artifacts"
        artifacts_vol.mkdir(parents=True, exist_ok=True)
        for artifact_name in ("results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png", "results.csv"):
            source = save_dir / artifact_name
            if source.exists():
                try:
                    artifact_dst = artifacts_vol / artifact_name
                    _shutil.copy2(source, artifact_dst)
                    _s3_put_file(f"jobs/{job_id}/artifacts/{artifact_name}", artifact_dst)
                except Exception:  # noqa: BLE001
                    pass

        raw = getattr(results, "results_dict", {}) or {}
        metrics = {
            "precision": _safe_float(raw.get("metrics/precision(B)") or raw.get("metrics/precision")),
            "recall": _safe_float(raw.get("metrics/recall(B)") or raw.get("metrics/recall")),
            "map50": _safe_float(raw.get("metrics/mAP50(B)") or raw.get("metrics/mAP50")),
            "map": _safe_float(raw.get("metrics/mAP50-95(B)") or raw.get("metrics/mAP50-95")),
        }

        try:
            _sys.stdout.flush()
            _sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        _upload_logs_now()
        _write_status("completed", completed_at=_time.time(), metrics=metrics)
        return {"status": "completed", "metrics": metrics}
    except Exception as exc:  # noqa: BLE001
        _traceback.print_exc()
        try:
            _sys.stdout.flush()
            _sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            _upload_logs_now()
            _write_status("failed", error=str(exc), completed_at=_time.time())
        except Exception:  # noqa: BLE001
            pass
        return {"status": "failed", "error": str(exc)}
    finally:
        try:
            _sys.stdout.flush()
            _sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        _sys.stdout, _sys.stderr = original_stdout, original_stderr
        # Detach the direct-to-file logger handler and close it so its buffer
        # is flushed into logs.txt before the final S3 upload.
        if logger_file_handler is not None:
            try:
                for _lg_name in ("", "ultralytics", "ultralytics.utils"):
                    _lg = _logging.getLogger(_lg_name)
                    if logger_file_handler in _lg.handlers:
                        _lg.removeHandler(logger_file_handler)
                logger_file_handler.close()
            except Exception:  # noqa: BLE001
                pass
        logs_upload_stop.set()
        try:
            logs_uploader_thread.join(timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        resources_stop.set()
        try:
            resources_sampler_thread.join(timeout=3.0)
        except Exception:  # noqa: BLE001
            pass
        _upload_logs_now()


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
