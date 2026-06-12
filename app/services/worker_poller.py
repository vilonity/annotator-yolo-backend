"""Pull-mode worker: long-polls the central Bun API for training work.

The central server cannot reach this machine (NAT, no tunnels), so all
traffic is outbound: this poller asks ``POST /api/worker/poll`` for queued
training jobs and pending cancellations, downloads staged datasets, starts
the regular local training pipeline, and lets the existing
``training_callback`` machinery report progress/logs/artifacts back.

Enabled when ``ANNOTATOR_API_URL`` and ``ANNOTATOR_WORKER_TOKEN`` are set
(see ``app/config.py``); the token is generated per user in the web app's
account settings.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

try:
    import requests
except ImportError:  # pragma: no cover - requests is in requirements but guard for tooling
    requests = None  # type: ignore[assignment]

from app.config import ANNOTATOR_API_URL, ANNOTATOR_API_VERIFY_TLS, ANNOTATOR_WORKER_TOKEN, TRAIN_JOBS_DIR
from app.services import training_callback
from app.services.training_service import TERMINAL_STATUSES, TrainingService, now_iso

# The server holds a poll open for up to ~20s; give it headroom.
POLL_TIMEOUT_SECONDS = 40.0
DOWNLOAD_TIMEOUT_SECONDS = 600.0
ERROR_BACKOFF_MIN_SECONDS = 5.0
ERROR_BACKOFF_MAX_SECONDS = 60.0
# The server re-queues a claimed job after STALE_CLAIM_MS (3 min) without a
# heartbeat, so refresh the claim well within that window while downloading.
CLAIM_HEARTBEAT_INTERVAL_SECONDS = 60.0


class JobCancelledDuringDownload(Exception):
    """A claim heartbeat reported the job was cancelled server-side."""


def remote_job_id(bun_job_id: int) -> str:
    """Deterministic local job id for a job dispatched by the central API."""
    return f"remote-{bun_job_id}"


def _format_download_progress(downloaded: int, total: int) -> str:
    downloaded_mb = downloaded / (1024 * 1024)
    if total > 0:
        return f"{downloaded_mb:.0f}/{total / (1024 * 1024):.0f} MB ({downloaded * 100 // total}%)"
    return f"{downloaded_mb:.0f} MB"


def _local_devices_payload() -> list[dict[str, Any]]:
    try:
        return [device.model_dump() for device in TrainingService.list_devices(provider="local")]
    except Exception as exc:  # noqa: BLE001
        print(f"[worker-poller] failed to collect devices: {exc}", flush=True)
        return []


class WorkerPoller:
    def __init__(self, api_url: str, token: str, *, verify_tls: bool = True) -> None:
        self._api_url = api_url.rstrip("/")
        self._token = token
        self._verify_tls = verify_tls
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="worker-poller", daemon=True)
        self._thread.start()
        print(f"[worker-poller] polling {self._api_url} for training jobs", flush=True)

    # ── poll loop ────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _rebase_callback_url(self, callback_url: Optional[str]) -> Optional[str]:
        """Point the callback at the same server we poll.

        The central API builds callback URLs from its BACKEND_HOST env, which
        is easy to leave at the localhost default. The worker already knows
        the right origin (ANNOTATOR_API_URL), so keep only the path.
        """
        if not callback_url:
            return callback_url
        return f"{self._api_url}{urlsplit(callback_url).path}"

    def _loop(self) -> None:
        backoff = ERROR_BACKOFF_MIN_SECONDS
        while True:
            try:
                response = requests.post(
                    f"{self._api_url}/api/worker/poll",
                    json={"devices": _local_devices_payload()},
                    headers=self._headers(),
                    timeout=POLL_TIMEOUT_SECONDS,
                    verify=self._verify_tls,
                )
                if response.status_code == 401:
                    print(
                        "[worker-poller] worker token rejected (401) — regenerate it in account settings",
                        flush=True,
                    )
                    time.sleep(ERROR_BACKOFF_MAX_SECONDS)
                    continue
                response.raise_for_status()
                data = response.json()

                for cancellation in data.get("cancellations") or []:
                    self._handle_cancellation(cancellation)

                job = data.get("job")
                if job:
                    self._handle_job(job)

                backoff = ERROR_BACKOFF_MIN_SECONDS
            except Exception as exc:  # noqa: BLE001
                print(f"[worker-poller] poll error: {exc} (retry in {backoff:.0f}s)", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, ERROR_BACKOFF_MAX_SECONDS)

    # ── cancellations ────────────────────────────────────────────────

    def _handle_cancellation(self, cancellation: dict[str, Any]) -> None:
        bun_job_id = cancellation.get("job_id")
        if bun_job_id is None:
            return
        local_id = remote_job_id(bun_job_id)
        callback_url = self._rebase_callback_url(cancellation.get("callback_url"))
        callback_secret = cancellation.get("callback_secret")
        try:
            metadata = TrainingService._read_metadata(TRAIN_JOBS_DIR / local_id)  # noqa: SLF001 - internal reuse
            if metadata is None:
                # The job never reached this machine (or its dir is gone) —
                # finalize it on the server so it stops being redelivered.
                training_callback.post_progress(
                    callback_url,
                    callback_secret,
                    {
                        "status": "cancelled",
                        "completed_at": now_iso(),
                        "error": "Training job was cancelled",
                    },
                )
            elif metadata.get("status") in TERMINAL_STATUSES:
                # Already finished locally but the server still thinks it is
                # cancelling (e.g. the original terminal callback was lost) —
                # re-post the final state to sync up.
                training_callback.post_progress(
                    callback_url,
                    callback_secret,
                    {
                        "status": metadata.get("status"),
                        "completed_at": metadata.get("completed_at") or now_iso(),
                        "produced_model_name": metadata.get("produced_model_name"),
                        "metrics": metadata.get("metrics"),
                        "error": metadata.get("error"),
                    },
                )
            else:
                asyncio.run(TrainingService.cancel_job(local_id))
                print(f"[worker-poller] cancellation forwarded to job {local_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker-poller] cancel of {local_id} failed: {exc}", flush=True)

    # ── job dispatch ─────────────────────────────────────────────────

    def _post_claim_heartbeat(self, bun_job_id: int) -> Optional[str]:
        """Refresh the server-side claim. Returns the job's current status, or
        None when the heartbeat failed (best-effort: one miss only matters if
        every retry within the stale-claim window misses too)."""
        try:
            response = requests.post(
                f"{self._api_url}/api/worker/jobs/{bun_job_id}/heartbeat",
                headers=self._headers(),
                timeout=10.0,
                verify=self._verify_tls,
            )
            response.raise_for_status()
            status = response.json().get("status")
            return status if isinstance(status, str) else None
        except Exception as exc:  # noqa: BLE001
            print(f"[worker-poller] claim heartbeat for job {bun_job_id} failed: {exc}", flush=True)
            return None

    def _download_dataset(
        self,
        bun_job_id: int,
        dataset_path: str,
        target: Path,
        callback_url: Optional[str],
        callback_secret: Optional[str],
    ) -> None:
        """Stream the dataset zip, heartbeating the claim and reporting progress.

        Large datasets take longer than the server's stale-claim window, so
        without the periodic heartbeat a second worker instance would re-claim
        and double-start the job mid-download.
        """
        with requests.get(
            f"{self._api_url}{dataset_path}",
            headers=self._headers(),
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            verify=self._verify_tls,
            stream=True,
        ) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            last_report = time.monotonic()
            with target.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if time.monotonic() - last_report < CLAIM_HEARTBEAT_INTERVAL_SECONDS:
                        continue
                    last_report = time.monotonic()
                    if self._post_claim_heartbeat(bun_job_id) in ("cancelling", "cancelled"):
                        raise JobCancelledDuringDownload()
                    progress = _format_download_progress(downloaded, total_bytes)
                    print(f"[worker-poller] job {bun_job_id} dataset download: {progress}", flush=True)
                    training_callback.post_log_chunk(
                        callback_url, callback_secret, f"[worker] downloading dataset: {progress}\n"
                    )

    def _handle_job(self, job: dict[str, Any]) -> None:
        local_id = remote_job_id(job["id"])
        callback_url = self._rebase_callback_url(job.get("callback_url"))
        callback_secret = job.get("callback_secret")

        if TrainingService.has_job(local_id):
            # Redelivery (e.g. the "running" callback was lost) — re-post the
            # current state instead of double-starting the job.
            metadata = TrainingService._read_metadata(TRAIN_JOBS_DIR / local_id)  # noqa: SLF001 - internal reuse
            if metadata:
                training_callback.post_progress(
                    callback_url,
                    callback_secret,
                    {"status": metadata.get("status"), "started_at": metadata.get("started_at")},
                )
            print(f"[worker-poller] job {local_id} already exists — skipped duplicate dispatch", flush=True)
            return

        print(f"[worker-poller] claimed training job {local_id}", flush=True)
        tmp_dir = tempfile.mkdtemp(prefix="annotator-worker-")
        tmp_zip = Path(tmp_dir) / "dataset.zip"
        try:
            # The job stays "queued" on the server until the spawned training
            # process posts "running" — surface the download in the job logs so
            # the UI is not silent for the whole transfer.
            training_callback.post_log_chunk(callback_url, callback_secret, "[worker] downloading dataset...\n")
            self._download_dataset(job["id"], job["dataset_path"], tmp_zip, callback_url, callback_secret)

            size_mb = tmp_zip.stat().st_size / (1024 * 1024)
            print(f"[worker-poller] job {local_id} dataset downloaded ({size_mb:.0f} MB)", flush=True)
            training_callback.post_log_chunk(
                callback_url,
                callback_secret,
                f"[worker] dataset downloaded ({size_mb:.0f} MB), starting training\n",
            )
            # Extraction + process spawn below can also take a while on big
            # datasets — refresh the claim once more before committing to it.
            self._post_claim_heartbeat(job["id"])

            hyperparams = job.get("hyperparams") or {}
            split = job.get("split") or {}
            detail = asyncio.run(
                TrainingService.start_job(
                    dataset_zip_source=tmp_zip,
                    job_id=local_id,
                    user_id=job["user_key"],
                    project_name=job["project_name"],
                    output_model_name=job["output_model_name"],
                    base_model=job["base_model"],
                    architecture=job.get("architecture") or "yolo",
                    epochs=job["epochs"],
                    imgsz=job.get("imgsz"),
                    batch=job.get("batch"),
                    workers=job.get("workers"),
                    device=job.get("device") or "auto",
                    provider="local",
                    train_percent=split.get("train_percent", 0),
                    val_percent=split.get("val_percent", 0),
                    test_percent=split.get("test_percent", 0),
                    total_images=job.get("total_images", 0),
                    boxed_images=job.get("boxed_images", 0),
                    empty_images=job.get("empty_images", 0),
                    classes_json=json.dumps(job.get("classes") or []),
                    external_callback_url=callback_url,
                    external_callback_secret=callback_secret,
                    hyperparams_json=json.dumps(hyperparams) if hyperparams else None,
                )
            )
            # The spawned training process posts "running" itself; device_name
            # is only known here, so sync it separately (no status override).
            if detail.device_name:
                training_callback.post_progress_async(
                    callback_url, callback_secret, {"device_name": detail.device_name}
                )
        except JobCancelledDuringDownload:
            print(f"[worker-poller] job {local_id} cancelled during dataset download", flush=True)
            training_callback.post_progress(
                callback_url,
                callback_secret,
                {"status": "cancelled", "completed_at": now_iso(), "error": "Training job was cancelled"},
            )
        except Exception as exc:  # noqa: BLE001
            detail_text = getattr(exc, "detail", None) or str(exc) or "Failed to start training job"
            print(f"[worker-poller] failed to start job {local_id}: {detail_text}", flush=True)
            training_callback.post_progress(
                callback_url,
                callback_secret,
                {"status": "failed", "error": str(detail_text), "completed_at": now_iso()},
            )
        finally:
            try:
                tmp_zip.unlink(missing_ok=True)
                Path(tmp_dir).rmdir()
            except OSError:
                pass


def start_worker_poller_if_configured() -> Optional[WorkerPoller]:
    if not ANNOTATOR_API_URL or not ANNOTATOR_WORKER_TOKEN:
        return None
    if requests is None:
        print("[worker-poller] requests not installed — pull-mode worker disabled", flush=True)
        return None
    poller = WorkerPoller(ANNOTATOR_API_URL, ANNOTATOR_WORKER_TOKEN, verify_tls=ANNOTATOR_API_VERIFY_TLS)
    poller.start()
    return poller
