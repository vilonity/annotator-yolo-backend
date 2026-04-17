import asyncio
import json
import shutil
import subprocess
import sys
import threading
import zipfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, AsyncIterator, Optional

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
LOG_STREAM_POLL_SECONDS = 0.5
LOG_STREAM_TRAILING_GRACE_SECONDS = 2.0

import yaml
from fastapi import HTTPException, UploadFile

from app.config import BASE_DIR, TRAIN_JOBS_DIR, YOLO_MODELS_DIR
from app.schemas.training import TrainingJobDetail, TrainingJobSummary


class TrainingService:
    _processes: dict[str, subprocess.Popen] = {}
    _log_handles: dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def initialize(cls) -> None:
        TRAIN_JOBS_DIR.mkdir(parents=True, exist_ok=True)

        for job_dir in TRAIN_JOBS_DIR.iterdir():
            if not job_dir.is_dir():
                continue

            metadata = cls._read_metadata(job_dir)
            if not metadata:
                continue

            if metadata.get("status") in {"queued", "running", "cancelling"}:
                metadata["status"] = "interrupted"
                metadata["completed_at"] = now_iso()
                metadata.setdefault("error", "Training job was interrupted before completion")
                cls._write_metadata(job_dir, metadata)

    @classmethod
    async def start_job(
        cls,
        *,
        dataset_file: UploadFile,
        user_id: str,
        project_name: str,
        output_model_name: str,
        base_model: str,
        epochs: int,
        train_percent: int,
        val_percent: int,
        test_percent: int,
        total_images: int,
        boxed_images: int,
        empty_images: int,
        classes_json: str,
        imgsz: Optional[int] = None,
        batch: Optional[int] = None,
        workers: Optional[int] = None,
        device: Optional[str] = None,
    ) -> TrainingJobDetail:
        if (YOLO_MODELS_DIR / output_model_name).exists():
            raise HTTPException(status_code=400, detail="Model with this name already exists")

        try:
            classes = json.loads(classes_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid classes_json payload") from exc

        if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
            raise HTTPException(status_code=400, detail="classes_json must be a JSON string array")

        job_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        job_dir = TRAIN_JOBS_DIR / job_id
        dataset_zip_path = job_dir / "dataset.zip"
        dataset_dir = job_dir / "dataset"
        logs_path = job_dir / "logs.txt"
        config_path = job_dir / "config.json"

        job_dir.mkdir(parents=True, exist_ok=False)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logs_path.touch()

        try:
            with dataset_zip_path.open("wb") as zip_buffer:
                shutil.copyfileobj(dataset_file.file, zip_buffer)

            try:
                with zipfile.ZipFile(dataset_zip_path, "r") as archive:
                    archive.extractall(dataset_dir)
            except zipfile.BadZipFile as exc:
                raise HTTPException(status_code=400, detail="Training dataset ZIP is invalid") from exc

            dataset_root = cls._resolve_dataset_root(dataset_dir)
            if not (dataset_root / "data.yaml").exists():
                raise HTTPException(status_code=400, detail="Training dataset ZIP does not contain data.yaml")
            if dataset_root != dataset_dir:
                cls._flatten_dataset_root(dataset_root, dataset_dir)

            cls._pin_data_yaml_path(dataset_dir)

            config = {
                "job_id": job_id,
                "user_id": user_id,
                "project_name": project_name,
                "output_model_name": output_model_name,
                "base_model": base_model,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "workers": workers,
                "device": device or "auto",
                "split": {
                    "train_percent": train_percent,
                    "val_percent": val_percent,
                    "test_percent": test_percent,
                },
                "classes": classes,
            }
            cls._write_json(config_path, config)

            metadata = {
                "id": job_id,
                "user_id": user_id,
                "project_name": project_name,
                "output_model_name": output_model_name,
                "produced_model_name": None,
                "base_model": base_model,
                "status": "queued",
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "workers": workers,
                "device": device or "auto",
                "total_images": total_images,
                "boxed_images": boxed_images,
                "empty_images": empty_images,
                "classes": classes,
                "split": config["split"],
                "created_at": now_iso(),
                "started_at": None,
                "completed_at": None,
                "metrics": None,
                "error": None,
                "artifacts": None,
            }
            cls._write_metadata(job_dir, metadata)

            command = [
                sys.executable,
                "-m",
                "app.services.training_worker",
                "--job-dir",
                str(job_dir),
            ]
            log_handle = logs_path.open("a", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(BASE_DIR),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                log_handle.close()
                raise

            metadata["status"] = "running"
            metadata["started_at"] = now_iso()
            cls._write_metadata(job_dir, metadata)

            with cls._lock:
                cls._processes[job_id] = process
                cls._log_handles[job_id] = log_handle

            monitor = threading.Thread(target=cls._monitor_process, args=(job_id, job_dir), daemon=True)
            monitor.start()

            return cls.get_job(job_id)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    @classmethod
    def list_jobs(cls, user_id: Optional[str] = None, project_name: Optional[str] = None) -> list[TrainingJobSummary]:
        jobs: list[TrainingJobSummary] = []

        for job_dir in TRAIN_JOBS_DIR.iterdir():
            if not job_dir.is_dir():
                continue

            metadata = cls._read_metadata(job_dir)
            if not metadata:
                continue
            if user_id and metadata.get("user_id") != user_id:
                continue
            if project_name and metadata.get("project_name") != project_name:
                continue

            jobs.append(TrainingJobSummary.model_validate(metadata))

        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    @classmethod
    def get_job(cls, job_id: str) -> TrainingJobDetail:
        job_dir = TRAIN_JOBS_DIR / job_id
        metadata = cls._read_metadata(job_dir)
        if not metadata:
            raise HTTPException(status_code=404, detail="Training job not found")

        return TrainingJobDetail.model_validate(
            {
                **metadata,
                "logs_tail": cls._read_log_tail(job_dir / "logs.txt"),
            }
        )

    @classmethod
    def cancel_job(cls, job_id: str) -> TrainingJobDetail:
        job = cls.get_job(job_id)
        if job.status in {"completed", "failed", "cancelled", "interrupted"}:
            return job

        with cls._lock:
            process = cls._processes.get(job_id)

        if process is None:
            metadata = cls._read_metadata(TRAIN_JOBS_DIR / job_id)
            if metadata:
                metadata["status"] = "interrupted"
                metadata["completed_at"] = now_iso()
                metadata["error"] = metadata.get("error") or "Training process is no longer active"
                cls._write_metadata(TRAIN_JOBS_DIR / job_id, metadata)
            return cls.get_job(job_id)

        metadata = cls._read_metadata(TRAIN_JOBS_DIR / job_id) or {}
        metadata["status"] = "cancelling"
        metadata["error"] = "Cancellation requested"
        cls._write_metadata(TRAIN_JOBS_DIR / job_id, metadata)

        process.terminate()
        return cls.get_job(job_id)

    @classmethod
    def has_job(cls, job_id: str) -> bool:
        return cls._read_metadata(TRAIN_JOBS_DIR / job_id) is not None

    @classmethod
    def read_full_logs(cls, job_id: str) -> str:
        logs_path = TRAIN_JOBS_DIR / job_id / "logs.txt"
        if not logs_path.exists():
            return ""
        with logs_path.open("r", encoding="utf-8", errors="ignore") as log_file:
            return log_file.read()

    @classmethod
    async def stream_logs(cls, job_id: str) -> AsyncIterator[bytes]:
        job_dir = TRAIN_JOBS_DIR / job_id
        logs_path = job_dir / "logs.txt"

        position = 0
        idle_after_terminal = 0.0

        while True:
            if logs_path.exists():
                with logs_path.open("rb") as log_file:
                    log_file.seek(position)
                    chunk = log_file.read()
                    position = log_file.tell()
                if chunk:
                    yield chunk

            metadata = cls._read_metadata(job_dir) or {}
            if metadata.get("status") in TERMINAL_STATUSES:
                idle_after_terminal += LOG_STREAM_POLL_SECONDS
                if idle_after_terminal >= LOG_STREAM_TRAILING_GRACE_SECONDS:
                    return
            else:
                idle_after_terminal = 0.0

            await asyncio.sleep(LOG_STREAM_POLL_SECONDS)

    @classmethod
    def _monitor_process(cls, job_id: str, job_dir: Path) -> None:
        with cls._lock:
            process = cls._processes.get(job_id)
        if process is None:
            return

        return_code = process.wait()

        with cls._lock:
            log_handle = cls._log_handles.pop(job_id, None)
            cls._processes.pop(job_id, None)

        if log_handle:
            log_handle.close()

        metadata = cls._read_metadata(job_dir) or {}
        metadata["completed_at"] = now_iso()

        if return_code == 0:
            metadata["status"] = "completed"
            metadata["produced_model_name"] = metadata.get("output_model_name")
            metadata["metrics"] = cls._read_json(job_dir / "metrics.json")
            metadata["artifacts"] = cls._read_json(job_dir / "artifacts.json")
            metadata["error"] = None
        elif metadata.get("status") == "cancelling":
            metadata["status"] = "cancelled"
            metadata["error"] = "Training job was cancelled"
        else:
            metadata["status"] = "failed"
            error_payload = cls._read_json(job_dir / "error.json") or {}
            metadata["error"] = error_payload.get("error") or "Training job failed"

        cls._write_metadata(job_dir, metadata)

    @classmethod
    def _resolve_dataset_root(cls, dataset_dir: Path) -> Path:
        if (dataset_dir / "data.yaml").exists():
            return dataset_dir

        child_directories = [entry for entry in dataset_dir.iterdir() if entry.is_dir()]
        if len(child_directories) == 1 and (child_directories[0] / "data.yaml").exists():
            return child_directories[0]

        return dataset_dir

    @classmethod
    def _pin_data_yaml_path(cls, dataset_dir: Path) -> None:
        data_yaml_path = dataset_dir / "data.yaml"
        with data_yaml_path.open("r", encoding="utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file) or {}

        data["path"] = dataset_dir.resolve().as_posix()

        with data_yaml_path.open("w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(data, yaml_file, sort_keys=False, allow_unicode=True)

    @classmethod
    def _flatten_dataset_root(cls, dataset_root: Path, dataset_dir: Path) -> None:
        temp_dir = dataset_dir.parent / f"{dataset_dir.name}-normalized"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for entry in dataset_root.iterdir():
            shutil.move(str(entry), str(temp_dir / entry.name))

        shutil.rmtree(dataset_dir, ignore_errors=True)
        temp_dir.rename(dataset_dir)

    @classmethod
    def _read_metadata(cls, job_dir: Path) -> Optional[dict[str, Any]]:
        return cls._read_json(job_dir / "metadata.json")

    @classmethod
    def _write_metadata(cls, job_dir: Path, payload: dict[str, Any]) -> None:
        cls._write_json(job_dir / "metadata.json", payload)

    @staticmethod
    def _read_json(path: Path) -> Optional[dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as json_file:
                return json.load(json_file)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, ensure_ascii=True, indent=2)

    @staticmethod
    def _read_log_tail(path: Path, max_lines: int = 80) -> list[str]:
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8", errors="ignore") as log_file:
            lines = [line.rstrip() for line in log_file.readlines()]
        return lines[-max_lines:]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
