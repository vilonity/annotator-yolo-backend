"""Standalone single-job trainer for on-demand RunPod Pods (provider "runpod-pod").

The API drives a bare GPU Pod over SSH: it uploads this backend-yolo source plus
the dataset, then runs `python -m app.pod_train --job-dir <dir>`. This module
runs exactly one training job from `<job-dir>/config.json` + `<job-dir>/dataset`
and writes its output as plain files in `<job-dir>`, which the API tails back
over SFTP. There are NO network calls, NO S3, and NO model registry — the Pod is
ephemeral and the API pulls the weights off it.

Files written under `<job-dir>`:

  logs.txt         append-only stdout/stderr
  progress.jsonl   one JSON line per finished epoch ({current_epoch, total_epochs, metrics})
  resources.jsonl  one JSON line every 5s (vram/ram/cpu/disk)
  status.json      terminal marker {status, error?, metrics?}
  weights.pt|.pth  best checkpoint (YOLO → .pt, RF-DETR → .pth)
  artifacts/*      result plots + results.csv / metrics.csv / log.txt

A `<job-dir>/cancel.json` dropped by the API triggers a graceful "stop & keep":
training halts at the next epoch boundary and the best checkpoint so far is kept,
finalized by the API as a cancellation that still recovers the model.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
import threading
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1024.0**3

# Artifacts copied into <job-dir>/artifacts so the API can pull them. Missing
# ones are skipped (RF-DETR produces metrics.csv/log.txt, not ultralytics plots).
_YOLO_ARTIFACTS = (
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "results.csv",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _Tee(io.TextIOBase):
    """Tee stdout/stderr writes into both the console and logs.txt so the API,
    which tails logs.txt over SFTP, sees the full training output."""

    def __init__(self, origin: Any, logs_path: Path, lock: threading.Lock) -> None:
        self._origin = origin
        self._logs_path = logs_path
        self._lock = lock
        self._buffer = ""

    def write(self, data: str) -> int:
        try:
            self._origin.write(data)
            self._origin.flush()
        except Exception:  # noqa: BLE001
            pass
        self._buffer += data
        if "\n" in self._buffer or len(self._buffer) > 4096:
            self._flush_buffer()
        return len(data)

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        try:
            with self._lock:
                with self._logs_path.open("ab") as fh:
                    fh.write(self._buffer.encode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            pass
        self._buffer = ""

    def flush(self) -> None:
        self._flush_buffer()
        try:
            self._origin.flush()
        except Exception:  # noqa: BLE001
            pass


class _JobIO:
    """Owns the per-job files + background resource sampler."""

    def __init__(self, job_dir: Path) -> None:
        self.job_dir = job_dir
        self.logs_path = job_dir / "logs.txt"
        self.progress_path = job_dir / "progress.jsonl"
        self.resources_path = job_dir / "resources.jsonl"
        self.status_path = job_dir / "status.json"
        self.cancel_path = job_dir / "cancel.json"
        self.artifacts_dir = job_dir / "artifacts"
        self.log_lock = threading.Lock()
        self._resources_stop = threading.Event()
        self._resources_thread: Optional[threading.Thread] = None
        job_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_path.touch()

    def cancel_requested(self) -> bool:
        return self.cancel_path.exists()

    def write_status(self, status: str, **extra: Any) -> None:
        blob = json.dumps({"status": status, **extra}, ensure_ascii=True).encode("utf-8")
        tmp = self.status_path.with_suffix(".json.tmp")
        with tmp.open("wb") as fh:
            fh.write(blob)
        tmp.replace(self.status_path)

    def write_progress(self, current_epoch: int, total_epochs: int, metrics: Optional[dict[str, Any]] = None) -> None:
        line = json.dumps(
            {"current_epoch": current_epoch, "total_epochs": total_epochs, "metrics": metrics or {}},
            ensure_ascii=True,
        )
        with self.progress_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def start_resource_sampler(self) -> None:
        self._resources_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._resources_thread.start()

    def stop_resource_sampler(self) -> None:
        self._resources_stop.set()
        if self._resources_thread:
            self._resources_thread.join(timeout=3.0)

    def _sample_loop(self) -> None:
        try:
            import psutil

            psutil.cpu_percent(interval=None)  # prime
        except Exception:  # noqa: BLE001
            psutil = None  # type: ignore[assignment]
        while not self._resources_stop.wait(5.0):
            try:
                sample = _sample_resources(psutil)
                with self.resources_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(sample, ensure_ascii=True) + "\n")
            except Exception:  # noqa: BLE001
                pass


def _sample_resources(psutil: Any) -> dict[str, Any]:
    sample: dict[str, Any] = {
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
        "sampled_at": _now(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
            used = max(0, total - free)
            sample["vram_used_gb"] = round(used / _BYTES_PER_GB, 2)
            sample["vram_total_gb"] = round(total / _BYTES_PER_GB, 2)
            if total > 0:
                sample["vram_used_pct"] = round(used / total * 100.0, 1)
    except Exception:  # noqa: BLE001
        pass
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            sample["ram_used_gb"] = round(vm.used / _BYTES_PER_GB, 2)
            sample["ram_total_gb"] = round(vm.total / _BYTES_PER_GB, 2)
            sample["ram_used_pct"] = round(vm.percent, 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            sample["cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            du = psutil.disk_usage("/")
            sample["disk_used_gb"] = round(du.used / _BYTES_PER_GB, 2)
            sample["disk_total_gb"] = round(du.total / _BYTES_PER_GB, 2)
            sample["disk_used_pct"] = round(du.percent, 1)
        except Exception:  # noqa: BLE001
            pass
    return sample


def _install_log_capture(jio: _JobIO) -> tuple[Any, Any, Optional[logging.Handler]]:
    """Route stdout/stderr + the ultralytics loggers into logs.txt."""
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, jio.logs_path, jio.log_lock)
    sys.stderr = _Tee(original_stderr, jio.logs_path, jio.log_lock)
    file_handler: Optional[logging.Handler] = None
    try:
        file_handler = logging.FileHandler(str(jio.logs_path), mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        for name in ("", "ultralytics", "ultralytics.utils"):
            lg = logging.getLogger(name)
            lg.addHandler(file_handler)
            if lg.level == logging.NOTSET or lg.level > logging.INFO:
                lg.setLevel(logging.INFO)
        logging.captureWarnings(True)
    except Exception:  # noqa: BLE001
        file_handler = None
    return original_stdout, original_stderr, file_handler


def _restore_log_capture(original_stdout: Any, original_stderr: Any, file_handler: Optional[logging.Handler]) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    sys.stdout, sys.stderr = original_stdout, original_stderr
    if file_handler is not None:
        for name in ("", "ultralytics", "ultralytics.utils"):
            lg = logging.getLogger(name)
            if file_handler in lg.handlers:
                lg.removeHandler(file_handler)
        try:
            file_handler.close()
        except Exception:  # noqa: BLE001
            pass


# ──────────────────────────────────────────────────────────────────────────
# YOLO
# ──────────────────────────────────────────────────────────────────────────


def _train_yolo(jio: _JobIO, config: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    dataset_dir = jio.job_dir / "dataset"
    data_yaml = _resolve_data_yaml(dataset_dir)

    runs_dir = jio.job_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    total_epochs = int(config.get("epochs") or 50)

    def _emit(trainer: Any) -> None:
        try:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            metrics = getattr(trainer, "metrics", {}) or {}
            jio.write_progress(
                epoch,
                total_epochs,
                {
                    "precision": _safe_float(metrics.get("metrics/precision(B)")),
                    "recall": _safe_float(metrics.get("metrics/recall(B)")),
                    "map50": _safe_float(metrics.get("metrics/mAP50(B)")),
                    "map": _safe_float(metrics.get("metrics/mAP50-95(B)")),
                },
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[progress] callback error: {exc}", flush=True)
        # Cooperative stop & keep: end training after this epoch.
        if jio.cancel_requested():
            try:
                trainer.stop = True
            except Exception:  # noqa: BLE001
                pass

    model = YOLO(config["base_model"])
    model.add_callback("on_fit_epoch_end", _emit)
    model.add_callback("on_train_epoch_end", _emit)

    device = 0
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:  # noqa: BLE001
        device = "cpu"

    train_args: dict[str, Any] = {
        "data": str(data_yaml),
        "project": str(runs_dir),
        "name": "train",
        "exist_ok": True,
        "epochs": total_epochs,
        "imgsz": int(config.get("imgsz") or 640),
        "batch": int(config.get("batch") or 16),
        "workers": int(config.get("workers") or 2),
        "device": device,
        "verbose": True,
    }
    reserved = set(train_args.keys())
    for key, value in (config.get("hyperparams") or {}).items():
        if key not in reserved and value is not None:
            train_args[key] = value

    print(f"[pod_train] yolo train args: {json.dumps(train_args, default=str)}", flush=True)
    results = model.train(**train_args)

    save_dir = Path(getattr(results, "save_dir", "") or "")
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError("best.pt was not produced by ultralytics")
    shutil.copy2(best_pt, jio.job_dir / "weights.pt")

    for name in _YOLO_ARTIFACTS:
        source = save_dir / name
        if source.exists():
            try:
                shutil.copy2(source, jio.artifacts_dir / name)
            except Exception:  # noqa: BLE001
                pass

    raw = getattr(results, "results_dict", {}) or {}
    return {
        "precision": _safe_float(raw.get("metrics/precision(B)") or raw.get("metrics/precision")),
        "recall": _safe_float(raw.get("metrics/recall(B)") or raw.get("metrics/recall")),
        "map50": _safe_float(raw.get("metrics/mAP50(B)") or raw.get("metrics/mAP50")),
        "map": _safe_float(raw.get("metrics/mAP50-95(B)") or raw.get("metrics/mAP50-95")),
    }


def _resolve_data_yaml(dataset_dir: Path) -> Path:
    """Locate data.yaml (tolerating a single nested wrapper dir) and rewrite its
    `path` to the absolute dataset dir so ultralytics resolves splits correctly."""
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        children = [entry for entry in dataset_dir.iterdir() if entry.is_dir()]
        if len(children) == 1 and (children[0] / "data.yaml").exists():
            nested = children[0]
            for entry in nested.iterdir():
                shutil.move(str(entry), str(dataset_dir / entry.name))
            nested.rmdir()
    if not data_yaml.exists():
        raise RuntimeError("data.yaml missing in the dataset")
    try:
        import yaml

        with data_yaml.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        cfg["path"] = str(dataset_dir.resolve())
        with data_yaml.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    except Exception:  # noqa: BLE001
        pass
    return data_yaml


# ──────────────────────────────────────────────────────────────────────────
# RF-DETR — reuse the canonical trainer in app.services.rfdetr_training
# ──────────────────────────────────────────────────────────────────────────


def _train_rfdetr(jio: _JobIO, config: dict[str, Any]) -> dict[str, Any]:
    from app.services import rfdetr_training

    total_epochs = int(config.get("epochs") or 50)

    def _on_epoch_end(epoch: int) -> None:
        # rfdetr's per-epoch metrics live in metrics.csv; surface epoch progress
        # here and let the final metrics + the metrics.csv artifact carry the rest.
        jio.write_progress(epoch, total_epochs, None)
        # Translate the API's cancel marker into the stop flag run() watches.
        if jio.cancel_requested():
            try:
                (jio.job_dir / "stop.flag").touch()
            except Exception:  # noqa: BLE001
                pass

    # run() expects a few descriptive fields for its training-metadata.json that
    # the Pod config doesn't carry — fill harmless defaults (the Pod registry dir
    # is throwaway; the API pulls weights.pth off the job dir).
    config.setdefault("job_id", str(jio.job_dir.name))
    config.setdefault("project_name", "")
    config.setdefault("user_id", "")
    config.setdefault("split", {})

    result = rfdetr_training.run(config, jio.job_dir, on_epoch_end=_on_epoch_end)

    # Stage the produced files where the API expects them: weights.pth in the job
    # root, metrics.csv/log.txt under artifacts/.
    for name, src_path in result.get("uploads", []):
        src = Path(src_path)
        if not src.exists():
            continue
        dst = jio.job_dir / "weights.pth" if name == "weights.pth" else jio.artifacts_dir / name
        try:
            shutil.copy2(src, dst)
        except Exception:  # noqa: BLE001
            pass

    metrics = result.get("metrics") or {}
    return {
        "precision": _safe_float(metrics.get("precision")),
        "recall": _safe_float(metrics.get("recall")),
        "map50": _safe_float(metrics.get("map50")),
        "map": _safe_float(metrics.get("map")),
    }


# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one RunPod-Pod training job from a job dir")
    parser.add_argument("--job-dir", required=True, help="Directory holding config.json + dataset/")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    jio = _JobIO(job_dir)
    original_stdout, original_stderr, file_handler = _install_log_capture(jio)
    jio.start_resource_sampler()

    try:
        config_path = job_dir / "config.json"
        if not config_path.exists():
            raise RuntimeError("config.json not found in the job dir")
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)

        jio.write_status("running", started_at=time.time())
        architecture = str(config.get("architecture") or "yolo").lower()
        print(f"[pod_train] starting {architecture} job in {job_dir}", flush=True)

        if architecture == "rfdetr":
            metrics = _train_rfdetr(jio, config)
        else:
            metrics = _train_yolo(jio, config)

        # A cancel marker present at the end means we stopped early but kept the
        # checkpoint — finalize as a cancellation so the API treats it as stop&keep.
        terminal = "cancelled" if jio.cancel_requested() else "completed"
        jio.write_status(terminal, completed_at=time.time(), metrics=metrics)
        print(f"[pod_train] job {terminal}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        jio.write_status("failed", error=str(exc), completed_at=time.time())
        return 1
    finally:
        jio.stop_resource_sampler()
        _restore_log_capture(original_stdout, original_stderr, file_handler)


if __name__ == "__main__":
    raise SystemExit(main())
