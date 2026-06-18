import argparse
import csv
import json
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, UTC
from pathlib import Path

from ultralytics import YOLO, settings as ultralytics_settings

from app.config import YOLO_MODELS_DIR

# This project does not use MLflow, but ultralytics auto-registers its MLflow
# callback whenever the package is installed, and mlflow>=3 raises on the
# default file-store backend — which aborts training before the first epoch.
ultralytics_settings.update({"mlflow": False})
from app.services import training_callback
from app.services.resource_sampler import ResourceSampler


_RESERVED_TRAIN_ARGS = frozenset(
    {"data", "project", "name", "exist_ok", "epochs", "imgsz", "batch", "workers", "device", "verbose"}
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a YOLO training job")
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    config_path = job_dir / "config.json"
    metrics_path = job_dir / "metrics.json"
    artifacts_path = job_dir / "artifacts.json"
    error_path = job_dir / "error.json"
    logs_path = job_dir / "logs.txt"

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    callback_url = config.get("external_callback_url")
    callback_secret = config.get("external_callback_secret")

    log_streamer_stop = threading.Event()
    log_streamer = None
    if callback_url and callback_secret:
        log_streamer = threading.Thread(
            target=_stream_logs_to_callback,
            args=(logs_path, callback_url, callback_secret, log_streamer_stop),
            daemon=True,
        )
        log_streamer.start()

    training_callback.post_progress(
        callback_url,
        callback_secret,
        {"status": "running", "started_at": _now_iso()},
    )

    resources_path = job_dir / "resources.jsonl"

    def _on_resource_sample(sample: dict) -> None:
        try:
            with resources_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample, ensure_ascii=True) + "\n")
        except OSError as exc:
            print(f"[training-resources] append error: {exc}", flush=True)
        training_callback.post_progress_async(
            callback_url,
            callback_secret,
            {"status": "running", "resources": sample},
        )

    resource_sampler = ResourceSampler(_on_resource_sample, interval=5.0)
    resource_sampler.start()

    try:
        total_epochs = int(config["epochs"])

        def _on_epoch_end(epoch: int) -> None:
            update_metadata_progress(job_dir, current_epoch=epoch, total_epochs=total_epochs)
            training_callback.post_progress(
                callback_url,
                callback_secret,
                {"status": "running", "current_epoch": epoch, "total_epochs": total_epochs},
            )

        architecture = str(config.get("architecture") or "yolo").strip().lower()
        if architecture == "rfdetr":
            from app.services import rfdetr_training

            outcome = rfdetr_training.run(config, job_dir, on_epoch_end=_on_epoch_end)
        else:
            outcome = _run_yolo_training(config, job_dir, on_epoch_end=_on_epoch_end)

        metrics = outcome["metrics"]
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, ensure_ascii=True, indent=2)
        if any(value is None for value in metrics.values()):
            print(f"[training] WARNING: some metrics are null after normalization: {metrics}", flush=True)

        with artifacts_path.open("w", encoding="utf-8") as artifacts_file:
            json.dump(outcome["artifacts"], artifacts_file, ensure_ascii=True, indent=2)

        resource_sampler.stop()

        log_streamer_stop.set()
        if log_streamer:
            log_streamer.join(timeout=3)
        _flush_remaining_logs(logs_path, callback_url, callback_secret)

        # A cooperative "stop & keep" (the API dropped stop.flag) finishes the current
        # epoch and registers the best checkpoint exactly like a full run, but the job
        # is reported as cancelled (stopped early) — keeping the produced model.
        stopped_early = (job_dir / "stop.flag").exists()
        progress_payload = {
            "status": "cancelled" if stopped_early else "completed",
            "completed_at": _now_iso(),
            "produced_model_name": config["output_model_name"],
            "total_epochs": total_epochs,
            "metrics": metrics,
        }
        # A full run reports the final epoch; a stop-&-keep leaves current_epoch at the
        # last epoch already reported per-epoch, so the UI shows where it stopped.
        if not stopped_early:
            progress_payload["current_epoch"] = total_epochs
        progress_ok = training_callback.post_progress(callback_url, callback_secret, progress_payload)

        artifacts_ok = _upload_artifacts(callback_url, callback_secret, outcome["uploads"])

        if stopped_early:
            # Keep job_dir so _monitor_process still sees stop.flag (and the metadata)
            # when it finalizes the job as cancelled-with-model.
            print(f"[training] stop & keep: keeping {job_dir} for finalization", flush=True)
        elif progress_ok and artifacts_ok:
            shutil.rmtree(job_dir, ignore_errors=True)
            print(f"[training] cleaned up {job_dir}", flush=True)
        else:
            print(
                f"[training] keeping {job_dir} (callback/artifact upload incomplete)",
                flush=True,
            )

        return 0
    except Exception as exc:  # noqa: BLE001
        with error_path.open("w", encoding="utf-8") as error_file:
            json.dump({"error": str(exc)}, error_file, ensure_ascii=True, indent=2)
        traceback.print_exc()

        resource_sampler.stop()

        log_streamer_stop.set()
        if log_streamer:
            log_streamer.join(timeout=3)
        _flush_remaining_logs(logs_path, callback_url, callback_secret)

        training_callback.post_progress(
            callback_url,
            callback_secret,
            {
                "status": "failed",
                "completed_at": _now_iso(),
                "error": str(exc),
            },
        )
        return 1
    finally:
        log_streamer_stop.set()
        resource_sampler.stop()


def _stream_logs_to_callback(
    logs_path: Path,
    callback_url: str,
    callback_secret: str,
    stop_event: threading.Event,
    poll_seconds: float = 1.5,
) -> None:
    position = 0
    while not stop_event.is_set():
        try:
            if logs_path.exists():
                with logs_path.open("rb") as fh:
                    fh.seek(position)
                    chunk = fh.read()
                    position = fh.tell()
                if chunk:
                    training_callback.post_log_chunk(
                        callback_url, callback_secret, chunk.decode("utf-8", errors="replace")
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"[training-logs] streaming error: {exc}", flush=True)
        if stop_event.wait(poll_seconds):
            return


def _flush_remaining_logs(logs_path: Path, callback_url, callback_secret) -> None:
    if not callback_url or not callback_secret or not logs_path.exists():
        return
    try:
        with logs_path.open("r", encoding="utf-8", errors="replace") as fh:
            full = fh.read()
        if full:
            training_callback.post_log_chunk(callback_url, callback_secret, full)
    except OSError as exc:
        print(f"[training-logs] flush error: {exc}", flush=True)


def _run_yolo_training(config: dict, job_dir: Path, *, on_epoch_end) -> dict:
    dataset_dir = job_dir / "dataset"
    data_yaml_path = dataset_dir / "data.yaml"
    runs_dir = job_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_model = resolve_base_model(config["base_model"])
    print(f"[training] loading model: {base_model}", flush=True)
    model = YOLO(base_model)

    total_epochs = int(config["epochs"])

    stop_flag_path = job_dir / "stop.flag"

    def _on_train_epoch_end(trainer) -> None:  # type: ignore[no-untyped-def]
        try:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
        except Exception:  # noqa: BLE001
            return
        on_epoch_end(epoch)
        # Cooperative "stop & keep": ask ultralytics to stop after this epoch so the
        # final validation runs, best.pt is saved, and the model registers normally.
        if stop_flag_path.exists():
            print("[training] stop.flag detected — stopping after this epoch", flush=True)
            trainer.stop = True

    model.add_callback("on_train_epoch_end", _on_train_epoch_end)

    train_args: dict[str, object] = {
        "data": str(data_yaml_path),
        "project": str(runs_dir),
        "name": "train",
        "exist_ok": True,
        "epochs": total_epochs,
        "imgsz": int(config.get("imgsz") or 640),
        "batch": int(config.get("batch") or 16),
        "workers": int(config["workers"]) if config.get("workers") is not None else 8,
        "verbose": True,
    }

    if config.get("device") and config["device"] != "auto":
        train_args["device"] = config["device"]

    hyperparams = config.get("hyperparams") or {}
    if isinstance(hyperparams, dict) and hyperparams:
        for key, value in hyperparams.items():
            if key in _RESERVED_TRAIN_ARGS:
                continue
            if value is None:
                continue
            train_args[key] = value

    print(f"[training] args: {json.dumps(train_args, default=str)}", flush=True)
    results = model.train(**train_args)

    best_weights = resolve_best_weights_path(results)
    if best_weights is None or not best_weights.exists():
        raise RuntimeError("Unable to locate best.pt after training")

    model_dir = YOLO_MODELS_DIR / config["output_model_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    registered_weights = model_dir / "weights.pt"
    shutil.copy2(best_weights, registered_weights)

    with (model_dir / "classes.json").open("w", encoding="utf-8") as classes_file:
        json.dump(config["classes"], classes_file, ensure_ascii=True, indent=2)

    with (model_dir / "training-metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "job_id": config["job_id"],
                "project_name": config["project_name"],
                "user_id": config["user_id"],
                "base_model": config["base_model"],
                "output_model_name": config["output_model_name"],
                "epochs": config["epochs"],
                "imgsz": config.get("imgsz"),
                "batch": config.get("batch"),
                "device": config.get("device"),
                "split": config["split"],
                "classes": config["classes"],
            },
            metadata_file,
            ensure_ascii=True,
            indent=2,
        )

    metrics = normalize_metrics(results)

    results_save_dir = Path(getattr(results, "save_dir", "") or runs_dir / "train")
    results_csv_path = results_save_dir / "results.csv"

    uploads: list[tuple[str, Path]] = [("weights.pt", registered_weights)]
    plot_filenames = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "results.csv",
    ]
    for filename in plot_filenames:
        candidate = results_save_dir / filename
        if candidate.exists():
            uploads.append((filename, candidate))

    print(f"[training] registered model at {registered_weights}", flush=True)

    return {
        "metrics": metrics,
        "artifacts": {
            "best_weights_path": str(registered_weights),
            "results_csv_path": str(results_csv_path),
        },
        "uploads": uploads,
    }


def _upload_artifacts(callback_url, callback_secret, uploads: list[tuple[str, Path]]) -> bool:
    if not callback_url or not callback_secret:
        return True

    all_ok = True
    for filename, path in uploads:
        ok = training_callback.put_artifact(callback_url, callback_secret, filename, path)
        all_ok = all_ok and ok
    return all_ok


def update_metadata_progress(job_dir: Path, *, current_epoch: int, total_epochs: int) -> None:
    """Merge epoch progress into metadata.json so the UI progress bar ticks up."""
    metadata_path = job_dir / "metadata.json"
    if not metadata_path.exists():
        return
    try:
        with metadata_path.open("r", encoding="utf-8") as source:
            metadata = json.load(source)
    except (OSError, json.JSONDecodeError):
        return

    metadata["current_epoch"] = current_epoch
    metadata["total_epochs"] = total_epochs

    tmp_path = metadata_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=True, indent=2)
    tmp_path.replace(metadata_path)


def resolve_base_model(base_model: str) -> str:
    local_model_dir = YOLO_MODELS_DIR / base_model
    if local_model_dir.exists():
        weights_candidates = sorted(local_model_dir.glob("weights.*"))
        if weights_candidates:
            return str(weights_candidates[0])
    return base_model


def resolve_best_weights_path(results) -> Path | None:
    save_dir = Path(getattr(results, "save_dir", ""))
    if save_dir:
        candidate = save_dir / "weights" / "best.pt"
        if candidate.exists():
            return candidate

    trainer = getattr(results, "trainer", None)
    best_attr = getattr(trainer, "best", None)
    if best_attr:
        candidate = Path(best_attr)
        if candidate.exists():
            return candidate

    return None


def normalize_metrics(results) -> dict[str, float | None]:
    raw_metrics = getattr(results, "results_dict", {}) or {}
    print(f"[training] results_dict keys: {list(raw_metrics.keys())}", flush=True)

    metrics: dict[str, float | None] = {
        "precision": safe_float(raw_metrics.get("metrics/precision(B)") or raw_metrics.get("metrics/precision")),
        "recall": safe_float(raw_metrics.get("metrics/recall(B)") or raw_metrics.get("metrics/recall")),
        "map50": safe_float(raw_metrics.get("metrics/mAP50(B)") or raw_metrics.get("metrics/mAP50")),
        "map": safe_float(raw_metrics.get("metrics/mAP50-95(B)") or raw_metrics.get("metrics/mAP50-95")),
    }

    if not all(value is not None for value in metrics.values()):
        fallback = read_metrics_from_csv(Path(getattr(results, "save_dir", "") or "") / "results.csv")
        if fallback:
            print(f"[training] filling missing metrics from results.csv: {fallback}", flush=True)
            for key, value in fallback.items():
                if metrics[key] is None and value is not None:
                    metrics[key] = value

    return metrics


def read_metrics_from_csv(csv_path: Path) -> dict[str, float | None] | None:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
    except OSError:
        return None
    if not rows:
        return None

    key_map = {
        "precision": ("metrics/precision(B)", "metrics/precision"),
        "recall": ("metrics/recall(B)", "metrics/recall"),
        "map50": ("metrics/mAP50(B)", "metrics/mAP50"),
        "map": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }

    for row in reversed(rows):
        candidate = {
            metric: safe_float(next((row[col] for col in columns if col in row and row[col] != ""), None))
            for metric, columns in key_map.items()
        }
        if any(value is not None for value in candidate.values()):
            return candidate
    return None


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    sys.exit(main())
