import argparse
import csv
import json
import shutil
import sys
import traceback
from pathlib import Path

from ultralytics import YOLO

from app.config import YOLO_MODELS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a YOLO training job")
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    config_path = job_dir / "config.json"
    metrics_path = job_dir / "metrics.json"
    artifacts_path = job_dir / "artifacts.json"
    error_path = job_dir / "error.json"

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    dataset_dir = job_dir / "dataset"
    data_yaml_path = dataset_dir / "data.yaml"
    runs_dir = job_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_model = resolve_base_model(config["base_model"])
        print(f"[training] loading model: {base_model}", flush=True)
        model = YOLO(base_model)

        train_args: dict[str, object] = {
            "data": str(data_yaml_path),
            "project": str(runs_dir),
            "name": "train",
            "exist_ok": True,
            "epochs": int(config["epochs"]),
            "imgsz": int(config.get("imgsz") or 640),
            "batch": int(config.get("batch") or 16),
            "workers": int(config["workers"]) if config.get("workers") is not None else 8,
            "verbose": True,
        }

        if config.get("device") and config["device"] != "auto":
            train_args["device"] = config["device"]

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
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, ensure_ascii=True, indent=2)
        if any(value is None for value in metrics.values()):
            print(f"[training] WARNING: some metrics are null after normalization: {metrics}", flush=True)

        artifacts = {
            "best_weights_path": str(registered_weights),
            "results_csv_path": str(Path(results.save_dir) / "results.csv"),
        }
        with artifacts_path.open("w", encoding="utf-8") as artifacts_file:
            json.dump(artifacts, artifacts_file, ensure_ascii=True, indent=2)

        print(f"[training] registered model at {registered_weights}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        with error_path.open("w", encoding="utf-8") as error_file:
            json.dump({"error": str(exc)}, error_file, ensure_ascii=True, indent=2)
        traceback.print_exc()
        return 1


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


if __name__ == "__main__":
    sys.exit(main())
