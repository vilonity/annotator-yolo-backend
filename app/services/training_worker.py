import argparse
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
            "verbose": True,
        }

        if config.get("imgsz"):
            train_args["imgsz"] = int(config["imgsz"])
        if config.get("batch"):
            train_args["batch"] = int(config["batch"])
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
    return {
        "precision": safe_float(raw_metrics.get("metrics/precision(B)") or raw_metrics.get("metrics/precision")),
        "recall": safe_float(raw_metrics.get("metrics/recall(B)") or raw_metrics.get("metrics/recall")),
        "map50": safe_float(raw_metrics.get("metrics/mAP50(B)") or raw_metrics.get("metrics/mAP50")),
        "map": safe_float(raw_metrics.get("metrics/mAP50-95(B)") or raw_metrics.get("metrics/mAP50-95")),
    }


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(main())
