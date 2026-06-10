"""RF-DETR training path for the training worker subprocess.

The staged dataset zip is always YOLO-format (data.yaml + images/ + labels/)
because the central API builds one zip for every architecture. rfdetr's own
YOLO loader expects a different, Roboflow-style layout, so the dataset is
converted to COCO JSON (train/valid/test dirs with _annotations.coco.json) —
the canonical rfdetr input format — before training starts.

Category ids are 0-based and match the index into the job's classes list, so
inference-time ``class_id`` values index straight into ``classes.json`` —
same convention as the YOLO models.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from app.config import RFDETR_MODELS_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BASE_MODEL_VARIANTS = {
    "rfdetr-nano": "nano",
    "rfdetr-small": "small",
    "rfdetr-medium": "medium",
    "rfdetr-large": "large",
}

_VARIANT_CLASS_NAMES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

# rfdetr Model.train() kwargs that job hyperparams may override.
ALLOWED_HYPERPARAMS = frozenset(
    {
        "lr",
        "lr_encoder",
        "batch_size",
        "grad_accum_steps",
        "weight_decay",
        "warmup_epochs",
        "ema_decay",
        "resolution",
        "checkpoint_interval",
        "early_stopping",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "use_ema",
        "gradient_checkpointing",
    }
)

CHECKPOINT_PREFERENCE = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint.pth",
)


def run(
    config: dict[str, Any],
    job_dir: Path,
    *,
    on_epoch_end: Callable[[int], None],
) -> dict[str, Any]:
    try:
        import rfdetr
    except ImportError as exc:
        raise RuntimeError("rfdetr package is not installed on this server (pip install rfdetr)") from exc

    dataset_dir = job_dir / "dataset"
    coco_dir = job_dir / "dataset-coco"
    output_dir = job_dir / "runs" / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [str(name) for name in config.get("classes") or []]
    convert_yolo_dataset_to_coco(dataset_dir, coco_dir, classes)

    variant, pretrain_weights = resolve_base_model(str(config["base_model"]))
    class_name = _VARIANT_CLASS_NAMES[variant]
    model_class = getattr(rfdetr, class_name, None)
    if model_class is None:
        raise RuntimeError(f"Installed rfdetr package does not provide {class_name}")

    init_kwargs: dict[str, Any] = {}
    if pretrain_weights is not None:
        init_kwargs["pretrain_weights"] = str(pretrain_weights)
    device = str(config.get("device") or "auto").strip()
    if device and device != "auto":
        init_kwargs["device"] = device

    print(f"[training] loading RF-DETR {variant} (pretrain={pretrain_weights or 'COCO default'})", flush=True)
    model = model_class(**init_kwargs)

    _attach_epoch_callback(model, on_epoch_end)

    train_args: dict[str, Any] = {
        "dataset_dir": str(coco_dir),
        "epochs": int(config["epochs"]),
        "output_dir": str(output_dir),
    }
    if config.get("batch"):
        train_args["batch_size"] = int(config["batch"])

    hyperparams = config.get("hyperparams") or {}
    if isinstance(hyperparams, dict):
        # The shared training form sends YOLO's lr0; map it onto rfdetr's lr.
        if hyperparams.get("lr0") is not None and hyperparams.get("lr") is None:
            train_args["lr"] = hyperparams["lr0"]
        for key, value in hyperparams.items():
            if key in ALLOWED_HYPERPARAMS and value is not None:
                train_args[key] = value

    print(f"[training] rfdetr args: {json.dumps(train_args, default=str)}", flush=True)
    model.train(**train_args)

    best_weights = resolve_best_checkpoint(output_dir)
    if best_weights is None:
        raise RuntimeError("Unable to locate an RF-DETR checkpoint after training")

    model_dir = RFDETR_MODELS_DIR / config["output_model_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    registered_weights = model_dir / "weights.pth"
    shutil.copy2(best_weights, registered_weights)

    with (model_dir / "classes.json").open("w", encoding="utf-8") as classes_file:
        json.dump(classes, classes_file, ensure_ascii=True, indent=2)

    with (model_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump({"variant": variant}, metadata_file, ensure_ascii=True)

    with (model_dir / "training-metadata.json").open("w", encoding="utf-8") as training_metadata_file:
        json.dump(
            {
                "job_id": config["job_id"],
                "project_name": config["project_name"],
                "user_id": config["user_id"],
                "architecture": "rfdetr",
                "variant": variant,
                "base_model": config["base_model"],
                "output_model_name": config["output_model_name"],
                "epochs": config["epochs"],
                "batch": config.get("batch"),
                "device": config.get("device"),
                "split": config["split"],
                "classes": classes,
            },
            training_metadata_file,
            ensure_ascii=True,
            indent=2,
        )

    metrics = read_metrics_from_log(output_dir / "log.txt")
    print(f"[training] registered RF-DETR model at {registered_weights}", flush=True)

    return {
        "metrics": metrics,
        "artifacts": {"best_weights_path": str(registered_weights), "results_csv_path": None},
        "uploads": [("weights.pth", registered_weights)],
    }


def resolve_base_model(base_model: str) -> tuple[str, Optional[Path]]:
    """Map a job base_model onto (variant, pretrain weights path).

    Either a built-in alias ("rfdetr-medium" → COCO checkpoint downloaded by
    the package) or the name of a locally registered RF-DETR model to
    fine-tune from.
    """
    normalized = base_model.strip().lower()
    if normalized in BASE_MODEL_VARIANTS:
        return BASE_MODEL_VARIANTS[normalized], None

    local_model_dir = RFDETR_MODELS_DIR / base_model
    if local_model_dir.exists():
        weights_candidates = sorted(local_model_dir.glob("weights.*"))
        if weights_candidates:
            variant = "medium"
            metadata_path = local_model_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with metadata_path.open() as metadata_file:
                        stored = json.load(metadata_file).get("variant")
                    if stored in _VARIANT_CLASS_NAMES:
                        variant = stored
                except (OSError, json.JSONDecodeError):
                    pass
            return variant, weights_candidates[0]

    raise RuntimeError(
        f"Unknown RF-DETR base model '{base_model}' "
        f"(expected one of {', '.join(sorted(BASE_MODEL_VARIANTS))} or a local RF-DETR model name)"
    )


def _attach_epoch_callback(model: Any, on_epoch_end: Callable[[int], None]) -> None:
    callbacks = getattr(model, "callbacks", None)
    if not isinstance(callbacks, dict):
        print("[training] rfdetr model exposes no callbacks — epoch progress disabled", flush=True)
        return

    def _on_fit_epoch_end(log: Any) -> None:
        epoch = log.get("epoch") if isinstance(log, dict) else None
        try:
            on_epoch_end(int(epoch) + 1)
        except (TypeError, ValueError):
            return

    try:
        callbacks["on_fit_epoch_end"].append(_on_fit_epoch_end)
    except Exception as exc:  # noqa: BLE001
        print(f"[training] failed to attach rfdetr epoch callback: {exc}", flush=True)


def resolve_best_checkpoint(output_dir: Path) -> Optional[Path]:
    for filename in CHECKPOINT_PREFERENCE:
        candidate = output_dir / filename
        if candidate.exists():
            return candidate
    remaining = sorted(output_dir.glob("checkpoint*.pth"))
    return remaining[-1] if remaining else None


def read_metrics_from_log(log_path: Path) -> dict[str, float | None]:
    """Extract mAP metrics from rfdetr's DETR-style log.txt (one JSON dict per line).

    ``test_coco_eval_bbox`` holds the 12 COCO eval stats for the validation
    set; [0] is AP@0.50:0.95 and [1] is AP@0.50. The best epoch is reported to
    match the best-checkpoint selection. Per-box precision/recall are not part
    of COCO eval output, so they stay null.
    """
    metrics: dict[str, float | None] = {"precision": None, "recall": None, "map50": None, "map": None}
    if not log_path.exists():
        return metrics

    best_map: float | None = None
    best_map50: float | None = None
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stats = entry.get("test_coco_eval_bbox")
                if not isinstance(stats, list) or len(stats) < 2:
                    continue
                try:
                    current_map = float(stats[0])
                    current_map50 = float(stats[1])
                except (TypeError, ValueError):
                    continue
                if best_map is None or current_map > best_map:
                    best_map = current_map
                    best_map50 = current_map50
    except OSError:
        return metrics

    metrics["map"] = best_map
    metrics["map50"] = best_map50
    return metrics


def convert_yolo_dataset_to_coco(dataset_dir: Path, target_dir: Path, class_names: list[str]) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)

    split_mapping = {"train": "train", "val": "valid", "test": "test"}
    converted: set[str] = set()

    for source_split, target_split in split_mapping.items():
        images_dir = dataset_dir / "images" / source_split
        if not images_dir.is_dir():
            continue
        image_paths = sorted(
            entry for entry in images_dir.iterdir() if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS
        )
        if not image_paths:
            continue

        split_dir = target_dir / target_split
        split_dir.mkdir(parents=True, exist_ok=True)

        coco: dict[str, Any] = {
            "info": {"description": "Converted from YOLO format for RF-DETR training"},
            "licenses": [],
            "categories": [
                {"id": index, "name": name, "supercategory": "none"} for index, name in enumerate(class_names)
            ],
            "images": [],
            "annotations": [],
        }

        annotation_id = 1
        for image_id, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                width, height = image.size
            shutil.copy2(image_path, split_dir / image_path.name)
            coco["images"].append(
                {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
            )

            label_path = dataset_dir / "labels" / source_split / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            with label_path.open("r", encoding="utf-8") as label_file:
                for line in label_file:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_index = int(float(parts[0]))
                        center_x, center_y, box_w, box_h = (float(value) for value in parts[1:5])
                    except ValueError:
                        continue

                    abs_w = box_w * width
                    abs_h = box_h * height
                    x_min = max(0.0, center_x * width - abs_w / 2)
                    y_min = max(0.0, center_y * height - abs_h / 2)
                    abs_w = min(abs_w, width - x_min)
                    abs_h = min(abs_h, height - y_min)
                    if abs_w <= 0 or abs_h <= 0:
                        continue

                    coco["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_index,
                            "bbox": [x_min, y_min, abs_w, abs_h],
                            "area": abs_w * abs_h,
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )
                    annotation_id += 1

        with (split_dir / "_annotations.coco.json").open("w", encoding="utf-8") as annotations_file:
            json.dump(coco, annotations_file, ensure_ascii=True)
        converted.add(target_split)

    if "train" not in converted:
        raise RuntimeError("Dataset has no train images — cannot start RF-DETR training")
    if "valid" not in converted:
        raise RuntimeError("RF-DETR training requires a validation split (set val percent > 0)")
