import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import HTTPException, UploadFile
from ultralytics import YOLO

from app.config import YOLO_MODELS_DIR
from app.schemas.yolo import YoloModelInfo, UploadModelResponse, AutoAnnotateRequest
from app.services.image_service import load_image_from_url

SUPPORTED_WEIGHTS_EXTS = {".pt", ".onnx"}
SUPPORTED_CLASSES_NAMES = {"classes.json", "classes.txt", "classes.yaml", "classes.yml"}


class YoloService:
    _cache: dict[str, YOLO] = {}

    @classmethod
    def get_model(cls, name: str) -> tuple[YOLO, list[str]]:
        if name in cls._cache:
            model_dir = YOLO_MODELS_DIR / name
            classes_file = model_dir / "classes.json"
            if not model_dir.exists() or not classes_file.exists():
                cls._cache.pop(name, None)
                raise HTTPException(status_code=404, detail="YOLO model not found")
            with classes_file.open() as f:
                classes_list = json.load(f)
            return cls._cache[name], classes_list

        model_dir = YOLO_MODELS_DIR / name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="YOLO model not found")

        weights_files = list(model_dir.glob("weights.*"))
        if not weights_files:
            raise HTTPException(status_code=404, detail="Model weights file not found")

        weights_path = weights_files[0]

        try:
            model = YOLO(str(weights_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load YOLO model: {exc}") from exc

        cls._cache[name] = model

        classes_file = model_dir / "classes.json"
        with classes_file.open() as f:
            classes_list = json.load(f)

        return model, classes_list

    @classmethod
    async def upload_model(
        cls,
        name: str,
        weights_file: UploadFile,
        classes_file: UploadFile,
    ) -> UploadModelResponse:
        cls._ensure_new_model(name)

        weights_ext = Path(weights_file.filename or "").suffix.lower()
        if weights_ext not in SUPPORTED_WEIGHTS_EXTS:
            raise HTTPException(status_code=400, detail="Weights must be a .pt or .onnx file")

        weights_bytes = await weights_file.read()
        classes_text = (await classes_file.read()).decode("utf-8")
        classes_list = cls._parse_classes_text(classes_text)

        return cls._store_model(name, weights_bytes, weights_ext, classes_list)

    @classmethod
    async def upload_model_archive(cls, name: str, archive_file: UploadFile) -> UploadModelResponse:
        cls._ensure_new_model(name)

        raw = await archive_file.read()
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Archive is not a valid ZIP file") from exc

        weights_ext: Optional[str] = None
        weights_bytes: Optional[bytes] = None
        classes_text: Optional[str] = None

        for info in archive.infolist():
            if info.is_dir():
                continue
            entry_name = Path(info.filename).name.lower()
            entry_ext = Path(entry_name).suffix

            if weights_bytes is None and entry_name.startswith("weights") and entry_ext in SUPPORTED_WEIGHTS_EXTS:
                weights_bytes = archive.read(info)
                weights_ext = entry_ext
            elif classes_text is None and entry_name in SUPPORTED_CLASSES_NAMES:
                classes_text = archive.read(info).decode("utf-8")

        if weights_bytes is None or weights_ext is None:
            raise HTTPException(
                status_code=400,
                detail="Archive must contain weights.pt or weights.onnx",
            )
        if classes_text is None:
            raise HTTPException(
                status_code=400,
                detail="Archive must contain classes.json (or classes.txt / classes.yaml)",
            )

        classes_list = cls._parse_classes_text(classes_text)
        return cls._store_model(name, weights_bytes, weights_ext, classes_list)

    @classmethod
    def get_model_files(cls, name: str) -> tuple[Path, Path]:
        model_dir = YOLO_MODELS_DIR / name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        weights_candidates = list(model_dir.glob("weights.*"))
        if not weights_candidates:
            raise HTTPException(status_code=404, detail="Model weights file not found")

        classes_path = model_dir / "classes.json"
        if not classes_path.exists():
            raise HTTPException(status_code=404, detail="Model classes file not found")

        return weights_candidates[0], classes_path

    @classmethod
    def _ensure_new_model(cls, name: str) -> None:
        cls._validate_model_name(name)
        model_dir = YOLO_MODELS_DIR / name
        if model_dir.exists():
            raise HTTPException(status_code=400, detail="Model with this name already exists")

    @classmethod
    def _validate_model_name(cls, name: str) -> None:
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid model name")

    @classmethod
    def _parse_classes_text(cls, raw_classes_text: str) -> List[str]:
        classes_list: Optional[List[str]] = None

        try:
            parsed_yaml = yaml.safe_load(raw_classes_text)
        except Exception:
            parsed_yaml = None

        if isinstance(parsed_yaml, list):
            classes_list = [str(n) for n in parsed_yaml]
        elif isinstance(parsed_yaml, dict) and "names" in parsed_yaml:
            names_field = parsed_yaml["names"]
            if isinstance(names_field, list):
                classes_list = [str(n) for n in names_field]
            elif isinstance(names_field, dict):
                try:
                    ordered_keys = sorted(names_field, key=lambda k: int(k))
                except Exception:
                    ordered_keys = sorted(names_field)
                classes_list = [str(names_field[k]) for k in ordered_keys]

        if classes_list is None:
            classes_list = [line.strip() for line in raw_classes_text.splitlines() if line.strip()]

        if not classes_list:
            raise HTTPException(status_code=400, detail="No class names found in classes file")

        return classes_list

    @classmethod
    def _store_model(
        cls,
        name: str,
        weights_bytes: bytes,
        weights_ext: str,
        classes_list: List[str],
    ) -> UploadModelResponse:
        model_dir = YOLO_MODELS_DIR / name
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            weights_storage_path = model_dir / f"weights{weights_ext}"
            with weights_storage_path.open("wb") as buffer:
                buffer.write(weights_bytes)

            classes_file_path = model_dir / "classes.json"
            with classes_file_path.open("w", encoding="utf-8") as f:
                json.dump(classes_list, f, ensure_ascii=False)
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

        return UploadModelResponse(
            name=name,
            classes=classes_list,
            message="Model uploaded successfully",
        )

    @classmethod
    def list_models(cls) -> List[YoloModelInfo]:
        models = []
        for model_dir in YOLO_MODELS_DIR.iterdir():
            if model_dir.is_dir():
                classes_file = model_dir / "classes.json"
                if classes_file.exists():
                    with classes_file.open() as f:
                        classes = json.load(f)
                    weights_files = list(model_dir.glob("weights.*"))
                    size_bytes = weights_files[0].stat().st_size if weights_files else 0
                    models.append(YoloModelInfo(
                        name=model_dir.name,
                        classes=classes,
                        date_add=datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
                        size_bytes=size_bytes,
                    ))
        return models

    @classmethod
    def delete_model(cls, model_name: str) -> None:
        model_dir = YOLO_MODELS_DIR / model_name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        cls._cache.pop(model_name, None)
        shutil.rmtree(model_dir)

    @classmethod
    def rename_model(cls, model_name: str, new_name: str) -> YoloModelInfo:
        cls._validate_model_name(new_name)
        source_dir = YOLO_MODELS_DIR / model_name
        if not source_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        target_dir = YOLO_MODELS_DIR / new_name
        if target_dir.exists():
            raise HTTPException(status_code=400, detail="Model with this name already exists")

        cls._cache.pop(model_name, None)
        source_dir.rename(target_dir)

        classes_file = target_dir / "classes.json"
        with classes_file.open() as f:
            classes = json.load(f)
        weights_files = list(target_dir.glob("weights.*"))
        size_bytes = weights_files[0].stat().st_size if weights_files else 0
        return YoloModelInfo(
            name=new_name,
            classes=classes,
            date_add=datetime.fromtimestamp(target_dir.stat().st_mtime).isoformat(),
            size_bytes=size_bytes,
        )

    @classmethod
    def run_inference(
        cls,
        model_name: str,
        payload: AutoAnnotateRequest,
    ) -> list[list[dict]]:
        if not payload.image_urls:
            raise HTTPException(status_code=400, detail="image_urls list cannot be empty")

        model, class_names = cls.get_model(model_name)

        results = []
        for url in payload.image_urls:
            img = load_image_from_url(url)

            kwargs = dict(
                source=img,
                conf=payload.conf_threshold or 0.25,
                save=False,
                verbose=False,
            )
            if payload.imgsz is not None:
                kwargs["imgsz"] = payload.imgsz
            try:
                pred = model.predict(**kwargs)
                results.append(pred[0])
            except Exception as exc:
                if payload.imgsz is None:
                    try:
                        pred = model.predict(**{**kwargs, "imgsz": 320})
                        results.append(pred[0])
                        continue
                    except Exception:
                        pass
                raise HTTPException(status_code=500, detail=f"YOLO inference failed: {exc}") from exc

        all_annotations = []
        for res in results:
            boxes = res.boxes
            annotations_list = []
            for xyxy, cls_idx, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                x1, y1, x2, y2 = xyxy
                raw_name = class_names[int(cls_idx)] if int(cls_idx) < len(class_names) else str(cls_idx)
                mapped_name = payload.class_map.get(raw_name) if payload.class_map else raw_name
                annotations_list.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class_id": int(cls_idx),
                        "class_name": mapped_name,
                        "confidence": float(conf),
                        "model": model_name,
                    }
                )
            all_annotations.append(annotations_list)

        return all_annotations
