import io
import json
import logging
import shutil
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import HTTPException, UploadFile
from ultralytics import YOLO

from app.config import YOLO_MODELS_DIR
from app.schemas.yolo import YoloModelInfo, UploadModelResponse, AutoAnnotateRequest
from app.services.image_service import free_inference_memory, load_image_from_url

logger = logging.getLogger(__name__)

SUPPORTED_WEIGHTS_EXTS = {".pt", ".onnx"}
SUPPORTED_CLASSES_NAMES = {"classes.json", "classes.txt", "classes.yaml", "classes.yml"}


class YoloService:
    _cache: dict[str, YOLO] = {}
    _export_locks: dict[str, threading.Lock] = {}
    _export_locks_guard = threading.Lock()
    # The annotate route is a plain `def` (runs in FastAPI's threadpool), so two
    # requests could otherwise call predict on the same model object concurrently —
    # ultralytics isn't thread-safe. Serialize inference; the event loop stays free.
    _inference_lock = threading.Lock()

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
        classes_file: Optional[UploadFile],
    ) -> UploadModelResponse:
        cls._ensure_new_model(name)

        weights_ext = Path(weights_file.filename or "").suffix.lower()
        if weights_ext not in SUPPORTED_WEIGHTS_EXTS:
            raise HTTPException(status_code=400, detail="Weights must be a .pt or .onnx file")

        weights_bytes = await weights_file.read()
        classes_text = await cls._read_optional_classes(classes_file)
        classes_list = cls._resolve_classes(classes_text, weights_bytes, weights_ext)

        return cls._store_model(name, weights_bytes, weights_ext, classes_list)

    @staticmethod
    async def _read_optional_classes(classes_file: Optional[UploadFile]) -> Optional[str]:
        if classes_file is None or not (classes_file.filename or "").strip():
            return None
        return (await classes_file.read()).decode("utf-8")

    @classmethod
    def _resolve_classes(cls, classes_text: Optional[str], weights_bytes: bytes, weights_ext: str) -> List[str]:
        if classes_text is not None:
            return cls._parse_classes_text(classes_text)
        return cls._derive_classes_from_weights(weights_bytes, weights_ext)

    @classmethod
    def _derive_classes_from_weights(cls, weights_bytes: bytes, weights_ext: str) -> List[str]:
        """Read class names straight from the checkpoint so weights can be
        uploaded without a classes file. Ultralytics ``.pt`` models embed
        ``model.names``; only a bare file with no recoverable names errors out.
        """
        with tempfile.NamedTemporaryFile(suffix=weights_ext, delete=False) as tmp:
            tmp.write(weights_bytes)
            tmp_path = Path(tmp.name)
        try:
            names = cls._names_to_list(getattr(YOLO(str(tmp_path)), "names", None))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read class names from the weights: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        if not names:
            raise HTTPException(
                status_code=400,
                detail="Could not read class names from the weights — attach a classes file",
            )
        return names

    @staticmethod
    def _names_to_list(names: object) -> Optional[List[str]]:
        if isinstance(names, dict):
            try:
                ordered = sorted(names.items(), key=lambda item: int(item[0]))
            except (TypeError, ValueError):
                ordered = sorted(names.items(), key=lambda item: str(item[0]))
            return [str(value) for _, value in ordered]
        if isinstance(names, (list, tuple)):
            return [str(value) for value in names]
        return None

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

        classes_list = cls._resolve_classes(classes_text, weights_bytes, weights_ext)
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
    def _read_imgsz(cls, model_dir: Path) -> Optional[int]:
        metadata_path = model_dir / "training-metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open() as f:
                    imgsz = json.load(f).get("imgsz")
                if isinstance(imgsz, int) and imgsz > 0:
                    return imgsz
            except (OSError, json.JSONDecodeError):
                pass
        return None

    @classmethod
    def export_onnx(cls, name: str) -> Path:
        weights_path, _ = cls.get_model_files(name)
        if weights_path.suffix == ".onnx":
            return weights_path

        model_dir = weights_path.parent
        # The export must not be named weights.* — get_model and friends resolve
        # weights via glob("weights.*") and weights.onnx would shadow weights.pt.
        onnx_path = model_dir / "export.onnx"
        if onnx_path.exists() and onnx_path.stat().st_mtime >= weights_path.stat().st_mtime:
            return onnx_path

        with cls._export_locks_guard:
            lock = cls._export_locks.setdefault(name, threading.Lock())

        with lock:
            if onnx_path.exists() and onnx_path.stat().st_mtime >= weights_path.stat().st_mtime:
                return onnx_path

            export_kwargs: dict = {"format": "onnx", "simplify": False, "device": "cpu"}
            imgsz = cls._read_imgsz(model_dir)
            if imgsz is not None:
                export_kwargs["imgsz"] = imgsz

            # Export in a temp dir on a fresh YOLO instance: export() fuses layers
            # in-place (the inference cache must stay pristine) and writes its
            # output next to the weights file.
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_weights = Path(tmp_dir) / weights_path.name
                shutil.copy2(weights_path, tmp_weights)
                try:
                    exported = YOLO(str(tmp_weights)).export(**export_kwargs)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"ONNX export failed: {exc}") from exc
                shutil.move(str(exported), str(onnx_path))

        return onnx_path

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
                        imgsz=cls._read_imgsz(model_dir),
                        trained=(model_dir / "training-metadata.json").exists(),
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
            imgsz=cls._read_imgsz(target_dir),
            trained=(target_dir / "training-metadata.json").exists(),
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
        total = len(payload.image_urls)
        logger.info("YOLO annotate '%s': %d image(s)", model_name, total)

        # Process one image at a time and discard the heavy prediction object before
        # the next — holding every result for the whole batch was a needless memory
        # peak. Each image is logged with timings, and a failure is logged with its
        # URL + traceback so the offending image is identifiable in the console.
        all_annotations = []
        for idx, url in enumerate(payload.image_urls):
            try:
                img = load_image_from_url(url)
                kwargs = dict(
                    source=img,
                    conf=payload.conf_threshold or 0.25,
                    save=False,
                    verbose=False,
                )
                if payload.imgsz is not None:
                    kwargs["imgsz"] = payload.imgsz

                t0 = time.monotonic()
                try:
                    with cls._inference_lock:
                        pred = model.predict(**kwargs)[0]
                except Exception as exc:
                    if payload.imgsz is None:
                        logger.warning("image %d/%d retrying at imgsz=320 after: %s", idx + 1, total, exc)
                        with cls._inference_lock:
                            pred = model.predict(**{**kwargs, "imgsz": 320})[0]
                    else:
                        raise
                infer_ms = (time.monotonic() - t0) * 1000

                boxes = pred.boxes
                annotations_list = []
                for xyxy, cls_idx, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                    x1, y1, x2, y2 = xyxy
                    raw_name = class_names[int(cls_idx)] if int(cls_idx) < len(class_names) else str(cls_idx)
                    mapped_name = payload.class_map.get(raw_name, raw_name) if payload.class_map else raw_name
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
                logger.info(
                    "image %d/%d ok: %d detection(s), infer=%.0f ms", idx + 1, total, len(annotations_list), infer_ms
                )
                del pred, img
            except HTTPException:
                logger.exception("image %d/%d FAILED (url=%s)", idx + 1, total, url)
                raise
            except Exception as exc:
                logger.exception("image %d/%d FAILED (url=%s)", idx + 1, total, url)
                raise HTTPException(
                    status_code=500, detail=f"YOLO inference failed on image {idx + 1}/{total}: {exc}"
                ) from exc
            finally:
                free_inference_memory()

        return all_annotations
