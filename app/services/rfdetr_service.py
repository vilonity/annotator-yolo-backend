import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from fastapi import HTTPException, UploadFile

from app.config import RFDETR_MODELS_DIR
from app.schemas.rfdetr import RfDetrAnnotateRequest, RfDetrModelInfo, RfDetrUploadResponse

SUPPORTED_WEIGHTS_EXTS = {".pth", ".pt"}
SUPPORTED_CLASSES_NAMES = {"classes.json", "classes.txt"}
RFDETR_VARIANTS = ("nano", "small", "medium", "large")
DEFAULT_VARIANT = "medium"

# rfdetr package class per checkpoint size — a fine-tuned checkpoint can only be
# loaded by the class it was trained with, so the variant is persisted per model.
_VARIANT_CLASS_NAMES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def load_rfdetr_class(variant: str):
    try:
        import rfdetr
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="rfdetr package is not installed on this server (pip install rfdetr)",
        ) from exc

    class_name = _VARIANT_CLASS_NAMES.get(variant)
    if class_name is None:
        raise HTTPException(status_code=400, detail=f"Unknown RF-DETR variant: {variant}")

    model_class = getattr(rfdetr, class_name, None)
    if model_class is None:
        raise HTTPException(
            status_code=500,
            detail=f"Installed rfdetr package does not provide {class_name}",
        )
    return model_class


class RfDetrService:
    _cache: dict[str, Any] = {}

    @classmethod
    def get_model(cls, name: str) -> tuple[Any, list[str]]:
        model_dir = RFDETR_MODELS_DIR / name
        classes_file = model_dir / "classes.json"
        if not model_dir.exists() or not classes_file.exists():
            cls._cache.pop(name, None)
            raise HTTPException(status_code=404, detail="RF-DETR model not found")

        with classes_file.open() as f:
            classes_list = json.load(f)

        if name in cls._cache:
            return cls._cache[name], classes_list

        weights_files = list(model_dir.glob("weights.*"))
        if not weights_files:
            raise HTTPException(status_code=404, detail="Model weights file not found")

        variant = cls._read_variant(model_dir)
        model_class = load_rfdetr_class(variant)
        load_kwargs: dict[str, Any] = {"pretrain_weights": str(weights_files[0])}
        resolution = cls._read_resolution(model_dir)
        if resolution is not None:
            # A checkpoint trained at a custom resolution stores positional
            # embeddings for that grid — it must be loaded at the same resolution.
            load_kwargs["resolution"] = resolution
        try:
            model = model_class(**load_kwargs)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load RF-DETR model: {exc}") from exc

        cls._cache[name] = model
        return model, classes_list

    @classmethod
    async def upload_model(
        cls,
        name: str,
        variant: str,
        weights_file: UploadFile,
        classes_file: UploadFile,
    ) -> RfDetrUploadResponse:
        cls._ensure_new_model(name)
        cls._validate_variant(variant)

        weights_ext = Path(weights_file.filename or "").suffix.lower()
        if weights_ext not in SUPPORTED_WEIGHTS_EXTS:
            raise HTTPException(status_code=400, detail="Weights must be a .pth or .pt file")

        weights_bytes = await weights_file.read()
        classes_text = (await classes_file.read()).decode("utf-8")
        classes_list = cls._parse_classes_text(classes_text)

        return cls._store_model(name, variant, weights_bytes, weights_ext, classes_list)

    @classmethod
    async def upload_model_archive(cls, name: str, variant: str, archive_file: UploadFile) -> RfDetrUploadResponse:
        cls._ensure_new_model(name)
        cls._validate_variant(variant)

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
            raise HTTPException(status_code=400, detail="Archive must contain weights.pth or weights.pt")
        if classes_text is None:
            raise HTTPException(status_code=400, detail="Archive must contain classes.json (or classes.txt)")

        classes_list = cls._parse_classes_text(classes_text)
        return cls._store_model(name, variant, weights_bytes, weights_ext, classes_list)

    @classmethod
    def get_model_files(cls, name: str) -> tuple[Path, Path]:
        model_dir = RFDETR_MODELS_DIR / name
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
        model_dir = RFDETR_MODELS_DIR / name
        if model_dir.exists():
            raise HTTPException(status_code=400, detail="Model with this name already exists")

    @classmethod
    def _validate_model_name(cls, name: str) -> None:
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid model name")

    @classmethod
    def _validate_variant(cls, variant: str) -> None:
        if variant not in RFDETR_VARIANTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid RF-DETR variant (expected one of: {', '.join(RFDETR_VARIANTS)})",
            )

    @classmethod
    def _read_variant(cls, model_dir: Path) -> str:
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open() as f:
                    metadata = json.load(f)
                variant = metadata.get("variant")
                if variant in RFDETR_VARIANTS:
                    return variant
            except (OSError, json.JSONDecodeError):
                pass
        return DEFAULT_VARIANT

    @classmethod
    def _read_resolution(cls, model_dir: Path) -> Optional[int]:
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open() as f:
                    resolution = json.load(f).get("resolution")
                if isinstance(resolution, int) and resolution > 0:
                    return resolution
            except (OSError, json.JSONDecodeError):
                pass
        return None

    @classmethod
    def _parse_classes_text(cls, raw_classes_text: str) -> List[str]:
        classes_list: Optional[List[str]] = None

        try:
            parsed = json.loads(raw_classes_text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            classes_list = [str(n) for n in parsed]

        if classes_list is None:
            classes_list = [line.strip() for line in raw_classes_text.splitlines() if line.strip()]

        if not classes_list:
            raise HTTPException(status_code=400, detail="No class names found in classes file")

        return classes_list

    @classmethod
    def _store_model(
        cls,
        name: str,
        variant: str,
        weights_bytes: bytes,
        weights_ext: str,
        classes_list: List[str],
    ) -> RfDetrUploadResponse:
        model_dir = RFDETR_MODELS_DIR / name
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            weights_storage_path = model_dir / f"weights{weights_ext}"
            with weights_storage_path.open("wb") as buffer:
                buffer.write(weights_bytes)

            with (model_dir / "classes.json").open("w", encoding="utf-8") as f:
                json.dump(classes_list, f, ensure_ascii=False)

            with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump({"variant": variant}, f, ensure_ascii=True)
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

        return RfDetrUploadResponse(
            name=name,
            classes=classes_list,
            variant=variant,
            message="Model uploaded successfully",
        )

    @classmethod
    def list_models(cls) -> List[RfDetrModelInfo]:
        models = []
        for model_dir in RFDETR_MODELS_DIR.iterdir():
            if model_dir.is_dir():
                classes_file = model_dir / "classes.json"
                if classes_file.exists():
                    with classes_file.open() as f:
                        classes = json.load(f)
                    weights_files = list(model_dir.glob("weights.*"))
                    size_bytes = weights_files[0].stat().st_size if weights_files else 0
                    models.append(RfDetrModelInfo(
                        name=model_dir.name,
                        classes=classes,
                        variant=cls._read_variant(model_dir),
                        resolution=cls._read_resolution(model_dir),
                        date_add=datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
                        size_bytes=size_bytes,
                    ))
        return models

    @classmethod
    def delete_model(cls, model_name: str) -> None:
        model_dir = RFDETR_MODELS_DIR / model_name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        cls._cache.pop(model_name, None)
        shutil.rmtree(model_dir)

    @classmethod
    def rename_model(cls, model_name: str, new_name: str) -> RfDetrModelInfo:
        cls._validate_model_name(new_name)
        source_dir = RFDETR_MODELS_DIR / model_name
        if not source_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        target_dir = RFDETR_MODELS_DIR / new_name
        if target_dir.exists():
            raise HTTPException(status_code=400, detail="Model with this name already exists")

        cls._cache.pop(model_name, None)
        source_dir.rename(target_dir)

        classes_file = target_dir / "classes.json"
        with classes_file.open() as f:
            classes = json.load(f)
        weights_files = list(target_dir.glob("weights.*"))
        size_bytes = weights_files[0].stat().st_size if weights_files else 0
        return RfDetrModelInfo(
            name=new_name,
            classes=classes,
            variant=cls._read_variant(target_dir),
            resolution=cls._read_resolution(target_dir),
            date_add=datetime.fromtimestamp(target_dir.stat().st_mtime).isoformat(),
            size_bytes=size_bytes,
        )

    @classmethod
    def run_inference(
        cls,
        model_name: str,
        payload: RfDetrAnnotateRequest,
    ) -> list[list[dict]]:
        from app.services.image_service import load_image_from_url

        if not payload.image_urls:
            raise HTTPException(status_code=400, detail="image_urls list cannot be empty")

        model, class_names = cls.get_model(model_name)
        threshold = payload.conf_threshold if payload.conf_threshold is not None else 0.5

        all_annotations = []
        for url in payload.image_urls:
            img = load_image_from_url(url)
            try:
                detections = model.predict(img, threshold=threshold)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"RF-DETR inference failed: {exc}") from exc

            annotations_list = []
            for xyxy, class_id, confidence in zip(
                detections.xyxy.tolist(),
                detections.class_id.tolist(),
                detections.confidence.tolist(),
            ):
                x1, y1, x2, y2 = xyxy
                class_index = int(class_id)
                raw_name = class_names[class_index] if 0 <= class_index < len(class_names) else str(class_index)
                mapped_name = payload.class_map.get(raw_name, raw_name) if payload.class_map else raw_name
                annotations_list.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class_id": class_index,
                        "class_name": mapped_name,
                        "confidence": float(confidence),
                        "model": model_name,
                    }
                )
            all_annotations.append(annotations_list)

        return all_annotations
