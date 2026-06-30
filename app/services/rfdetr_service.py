import io
import json
import logging
import shutil
import tempfile
import threading
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

logger = logging.getLogger(__name__)

# Loading a fine-tuned checkpoint at its trained resolution always makes the
# rfdetr package warn that DINOv2 backbone weights are skipped — expected for
# every checkpoint we serve, so these messages are pure noise here.
_BENIGN_LOAD_WARNINGS = (
    "not loading DINOv2 backbone weights",
    "args.num_queries absent",
)


class _BenignLoadWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(marker in message for marker in _BENIGN_LOAD_WARNINGS)


logging.getLogger("rf-detr").addFilter(_BenignLoadWarningFilter())


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
    _export_locks: dict[str, threading.Lock] = {}
    _export_locks_guard = threading.Lock()
    # Annotate runs in FastAPI's threadpool (plain `def`); serialize predict so two
    # requests can't hit the same model object concurrently. See YoloService.
    _inference_lock = threading.Lock()

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
        load_kwargs: dict[str, Any] = {
            "pretrain_weights": str(weights_files[0]),
            "num_classes": len(classes_list),
        }
        resolution = cls._read_resolution(model_dir)
        if resolution is not None:
            # A checkpoint trained at a custom resolution stores positional
            # embeddings for that grid — it must be loaded at the same resolution.
            load_kwargs["resolution"] = resolution
        try:
            model = model_class(**load_kwargs)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load RF-DETR model: {exc}") from exc

        try:
            model.optimize_for_inference()
        except Exception as exc:
            # Best-effort: tracing/half-precision can fail on some torch builds
            # (e.g. CPU-only); the un-optimized model still predicts correctly.
            logger.warning("optimize_for_inference failed for %s, serving un-optimized model: %s", name, exc)

        cls._cache[name] = model
        return model, classes_list

    @classmethod
    async def upload_model(
        cls,
        name: str,
        variant: Optional[str],
        weights_file: UploadFile,
        classes_file: Optional[UploadFile],
    ) -> RfDetrUploadResponse:
        cls._ensure_new_model(name)

        weights_ext = Path(weights_file.filename or "").suffix.lower()
        if weights_ext not in SUPPORTED_WEIGHTS_EXTS:
            raise HTTPException(status_code=400, detail="Weights must be a .pth or .pt file")

        weights_bytes = await weights_file.read()
        classes_text = await cls._read_optional_classes(classes_file)

        resolved_variant, classes_list, resolution = cls._resolve_variant_and_classes(
            variant, classes_text, weights_bytes
        )
        return cls._store_model(name, resolved_variant, weights_bytes, weights_ext, classes_list, resolution)

    @classmethod
    async def upload_model_archive(
        cls, name: str, variant: Optional[str], archive_file: UploadFile
    ) -> RfDetrUploadResponse:
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
            raise HTTPException(status_code=400, detail="Archive must contain weights.pth or weights.pt")

        resolved_variant, classes_list, resolution = cls._resolve_variant_and_classes(
            variant, classes_text, weights_bytes
        )
        return cls._store_model(name, resolved_variant, weights_bytes, weights_ext, classes_list, resolution)

    @staticmethod
    async def _read_optional_classes(classes_file: Optional[UploadFile]) -> Optional[str]:
        if classes_file is None or not (classes_file.filename or "").strip():
            return None
        return (await classes_file.read()).decode("utf-8")

    @classmethod
    def _resolve_variant_and_classes(
        cls,
        provided_variant: Optional[str],
        classes_text: Optional[str],
        weights_bytes: bytes,
    ) -> tuple[str, List[str], Optional[int]]:
        """Settle the variant + class list for an upload, filling gaps from the checkpoint.

        Reading the checkpoint is only needed to cover a missing variant or
        classes file, so the torch import/load is skipped when both are supplied.
        """
        normalized_provided = provided_variant.strip().lower() if provided_variant else None

        detected_variant: Optional[str] = None
        head_num_classes: Optional[int] = None
        resolution: Optional[int] = None
        if normalized_provided is None or classes_text is None:
            detected_variant, head_num_classes, resolution = cls._inspect_checkpoint(weights_bytes)

        # The size class is baked into the weights, so a value read from the
        # checkpoint is authoritative over the caller's hint.
        variant = detected_variant or normalized_provided or DEFAULT_VARIANT
        cls._validate_variant(variant)

        if classes_text is not None:
            classes_list = cls._parse_classes_text(classes_text)
        elif head_num_classes is not None:
            classes_list = [f"class{index + 1}" for index in range(head_num_classes)]
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No classes file provided and the class count could not be read "
                    "from the checkpoint — attach a classes file"
                ),
            )
        return variant, classes_list, resolution

    @classmethod
    def _inspect_checkpoint(cls, weights_bytes: bytes) -> tuple[Optional[str], Optional[int], Optional[int]]:
        """Best-effort (variant, class count, resolution) read from a raw checkpoint.

        Lets a user upload bare weights: the rfdetr detection head is
        ``Linear(hidden, num_classes + 1)``, so the real class count is
        ``class_embed.bias.shape[0] - 1``; newer checkpoints also carry a
        ``model_name`` (e.g. ``"RFDETRMedium"``) and the trained ``resolution``
        in ``args``. ``torch.load`` runs the pickle, same as rfdetr's own
        ``from_checkpoint`` — acceptable for a model the user registers on their
        own server. Returns ``(None, None, None)`` on any failure so the caller
        falls back to the supplied variant / classes file.
        """
        try:
            import torch
        except ImportError:
            return None, None, None
        try:
            checkpoint = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning("Could not read RF-DETR checkpoint for auto-detection: %s", exc)
            return None, None, None

        return (
            cls._variant_from_checkpoint(checkpoint),
            cls._num_classes_from_checkpoint(checkpoint),
            cls._resolution_from_checkpoint(checkpoint),
        )

    @staticmethod
    def _variant_from_checkpoint(checkpoint: Any) -> Optional[str]:
        if not isinstance(checkpoint, dict):
            return None
        # New rfdetr checkpoints store the variant class symbol directly.
        model_name = checkpoint.get("model_name")
        if isinstance(model_name, str):
            lowered = model_name.strip().lower()
            for variant in RFDETR_VARIANTS:
                if lowered in (variant, f"rfdetr{variant}"):
                    return variant
        # Legacy checkpoints only record the base weights they fine-tuned from.
        args = checkpoint.get("args")
        pretrain = args.get("pretrain_weights") if isinstance(args, dict) else getattr(args, "pretrain_weights", None)
        pretrain_name = str(pretrain or "").lower()
        for variant in RFDETR_VARIANTS:
            if variant in pretrain_name:
                return variant
        return None

    @staticmethod
    def _num_classes_from_checkpoint(checkpoint: Any) -> Optional[int]:
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            state = checkpoint if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            return None
        bias = state.get("class_embed.bias")
        if bias is None:
            bias = next(
                (value for key, value in state.items() if isinstance(key, str) and key.endswith("class_embed.bias")),
                None,
            )
        shape = getattr(bias, "shape", None)
        if not shape:
            return None
        try:
            # The head carries an extra no-object slot — drop it for the real count.
            num_classes = int(shape[0]) - 1
        except (TypeError, ValueError, IndexError):
            return None
        return num_classes if num_classes >= 1 else None

    @staticmethod
    def _resolution_from_checkpoint(checkpoint: Any) -> Optional[int]:
        if not isinstance(checkpoint, dict):
            return None
        args = checkpoint.get("args")
        resolution = args.get("resolution") if isinstance(args, dict) else getattr(args, "resolution", None)
        if isinstance(resolution, bool):
            return None
        if isinstance(resolution, int) and resolution > 0:
            return resolution
        return None

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
    def export_onnx(cls, name: str) -> Path:
        weights_path, _ = cls.get_model_files(name)
        model_dir = weights_path.parent
        # Not weights.* — get_model and friends resolve weights via
        # glob("weights.*"), so an export named weights.onnx would shadow them.
        onnx_path = model_dir / "export.onnx"
        if onnx_path.exists() and onnx_path.stat().st_mtime >= weights_path.stat().st_mtime:
            return onnx_path

        with cls._export_locks_guard:
            lock = cls._export_locks.setdefault(name, threading.Lock())

        with lock:
            if onnx_path.exists() and onnx_path.stat().st_mtime >= weights_path.stat().st_mtime:
                return onnx_path

            # rfdetr's export() deepcopies the live model and restores its
            # device on exit, so the cached inference model is safe to reuse.
            model, _ = cls.get_model(name)
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    exported = model.export(output_dir=tmp_dir, format="onnx", verbose=False)
                except ImportError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="ONNX export dependencies missing on this server (pip install rfdetr[onnx])",
                    ) from exc
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"ONNX export failed: {exc}") from exc
                shutil.move(str(exported), str(onnx_path))

        return onnx_path

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
        resolution: Optional[int] = None,
    ) -> RfDetrUploadResponse:
        model_dir = RFDETR_MODELS_DIR / name
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            weights_storage_path = model_dir / f"weights{weights_ext}"
            with weights_storage_path.open("wb") as buffer:
                buffer.write(weights_bytes)

            with (model_dir / "classes.json").open("w", encoding="utf-8") as f:
                json.dump(classes_list, f, ensure_ascii=False)

            # Persist the resolution only when it was recoverable from the
            # checkpoint, so loads of a custom-resolution upload size their
            # positional embeddings correctly instead of using the variant default.
            metadata: dict[str, Any] = {"variant": variant}
            if resolution is not None:
                metadata["resolution"] = resolution
            with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=True)
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
                        trained=(model_dir / "training-metadata.json").exists(),
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
            trained=(target_dir / "training-metadata.json").exists(),
        )

    @classmethod
    def run_inference(
        cls,
        model_name: str,
        payload: RfDetrAnnotateRequest,
    ) -> list[list[dict]]:
        import time

        from app.services.image_service import free_inference_memory, load_image_from_url

        if not payload.image_urls:
            raise HTTPException(status_code=400, detail="image_urls list cannot be empty")

        model, class_names = cls.get_model(model_name)
        threshold = payload.conf_threshold if payload.conf_threshold is not None else 0.5
        total = len(payload.image_urls)
        logger.info("RF-DETR annotate '%s': %d image(s)", model_name, total)

        # One image at a time with per-image timings and failure logging (url +
        # traceback), freeing memory between images — see YoloService.run_inference.
        all_annotations = []
        for idx, url in enumerate(payload.image_urls):
            try:
                img = load_image_from_url(url)
                t0 = time.monotonic()
                try:
                    with cls._inference_lock:
                        detections = model.predict(img, threshold=threshold)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"RF-DETR inference failed: {exc}") from exc
                infer_ms = (time.monotonic() - t0) * 1000

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
                logger.info(
                    "image %d/%d ok: %d detection(s), infer=%.0f ms", idx + 1, total, len(annotations_list), infer_ms
                )
                del detections, img
            except HTTPException:
                logger.exception("image %d/%d FAILED (url=%s)", idx + 1, total, url)
                raise
            except Exception as exc:
                logger.exception("image %d/%d FAILED (url=%s)", idx + 1, total, url)
                raise HTTPException(
                    status_code=500, detail=f"RF-DETR inference failed on image {idx + 1}/{total}: {exc}"
                ) from exc
            finally:
                free_inference_memory()

        return all_annotations
