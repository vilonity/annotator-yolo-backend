import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from fastapi import HTTPException, UploadFile
from ultralytics import SAM

from app.config import SAM2_MODELS_DIR
from app.services.image_service import (
    mask_to_base64_png,
    mask_to_polygon,
    merge_overlapping_detections,
)

# Ultralytics picks the SAM 2 architecture variant from the filename stem (e.g.
# "sam2_t", "sam2.1_b"). If the file is named anything else, loading falls
# through the generic path and inference fails. We preserve the original name
# when the user supplied one that matches, otherwise we normalize to a default.
SAM2_FILENAME_PATTERN = re.compile(r"^sam2(?:\.1)?_[tsbl](?:\.pt)?$", re.IGNORECASE)
DEFAULT_SAM2_FILENAME = "sam2.1_b.pt"


def _pick_storage_filename(source_filename: Optional[str]) -> str:
    if not source_filename:
        return DEFAULT_SAM2_FILENAME
    stem = Path(source_filename).name
    if SAM2_FILENAME_PATTERN.match(stem):
        return stem if stem.lower().endswith(".pt") else f"{stem}.pt"
    return DEFAULT_SAM2_FILENAME
from app.schemas.sam2 import (
    Sam2ModelInfo,
    UploadSam2ModelResponse,
    DownloadSam2FromHuggingFaceRequest,
    Sam2AnnotateRequest,
    Sam2AnnotateResponse,
)
from app.services.image_service import load_image_from_url


class Sam2Service:
    _cache: dict[str, SAM] = {}

    @staticmethod
    def _validate_model_name(name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise HTTPException(status_code=400, detail="Model name is required")
        if normalized in {".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid model name")
        if any(ch in normalized for ch in "\\/:*?\"<>|"):
            raise HTTPException(status_code=400, detail="Model name contains unsupported characters")
        return normalized

    @staticmethod
    def _write_metadata(model_dir: Path, metadata: dict) -> None:
        metadata_path = model_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)

    @classmethod
    def _prepare_model_dir(cls, name: str) -> tuple[str, Path]:
        normalized_name = cls._validate_model_name(name)
        model_dir = SAM2_MODELS_DIR / normalized_name
        if model_dir.exists():
            raise HTTPException(status_code=400, detail="SAM2 model with this name already exists")
        model_dir.mkdir(parents=True, exist_ok=False)
        return normalized_name, model_dir

    @classmethod
    def _find_weights_file(cls, model_dir: Path) -> Optional[Path]:
        # Prefer the SAM2-style filenames Ultralytics can parse, else fall back
        # to any .pt file stored in the model dir.
        for name in sorted(p.name for p in model_dir.glob("*.pt")):
            if SAM2_FILENAME_PATTERN.match(name):
                return model_dir / name
        pt_files = list(model_dir.glob("*.pt"))
        return pt_files[0] if pt_files else None

    @classmethod
    def get_model(cls, name: str) -> SAM:
        if name in cls._cache:
            model_dir = SAM2_MODELS_DIR / name
            if not model_dir.exists():
                cls._cache.pop(name, None)
                raise HTTPException(status_code=404, detail="SAM2 model not found")
            return cls._cache[name]

        model_dir = SAM2_MODELS_DIR / name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="SAM2 model not found")

        weights_path = cls._find_weights_file(model_dir)
        if weights_path is None:
            raise HTTPException(status_code=404, detail="SAM2 model weights file not found")

        try:
            model = SAM(str(weights_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load SAM2 model: {exc}") from exc

        cls._cache[name] = model
        return model

    @classmethod
    async def upload_model(
        cls,
        name: str,
        weights_file: UploadFile,
    ) -> UploadSam2ModelResponse:
        normalized_name, model_dir = cls._prepare_model_dir(name)

        storage_filename = _pick_storage_filename(weights_file.filename)

        try:
            weights_storage_path = model_dir / storage_filename
            with weights_storage_path.open("wb") as buffer:
                buffer.write(await weights_file.read())

            cls._write_metadata(
                model_dir,
                {
                    "name": normalized_name,
                    "source": "upload",
                    "original_filename": weights_file.filename,
                    "storage_filename": storage_filename,
                },
            )
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

        return UploadSam2ModelResponse(name=normalized_name, message="SAM2 model uploaded successfully")

    @classmethod
    def download_from_huggingface(
        cls,
        payload: DownloadSam2FromHuggingFaceRequest,
    ) -> UploadSam2ModelResponse:
        normalized_name, model_dir = cls._prepare_model_dir(payload.name)
        source_filename = payload.filename.strip() or "sam2.1_b.pt"

        if not source_filename.lower().endswith(".pt"):
            shutil.rmtree(model_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=(
                    "Only Ultralytics-compatible .pt weights can be installed automatically. "
                    f"Received '{source_filename}'. Download the SAM 2.1 .pt checkpoint from Hugging Face instead."
                ),
            )

        download_url = f"https://huggingface.co/{payload.repo_id}/resolve/main/{source_filename}"
        headers = {}
        if payload.token:
            headers["Authorization"] = f"Bearer {payload.token}"

        storage_filename = _pick_storage_filename(source_filename)

        try:
            with requests.get(download_url, headers=headers, stream=True, timeout=(10, 600)) as response:
                if response.status_code == 401:
                    raise HTTPException(status_code=401, detail="Hugging Face token is invalid or missing access")
                if response.status_code == 404:
                    raise HTTPException(
                        status_code=404,
                        detail=f"File '{source_filename}' was not found in Hugging Face repo '{payload.repo_id}'",
                    )

                response.raise_for_status()

                weights_storage_path = model_dir / storage_filename
                with weights_storage_path.open("wb") as buffer:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            buffer.write(chunk)

            cls._write_metadata(
                model_dir,
                {
                    "name": normalized_name,
                    "source": "huggingface",
                    "repo_id": payload.repo_id,
                    "filename": source_filename,
                    "storage_filename": storage_filename,
                },
            )
        except HTTPException:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise
        except requests.RequestException as exc:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise HTTPException(status_code=502, detail=f"Failed to download SAM2 model from Hugging Face: {exc}") from exc
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

        return UploadSam2ModelResponse(
            name=normalized_name,
            message=f"SAM2 model downloaded from Hugging Face and saved as {storage_filename}",
        )

    @classmethod
    def list_models(cls) -> List[Sam2ModelInfo]:
        models = []
        for model_dir in SAM2_MODELS_DIR.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    models.append(Sam2ModelInfo(
                        name=model_dir.name,
                        date_add=datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
                    ))
        return models

    @classmethod
    def delete_model(cls, model_name: str) -> None:
        model_dir = SAM2_MODELS_DIR / model_name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="SAM2 model not found")

        cls._cache.pop(model_name, None)
        shutil.rmtree(model_dir)

    @classmethod
    def annotate(
        cls,
        model_name: str,
        payload: Sam2AnnotateRequest,
    ) -> Sam2AnnotateResponse:
        model = cls.get_model(model_name)
        img = load_image_from_url(payload.image_url)

        try:
            if payload.prompt_type == "bbox":
                if not payload.bboxes or len(payload.bboxes) != 4:
                    raise HTTPException(status_code=400, detail="bboxes must contain exactly 4 values [x1, y1, x2, y2]")
                results = model(img, bboxes=payload.bboxes, retina_masks=True)

            elif payload.prompt_type == "point":
                if not payload.points or len(payload.points) != 1 or len(payload.points[0]) != 2:
                    raise HTTPException(status_code=400, detail="points must contain exactly one point [x, y]")
                if not payload.labels or len(payload.labels) != 1:
                    raise HTTPException(status_code=400, detail="labels must contain exactly one label")
                results = model(img, points=payload.points[0], labels=payload.labels, retina_masks=True)

            elif payload.prompt_type == "points":
                if not payload.points:
                    raise HTTPException(status_code=400, detail="points list cannot be empty")
                if not payload.labels or len(payload.labels) != len(payload.points):
                    raise HTTPException(status_code=400, detail="labels must have same length as points")
                results = model(img, points=payload.points, labels=payload.labels, retina_masks=True)

            elif payload.prompt_type == "points_per_object":
                if not payload.points:
                    raise HTTPException(status_code=400, detail="points list cannot be empty")
                if not payload.labels or len(payload.labels) != len(payload.points):
                    raise HTTPException(status_code=400, detail="labels must have same length as points")
                results = model(img, points=[payload.points], labels=[payload.labels], retina_masks=True)

            elif payload.prompt_type == "negative_points":
                if not payload.points:
                    raise HTTPException(status_code=400, detail="points list cannot be empty")
                if not payload.labels or len(payload.labels) != len(payload.points):
                    raise HTTPException(status_code=400, detail="labels must have same length as points")
                results = model(img, points=[payload.points], labels=[payload.labels], retina_masks=True)

            else:
                raise HTTPException(status_code=400, detail="Invalid prompt_type")

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SAM2 inference failed: {exc}") from exc

        masks_list = []
        boxes_list = []
        confidences_list = []
        mask_images_list = []

        if results and len(results) > 0:
            result = results[0]

            if hasattr(result, 'masks') and result.masks is not None:
                raw_masks = [m.cpu().numpy().astype(np.uint8) for m in result.masks.data]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    raw_boxes = result.boxes.xyxy.cpu().numpy().tolist()
                    if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                        raw_confs = result.boxes.conf.cpu().numpy().tolist()
                    else:
                        raw_confs = [1.0] * len(raw_masks)
                else:
                    raw_boxes = [[0.0, 0.0, 0.0, 0.0]] * len(raw_masks)
                    raw_confs = [1.0] * len(raw_masks)
                merged_masks, boxes_list, confidences_list = merge_overlapping_detections(
                    raw_masks, raw_boxes, raw_confs,
                )
                masks_list = [mask_to_polygon(m) for m in merged_masks]
                mask_images_list = [mask_to_base64_png(m) for m in merged_masks]

        return Sam2AnnotateResponse(
            masks=masks_list,
            boxes=boxes_list,
            confidences=confidences_list if confidences_list else [1.0] * len(masks_list),
            mask_images=mask_images_list
        )
