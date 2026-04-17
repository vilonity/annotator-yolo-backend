import json
import shutil
import base64
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
import requests
from fastapi import HTTPException, UploadFile
from ultralytics import SAM
from ultralytics.models.sam import SAM3SemanticPredictor

from app.config import SAM3_MODELS_DIR
from app.schemas.sam3 import (
    Sam3ModelInfo,
    UploadSam3ModelResponse,
    DownloadSam3FromHuggingFaceRequest,
    Sam3AnnotateRequest,
    Sam3AnnotateResponse,
    Sam3ConceptRequest,
    Sam3ConceptResponse,
    Sam3ConceptBatchRequest,
    Sam3ConceptBatchResponse,
    Sam3ConceptBatchResultItem,
)
from app.services.image_service import load_image_from_url


def mask_to_base64_png(mask_tensor) -> str:
    mask = mask_tensor.cpu().numpy().astype(np.uint8)
    _, png_data = cv2.imencode('.png', mask * 255)
    return base64.b64encode(png_data).decode('utf-8')


class Sam3Service:
    _visual_cache: dict[str, SAM] = {}
    _concept_cache: dict[str, SAM3SemanticPredictor] = {}

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
        model_dir = SAM3_MODELS_DIR / normalized_name
        if model_dir.exists():
            raise HTTPException(status_code=400, detail="SAM3 model with this name already exists")
        model_dir.mkdir(parents=True, exist_ok=False)
        return normalized_name, model_dir

    @classmethod
    def get_visual_model(cls, name: str) -> SAM:
        if name in cls._visual_cache:
            model_dir = SAM3_MODELS_DIR / name
            if not model_dir.exists():
                cls._visual_cache.pop(name, None)
                raise HTTPException(status_code=404, detail="SAM3 model not found")
            return cls._visual_cache[name]

        model_dir = SAM3_MODELS_DIR / name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="SAM3 model not found")

        weights_path = model_dir / "sam3.pt"
        if not weights_path.exists():
            raise HTTPException(status_code=404, detail="SAM3 model weights file not found")

        try:
            model = SAM(str(weights_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load SAM3 model: {exc}") from exc

        cls._visual_cache[name] = model
        return model

    @classmethod
    def get_concept_predictor(cls, name: str) -> SAM3SemanticPredictor:
        if name in cls._concept_cache:
            model_dir = SAM3_MODELS_DIR / name
            if not model_dir.exists():
                cls._concept_cache.pop(name, None)
                raise HTTPException(status_code=404, detail="SAM3 model not found")
            return cls._concept_cache[name]

        model_dir = SAM3_MODELS_DIR / name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="SAM3 model not found")

        weights_path = model_dir / "sam3.pt"
        if not weights_path.exists():
            raise HTTPException(status_code=404, detail="SAM3 model weights file not found")

        try:
            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=str(weights_path),
                half=True,
            )
            predictor = SAM3SemanticPredictor(overrides=overrides)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load SAM3 semantic predictor: {exc}") from exc

        cls._concept_cache[name] = predictor
        return predictor

    @classmethod
    async def upload_model(
        cls,
        name: str,
        weights_file: UploadFile,
    ) -> UploadSam3ModelResponse:
        normalized_name, model_dir = cls._prepare_model_dir(name)

        try:
            weights_storage_path = model_dir / "sam3.pt"
            with weights_storage_path.open("wb") as buffer:
                buffer.write(await weights_file.read())

            cls._write_metadata(
                model_dir,
                {
                    "name": normalized_name,
                    "source": "upload",
                    "original_filename": weights_file.filename,
                },
            )
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

        return UploadSam3ModelResponse(name=normalized_name, message="SAM3 model uploaded successfully")

    @classmethod
    def download_from_huggingface(
        cls,
        payload: DownloadSam3FromHuggingFaceRequest,
    ) -> UploadSam3ModelResponse:
        normalized_name, model_dir = cls._prepare_model_dir(payload.name)
        source_filename = payload.filename.strip() or "sam3.pt"

        if not source_filename.lower().endswith(".pt"):
            shutil.rmtree(model_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=(
                    "Only Ultralytics-compatible .pt weights can be installed automatically. "
                    f"Received '{source_filename}'. Download the SAM3 .pt checkpoint from Hugging Face instead."
                ),
            )

        download_url = f"https://huggingface.co/{payload.repo_id}/resolve/main/{source_filename}"
        headers = {}
        if payload.token:
            headers["Authorization"] = f"Bearer {payload.token}"

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

                weights_storage_path = model_dir / "sam3.pt"
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
                },
            )
        except HTTPException:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise
        except requests.RequestException as exc:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise HTTPException(status_code=502, detail=f"Failed to download SAM3 model from Hugging Face: {exc}") from exc
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

        return UploadSam3ModelResponse(
            name=normalized_name,
            message=f"SAM3 model downloaded from Hugging Face and saved as sam3.pt",
        )

    @classmethod
    def list_models(cls) -> List[Sam3ModelInfo]:
        models = []
        for model_dir in SAM3_MODELS_DIR.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    models.append(Sam3ModelInfo(
                        name=model_dir.name,
                        date_add=datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
                    ))
        return models

    @classmethod
    def delete_model(cls, model_name: str) -> None:
        model_dir = SAM3_MODELS_DIR / model_name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="SAM3 model not found")

        cls._visual_cache.pop(model_name, None)
        cls._concept_cache.pop(model_name, None)
        shutil.rmtree(model_dir)

    @classmethod
    def annotate(
        cls,
        model_name: str,
        payload: Sam3AnnotateRequest,
    ) -> Sam3AnnotateResponse:
        model = cls.get_visual_model(model_name)
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
            raise HTTPException(status_code=500, detail=f"SAM3 inference failed: {exc}") from exc

        masks_list = []
        boxes_list = []
        confidences_list = []
        mask_images_list = []

        if results and len(results) > 0:
            result = results[0]

            if hasattr(result, 'masks') and result.masks is not None:
                masks_list = [p.tolist() for p in result.masks.xy]
                for mask_data in result.masks.data:
                    mask_images_list.append(mask_to_base64_png(mask_data))

            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes_list = result.boxes.xyxy.cpu().numpy().tolist()
                if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                    confidences_list = result.boxes.conf.cpu().numpy().tolist()

        return Sam3AnnotateResponse(
            masks=masks_list,
            boxes=boxes_list,
            confidences=confidences_list if confidences_list else [1.0] * len(masks_list),
            mask_images=mask_images_list
        )

    @classmethod
    def concept_segment(
        cls,
        model_name: str,
        payload: Sam3ConceptRequest,
    ) -> Sam3ConceptResponse:
        predictor = cls.get_concept_predictor(model_name)
        img = load_image_from_url(payload.image_url)

        try:
            predictor.set_image(img)
            results = predictor(text=payload.text_prompts, save=False, retina_masks=True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SAM3 concept segmentation failed: {exc}") from exc

        masks_list = []
        boxes_list = []
        confidences_list = []
        prompt_indices_list = []
        mask_images_list = []

        if results and len(results) > 0:
            for prompt_idx, result in enumerate(results):
                if hasattr(result, 'masks') and result.masks is not None:
                    polygons = [p.tolist() for p in result.masks.xy]
                    masks_list.extend(polygons)
                    for mask_data in result.masks.data:
                        mask_images_list.append(mask_to_base64_png(mask_data))

                    if hasattr(result, 'boxes') and result.boxes is not None:
                        bxs = result.boxes.xyxy.cpu().numpy().tolist()
                        boxes_list.extend(bxs)

                        if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                            confs = result.boxes.conf.cpu().numpy().tolist()
                            confs = [c for c in confs if c >= (payload.conf_threshold or 0.25)]
                            confidences_list.extend(confs)
                        else:
                            confidences_list.extend([1.0] * len(bxs))

                    prompt_indices_list.extend([prompt_idx] * len(polygons))

        return Sam3ConceptResponse(
            masks=masks_list,
            boxes=boxes_list,
            confidences=confidences_list if confidences_list else [1.0] * len(masks_list),
            prompt_indices=prompt_indices_list,
            mask_images=mask_images_list
        )

    @classmethod
    def concept_batch(
        cls,
        model_name: str,
        payload: Sam3ConceptBatchRequest,
    ) -> Sam3ConceptBatchResponse:
        predictor = cls.get_concept_predictor(model_name)
        results_list = []

        for image_url in payload.image_urls:
            try:
                img = load_image_from_url(image_url)
                predictor.set_image(img)
                results = predictor(text=payload.text_prompts, save=False, retina_masks=True)
            except Exception as exc:
                results_list.append(Sam3ConceptBatchResultItem(
                    masks=[],
                    boxes=[],
                    confidences=[],
                    prompt_indices=[],
                    mask_images=[]
                ))
                continue

            masks_list = []
            boxes_list = []
            confidences_list = []
            prompt_indices_list = []
            mask_images_list = []

            if results and len(results) > 0:
                for prompt_idx, result in enumerate(results):
                    if hasattr(result, 'masks') and result.masks is not None:
                        polygons = [p.tolist() for p in result.masks.xy]
                        mask_imgs = [mask_to_base64_png(m) for m in result.masks.data]

                        if hasattr(result, 'boxes') and result.boxes is not None:
                            bxs = result.boxes.xyxy.cpu().numpy().tolist()
                            
                            if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                                confs = result.boxes.conf.cpu().numpy().tolist()
                            else:
                                confs = [1.0] * len(bxs)

                            for i, (polygon, box, conf, mask_img) in enumerate(zip(polygons, bxs, confs, mask_imgs)):
                                if conf >= (payload.conf_threshold or 0.25):
                                    masks_list.append(polygon)
                                    boxes_list.append(box)
                                    confidences_list.append(conf)
                                    prompt_indices_list.append(prompt_idx)
                                    mask_images_list.append(mask_img)

            results_list.append(Sam3ConceptBatchResultItem(
                masks=masks_list,
                boxes=boxes_list,
                confidences=confidences_list if confidences_list else [],
                prompt_indices=prompt_indices_list,
                mask_images=mask_images_list
            ))

        return Sam3ConceptBatchResponse(results=results_list)
