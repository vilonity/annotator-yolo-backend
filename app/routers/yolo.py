from typing import List

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.yolo import (
    YoloModelInfo,
    AutoAnnotateRequest,
    AutoAnnotateResponse,
    UploadModelResponse,
)
from app.services.yolo_service import YoloService

router = APIRouter(prefix="/yolo-models", tags=["yolo"])


@router.post("", response_model=UploadModelResponse)
async def upload_yolo_model(
    name: str = Form(...),
    weights_file: UploadFile = File(...),
    classes_file: UploadFile = File(...),
):
    return await YoloService.upload_model(name, weights_file, classes_file)


@router.get("", response_model=List[YoloModelInfo])
def list_yolo_models():
    return YoloService.list_models()


@router.delete("/{model_name}", status_code=204)
def delete_yolo_model(model_name: str):
    YoloService.delete_model(model_name)


@router.post("/{model_name}/annotate", response_model=AutoAnnotateResponse)
async def auto_annotate_images(
    model_name: str,
    payload: AutoAnnotateRequest,
):
    annotations = YoloService.run_inference(model_name, payload)
    return AutoAnnotateResponse(annotations=annotations)
