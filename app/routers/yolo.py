from typing import List

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from app.schemas.yolo import (
    YoloModelInfo,
    AutoAnnotateRequest,
    AutoAnnotateResponse,
    RenameModelRequest,
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


@router.post("/archive", response_model=UploadModelResponse)
async def upload_yolo_model_archive(
    name: str = Form(...),
    archive_file: UploadFile = File(...),
):
    return await YoloService.upload_model_archive(name, archive_file)


@router.get("", response_model=List[YoloModelInfo])
def list_yolo_models():
    return YoloService.list_models()


@router.get("/{model_name}/weights")
def download_yolo_model_weights(model_name: str):
    weights_path, _ = YoloService.get_model_files(model_name)
    return FileResponse(
        path=str(weights_path),
        media_type="application/octet-stream",
        filename=f"{model_name}{weights_path.suffix}",
    )


@router.get("/{model_name}/onnx")
def export_yolo_model_onnx(model_name: str):
    onnx_path = YoloService.export_onnx(model_name)
    return FileResponse(
        path=str(onnx_path),
        media_type="application/octet-stream",
        filename=f"{model_name}.onnx",
    )


@router.get("/{model_name}/classes")
def download_yolo_model_classes(model_name: str):
    _, classes_path = YoloService.get_model_files(model_name)
    return FileResponse(
        path=str(classes_path),
        media_type="application/json",
        filename=f"{model_name}-classes.json",
    )


@router.delete("/{model_name}", status_code=204)
def delete_yolo_model(model_name: str):
    YoloService.delete_model(model_name)


@router.put("/{model_name}/rename", response_model=YoloModelInfo)
def rename_yolo_model(model_name: str, payload: RenameModelRequest):
    return YoloService.rename_model(model_name, payload.new_name)


@router.post("/{model_name}/annotate", response_model=AutoAnnotateResponse)
async def auto_annotate_images(
    model_name: str,
    payload: AutoAnnotateRequest,
):
    annotations = YoloService.run_inference(model_name, payload)
    return AutoAnnotateResponse(annotations=annotations)
