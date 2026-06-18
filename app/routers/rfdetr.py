from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from app.schemas.rfdetr import (
    RfDetrModelInfo,
    RfDetrAnnotateRequest,
    RfDetrAnnotateResponse,
    RfDetrUploadResponse,
)
from app.schemas.yolo import RenameModelRequest
from app.services.rfdetr_service import RfDetrService

router = APIRouter(prefix="/rfdetr-models", tags=["rfdetr"])


@router.post("", response_model=RfDetrUploadResponse)
async def upload_rfdetr_model(
    name: str = Form(...),
    variant: Optional[str] = Form(None),
    weights_file: UploadFile = File(...),
    classes_file: Optional[UploadFile] = File(None),
):
    return await RfDetrService.upload_model(name, variant, weights_file, classes_file)


@router.post("/archive", response_model=RfDetrUploadResponse)
async def upload_rfdetr_model_archive(
    name: str = Form(...),
    variant: Optional[str] = Form(None),
    archive_file: UploadFile = File(...),
):
    return await RfDetrService.upload_model_archive(name, variant, archive_file)


@router.get("", response_model=List[RfDetrModelInfo])
def list_rfdetr_models():
    return RfDetrService.list_models()


@router.get("/{model_name}/weights")
def download_rfdetr_model_weights(model_name: str):
    weights_path, _ = RfDetrService.get_model_files(model_name)
    return FileResponse(
        path=str(weights_path),
        media_type="application/octet-stream",
        filename=f"{model_name}{weights_path.suffix}",
    )


@router.get("/{model_name}/onnx")
def export_rfdetr_model_onnx(model_name: str):
    onnx_path = RfDetrService.export_onnx(model_name)
    return FileResponse(
        path=str(onnx_path),
        media_type="application/octet-stream",
        filename=f"{model_name}.onnx",
    )


@router.get("/{model_name}/classes")
def download_rfdetr_model_classes(model_name: str):
    _, classes_path = RfDetrService.get_model_files(model_name)
    return FileResponse(
        path=str(classes_path),
        media_type="application/json",
        filename=f"{model_name}-classes.json",
    )


@router.delete("/{model_name}", status_code=204)
def delete_rfdetr_model(model_name: str):
    RfDetrService.delete_model(model_name)


@router.put("/{model_name}/rename", response_model=RfDetrModelInfo)
def rename_rfdetr_model(model_name: str, payload: RenameModelRequest):
    return RfDetrService.rename_model(model_name, payload.new_name)


@router.post("/{model_name}/annotate", response_model=RfDetrAnnotateResponse)
async def auto_annotate_images_rfdetr(
    model_name: str,
    payload: RfDetrAnnotateRequest,
):
    annotations = RfDetrService.run_inference(model_name, payload)
    return RfDetrAnnotateResponse(annotations=annotations)
