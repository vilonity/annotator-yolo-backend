from typing import List

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.sam2 import (
    Sam2ModelInfo,
    UploadSam2ModelResponse,
    DownloadSam2FromHuggingFaceRequest,
    Sam2AnnotateRequest,
    Sam2AnnotateResponse,
)
from app.services.sam2_service import Sam2Service

router = APIRouter(prefix="/sam2-models", tags=["sam2"])


@router.post("", response_model=UploadSam2ModelResponse)
async def upload_sam2_model(
    name: str = Form(...),
    weights_file: UploadFile = File(...),
):
    return await Sam2Service.upload_model(name, weights_file)


@router.post("/download-from-huggingface", response_model=UploadSam2ModelResponse)
async def download_sam2_model_from_huggingface(payload: DownloadSam2FromHuggingFaceRequest):
    return Sam2Service.download_from_huggingface(payload)


@router.get("", response_model=List[Sam2ModelInfo])
def list_sam2_models():
    return Sam2Service.list_models()


@router.delete("/{model_name}", status_code=204)
def delete_sam2_model(model_name: str):
    Sam2Service.delete_model(model_name)


@router.post("/{model_name}/annotate", response_model=Sam2AnnotateResponse)
async def sam2_annotate(
    model_name: str,
    payload: Sam2AnnotateRequest,
):
    return Sam2Service.annotate(model_name, payload)
