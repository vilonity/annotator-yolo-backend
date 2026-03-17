from typing import List

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.sam3 import (
    Sam3ModelInfo,
    UploadSam3ModelResponse,
    Sam3AnnotateRequest,
    Sam3AnnotateResponse,
    Sam3ConceptRequest,
    Sam3ConceptResponse,
    Sam3ConceptBatchRequest,
    Sam3ConceptBatchResponse,
)
from app.services.sam3_service import Sam3Service

router = APIRouter(prefix="/sam3-models", tags=["sam3"])


@router.post("", response_model=UploadSam3ModelResponse)
async def upload_sam3_model(
    name: str = Form(...),
    weights_file: UploadFile = File(...),
):
    return await Sam3Service.upload_model(name, weights_file)


@router.get("", response_model=List[Sam3ModelInfo])
def list_sam3_models():
    return Sam3Service.list_models()


@router.delete("/{model_name}", status_code=204)
def delete_sam3_model(model_name: str):
    Sam3Service.delete_model(model_name)


@router.post("/{model_name}/annotate", response_model=Sam3AnnotateResponse)
async def sam3_annotate(
    model_name: str,
    payload: Sam3AnnotateRequest,
):
    return Sam3Service.annotate(model_name, payload)


@router.post("/{model_name}/concept", response_model=Sam3ConceptResponse)
async def sam3_concept_segment(
    model_name: str,
    payload: Sam3ConceptRequest,
):
    return Sam3Service.concept_segment(model_name, payload)


@router.post("/{model_name}/concept-batch", response_model=Sam3ConceptBatchResponse)
async def sam3_concept_batch(
    model_name: str,
    payload: Sam3ConceptBatchRequest,
):
    return Sam3Service.concept_batch(model_name, payload)
