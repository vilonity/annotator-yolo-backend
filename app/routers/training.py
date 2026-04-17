from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.training import TrainingJobDetail, TrainingJobSummary
from app.services.training_service import TrainingService

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/jobs", response_model=TrainingJobDetail)
async def start_training_job(
    dataset_file: UploadFile = File(...),
    user_id: str = Form(...),
    project_name: str = Form(...),
    output_model_name: str = Form(...),
    base_model: str = Form(...),
    epochs: int = Form(...),
    train_percent: int = Form(...),
    val_percent: int = Form(...),
    test_percent: int = Form(...),
    total_images: int = Form(...),
    boxed_images: int = Form(...),
    empty_images: int = Form(...),
    classes_json: str = Form(...),
    imgsz: Optional[int] = Form(None),
    batch: Optional[int] = Form(None),
    device: Optional[str] = Form("auto"),
):
    return await TrainingService.start_job(
        dataset_file=dataset_file,
        user_id=user_id,
        project_name=project_name,
        output_model_name=output_model_name,
        base_model=base_model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        train_percent=train_percent,
        val_percent=val_percent,
        test_percent=test_percent,
        total_images=total_images,
        boxed_images=boxed_images,
        empty_images=empty_images,
        classes_json=classes_json,
    )


@router.get("/jobs", response_model=List[TrainingJobSummary])
def list_training_jobs(user_id: Optional[str] = None, project_name: Optional[str] = None):
    return TrainingService.list_jobs(user_id=user_id, project_name=project_name)


@router.get("/jobs/{job_id}", response_model=TrainingJobDetail)
def get_training_job(job_id: str):
    return TrainingService.get_job(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobDetail)
def cancel_training_job(job_id: str):
    return TrainingService.cancel_job(job_id)
