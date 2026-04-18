from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from app.config import TRAIN_JOBS_DIR
from app.schemas.training import DevicesResponse, TrainingJobDetail, TrainingJobSummary
from app.services import runpod_training
from app.services.training_service import TrainingService

ARTIFACT_MEDIA_TYPES: dict[str, str] = {
    "results.png": "image/png",
    "confusion_matrix.png": "image/png",
    "confusion_matrix_normalized.png": "image/png",
    "PR_curve.png": "image/png",
    "F1_curve.png": "image/png",
    "P_curve.png": "image/png",
    "R_curve.png": "image/png",
    "results.csv": "text/csv",
}

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/devices", response_model=DevicesResponse)
def list_training_devices(provider: Optional[str] = None):
    return DevicesResponse(devices=TrainingService.list_devices(provider=provider))


@router.get("/runpod/diagnostics")
def runpod_diagnostics() -> dict[str, object]:
    missing = runpod_training.diagnose_runpod_availability()
    return {"ready": not missing, "missing": missing}


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
    workers: Optional[int] = Form(None),
    device: Optional[str] = Form("auto"),
    provider: Optional[str] = Form("local"),
    external_callback_url: Optional[str] = Form(None),
    external_callback_secret: Optional[str] = Form(None),
    hyperparams_json: Optional[str] = Form(None),
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
        workers=workers,
        device=device,
        provider=provider or "local",
        train_percent=train_percent,
        val_percent=val_percent,
        test_percent=test_percent,
        total_images=total_images,
        boxed_images=boxed_images,
        empty_images=empty_images,
        classes_json=classes_json,
        external_callback_url=external_callback_url,
        external_callback_secret=external_callback_secret,
        hyperparams_json=hyperparams_json,
    )


@router.get("/jobs", response_model=List[TrainingJobSummary])
def list_training_jobs(user_id: Optional[str] = None, project_name: Optional[str] = None):
    return TrainingService.list_jobs(user_id=user_id, project_name=project_name)


@router.get("/jobs/{job_id}", response_model=TrainingJobDetail)
def get_training_job(job_id: str):
    return TrainingService.get_job(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobDetail)
async def cancel_training_job(job_id: str):
    return await TrainingService.cancel_job(job_id)


@router.get("/jobs/{job_id}/logs", response_class=PlainTextResponse)
def get_training_job_logs(job_id: str):
    if not TrainingService.has_job(job_id):
        raise HTTPException(status_code=404, detail="Training job not found")
    return PlainTextResponse(TrainingService.read_full_logs(job_id), media_type="text/plain; charset=utf-8")


@router.get("/jobs/{job_id}/artifacts/{filename}")
def get_training_job_artifact(job_id: str, filename: str):
    if filename not in ARTIFACT_MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported artifact")
    if not TrainingService.has_job(job_id):
        raise HTTPException(status_code=404, detail="Training job not found")

    artifact_path = TRAIN_JOBS_DIR / job_id / "runs" / "train" / filename
    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not available")

    return FileResponse(artifact_path, media_type=ARTIFACT_MEDIA_TYPES[filename], filename=filename)


@router.get("/jobs/{job_id}/logs/stream")
async def stream_training_job_logs(job_id: str):
    if not TrainingService.has_job(job_id):
        raise HTTPException(status_code=404, detail="Training job not found")
    return StreamingResponse(
        TrainingService.stream_logs(job_id),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
