from typing import Literal, Optional
from pydantic import BaseModel, Field


TrainingJobStatus = Literal[
    "queued",
    "running",
    "cancelling",
    "cancelled",
    "completed",
    "failed",
    "interrupted",
]

TrainingProvider = Literal["local", "runpod"]


class SplitConfig(BaseModel):
    train_percent: int
    val_percent: int
    test_percent: int


class TrainingMetricsSummary(BaseModel):
    precision: Optional[float] = None
    recall: Optional[float] = None
    map50: Optional[float] = None
    map: Optional[float] = None


class TrainingArtifacts(BaseModel):
    best_weights_path: Optional[str] = None
    results_csv_path: Optional[str] = None


class TrainingJobSummary(BaseModel):
    id: str
    user_id: str
    project_name: str
    output_model_name: str
    produced_model_name: Optional[str] = None
    base_model: str
    status: TrainingJobStatus
    provider: TrainingProvider = "local"
    epochs: int
    imgsz: Optional[int] = None
    batch: Optional[int] = None
    workers: Optional[int] = None
    device: Optional[str] = None
    device_name: Optional[str] = None
    total_images: int
    boxed_images: int
    empty_images: int
    classes: list[str] = Field(default_factory=list)
    split: SplitConfig
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics: Optional[TrainingMetricsSummary] = None
    error: Optional[str] = None
    artifacts: Optional[TrainingArtifacts] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    runpod_endpoint_id: Optional[str] = None
    runpod_job_id: Optional[str] = None


class TrainingJobDetail(TrainingJobSummary):
    logs_tail: list[str] = Field(default_factory=list)


class DeviceInfo(BaseModel):
    id: str
    name: str
    provider: TrainingProvider = "local"
    vram_gb: Optional[int] = None
    ram_gb: Optional[int] = None
    vcpus: Optional[int] = None
    price_per_hour_usd: Optional[float] = None


class DevicesResponse(BaseModel):
    devices: list[DeviceInfo]
