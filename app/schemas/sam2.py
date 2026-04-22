from typing import Optional, Literal
from pydantic import BaseModel


class Sam2ModelInfo(BaseModel):
    name: str
    date_add: str


class UploadSam2ModelResponse(BaseModel):
    name: str
    message: str


class DownloadSam2FromHuggingFaceRequest(BaseModel):
    name: str
    repo_id: str = "Ultralytics/SAM2.1"
    filename: str = "sam2.1_b.pt"
    token: Optional[str] = None


class Sam2AnnotateRequest(BaseModel):
    image_url: str
    prompt_type: Literal["bbox", "point", "points", "points_per_object", "negative_points"]
    bboxes: Optional[list[float]] = None
    points: Optional[list[list[float]]] = None
    labels: Optional[list[int]] = None


class Sam2AnnotateResponse(BaseModel):
    masks: list[list[list[float]]]
    boxes: list[list[float]]
    confidences: list[float]
    mask_images: list[str]
