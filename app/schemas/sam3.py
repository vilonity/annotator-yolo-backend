from typing import Optional, Literal
from pydantic import BaseModel


class Sam3ModelInfo(BaseModel):
    name: str
    date_add: str


class UploadSam3ModelResponse(BaseModel):
    name: str
    message: str


class Sam3AnnotateRequest(BaseModel):
    image_url: str
    prompt_type: Literal["bbox", "point", "points", "points_per_object", "negative_points"]
    bboxes: Optional[list[float]] = None
    points: Optional[list[list[float]]] = None
    labels: Optional[list[int]] = None


class Sam3AnnotateResponse(BaseModel):
    masks: list[list[list[float]]]
    boxes: list[list[float]]
    confidences: list[float]
    mask_images: list[str]


class Sam3ConceptRequest(BaseModel):
    image_url: str
    text_prompts: list[str]
    conf_threshold: Optional[float] = 0.25


class Sam3ConceptResponse(BaseModel):
    masks: list[list[list[float]]]
    boxes: list[list[float]]
    confidences: list[float]
    prompt_indices: list[int]
    mask_images: list[str]


class Sam3ConceptBatchRequest(BaseModel):
    image_urls: list[str]
    text_prompts: list[str]
    conf_threshold: Optional[float] = 0.25
    class_name: str
    skip_duplicates: Optional[bool] = False


class Sam3ConceptBatchResultItem(BaseModel):
    masks: list[list[list[float]]]
    boxes: list[list[float]]
    confidences: list[float]
    prompt_indices: list[int]
    mask_images: list[str]


class Sam3ConceptBatchResponse(BaseModel):
    results: list[Sam3ConceptBatchResultItem]