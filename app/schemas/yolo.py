from typing import Optional
from pydantic import BaseModel


class YoloModelInfo(BaseModel):
    name: str
    classes: list[str]
    date_add: str
    size_bytes: int
    # Square training resolution from training-metadata.json; None for uploaded
    # models or jobs that trained with auto imgsz.
    imgsz: Optional[int] = None


class AutoAnnotateRequest(BaseModel):
    image_urls: list[str]
    conf_threshold: Optional[float] = 0.25
    imgsz: Optional[int] = None
    class_map: Optional[dict[str, str]] = None


class AutoAnnotateResponse(BaseModel):
    annotations: list[list[dict]]


class UploadModelResponse(BaseModel):
    name: str
    classes: list[str]
    message: str


class RenameModelRequest(BaseModel):
    new_name: str
