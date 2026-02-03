from typing import Optional
from pydantic import BaseModel


class YoloModelInfo(BaseModel):
    name: str
    classes: list[str]
    date_add: str


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
