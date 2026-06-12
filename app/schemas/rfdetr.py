from typing import Literal, Optional
from pydantic import BaseModel


RfDetrVariant = Literal["nano", "small", "medium", "large"]


class RfDetrModelInfo(BaseModel):
    name: str
    classes: list[str]
    variant: RfDetrVariant
    # Square input resolution the model was trained at; None for models
    # uploaded without metadata (the variant default applies).
    resolution: Optional[int] = None
    date_add: str
    size_bytes: int
    # True when the model was produced by a training job on this server
    # (training-metadata.json exists); False for uploaded weights.
    trained: bool = False


class RfDetrAnnotateRequest(BaseModel):
    image_urls: list[str]
    conf_threshold: Optional[float] = 0.5
    class_map: Optional[dict[str, str]] = None


class RfDetrAnnotateResponse(BaseModel):
    annotations: list[list[dict]]


class RfDetrUploadResponse(BaseModel):
    name: str
    classes: list[str]
    variant: RfDetrVariant
    message: str
