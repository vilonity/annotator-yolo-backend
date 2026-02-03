from .health import router as health_router
from .yolo import router as yolo_router
from .sam3 import router as sam3_router

__all__ = [
    "health_router",
    "yolo_router",
    "sam3_router",
]
