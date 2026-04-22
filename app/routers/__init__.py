from .health import router as health_router
from .yolo import router as yolo_router
from .sam3 import router as sam3_router
from .sam2 import router as sam2_router
from .training import router as training_router

__all__ = [
    "health_router",
    "yolo_router",
    "sam3_router",
    "sam2_router",
    "training_router",
]
