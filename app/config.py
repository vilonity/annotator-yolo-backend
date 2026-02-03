from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODELS_DIR = BASE_DIR / "yolo_models"
YOLO_MODELS_DIR.mkdir(exist_ok=True)

SAM3_MODELS_DIR = BASE_DIR / "sam3_models"
SAM3_MODELS_DIR.mkdir(exist_ok=True)
