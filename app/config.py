import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Precedence: process env > backend-yolo/.env > monorepo-root/.env.
    # load_dotenv() respects existing keys (override=False) so shell env always wins.
    _BACKEND_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_BACKEND_ROOT / ".env")
    load_dotenv(_BACKEND_ROOT.parent / ".env")
except ImportError:
    # python-dotenv not installed — fall back to only shell env.
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODELS_DIR = BASE_DIR / "yolo_models"
YOLO_MODELS_DIR.mkdir(exist_ok=True)

SAM3_MODELS_DIR = BASE_DIR / "sam3_models"
SAM3_MODELS_DIR.mkdir(exist_ok=True)

TRAIN_JOBS_DIR = BASE_DIR / "train_jobs"
TRAIN_JOBS_DIR.mkdir(exist_ok=True)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


RUNPOD_ENABLED: bool = _env_bool("RUNPOD_ENABLED", False)
RUNPOD_API_KEY: str | None = os.getenv("RUNPOD_API_KEY") or None
RUNPOD_NETWORK_VOLUME_NAME: str = os.getenv("RUNPOD_NETWORK_VOLUME_NAME", "annotator-training-vol")
RUNPOD_VOLUME_SIZE_GB: int = int(os.getenv("RUNPOD_VOLUME_SIZE_GB", "100"))
# S3-compatible access to the NetworkVolume. The volume id acts as the bucket name.
# Datacenter-specific endpoint URL, e.g. https://s3api-eu-ro-1.runpodio.com
RUNPOD_S3_ENDPOINT_URL: str | None = os.getenv("RUNPOD_S3_ENDPOINT_URL") or None
RUNPOD_S3_ACCESS_KEY_ID: str | None = os.getenv("RUNPOD_S3_ACCESS_KEY_ID") or None
RUNPOD_S3_SECRET_ACCESS_KEY: str | None = os.getenv("RUNPOD_S3_SECRET_ACCESS_KEY") or None
RUNPOD_NETWORK_VOLUME_ID: str | None = os.getenv("RUNPOD_NETWORK_VOLUME_ID") or None
