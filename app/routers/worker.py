from fastapi import APIRouter

from app.schemas.worker import WorkerSyncRequest, WorkerSyncResponse
from app.services.worker_poller import sync_models_now

router = APIRouter(prefix="/worker", tags=["worker"])


@router.post("/sync-models", response_model=WorkerSyncResponse)
def sync_models(request: WorkerSyncRequest) -> WorkerSyncResponse:
    """Pull cloud-trained models from the central API into the local registry on
    demand, so the browser can deliver a model just before running inference
    (e.g. a benchmark run) instead of waiting for the periodic background sync."""
    result = sync_models_now(request.name)
    return WorkerSyncResponse(configured=result["configured"], models=result["models"])
