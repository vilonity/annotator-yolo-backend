from typing import Optional

from pydantic import BaseModel


class WorkerSyncRequest(BaseModel):
    # Sync a single cloud-trained model by its produced name; None syncs all of
    # the user's pending cloud models.
    name: Optional[str] = None


class WorkerSyncResponse(BaseModel):
    # False when pull-mode isn't configured (no API URL / worker token), so the
    # caller can tell the user to start the server as a worker.
    configured: bool
    # Names of cloud-trained models now present in the local registry after the
    # sync attempt.
    models: list[str]
