"""Background sampler for host/GPU resource utilization during a training job.

Emits a snapshot every `interval` seconds. The snapshot is a flat JSON-ready
dict consumed both by the local training worker (forwards via HTTP callback)
and by the RunPod bridge (forwards via S3 + callback).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime, UTC
from typing import Any, Callable, Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1024.0 ** 3


def _safe_round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _default_disk_path() -> str:
    if sys.platform.startswith("win"):
        drive = os.getenv("SystemDrive", "C:")
        return drive + os.sep
    return "/"


def sample_resources(disk_path: Optional[str] = None) -> dict[str, Any]:
    """Snapshot current VRAM/RAM/CPU/disk usage. Any unavailable field is None."""
    sample: dict[str, Any] = {
        "vram_used_gb": None,
        "vram_total_gb": None,
        "vram_used_pct": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "ram_used_pct": None,
        "cpu_pct": None,
        "disk_used_gb": None,
        "disk_total_gb": None,
        "disk_used_pct": None,
        "sampled_at": datetime.now(UTC).isoformat(),
    }

    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            used_bytes = max(0, total_bytes - free_bytes)
            sample["vram_used_gb"] = _safe_round(used_bytes / _BYTES_PER_GB)
            sample["vram_total_gb"] = _safe_round(total_bytes / _BYTES_PER_GB)
            if total_bytes > 0:
                sample["vram_used_pct"] = _safe_round(used_bytes / total_bytes * 100.0, 1)
    except Exception:  # noqa: BLE001
        pass

    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            sample["ram_used_gb"] = _safe_round(vm.used / _BYTES_PER_GB)
            sample["ram_total_gb"] = _safe_round(vm.total / _BYTES_PER_GB)
            sample["ram_used_pct"] = _safe_round(vm.percent, 1)
        except Exception:  # noqa: BLE001
            pass

        try:
            # interval=None returns value since last call; caller should prime
            # with a throwaway sample at startup to avoid an initial zero.
            sample["cpu_pct"] = _safe_round(psutil.cpu_percent(interval=None), 1)
        except Exception:  # noqa: BLE001
            pass

        try:
            path = disk_path or _default_disk_path()
            du = psutil.disk_usage(path)
            sample["disk_used_gb"] = _safe_round(du.used / _BYTES_PER_GB)
            sample["disk_total_gb"] = _safe_round(du.total / _BYTES_PER_GB)
            sample["disk_used_pct"] = _safe_round(du.percent, 1)
        except Exception:  # noqa: BLE001
            pass

    return sample


class ResourceSampler:
    """Daemon thread that calls `on_sample(dict)` every `interval` seconds.

    Caller is responsible for `start()` and `stop()`. Stop is idempotent and
    waits briefly for the thread to exit.
    """

    def __init__(
        self,
        on_sample: Callable[[dict[str, Any]], None],
        *,
        interval: float = 5.0,
        disk_path: Optional[str] = None,
    ) -> None:
        self._on_sample = on_sample
        self._interval = max(1.0, float(interval))
        self._disk_path = disk_path
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)  # prime
            except Exception:  # noqa: BLE001
                pass

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="resource-sampler")
        self._thread.start()

    def stop(self, join_timeout: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        try:
            thread.join(timeout=join_timeout)
        except Exception:  # noqa: BLE001
            pass
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                sample = sample_resources(disk_path=self._disk_path)
                self._on_sample(sample)
            except Exception:  # noqa: BLE001
                logger.exception("resource sampler callback raised")
