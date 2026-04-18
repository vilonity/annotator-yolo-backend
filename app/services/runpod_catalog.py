"""Live pricing catalog for RunPod GPUs.

Uses the `runpod` SDK (GraphQL under the hood) to fetch real hourly prices
and GPU memory for each SKU we expose. Results are cached for an hour so
we don't hit the API on every devices listing.

Note: the RunPod GraphQL `GpuType` entity does NOT expose system RAM or
vCPU allocations (they're Pod config fields, not GPU properties). We
hard-code them here based on RunPod's documented per-SKU default configs.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from app.config import RUNPOD_API_KEY

logger = logging.getLogger(__name__)

_CATALOG_TTL_SECONDS = 3600.0


# Maps our synthetic device ids to the RunPod GPU type id used by the SDK,
# plus the typical Pod RAM/vCPU allocations shown on runpod.io.
_GPU_SKU_MAP: dict[str, dict[str, Any]] = {
    "runpod:ada_24": {
        "gpu_ids": ["NVIDIA GeForce RTX 4090"],
        "display": "RunPod RTX 4090",
        "ram_gb": 62,
        "vcpus": 16,
    },
    "runpod:ampere_80": {
        "gpu_ids": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
        "display": "RunPod A100",
        "ram_gb": 117,
        "vcpus": 8,
    },
}


class _CatalogCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._expires_at: float = 0.0
        self._value: dict[str, dict[str, Any]] = {}

    def get(self) -> Optional[dict[str, dict[str, Any]]]:
        with self._lock:
            if time.time() < self._expires_at and self._value:
                return self._value
            return None

    def set(self, value: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            self._value = value
            self._expires_at = time.time() + _CATALOG_TTL_SECONDS


_cache = _CatalogCache()


def _fetch_gpu_detail(gpu_id: str) -> Optional[dict[str, Any]]:
    try:
        import runpod

        runpod.api_key = RUNPOD_API_KEY
        return runpod.get_gpu(gpu_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to fetch RunPod GPU %s", gpu_id)
        return None


def _resolve_sku(sku_id: str, sku_cfg: dict[str, Any]) -> dict[str, Any]:
    """Pick the cheapest live variant among this SKU's fallback GPU ids."""
    detail: Optional[dict[str, Any]] = None
    for gpu_id in sku_cfg["gpu_ids"]:
        candidate = _fetch_gpu_detail(gpu_id)
        if candidate is None:
            continue
        if detail is None:
            detail = candidate
            continue
        if (candidate.get("communityPrice") or candidate.get("securePrice") or float("inf")) < (
            detail.get("communityPrice") or detail.get("securePrice") or float("inf")
        ):
            detail = candidate

    entry: dict[str, Any] = {
        "name": sku_cfg["display"],
        "vram_gb": None,
        "ram_gb": sku_cfg.get("ram_gb"),
        "vcpus": sku_cfg.get("vcpus"),
        "price_per_hour_usd": None,
    }

    if detail is None:
        return entry

    entry["name"] = sku_cfg["display"]
    entry["vram_gb"] = detail.get("memoryInGb")

    price_candidates = [
        detail.get("communityPrice"),
        detail.get("securePrice"),
        (detail.get("lowestPrice") or {}).get("uninterruptablePrice"),
    ]
    prices = [p for p in price_candidates if isinstance(p, (int, float)) and p > 0]
    if prices:
        entry["price_per_hour_usd"] = round(float(min(prices)), 2)
    return entry


def get_sku_catalog() -> dict[str, dict[str, Any]]:
    """Return live spec/price for each RunPod SKU, keyed by our synthetic id."""
    cached = _cache.get()
    if cached is not None:
        return cached

    catalog: dict[str, dict[str, Any]] = {}
    for sku_id, sku_cfg in _GPU_SKU_MAP.items():
        catalog[sku_id] = _resolve_sku(sku_id, sku_cfg)
    _cache.set(catalog)
    return catalog


def build_runpod_device_options() -> list[dict[str, Any]]:
    catalog = get_sku_catalog()

    ada = catalog.get("runpod:ada_24") or {}
    ampere = catalog.get("runpod:ampere_80") or {}

    auto_price = None
    for candidate in (ada.get("price_per_hour_usd"), ampere.get("price_per_hour_usd")):
        if candidate is not None:
            auto_price = candidate if auto_price is None else min(auto_price, candidate)

    return [
        {
            "id": "runpod:auto",
            "name": "RunPod (auto 4090 / A100)",
            "vram_gb": ada.get("vram_gb") or ampere.get("vram_gb"),
            "ram_gb": None,
            "vcpus": None,
            "price_per_hour_usd": auto_price,
        },
        {
            "id": "runpod:ada_24",
            "name": ada.get("name") or "RunPod RTX 4090",
            "vram_gb": ada.get("vram_gb") or 24,
            "ram_gb": ada.get("ram_gb"),
            "vcpus": ada.get("vcpus"),
            "price_per_hour_usd": ada.get("price_per_hour_usd"),
        },
        {
            "id": "runpod:ampere_80",
            "name": ampere.get("name") or "RunPod A100",
            "vram_gb": ampere.get("vram_gb") or 80,
            "ram_gb": ampere.get("ram_gb"),
            "vcpus": ampere.get("vcpus"),
            "price_per_hour_usd": ampere.get("price_per_hour_usd"),
        },
    ]
