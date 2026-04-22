import base64
from io import BytesIO

import cv2
import numpy as np
import requests
from fastapi import HTTPException


def load_image_from_url(url: str) -> np.ndarray:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        image_bytes = BytesIO(response.content)
        file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {exc}") from exc


def simplify_polygon(points, epsilon_ratio: float = 0.005, min_epsilon: float = 1.0) -> list[list[float]]:
    """Reduce polygon vertex count using Douglas-Peucker.

    Ultralytics `result.masks.xy` returns a vertex per edge pixel, which produces
    hundreds of points on human-sized masks. This thins them while preserving
    overall shape. `epsilon_ratio` scales with perimeter so the density stays
    consistent across small and large objects.
    """
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 4:
        return arr.reshape(-1, 2).tolist()
    contour = arr.reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = max(epsilon_ratio * perimeter, min_epsilon)
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
    return simplified.reshape(-1, 2).tolist()


def _to_numpy_mask(mask) -> np.ndarray:
    arr = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
    return (arr > 0).astype(np.uint8)


def mask_to_base64_png(mask) -> str:
    """Encode a binary mask (tensor or numpy array) as a base64 PNG."""
    arr = _to_numpy_mask(mask) * 255
    _, png_data = cv2.imencode(".png", arr)
    return base64.b64encode(png_data).decode("utf-8")


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """Extract a single outer contour from a binary mask, simplified.

    When the mask has multiple disconnected components (common after merging
    two SAM3 detections whose pixels don't touch), apply morphological closing
    with a progressively larger kernel until the components bridge into one
    contour. Falls back to the largest contour if bridging can't unify them.
    """
    bin_mask = _to_numpy_mask(mask)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    if len(contours) > 1:
        for kernel_size in (9, 17, 31):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
            bridged, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(bridged) == 1:
                contours = bridged
                break
        else:
            contours = bridged or contours
    largest = max(contours, key=cv2.contourArea)
    return simplify_polygon(largest.reshape(-1, 2))


def _bbox_contained(a, b, tol: float = 1.0) -> bool:
    return (
        a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol
    ) or (
        b[0] >= a[0] - tol and b[1] >= a[1] - tol and b[2] <= a[2] + tol and b[3] <= a[3] + tol
    )


def merge_overlapping_detections(masks, boxes, confidences):
    """Collapse detections that belong to the same physical object.

    SAM3 concept segmentation frequently returns multiple overlapping masks
    for one object — e.g. a torso mask plus a smaller mask over the collar or
    a bright patch. We merge entries whose masks share any pixel or whose
    bounding boxes are fully contained one within the other. Merged entries
    get the pixel union as mask, the union bbox, and the max confidence.

    Returns parallel lists: (masks np.uint8 HxW, boxes [x1,y1,x2,y2], confs).
    """
    n = len(masks)
    if n == 0:
        return [], [], []

    masks_np = [_to_numpy_mask(m) for m in masks]
    boxes_list = [list(b) for b in boxes]
    confs_list = [float(c) for c in confidences]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            overlap = bool(np.any(np.logical_and(masks_np[i], masks_np[j])))
            if overlap or _bbox_contained(boxes_list[i], boxes_list[j]):
                parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out_masks, out_boxes, out_confs = [], [], []
    for idxs in groups.values():
        if len(idxs) == 1:
            k = idxs[0]
            out_masks.append(masks_np[k])
            out_boxes.append(boxes_list[k])
            out_confs.append(confs_list[k])
            continue
        combined = masks_np[idxs[0]].copy()
        for k in idxs[1:]:
            combined |= masks_np[k]
        out_masks.append(combined)
        out_boxes.append([
            min(boxes_list[k][0] for k in idxs),
            min(boxes_list[k][1] for k in idxs),
            max(boxes_list[k][2] for k in idxs),
            max(boxes_list[k][3] for k in idxs),
        ])
        out_confs.append(max(confs_list[k] for k in idxs))

    return out_masks, out_boxes, out_confs


def extract_polygons_from_masks(masks_data) -> list[list[list[float]]]:
    return [mask_to_polygon(m) for m in masks_data]
