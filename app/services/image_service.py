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


def extract_polygons_from_masks(masks_data) -> list[list[list[float]]]:
    polygons = []
    for mask_data in masks_data:
        mask_np = mask_data.cpu().numpy()
        contours, _ = cv2.findContours(
            mask_np.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_L1
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            polygon = largest_contour.reshape(-1, 2).tolist()
            polygons.append(polygon)
        else:
            polygons.append([])
    return polygons
