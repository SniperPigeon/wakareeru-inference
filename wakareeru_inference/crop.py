from dataclasses import dataclass

from PIL import Image

from wakareeru_inference.config import CropConfig
from wakareeru_inference.detector import Detection
from wakareeru_inference.response_schema import DetectionStatus


@dataclass(frozen=True)
class CropCandidate:
    image: Image.Image
    detection: Detection | None
    bbox: tuple[int, int, int, int] | None
    status: DetectionStatus


def make_crop_candidates(
    *,
    image: Image.Image,
    detections: list[Detection],
    config: CropConfig,
) -> list[CropCandidate]:
    selected = select_detections(detections=detections, policy=config.select_policy)
    if not selected:
        if config.fallback_policy == "whole_image":
            return [
                CropCandidate(
                    image=image.copy(),
                    detection=None,
                    bbox=None,
                    status=DetectionStatus.FALLBACK_WHOLE_IMAGE,
                )
            ]
        return []

    candidates = []
    for detection in selected:
        bbox = expand_bbox(
            bbox=detection.bbox,
            image_size=image.size,
            padding_ratio=float(config.padding_ratio),
        )
        candidates.append(
            CropCandidate(
                image=image.crop(bbox),
                detection=detection,
                bbox=bbox,
                status=DetectionStatus.DETECTED,
            )
        )
    return candidates


def select_detections(*, detections: list[Detection], policy: str) -> list[Detection]:
    if policy == "all":
        return detections
    if not detections:
        return []
    if policy == "highest_score":
        return [max(detections, key=lambda item: item.score)]
    if policy == "largest_area":
        return [max(detections, key=lambda item: item.area)]
    raise ValueError(f"Unknown crop select_policy: {policy!r}")


def expand_bbox(
    *,
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    pad = max(x2 - x1, y2 - y1) * padding_ratio
    left = max(0, int(x1 - pad))
    top = max(0, int(y1 - pad))
    right = min(width, int(x2 + pad))
    bottom = min(height, int(y2 + pad))
    if right <= left or bottom <= top:
        raise ValueError(f"Bad crop bbox: {(left, top, right, bottom)}")
    return left, top, right, bottom
