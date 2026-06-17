from dataclasses import dataclass
from typing import Any

import torch

from model_core.loader import LoadedClassifier
from model_core.predict import predict_crops
from wakareeru_inference.crop import CropCandidate


@dataclass(frozen=True)
class SubjectPrediction:
    candidate: CropCandidate
    predictions: list[dict[str, Any]]


def predict_subjects(
    *,
    classifier: LoadedClassifier,
    crop_candidates: list[CropCandidate],
    top_k: int,
    device: torch.device,
) -> list[SubjectPrediction]:
    if not crop_candidates:
        return []

    batch_predictions = predict_crops(
        loaded=classifier,
        images=[candidate.image for candidate in crop_candidates],
        top_k=top_k,
        device=device,
    )
    return [
        SubjectPrediction(
            candidate=candidate,
            predictions=predictions,
        )
        for candidate, predictions in zip(crop_candidates, batch_predictions)
    ]
