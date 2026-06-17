from typing import Any

from PIL import Image

from model_core.loader import LoadedClassifier, load_classifier
from model_core.predict import predict_crops
from wakareeru_inference.config import ServiceConfig
from wakareeru_inference.crop import CropCandidate, make_crop_candidates
from wakareeru_inference.detector import GroundingDinoDetector, get_torch_device
from wakareeru_inference.image_io import load_image_from_event
from wakareeru_inference.postprocess import attach_postprocess, build_confusion_group_index


class WakareeruService:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.device = get_torch_device(config.device)
        self.detector = GroundingDinoDetector(
            config=config.detector,
            device=self.device,
        )
        self.classifier: LoadedClassifier = load_classifier(
            config.classifier.model_dir,
            device=self.device,
            local_files_only=config.classifier.local_files_only,
        )
        self.confusion_groups = build_confusion_group_index(config.postprocess)

    def predict_event(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = event.get("input", event)
        top_k = int(payload.get("top_k", self.config.classifier.top_k))
        image = load_image_from_event(event)
        return self.predict_image(image=image, top_k=top_k)

    def predict_image(self, *, image: Image.Image, top_k: int | None = None) -> dict[str, Any]:
        if top_k is None:
            top_k = int(self.config.classifier.top_k)
        detections = self.detector.detect(image)
        crop_candidates = make_crop_candidates(
            image=image,
            detections=detections,
            config=self.config.crop,
        )
        if not crop_candidates:
            return {
                "status": "no_detection",
                "subjects": [],
            }

        predictions = predict_crops(
            loaded=self.classifier,
            images=[candidate.image for candidate in crop_candidates],
            top_k=top_k,
            device=self.device,
        )
        subjects = [
            self._build_subject_payload(
                index=index,
                candidate=candidate,
                predictions=subject_predictions,
            )
            for index, (candidate, subject_predictions) in enumerate(
                zip(crop_candidates, predictions)
            )
        ]
        return {
            "status": "ok",
            "subject_count": len(subjects),
            "subjects": subjects,
        }

    def _build_subject_payload(
        self,
        *,
        index: int,
        candidate: CropCandidate,
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        detection_payload = {
            "status": candidate.status,
            "bbox": list(candidate.bbox) if candidate.bbox else None,
            "score": candidate.detection.score if candidate.detection else None,
            "label": candidate.detection.label if candidate.detection else None,
        }
        return {
            "index": index,
            "detection": detection_payload,
            "classification": attach_postprocess(
                predictions=predictions,
                confusion_groups=self.confusion_groups,
            ),
        }
