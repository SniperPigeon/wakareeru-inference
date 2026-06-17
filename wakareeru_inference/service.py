from typing import Any

from PIL import Image

from model_core.loader import LoadedClassifier, load_classifier
from wakareeru_inference.config import ServiceConfig
from wakareeru_inference.detector import GroundingDinoDetector, get_torch_device
from wakareeru_inference.postprocess import build_response
from wakareeru_inference.predict import predict_subjects
from wakareeru_inference.preprocess import preprocess_event, preprocess_image


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

    def predict_event(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = event.get("input", event)
        top_k = int(payload.get("top_k", self.config.classifier.top_k))

        preprocess_result = preprocess_event(
            event=event,
            detector=self.detector,
            crop_config=self.config.crop,
        )

        subject_predictions = predict_subjects(
            classifier=self.classifier,
            crop_candidates=preprocess_result.crop_candidates,
            top_k=top_k,
            device=self.device,
        )

        return build_response(
            subject_predictions=subject_predictions,
            postprocess_config=self.config.postprocess,
        )

    def predict_image(self, *, image: Image.Image, top_k: int | None = None) -> dict[str, Any]:
        if top_k is None:
            top_k = int(self.config.classifier.top_k)

        preprocess_result = preprocess_image(
            image=image,
            detector=self.detector,
            crop_config=self.config.crop,
        )

        subject_predictions = predict_subjects(
            classifier=self.classifier,
            crop_candidates=preprocess_result.crop_candidates,
            top_k=top_k,
            device=self.device,
        )

        return build_response(
            subject_predictions=subject_predictions,
            postprocess_config=self.config.postprocess,
        )
