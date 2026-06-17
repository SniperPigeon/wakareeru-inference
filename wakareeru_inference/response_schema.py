from enum import StrEnum


class ResponseStatus(StrEnum):
    OK = "ok"
    NO_DETECTION = "no_detection"
    ERROR = "error"


class DetectionStatus(StrEnum):
    DETECTED = "detected"
    FALLBACK_WHOLE_IMAGE = "fallback_whole_image"


class ClassificationStatus(StrEnum):
    CLASSIFIED = "classified"
    LOW_CONFIDENCE = "low_confidence"
    NO_PREDICTION = "no_prediction"
