from dataclasses import dataclass
from typing import Any

from wakareeru_inference.config import ConfusionGroup, PostprocessConfig


@dataclass(frozen=True)
class ConfusionGroupIndex:
    label_to_group: dict[str, str]


def build_confusion_group_index(config: PostprocessConfig) -> ConfusionGroupIndex:
    label_to_group = {}
    for group in config.confusion_groups:
        for label in group.labels:
            label_to_group[label] = group.id
    return ConfusionGroupIndex(label_to_group=label_to_group)


def attach_postprocess(
    *,
    predictions: list[dict[str, Any]],
    confusion_groups: ConfusionGroupIndex,
) -> dict[str, Any]:
    if not predictions:
        return {
            "top_prediction": None,
            "top_k": [],
            "confusion_group": None,
            "group_candidates": [],
        }
    top_prediction = predictions[0]
    group_id = confusion_groups.label_to_group.get(str(top_prediction["label"]))
    group_candidates = []
    if group_id is not None:
        group_candidates = [
            prediction
            for prediction in predictions
            if confusion_groups.label_to_group.get(str(prediction["label"])) == group_id
        ]
    return {
        "top_prediction": top_prediction,
        "top_k": predictions,
        "confusion_group": group_id,
        "group_candidates": group_candidates,
    }


def normalize_confusion_groups(groups: list[dict[str, Any]] | list[ConfusionGroup]) -> list[dict[str, Any]]:
    return [
        group.model_dump() if isinstance(group, ConfusionGroup) else group
        for group in groups
    ]
