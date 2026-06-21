from copy import deepcopy

import pytest

from wakareeru_inference.config import apply_environment_overrides


BASE_PAYLOAD = {
    "version": {"classifier": "configured-version"},
    "classifier": {"model_dir": "models/configured-version"},
}


def test_classifier_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAKAREERU_CLASSIFIER_VERSION", "wakareeru-v0.1.1-alpha.1")
    monkeypatch.setenv(
        "WAKAREERU_CLASSIFIER_MODEL_DIR",
        "/app/models/wakareeru-v0.1.1-alpha.1",
    )

    payload = apply_environment_overrides(deepcopy(BASE_PAYLOAD))

    assert payload["version"]["classifier"] == "wakareeru-v0.1.1-alpha.1"
    assert payload["classifier"]["model_dir"] == "/app/models/wakareeru-v0.1.1-alpha.1"


def test_classifier_environment_override_requires_both_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WAKAREERU_CLASSIFIER_VERSION", "wakareeru-v0.1.1-alpha.1")

    with pytest.raises(ValueError, match="must be set together"):
        apply_environment_overrides(deepcopy(BASE_PAYLOAD))
