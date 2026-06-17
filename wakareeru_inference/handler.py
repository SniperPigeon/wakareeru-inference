import os
from pathlib import Path
from typing import Any

from wakareeru_inference.config import load_service_config
from wakareeru_inference.service import WakareeruService


DEFAULT_CONFIG_PATH = Path("configs/service_config.yaml")


def build_service() -> WakareeruService:
    config_path = Path(os.getenv("WAKAREERU_SERVICE_CONFIG", DEFAULT_CONFIG_PATH))
    return WakareeruService(load_service_config(config_path))


SERVICE = build_service()


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        return SERVICE.predict_event(event)
    except Exception as exc:
        return {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
