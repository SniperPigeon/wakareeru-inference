import os
from pathlib import Path
from typing import Any

from wakareeru_inference.response_schema import ResponseStatus
from wakareeru_inference.config import load_service_config
from wakareeru_inference.service import WakareeruService

from dotenv import load_dotenv
import runpod
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
runpod.api_key = os.getenv("RUNPOD_API_KEY")
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
            "status": ResponseStatus.ERROR.value,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        
# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
