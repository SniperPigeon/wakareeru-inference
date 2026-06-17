import argparse
import base64
import json
from pathlib import Path
from typing import Any

from wakareeru_inference.config import load_service_config
from wakareeru_inference.service import WakareeruService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local serverless-style smoke test against WakareeruService.",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to a local image.",
    )
    parser.add_argument(
        "--config",
        default=Path("configs/service_config.yaml"),
        type=Path,
        help="Path to service_config.yaml.",
    )
    parser.add_argument(
        "--top-k",
        default=None,
        type=int,
        help="Override classifier top_k for this request.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Optional path to write JSON response.",
    )
    return parser.parse_args()


def make_event(*, image_path: Path, top_k: int | None) -> dict[str, Any]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload: dict[str, Any] = {
        "image_base64": encoded,
    }
    if top_k is not None:
        payload["top_k"] = top_k
    return {
        "input": payload,
    }


def main() -> None:
    args = parse_args()
    service = WakareeruService(load_service_config(args.config))
    result = service.predict_event(
        make_event(
            image_path=args.image,
            top_k=args.top_k,
        )
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
