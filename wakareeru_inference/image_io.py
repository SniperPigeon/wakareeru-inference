import base64
import binascii
from io import BytesIO
from typing import Any

from PIL import Image, ImageOps


def load_image_from_event(event: dict[str, Any]) -> Image.Image:
    payload = event.get("input", event)
    image_base64 = payload.get("image_base64")
    if image_base64 is None:
        raise ValueError("Request input must contain input.image_base64")
    if not isinstance(image_base64, str):
        raise TypeError("input.image_base64 must be a base64 string")
    return load_rgb_image_from_base64(image_base64)


def load_rgb_image_from_base64(value: str) -> Image.Image:
    if value.startswith("data:"):
        _, encoded = value.split(",", 1)
        return load_rgb_image_from_bytes(decode_base64_image(encoded))
    return load_rgb_image_from_bytes(decode_base64_image(value))


def decode_base64_image(value: str) -> bytes:
    normalized_value = "".join(value.split())
    try:
        return base64.b64decode(normalized_value, validate=True)
    except binascii.Error as exc:
        raise ValueError("input.image_base64 is not valid base64") from exc


def load_rgb_image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.load()
        return image.copy()
