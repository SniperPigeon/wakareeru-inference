import base64
from io import BytesIO

from PIL import Image

from wakareeru_inference.image_io import load_image_from_event, load_rgb_image_from_base64


def make_base64_jpeg() -> str:
    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_load_rgb_image_from_plain_jpeg_base64() -> None:
    encoded = make_base64_jpeg()
    assert encoded.startswith("/9j/")

    image = load_rgb_image_from_base64(encoded)

    assert image.mode == "RGB"
    assert image.size == (2, 2)


def test_load_image_from_event_plain_base64() -> None:
    image = load_image_from_event({"input": {"image_base64": make_base64_jpeg()}})

    assert image.mode == "RGB"
    assert image.size == (2, 2)
