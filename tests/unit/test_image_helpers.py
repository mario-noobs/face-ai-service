import base64
import io

import numpy as np
import pytest
from PIL import Image

from face_ai_service.utils.image_helpers import base64_to_numpy_image, is_valid_base64_image


def _create_test_image_base64(width=100, height=100):
    """Create a small valid JPEG image as base64."""
    img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestBase64ToNumpyImage:
    def test_valid_image(self):
        b64 = _create_test_image_base64()
        image = base64_to_numpy_image(b64)
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3  # BGR channels

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            base64_to_numpy_image("not_valid_base64!!!")


class TestIsValidBase64Image:
    def test_valid_image(self):
        b64 = _create_test_image_base64()
        assert is_valid_base64_image(b64) is True

    def test_invalid_base64(self):
        assert is_valid_base64_image("not_an_image") is False

    def test_empty_string(self):
        assert is_valid_base64_image("") is False
