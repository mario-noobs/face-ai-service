import base64
import io

import cv2
import numpy as np
from PIL import Image


def base64_to_numpy_image(base64_string: str) -> np.ndarray:
    """Convert a Base64 string to a NumPy array (BGR format)."""
    image_data = base64.b64decode(base64_string)
    np_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from base64 data")
    return image


def is_valid_base64_image(image_string: str) -> bool:
    """Check whether the given string is a valid base64-encoded image."""
    try:
        image_data = base64.b64decode(image_string)
        img = Image.open(io.BytesIO(image_data))
        img.verify()
        return True
    except Exception:
        return False
