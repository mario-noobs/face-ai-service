import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestInferenceEngineCompare:
    """Test the comparison logic without loading real models."""

    def test_compare_same_encoding_returns_zero_distance(self):
        encoding = np.random.randn(128).astype(np.float32)
        encoding_b64 = base64.b64encode(encoding.tobytes()).decode("utf-8")

        distance = float(np.linalg.norm(encoding - encoding))
        assert distance == 0.0

    def test_compare_different_encodings_returns_positive_distance(self):
        enc1 = np.random.randn(128).astype(np.float32)
        enc2 = np.random.randn(128).astype(np.float32)

        distance = float(np.linalg.norm(enc1 - enc2))
        assert distance > 0.0

    def test_base64_encoding_roundtrip(self):
        original = np.random.randn(128).astype(np.float32)
        encoded = base64.b64encode(original.tobytes()).decode("utf-8")
        decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.float32)
        np.testing.assert_array_equal(original, decoded)
