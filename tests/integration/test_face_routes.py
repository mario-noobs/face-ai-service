import base64
import io

import numpy as np
from PIL import Image


def _create_test_image_base64():
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestHealthRoutes:
    def test_ping(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "retinaface_mobilenet" in data["checks"]["models"]["detection"]


class TestEncodeRoute:
    def test_encode_missing_image(self, client):
        resp = client.post("/api/v1/face/encode", json={})
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "4000"

    def test_encode_invalid_base64(self, client):
        resp = client.post("/api/v1/face/encode", json={"imageBase64": "not_valid"})
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "4001"

    def test_encode_model_not_loaded(self, client):
        b64 = _create_test_image_base64()
        resp = client.post("/api/v1/face/encode", json={
            "imageBase64": b64,
            "algorithmDet": "unknown_model",
        })
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "5003"

    def test_encode_success(self, client, mock_engine):
        mock_engine.encode_face.return_value = {
            "encoding": "abc123",
            "encoding_shape": [128],
            "confidence": 0.99,
            "bbox": [10, 20, 100, 120],
            "algorithmDet": "retinaface_mobilenet",
            "algorithmReg": "facenet_mobilenet",
        }
        b64 = _create_test_image_base64()
        resp = client.post("/api/v1/face/encode", json={"imageBase64": b64})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["code"] == "0000"
        assert data["data"]["encoding"] == "abc123"

    def test_encode_no_face(self, client, mock_engine):
        mock_engine.encode_face.return_value = None
        b64 = _create_test_image_base64()
        resp = client.post("/api/v1/face/encode", json={"imageBase64": b64})
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "5002"


class TestSearchRoute:
    def test_search_missing_image(self, client):
        resp = client.post("/api/v1/face/search", json={"candidates": [{"userId": "1", "encoding": "enc"}]})
        assert resp.status_code == 400

    def test_search_no_candidates(self, client):
        b64 = _create_test_image_base64()
        resp = client.post("/api/v1/face/search", json={"imageBase64": b64})
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "4003"

    def test_search_success(self, client, mock_engine):
        mock_engine.search_faces.return_value = {
            "query_encoding": "qenc",
            "matches": [
                {"userId": "1", "distance": 0.5, "matched": True},
            ],
        }
        b64 = _create_test_image_base64()
        resp = client.post("/api/v1/face/search", json={
            "imageBase64": b64,
            "candidates": [{"userId": "1", "encoding": "enc1"}],
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["code"] == "0000"
        assert len(data["data"]["matches"]) == 1
        assert data["data"]["matches"][0]["matched"] is True
