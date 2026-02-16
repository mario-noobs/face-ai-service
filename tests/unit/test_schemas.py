from face_ai_service.schemas.requests import EncodeRequest, SearchRequest, CandidateEntry
from face_ai_service.schemas.responses import success_response, error_response
from face_ai_service.schemas.exceptions import FaceAIException
from face_ai_service.schemas.messages import Messages


class TestEncodeRequest:
    def test_from_dict_defaults(self):
        req = EncodeRequest.from_dict({"imageBase64": "abc123"})
        assert req.imageBase64 == "abc123"
        assert req.algorithmDet == "retinaface_mobilenet"
        assert req.algorithmReg == "facenet_mobilenet"

    def test_from_dict_custom_algorithms(self):
        req = EncodeRequest.from_dict({
            "imageBase64": "abc",
            "algorithmDet": "retinaface_resnet50",
            "algorithmReg": "facenet_inception_resnetv1",
        })
        assert req.algorithmDet == "retinaface_resnet50"
        assert req.algorithmReg == "facenet_inception_resnetv1"

    def test_from_dict_empty(self):
        req = EncodeRequest.from_dict({})
        assert req.imageBase64 == ""


class TestSearchRequest:
    def test_from_dict_with_candidates(self):
        req = SearchRequest.from_dict({
            "imageBase64": "abc",
            "candidates": [
                {"userId": "1", "encoding": "enc1"},
                {"userId": "2", "encoding": "enc2"},
            ],
            "threshold": 1.0,
        })
        assert len(req.candidates) == 2
        assert req.candidates[0].userId == "1"
        assert req.threshold == 1.0

    def test_from_dict_no_threshold(self):
        req = SearchRequest.from_dict({"imageBase64": "abc", "candidates": []})
        assert req.threshold is None


class TestResponses:
    def test_success_response(self):
        resp = success_response({"key": "value"})
        assert resp["code"] == "0000"
        assert resp["data"]["key"] == "value"

    def test_error_response(self):
        resp = error_response("5002", "No face detected")
        assert resp["code"] == "5002"
        assert "detail" not in resp

    def test_error_response_with_detail(self):
        resp = error_response("5002", "No face", "extra info")
        assert resp["detail"] == "extra info"


class TestFaceAIException:
    def test_exception_from_message(self):
        exc = FaceAIException(Messages.NO_FACE)
        assert exc.code == "5002"
        assert "No face" in exc.message

    def test_exception_with_detail(self):
        exc = FaceAIException(Messages.MODEL_NOT_LOADED, "retinaface_resnet50")
        assert exc.detail == "retinaface_resnet50"
