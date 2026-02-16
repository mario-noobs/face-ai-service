import logging

from flask import Blueprint, current_app, jsonify, request

from face_ai_service.schemas.exceptions import FaceAIException
from face_ai_service.schemas.messages import Messages
from face_ai_service.schemas.requests import EncodeRequest, SearchRequest
from face_ai_service.schemas.responses import error_response, success_response
from face_ai_service.utils.image_helpers import base64_to_numpy_image, is_valid_base64_image

logger = logging.getLogger(__name__)

face_bp = Blueprint("face", __name__, url_prefix="/api/v1/face")


@face_bp.route("/encode", methods=["POST"])
def encode():
    """Detect the largest face and return its encoding."""
    data = request.get_json(silent=True) or {}
    req = EncodeRequest.from_dict(data)

    if not req.imageBase64:
        raise FaceAIException(Messages.MISSING_FIELDS, "imageBase64 is required")

    if not is_valid_base64_image(req.imageBase64):
        raise FaceAIException(Messages.IMAGE_BASE64_ERROR)

    engine = current_app.engine

    # Validate algorithms are loaded
    algorithms = engine.get_loaded_algorithms()
    if req.algorithmDet not in algorithms["detection"]:
        raise FaceAIException(
            Messages.MODEL_NOT_LOADED,
            f"Detection model not loaded: {req.algorithmDet}",
        )
    if req.algorithmReg not in algorithms["recognition"]:
        raise FaceAIException(
            Messages.MODEL_NOT_LOADED,
            f"Recognition model not loaded: {req.algorithmReg}",
        )

    image_np = base64_to_numpy_image(req.imageBase64)
    result = engine.encode_face(image_np, req.algorithmDet, req.algorithmReg)

    if result is None:
        raise FaceAIException(Messages.NO_FACE)

    return jsonify(success_response(result))


@face_bp.route("/search", methods=["POST"])
def search():
    """Detect face, encode, and compare against candidate encodings."""
    data = request.get_json(silent=True) or {}
    req = SearchRequest.from_dict(data)

    if not req.imageBase64:
        raise FaceAIException(Messages.MISSING_FIELDS, "imageBase64 is required")

    if not req.candidates:
        raise FaceAIException(Messages.INVALID_CANDIDATES)

    if not is_valid_base64_image(req.imageBase64):
        raise FaceAIException(Messages.IMAGE_BASE64_ERROR)

    engine = current_app.engine

    # Validate algorithms are loaded
    algorithms = engine.get_loaded_algorithms()
    if req.algorithmDet not in algorithms["detection"]:
        raise FaceAIException(
            Messages.MODEL_NOT_LOADED,
            f"Detection model not loaded: {req.algorithmDet}",
        )
    if req.algorithmReg not in algorithms["recognition"]:
        raise FaceAIException(
            Messages.MODEL_NOT_LOADED,
            f"Recognition model not loaded: {req.algorithmReg}",
        )

    image_np = base64_to_numpy_image(req.imageBase64)

    candidates_dicts = [
        {"userId": c.userId, "encoding": c.encoding}
        for c in req.candidates
    ]

    result = engine.search_faces(
        image_np,
        candidates_dicts,
        req.algorithmDet,
        req.algorithmReg,
        req.threshold,
    )

    if result is None:
        raise FaceAIException(Messages.NO_FACE)

    return jsonify(success_response(result))
