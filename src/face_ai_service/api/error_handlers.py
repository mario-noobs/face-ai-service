import logging

from flask import jsonify

from face_ai_service.schemas.exceptions import FaceAIException
from face_ai_service.schemas.responses import error_response

logger = logging.getLogger(__name__)


def register_error_handlers(app):
    @app.errorhandler(FaceAIException)
    def handle_face_ai_exception(exc):
        logger.warning("FaceAIException: code=%s message=%s", exc.code, exc.message)
        return jsonify(error_response(exc.code, exc.message, exc.detail)), 400

    @app.errorhandler(400)
    def handle_bad_request(exc):
        return jsonify(error_response("4000", "Bad request", str(exc))), 400

    @app.errorhandler(404)
    def handle_not_found(exc):
        return jsonify(error_response("4004", "Not found")), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(exc):
        return jsonify(error_response("4005", "Method not allowed")), 405

    @app.errorhandler(500)
    def handle_internal_error(exc):
        logger.error("Internal server error: %s", exc, exc_info=True)
        return jsonify(error_response("1111", "Internal server error")), 500
