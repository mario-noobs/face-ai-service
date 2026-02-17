import logging
import time

from flask import Flask, g, request

from face_ai_service.utils.request_context import get_request_id

logger = logging.getLogger(__name__)


def register_request_logging(app: Flask):
    """Register before/after request hooks for structured request logging."""

    @app.before_request
    def _before_request():
        g.request_id = get_request_id()
        g.start_time = time.perf_counter()
        logger.info(
            ">>> %s %s request_id=%s",
            request.method,
            request.path,
            g.request_id,
        )

    @app.after_request
    def _after_request(response):
        duration_ms = (time.perf_counter() - getattr(g, "start_time", time.perf_counter())) * 1000
        logger.info(
            "<<< %s %s %s duration=%.0fms request_id=%s",
            response.status_code,
            request.method,
            request.path,
            duration_ms,
            getattr(g, "request_id", "unknown"),
        )
        return response
