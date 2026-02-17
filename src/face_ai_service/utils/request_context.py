import uuid

from flask import g, has_request_context, request


def get_request_id() -> str:
    """Get the current request ID, or 'no-request' if outside a request context.

    Propagates X-Request-ID from upstream callers (e.g. backend-service)
    to maintain a single trace ID across the entire request flow.
    Falls back to generating a short local ID if no header is present.
    """
    if has_request_context():
        req_id = getattr(g, "request_id", None)
        if req_id is None:
            req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
            g.request_id = req_id
        return req_id
    return "no-request"
