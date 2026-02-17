import functools
import logging
import time

from face_ai_service.utils.request_context import get_request_id

logger = logging.getLogger(__name__)


def traceable(name: str):
    """Decorator for method-level structured logging with timing.

    Usage:
        @traceable("engine.encode_face")
        def encode_face(self, image_np, det_algorithm, reg_algorithm):
            ...

    Logs at DEBUG level:
        >>> ENTER [engine.encode_face] args={...}
        <<< EXIT  [engine.encode_face] duration=245ms
        !!! ERROR [engine.encode_face] duration=12ms exception=ValueError
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            request_id = get_request_id()
            extra = kwargs.get("_trace_extra", "")
            if "_trace_extra" in kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k != "_trace_extra"}

            logger.debug(
                ">>> ENTER [%s] %srequest_id=%s",
                name,
                f"{extra} " if extra else "",
                request_id,
            )

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000

                result_summary = kwargs.get("_trace_result", "")
                if "_trace_result" in kwargs:
                    kwargs = {k: v for k, v in kwargs.items() if k != "_trace_result"}

                logger.debug(
                    "<<< EXIT  [%s] duration=%.0fms request_id=%s",
                    name,
                    duration_ms,
                    request_id,
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    "!!! ERROR [%s] duration=%.0fms exception=%s: %s request_id=%s",
                    name,
                    duration_ms,
                    type(e).__name__,
                    str(e),
                    request_id,
                )
                raise

        return wrapper

    return decorator
