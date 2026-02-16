from dataclasses import dataclass, field
from typing import Any, Optional


def success_response(data: Any = None) -> dict:
    return {"code": "0000", "message": "Operation successful.", "data": data}


def error_response(code: str, message: str, detail: str = None) -> dict:
    resp = {"code": code, "message": message}
    if detail:
        resp["detail"] = detail
    return resp
