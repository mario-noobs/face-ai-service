from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EncodeRequest:
    imageBase64: str = ""
    algorithmDet: str = "retinaface_mobilenet"
    algorithmReg: str = "facenet_mobilenet"

    @classmethod
    def from_dict(cls, data: dict) -> "EncodeRequest":
        return cls(
            imageBase64=data.get("imageBase64", ""),
            algorithmDet=data.get("algorithmDet", "retinaface_mobilenet"),
            algorithmReg=data.get("algorithmReg", "facenet_mobilenet"),
        )


@dataclass
class CandidateEntry:
    userId: str = ""
    encoding: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateEntry":
        return cls(
            userId=data.get("userId", ""),
            encoding=data.get("encoding", ""),
        )


@dataclass
class SearchRequest:
    imageBase64: str = ""
    algorithmDet: str = "retinaface_mobilenet"
    algorithmReg: str = "facenet_mobilenet"
    candidates: list = field(default_factory=list)
    threshold: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "SearchRequest":
        candidates = [
            CandidateEntry.from_dict(c)
            for c in data.get("candidates", [])
        ]
        threshold = data.get("threshold")
        if threshold is not None:
            threshold = float(threshold)
        return cls(
            imageBase64=data.get("imageBase64", ""),
            algorithmDet=data.get("algorithmDet", "retinaface_mobilenet"),
            algorithmReg=data.get("algorithmReg", "facenet_mobilenet"),
            candidates=candidates,
            threshold=threshold,
        )
