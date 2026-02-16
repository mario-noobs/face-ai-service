import hashlib
import logging
import os
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


MODEL_REGISTRY = {
    "detection": {
        "retinaface_mobilenet": {
            "s3_key": "models/Retinaface_mobilenet0.25.pth",
            "local_name": "Retinaface_mobilenet0.25.pth",
        },
        "retinaface_resnet50": {
            "s3_key": "models/Retinaface_resnet50.pth",
            "local_name": "Retinaface_resnet50.pth",
        },
    },
    "recognition": {
        "facenet_mobilenet": {
            "s3_key": "models/facenet_mobilenet.pth",
            "local_name": "facenet_mobilenet.pth",
        },
        "facenet_inception_resnetv1": {
            "s3_key": "models/facenet_inception_resnetv1.pth",
            "local_name": "facenet_inception_resnetv1.pth",
        },
    },
}


class ModelStore:
    def __init__(self, config):
        self.cache_dir = Path(config.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bucket = config.MODEL_S3_BUCKET

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=config.MODEL_S3_ENDPOINT,
            aws_access_key_id=config.MODEL_S3_ACCESS_KEY,
            aws_secret_access_key=config.MODEL_S3_SECRET_KEY,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def _download_file(self, s3_key: str, local_path: Path) -> None:
        logger.info("Downloading model: %s -> %s", s3_key, local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket, s3_key, str(local_path))
        logger.info("Downloaded model: %s (%.1f MB)", s3_key, local_path.stat().st_size / 1e6)

    def get_model_path(self, model_type: str, algorithm: str) -> str:
        """Get local path for a model, downloading from S3 if not cached."""
        registry = MODEL_REGISTRY.get(model_type, {})
        model_info = registry.get(algorithm)
        if model_info is None:
            raise ValueError(f"Unknown {model_type} algorithm: {algorithm}")

        local_path = self.cache_dir / model_info["local_name"]

        if not local_path.exists():
            try:
                self._download_file(model_info["s3_key"], local_path)
            except ClientError as e:
                logger.error("Failed to download model %s: %s", model_info["s3_key"], e)
                raise RuntimeError(
                    f"Failed to download model {algorithm} from S3: {e}"
                ) from e

        return str(local_path)

    def ensure_models(self, enabled_models: str) -> dict:
        """Download all enabled models. Returns dict of {algorithm: local_path}."""
        pairs = [m.strip() for m in enabled_models.split(",") if m.strip()]
        paths = {}

        for pair in pairs:
            for model_type, registry in MODEL_REGISTRY.items():
                if pair in registry:
                    paths[pair] = self.get_model_path(model_type, pair)

        return paths
