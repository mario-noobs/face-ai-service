#!/usr/bin/env python3
"""Download model weights from S3/MinIO to local cache."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from face_ai_service.config import get_config
from face_ai_service.storage.model_store import ModelStore
from face_ai_service.utils.logging_config import configure_logging


def main():
    config = get_config()
    configure_logging(config.LOG_LEVEL, "text")

    print(f"Downloading models from {config.MODEL_S3_ENDPOINT}/{config.MODEL_S3_BUCKET}")
    print(f"Enabled models: {config.ENABLED_MODELS}")
    print(f"Cache dir: {config.MODEL_CACHE_DIR}")

    store = ModelStore(config)
    paths = store.ensure_models(config.ENABLED_MODELS)

    print(f"\nDownloaded {len(paths)} models:")
    for name, path in paths.items():
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {name}: {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
