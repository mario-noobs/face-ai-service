import os


class BaseConfig:
    # S3/MinIO model storage
    MODEL_S3_ENDPOINT = os.getenv("MODEL_S3_ENDPOINT", "http://localhost:9000")
    MODEL_S3_ACCESS_KEY = os.getenv("MODEL_S3_ACCESS_KEY", "admin")
    MODEL_S3_SECRET_KEY = os.getenv("MODEL_S3_SECRET_KEY", "123456789")
    MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET", "face-models")

    # Model cache
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")

    # Model selection
    ENABLED_MODELS = os.getenv(
        "ENABLED_MODELS", "retinaface_mobilenet,facenet_mobilenet"
    )

    # RetinaFace settings
    RETINAFACE_INPUT_SHAPE = [640, 640, 3]
    RETINAFACE_CONFIDENCE = 0.5
    RETINAFACE_NMS_IOU = 0.3
    LETTERBOX_IMAGE = True

    # FaceNet settings
    FACENET_INPUT_SHAPE = [160, 160, 3]
    FACENET_THRESHOLD = float(os.getenv("FACENET_THRESHOLD", "1.05"))

    # CUDA
    USE_CUDA = os.getenv("USE_CUDA", "false").lower() == "true"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "json")


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    LOG_FORMAT = "text"
    LOG_LEVEL = "DEBUG"


class TestingConfig(BaseConfig):
    TESTING = True
    MODEL_CACHE_DIR = "/tmp/test_model_cache"
    LOG_FORMAT = "text"
    LOG_LEVEL = "DEBUG"


class ProductionConfig(BaseConfig):
    pass


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(config_name: str = None) -> BaseConfig:
    name = config_name or os.getenv("FLASK_ENV", "production")
    return config_by_name.get(name, ProductionConfig)()
