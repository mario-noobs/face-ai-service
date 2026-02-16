import logging

from flask import Flask

from face_ai_service.api.error_handlers import register_error_handlers
from face_ai_service.api.v1 import face_bp, health_bp
from face_ai_service.config import get_config
from face_ai_service.core.inference_engine import InferenceEngine
from face_ai_service.storage.model_store import ModelStore
from face_ai_service.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(config_name: str = None) -> Flask:
    config = get_config(config_name)

    configure_logging(config.LOG_LEVEL, config.LOG_FORMAT)

    app = Flask(__name__)
    app.config.from_object(config)

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(face_bp)

    # Download models and initialize inference engine
    logger.info("Initializing model store and downloading models...")
    model_store = ModelStore(config)
    model_store.ensure_models(config.ENABLED_MODELS)

    logger.info("Initializing inference engine...")
    app.engine = InferenceEngine(model_store, config)

    logger.info("Face AI Service initialized successfully")
    return app
