import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.get_loaded_algorithms.return_value = {
        "detection": ["retinaface_mobilenet"],
        "recognition": ["facenet_mobilenet"],
    }
    return engine


@pytest.fixture
def app(mock_engine):
    """Create a test Flask app with mocked inference engine."""
    from face_ai_service.app import Flask
    from face_ai_service.api.error_handlers import register_error_handlers
    from face_ai_service.api.v1 import face_bp, health_bp

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.engine = mock_engine

    register_error_handlers(app)
    app.register_blueprint(health_bp)
    app.register_blueprint(face_bp)

    return app


@pytest.fixture
def client(app):
    return app.test_client()
