from flask import Blueprint, jsonify, current_app

health_bp = Blueprint("health", __name__)


@health_bp.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@health_bp.route("/health", methods=["GET"])
def health():
    engine = current_app.engine
    algorithms = engine.get_loaded_algorithms()

    models_ok = bool(algorithms["detection"]) and bool(algorithms["recognition"])

    status = "healthy" if models_ok else "unhealthy"
    http_status = 200 if models_ok else 503

    return jsonify({
        "status": status,
        "checks": {
            "models": {
                "status": "ok" if models_ok else "error",
                "detection": algorithms["detection"],
                "recognition": algorithms["recognition"],
            }
        },
    }), http_status
