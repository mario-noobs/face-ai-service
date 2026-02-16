# Face AI Service

Stateless face detection and recognition inference microservice. Provides HTTP endpoints for encoding face images into embeddings and searching against candidate lists.

## Architecture

This service is **stateless** — it owns no database or cache. The backend-service owns all state (MySQL) and passes candidates to this service for comparison.

```
Backend (MySQL) ──→ face-ai-service (encode / search) ──→ S3/MinIO (model weights)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/face/encode` | Detect and encode a face from a base64 image |
| `POST` | `/api/v1/face/search` | Encode a face and compare against candidate embeddings |
| `GET` | `/ping` | Liveness check |
| `GET` | `/health` | Readiness check (verifies models loaded) |

## Supported Algorithms

| Type | Algorithm | Default |
|------|-----------|---------|
| Detection | `retinaface_mobilenet` | Yes |
| Detection | `retinaface_resnet50` | No |
| Recognition | `facenet_mobilenet` | Yes |
| Recognition | `facenet_inception_resnetv1` | No |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run in development
FLASK_ENV=development python -m face_ai_service.app

# Run with gunicorn
gunicorn "face_ai_service.app:create_app()" -b 0.0.0.0:5000
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `production` | Flask environment |
| `MODEL_S3_ENDPOINT` | `http://localhost:9000` | S3/MinIO endpoint |
| `MODEL_S3_ACCESS_KEY` | `admin` | S3 access key |
| `MODEL_S3_SECRET_KEY` | `123456789` | S3 secret key |
| `MODEL_S3_BUCKET` | `face-models` | S3 bucket for model weights |
| `MODEL_CACHE_DIR` | `/app/model_cache` | Local cache directory for downloaded models |
| `ENABLED_MODELS` | `retinaface_mobilenet,facenet_mobilenet` | Comma-separated model pairs to load |
| `USE_CUDA` | `false` | Enable GPU inference |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format (`json` or `text`) |

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Docker

```bash
docker build -t face-ai-service .
docker run -p 5000:5000 \
  -e MODEL_S3_ENDPOINT=http://minio:9000 \
  -e MODEL_S3_ACCESS_KEY=admin \
  -e MODEL_S3_SECRET_KEY=123456789 \
  face-ai-service
```

## Related Repositories

- [face-microservice](https://github.com/mario-noobs/face-microservice) — Parent orchestration repo
- [face-ai-lab](https://github.com/mario-noobs/face-ai-lab) — Training and evaluation tooling
