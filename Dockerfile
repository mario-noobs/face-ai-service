FROM python:3.10-slim-bookworm AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application code
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Create model cache directory
RUN mkdir -p /app/model_cache

ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=120s \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--preload", \
     "face_ai_service.app:create_app()"]
