FROM python:3.13.14-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY pyproject.toml ./
COPY app/ ./app/
COPY experiments/ ./experiments/

RUN python -m pip wheel \
    --no-cache-dir \
    --wheel-dir /wheels \
    '.[api]'


FROM python:3.13.14-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PODCAST_API_HOST=0.0.0.0

RUN groupadd --system --gid 10001 podcast \
    && useradd --system --uid 10001 --gid podcast \
        --home-dir /nonexistent --shell /usr/sbin/nologin podcast

COPY --from=builder /wheels /wheels

RUN python -m pip install \
    --no-cache-dir \
    --no-index \
    --find-links=/wheels \
    'podcast-ai-assistant[api]==0.1.0' \
    && rm -rf /wheels

WORKDIR /app

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.environ.get('PODCAST_API_PORT', '8000'); response = urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2); raise SystemExit(0 if response.status == 200 else 1)"]

CMD ["python", "-m", "app.api"]
