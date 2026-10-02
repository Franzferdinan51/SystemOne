# SystemOne shim image: slim CPU by default, CUDA stage for GPU judges.
#
#   docker build -t systemone:0.2.0 .
#   docker run -p 8765:8765 systemone:0.2.0
#   docker build --target cuda -t systemone:0.2.0-cuda .   # + torch/GLiClass

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SYSTEMONE_ENGINE=auto \
    SYSTEMONE_SHIM_URL=http://127.0.0.1:8765
WORKDIR /app
COPY pyproject.toml README.md ./
COPY systemone/ ./systemone/
# Slim install: numpy-only runtime (remote engines + ONNX judge ready).
RUN pip install --no-cache-dir .
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s CMD \
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=4)"
CMD ["python", "-m", "systemone.cli", "serve", "--port", "8765"]

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS cuda
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    SYSTEMONE_ENGINE=local
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY systemone/ ./systemone/
RUN pip install --no-cache-dir '.[local]'
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=10s CMD \
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=8)"
CMD ["python3", "-m", "systemone.cli", "serve", "--port", "8765"]
