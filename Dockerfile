FROM docker.io/pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS builder
WORKDIR /build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
RUN MAX_JOBS=0 uv pip install --system --no-cache "causal-conv1d>=1.4.0" "mamba-ssm>=1.2.0"

FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
CMD ["python3", "train.py", "--help"]
