# syntax=docker/dockerfile:1
#
# Build command:
# env -u http_proxy -u https_proxy buildah bud --network=host --layers -t mamba-train .
#
FROM docker.io/nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

# [优化 1]：安装系统依赖 + Python3
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    git \
    ninja-build \
    gcc \
    g++ \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# [优化 2]：配置 PIP 镜像 + 安装基础工具（使用持久化缓存卷）
RUN --mount=type=cache,id=pip-tools-cache,target=/root/.cache/pip \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --break-system-packages ninja setuptools wheel packaging

# [优化 3]：安装 PyTorch（通过清华镜像获取 nvidia 包，PyPI 下载 torch）
RUN --mount=type=cache,target=/var/tmp/pip \
    pip install --break-system-packages \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        torch torchvision torchaudio

# [优化 4]：编译 causal-conv1d (全核心编译)
RUN --mount=type=cache,target=/var/tmp/pip-install \
    pip install --break-system-packages packaging && \
    git clone -b v1.4.0 --depth 1 https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d && \
    cd /tmp/causal-conv1d && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=$(nproc) pip install --break-system-packages . --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/causal-conv1d

# [优化 5]：编译 mamba (全核心编译)
RUN --mount=type=cache,target=/var/tmp/pip-install \
    git clone -b v2.2.2 --depth 1 https://github.com/state-spaces/mamba.git /tmp/mamba && \
    cd /tmp/mamba && \
    MAMBA_FORCE_BUILD=TRUE MAX_JOBS=$(nproc) pip install --break-system-packages . --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/mamba

# [优化 6]：安装常用训练库
RUN --mount=type=cache,target=/var/tmp/pip \
    pip install --break-system-packages transformers datasets accelerate tensorboard

# 终极验证
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}'); import causal_conv1d; import mamba_ssm; print('Mamba OK')"