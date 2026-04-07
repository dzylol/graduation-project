# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="12.0;12.0+ptx" \
    CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    FORCE_CUDA=1 \
    MAX_JOBS=12

WORKDIR /workspace

# 1. 系统依赖
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
        ninja-build gcc g++ python3-pip python3-dev \
        libxrender1 libxext6 libfontconfig1 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# 2. 安装 PyTorch
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install setuptools wheel packaging ninja pyyaml && \
    pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 \
                --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
                torch torchvision torchaudio

# 3. 安装 causal-conv1d
COPY causal-conv1d-v1.4.0.tar.gz /tmp/
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    cd /tmp && \
    tar -xzf causal-conv1d-v1.4.0.tar.gz && \
    cd causal-conv1d-1.4.0 && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    pip install --no-build-isolation --no-deps . && \
    cd / && rm -rf /tmp/causal-conv1d*

# 4. 安装 mamba-ssm
COPY mamba-v2.2.2.tar.gz /tmp/
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    cd /tmp && \
    tar -xzf mamba-v2.2.2.tar.gz && \
    cd mamba-2.2.2 && \
    MAMBA_FORCE_BUILD=TRUE \
    pip install --no-build-isolation --no-deps . && \
    cd / && rm -rf /tmp/mamba*
# 5. 安装分子预测常用库
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install transformers==4.38.2 datasets accelerate tensorboard \
                scikit-learn rdkit pandas matplotlib tqdm einops
# 6. 验证
RUN python3 -c "import torch; import mamba_ssm; import causal_conv1d; \
    from rdkit import Chem; \
    print(f'CUDA: {torch.version.cuda}'); print('SUCCESS!')"

CMD ["/bin/bash"]
