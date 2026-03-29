FROM docker.io/nvidia/cuda:13.2.0-cudnn-devel-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    ninja-build git wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

RUN python3.11 -m pip install --no-cache-dir causal-conv1d>=1.4.0

RUN python3.11 -m pip install --no-cache-dir mamba-ssm

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

RUN python3.11 -m pip install --no-cache-dir rdkit-pypi scikit-learn
RUN python3.11 -m pip install --no-cache-dir "numpy<2"

COPY . .
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

# Verify critical dependencies
RUN python3.11 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" && \
    python3.11 -c "import mamba_ssm; print('mamba-ssm: OK')" && \
    python3.11 -c "from rdkit import Chem; print('RDKit: OK')"

CMD ["python3.11", "train.py", "--help"]
