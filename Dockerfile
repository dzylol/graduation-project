FROM docker.io/pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir causal-conv1d>=1.4.0 mamba-ssm --no-build-isolation
# 使用 conda 安装 rdkit（更稳定）
RUN conda install -c conda-forge rdkit scikit-learn -y && conda clean -a
COPY . .
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
CMD ["python", "train.py", "--help"]
