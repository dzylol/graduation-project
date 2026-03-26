FROM docker.io/pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
WORKDIR /app

COPY requirements.txt .
RUN conda install --yes -c conda-forge causal-conv1d mamba-ssm && \
    conda clean -afy
COPY . .
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
CMD ["python", "train.py", "--help"]
