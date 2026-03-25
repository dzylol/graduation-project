# Bi-Mamba-Chem Training TODO

## 镜像构建

- [ ] 拉取基础镜像 `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
  ```bash
  podman pull docker.io/pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
  ```

- [ ] 构建训练镜像（含 mamba-ssm，全核心编译）
  ```bash
  cd ~/graduation-project
  podman build -t bimamba-train:latest .
  ```

- [ ] 验证镜像构建成功
  ```bash
  podman images | grep bimamba
  ```

- [ ] 验证 GPU 访问
  ```bash
  podman run --rm --gpus all bimamba-train:latest nvidia-smi
  ```

## 模型训练

### Manual 模型（无需 mamba-ssm）

- [ ] 运行 manual 模型训练
  ```bash
  podman run --gpus all \
    -v ~/graduation-project:/app:rw \
    bimamba-train:latest \
    python train.py --dataset ESOL --epochs 100 --batch_size 32 --device cuda --model_type manual
  ```

### mamba_ssm 模型（需要 mamba-ssm）

- [ ] 运行 mamba_ssm 模型训练
  ```bash
  podman run --gpus all \
    -v ~/graduation-project:/app:rw \
    bimamba-train:latest \
    python train.py --dataset ESOL --epochs 100 --batch_size 32 --device cuda --model_type mamba_ssm
  ```

### 其他数据集

可选数据集：BBBP, ClinTox

```bash
# BBBP
podman run --gpus all -v ~/graduation-project:/app:rw bimamba-train:latest \
  python train.py --dataset BBBP --epochs 100 --batch_size 32 --device cuda --model_type manual

# ClinTox
podman run --gpus all -v ~/graduation-project:/app:rw bimamba-train:latest \
  python train.py --dataset ClinTox --epochs 100 --batch_size 32 --device cuda --model_type manual
```

## 训练后验证

- [ ] 检查检查点文件
  ```bash
  ls -la ~/graduation-project/checkpoints/
  ```

- [ ] 对比两个模型的训练结果（loss, MAE, RMSE 等指标）
