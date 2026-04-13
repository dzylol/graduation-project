# squeez [cat] 367→365 tokens (-1%) 54ms [adaptive: Ultra]
#!/bin/bash
cd /home/qfh/graduation-project
echo "=== Training ESOL pooling=mean ==="
/usr/bin/podman run --rm -v /home/qfh/graduation-project:/workspace --workdir /workspace --device nvidia.com/gpu=all localhost/bimamba python train.py --dataset ESOL --data_dir ./dataset/ESOL --train_file train.csv --val_file val.csv --test_file test.csv --model_type mamba_ssm --task_type regression --d_model 256 --n_layers 4 --pooling mean --epochs 100 --batch_size 32 --learning_rate 1e-3 --device cuda --no_db
echo "=== Training ESOL pooling=max ==="
/usr/bin/podman run --rm -v /home/qfh/graduation-project:/workspace --workdir /workspace --device nvidia.com/gpu=all localhost/bimamba python train.py --dataset ESOL --data_dir ./dataset/ESOL --train_file train.csv --val_file val.csv --test_file test.csv --model_type mamba_ssm --task_type regression --d_model 256 --n_layers 4 --pooling max --epochs 100 --batch_size 32 --learning_rate 1e-3 --device cuda --no_db
echo "=== Training ESOL pooling=cls ==="
/usr/bin/podman run --rm -v /home/qfh/graduation-project:/workspace --workdir /workspace --device nvidia.com/gpu=all localhost/bimamba python train.py --dataset ESOL --data_dir ./dataset/ESOL --train_file train.csv --val_file val.csv --test_file test.csv --model_type mamba_ssm --task_type regression --d_model 256 --n_layers 4 --pooling cls --epochs 100 --batch_size 32 --learning_rate 1e-3 --device cuda --no_db
echo "=== All ESOL pooling experiments complete ==="
