#!/bin/bash
# CheXpert FL — Dirichlet non-IID (concentration 0.5)
# ResNet18 (hidden_size=32), 224x224 input, 25% dataset, 3 local epochs

python3 main.py \
    --exp_name "FedAvg_CheXpert_ResNet18_Diri05" --seed 42 --device cuda \
    --dataset CheXpert --data_path ./external-data \
    --split_type diri --test_size 0.2 --rawsmpl 0.25 --cncntrtn 0.5 --mincls 2 \
    --model_name ResNet18 --resize 224 --hidden_size 32 --imnorm --init_type kaiming \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 5 --eval_metrics mlacc mlauroc \
    --K 50 --R 100 --E 3 --C 0.2 --B 16 --beta1 0 \
    --optimizer Adam --lr 0.0001 --lr_decay 0.99 --lr_decay_step 10 --criterion BCEWithLogitsLoss
