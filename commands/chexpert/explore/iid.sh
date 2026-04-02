#!/bin/bash

python3 main.py \
    --exp_name "FedAvg_CheXpert_ResNet18_IID" --seed 42 --device cuda --bf16 \
    --dataset CheXpert --data_path ./external-data \
    --split_type iid --test_size 0.2 --rawsmpl 1.0 \
    --randhf 0.5 --randrot 10 \
    --model_name ResNet18 --resize 224 --hidden_size 64 --imnorm --init_type kaiming \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics mlacc mlauroc \
    --K 10 --R 100 --E 2 --C 0.5 --B 16 --beta1 0 \
    --optimizer Adam --lr 0.0001 --lr_decay 1. --lr_decay_step 10 --criterion BCEWithLogitsLoss
