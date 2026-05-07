#!/bin/sh

# Cluster (CUDA) explore: FedAvg baseline on Fashion-MNIST, pathological non-IID, sudden drift (rounds 40-60).
# Switched from Dirichlet to pathological partitioning (matching the MNIST setup) to compare
# how loss-based selection behaves under categorical vs continuous corruption heterogeneity.
: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "FMNIST_NON_IID_SUD_FEDAVG${EXP_SUFFIX}" --seed ${SEED} --device cuda \
    --dataset FashionMNIST \
    --split_type patho --test_size 0 \
    --model_name TwoCNN --resize 28 --hidden_size 200 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision recall \
    --K 100 --R 100 --E 3 --C 0.1 --B 10 \
    --optimizer SGD --lr 0.1 --lr_decay 1 --lr_decay_step 25 --criterion CrossEntropyLoss \
    --concept_drift --drift_mode sudden --drift_start 40 --drift_duration 20 \
    --max_grad_norm 10 \
    --no_save_model --no_save_results
