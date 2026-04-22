#!/bin/sh

# FedAvg baseline on MNIST, pathological non-IID, sudden label-swap drift (rounds 40-60).
# C=0.1 (10/100 clients trained per round, uniformly sampled). Identical to c_small/FEDAVG
# — baseline held at 10 trained clients to match the active-sampling variants' training count.
: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "MNIST_NON_IID_SUD_FEDAVG${EXP_SUFFIX}" --seed ${SEED} --device cuda \
    --dataset MNIST \
    --split_type patho --test_size 0 \
    --model_name TwoCNN --resize 28 --hidden_size 200 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision recall \
    --K 100 --R 100 --E 2 --C 0.1 --B 10 \
    --optimizer SGD --lr 0.1 --lr_decay 1 --lr_decay_step 25 --criterion CrossEntropyLoss \
    --concept_drift --drift_mode sudden --drift_start 40 --drift_duration 20 \
    --max_grad_norm 10 \
    --no_save_model --no_save_results
