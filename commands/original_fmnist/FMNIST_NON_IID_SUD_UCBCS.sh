#!/bin/sh

# UCB candidate sampling + max final selection on Fashion-MNIST Dirichlet non-IID sudden drift.
# Candidate set big: C=0.4 (40/100 via UCB1), sampling_fraction=0.25 → train 10 highest-loss per round.
: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "FMNIST_NON_IID_SUD_UCBCS${EXP_SUFFIX}" --seed ${SEED} --device cuda \
    --dataset FashionMNIST \
    --split_type diri --test_size 0 --cncntrtn 0.1 --mincls 2 \
    --model_name TwoCNN --resize 28 --hidden_size 200 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision recall \
    --K 100 --R 100 --E 3 --C 0.4 --B 10 \
    --optimizer SGD --lr 0.1 --lr_decay 1 --lr_decay_step 25 --criterion CrossEntropyLoss \
    --concept_drift --drift_mode sudden --drift_start 40 --drift_duration 20 \
    --active_sampling --sampling_fraction 0.25 --sampling_type stoch \
    --candidate_sampling ucb --ucb_c 1.0 \
    --max_grad_norm 10 \
    --no_save_model --no_save_results
