#!/bin/sh

# UCB candidate sampling + max final selection on MNIST non-IID sudden drift.
# Candidate set small: C=0.2 (20/100 via UCB1), sampling_fraction=0.5 → train 10 highest-loss per round.
python3 main.py \
    --exp_name MNIST_NON_IID_SUD_UCBCS_CSMALL --seed 42 --device cuda \
    --dataset MNIST \
    --split_type patho --test_size 0 \
    --model_name TwoCNN --resize 28 --hidden_size 200 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision recall \
    --K 100 --R 100 --E 3 --C 0.2 --B 10 \
    --optimizer SGD --lr 0.1 --lr_decay 1 --lr_decay_step 25 --criterion CrossEntropyLoss \
    --use_tb \
    --concept_drift --drift_mode sudden --drift_start 40 --drift_duration 20 \
    --active_sampling --sampling_fraction 0.5 --sampling_type max \
    --candidate_sampling ucb --ucb_c 1.0
