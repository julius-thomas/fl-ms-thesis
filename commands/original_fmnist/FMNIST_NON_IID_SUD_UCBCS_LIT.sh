#!/bin/sh

# UCB-CS-LIT (Cho, Gupta, Joshi, Yagan 2020 - Algorithm 1) baseline on Fashion-MNIST
# pathological non-IID with sudden label-swap drift (rounds 60-80).
# Matches FEDAVG settings; only the algorithm and gamma differ.
# No active_sampling / candidate_sampling - UCB-CS-LIT picks m clients directly.
: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "FMNIST_NON_IID_SUD_UCBCS_LIT${EXP_SUFFIX}" --seed ${SEED} --device cuda \
    --dataset FashionMNIST \
    --split_type patho --test_size 0 \
    --model_name TwoCNN --resize 28 --hidden_size 200 \
    --algorithm ucbcs_lit --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision recall \
    --K 100 --R 100 --E 3 --C 0.1 --B 10 \
    --optimizer SGD --lr 0.1 --lr_decay 1 --lr_decay_step 25 --criterion CrossEntropyLoss \
    --concept_drift --drift_mode sudden --drift_start 60 --drift_duration 20 \
    --ucb_gamma 0.9 \
    --max_grad_norm 10 \
    --no_save_model --no_save_results
