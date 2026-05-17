#!/bin/sh

# UCB-CS-LIT (Cho, Gupta, Joshi, Yagan 2020 - Algorithm 1) baseline on MNIST
# pathological non-IID with incremental hard-blur drift (rounds 40-60).
# Paper-aligned settings: K=100, C=0.03 (m=3 trained per round), gamma=0.9.
: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "MNIST_NON_IID_INC_UCBCS_LIT${EXP_SUFFIX}" --seed ${SEED} --device cuda \
    --dataset MNIST \
    --split_type patho --test_size 0 \
    --model_name TwoCNN --resize 28 --hidden_size 200 \
    --algorithm ucbcs_lit --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision recall \
    --K 100 --R 100 --E 2 --C 0.03 --B 10 \
    --optimizer SGD --lr 0.1 --lr_decay 1 --lr_decay_step 25 --criterion CrossEntropyLoss \
    --concept_drift --drift_mode hard --drift_start 40 --drift_duration 20 \
    --ucb_gamma 0.9 \
    --max_grad_norm 10 \
    --no_save_model --no_save_results
