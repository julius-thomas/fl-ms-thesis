#!/bin/sh

# Local smoke test: UCB-CS-LIT (Cho, Gupta, Joshi, Yagan 2020 - Algorithm 1)
# on MNIST pathological non-IID with sudden label-swap drift (rounds 40-60).
# Full-size MNIST, MPS shaders for Mac-local iteration.
: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "MNIST_NON_IID_SUD_UCBCS_LIT_TEST${EXP_SUFFIX}" --seed ${SEED} --device mps \
    --dataset MNIST \
    --split_type patho --test_size 0 \
    --model_name TwoCNN --resize 28 --hidden_size 200 \
    --algorithm ucbcs_lit --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision recall \
    --K 100 --R 100 --E 2 --C 0.1 --B 10 \
    --optimizer SGD --lr 0.1 --lr_decay 1 --lr_decay_step 25 --criterion CrossEntropyLoss \
    --use_tb \
    --concept_drift --drift_mode sudden --drift_start 40 --drift_duration 20 \
    --ucb_gamma 0.9 \
    --max_grad_norm 10 \
    --no_save_model --no_save_results
