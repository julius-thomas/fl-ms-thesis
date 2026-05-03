#!/bin/sh

# UCB candidate sampling + stoch final selection on MNIST non-IID incremental (hard blur) drift,
# with a sliding-window UCB of 5 rounds: any client not pulled in the last 5
# rounds is treated as "unvisited" and force-included in the next candidate pool.
: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "MNIST_NON_IID_INC_UCBCS_W5${EXP_SUFFIX}" --seed ${SEED} --device cuda \
    --dataset MNIST \
    --split_type patho --test_size 0 \
    --model_name TwoCNN --resize 28 --hidden_size 200 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 precision recall \
    --K 100 --R 100 --E 3 --C 0.4 --B 10 \
    --optimizer SGD --lr 0.1 --lr_decay 1 --lr_decay_step 25 --criterion CrossEntropyLoss \
    --concept_drift --drift_mode hard --drift_start 40 --drift_duration 20 \
    --active_sampling --sampling_fraction 0.25 --sampling_type stoch \
    --candidate_sampling ucb --ucb_c 1.0 --ucb_window 5 \
    --max_grad_norm 10 \
    --no_save_model --no_save_results
