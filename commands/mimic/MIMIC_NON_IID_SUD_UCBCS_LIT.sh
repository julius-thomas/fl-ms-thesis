# UCB-CS-LIT (Cho, Gupta, Joshi, Yagan 2020 - Algorithm 1) baseline on MIMIC4
# custom (care-unit) non-IID with custom (ICD-9 -> ICD-10) sudden drift at round 400.
# Paper-aligned settings: gamma=0.9, C=0.03 per user choice. NOTE: K=20 with C=0.03 yields
# m=1 (clamped), which means sigma_max stays at the initial 1.0 since std needs >=2 samples;
# bump --C if you want a richer bandit signal here.
: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "MIMIC_NON_IID_SUD_UCBCS_LIT${EXP_SUFFIX}" --seed ${SEED} --device mps \
    --dataset MIMIC4 --data_path ./external-data --rawsmpl 1 \
    --split_type custom --cncntrtn 0.3 --test_size 0.0 \
    --model_name LogReg --num_layers 1 --hidden_size 64 \
    --algorithm ucbcs_lit --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 auroc auprc \
    --K 20 --R 600 --E 1 --C 0.03 --B 256 --beta1 0 \
    --concept_drift --drift_mode custom --drift_start 400 --drift_duration 0 \
    --ucb_gamma 0.9 \
    --optimizer SGD --lr 0.01 --lr_decay 1 --lr_decay_step 5 --criterion CrossEntropyLoss \
    --no_save_model --no_save_results
