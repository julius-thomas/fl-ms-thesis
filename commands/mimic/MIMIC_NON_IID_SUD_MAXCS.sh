: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "FedAvg_MIMIC4_LogReg_MaxCS${EXP_SUFFIX}" --seed ${SEED} --device mps \
    --dataset MIMIC4 --data_path ./external-data --rawsmpl 0.25 \
    --split_type custom --cncntrtn 0.3 --test_size 0.2 \
    --model_name LogReg --num_layers 1 --hidden_size 64 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 auroc auprc \
    --K 20 --R 150 --E 1 --C 0.5 --B 64 --beta1 0 \
    --active_sampling --sampling_fraction 0.5 --sampling_type max \
    --concept_drift --drift_mode custom --drift_start 75 --drift_duration 0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.99 --lr_decay_step 5 --criterion CrossEntropyLoss \
    --no_save_model --no_save_results
