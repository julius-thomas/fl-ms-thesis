: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "FedAvg_CheXpert_ResNet18_Vanilla${EXP_SUFFIX}" --seed ${SEED} --device cuda --bf16 \
    --dataset CheXpert --data_path ./external-data --rawsmpl 1.0 \
    --split_type custom --cncntrtn 0.3 --test_size 0.0 \
    --randhf 0.5 --randrot 10 \
    --model_name ResNet18 --resize 224 --hidden_size 64 --imnorm --init_type kaiming \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics mlacc mlauroc \
    --K 100 --R 150 --E 1 --C 0.1 --B 64 --beta1 0 \
    --concept_drift --drift_mode custom --drift_start 100 --drift_duration 1 \
    --optimizer Adam --lr 0.0001 --lr_decay 1.0 --lr_decay_step 10 --criterion BCEWithLogitsLoss \
    --no_save_model --no_save_results
