: "${SEED:=42}"
: "${EXP_SUFFIX:=}"

python3 main.py \
    --exp_name "FedAvg_CheXpert_ResNet18_MaxCS${EXP_SUFFIX}" --seed ${SEED} --device cuda --bf16 \
    --dataset CheXpert --data_path ./external-data --rawsmpl 1.0 \
    --split_type custom --cncntrtn 0.3 --test_size 0.2 \
    --randhf 0.5 --randrot 10 \
    --model_name ResNet18 --resize 224 --hidden_size 64 --imnorm --init_type kaiming \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics mlacc mlauroc \
    --K 20 --R 150 --E 2 --C 0.5 --B 64 --beta1 0 --num_workers 4 \
    --active_sampling --sampling_fraction 0.5 --sampling_type max \
    --concept_drift --drift_mode custom --drift_start 75 --drift_duration 0 \
    --optimizer Adam --lr 0.0001 --lr_decay 1.0 --lr_decay_step 10 --criterion BCEWithLogitsLoss \
    --no_save_model --no_save_results
