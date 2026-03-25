python3 main.py \
    --exp_name "FedAvg_CheXpert_ResNet10_IID_tiny" --seed 42 --device mps \
    --dataset CheXpert --data_path ./downloads \
    --split_type iid --test_size 0.2 --rawsmpl 0.05 \
    --model_name ResNet18 --resize 112 --hidden_size 12 --imnorm --init_type kaiming \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics mlacc \
    --K 50 --R 50 --E 1 --C 0.2 --B 16 --beta1 0 \
    --optimizer Adam --lr 0.001 --lr_decay 1.0 --lr_decay_step 5 --criterion BCEWithLogitsLoss \
    --use_tb
