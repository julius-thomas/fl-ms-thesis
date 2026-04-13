python3 main.py \
    --exp_name "FedAvg_MIMIC4_LogReg_IID" --seed 42 --device mps \
    --dataset MIMIC4 --data_path ./external-data --rawsmpl 0.25 \
    --split_type iid --test_size 0.2 \
    --model_name LogReg --num_layers 1 --hidden_size 64 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 \
    --K 5 --R 20 --E 1 --C 1.0 --B 64 --beta1 0 \
    --optimizer SGD --lr 0.01 --lr_decay 0.99 --lr_decay_step 5 --criterion CrossEntropyLoss \
    --use_tb --tb_host localhost --tb_port 6007