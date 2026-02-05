python3 main.py \
    --exp_name "FedAvg_MNIST_2NN_IID_C10_B0.2" --seed 42 --device mps \
    --dataset MNIST \
    --split_type iid --test_size 0 \
    --model_name TwoNN --resize 28 --hidden_size 200 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
    --K 100 --R 1000 --E 1 --C 0.2 --B 10 --beta1 0 \
    --optimizer SGD --lr 0.1 --lr_decay 0.99 --lr_decay_step 25 --criterion CrossEntropyLoss