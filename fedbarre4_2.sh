#!/bin/bash
# FedBarre4 on MNIST only, 10 clients, 1000 data per client

eps_start=0.2
eps_end=0.2
eps_step=0.1

out_base="runs/fedbarre4_2_2/M_5/mnist"
mkdir -p "$out_base"

eps=$eps_start
while (( $(echo "$eps <= $eps_end" | bc -l) )); do
    eps_fmt=$(printf "%.1f" "$eps")
    echo "FedBarre4: eps=$eps_fmt"

    python main.py \
        --dataset mnist \
        --nfl "eps=$eps_fmt,privacy=barre,distort=barre,barre_noise_type=2,barre_M=5" \
        --out_dir "$out_base" \
        --global_epoch 30 \
        --use_rp False \
        --n_clients 10 \
        --gpu 7

    eps=$(echo "$eps + $eps_step" | bc)
done
