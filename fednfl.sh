

#!/bin/bash
# Script to run FedNFL experiments across multiple datasets and privacy budgets

datasets=(cifar10)
# epsilon values from 0.05 to 1.0 with step 0.05
eps_start=0.05
eps_end=0.95
eps_step=0.05

for dataset in "${datasets[@]}"; do
    out_base="runs/fednfl2/$dataset"
    mkdir -p "$out_base"
    echo "\n==== Running experiments on $dataset dataset ===="

    eps=$eps_start
    while (( $(echo "$eps <= $eps_end" | bc -l) )); do
        # Format eps to two decimal places
        eps_fmt=$(printf "%.2f" "$eps")
        echo "Running with eps=$eps_fmt"

        python main.py \
            --dataset "$dataset" \
            --nfl "eps=$eps_fmt,privacy=nfl,distort=nfl,clipDP=1.0" \
            --out_dir "$out_base" \
            --local_epoch 1 \
            --global_epoch 100 \
            --use_rp False

        # Increment epsilon
        eps=$(echo "$eps + $eps_step" | bc)
    done
done
