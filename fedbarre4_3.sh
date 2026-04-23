# 这是添加优化噪声的。
#!/bin/bash
# Script to run FedBarre (barre_noise_type=4) across multiple datasets and privacy budgets

datasets=(mnist fmnist cifar10)
# epsilon values from 0.1 to 1.0 with step 0.1
eps_start=0.1
eps_end=1.0
eps_step=0.1

echo "==== Starting FedBarre4 experiments (barre_noise_type=4) ===="
for dataset in "${datasets[@]}"; do
    out_base="runs/fedbarre4_2/M_9/$dataset"
    mkdir -p "$out_base"
    echo "\n-- FedBarre4 on $dataset --"

    eps=$eps_start
    while (( $(echo "$eps <= $eps_end" | bc -l) )); do
        eps_fmt=$(printf "%.1f" "$eps")
        echo "FedBarre4: eps=$eps_fmt"

        python main.py \
            --nfl "eps=0.7,privacy=barre,distort=barre,barre_noise_type=4,barre_M=3" \
            --use_rp False
        
        python main.py \
            --nfl "eps=0.7,privacy=barre,distort=barre,barre_noise_type=4,barre_M=3" \
            --use_rp False

        eps=$(echo "$eps + $eps_step" | bc)
    done
done
