#!/bin/bash
# Script to run FedRPF0 (no optimized noise, noise_type=0) across multiple datasets
# with number of random projection filters from 1 to 12.
# 仅RPF无噪声的消融实验


datasets=(mnist fmnist)

echo "==== Starting FedRPF0 experiments without noise (noise_type=0) ===="
for dataset in "${datasets[@]}"; do
    echo -e "\n-- Dataset: $dataset --"
    for rp in {1..12}; do
        # 计算 rp_ratio = rp / 12，保留三位小数
        rp_ratio=$(awk "BEGIN {printf \"%.3f\", $rp/12}")
        out_base="runs/ablation/fedrpf/${dataset}/rp${rp}"
        mkdir -p "$out_base"
        echo "--> Running with $rp random projection filter(s) (rp_ratio=$rp_ratio)"

        python main.py \
            --dataset "$dataset" \
            --use_rp True \
            --nfl "privacy=rpf,distort=rpf" \
            --out_dir "$out_base" \
            --global_epoch 50 \
            --noise_type 0 \
            --rp_ratio "$rp_ratio"

    done
done
