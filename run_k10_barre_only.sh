#!/bin/bash
# K=10 BARRE-only rerun with noise_type=2 (optimized PGD noise)
GPU=3
echo "[BARRE K=10] Starting with noise_type=2 (optimized PGD noise) ..."
python main.py \
    --dataset mnist \
    --n_clients 10 \
    --global_epoch 30 \
    --gpu $GPU \
    --out_dir runs/k10_rebuttal \
    --name barre_k10_v2 \
    --nfl "eps=0.7,privacy=barre,distort=barre,barre_noise_type=2,barre_M=3,dlg=True,known_grad=raw,dlg_attack_epochs=8-9-10,warm_up_rounds=8,data_per_client=1000" \
    --use_rp False \
    --v
echo "[BARRE K=10] Done."
