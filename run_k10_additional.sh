#!/bin/bash
# K=10 additional experiments: No-defense baseline + BARRE at different eps
GPU=3

echo "=== Additional K=10 Experiments ==="

# --- Experiment A: No defense baseline ---
echo "[A] Running no-defense baseline (FedAvg, no privacy) K=10..."
python main.py \
    --dataset mnist \
    --n_clients 10 \
    --global_epoch 30 \
    --gpu $GPU \
    --out_dir runs/k10_rebuttal \
    --name nodefense_k10 \
    --nfl "eps=0.7,privacy=barre,distort=no,dlg=True,known_grad=raw,dlg_attack_epochs=8-9-10,warm_up_rounds=0,data_per_client=1000" \
    --use_rp False \
    --v
echo "[A] No-defense done."

# --- Experiment B: BARRE eps=0.3 (stronger privacy) ---
echo "[B] Running BARRE eps=0.3 K=10..."
python main.py \
    --dataset mnist \
    --n_clients 10 \
    --global_epoch 30 \
    --gpu $GPU \
    --out_dir runs/k10_rebuttal \
    --name barre_k10_eps03 \
    --nfl "eps=0.3,privacy=barre,distort=barre,barre_noise_type=2,barre_M=3,dlg=True,known_grad=raw,dlg_attack_epochs=8-9-10,warm_up_rounds=8,data_per_client=1000" \
    --use_rp False \
    --v
echo "[B] BARRE eps=0.3 done."

# --- Experiment C: BARRE eps=0.5 ---
echo "[C] Running BARRE eps=0.5 K=10..."
python main.py \
    --dataset mnist \
    --n_clients 10 \
    --global_epoch 30 \
    --gpu $GPU \
    --out_dir runs/k10_rebuttal \
    --name barre_k10_eps05 \
    --nfl "eps=0.5,privacy=barre,distort=barre,barre_noise_type=2,barre_M=3,dlg=True,known_grad=raw,dlg_attack_epochs=8-9-10,warm_up_rounds=8,data_per_client=1000" \
    --use_rp False \
    --v
echo "[C] BARRE eps=0.5 done."

echo "=== All additional experiments complete ==="
