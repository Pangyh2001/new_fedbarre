#!/bin/bash
# K=10 v2 Baselines: DP-Gas, DP-Lap, No Defense (on separate GPU from BARRE)
# BARRE is already running on GPU 6 — run baselines on GPU 5

GPU=5
DATASET=mnist
N_CLIENTS=10
GLOBAL_EPOCH=30
EPS=0.7
OUT_DIR="runs/k10_v2"
COMMON="data_per_client=3000,dlg=True,known_grad=raw,dlg_attack_epochs=8-9-10"

echo "=========================================="
echo "K=10 v2 Baselines — GPU $GPU"
echo "=========================================="

# --- 1. DP-Gaussian ---
echo ""
echo "[1/3] DP-Gaussian (eps=$EPS) ..."
python main.py \
    --dataset $DATASET \
    --n_clients $N_CLIENTS \
    --global_epoch $GLOBAL_EPOCH \
    --gpu $GPU \
    --out_dir $OUT_DIR \
    --name dpgas_k10 \
    --nfl "eps=$EPS,privacy=dp,distort=dp-gaussian,clipDP=1.0,warm_up_rounds=8,$COMMON" \
    --use_rp False \
    --v
echo "[1/3] DP-Gaussian done."

# --- 2. DP-Laplace ---
echo ""
echo "[2/3] DP-Laplace (eps=$EPS) ..."
python main.py \
    --dataset $DATASET \
    --n_clients $N_CLIENTS \
    --global_epoch $GLOBAL_EPOCH \
    --gpu $GPU \
    --out_dir $OUT_DIR \
    --name dplap_k10 \
    --nfl "eps=$EPS,privacy=dp,distort=dp-laplace,clipDP=1.0,warm_up_rounds=8,$COMMON" \
    --use_rp False \
    --v
echo "[2/3] DP-Laplace done."

# --- 3. No Defense (FedAvg) ---
echo ""
echo "[3/3] No Defense (FedAvg) ..."
python main.py \
    --dataset $DATASET \
    --n_clients $N_CLIENTS \
    --global_epoch $GLOBAL_EPOCH \
    --gpu $GPU \
    --out_dir $OUT_DIR \
    --name nodefense_k10 \
    --nfl "eps=$EPS,privacy=barre,distort=barre,barre_noise_type=2,barre_M=3,barre_k_noise=3,warm_up_rounds=31,$COMMON" \
    --use_rp False \
    --v
echo "[3/3] No Defense done."

echo ""
echo "=========================================="
echo "All baselines complete!"
echo "=========================================="
