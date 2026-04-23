#!/bin/bash
# K=10 Rebuttal Experiment v2: Matching actual paper experiment parameters
# Addresses reviewer concern R1-C3 / R3-C10 (narrow setup, only 4 clients)
#
# Parameters verified against actual experiment logs (test_noShuf_* runs):
#   - model_optim=adam, lr=0.01 (code defaults, used in all paper experiments)
#   - data_per_client=3000 (paper default, v1 erroneously used 1000)
#   - barre_k_noise=3 (code default, used in paper experiments)
#   - barre_M=3, noise_type=2, warm_up_rounds=8, dlg_attack_epochs=8-9-10
#   - global_epoch=30, batch_size=8, n_clients=10
#
# Note: Paper text states "SGD lr=0.1, 5 PGD steps" but all actual experiment
# logs in this codebase used Adam lr=0.01, k_noise=3. Using actual parameters
# ensures K=10 results are directly comparable to K=4 results in Table 2.

GPU=6
DATASET=mnist
N_CLIENTS=10
GLOBAL_EPOCH=30
EPS=0.7
OUT_DIR="runs/k10_v2"

# Common --nfl params matching actual paper experiments
COMMON="data_per_client=3000,dlg=True,known_grad=raw,dlg_attack_epochs=8-9-10"

echo "=========================================="
echo "K=10 Rebuttal v2 — Actual Paper Params"
echo "GPU: $GPU | Dataset: $DATASET | K: $N_CLIENTS"
echo "Adam lr=0.01 | data_per_client=3000 | k_noise=3"
echo "=========================================="

# --- 1. FedBARRE (M=3, eps=0.7, noise_type=2, k_noise=3) ---
echo ""
echo "[1/4] FedBARRE (M=3, eps=$EPS, k_noise=3) ..."
python main.py \
    --dataset $DATASET \
    --n_clients $N_CLIENTS \
    --global_epoch $GLOBAL_EPOCH \
    --gpu $GPU \
    --out_dir $OUT_DIR \
    --name barre_k10 \
    --nfl "eps=$EPS,privacy=barre,distort=barre,barre_noise_type=2,barre_M=3,barre_k_noise=3,warm_up_rounds=8,$COMMON" \
    --use_rp False \
    --v
echo "[1/4] FedBARRE done."

# --- 2. DP-Gaussian (eps=0.7) ---
echo ""
echo "[2/4] DP-Gaussian (eps=$EPS) ..."
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
echo "[2/4] DP-Gaussian done."

# --- 3. DP-Laplace (eps=0.7) ---
echo ""
echo "[3/4] DP-Laplace (eps=$EPS) ..."
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
echo "[3/4] DP-Laplace done."

# --- 4. No Defense (FedAvg) ---
echo ""
echo "[4/4] No Defense (FedAvg) ..."
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
echo "[4/4] No Defense done."

echo ""
echo "=========================================="
echo "All K=10 v2 experiments complete!"
echo "Results in: $OUT_DIR"
echo "=========================================="
