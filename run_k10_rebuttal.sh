#!/bin/bash
# K=10 Rebuttal Experiment: FedBARRE vs DP-Laplace on MNIST
# Addresses reviewer concern R1-C3 (narrow setup, only 4 clients)
# Config: K=10 clients, IID, MNIST, 30 epochs, DLG attack on rounds 8-10

GPU=3
DATASET=mnist
N_CLIENTS=10
GLOBAL_EPOCH=30
EPS=0.7
OUT_DIR="runs/k10_rebuttal"

echo "=========================================="
echo "K=10 Rebuttal Experiment"
echo "GPU: $GPU, Dataset: $DATASET, K: $N_CLIENTS"
echo "=========================================="

# --- Experiment 1: FedBARRE (M=3, noise_type=4) ---
echo ""
echo "[1/2] Running FedBARRE (M=3, eps=$EPS) with K=$N_CLIENTS ..."
python main.py \
    --dataset $DATASET \
    --n_clients $N_CLIENTS \
    --global_epoch $GLOBAL_EPOCH \
    --gpu $GPU \
    --out_dir $OUT_DIR \
    --name barre_k10 \
    --nfl "eps=$EPS,privacy=barre,distort=barre,barre_noise_type=2,barre_M=3,dlg=True,known_grad=raw,dlg_attack_epochs=8-9-10,warm_up_rounds=8,data_per_client=1000" \
    --use_rp False \
    --v

echo ""
echo "[1/2] FedBARRE K=10 done."

# --- Experiment 2: DP-Laplace baseline ---
echo ""
echo "[2/2] Running DP-Laplace (eps=$EPS) with K=$N_CLIENTS ..."
python main.py \
    --dataset $DATASET \
    --n_clients $N_CLIENTS \
    --global_epoch $GLOBAL_EPOCH \
    --gpu $GPU \
    --out_dir $OUT_DIR \
    --name dp_k10 \
    --nfl "eps=$EPS,privacy=dp,distort=dp-laplace,clipDP=1.0,dlg=True,known_grad=raw,dlg_attack_epochs=8-9-10,warm_up_rounds=8,data_per_client=1000" \
    --use_rp False \
    --v

echo ""
echo "[2/2] DP-Laplace K=10 done."

echo ""
echo "=========================================="
echo "All K=10 experiments complete!"
echo "Results in: $OUT_DIR"
echo "=========================================="


# # 这是添加优化噪声的。
# #!/bin/bash
# # Script to run FedBarre (barre_noise_type=4) across multiple datasets and privacy budgets

# datasets=(mnist fmnist cifar10)
# # epsilon values from 0.1 to 1.0 with step 0.1
# eps_start=0.1
# eps_end=1.0
# eps_step=0.1

# echo "==== Starting FedBarre4 experiments (barre_noise_type=4) ===="
# for dataset in "${datasets[@]}"; do
#     out_base="runs/fedbarre4_2/M_5/$dataset"
#     mkdir -p "$out_base"
#     echo "\n-- FedBarre4 on $dataset --"

#     eps=$eps_start
#     while (( $(echo "$eps <= $eps_end" | bc -l) )); do
#         eps_fmt=$(printf "%.1f" "$eps")
#         echo "FedBarre4: eps=$eps_fmt"

#         python main.py \
#             --dataset "$dataset" \
#             --nfl "eps=$eps_fmt,privacy=barre,distort=barre,barre_noise_type=4,barre_M=5" \
#             --out_dir "$out_base" \
#             --global_epoch 30 \
#             --use_rp False

#         eps=$(echo "$eps + $eps_step" | bc)
#     done
# done
