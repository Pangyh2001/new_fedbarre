#!/usr/bin/env bash
set -euo pipefail

# Compare DP-Laplace vs FedBARRE across multiple eps values after 30 global rounds.
# Usage:
#   bash run_dp_vs_fedbarre_eps30.sh
# Optional env vars:
#   GPU=0 DATASET=mnist N_CLIENTS=4 EPS_LIST="0.3 0.5 0.7" OUT_DIR=runs/eps30_compare

GPU="${GPU:-0}"
DATASET="${DATASET:-mnist}"
N_CLIENTS="${N_CLIENTS:-4}"
GLOBAL_EPOCH="${GLOBAL_EPOCH:-30}"
OUT_DIR="${OUT_DIR:-runs/eps30_compare}"
EPS_LIST="${EPS_LIST:-0.3 0.5 0.7}"

# Keep DLG only on the final round (round index = GLOBAL_EPOCH-1) so that MSE/PSNR correspond to 30 rounds.
FINAL_DLG_EPOCH=$((GLOBAL_EPOCH - 1))
COMMON_CFG="data_per_client=1000,dlg=True,known_grad=noisy,dlg_attack_epochs=${FINAL_DLG_EPOCH},warm_up_rounds=0"

mkdir -p "$OUT_DIR"

echo "==========================================================="
echo "DP vs FedBARRE @ ${GLOBAL_EPOCH} rounds"
echo "dataset=${DATASET}, clients=${N_CLIENTS}, gpu=${GPU}, out_dir=${OUT_DIR}"
echo "eps list: ${EPS_LIST}"
echo "==========================================================="

for eps in $EPS_LIST; do
  echo ""
  echo "[EPS=${eps}] Running DP-Laplace ..."
  python main.py \
    --dataset "$DATASET" \
    --n_clients "$N_CLIENTS" \
    --global_epoch "$GLOBAL_EPOCH" \
    --gpu "$GPU" \
    --out_dir "$OUT_DIR" \
    --name "dp_eps${eps}" \
    --nfl "eps=${eps},privacy=dp,distort=dp-laplace,clipDP=1.0,${COMMON_CFG}" \
    --use_rp False

  echo "[EPS=${eps}] Running FedBARRE ..."
  python main.py \
    --dataset "$DATASET" \
    --n_clients "$N_CLIENTS" \
    --global_epoch "$GLOBAL_EPOCH" \
    --gpu "$GPU" \
    --out_dir "$OUT_DIR" \
    --name "barre_eps${eps}" \
    --nfl "eps=${eps},privacy=barre,distort=barre,barre_noise_type=2,barre_M=5,barre_tau=1.0,${COMMON_CFG}" \
    --use_rp False

done

echo ""
echo "================ Summary (round ${GLOBAL_EPOCH}) ================"
python tools/summarize_eps_runs.py \
  --base_dir "$OUT_DIR" \
  --eps_list $EPS_LIST \
  --round_idx "$FINAL_DLG_EPOCH"
echo "==========================================================="
