#!/usr/bin/env bash
set -euo pipefail

# Compare DP-Laplace vs FedBARRE across multiple eps values after 30 global rounds.
# Usage:
#   bash run_dp_vs_fedbarre_eps30.sh
# Optional env vars:
#   GPU=0 DATASET=mnist N_CLIENTS=4 EPS_LIST="0.3 0.5 0.7" OUT_DIR=runs/eps30_compare

GPU="${GPU:-0}"
GPU_LIST="${GPU_LIST:-$GPU}" # e.g. "0 1 2" or "0,1,2"
DATASET="${DATASET:-mnist}"
N_CLIENTS="${N_CLIENTS:-4}"
GLOBAL_EPOCH="${GLOBAL_EPOCH:-30}"
OUT_DIR="${OUT_DIR:-runs/eps30_compare}"
EPS_LIST="${EPS_LIST:-0.3 0.5 0.7}"


# Keep DLG only on the final round (round index = GLOBAL_EPOCH-1) so that MSE/PSNR correspond to 30 rounds.
FINAL_DLG_EPOCH=$((GLOBAL_EPOCH - 1))
COMMON_CFG="data_per_client=1000,dlg=True,known_grad=noisy,dlg_attack_epochs=${FINAL_DLG_EPOCH},warm_up_rounds=0"

mkdir -p "$OUT_DIR"

IFS=', ' read -r -a GPU_ARR <<< "$GPU_LIST"
if [[ "${#GPU_ARR[@]}" -eq 0 ]]; then
  GPU_ARR=("$GPU")
fi
if [[ "$MAX_PARALLEL" -eq 0 ]]; then
  MAX_PARALLEL="${#GPU_ARR[@]}"
fi
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  MAX_PARALLEL=1
fi

echo "==========================================================="
echo "DP vs FedBARRE @ ${GLOBAL_EPOCH} rounds"
echo "dataset=${DATASET}, clients=${N_CLIENTS}, gpu_list=${GPU_LIST}, out_dir=${OUT_DIR}"
echo "eps list: ${EPS_LIST}"
echo "max parallel jobs: ${MAX_PARALLEL}"
echo "==========================================================="

run_one() {
  local eps="$1"


echo ""
echo "================ Summary (round ${GLOBAL_EPOCH}) ================"
python tools/summarize_eps_runs.py \
  --base_dir "$OUT_DIR" \
  --eps_list $EPS_LIST \
  --round_idx "$FINAL_DLG_EPOCH"
echo "==========================================================="
