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
MAX_PARALLEL="${MAX_PARALLEL:-0}" # 0 means auto (= number of visible GPUs from GPU_LIST)

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
  local method="$2"   # dp | barre
  local run_gpu="$3"

  echo ""
  if [[ "$method" == "dp" ]]; then
    echo "[EPS=${eps}] Running DP-Laplace on GPU ${run_gpu} ..."
    python main.py \
      --dataset "$DATASET" \
      --n_clients "$N_CLIENTS" \
      --global_epoch "$GLOBAL_EPOCH" \
      --gpu "$run_gpu" \
      --out_dir "$OUT_DIR" \
      --name "dp_eps${eps}" \
      --nfl "eps=${eps},privacy=dp,distort=dp-laplace,clipDP=1.0,${COMMON_CFG}" \
      --use_rp False
  else
    echo "[EPS=${eps}] Running FedBARRE on GPU ${run_gpu} ..."
    python main.py \
      --dataset "$DATASET" \
      --n_clients "$N_CLIENTS" \
      --global_epoch "$GLOBAL_EPOCH" \
      --gpu "$run_gpu" \
      --out_dir "$OUT_DIR" \
      --name "barre_eps${eps}" \
      --nfl "eps=${eps},privacy=barre,distort=barre,barre_noise_type=2,barre_M=5,barre_tau=1.0,${COMMON_CFG}" \
      --use_rp False
  fi
}

job_pids=()
job_desc=()
job_fail=0
gpu_idx=0

launch_job() {
  local eps="$1"
  local method="$2"
  local assigned_gpu="${GPU_ARR[$((gpu_idx % ${#GPU_ARR[@]}))]}"
  gpu_idx=$((gpu_idx + 1))

  run_one "$eps" "$method" "$assigned_gpu" &
  job_pids+=("$!")
  job_desc+=("eps=${eps},method=${method},gpu=${assigned_gpu}")
}

for eps in $EPS_LIST; do
  launch_job "$eps" "dp"
  while [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    sleep 1
  done
  launch_job "$eps" "barre"
  while [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    sleep 1
  done
done

for i in "${!job_pids[@]}"; do
  if ! wait "${job_pids[$i]}"; then
    echo "[ERROR] job failed: ${job_desc[$i]}"
    job_fail=1
  fi
done

if [[ "$job_fail" -ne 0 ]]; then
  echo "[ERROR] One or more runs failed; skip summary."
  exit 1
fi

echo ""
echo "================ Summary (round ${GLOBAL_EPOCH}) ================"
python tools/summarize_eps_runs.py \
  --base_dir "$OUT_DIR" \
  --eps_list $EPS_LIST \
  --round_idx "$FINAL_DLG_EPOCH"
echo "==========================================================="
