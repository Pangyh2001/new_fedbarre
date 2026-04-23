#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import pickle
from typing import Dict, Optional


def find_latest_run(base_dir: str, prefix: str) -> Optional[str]:
    pattern = os.path.join(base_dir, f"{prefix}_*")
    candidates = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def read_test_acc(run_dir: str, round_idx: int) -> Optional[float]:
    metric_file = os.path.join(run_dir, "all_metrics_log.txt")
    if not os.path.exists(metric_file):
        return None
    value = None
    with open(metric_file, "r", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if int(row[0]) == round_idx:
                value = float(row[2])
    return value


def read_privacy(run_dir: str, round_idx: int) -> Dict[str, Optional[float]]:
    pkl_file = os.path.join(run_dir, f"dlg_result_E{round_idx}.pkl")
    if not os.path.exists(pkl_file):
        return {"mse": None, "psnr": None}
    with open(pkl_file, "rb") as f:
        obj = pickle.load(f)
    return {
        "mse": float(obj.get("test_mse")) if obj.get("test_mse") is not None else None,
        "psnr": float(obj.get("test_psnr")) if obj.get("test_psnr") is not None else None,
    }


def fmt(v: Optional[float], p: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{p}f}"


def summarize_method(base_dir: str, method_prefix: str, eps: str, round_idx: int):
    run_dir = find_latest_run(base_dir, f"{method_prefix}_eps{eps}")
    if run_dir is None:
        return None
    acc = read_test_acc(run_dir, round_idx)
    privacy = read_privacy(run_dir, round_idx)
    return {
        "run_dir": run_dir,
        "acc": acc,
        "mse": privacy["mse"],
        "psnr": privacy["psnr"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--eps_list", nargs="+", required=True)
    parser.add_argument("--round_idx", type=int, default=29)
    args = parser.parse_args()

    print(f"Round index used for metrics: {args.round_idx}")
    header = f"{'eps':<8} {'method':<12} {'test_acc':>10} {'mse':>10} {'psnr':>10}"
    print(header)
    print("-" * len(header))

    for eps in args.eps_list:
        dp = summarize_method(args.base_dir, "dp", eps, args.round_idx)
        barre = summarize_method(args.base_dir, "barre", eps, args.round_idx)

        if dp is None:
            print(f"{eps:<8} {'DP-Laplace':<12} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            print(f"{eps:<8} {'DP-Laplace':<12} {fmt(dp['acc']):>10} {fmt(dp['mse']):>10} {fmt(dp['psnr']):>10}")

        if barre is None:
            print(f"{eps:<8} {'FedBARRE':<12} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            print(f"{eps:<8} {'FedBARRE':<12} {fmt(barre['acc']):>10} {fmt(barre['mse']):>10} {fmt(barre['psnr']):>10}")


if __name__ == "__main__":
    main()
