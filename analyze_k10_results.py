#!/usr/bin/env python3
"""Analyze K=10 experiment results: visual reconstructions + eps calibration"""

import pickle
import numpy as np
import os
import json

def load_dlg_results(run_dir, epochs=[8, 9, 10]):
    """Load DLG results from a run directory"""
    results = []
    for epoch in epochs:
        pkl_path = os.path.join(run_dir, f'dlg_result_E{epoch}.pkl')
        if os.path.exists(pkl_path):
            d = pickle.load(open(pkl_path, 'rb'))
            d['epoch'] = epoch
            results.append(d)
    return results

def compute_per_sample_metrics(gt, rec):
    """Compute per-sample MSE for each image in the batch"""
    batch_size = gt.shape[0]
    mses = []
    for i in range(batch_size):
        mse = np.mean((gt[i] - rec[i]) ** 2)
        mses.append(mse)
    return mses

def main():
    configs = {
        'No Defense': 'runs/k10_rebuttal/nodefense_k10_noShuf_C12.0_eps0.7_lba10_zeta1e-05_1776533576',
        'DP-Laplace': 'runs/k10_rebuttal/dp_k10_noShuf_C12.0_batch_eps0.7_1776529902',
        'BARRE_eps03': 'runs/k10_rebuttal/barre_k10_eps03_noShuf_C12.0_eps0.3_lba10_zeta1e-05_1776534704',
        'BARRE_eps07': 'runs/k10_rebuttal/barre_k10_v2_noShuf_C12.0_eps0.7_lba10_zeta1e-05_1776530858',
    }

    all_results = {}
    print("=" * 80)
    print("K=10 Comprehensive Results Analysis")
    print("=" * 80)

    # 1. Per-epoch per-sample analysis
    print("\n## 1. Per-Epoch Detailed Metrics")
    print("-" * 80)

    for name, run_dir in configs.items():
        results = load_dlg_results(run_dir)
        if not results:
            print(f"  {name}: No DLG results found")
            continue

        all_results[name] = results
        print(f"\n  {name}:")
        for r in results:
            gt = r['gt']  # (batch, C, H, W)
            rec = r['rec_img']
            per_sample_mse = compute_per_sample_metrics(gt, rec)
            print(f"    E{r['epoch']}: MSE={r['test_mse']:.4f}, PSNR={r['test_psnr']:.4f}, "
                  f"SSIM={r['test_ssim']:.4f}")
            print(f"           Per-sample MSE: {[f'{m:.4f}' for m in per_sample_mse]}")
            print(f"           MSE std={np.std(per_sample_mse):.4f}, "
                  f"min={np.min(per_sample_mse):.4f}, max={np.max(per_sample_mse):.4f}")

    # 2. Aggregated comparison
    print("\n\n## 2. Aggregated Comparison (avg over epochs 8-10)")
    print("-" * 80)
    print(f"  {'Method':>20s} | {'MSE':>8s} | {'PSNR':>8s} | {'SSIM':>8s} | {'Privacy Gain':>12s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")

    baseline_mse = None
    for name, results in all_results.items():
        avg_mse = np.mean([r['test_mse'] for r in results])
        avg_psnr = np.mean([r['test_psnr'] for r in results])
        avg_ssim = np.mean([r['test_ssim'] for r in results])
        if baseline_mse is None:
            baseline_mse = avg_mse
            baseline_psnr = avg_psnr
            gain = "baseline"
        else:
            psnr_gain = (baseline_psnr - avg_psnr) / baseline_psnr * 100
            gain = f"PSNR -{psnr_gain:.1f}%"
        print(f"  {name:>20s} | {avg_mse:>8.4f} | {avg_psnr:>8.4f} | {avg_ssim:>8.4f} | {gain:>12s}")

    # 3. Eps calibration analysis
    print("\n\n## 3. Perturbation Bounds Analysis (eps calibration)")
    print("-" * 80)

    eps_values = [0.3, 0.5, 0.7, 0.9]
    for eps in eps_values:
        l = 100 - 30.0 * eps
        u = l + 25.0
        # MNIST image: 1x28x28 = 784 dimensions
        # Relative perturbation = noise_norm / image_norm
        # Typical MNIST pixel in [0,1], so ||x||_2 ~ sqrt(784 * 0.3^2) ~ 8.4 (rough)
        img_norm_estimate = 8.4  # rough L2 norm of normalized MNIST image
        rel_pert = l / img_norm_estimate
        print(f"  eps={eps:.1f}: l={l:.1f}, u={u:.1f}, "
              f"relative_perturbation={rel_pert:.1f}x image norm")

    print(f"\n  PROBLEM: All perturbation norms ({100-30*0.7:.0f} to {100-30*0.3:.0f}) >> image norm (~8.4)")
    print(f"  This explains why eps=0.3 and eps=0.7 give similar results:")
    print(f"  Both perturbations are so large they completely destroy the original signal.")
    print(f"  The defense is effective but eps is NOT a meaningful privacy knob in this regime.")

    # 4. Reconstruction quality assessment
    print("\n\n## 4. Reconstruction Quality Assessment")
    print("-" * 80)

    for name, results in all_results.items():
        for r in results:
            gt = r['gt']
            rec = r['rec_img']
            # Check if reconstruction looks like original
            gt_mean = np.mean(gt)
            rec_mean = np.mean(rec)
            gt_std = np.std(gt)
            rec_std = np.std(rec)
            correlation = np.corrcoef(gt.flatten(), rec.flatten())[0, 1]
            print(f"  {name} E{r['epoch']}: GT(mean={gt_mean:.3f}, std={gt_std:.3f}), "
                  f"Rec(mean={rec_mean:.3f}, std={rec_std:.3f}), corr={correlation:.4f}")

    # 5. Save structured results for review
    summary = {
        'experiment': 'K=10 MNIST IID DLG Attack',
        'methods': {},
    }
    for name, results in all_results.items():
        summary['methods'][name] = {
            'avg_mse': float(np.mean([r['test_mse'] for r in results])),
            'avg_psnr': float(np.mean([r['test_psnr'] for r in results])),
            'avg_ssim': float(np.mean([r['test_ssim'] for r in results])),
            'per_epoch': [{
                'epoch': r['epoch'],
                'mse': float(r['test_mse']),
                'psnr': float(r['test_psnr']),
                'ssim': float(r['test_ssim']),
            } for r in results]
        }

    os.makedirs('review-stage', exist_ok=True)
    with open('review-stage/k10_results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nResults saved to review-stage/k10_results_summary.json")


if __name__ == '__main__':
    main()
