#!/usr/bin/env python3
"""Analyze K=10 v2 experiment results (paper-matched parameters)"""

import pickle
import numpy as np
import os
import json
import re

def load_dlg_results(run_dir, epochs=[8, 9, 10]):
    """Load DLG results from a run directory.
    Pickle files contain pre-computed: test_mse, test_psnr, test_ssim, rec_img, gt"""
    results = []
    for epoch in epochs:
        pkl_path = os.path.join(run_dir, f'dlg_result_E{epoch}.pkl')
        if os.path.exists(pkl_path):
            d = pickle.load(open(pkl_path, 'rb'))
            d['epoch'] = epoch
            results.append(d)
    return results

def extract_accuracy_from_log(run_dir):
    """Extract best average test accuracy across all clients from log.txt.
    Returns (best_avg_acc, best_epoch, acc_trajectory).
    acc_trajectory maps epoch -> average accuracy across clients."""
    log_path = os.path.join(run_dir, 'log.txt')
    if not os.path.exists(log_path):
        return None, None, None

    acc_trajectory = {}
    # Also check for final summary lines (after "Federated Learning Finish!")
    final_accs = {}

    with open(log_path, 'r') as f:
        current_epoch = -1
        epoch_accs = []
        in_final = False
        for line in f:
            if 'Federated Learning Finish' in line:
                in_final = True
                continue

            if in_final:
                m = re.search(r'client:\s*(\d+), test acc:([0-9.]+), best epoch:\s*(\d+)$', line)
                if m:
                    final_accs[int(m.group(1))] = float(m.group(2))
                continue

            m = re.search(r'Training Epoch (\d+)', line)
            if m:
                if current_epoch >= 0 and epoch_accs:
                    avg_acc = np.mean(epoch_accs)
                    acc_trajectory[current_epoch] = avg_acc
                current_epoch = int(m.group(1))
                epoch_accs = []

            m = re.search(r'test acc:([0-9.]+), best epoch: (\d+)', line)
            if m:
                acc = float(m.group(1))
                epoch_accs.append(acc)

        # Last epoch
        if current_epoch >= 0 and epoch_accs:
            acc_trajectory[current_epoch] = np.mean(epoch_accs)

    # Use final summary if available (per-client best), else trajectory
    if final_accs:
        best_avg_acc = np.mean(list(final_accs.values()))
        best_epoch = -1  # from final summary
    elif acc_trajectory:
        best_epoch = max(acc_trajectory, key=acc_trajectory.get)
        best_avg_acc = acc_trajectory[best_epoch]
    else:
        return None, None, None

    return best_avg_acc, best_epoch, acc_trajectory

def extract_global_test_acc(run_dir):
    """Extract global test accuracy from log.txt"""
    log_path = os.path.join(run_dir, 'log.txt')
    if not os.path.exists(log_path):
        return None, None

    best_global_acc = 0
    best_epoch = 0

    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(r'Global Test Acc:\s*([0-9.]+)', line)
            if m:
                acc = float(m.group(1))
                if acc > best_global_acc:
                    best_global_acc = acc

            m = re.search(r'train epoch round:(\d+)', line)
            if m:
                current_epoch = int(m.group(1))

    return best_global_acc, best_epoch

def compute_metrics(gt, rec):
    """Compute MSE, PSNR, SSIM for batch"""
    mse = np.mean((gt - rec) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))

    # Simple SSIM approximation
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu_gt = np.mean(gt)
    mu_rec = np.mean(rec)
    sigma_gt = np.var(gt)
    sigma_rec = np.var(rec)
    sigma_gt_rec = np.mean((gt - mu_gt) * (rec - mu_rec))
    ssim = ((2 * mu_gt * mu_rec + C1) * (2 * sigma_gt_rec + C2)) / \
           ((mu_gt**2 + mu_rec**2 + C1) * (sigma_gt + sigma_rec + C2))

    return mse, psnr, ssim

def main():
    base_dir = 'runs/k10_v2'

    # Auto-discover run directories
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist")
        return

    dirs = sorted(os.listdir(base_dir))
    configs = {}
    for d in dirs:
        full_path = os.path.join(base_dir, d)
        if not os.path.isdir(full_path):
            continue
        # Check for DLG results to pick the correct run (skip crashed runs)
        has_dlg = os.path.exists(os.path.join(full_path, 'dlg_result_E8.pkl'))
        if d.startswith('barre_k10'):
            configs['FedBARRE'] = full_path
        elif d.startswith('dpgas_k10'):
            # DP-Gaussian crashed (NotImplementedError) — skip
            pass
        elif d.startswith('dplap_k10'):
            configs['DP-Laplace'] = full_path
        elif d.startswith('nodefense_k10') and has_dlg:
            configs['No Defense'] = full_path  # Use the run with DLG results

    print("=" * 80)
    print("K=10 v2 Results (Paper-Matched Parameters)")
    print("Adam lr=0.01 | data_per_client=3000 | k_noise=3 | n_clients=10")
    print("=" * 80)

    # 1. Accuracy
    print("\n## 1. Test Accuracy")
    print("-" * 60)
    print(f"{'Method':<15} {'Best Avg Acc':>12} {'Best Epoch':>12}")
    print("-" * 60)

    acc_results = {}
    for name, run_dir in configs.items():
        best_acc, best_ep, trajectory = extract_accuracy_from_log(run_dir)
        if best_acc is not None:
            print(f"{name:<15} {best_acc*100:>11.2f}% {best_ep:>12}")
            acc_results[name] = {'best_acc': best_acc, 'best_epoch': best_ep, 'trajectory': trajectory}

    # 2. Privacy metrics (DLG)
    print("\n## 2. Privacy Metrics (DLG Attack, avg over epochs 8-10)")
    print("-" * 70)
    print(f"{'Method':<15} {'MSE (up)':>10} {'PSNR (dn)':>10} {'SSIM (dn)':>10}")
    print("-" * 70)

    privacy_results = {}
    for name, run_dir in configs.items():
        dlg_results = load_dlg_results(run_dir)
        if not dlg_results:
            print(f"{name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue

        mses, psnrs, ssims = [], [], []
        epoch_details = []
        for r in dlg_results:
            # Use pre-computed metrics from pickle files
            mse = float(r['test_mse'])
            psnr = float(r['test_psnr'])
            ssim = float(r['test_ssim'])
            mses.append(mse)
            psnrs.append(psnr)
            ssims.append(ssim)
            epoch_details.append({
                'epoch': r['epoch'], 'mse': mse, 'psnr': psnr, 'ssim': ssim
            })

        if mses:
            avg_mse = np.mean(mses)
            avg_psnr = np.mean(psnrs)
            avg_ssim = np.mean(ssims)
            print(f"{name:<15} {avg_mse:>10.4f} {avg_psnr:>10.4f} {avg_ssim:>10.4f}")
            privacy_results[name] = {
                'avg_mse': float(avg_mse),
                'avg_psnr': float(avg_psnr),
                'avg_ssim': float(avg_ssim),
                'per_epoch': epoch_details
            }

    # 3. Per-epoch breakdown
    print("\n## 3. Per-Epoch DLG Breakdown")
    print("-" * 70)
    print(f"{'Method':<15} {'Epoch':>6} {'MSE':>10} {'PSNR':>10} {'SSIM':>10}")
    print("-" * 70)
    for name, res in privacy_results.items():
        for ep in res['per_epoch']:
            print(f"{name:<15} {ep['epoch']:>6} {ep['mse']:>10.4f} {ep['psnr']:>10.4f} {ep['ssim']:>10.4f}")
        print()

    # 4. Comparison with paper's K=4 results
    print("\n## 4. Comparison with Paper K=4 Results (Table 2, eps=0.7)")
    print("-" * 70)
    paper_k4 = {
        'FedBARRE': {'acc': 93.32, 'mse': 2.030, 'psnr': 7.28, 'ssim': 0.023},
        'DP-Gaussian': {'acc': 87.34, 'mse': 1.410, 'psnr': 9.11, 'ssim': 0.081},
        'DP-Laplace': {'acc': 87.70, 'mse': 1.399, 'psnr': 9.28, 'ssim': 0.085},
        'No Defense': {'acc': 96.80, 'mse': 1.37, 'psnr': 9.44, 'ssim': 0.12},
    }
    print(f"{'Method':<15} {'Metric':<8} {'K=4 (paper)':>12} {'K=10 (ours)':>12} {'Change':>10}")
    print("-" * 70)
    for name in ['FedBARRE', 'DP-Gaussian', 'DP-Laplace', 'No Defense']:
        if name in paper_k4 and name in acc_results:
            k4_acc = paper_k4[name]['acc']
            k10_acc = acc_results[name]['best_acc'] * 100
            print(f"{name:<15} {'Acc%':<8} {k4_acc:>12.2f} {k10_acc:>12.2f} {k10_acc-k4_acc:>+10.2f}")
        if name in paper_k4 and name in privacy_results:
            for metric in ['mse', 'psnr', 'ssim']:
                k4_val = paper_k4[name][metric]
                k10_val = privacy_results[name][f'avg_{metric}']
                print(f"{'':<15} {metric.upper():<8} {k4_val:>12.4f} {k10_val:>12.4f} {k10_val-k4_val:>+10.4f}")
        print()

    # 5. Save structured results
    output = {
        'experiment': 'K=10 v2 paper-matched',
        'parameters': {
            'n_clients': 10,
            'global_epoch': 30,
            'batch_size': 8,
            'model_optim': 'adam',
            'lr': 0.01,
            'data_per_client': 3000,
            'barre_k_noise': 3,
            'barre_M': 3,
            'barre_noise_type': 2,
            'warm_up_rounds': 8,
            'eps': 0.7,
        },
        'accuracy': {k: {'best_acc': v['best_acc'], 'best_epoch': v['best_epoch']}
                     for k, v in acc_results.items()},
        'privacy': privacy_results,
        'paper_k4_comparison': paper_k4,
    }

    out_path = os.path.join('..', 'review-stage', 'k10_v2_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
