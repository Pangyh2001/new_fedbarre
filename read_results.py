#!/usr/bin/env python3
"""读取实验结果的工具脚本。

用法:
    python read_results.py <run_dir>                    # 单个实验
    python read_results.py <run_dir1> <run_dir2> ...    # 多个实验对比
    python read_results.py runs/fedbarre4_2/M_5/mnist/  # 扫描目录下所有实验
"""
import os
import sys
import re
import pickle
import numpy as np
from collections import OrderedDict


def parse_config(log_path):
    """从 log.txt 解析实验配置"""
    config = {}
    with open(log_path) as f:
        for line in f:
            if 'Namespace' not in line:
                continue
            params = [
                'eps', 'apply_distortion', 'clipDP', 'warm_up_rounds',
                'data_per_client', 'n_clients', 'global_epoch', 'lr',
                'batch_size', 'model_optim', 'dlg_know_grad',
                'barre_M', 'barre_noise_type', 'barre_k_noise',
            ]
            for p in params:
                m = re.search(rf"{p}=([^,)\s]+)", line)
                if m:
                    val = m.group(1).strip("'\"")
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    config[p] = val
            break
    return config


def read_accuracy(run_dir):
    """读取 best_metric.txt 和 all_metrics_log.txt"""
    result = {}

    bm_path = os.path.join(run_dir, 'best_metric.txt')
    if os.path.exists(bm_path):
        with open(bm_path) as f:
            parts = f.read().strip().split(',')
        result['best_epoch'] = int(parts[0])
        result['best_val_acc'] = float(parts[1])
        result['best_test_acc'] = float(parts[2])

    am_path = os.path.join(run_dir, 'all_metrics_log.txt')
    if os.path.exists(am_path):
        epochs = []
        with open(am_path) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    epochs.append({
                        'epoch': int(parts[0]),
                        'val_acc': float(parts[1]),
                        'test_acc': float(parts[2]),
                    })
        result['epoch_history'] = epochs
        if epochs:
            result['final_epoch'] = epochs[-1]['epoch']
            result['final_test_acc'] = epochs[-1]['test_acc']

    return result


def read_dlg(run_dir):
    """读取 DLG 隐私指标"""
    dlg_results = []
    for ep in range(50):
        pkl_path = os.path.join(run_dir, f'dlg_result_E{ep}.pkl')
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        dlg_results.append({
            'epoch': ep,
            'mse': data['test_mse'],
            'psnr': data['test_psnr'],
            'ssim': data['test_ssim'],
            'feat_mse': data.get('feat_mse', None),
        })

    if not dlg_results:
        return {}

    return {
        'per_epoch': dlg_results,
        'avg_mse': np.mean([r['mse'] for r in dlg_results]),
        'avg_psnr': np.mean([r['psnr'] for r in dlg_results]),
        'avg_ssim': np.mean([r['ssim'] for r in dlg_results]),
    }


def read_run(run_dir):
    """读取单个实验的全部结果"""
    result = {'dir': run_dir, 'name': os.path.basename(run_dir)}

    log_path = os.path.join(run_dir, 'log.txt')
    if os.path.exists(log_path):
        result['config'] = parse_config(log_path)

    result['accuracy'] = read_accuracy(run_dir)
    result['dlg'] = read_dlg(run_dir)

    return result


def infer_method(result):
    """从配置推断方法名"""
    cfg = result.get('config', {})
    distort = cfg.get('apply_distortion', '')
    if distort == 'barre':
        M = cfg.get('barre_M', '?')
        return f"BARRE(M={M})"
    elif distort == 'dp-laplace':
        return 'DP-Laplace'
    elif distort == 'dp-gaussian':
        return 'DP-Gaussian'
    elif distort == 'no':
        return 'No Defense'
    elif distort == 'nfl':
        return 'NFL'
    elif distort == 'rpf':
        return 'RPF'
    return distort or result['name']


def print_single(result):
    """打印单个实验结果"""
    method = infer_method(result)
    cfg = result.get('config', {})
    acc = result.get('accuracy', {})
    dlg = result.get('dlg', {})

    print(f"{'=' * 70}")
    print(f"Method: {method}")
    print(f"Dir:    {result['dir']}")
    print(f"{'=' * 70}")

    if cfg:
        print("\n[Config]")
        for k, v in cfg.items():
            print(f"  {k}: {v}")

    if acc:
        print("\n[Accuracy]")
        if 'best_test_acc' in acc:
            print(f"  Best test acc: {acc['best_test_acc']*100:.2f}% (epoch {acc['best_epoch']})")
        if 'epoch_history' in acc:
            hist = acc['epoch_history']
            warmup = cfg.get('warm_up_rounds', 8)
            warmup_accs = [e['test_acc'] for e in hist if e['epoch'] < warmup]
            post_accs = [e['test_acc'] for e in hist if e['epoch'] >= warmup]
            if warmup_accs:
                print(f"  Warm-up acc:   {warmup_accs[0]*100:.2f}% -> {warmup_accs[-1]*100:.2f}% (epoch 0-{warmup-1})")
            if post_accs:
                print(f"  Post-warmup:   {post_accs[0]*100:.2f}% -> {post_accs[-1]*100:.2f}% (epoch {warmup}-{hist[-1]['epoch']})")

    if dlg:
        print("\n[DLG Privacy Metrics]  (MSE↑ PSNR↓ SSIM↓ = better privacy)")
        print(f"  {'Epoch':<8} {'MSE':>8} {'PSNR':>8} {'SSIM':>8}")
        for r in dlg['per_epoch']:
            print(f"  E{r['epoch']:<6} {r['mse']:>8.4f} {r['psnr']:>8.4f} {r['ssim']:>8.4f}")
        print(f"  {'Avg':<8} {dlg['avg_mse']:>8.4f} {dlg['avg_psnr']:>8.4f} {dlg['avg_ssim']:>8.4f}")

    print()


def print_comparison(results):
    """打印多个实验的对比表格"""
    print(f"\n{'=' * 90}")
    print("COMPARISON TABLE")
    print(f"{'=' * 90}")

    # Header
    methods = [infer_method(r) for r in results]
    eps_list = [r.get('config', {}).get('eps', '?') for r in results]
    labels = [f"{m} (ε={e})" for m, e in zip(methods, eps_list)]

    col_w = max(20, max(len(l) for l in labels) + 2)
    header = f"{'Metric':<16}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    # Accuracy
    accs = [r.get('accuracy', {}).get('best_test_acc', None) for r in results]
    row = f"{'Test Acc':<16}"
    for a in accs:
        row += f"{f'{a*100:.2f}%' if a else 'N/A':>{col_w}}"
    print(row)

    # DLG metrics
    has_dlg = any(r.get('dlg') for r in results)
    if has_dlg:
        for metric, key, better in [('MSE ↑', 'avg_mse', 'max'), ('PSNR ↓', 'avg_psnr', 'min'), ('SSIM ↓', 'avg_ssim', 'min')]:
            vals = [r.get('dlg', {}).get(key, None) for r in results]
            valid = [v for v in vals if v is not None]
            best = (max if better == 'max' else min)(valid) if valid else None
            row = f"{metric:<16}"
            for v in vals:
                s = f"{v:.4f}" if v is not None else "N/A"
                if v is not None and v == best and len(valid) > 1:
                    s = f"*{s}"
                row += f"{s:>{col_w}}"
            print(row)

    print(f"\n* = best on that metric")

    # Config diff
    all_keys = set()
    for r in results:
        all_keys.update(r.get('config', {}).keys())
    diff_keys = []
    for k in sorted(all_keys):
        vals = [r.get('config', {}).get(k, None) for r in results]
        if len(set(str(v) for v in vals)) > 1:
            diff_keys.append(k)
    if diff_keys:
        print(f"\n[Config differences]")
        for k in diff_keys:
            row = f"  {k:<20}"
            for r in results:
                v = r.get('config', {}).get(k, '-')
                row += f"{str(v):>{col_w}}"
            print(row)


def find_runs(path):
    """在目录下查找所有包含 log.txt 的实验目录"""
    runs = []
    if os.path.isfile(os.path.join(path, 'log.txt')):
        return [path]
    for entry in sorted(os.listdir(path)):
        sub = os.path.join(path, entry)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, 'log.txt')):
            runs.append(sub)
    return runs


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = sys.argv[1:]
    all_runs = []
    for p in paths:
        all_runs.extend(find_runs(p))

    if not all_runs:
        print(f"No experiments found in: {paths}")
        sys.exit(1)

    results = [read_run(d) for d in all_runs]

    if len(results) == 1:
        print_single(results[0])
    else:
        for r in results:
            print_single(r)
        print_comparison(results)


if __name__ == '__main__':
    main()
