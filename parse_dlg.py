#!/usr/bin/env python3
"""Extract DLG metrics from pickle files using restricted unpickler to skip large arrays."""
import pickle
import struct
import os
import sys
import io

BASE = '/data1/data2/liym/Projects/FedBarre/BARRE/runs/fedbarre10_1000_v2/M_5/emnist'

METHODS = {
    'DP-Lap': 'dplap_noShuf_C12.0_batch_eps0.7_1776789740',
    'BARRE':  'barre_noShuf_C12.0_eps0.7_lba10_zeta1e-05_1776791062',
    'FedAvg': 'fedavg_noShuf_C12.0_eps0.7_lba10_zeta1e-05_1776801441',
    'DP-GAS': 'dpgas_noShuf_C12.0_batch_eps0.7_1776802563',
}

EPOCHS = [8, 9, 10]


def extract_floats_from_pickle(filepath):
    """Extract the first 4 float values from a pickle dict.

    The pickle format for this file stores:
      test_mse (float), feat_mse (float), test_psnr (float), test_ssim (float),
      then rec_img and gt (numpy arrays).

    We'll use a restricted unpickler to avoid loading numpy arrays.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return {
        'test_mse': float(data['test_mse']),
        'feat_mse': float(data['feat_mse']),
        'test_psnr': float(data['test_psnr']),
        'test_ssim': float(data['test_ssim']),
    }


def extract_floats_manual(filepath):
    """Manually parse pickle protocol to extract float values without loading numpy arrays."""
    results = {}
    with open(filepath, 'rb') as f:
        content = f.read(2000)  # Read just the beginning -- floats come first

    # Look for the pattern: key string followed by float
    # In pickle protocol 2, floats are stored as BINFLOAT (G) + 8 bytes big-endian double
    keys = ['test_mse', 'feat_mse', 'test_psnr', 'test_ssim']

    for key in keys:
        # Find the key in the binary content
        key_bytes = key.encode('utf-8')
        # Pickle short string: opcode 0x8c + length byte + string
        needle = bytes([0x8c, len(key_bytes)]) + key_bytes
        idx = content.find(needle)
        if idx < 0:
            # Try long string format
            needle = bytes([0x8a]) + struct.pack('<I', len(key_bytes)) + key_bytes
            idx = content.find(needle)

        if idx >= 0:
            # After the key, look for BINFLOAT opcode (G = 0x47)
            search_start = idx + len(needle)
            for j in range(search_start, min(search_start + 20, len(content))):
                if content[j] == 0x47:  # BINFLOAT
                    val = struct.unpack('>d', content[j+1:j+9])[0]
                    results[key] = val
                    break

    return results


if __name__ == '__main__':
    print(f"{'Method':<12} {'Epoch':>5} {'MSE':>12} {'Feat_MSE':>12} {'PSNR':>10} {'SSIM':>10}", flush=True)
    print('-' * 65, flush=True)

    all_results = {}

    for method_name, method_dir in METHODS.items():
        all_results[method_name] = {'mse': [], 'feat_mse': [], 'psnr': [], 'ssim': []}
        for ep in EPOCHS:
            pkl_path = os.path.join(BASE, method_dir, f'dlg_result_E{ep}.pkl')
            if not os.path.exists(pkl_path):
                print(f'{method_name:<12} {ep:>5}  --- FILE NOT FOUND ---', flush=True)
                continue

            try:
                vals = extract_floats_manual(pkl_path)
            except Exception as e:
                print(f'{method_name:<12} {ep:>5}  ERROR: {e}', flush=True)
                continue

            mse = vals.get('test_mse', float('nan'))
            feat_mse = vals.get('feat_mse', float('nan'))
            psnr = vals.get('test_psnr', float('nan'))
            ssim = vals.get('test_ssim', float('nan'))

            all_results[method_name]['mse'].append(mse)
            all_results[method_name]['feat_mse'].append(feat_mse)
            all_results[method_name]['psnr'].append(psnr)
            all_results[method_name]['ssim'].append(ssim)

            print(f'{method_name:<12} {ep:>5} {mse:>12.6f} {feat_mse:>12.6f} {psnr:>10.4f} {ssim:>10.6f}', flush=True)
        print(flush=True)

    print(flush=True)
    print('=== Averages across Epochs 8-10 ===', flush=True)
    print(f"{'Method':<12} {'Avg MSE':>12} {'Avg Feat_MSE':>14} {'Avg PSNR':>10} {'Avg SSIM':>10}", flush=True)
    print('-' * 62, flush=True)
    for method_name in METHODS:
        r = all_results[method_name]
        if r['mse']:
            n = len(r['mse'])
            avg_mse = sum(r['mse']) / n
            avg_feat = sum(r['feat_mse']) / n
            avg_psnr = sum(r['psnr']) / n
            avg_ssim = sum(r['ssim']) / n
            print(f"{method_name:<12} {avg_mse:>12.6f} {avg_feat:>14.6f} {avg_psnr:>10.4f} {avg_ssim:>10.6f}", flush=True)
        else:
            print(f"{method_name:<12}  --- NO DATA ---", flush=True)
