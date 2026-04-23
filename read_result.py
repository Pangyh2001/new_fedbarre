#!/usr/bin/env python3
# summarize_tb.py

import os
import re
from tensorboard.backend.event_processing import event_accumulator

# 需要读取的 scalar tag 列表
TAGS = {
    "psnr": "C0/dlg_B0_psnr",
    "mse":  "C0/dlg_B0_mse",
    "acc":  "global/test_acc",
    "ssim": "C0/dlg_B0_ssim",
}

# 指定哪些 tags 需要计算平均，哪些需要取最后值
AVERAGE_TAGS = {"psnr", "mse", "ssim"}
LAST_TAGS    = {"acc"}


def summarize_run(run_dir):
    """
    读取单个实验目录下的 event 文件，返回各指标的平均或最后值
    """
    # 只关心 SCALARS，提高加载速度
    ea = event_accumulator.EventAccumulator(
        run_dir,
        size_guidance={
            event_accumulator.SCALARS:   1000,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES:     0,
            event_accumulator.AUDIO:      0,
            event_accumulator.GRAPH:      0,
        }
    )
    ea.Reload()

    results = {}
    for key, tag in TAGS.items():
        try:
            events = ea.Scalars(tag)
        except KeyError:
            events = []

        values = [e.value for e in events]
        if key in AVERAGE_TAGS:
            results[key] = sum(values) / len(values) if values else None
        elif key in LAST_TAGS:
            results[key] = values[-1] if values else None
        else:
            results[key] = None

    return (
        results["psnr"],
        results["mse"],
        results["ssim"],
        results["acc"],
    )


def main():
    root = "runs/ablation/fedrpf/mnist/rp12"   # TODO 修改为实际的 runs 目录路径

    # 匹配子目录名，并捕获 rp 后的浮点数（命名为 eps）和尾部的整数种子
    # pattern = re.compile(
    #     r"^test_noShuf_C12\.0_"        # 固定 C12.0
    #     r"rpeps1_lba10_zeta1e-05_"     # 固定 rpeps1、lba10、zeta1e-05
    #     r"rp(?P<eps>\d+(?:\.\d+)?)_"   # 捕获 rp 后面可带小数的数字，组名为 eps
    #     r"eps0.9_(?P<seed>\d+)$"         # 固定 eps1_，再捕获尾部整数种子
    # )
    # pattern = re.compile(r"^test_noShuf_noisetype2_rpratio0\.3_eps(?P<eps>\d+\.?\d*)_.+")  # TODO 这是fedRPF的命名规则
    pattern = re.compile(r"^test_noShuf_noisetype0_rpratio(?P<eps>\d+\.?\d*)_.+")  # TODO 这是fedRPF的命名规则

    # pattern = re.compile(r"^test_noShuf_C12\.0_eps(?P<eps>\d+\.?\d*)_.+")  # 这是fednfl和fedbarre的命名规则
    
    # pattern = re.compile(r"^test_noShuf_C12\.0_batch_eps(?P<eps>\d+\.?\d*)_.+")  # 这是DP的命名规则


    results = []
    for entry in os.listdir(root):
        # 只匹配符合命名规则的子目录
        m = pattern.match(entry)
        if not m:
            continue

        eps = float(m.group("eps"))  # rp 后的数字
        # seed = int(m.group("seed")) # 如果需要，也可以用这个随机种子

        run_path = os.path.join(root, entry)
        psnr_avg, mse_avg, ssim_avg, acc_last = summarize_run(run_path)
        results.append((eps, psnr_avg, mse_avg, ssim_avg, acc_last))

    # 排序并打印结果
    results.sort(key=lambda x: x[1])
    header = f"{'epsilon':>7s} | {'psnr_avg':>10s} | {'mse_avg':>10s} | {'ssim_avg':>10s} | {'acc_last':>10s}"
    print(header)
    print("-" * len(header))
    for eps, psnr, mse, ssim, acc in results:
        print(f"{eps:7.3f} | {psnr:10.4f} | {mse:10.4f} | {ssim:10.4f} | {acc:10.4f}")


if __name__ == "__main__":
    main()
