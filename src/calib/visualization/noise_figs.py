"""噪声阶段的出图:传递曲线误差棒图(像素噪声 σ → 角点定位误差 RMS)。

matplotlib Agg 后端直接写 PNG(离线批处理:不弹窗、不依赖显示服务)。
本机装有 Noto CJK 字体,图内轴标签按项目要求用通俗中文;数字结论以
同目录的 transfer_curve.json 为准,图只是同一份数据的可视化。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 离线后端先于 pyplot 就位,导入顺序是刻意的
import matplotlib.pyplot as plt
import numpy as np

# 中文字体优先级(本机实测在列):找不到时 matplotlib 回退默认字体,
# 中文会显示为方框,但不影响出图与数据本身
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                   "Noto Sans CJK HK", "AR PL UMing CN",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_transfer_curve(curve: dict, out_path: Path | str) -> Path:
    """把传递曲线画成误差棒图(PNG),返回图路径。

    主轴:每档 σ 的角点定位误差 RMS(圆点)+ 逐图 25/75 分位散带(误差棒);
    次轴(红色虚线):该档检测失败率——高 σ 段失败率抬头本身就是
    "检测器失效点"的证据,与误差抬头互为印证。
    σ=0 校准行不画进对数轴(它只用来核验基线对齐,meta 里有它的值)。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = [e for e in curve["levels"] if e["sigma"] > 0.0]
    sigmas = np.asarray([e["sigma"] for e in entries], dtype=np.float64)
    rms = np.asarray([np.nan if e["rms_euclidean"] is None else e["rms_euclidean"]
                      for e in entries], dtype=np.float64)
    low = np.asarray([np.nan if e["scatter_low"] is None else e["scatter_low"]
                      for e in entries], dtype=np.float64)
    high = np.asarray([np.nan if e["scatter_high"] is None else e["scatter_high"]
                       for e in entries], dtype=np.float64)
    failure = np.asarray([e["failure_rate"] for e in entries], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax2 = ax1.twinx()

    yerr = np.vstack([np.clip(rms - low, 0.0, None), np.clip(high - rms, 0.0, None)])
    ax1.errorbar(sigmas, rms, yerr=yerr, marker="o", capsize=3,
                 color="#1f77b4", linewidth=1.2,
                 label="角点定位误差 RMS(误差棒 = 逐图 25/75 分位)")
    ax2.plot(sigmas, failure, linestyle="--", marker="x", color="#d62728",
             linewidth=1.2, label="检测失败率")

    ax1.set_xscale("log")
    ax1.set_xlabel("注入像素噪声 σ(px)")
    ax1.set_ylabel("角点定位误差 RMS(px)")
    ax2.set_ylabel("检测失败率")
    ax1.set_title("像素噪声 → 角点定位误差 传递曲线(σ=0 校准行未画出)")
    handles = ax1.get_lines() + ax2.get_lines()
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper left", fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
