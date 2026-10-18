"""存疑帧 vs 正常帧的重投影误差对比图(matplotlib PNG,佐证材料)。

D-04 闭环的出图件:把复核逐帧表(见 solve/scoring.py 的 review_suspects)
里两组照片的逐帧重投影误差画成两张并排小图——直方图叠加看分布形状、
累积分布曲线叠加看分位数差距。**PNG 只作佐证,数字以同目录的
suspect_vs_normal.json 为准**(图是同一份逐帧表画的,但结论判定、
阈值与血缘都记在 JSON 里,T-02-06-03)。

图内标注只用 ASCII("suspect"/"normal",沿用 mosaic.py 的 cv2 时代约定,
保持全项目图内文字风格一致);中文说明全部进 JSON 与控制台。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 离线批处理:不弹窗、不依赖显示服务,直接写 PNG 文件
# use() 之后的 pyplot/numpy 导入必须在 use 之后执行(离线后端先于 pyplot
# 就位),顺序是刻意的,不是笔误
import matplotlib.pyplot as plt
import numpy as np


def _valid_rms(frame_table: list[dict], suspect: bool) -> np.ndarray:
    """从逐帧表取一组的有效误差值(NaN 行——单帧求解失败的占位——不计入分布)。"""
    values = [row["per_view_rms"] for row in frame_table
              if row["suspect"] == suspect and row["per_view_rms"] is not None
              and np.isfinite(row["per_view_rms"])]
    return np.asarray(values, dtype=np.float64)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """经验累积分布曲线:排序值 vs "不超过该值的样本占比"(0 到 1)。"""
    ordered = np.sort(values)
    cumulative = np.arange(1, ordered.size + 1, dtype=np.float64) / ordered.size
    return ordered, cumulative


def plot_suspect_compare(frame_table: list[dict], out_png: Path | str) -> Path:
    """按逐帧表画两组误差分布对比图,落盘 PNG,返回图片路径。

    参数 frame_table 是 review_suspects 逐帧表(每行含 suspect 布尔与
    per_view_rms 数值或 None)。左图 = 两组直方图叠加(按密度归一,两组
    张数不同也能比形状);右图 = 两组累积分布曲线叠加(同一误差水平下
    哪组"攒"得更快一眼可见)。任一组没有有效数值时对应曲线留空并在
    图内标 ASCII 说明,不硬画空图。
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    suspect_vals = _valid_rms(frame_table, suspect=True)
    normal_vals = _valid_rms(frame_table, suspect=False)
    if suspect_vals.size == 0 and normal_vals.size == 0:
        raise ValueError(
            "对比图没有可画的数值:逐帧表里两组都没有有效误差值"
            "(全部是失败占位),请先检查复核逐帧表。"
        )

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    colors = {"suspect": "#d62728", "normal": "#1f77b4"}

    # 左图:直方图叠加(密度归一,张数不同也可比形状)
    all_vals = np.concatenate([v for v in (suspect_vals, normal_vals) if v.size])
    bins = np.histogram_bin_edges(all_vals, bins=40)
    for name, vals in (("suspect", suspect_vals), ("normal", normal_vals)):
        if vals.size:
            ax_hist.hist(vals, bins=bins, density=True, alpha=0.55,
                         color=colors[name], label=f"{name} (n={vals.size})")
    ax_hist.set_xlabel("per-view reprojection RMS (px)")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("RMS distribution: suspect vs normal")
    ax_hist.legend(loc="upper right", fontsize=9)

    # 右图:经验累积分布曲线叠加
    for name, vals in (("suspect", suspect_vals), ("normal", normal_vals)):
        if vals.size:
            x, y = _ecdf(vals)
            ax_cdf.plot(x, y, color=colors[name],
                        label=f"{name} (n={vals.size})")
    ax_cdf.set_xlabel("per-view reprojection RMS (px)")
    ax_cdf.set_ylabel("cumulative fraction")
    ax_cdf.set_title("ECDF: suspect vs normal")
    ax_cdf.legend(loc="lower right", fontsize=9)
    if suspect_vals.size == 0:
        ax_cdf.text(0.5, 0.5, "no valid suspect frames", ha="center",
                    transform=ax_cdf.transAxes, color=colors["suspect"])
    if normal_vals.size == 0:
        ax_cdf.text(0.5, 0.4, "no valid normal frames", ha="center",
                    transform=ax_cdf.transAxes, color=colors["normal"])

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
