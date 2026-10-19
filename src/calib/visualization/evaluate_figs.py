"""评估阶段的出图:判行表证据的三张图(CV 误差棒 / LOPO 逐姿态散点 / 离散对比)。

matplotlib Agg 后端直接写 PNG(离线批处理:不弹窗、不依赖显示服务),
模块头三件套照 noise_figs.py(Agg 先于 pyplot、中文字体、unicode_minus)。
三张图都是同一份数据的可视化:数字结论一律以 evaluate.json 为准,图只是
帮人一眼看出差距;库函数只管画图,图失败不拦主产物(CLI 侧逐图独立
try/except,共享模式 7 / T-04-05-03)。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 离线后端先于 pyplot 就位,导入顺序是刻意的
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation

# 中文字体优先级(本机实测在列):找不到时 matplotlib 回退默认字体,
# 中文会显示为方框,但不影响出图与数据本身
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK HK",
    "AR PL UMing CN",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 离群判据线的默认阈值(留一姿态的稳健 z 分数线):与 EvaluateConfig 的
# mad_z_threshold 默认同源;lopo() 报告的 meta 里带实际用值,优先取那份
_DEFAULT_MAD_Z_THRESHOLD = 3.5


def plot_cv_errorbars(cv_summary: dict, out_path: Path | str) -> Path:
    """把交叉验证结果画成误差棒图(PNG),返回图路径。

    输入是 cv_pose_grouped 返回值的 models 段:{模型: {"fold_heldout":
    折级 held-out 误差列表, "mean_heldout_px": 均值}}。每模型一个点
    (held-out 均值)+ 95% 置信区间误差棒(折级样本的均值波动,1.96 ×
    标准误);求解失败的折(None)不进统计。轴标签通俗中文,标题注明
    每模型折级样本数——样本数是"这根误差棒有多可信"的前提。

    图是可视化,数字结论以 evaluate.json 的 cv 段为准。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_ids = list(cv_summary)
    xs = np.arange(len(model_ids), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    n_samples = 0
    for x, model_id in zip(xs, model_ids):
        valid = np.asarray(
            [v for v in cv_summary[model_id]["fold_heldout"] if v is not None],
            dtype=np.float64,
        )
        if valid.size == 0:
            continue  # 全折失败:无点可画(失败率看 json 的 n_failed_folds)
        n_samples = max(n_samples, int(valid.size))
        mean = float(np.mean(valid))
        # 95% 置信区间 = 均值 ± 1.96 × 标准误(标准误 = 样本标准差 / √样本数)
        ci95 = 1.96 * float(np.std(valid, ddof=1)) / np.sqrt(valid.size) if valid.size >= 2 else 0.0
        ax.errorbar(
            [x],
            [mean],
            yerr=[[ci95], [ci95]],
            fmt="o",
            capsize=4,
            markersize=7,
            linewidth=1.4,
            label=model_id,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(model_ids)
    ax.set_xlabel("模型")
    ax.set_ylabel("验证集重投影 RMS(像素)")
    ax.set_title(f"交叉验证 held-out 误差对比(每模型 {n_samples} 个折级样本,误差棒 = 95% 置信区间)")
    if model_ids:
        ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_lopo_pose_errors(lopo_report: dict, out_path: Path | str) -> Path:
    """把留一姿态结果画成逐姿态误差散点(PNG),返回图路径。

    输入是 lopo() 的完整返回值({"schema", "models", "meta"};只传 models
    段也认,离群判据线退回默认阈值)。每模型一个子图:x = 姿态序号、
    y = 该姿态的 held-out 误差;被判"问题姿态"的点用红色三角并标注姿态
    编号;虚线 = 离群判据线(中位数 + 阈值 × MAD,稳健 z 分数线在误差
    轴上的位置;阈值优先取报告 meta 的实际用值)。MAD = 0(全体相同)
    时画不出判据线,省略并在子图标题注明。

    图是可视化,点名结论以 evaluate.json 的 lopo 段(outlier_poses)为准。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "models" in lopo_report:
        models = lopo_report["models"]
        threshold = float(
            (lopo_report.get("meta") or {}).get("mad_z_threshold", _DEFAULT_MAD_Z_THRESHOLD)
        )
    else:
        models = lopo_report
        threshold = _DEFAULT_MAD_Z_THRESHOLD
    model_ids = list(models)
    if not model_ids:
        raise ValueError("留一姿态报告里没有任何模型,画不出逐姿态散点图。")

    fig, axes = plt.subplots(
        1,
        len(model_ids),
        figsize=(4.2 * len(model_ids), 4.2),
        squeeze=False,
        sharey=True,
    )

    for ax, model_id in zip(axes[0], model_ids):
        summary = models[model_id]
        per_pose = summary.get("per_pose") or {}
        outlier_ids = {item["pose_id"] for item in (summary.get("outlier_poses") or [])}
        poses = list(per_pose)
        xs: list[float] = []
        ys: list[float] = []
        ox: list[float] = []
        oy: list[float] = []
        for i, pose in enumerate(poses):
            value = per_pose[pose]
            if value is None:
                continue  # 求解失败的姿态无数值可画(占比看 json 的 n_failed_poses)
            if pose in outlier_ids:
                ox.append(float(i))
                oy.append(float(value))
            else:
                xs.append(float(i))
                ys.append(float(value))
        ax.scatter(xs, ys, s=22, color="#1f77b4", alpha=0.8, label="正常姿态")
        if ox:
            ax.scatter(ox, oy, s=52, color="#d62728", marker="^", label="问题姿态")
            for x, y, pose in zip(ox, oy, [p for p in poses if p in outlier_ids]):
                ax.annotate(
                    str(pose),
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color="#d62728",
                )

        values = np.asarray(ys, dtype=np.float64)
        mad = float(median_abs_deviation(values, scale="normal")) if values.size else 0.0
        if mad > 0.0:
            line = float(np.median(values)) + threshold * mad
            ax.axhline(
                line,
                linestyle="--",
                color="#d62728",
                linewidth=1.0,
                label=f"离群判据线(z>{threshold:g})",
            )
            note = ""
        else:
            note = "(全体相同,无判据线)"

        ax.set_title(f"{model_id}({len(outlier_ids)} 个问题姿态){note}", fontsize=10)
        ax.set_xlabel("姿态序号")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].set_ylabel("留一姿态 held-out RMS(像素)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle("留一姿态逐姿态误差(红 = 被点名的问题姿态)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_dispersion_compare(
    bootstrap_summary: dict, slices_summary: dict, out_path: Path | str
) -> Path:
    """把两种不确定度来源的参数离散画成并排条形图(PNG),返回图路径。

    输入:bootstrap_pose 返回值的 models 段(键 max_disp_px,bootstrap =
    换一批姿态 + 噪声,总晃动)与 slices_by_index 返回值的 models 段
    (键 slice_max_disp_px,重复张分组 = 姿态固定只换噪声,纯噪声晃动),
    单位都是像素位移当量。两根条并排,差值就是"换姿态本身"的贡献
    (D-07 来源分离,一眼可读);某模型数值缺失(None)就不画那根条,
    不冒充 0。

    图是可视化,数字结论以 evaluate.json 的 bootstrap / slices 段为准。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_ids = [m for m in bootstrap_summary if m in slices_summary]
    if not model_ids:
        raise ValueError(
            "bootstrap 与分组的模型名单对不上,画不出离散对比图"
            f"(bootstrap:{list(bootstrap_summary)},分组:{list(slices_summary)})。"
        )
    xs = np.arange(len(model_ids), dtype=np.float64)
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    boot_vals = [bootstrap_summary[m].get("max_disp_px") for m in model_ids]
    slice_vals = [slices_summary[m].get("slice_max_disp_px") for m in model_ids]
    boot_bars = ax.bar(
        xs - width / 2,
        [v if v is not None else np.nan for v in boot_vals],
        width,
        color="#1f77b4",
        label="bootstrap(换姿态 + 噪声)",
    )
    slice_bars = ax.bar(
        xs + width / 2,
        [v if v is not None else np.nan for v in slice_vals],
        width,
        color="#ff7f0e",
        label="重复张分组(只换噪声)",
    )
    for bars, vals in ((boot_bars, boot_vals), (slice_bars, slice_vals)):
        for rect, value in zip(bars, vals):
            if value is None:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    0.0,
                    "无数据",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#777777",
                )
            else:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(xs)
    ax.set_xticklabels(model_ids)
    ax.set_xlabel("模型")
    ax.set_ylabel("参数位移像素当量(px)")
    ax.set_title("两种不确定度来源的参数离散对比:差值 = 换姿态本身的贡献(D-07)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
