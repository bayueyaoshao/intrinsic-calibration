"""误差归因阶段的出图:ANA-01 误差预算证据图(预算条形图)。

matplotlib Agg 后端直接写 PNG(离线批处理:不弹窗、不依赖显示服务),
模块头三件套照 evaluate_figs.py(Agg 先于 pyplot、中文字体、unicode_minus)。
图都是同一份数据的可视化:数字结论一律以 analyze.json 为准,图只是帮人
一眼看出量级;库函数只管画图,图失败不拦主产物(CLI 侧逐图独立
try/except,体例照 evaluate 三张判行表图)。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 离线后端先于 pyplot 就位,导入顺序是刻意的
import matplotlib.pyplot as plt
import numpy as np

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

# 右子图的两条判读线(px = 像素):
# 0.1 px 是"晃动开始可见"的量级参照;低于 0.05 px 的参数在画面上的
# 影响可忽略(findings 结论 1 的判读口径:p1/p2 百分比晃得凶但画面影响
# 不到 0.02 px,必须换算像素当量再上天平)
_PX_VISIBLE_LINE = 0.1
_PX_NEGLIGIBLE = 0.05


def plot_budget_bars(budget_section: dict | None, out_path: Path | str) -> Path:
    """把误差预算段画成左右两张条形图(PNG),返回图路径。

    输入是 analyze.json 的 budget 段(rows 逐参数带分组经验标准差、像素
    当量与 CRLB 参数空间下界;只消费段内现成键,不从数组倒推)。左子图 =
    参数空间:每组两根条——8 组独立标定的经验 σ 与 CRLB 理论下界,对数
    纵轴;k1–k3 与 p1/p2 只画 σ 条不画下界条——它们的单独下界被参数间
    补偿空间撑大(不是噪声小),画出来会诱导无意义的对照,口径见预算表
    的 crlb_caliber_note。右子图 = 画面影响:经验 σ 换算成"画面上挪多少
    像素"的当量(不同量纲参数唯一能放上天平的公共货币),0.1 px 参考
    线,近零参数就地标注"画面影响可忽略"。

    budget 段缺席(None)或没有预算行时不冒充数据:画"无数据"文字标注
    (evaluate_figs 同一体例)。

    图是可视化,数字结论以 analyze.json 的 budget 段为准。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = ((budget_section or {}).get("rows")) or []
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    if not rows:
        for ax in (ax_left, ax_right):
            ax.text(0.5, 0.5, "无数据", ha="center", va="center",
                    fontsize=14, color="#777777", transform=ax.transAxes)
            ax.set_axis_off()
        fig.suptitle("误差预算:参数晃动 vs 理论下界(检测器级口径)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    names = [str(row["param"]) for row in rows]
    xs = np.arange(len(rows), dtype=np.float64)
    width = 0.38

    # ---- 左子图:参数空间的 σ 与理论下界(对数纵轴)----
    empirical = [float(row["empirical_std"]) for row in rows]
    ax_left.bar(xs - width / 2, empirical, width, color="#1f77b4",
                label="分组经验 σ(8 组独立标定)")
    bound_xs: list[float] = []
    bound_vals: list[float] = []
    for x, row in zip(xs, rows):
        # 只给 fx/fy/cx/cy 这类"数据定得住"的参数画下界条(预算表里只有
        # 它们的利用率比值有值);k1–k3/p1/p2 的下界对照无意义,不画
        if row.get("utilization_ratio") is not None:
            bound_xs.append(float(x + width / 2))
            bound_vals.append(float(row["crlb_se"]))
    if bound_xs:
        ax_left.bar(bound_xs, bound_vals, width, color="#ff7f0e",
                    label="CRLB 理论下界(参数空间,仅参考)")
    ax_left.set_yscale("log")
    ax_left.set_xticks(xs)
    ax_left.set_xticklabels(names)
    ax_left.set_xlabel("内参")
    ax_left.set_ylabel("参数标准差(对数轴)")
    ax_left.set_title(
        "参数空间:经验 σ vs 理论下界\n(k1–k3/p1/p2 只画 σ:单独下界被"
        "参数间补偿空间撑大,对照无意义)", fontsize=9)
    ax_left.legend(loc="best", fontsize=8)
    ax_left.grid(True, axis="y", alpha=0.3, which="both")

    # ---- 右子图:经验 σ 的像素当量(画面影响)----
    px_equiv = [float(row["empirical_std_px"]) for row in rows]
    bars = ax_right.bar(xs, px_equiv, 0.6, color="#2ca02c")
    ax_right.axhline(
        _PX_VISIBLE_LINE, linestyle="--", color="#d62728", linewidth=1.0,
        label=f"{_PX_VISIBLE_LINE:g} px 参考线(晃动开始可见)")
    for rect, value in zip(bars, px_equiv):
        if value < _PX_NEGLIGIBLE:
            ax_right.annotate(
                "画面影响可忽略",
                (rect.get_x() + rect.get_width() / 2, value),
                textcoords="offset points", xytext=(2, 3),
                ha="center", va="bottom", fontsize=7, rotation=90,
                color="#777777")
        else:
            ax_right.annotate(
                f"{value:.3f}",
                (rect.get_x() + rect.get_width() / 2, value),
                textcoords="offset points", xytext=(0, 3),
                ha="center", va="bottom", fontsize=7)
    ax_right.set_xticks(xs)
    ax_right.set_xticklabels(names)
    ax_right.set_xlabel("内参")
    ax_right.set_ylabel("经验 σ 的像素当量(px)")
    ax_right.set_title("画面影响:σ 换算成像素当量才好比较\n"
                       "(近零参数看百分比没有意义)", fontsize=9)
    ax_right.legend(loc="best", fontsize=8)
    ax_right.grid(True, axis="y", alpha=0.3)

    fig.suptitle("误差预算:参数晃动 vs 理论下界(检测器级口径)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
