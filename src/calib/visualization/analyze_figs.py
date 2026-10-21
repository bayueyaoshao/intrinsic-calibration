"""误差归因阶段的出图:ANA-01 误差预算证据图 + ANA-02 残差结构两张空间图。

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


# r 等值线固定档位(px):findings 结论 5 判据表的关键半径——0.4 中区
# (现状够用)、0.5 外环起点、0.685 当前数据最远观测、0.75 接近四角的
# 预警圈。等值线是"如果畸变参数晃一动,画面哪里跟着动"的等高线
_R_CONTOUR_VALUES = (0.4, 0.5, 0.685, 0.75)


def _draw_extrapolation_shading(ax, section: dict) -> None:
    """给带 r 横轴的图叠外推区阴影 [r_max_obs, r_corner_frame]。

    外推区(未观测区)是畸变纯靠外推猜的 r 区间,阴影 + 中文标注让它
    在图上一眼可见;全域覆盖(区间为 None)时不画任何阴影。
    """
    interval = ((section.get("train") or {}).get("coverage") or {}).get(
        "extrapolation_interval")
    if not interval:
        return
    r_lo, r_hi = float(interval[0]), float(interval[1])
    ax.axvspan(r_lo, r_hi, color="#d62728", alpha=0.12, zorder=0)
    ax.annotate(
        "外推区(未观测,k2/k3 纯外推)",
        (0.5, 0.92), xycoords="axes fraction",
        ha="center", va="top", fontsize=8, color="#d62728")


def plot_radial_bands(residual_section: dict | None,
                      out_path: Path | str) -> Path:
    """把残差结构段画成 r 分带两张曲线图(PNG),返回图路径。

    输入是 analyze.json 的 residual_structure 段(train/test 各一份逐带
    统计,只消费段内现成键)。左子图 = 每带残差散布 RMS(px),train 与
    test 两条曲线;右子图 = 每带系统偏移 mean_du / mean_dv(整带残差
    平均往哪边挪——与散布分开看才分得出"整体偏"还是"乱散")。横轴是
    带中心 r(带边界的中点);空带(没有观测落进)断线不冒充 0。
    外推区 [r_max_obs, r_corner_frame] 以红色阴影带叠加并标注。

    段缺席(None)或没有任何带统计时不冒充数据:画"无数据"文字标注。

    图是可视化,数字结论以 analyze.json 的 residual_structure 段为准。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_bands = ((residual_section or {}).get("train") or {}).get("bands") or []
    test_bands = ((residual_section or {}).get("test") or {}).get("bands") or []
    fig, (ax_rms, ax_shift) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    if not train_bands and not test_bands:
        for ax in (ax_rms, ax_shift):
            ax.text(0.5, 0.5, "无数据", ha="center", va="center",
                    fontsize=14, color="#777777", transform=ax.transAxes)
            ax.set_axis_off()
        fig.suptitle("残差方向分解:r 分带散布与系统偏移", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def _centers(rows: list[dict]) -> np.ndarray:
        return np.asarray(
            [(float(r["band_lo"]) + float(r["band_hi"])) / 2.0 for r in rows],
            dtype=np.float64)

    def _series(rows: list[dict], key: str) -> np.ndarray:
        # 空带的 None 转 NaN:曲线在缺数据处断线,不把 None 画成 0
        return np.asarray(
            [np.nan if r.get(key) is None else float(r[key]) for r in rows],
            dtype=np.float64)

    # ---- 左子图:每带散布 RMS(px),train/test 两条 ----
    for rows, label, color in ((train_bands, "train(标定用照片)", "#1f77b4"),
                               (test_bands, "test(没参与标定的照片)", "#ff7f0e")):
        if rows:
            ax_rms.plot(_centers(rows), _series(rows, "rms_px"), marker="o",
                        color=color, label=f"{label} 每带 RMS")
    ax_rms.set_xlabel("归一化半径 r(带中心,r 为角点离画面中心的距离,以焦距为单位)")
    ax_rms.set_ylabel("每带残差散布 RMS(px)")
    ax_rms.set_title("散布:每带残差乱散程度(减去本带均值后)", fontsize=9)
    ax_rms.grid(True, alpha=0.3)
    ax_rms.legend(loc="best", fontsize=8)

    # ---- 右子图:每带系统偏移 mean_du / mean_dv ----
    for rows, side, color in ((train_bands, "train", "#1f77b4"),
                              (test_bands, "test", "#ff7f0e")):
        if rows:
            ax_shift.plot(_centers(rows), _series(rows, "mean_du"), marker="s",
                          linestyle="-", color=color, label=f"{side} mean_du(水平偏)")
            ax_shift.plot(_centers(rows), _series(rows, "mean_dv"), marker="^",
                          linestyle="--", color=color, label=f"{side} mean_dv(垂直偏)")
    ax_shift.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.6)
    ax_shift.set_xlabel("归一化半径 r(带中心)")
    ax_shift.set_ylabel("每带系统偏移(px)")
    ax_shift.set_title("系统偏移:每带残差平均往哪边挪(与散布分开看)", fontsize=9)
    ax_shift.grid(True, alpha=0.3)
    ax_shift.legend(loc="best", fontsize=8)

    for ax in (ax_rms, ax_shift):
        _draw_extrapolation_shading(ax, residual_section or {})

    fig.suptitle("残差方向分解:r 分带散布与系统偏移", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_coverage_footprint(residual_section: dict | None,
                            uv_all: np.ndarray | None,
                            out_path: Path | str) -> Path:
    """把观测覆盖足迹画成像素坐标系散点图(PNG),返回图路径。

    输入:residual_structure 段(r 归一化基准 fx/fy/cx/cy、画面尺寸、
    最远观测 r 与外推区间都在段内现成键里)+ 全部观测角点像素坐标
    uv_all(M,2,train 与 test 拼一起,由调用方拼接)。图上叠:
    - 观测足迹散点(拍到过哪里);
    - r = 0.4 / 0.5 / 0.685 / 0.75 的等值线(虚线椭圆——r 相同的画面
      位置连成一圈,是"参数晃一动哪里跟着动"的等高线);
    - 外推区阴影(r 从最远观测到画面最远角的环带,未拍到、纯外推);
    - 画面四角标记(畸变最难约束的位置)。
    标题注明 r 归一化基准的产物轮名(跨轮比较先对齐口径)。

    uv_all 为空或段缺席时画"无数据",不冒充、不抛异常。

    图是可视化,数字结论以 analyze.json 的 residual_structure 段为准。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    section = residual_section or {}
    pts = np.asarray(uv_all, dtype=np.float64) if uv_all is not None \
        else np.zeros((0, 2), dtype=np.float64)
    basis = str(section.get("r_normalization_basis") or "未知基准")
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0 \
            or "r_normalization_params" not in section:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center",
                fontsize=14, color="#777777", transform=ax.transAxes)
        ax.set_axis_off()
        fig.suptitle("覆盖足迹:拍到过哪里、哪里靠外推", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    norm = section["r_normalization_params"]
    fx, fy = float(norm["fx"]), float(norm["fy"])
    cx, cy = float(norm["cx"]), float(norm["cy"])
    width, height = (int(x) for x in section.get("image_size") or (0, 0))
    coverage = (section.get("train") or {}).get("coverage") or {}

    # ---- 外推区阴影:最远观测 r 到画面最远角 r 之间的环带(先画,垫底)----
    r_max_obs = coverage.get("r_max_obs")
    r_corner = coverage.get("r_corner_frame")
    if r_max_obs is not None and r_corner is not None and r_corner > r_max_obs:
        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        outer_u = cx + fx * float(r_corner) * np.cos(theta)
        outer_v = cy + fy * float(r_corner) * np.sin(theta)
        inner_u = cx + fx * float(r_max_obs) * np.cos(theta[::-1])
        inner_v = cy + fy * float(r_max_obs) * np.sin(theta[::-1])
        ax.fill(np.concatenate([outer_u, inner_u]),
                np.concatenate([outer_v, inner_v]),
                color="#d62728", alpha=0.12, zorder=1,
                label="外推区(未观测,k2/k3 纯外推)")

    # ---- 观测足迹散点 ----
    ax.scatter(pts[:, 0], pts[:, 1], s=4, color="#1f77b4", alpha=0.25,
               zorder=2, label="观测到的角点(足迹)")

    # ---- r 等值线(虚线椭圆)----
    theta = np.linspace(0.0, 2.0 * np.pi, 181)
    for i, r_val in enumerate(_R_CONTOUR_VALUES):
        ax.plot(cx + fx * r_val * np.cos(theta), cy + fy * r_val * np.sin(theta),
                linestyle="--", linewidth=0.9, color="#7f7f7f", zorder=3,
                label=f"r = {r_val:g} 等值线" if i == 0 else None)
        # 每条等值线在正右方向标注数值
        ax.annotate(f"r={r_val:g}", (cx + fx * r_val, cy),
                    textcoords="offset points", xytext=(2, 2),
                    fontsize=6.5, color="#555555", rotation=0)
    ax.plot([], [], linestyle="--", color="#7f7f7f",
            label="r 等值线(0.4 / 0.5 / 0.685 / 0.75)")

    # ---- 画面四角与中心标记 ----
    if width > 0 and height > 0:
        corners = np.asarray([(0.0, 0.0), (width - 1.0, 0.0),
                              (0.0, height - 1.0), (width - 1.0, height - 1.0)])
        ax.scatter(corners[:, 0], corners[:, 1], marker="x", s=70,
                   color="#d62728", zorder=4, label="画面四角(最外环位置)")
        ax.plot(cx, cy, marker="+", markersize=12, color="#2ca02c",
                linestyle="none", zorder=4, label="画面中心(主点)")
        x_lo = min(0.0, float(np.nanmin(pts[:, 0]))) - 0.05 * width
        x_hi = max(float(width - 1), float(np.nanmax(pts[:, 0]))) + 0.05 * width
        y_lo = min(0.0, float(np.nanmin(pts[:, 1]))) - 0.05 * height
        y_hi = max(float(height - 1), float(np.nanmax(pts[:, 1]))) + 0.05 * height
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_hi, y_lo)  # 图像坐标 v 朝下,倒轴照片方向才对
        ax.set_aspect("equal", adjustable="box")  # 像素等比,不动数据范围

    ax.set_xlabel("u(px)")
    ax.set_ylabel("v(px)")
    ax.set_title(f"覆盖足迹:拍到过哪里、哪里靠外推\n"
                 f"r 归一化基准:{basis}", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("覆盖足迹:拍到过哪里、哪里靠外推", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
