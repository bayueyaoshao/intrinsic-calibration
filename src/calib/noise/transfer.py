"""传递曲线:把注入实验的逐角点位移按 σ 档汇总成"像素噪声 → 角点定位误差"
的经验曲线,并给出 σ_px → 每轴角点 σ 的插值换算(D-03 检测器口径的折算依据)。

两条硬规则(研究 Pitfall 2 / Pitfall 4):
- σ=0 档是校准行(role="reference"):位移应恰为 0,只用来核验基线对齐,
  绝不参与曲线拟合;
- 换算输出字段名一律 sigma_corner_per_axis(每轴角点 σ):欧氏误差在两轴
  各向同性假设下 = √2 × 单轴误差,所以每轴 σ = 曲线值 ÷ √2。字段名上把
  "像素噪声 σ"与"角点误差 σ"两个口径分开,防止混线喂给理论精度计算。

插值用 scipy 的 PCHIP(保形插值,白话:插值曲线只从数据点中间平稳地过,
不会在数据点之间鼓出数据里不存在的假包),拟合在对数 σ 轴上。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

# 每档结果必须齐备的五个统计键(下游与测试按这套键取数)
LEVEL_KEYS = ("rms_euclidean", "scatter_low", "scatter_high",
              "failure_rate", "fallback_rate")


def build_transfer_curve(displacements: np.ndarray, grid: list[float],
                         detected: np.ndarray, source: np.ndarray) -> dict:
    """把注入位移数据汇总成传递曲线。

    参数(全部来自 injection_displacements.npz):
        displacements: (N, L, R, P, 2) float64,未检出副本为 NaN;
        grid:          长度 L 的 σ 档列表,第 0 档必须是 σ=0(校准行);
        detected:      (N, L, R) bool,该副本是否检出;
        source:        (N, L, R) 文字,每副本来源("sb"/"classic_fallback"/"")。

    返回:
        {"grid": [...], "levels": [每档一条记录], "meta": {...}}。
        每档记录含五键:rms_euclidean(方均根欧氏角点误差,对图像 × 重复 ×
        角点全平均)、scatter_low/high(逐图 rms 的 25/75 分位散带)、
        failure_rate(该档未检出副本占比)、fallback_rate(兜底占比);
        σ=0 档另标 role="reference"。
    """
    disp = np.asarray(displacements, dtype=np.float64)
    detected = np.asarray(detected, dtype=bool)
    source = np.asarray(source)
    n, _L, _R, n_points, _two = disp.shape

    levels = []
    for li, sigma in enumerate(grid):
        det = detected[:, li]                     # (N, R) 该档的检出掩码
        d = disp[:, li][det]                      # (n_det, P, 2) 只留检出副本
        rms = None
        scatter_low = scatter_high = None
        if d.shape[0] > 0 and np.isfinite(d).any():
            rms = float(np.sqrt((d ** 2).sum(axis=-1).mean()))
            per_image = []
            for i in range(n):
                di = disp[i, li][detected[i, li]]
                if di.shape[0] > 0 and np.isfinite(di).any():
                    per_image.append(float(np.sqrt((di ** 2).sum(axis=-1).mean())))
            if per_image:
                scatter_low = float(np.percentile(per_image, 25))
                scatter_high = float(np.percentile(per_image, 75))
        entry = {
            "sigma": float(sigma),
            "rms_euclidean": rms,
            "scatter_low": scatter_low,
            "scatter_high": scatter_high,
            "failure_rate": float(1.0 - det.mean()),
            "fallback_rate": float((source[:, li] == "classic_fallback").mean()),
            "n_total": int(det.size),
            "n_detected": int(det.sum()),
            "n_points": int(n_points),
        }
        if float(sigma) == 0.0:
            entry["role"] = "reference"           # 校准行:位移应恒 0,不参与拟合
        levels.append(entry)

    return {
        "grid": [float(s) for s in grid],
        "levels": levels,
        "meta": {
            "n_images": int(n),
            "axis_note": ("rms_euclidean 是方均根欧氏角点误差(像素);"
                          "喂理论精度计算的每轴 σ 用 interp_corner_sigma 换算,"
                          "输出字段名一律 sigma_corner_per_axis。"),
        },
    }


def fit_interp(curve: dict) -> PchipInterpolator:
    """在除 σ=0 校准行外的对数 σ 轴上拟合 PCHIP 保形插值,返回插值函数。

    保形(白话):插值曲线不会在数据点之间鼓出数据里不存在的假包——
    两档之间的取值总落在相邻档曲线值之间。这是传递曲线该有的物理形态
    (噪声越大误差越大,单调抬头),普通样条做不到这条保证。
    有效档不足两档时中文报错(没有曲线可言,不能静默硬插)。
    """
    xs: list[float] = []
    ys: list[float] = []
    for entry in curve["levels"]:
        if entry["sigma"] <= 0.0 or entry.get("role") == "reference":
            continue
        if entry.get("rms_euclidean") is None:
            continue                              # 全失败档:没有纵坐标,跳过
        xs.append(float(np.log(entry["sigma"])))
        ys.append(float(entry["rms_euclidean"]))
    if len(xs) < 2:
        raise ValueError(
            f"传递曲线至少要两档有效 σ 才能拟合插值;实际只有 {len(xs)} 档。"
            f"请检查注入实验是否失败档太多(曲线在高 σ 段全失败是网格上限问题)。")
    return PchipInterpolator(np.asarray(xs), np.asarray(ys))


def interp_corner_sigma(sigma_px: float, interp: PchipInterpolator) -> float:
    """把像素噪声 σ 换算成每轴角点 σ(sigma_corner_per_axis)。

    √2 换算(写死,别改):欧氏距离在 x、y 两轴各向同性的假设下,
    E[dx² + dy²] = 2σ²_轴,所以每轴 σ = 曲线值(欧氏 RMS)÷ √2。
    输出字段名一律叫 sigma_corner_per_axis(Pitfall 2:防两口径混线)。
    """
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError(
            f"interp_corner_sigma 的输入必须是 > 0 的像素噪声 σ;实际是 "
            f"{sigma_px!r}(σ=0 或负数没有对数坐标可查,请先判空再调)。")
    return float(interp(np.log(sigma_px))) / np.sqrt(2.0)


def save_transfer_curve(curve: dict, out_dir: Path | str) -> Path:
    """传递曲线落盘 transfer_curve.json(中文 meta 原样可读),返回文件路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "transfer_curve.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(curve, f, ensure_ascii=False, indent=2)
    return path
