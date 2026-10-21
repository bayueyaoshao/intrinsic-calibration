"""ANA-02 残差方向分解 + r 覆盖足迹:误差到底偏在哪个方向、画面哪个区域。

白话背景:标定解出来之后,"误差长什么样"比"误差平均多大"更能指出病根。
本模块把残差(投影位置减实测位置的像素偏差,重投影误差的逐角点版本)
按三个维度切开看:逐视图(哪张照片差)、画面区域(四象限哪块差)、
归一化半径 r 分带(离画面中心多远的地方差)。同时回答另一个要命的问题:
畸变模型在哪些像素位置**没有观测撑着**——那些位置是纯靠外推猜的,
用之前必须知道。

公式定案于 docs/findings.md 结论 5(2026-10-21):
    r = √(((u−cx)/fx)² + ((v−cy)/fy)²)   归一化半径:角点到画面中心的
                                         距离,以焦距为单位(中心 r=0);
    信息权重 ∝ r⁴(k2)、r⁶(k3)          中心观测对高阶畸变近乎失明;
    当前数据实测:最远观测 r=0.685,画面四角 r=0.796,
    r>0.5 的观测占 21.1% 却贡献 k2 信息的 61.4%,r 0.69–0.80 零观测
    = 纯外推区(判据:r≤0.4 晃动≤0.02px 现状够用;0.5–0.685 记入
    预算;r>0.69 必须先补覆盖)。

分带器(assign_bands / band_stats)一处实现,ANA-02 与 COR-03
(连拍各向异性,anisotropy.py)共用同一把尺,严禁第二份实现。

单一数学源红线(D-06/D-10):残差与位姿评分只走既有接口——train 侧
直接读标定落盘的 residuals.npz,test 侧逐帧调 solve_pose_only 只估位姿
再回代,投影只经 model.project;本模块不出现任何投影/导数/位姿求解
公式,也绝不 import OpenCV。
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import numpy as np

from calib.evaluation.experiments import _px_dispersion
from calib.models import create_model
from calib.models.base import transform_to_camera
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet
from calib.solve.result import load_extrinsics, load_residuals, load_result
from calib.solve.scoring import solve_pose_only

# 段模式版本:残差结构段的结构约定(加键不改键算兼容,删键必须升版本)
SCHEMA_RESIDUAL = "analyze-residual-v1"

# 位姿求解下限(solve_pose_only 的失败语义:4 点以下方程组欠定)。
# 单帧可用角点少于此数时按失败帧占行,不中断整批。
_MIN_PNP_POINTS = 4

# r⁴ 折算的参考半径口径说明(findings 结论 5:σ 级晃动 0.135px 当量
# 实测在 r=0.685 处,即当前数据的最远观测半径;换数据后由 run 时从
# 覆盖足迹现算,公式不变)
_WOBBLE_LAW_NOTE = "k2 晃动按 r⁴ 律折算(畸变参数 k2 的作用量正比于 r⁴)"


def _require(mapping, key: str, what: str):
    """从上游报告取键,缺失当场中文报错(体例照 budget.py 的取数防线)。"""
    if not isinstance(mapping, dict) or key not in mapping:
        raise AssertionError(
            f"上游报告缺键 {key}({what})。残差方向分解按契约消费 "
            f"evaluate.json 的落盘段,缺键多半是产物与当前代码不是同一套"
            f"契约,或串了轮次,请先重跑 calib evaluate 再来。"
        )
    return mapping[key]


def radius_footprint(
    uv: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> tuple[np.ndarray, np.ndarray]:
    """归一化半径 r 与 k2 信息权重 r⁴(逐角点,findings 结论 5 的公式)。

    uv 为 (M,2) 角点像素坐标;返回 (r (M,), r4 (M,))。
    r 是"角点到画面中心的距离,以焦距为单位":中心 r=0,数值越大离
    中心越远。k2 的作用量正比于 r⁴,所以 r⁴ 就是这个观测对 k2 的
    信息权重——中心观测权重近乎零,外环观测才是 k2 的信息来源。
    """
    arr = np.asarray(uv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise AssertionError(
            f"radius_footprint 的 uv 形状不合:期望 (M,2) 角点像素坐标,"
            f"实际是 {arr.shape}。请检查调用方的数据排布。"
        )
    r = np.sqrt(
        ((arr[:, 0] - cx) / fx) ** 2 + ((arr[:, 1] - cy) / fy) ** 2
    )
    return r, r**4


def assign_bands(r: np.ndarray, edges: tuple[float, ...]) -> tuple[np.ndarray, int]:
    """按分带边界把逐点 r 归到带号:左闭右开 [e_i, e_{i+1})。

    返回 (band_ids (M,) 整型, n_beyond 超界计数):带号 0..len(edges)-2;
    超出最大边界 edges[-1] 的观测归最外带(最后一带向无穷大开口),
    并单独计数返回——超界观测不丢弃,它们正是最外环的信息来源。

    NaN 的 r 不属于任何带(带号随 numpy 落到最外带,但超界计数只数
    有限值),统计由调用方的有限值掩码兜底。
    """
    e = tuple(float(x) for x in edges)
    if len(e) < 2:
        raise AssertionError(
            f"分带边界至少要 2 个数(1 条带),实际拿到 {len(e)} 个:{e}。"
            f"请检查配置 analyze.radial_bin_edges。"
        )
    if not all(math.isfinite(x) for x in e):
        raise AssertionError(
            f"分带边界必须是有限数,实际拿到 {e}。请检查配置 "
            f"analyze.radial_bin_edges。"
        )
    if any(b <= a for a, b in pairwise(e)):
        raise AssertionError(
            f"分带边界必须严格递增(带与带不重叠不倒序),实际拿到 {e}。"
            f"请检查配置 analyze.radial_bin_edges。"
        )
    r_arr = np.asarray(r, dtype=np.float64).ravel()
    # np.digitize(bins=内部边界, right=False):x < bins[0] → 0;
    # bins[i] <= x < bins[i+1] → i+1(左闭右开);x ≥ 最大内部边界 → 最后一带
    interior = np.asarray(e[1:-1], dtype=np.float64)
    band_ids = np.digitize(r_arr, bins=interior, right=False)
    finite = np.isfinite(r_arr)
    n_beyond = int(np.count_nonzero(finite & (r_arr >= e[-1])))
    return band_ids, n_beyond


def band_stats(
    resid: np.ndarray, r: np.ndarray, edges: tuple[float, ...]
) -> list[dict]:
    """逐带统计:系统偏移(mean_du/mean_dv)与散布(rms)分开呈现。

    resid (M,2) 为逐角点像素残差(du, dv),r (M,) 为对应归一化半径,
    两者逐行对齐。每带一行字典:band_lo/band_hi(带边界)、n(带内有限
    残差数)、mean_du / mean_dv(分量均值 = 系统性偏移)、
    rms_px / median_px / max_px(**散布**统计——先把带内残差减去本带
    均值再取欧氏长度,所以整带同向挪动不改这三个数;"整体偏"记在
    mean_du/mean_dv,"乱散"记在 rms,两套各说各话)。含 NaN 的残差行
    不计入(test 侧失败帧占行后自然被跳过)。最后一带额外给
    n_beyond_hi:超出带上界 edges[-1] 的观测数(归最外带、不丢)。
    """
    res = np.asarray(resid, dtype=np.float64)
    if res.ndim != 2 or res.shape[1] != 2:
        raise AssertionError(
            f"band_stats 的 resid 形状不合:期望 (M,2) 逐角点 (du, dv),"
            f"实际是 {res.shape}。请检查调用方的数据排布。"
        )
    r_arr = np.asarray(r, dtype=np.float64).ravel()
    if r_arr.shape[0] != res.shape[0]:
        raise AssertionError(
            f"band_stats 的 resid 与 r 行数对不上:{res.shape[0]} vs "
            f"{r_arr.shape[0]}。残差与半径必须逐角点对齐。"
        )
    e = tuple(float(x) for x in edges)
    band_ids, _n_beyond = assign_bands(r_arr, e)
    finite = np.isfinite(res).all(axis=1) & np.isfinite(r_arr)
    rows: list[dict] = []
    for i in range(len(e) - 1):
        m = finite & (band_ids == i)
        n = int(np.count_nonzero(m))
        row: dict = {
            "band_lo": e[i],
            "band_hi": e[i + 1],
            "n": n,
            "rms_px": None,
            "median_px": None,
            "max_px": None,
            "mean_du": None,
            "mean_dv": None,
        }
        if n:
            d = res[m]
            mean_du = float(np.mean(d[:, 0]))
            mean_dv = float(np.mean(d[:, 1]))
            # 散布 = 残差减去本带均值向量后的长度(系统偏移不掺进 rms)
            centered = d - np.array([mean_du, mean_dv])
            norms = np.linalg.norm(centered, axis=1)
            row.update(
                mean_du=mean_du,
                mean_dv=mean_dv,
                rms_px=float(np.sqrt(np.mean(norms**2))),
                median_px=float(np.median(norms)),
                max_px=float(np.max(norms)),
            )
        if i == len(e) - 2:
            row["n_beyond_hi"] = int(
                np.count_nonzero(finite & (r_arr >= e[-1]))
            )
        rows.append(row)
    return rows


def coverage_summary(
    uv: np.ndarray, fx: float, fy: float, cx: float, cy: float,
    image_size: tuple[int, int],
) -> dict:
    """覆盖足迹:拍到过哪里、哪里是纯外推(findings 结论 5 的口径)。

    返回字典:
        r_max_obs:              观测到的最远 r(没有有限观测时为 None);
        r_corner_frame:         画面四角像素按同一公式算出的最大 r;
        extrapolation_interval: [r_max_obs, r_corner_frame],r_max_obs ≥
                                r_corner_frame(全域覆盖)时为 None;
        frac_beyond_05:         r > 0.5 的观测占比(外环两成观测撑起六成
                                k2 信息的那把尺);
        info_share_beyond_05:   Σr⁴(r>0.5) / Σr⁴ 全体——r>0.5 观测承载
                                的 k2 信息份额。
    """
    arr = np.asarray(uv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise AssertionError(
            f"coverage_summary 的 uv 形状不合:期望 (M,2),实际是 {arr.shape}。"
        )
    width, height = int(image_size[0]), int(image_size[1])
    corner_pixels = np.asarray(
        [(0.0, 0.0), (width - 1.0, 0.0), (0.0, height - 1.0),
         (width - 1.0, height - 1.0)],
        dtype=np.float64,
    )
    r_corner = float(
        radius_footprint(corner_pixels, fx, fy, cx, cy)[0].max()
    )
    finite = np.isfinite(arr).all(axis=1)
    if not np.any(finite):
        return {
            "r_max_obs": None,
            "r_corner_frame": r_corner,
            "extrapolation_interval": None,
            "frac_beyond_05": None,
            "info_share_beyond_05": None,
        }
    r, r4 = radius_footprint(arr[finite], fx, fy, cx, cy)
    r_max = float(np.max(r))
    interval = [r_max, r_corner] if r_max < r_corner else None
    beyond = r > 0.5
    total = float(np.sum(r4))
    return {
        "r_max_obs": r_max,
        "r_corner_frame": r_corner,
        "extrapolation_interval": interval,
        "frac_beyond_05": float(np.mean(beyond)),
        "info_share_beyond_05": (
            float(np.sum(r4[beyond]) / total) if total > 0 else None
        ),
    }


def k2_wobble_px(px_equiv_at_ref: float, r: float, r_ref: float) -> float:
    """k2 晃动按 r⁴ 律折算:参考半径处的像素当量 → 目标半径处的晃动量。

    findings 结论 5 的判据表就是这么折算的:σ 级晃动 0.135px 当量在
    r=0.685 处实测,折到 r=0.80 的四角方向约 0.25px,折到 r=0.4 的中区
    约 0.016px(现状够用)。公式:晃动(r) = 当量(r_ref) × (r/r_ref)⁴。
    """
    if not (math.isfinite(px_equiv_at_ref) and math.isfinite(r)
            and math.isfinite(r_ref)) or r_ref <= 0 or r < 0:
        raise AssertionError(
            f"k2 晃动折算的输入不合法:当量 {px_equiv_at_ref!r}、"
            f"目标 r {r!r}、参考 r {r_ref!r}(参考半径必须为正)。"
        )
    return float(px_equiv_at_ref) * (float(r) / float(r_ref)) ** 4


def _quadrant_stats(resid: np.ndarray, uv: np.ndarray, cx: float,
                    cy: float) -> list[dict]:
    """按画面四象限分桶统计(象限以主点 cx/cy 分界,系统偏移与散布同尺)。

    图像坐标 v 轴朝下:v ≥ cy 是下半幅。四个象限的命名带判据原文,
    每桶给 n / rms_px / mean_du / mean_dv(与 band_stats 同一把尺)。
    """
    res = np.asarray(resid, dtype=np.float64)
    arr = np.asarray(uv, dtype=np.float64)
    finite = np.isfinite(res).all(axis=1) & np.isfinite(arr).all(axis=1)
    right = arr[:, 0] >= cx
    lower = arr[:, 1] >= cy
    named = (
        ("左上(u<cx, v<cy)", ~right & ~lower),
        ("右上(u>=cx, v<cy)", right & ~lower),
        ("左下(u<cx, v>=cy)", ~right & lower),
        ("右下(u>=cx, v>=cy)", right & lower),
    )
    rows: list[dict] = []
    for name, raw_mask in named:
        m = finite & raw_mask
        n = int(np.count_nonzero(m))
        row: dict = {"quadrant": name, "n": n, "rms_px": None,
                     "mean_du": None, "mean_dv": None}
        if n:
            d = res[m]
            norms = np.linalg.norm(d, axis=1)
            row.update(
                rms_px=float(np.sqrt(np.mean(norms**2))),
                mean_du=float(np.mean(d[:, 0])),
                mean_dv=float(np.mean(d[:, 1])),
            )
        rows.append(row)
    return rows


def _per_view_summary(per_view_rms: np.ndarray) -> dict:
    """逐视图 RMS 表的汇总(中位/均值/最大;NaN 计数不算进统计)。"""
    rms = np.asarray(per_view_rms, dtype=np.float64).ravel()
    finite = rms[np.isfinite(rms)]
    summary: dict = {
        "n_views": int(rms.size),
        "rms_mean_px": None,
        "rms_median_px": None,
        "rms_max_px": None,
    }
    if finite.size:
        summary.update(
            rms_mean_px=float(np.mean(finite)),
            rms_median_px=float(np.median(finite)),
            rms_max_px=float(np.max(finite)),
        )
    return summary


def _extrapolation_area_frac(image_size: tuple[int, int], fx: float, fy: float,
                             cx: float, cy: float, r_max_obs: float) -> float:
    """外推区的画面像素占比:r > 最远观测 r 的像素占整幅画面的比例。

    逐像素按同一条归一化半径公式现算(2560×1440 量级毫秒级),这是
    "纯外推区到底多大"的直观物理量。
    """
    width, height = int(image_size[0]), int(image_size[1])
    ru = ((np.arange(width, dtype=np.float64) - cx) / fx).reshape(1, width)
    rv = ((np.arange(height, dtype=np.float64) - cy) / fy).reshape(height, 1)
    r_grid = np.sqrt(ru**2 + rv**2)
    return float(np.mean(r_grid > r_max_obs))


def _extrapolation_block(
    coverage: dict, sigma_k2: float, params_vec: np.ndarray, model,
    object_points: np.ndarray, ref_rvec: np.ndarray, ref_tvec: np.ndarray,
    image_size: tuple[int, int],
) -> dict:
    """外推披露的数值块:σ 级 k2 晃动按 r⁴ 律折算 + findings 结论 5 判据表。

    σ 级晃动基准与 budget 段同源:k2 的分组经验 σ(取自 evaluate.json
    的 slices.empirical_std)经 ±σ 扰动投影 88 角点换算成像素当量
    (_px_dispersion 同款调用);参考半径取当前数据的最远观测 r。
    判据表三行的数值全部由公式从当前数据现算,不抄死 findings 数字。
    """
    interval = coverage["extrapolation_interval"]
    r_max_obs = coverage["r_max_obs"]
    r_corner = coverage["r_corner_frame"]
    if interval is None or r_max_obs is None or r_max_obs <= 0:
        return {
            "applicable": False,
            "note": "全域覆盖(观测到了画面四角),无畸变外推区,无需折算。",
        }
    unit_k2 = np.zeros_like(params_vec)
    unit_k2[list(model.param_names).index("k2")] = 1.0
    px_equiv = _px_dispersion(
        [params_vec, params_vec + sigma_k2 * unit_k2],
        model, object_points, ref_rvec, ref_tvec,
    )
    w_04 = k2_wobble_px(px_equiv, 0.4, r_max_obs)
    w_05 = k2_wobble_px(px_equiv, 0.5, r_max_obs)
    w_corner = k2_wobble_px(px_equiv, r_corner, r_max_obs)
    names = list(model.param_names)
    i_fx, i_fy = names.index("fx"), names.index("fy")
    i_cx, i_cy = names.index("cx"), names.index("cy")
    area_frac = _extrapolation_area_frac(
        image_size,
        float(params_vec[i_fx]), float(params_vec[i_fy]),
        float(params_vec[i_cx]), float(params_vec[i_cy]),
        r_max_obs,
    )
    return {
        "applicable": True,
        "law_note": _WOBBLE_LAW_NOTE,
        "k2_sigma": float(sigma_k2),
        "k2_sigma_px_at_r_max_obs": float(px_equiv),
        "r_max_obs": float(r_max_obs),
        "r_corner_frame": float(r_corner),
        "k2_wobble_at_r_04": float(w_04),
        "k2_wobble_at_r_05": float(w_05),
        "k2_wobble_at_corner_frame": float(w_corner),
        "extrapolation_area_frac": float(area_frac),
        "criteria_table": [
            {
                "band": "r <= 0.4(中区)",
                "wobble_px": float(w_04),
                "conclusion": "现状够用,不用治",
            },
            {
                "band": f"r 0.5–{r_max_obs:.3f}(当前外环)",
                "wobble_px_min": float(w_05),
                "wobble_px_max": float(px_equiv),
                "conclusion": "记入误差预算;指标要求苛刻再补",
            },
            {
                "band": f"r > {r_max_obs:.2f}(四角区,未拍到)",
                "wobble_px_max": float(w_corner),
                "conclusion": "纯外推区,必须先补边缘覆盖或上先验,不要直接用",
            },
        ],
    }


def _build_extrapolation_note(block: dict) -> str:
    """把外推披露数值块写成一句通俗中文(自动文案,数字全部现算)。"""
    if not block.get("applicable"):
        return str(block.get("note", ""))
    return (
        f"最远观测 r={block['r_max_obs']:.3f},画面四角 r="
        f"{block['r_corner_frame']:.3f};r∈[{block['r_max_obs']:.2f}, "
        f"{block['r_corner_frame']:.2f}] 区间零观测,是畸变外推区"
        f"(占画面像素 {block['extrapolation_area_frac']:.1%}),k2/k3 在这里"
        f"没有观测约束、纯靠外推。k2 σ 级晃动当量 "
        f"{block['k2_sigma_px_at_r_max_obs']:.3f}px(在 r="
        f"{block['r_max_obs']:.3f} 处),按 r⁴ 律折算:四角方向约 "
        f"{block['k2_wobble_at_corner_frame']:.3f}px。判据:中区 r<=0.4 约 "
        f"{block['k2_wobble_at_r_04']:.3f}px 现状够用;外环 r 0.5–"
        f"{block['r_max_obs']:.3f} 约 {block['k2_wobble_at_r_05']:.3f}–"
        f"{block['k2_sigma_px_at_r_max_obs']:.3f}px 记入误差预算;外推区"
        f"必须先补覆盖再谈精度。"
    )


def run_residual_structure(
    model_id: str,
    evaluate_report: dict,
    calib_run: Path | str,
    corners_train: ObservationSet,
    corners_test: ObservationSet,
    board,
    cfg,
) -> tuple[dict, dict]:
    """编排 ANA-02 残差方向分解:train/test 双侧分带统计 + 覆盖足迹 + 外推披露。

    纯消费既有产物、零标定求解:test 侧只做"内参当尺子"的只估位姿小
    求解(与存疑帧复核/评估同一把尺)。返回 (section, detail):

    - section 进 analyze.json 的 residual_structure 段
      (schema analyze-residual-v1);
    - detail 是逐点明细大数组(r_train / band_ids_train / uv_train /
      resid_test / r_test / image_ids_train / image_ids_test),纯数组,
      交给 _residual_detail 走 npz 旁车。

    参数:
        model_id:        目标模型短名(evaluate.json 的 selection.selected);
        evaluate_report: evaluate.json 读回的完整报告(r 归一化基准取
                         delivery 的 final_params,k2 经验 σ 取 slices);
        calib_run:       标定轮目录(residuals.npz / extrinsics.npz /
                         result.json 三件套按 {model_id}__scipy 取);
        corners_train:   train 观测集(经 load_observation_set 装载,
                         行序契约的唯一承载);
        corners_test:    test 观测集(同上);
        board:           棋盘规格(物点网格由此构造);
        cfg:             全量配置(分带边界读 cfg.analyze.radial_bin_edges)。
    """
    edges = tuple(float(x) for x in cfg.analyze.radial_bin_edges)
    model = create_model(model_id)
    object_points = build_object_points(board)

    # ---- 1) 归一化基准:evaluate delivery 的 final_params(与选型同源),
    #         基准随产物声明,并与 findings 结论 5 的基准(标定轮的
    #         fx/fy/cx/cy)做相对差核对——漂移披露不静默(T-05-10)----
    delivery_models = _require(
        evaluate_report.get("delivery", {}), "models", "evaluate.json 的 delivery 段"
    )
    final_params = _require(
        delivery_models[model_id], "final_params",
        f"delivery.models[{model_id}] 的交付参数",
    )
    param_names = list(model.param_names)
    for name in ("fx", "fy", "cx", "cy", "k2"):
        _require(final_params, name, f"delivery final_params 的 {name}")
    params_vec = np.array(
        [float(final_params[name]) for name in param_names], dtype=np.float64
    )
    idx = {name: i for i, name in enumerate(param_names)}
    fx, fy, cx, cy = (float(params_vec[idx[n]]) for n in ("fx", "fy", "cx", "cy"))
    evaluate_round = str(
        (evaluate_report.get("lineage") or {}).get("round_id") or "未知轮次"
    )
    r_normalization_basis = f"evaluate delivery final_params @ {evaluate_round}"

    combo_dir = Path(calib_run) / f"{model_id}__scipy"
    baseline = load_result(combo_dir).get("params") or {}
    max_rel = 0.0
    have_baseline = all(name in baseline for name in ("fx", "fy", "cx", "cy"))
    if have_baseline:
        for name in ("fx", "fy", "cx", "cy"):
            a = float(final_params[name])
            b = float(baseline[name])
            max_rel = max(max_rel, abs(a - b) / max(abs(a), abs(b), 1e-12))
    baseline_status = "consistent" if (have_baseline and max_rel < 1e-6) else "drifted"
    baseline_note = (
        f"r 归一化基准与 findings 结论 5 所用基准(标定 {Path(calib_run).name} "
        f"{model_id}__scipy 的 fx/fy/cx/cy)逐位一致(最大相对差 {max_rel:.2e})"
        if baseline_status == "consistent" else
        f"r 归一化基准与 findings 结论 5 所用基准(标定 {Path(calib_run).name} "
        f"{model_id}__scipy)出现漂移:fx/fy/cx/cy 最大相对差 {max_rel:.2e}。"
        f"上游换轮属正常现象,本段全部 r 值按声明基准(evaluate delivery "
        f"{evaluate_round})计算,跨轮比较时请先对齐口径。"
    )

    # ---- 2) train 侧:残差直接读标定落盘产物(标定解过的位姿,不重解)----
    residuals_doc = load_residuals(combo_dir)
    extrinsics_doc = load_extrinsics(combo_dir)
    resid_train = residuals_doc["residuals"]
    train_ids = residuals_doc["image_ids"]
    if train_ids != extrinsics_doc["image_ids"]:
        raise AssertionError(
            "residuals.npz 与 extrinsics.npz 的照片编号对不上:"
            "两份产物必须逐行同序(行序契约),串了轮次或文件损坏都会"
            "触发本断言,请重跑 calib calibrate。"
        )
    if tuple(corners_train.image_ids) != train_ids:
        raise AssertionError(
            "train 观测集的照片编号与标定结果 residuals.npz 对不上:"
            "观测集顺序 = 求解结果顺序 = 落盘顺序(行序契约)。"
            "两边多半不是同一轮角点产物,请核对 corners 轮次后重跑。"
        )
    uv_train = np.asarray(corners_train.image_points, dtype=np.float64)
    if uv_train.shape != resid_train.shape:
        raise AssertionError(
            f"train 角点数组形状 {uv_train.shape} 与残差数组形状 "
            f"{resid_train.shape} 对不上:逐角点残差必须与实测角点一一"
            f"对齐,请核对棋盘规格与角点轮次。"
        )
    image_size = tuple(int(x) for x in corners_train.image_size)
    per_view = _per_view_summary(residuals_doc["per_view_rms"])

    uv_flat = uv_train.reshape(-1, 2)
    r_train_flat, _r4_train = radius_footprint(uv_flat, fx, fy, cx, cy)
    band_ids_flat, n_beyond_train = assign_bands(r_train_flat, edges)
    bands_train = band_stats(resid_train.reshape(-1, 2), r_train_flat, edges)
    quadrants_train = _quadrant_stats(resid_train.reshape(-1, 2), uv_flat, cx, cy)
    coverage_train = coverage_summary(uv_flat, fx, fy, cx, cy, image_size)

    # ---- 3) test 侧:位姿未知,逐帧只估位姿小求解后回代出逐角点残差
    #         (与 evaluate 交付评分同一把尺);单帧失败 NaN 占行不中断
    #         (T-05-11:失败留痕,不冒充、不静默)----
    uv_test = np.asarray(corners_test.image_points, dtype=np.float64)
    n_test = uv_test.shape[0]
    n_points = uv_test.shape[1]
    resid_test = np.full((n_test, n_points, 2), np.nan, dtype=np.float64)
    rms_rows = np.full(n_test, np.nan, dtype=np.float64)
    n_failed = 0
    for i in range(n_test):
        uv_view = uv_test[i]
        finite_pts = int(np.count_nonzero(np.isfinite(uv_view).all(axis=1)))
        try:
            if finite_pts < _MIN_PNP_POINTS:
                raise ValueError(
                    f"可用角点 {finite_pts} 个,少于位姿求解下限 "
                    f"{_MIN_PNP_POINTS} 个:方程组欠定,该帧按失败占行。"
                )
            rvec, tvec, rms = solve_pose_only(
                uv_view, object_points, params_vec, model)
            pts_cam, _pose_block = transform_to_camera(object_points, rvec, tvec)
            uv_pred = model.project(pts_cam, params_vec)
            resid_test[i] = uv_pred - uv_view
            rms_rows[i] = float(rms)
        except ValueError:
            n_failed += 1  # 失败帧 NaN 占行 + 计数,不中断整批
    rms_finite = rms_rows[np.isfinite(rms_rows)]
    per_frame = {
        "median_px": float(np.median(rms_finite)) if rms_finite.size else None,
        "max_px": float(np.max(rms_finite)) if rms_finite.size else None,
    }

    uv_test_flat = uv_test.reshape(-1, 2)
    r_test_flat, _r4_test = radius_footprint(uv_test_flat, fx, fy, cx, cy)
    bands_test = band_stats(resid_test.reshape(-1, 2), r_test_flat, edges)
    coverage_test = coverage_summary(uv_test_flat, fx, fy, cx, cy, image_size)

    # ---- 4) 外推披露:σ 级 k2 晃动按 r⁴ 律折算 + findings 结论 5 判据表
    #         (σ 取 slices.empirical_std 的 k2,与 budget 段同源;
    #          参考位姿取 extrinsics 首视图,findings 结论 1 惯例)----
    slices_models = _require(
        evaluate_report.get("slices", {}), "models", "evaluate.json 的 slices 段"
    )
    empirical_std = _require(
        slices_models[model_id], "empirical_std",
        f"slices.models[{model_id}] 的分组经验 σ",
    )
    sigma_k2 = float(_require(empirical_std, "k2", "empirical_std 的 k2"))
    ref_rvec = np.asarray(extrinsics_doc["rvecs"][0], dtype=np.float64)
    ref_tvec = np.asarray(extrinsics_doc["tvecs"][0], dtype=np.float64)
    extrapolation = _extrapolation_block(
        coverage_train, sigma_k2, params_vec, model, object_points,
        ref_rvec, ref_tvec, image_size,
    )
    extrapolation_note = _build_extrapolation_note(extrapolation)

    # ---- 5) 段与明细组装 ----
    section = {
        "schema": SCHEMA_RESIDUAL,
        "model_id": model_id,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "r_normalization_basis": r_normalization_basis,
        "r_normalization_params": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "r_normalization_baseline_check": {
            "status": baseline_status,
            "max_rel_diff": float(max_rel),
            "findings_baseline": (
                f"标定 {Path(calib_run).name} {model_id}__scipy 的 fx/fy/cx/cy"
                "(docs/findings.md 结论 5 所用基准)"
            ),
            "note": baseline_note,
        },
        "band_edges": list(edges),
        "train": {
            "per_view": per_view,
            "bands": bands_train,
            "quadrants": quadrants_train,
            "coverage": coverage_train,
            "n_beyond_last_edge": int(n_beyond_train),
        },
        "test": {
            "n_frames": int(n_test),
            "n_failed": int(n_failed),
            "bands": bands_test,
            "coverage": coverage_test,
            "per_frame_rms": per_frame,
        },
        "extrapolation": extrapolation,
        "extrapolation_note": extrapolation_note,
        "sources": {
            "calibration_run": Path(calib_run).name,
            "residuals": f"{model_id}__scipy/residuals.npz(行序经 extrinsics 对齐)",
            "corners_train": "corners_train_sb.npz(经 load_observation_set 装载)",
            "corners_test": "corners_test_sb.npz(经 load_observation_set 装载)",
            "evaluate_round": evaluate_round,
            "findings_basis": "docs/findings.md 结论 5(r⁴ 覆盖判据与折算公式)",
        },
    }
    detail = {
        "r_train": r_train_flat.reshape(uv_train.shape[:2]),
        "band_ids_train": band_ids_flat.reshape(uv_train.shape[:2]),
        "uv_train": uv_train,
        "resid_test": resid_test,
        "r_test": r_test_flat.reshape(uv_test.shape[:2]),
        "image_ids_train": np.asarray(train_ids),
        "image_ids_test": np.asarray(corners_test.image_ids),
    }
    return section, detail
