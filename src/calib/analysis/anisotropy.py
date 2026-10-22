"""COR-03 连拍各向异性:同一姿态连拍的同一角点,横向(u)与纵向(v)的
重复定位散布之比——横竖两个方向"抖得一样厉害吗"。

白话背景:相机对着同一摆放位置连拍 8 张,画面里的棋盘没动,角点位置
却每次差一点。把每个角点在连拍间的横向散布(σu)与纵向散布(σv)相比,
就得到"各向异性"(aniso = σu/σv,越接近 1 = 横纵对称)。如果画面某片
区域显著大于 1,说明那个方向的定位更不稳,结合角点寻优(corner_opt)的
u/v 分量证据可以判断要不要换角点方法——只提示方向,不自动执行任何改动。

原料是阶段三已落盘的逐角点重复性(output/noise/<轮>/temporal_sigma_per_corner.npz,
纯消费,本模块不做任何检测/求解):折算姿态(immerkaer_transfer)没有
真实连拍重复,行可能是折算值或 NaN,必须按 sources == "temporal" 硬过滤
(Pitfall 9:来源混入会污染统计),NaN 行显式排除并计数落盘。

防零除:分母 σv 先 clamp 到检测器级 σ 量级的下限(检测器噪声约
0.0087px,散布比它还小说明量在测量不确定度之内,比值再大也是假大),
杜绝 inf/近零放大。判读给物理量级白话:对照检测器 σ 给"几倍",按照片里
方格边长折算毫米,再对照发丝直径(约 50µm)给几分之一。

空间切片与 ANA-02 残差方向分解共用同一把 r 分带尺
(residual_structure 的 radius_footprint / assign_bands,一处实现两处用,
严禁第二份);画面四象限以主点 cx/cy 分界,象限命名与 residual_structure
同一套。单一数学源红线(D-06/D-10):本模块不出现任何投影/导数/求解
公式,也绝不 import OpenCV。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calib.analysis.residual_structure import assign_bands, radius_footprint
from calib.solve.observation import ObservationSet

# 段模式版本:连拍各向异性段的结构约定(加键不改键算兼容,删键必须升版本)
SCHEMA_ANISOTROPY = "analyze-anisotropy-v1"

# 显著各向异性判读线:σu/σv > 1.5 记"横纵不对称显著"(显著线随段落账,
# 判读文案引用,不抄死在文案里)
_SIGNIFICANT_ANISO = 1.5

# 发丝直径参照(白话物理量级锚点,memory 规则:误差给参照物量级)
_HAIR_DIAMETER_MM = 0.05

# 画面四象限命名(与 residual_structure._quadrant_stats 同一套顺序与写法,
# 图像坐标 v 轴朝下:v ≥ cy 是下半幅)
_QUADRANT_NAMES = (
    "左上(u<cx, v<cy)",
    "右上(u>=cx, v<cy)",
    "左下(u<cx, v>=cy)",
    "右下(u>=cx, v>=cy)",
)


def _require(mapping, key: str, what: str):
    """从上游报告取键,缺失当场中文报错(体例照 residual_structure 同名防线)。"""
    if not isinstance(mapping, dict) or key not in mapping:
        raise AssertionError(
            f"上游报告缺键 {key}({what})。连拍各向异性按契约消费落盘产物,"
            f"缺键多半是产物与当前代码不是同一套契约,或串了轮次,请重跑"
            f"对应上游命令再来。"
        )
    return mapping[key]


def _load_temporal_npz(npz_path: Path | str) -> dict:
    """读回阶段三的逐角点重复性 npz,读前先验形状(上游文件坏了当场响)。

    键契约(与生产端 calib/noise/temporal.py 同一套):pose_ids (80,) /
    sigma_per_corner_all (80, 点数, 2) / sources (80,) / sigma_detector (80,)。
    读取强制 allow_pickle=False(npz 的 pickle 模式可以执行任意代码,
    项目全部读回路径的既定防线)。
    """
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        if "sigma_per_corner_all" not in data.files \
                or "sources" not in data.files or "pose_ids" not in data.files:
            raise AssertionError(
                f"时域逐角点文件 {path.name} 缺契约键"
                f"(需要 pose_ids / sigma_per_corner_all / sources,实际有 "
                f"{sorted(data.files)})。这份文件多半不是 calib noise 的"
                f" temporal_sigma_per_corner.npz,请核对轮次后重跑 calib noise。"
            )
        pose_ids = np.asarray(data["pose_ids"]).ravel()
        sigma_all = np.asarray(data["sigma_per_corner_all"], dtype=np.float64)
        sources = np.asarray([str(x) for x in np.asarray(data["sources"]).ravel()])
        sigma_detector_col = (
            np.asarray(data["sigma_detector"], dtype=np.float64).ravel()
            if "sigma_detector" in data.files else None
        )
    n = pose_ids.shape[0]
    if sigma_all.ndim != 3 or sigma_all.shape[0] != n \
            or sigma_all.shape[2] != 2 or sources.shape[0] != n:
        raise AssertionError(
            f"时域逐角点文件 {path.name} 的形状/行数对不上:pose_ids {n} 行、"
            f"sigma_per_corner_all {sigma_all.shape}、sources "
            f"{sources.shape[0]} 条。期望 (姿态数, 点数, 2) 且三者行数一致,"
            f"请重跑 calib noise。"
        )
    return {
        "pose_ids": pose_ids,
        "sigma_all": sigma_all,
        "sources": sources,
        "sigma_detector_col": sigma_detector_col,
    }


def run_anisotropy(
    model_id: str,
    evaluate_report: dict,
    temporal_npz_path: Path | str,
    corners_train: ObservationSet,
    board,
    cfg,
    sigma_detector_floor: float | None = None,
    corner_opt_section: dict | None = None,
) -> tuple[dict, dict]:
    """编排 COR-03 连拍各向异性:temporal 过滤 + 防零除 + 分带/象限切片。

    纯消费块(零检测零求解):原料是阶段三已落盘的逐角点重复性 npz;
    train 观测集只用来给 88 个角点定位(算空间切片的分带与象限)。
    返回 (section, detail):

    - section 进 analyze.json 的 anisotropy 段
      (schema analyze-anisotropy-v1);
    - detail 是逐姿态明细大数组(pose_ids / aniso / sigma_uv / valid_mask /
      band_ids / quadrant_ids),纯数组,交给 _anisotropy_detail 走 npz 旁车。

    参数:
        model_id:             目标模型短名(evaluate.json 的 selection.selected);
        evaluate_report:      evaluate.json 读回的完整报告(r 归一化基准取
                              delivery 的 final_params,与残差方向分解同源);
        temporal_npz_path:    阶段三 temporal_sigma_per_corner.npz 路径;
        corners_train:        train 观测集(经 load_observation_set 装载,
                              角点像素坐标的唯一承载,行序契约);
        board:                棋盘规格(角点 (行,列) 网格坐标由此映射);
        cfg:                  全量配置(分带边界读 cfg.analyze.radial_bin_edges);
        sigma_detector_floor: 检测器级 σ(noise.json 的 sigma_detector_corner,
                              由 CLI 层读入传入;缺省时退回 npz 自带
                              sigma_detector 列的中位,并把口径来源落账);
        corner_opt_section:   本轮 corner_opt 段(在盘时给互证行引用其
                              u/v 分量均值方向;None 时互证行注明缺席)。
    """
    npz = _load_temporal_npz(temporal_npz_path)
    sigma_all = npz["sigma_all"]
    sources = npz["sources"]
    pose_ids_all = npz["pose_ids"]
    n_points = sigma_all.shape[1]

    # ---- 1) 来源硬过滤(Pitfall 9):只有 temporal 是真连拍重复,
    #         折算姿态(immerkaer_transfer)混进来就是假数据 ----
    mask_temporal = sources == "temporal"
    n_temporal = int(np.count_nonzero(mask_temporal))
    if n_temporal == 0:
        raise AssertionError(
            f"时域逐角点文件 {Path(temporal_npz_path).name} 里没有任何 "
            f"sources == 'temporal' 的真连拍姿态(全部来源:"
            f"{sorted(set(sources.tolist()))})。各向异性必须有真实连拍重复,"
            f"请先排查 calib noise 的时域噪声底再重跑。"
        )
    n_excluded_sources = int(sources.shape[0]) - n_temporal
    excluded_source_list = sorted({s for s in sources.tolist() if s != "temporal"})
    excluded_pose_list = [
        str(p) for p, m in zip(pose_ids_all, mask_temporal) if not m
    ]

    sigma_t = sigma_all[mask_temporal]                      # (n_temporal, P, 2)
    pose_ids = pose_ids_all[mask_temporal]
    # temporal 姿态内仍可能有个别角点行含 NaN(检出缺角等):角点级显式
    # 排除并计数,不静默也不中断
    valid_mask = np.isfinite(sigma_t).all(axis=2)           # (n_temporal, P)
    n_excluded_nan = int(np.count_nonzero(~valid_mask))

    # ---- 2) 防零除下限:检测器级 σ(口径铁律唯一入口 noise.json 的
    #         sigma_detector_corner,由 CLI 传入;未传退回 npz 列中位,
    #         来源随段落账不静默)----
    if sigma_detector_floor is not None:
        floor = float(sigma_detector_floor)
        floor_source = "noise.json 的 sigma_detector_corner(CLI 层读入传入)"
    elif npz["sigma_detector_col"] is not None:
        floor = float(np.median(npz["sigma_detector_col"]))
        floor_source = "npz 自带 sigma_detector 列的中位(调用方未传入 noise 口径)"
    else:
        raise AssertionError(
            "防零除下限拿不到:调用方未传 sigma_detector_floor,且时域 npz "
            "里也没有 sigma_detector 列。请改由 noise.json 的 "
            "sigma_detector_corner 传入(口径铁律唯一入口)后重跑。"
        )
    if not np.isfinite(floor) or floor <= 0:
        raise AssertionError(
            f"检测器级 σ 下限必须是正的有限值,实际拿到 {floor!r}"
            f"(来源:{floor_source})。坏口径进统计等于假比值,"
            f"请先排查 calib noise 再来。"
        )

    # ---- 3) 各向异性:aniso = σu / max(σv, 下限);NaN 行输出 NaN ----
    sigma_u = sigma_t[:, :, 0]
    sigma_v = sigma_t[:, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        aniso = np.where(valid_mask, sigma_u / np.maximum(sigma_v, floor),
                         np.nan)

    # ---- 4) 汇总统计(只算有效值;NaN 不掺进中位/分位)----
    valid_vals = aniso[valid_mask]
    n_valid = int(valid_vals.size)
    summary: dict = {
        "n_valid_values": n_valid,
        "median": float(np.median(valid_vals)) if n_valid else None,
        "p90": float(np.percentile(valid_vals, 90)) if n_valid else None,
        "max": float(np.max(valid_vals)) if n_valid else None,
        "sigma_u_median": float(np.median(sigma_u[valid_mask])) if n_valid else None,
        "sigma_v_median": float(np.median(sigma_v[valid_mask])) if n_valid else None,
        "n_aniso_gt_1p5": int(np.count_nonzero(valid_vals > _SIGNIFICANT_ANISO)),
    }
    per_corner_median = (
        np.nanmedian(aniso, axis=0) if n_valid else np.full(n_points, np.nan)
    )   # (P,) 逐角点跨姿态中位:NaN 不进中位,整列全 NaN 的角点保持 NaN
    sig_line = _SIGNIFICANT_ANISO
    summary["significant_line"] = sig_line
    summary["aniso_gt_1p5_positions"] = [
        {"row": int(idx // board.cols), "col": int(idx % board.cols),
         "point_idx": int(idx), "median_aniso": float(per_corner_median[idx])}
        for idx in range(n_points) if np.isfinite(per_corner_median[idx])
        and per_corner_median[idx] > sig_line
    ]

    # ---- 5) 空间切片:r 归一化基准与残差方向分解同源(evaluate delivery
    #         的 final_params),分带器共用 residual_structure 同一把尺 ----
    delivery_models = _require(
        evaluate_report.get("delivery", {}), "models", "evaluate.json 的 delivery 段"
    )
    final_params = _require(
        delivery_models[model_id], "final_params",
        f"delivery.models[{model_id}] 的交付参数",
    )
    for name in ("fx", "fy", "cx", "cy"):
        _require(final_params, name, f"delivery final_params 的 {name}")
    fx, fy = float(final_params["fx"]), float(final_params["fy"])
    cx, cy = float(final_params["cx"]), float(final_params["cy"])
    evaluate_round = str(
        (evaluate_report.get("lineage") or {}).get("round_id") or "未知轮次"
    )
    r_normalization_basis = f"evaluate delivery final_params @ {evaluate_round}"

    uv_train = np.asarray(corners_train.image_points, dtype=np.float64)
    if uv_train.ndim != 3 or uv_train.shape[1] != n_points or uv_train.shape[2] != 2:
        raise AssertionError(
            f"train 观测集的角点数组形状 {uv_train.shape} 与时域 npz 的 "
            f"角点数 {n_points} 对不上:两边必须是同一块棋盘的同一套角点"
            f"编号(逐角点一一对应),请核对角点轮次与棋盘规格。"
        )
    edges = tuple(float(x) for x in cfg.analyze.radial_bin_edges)
    # 逐角点的代表位置:r 取该角点全部观测的归一化半径中位,象限取
    # 全部观测的像素坐标中位(单角点跨照片位置近似不动,取中位抗单帧离群)
    r_grid = radius_footprint(uv_train.reshape(-1, 2), fx, fy, cx, cy)[0] \
        .reshape(uv_train.shape[0], n_points)
    r_per_corner = np.median(r_grid, axis=0)                # (P,)
    band_ids, _n_beyond = assign_bands(r_per_corner, edges)
    uv_median = np.median(uv_train, axis=0)                 # (P, 2)
    right = uv_median[:, 0] >= cx
    lower = uv_median[:, 1] >= cy
    quadrant_ids = np.where(right, np.where(lower, 3, 1),
                            np.where(lower, 2, 0)).astype(np.int64)

    def _median_aniso(point_mask: np.ndarray) -> tuple[int, int, float | None]:
        """带/象限内 aniso 中位(逐姿态逐角点全摊开算,NaN 不掺)。"""
        vals = aniso[:, point_mask]
        finite = vals[np.isfinite(vals)]
        return int(point_mask.sum()), int(finite.size), (
            float(np.median(finite)) if finite.size else None
        )

    by_band: list[dict] = []
    for i in range(len(edges) - 1):
        n_pts, n_vals, med = _median_aniso(band_ids == i)
        by_band.append({
            "band_lo": edges[i], "band_hi": edges[i + 1],
            "n_points": n_pts, "n_values": n_vals, "median_aniso": med,
        })
    by_quadrant: list[dict] = []
    for q, name in enumerate(_QUADRANT_NAMES):
        n_pts, n_vals, med = _median_aniso(quadrant_ids == q)
        by_quadrant.append({
            "quadrant": name, "n_points": n_pts,
            "n_values": n_vals, "median_aniso": med,
        })

    # 判读行(开放判读:只报数据不下强结论)
    def _top(rows, name_key):
        scored = [(r.get("median_aniso"), r) for r in rows
                  if r.get("median_aniso") is not None]
        if not scored:
            return "无有效数据"
        best_val, best = max(scored, key=lambda t: t[0])
        return f"{best[name_key]}(中位 {best_val:.2f})"

    band_txt = _top(by_band, "band_lo")
    quad_txt = _top(by_quadrant, "quadrant")
    reading = (
        f"各向异性中位 {summary['median']:.2f};显著(> {sig_line:g})的"
        f"(姿态, 角点)值有 {summary['n_aniso_gt_1p5']} 个,集中在角点"
        f"{len(summary['aniso_gt_1p5_positions'])} 个;分带里最高:{band_txt},"
        f"象限里最高:{quad_txt}。是否与画面内容方向相关属开放判读,"
        f"本段只报数据不下强结论。"
    ) if n_valid else "无有效各向异性数据。"

    # ---- 6) 物理量级白话(memory 规则:误差给参照物量级,数值全部
    #         由公式现算;方格间距量不出时毫米折算如实缺席不硬除)----
    if n_valid and summary["sigma_u_median"] is not None:
        px_part = (
            f"横向重复定位散布中位约 {summary['sigma_u_median']:.4f}px,"
            f"纵向约 {summary['sigma_v_median']:.4f}px,横纵之比中位 "
            f"{summary['median']:.2f}(接近 1 = 横纵对称);对照检测器噪声"
            f" σ ≈ {floor:.4f}px,横向散布约是它的 "
            f"{summary['sigma_u_median'] / floor:.0f} 倍。"
        )
        sq_px_median = _median_square_px(uv_train, board)
        if sq_px_median > 0:
            mm_per_px = float(board.square_size_mm) / sq_px_median
            sigma_u_mm = summary["sigma_u_median"] * mm_per_px
            hair_ratio = _HAIR_DIAMETER_MM / sigma_u_mm if sigma_u_mm > 0 else np.inf
            hair_txt = (
                f"约为发丝直径(约 {_HAIR_DIAMETER_MM * 1000:.0f}µm)的 "
                f"{hair_ratio:.0f} 分之一"
                if hair_ratio >= 1 else
                f"已超过发丝直径(约 {_HAIR_DIAMETER_MM * 1000:.0f}µm)在棋盘"
                f"平面上的投影({sigma_u_mm:.3f}mm > {_HAIR_DIAMETER_MM}mm)"
            )
            px_part += (
                f"物理量级:按照片里方格边长约 {sq_px_median:.0f}px 折算,"
                f"棋盘平面上 1px ≈ {mm_per_px:.3f}mm,横向散布中位约 "
                f"{sigma_u_mm:.3f}mm,{hair_txt}。"
            )
        else:
            px_part += (
                "物理量级:照片里量不出方格间距(角点布局退化),毫米折算"
                "缺席;像素级参照见上。"
            )
        plain_language = px_part
    else:
        plain_language = "无有效各向异性数据,无法给物理量级参照。"

    # ---- 7) 互证行:与 corner_opt 段 u/v 分量证据链互引(缺席不冒充)----
    cross_ref = _build_cross_ref(corner_opt_section)

    section = {
        "schema": SCHEMA_ANISOTROPY,
        "model_id": model_id,
        "n_poses_total": int(sources.shape[0]),
        "n_temporal_poses": n_temporal,
        "n_excluded_sources": n_excluded_sources,
        "excluded_source_list": excluded_source_list,
        "excluded_pose_ids": excluded_pose_list,
        "n_excluded_nan": n_excluded_nan,
        "sigma_detector_floor_px": floor,
        "sigma_detector_floor_source": floor_source,
        "band_edges": list(edges),
        "r_normalization_basis": r_normalization_basis,
        "r_normalization_params": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "image_size": [int(corners_train.image_size[0]),
                       int(corners_train.image_size[1])],
        "summary": summary,
        "by_band": by_band,
        "by_quadrant": by_quadrant,
        "reading": reading,
        "plain_language": plain_language,
        "cross_ref": cross_ref,
        "per_corner_median_aniso": [
            float(x) if np.isfinite(x) else None for x in per_corner_median
        ],
        "aniso_hist": _hist_block(valid_vals),
        "sources": {
            "temporal_npz": (
                f"output/noise/{Path(temporal_npz_path).parent.name}/"
                f"{Path(temporal_npz_path).name}"
                "(sources == 'temporal' 硬过滤,NaN 角点行显式排除并计数)"
            ),
            "corners_train": "corners_train_sb.npz(经 load_observation_set 装载,"
                             "只取角点像素坐标做空间切片)",
            "r_normalization": f"{r_normalization_basis}(与残差方向分解同源)",
            "band_tool": "residual_structure.assign_bands / radius_footprint"
                         "(与 ANA-02 同一把分带尺,一处实现两处用)",
            "evaluate_round": evaluate_round,
        },
    }
    detail = {
        "pose_ids": pose_ids.astype(str),
        "aniso": aniso,
        "sigma_uv": sigma_t,
        "valid_mask": valid_mask,
        "band_ids": band_ids.astype(np.int64),
        "quadrant_ids": quadrant_ids,
    }
    return section, detail


def _median_square_px(uv_train: np.ndarray, board) -> float:
    """照片里方格边长的像素数(横向相邻角点间距的中位,白话折算用)。"""
    grid = uv_train.reshape(uv_train.shape[0], board.rows, board.cols, 2)
    dx = np.linalg.norm(grid[:, :, 1:, :] - grid[:, :, :-1, :], axis=3)
    return float(np.median(dx))


def _hist_block(valid_vals: np.ndarray) -> dict:
    """全部有效 aniso 值的直方图(等宽 40 桶,全域不截尾),供出图消费。"""
    if valid_vals.size == 0:
        return {"edges": [], "counts": [], "n_values": 0}
    hi = float(np.ceil(float(np.max(valid_vals)) * 10.0) / 10.0)
    lo = 0.0
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(valid_vals, bins=40, range=(lo, hi))
    return {
        "edges": [float(x) for x in edges],
        "counts": [int(c) for c in counts],
        "n_values": int(valid_vals.size),
    }


def _build_cross_ref(corner_opt_section: dict | None) -> str:
    """与角点寻优段互证:引其生产配置 u/v 分量均值的方向;缺席如实注明。"""
    if not corner_opt_section:
        return "本轮无退化实验数据(corner_opt 段缺席),无法与 u/v 分量证据互引。"
    prod = corner_opt_section.get("production_method")
    per_method = ((corner_opt_section.get("stats") or {}).get("_self") or {}).get(prod)
    if not prod or not isinstance(per_method, dict):
        return "corner_opt 段在盘但缺生产配置的自参照统计,无法互证。"
    mean_du = [float(u["mean_du"]) for u in per_method.values()
               if isinstance(u, dict) and u.get("mean_du") is not None]
    mean_dv = [float(u["mean_dv"]) for u in per_method.values()
               if isinstance(u, dict) and u.get("mean_dv") is not None]
    if not mean_du or not mean_dv:
        return "corner_opt 段的生产配置均值全为缺席,无法互证。"
    med_du, med_dv = float(np.median(mean_du)), float(np.median(mean_dv))
    bigger = "水平(u)" if abs(med_du) >= abs(med_dv) else "垂直(v)"
    return (
        f"与 corner_opt 段互证:生产配置 {prod} 在各退化单元的系统偏移均值"
        f"中位为 u {med_du:+.4f}px / v {med_dv:+.4f}px(|u| 与 |v| 相比,"
        f"{bigger}方向的系统偏移更明显);与各向异性的横纵对比互相参照。"
        f"只报方向,不下强结论。"
    )
