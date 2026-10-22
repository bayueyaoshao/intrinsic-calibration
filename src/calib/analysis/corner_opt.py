"""COR-02 角点合成退化寻优实验(阶段⑦唯一重计算块):把真实拍到的棋盘照
人为变差(加噪/模糊/低对比),用各候选检测配置重新提角点,与参照逐角点
比对,产出"退化强度 → 定位误差"曲线,用数据回答"当前角点配置是否最优"。

分工(检测逻辑唯一来源,红线):本模块只负责退化、调度、统计、留痕;
检测一律走 calib.corners.detect 的 detect_classic / detect_sb 两个入口
(输入规范化与缺陷闸门两道防线都在入口里),绝不直接调 cv2 的角点函数,
更不复制第二份检测实现——源码防线测试锁定本文件不含 cv2 角点检测与
亚像素精化函数的英文名,SB 结果也绝不进二次亚像素精化路径。

CALIB_CB_ACCURACY 语义实测(2026-10-22,合成棋盘双帧,干净图与 σ=8 加噪
各检一遍,本机 OpenCV 4.14.0):flags=0 与 flags=CALIB_CB_ACCURACY(=32)
两档检出数相同、逐角点坐标逐位一致(位移 0.0000px)——"SB 追加亚像素
精化"的预期语义未获证实,该 flag 在本机对 SB 输出无可测差异。两档仍都
保留为候选(结论由数据出,不预设):真实数据退化网格若两档曲线完全重合,
即可判"该 flag 对本数据无效"。

可复现承诺(退化图不落盘的正当性来源,体例照 noise/injection.py):每个
退化副本的随机种子由(图像, 退化单元, 方法, 重复)四元组从 base_seed
确定性推导(base_seed + sha1(...) mod 2**32)——同种子完整重跑逐位一致,
复现只需原图 + 配置里的 corner_opt_seed,不需要落盘任何退化副本(全量
会有几十 GB)。保真规则(D-13)约束的是测量结果:逐帧逐单元逐方法的
角点位移数组全量落盘(corner_opt.npz,纯数值数组无 pickle)。

确定性红线(04-02 教训):并行子进程是全新进程,worker 内首行
cv2.setNumThreads(任务携带值)——OpenCV 内部多线程的归约顺序不定,不锁
线程则并行与串行出现末位漂移。任务经 evaluation/parallel.run_parallel
调度,单任务异常包 TaskFailed 信封转 NaN 占行 + 计数留痕,任务流不断。

退化收尾纪律(照 injection Pitfall 4):三族退化统一 float 域计算 →
np.clip(0, 255) → np.rint 回 uint8。最后一步量化不能省——真实相机输出
的就是量化后的整数灰阶,跳过量化直接把 float 图喂检测器,量出来的误差
曲线与真实采集链路对不上。

指标口径(头号陷阱的教训,Pitfall 4):位移是"与参照的逐角点欧氏距离",
不是重投影误差——窗口吞邻角的系统性偏差在每个角点方向相同,会被求解
整个吸收、重投影完全看不出来;逐角点位移才看得见。

双口径并报(裁定①):主口径 _self = 各方法对"自己干净检出"的偏移
(量"退化敏感度",主曲线);旁证口径 _prod_sb = 各方法对"生产 SB
(flags=0)干净检出"的绝对偏移(跨方法可比)。结论文案两列都给。
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from calib.analysis.residual_structure import assign_bands, radius_footprint
from calib.corners.detect import detect_classic, detect_sb
from calib.corners.window import estimate_sq_px
from calib.evaluation.parallel import TaskFailed, run_parallel
from calib.io.config import AnalyzeConfig, CalibrationConfig, CornersConfig
from calib.io.dataset import imread_checked
from calib.solve.observation import ObservationSet

# 段模式版本:角点寻优段的结构约定(加键不改键算兼容,删键必须升版本)
SCHEMA_CORNER_OPT = "analyze-corner-opt-v1"

# 生产配置的方法标签:SB flags=0(corners 生产路线的定版位)
PRODUCTION_LABEL = "sb"

# full 档每退化单元重复次数(quick 档读配置 corner_opt_n_reps,默认 1;
# full 固定 2 压随机性,配置键保留只为 quick/full 契约对称,见 AnalyzeConfig 注)
_FULL_REPS = 2

# CALIB_CB_ACCURACY 语义实测结论(A1 销险,随段落盘 + 模块 docstring 固定段)
SB_ACCURACY_VERDICT = (
    "CALIB_CB_ACCURACY 语义实测(2026-10-22,合成棋盘双帧,干净图与 σ=8 加噪"
    "各检一遍,OpenCV 4.14.0):与 flags=0 检出数相同、逐角点坐标逐位一致"
    "(位移 0.0000px),「SB 追加亚像素精化」的预期语义未获证实,该 flag 在本机"
    "对 SB 输出无可测差异;两档仍都保留为候选,真实数据退化网格若两档曲线"
    "完全重合,即可判该 flag 对本数据无效。"
)


def apply_degradation(
    gray_u8: np.ndarray, family: str, level: float, rng: np.random.Generator
) -> np.ndarray:
    """给灰度图施加一族一档退化,返回 uint8 退化副本。

    三族(配置 analyze.noise_levels / blur_levels / contrast_gains 的档位):
    - noise:   加高斯噪声,σ = level(灰阶);语义与收尾照 injection 的
               add_gaussian_noise;
    - blur:    高斯模糊,σ_px = level,核径 ≈ 6σ+1(奇数);level <= 0 恒等;
    - contrast:对比度增益,输出 = 128 + (输入 − 128) × gain;gain = 1.0 恒等。

    收尾纪律(injection Pitfall 4,不能省):三族统一 float 域计算 →
    np.clip(0, 255) → np.rint 回 uint8——真实相机输出的就是量化后的整数
    灰阶,跳过量化会让量出的误差曲线与真实采集链路对不上。

    参数:
        gray_u8: 二维灰度图(uint8);
        family:  "noise" / "blur" / "contrast"(其他值中文报错,T-05-19);
        level:   该档强度(非负;强度没有负值);
        rng:     numpy 随机数生成器(种子由调用方按四元组推导)。
    """
    arr = np.asarray(gray_u8)
    if arr.ndim != 2:
        raise AssertionError(
            f"apply_degradation 的输入必须是二维灰度图;实际是 {arr.ndim} 维,"
            f"形状 {arr.shape}。请用 cv2.imread 的 IMREAD_GRAYSCALE 读图。")
    level = float(level)
    if not np.isfinite(level) or level < 0.0:
        raise AssertionError(
            f"退化强度不合法:{level!r}(必须是有限的非负数,强度没有负值)。"
            f"请检查 analyze 配置里的退化档位。")
    if family == "noise":
        out = arr.astype(np.float64) + rng.normal(0.0, level, size=arr.shape)
    elif family == "blur":
        if level <= 0.0:
            out = arr.astype(np.float64)  # 恒等档:逐位等于输入
        else:
            k = int(2 * round(3.0 * level) + 1)
            out = cv2.GaussianBlur(arr, (k, k), level).astype(np.float64)
    elif family == "contrast":
        mid = 128.0
        out = mid + (arr.astype(np.float64) - mid) * level
    else:
        raise ValueError(
            f"未知退化族 {family!r}(支持 noise/blur/contrast)。"
            f"请检查 analyze 配置里的退化档位与寻优实验的单元枚举。")
    np.clip(out, 0.0, 255.0, out=out)
    return np.rint(out).astype(np.uint8)


def _seed_for(image_id: str, unit_label: str, method_label: str, rep: int,
              base_seed: int) -> int:
    """从(图像, 退化单元, 方法, 重复)四元组推导该副本的随机种子。

    公式:base_seed + sha1("image|unit|method|rep") mod 2**32——确定性散列,
    任何一元不同种子就不同;同四元组永远同种子。这是"退化图不落盘也能
    精确复现"的正当性来源(体例照 injection._seed_for 三元组扩展一元)。
    """
    digest = hashlib.sha1(
        f"{image_id}|{unit_label}|{method_label}|{rep}".encode()).hexdigest()
    return (base_seed + int(digest, 16)) % 2**32


def build_method_candidates(sq_px: float, cfg: AnalyzeConfig,
                            profile: str = "quick") -> list[dict]:
    """按本帧方格像素边长枚举候选检测配置(方法 = 检测入口参数,不新增实现)。

    - classic 各窗口档 w ∈ cfg.classic_win_ladder 中满足 2w+1 < sq_px 者:
      超档直接不进候选,不报错——这是设计内过滤(窗口吞邻角是本项目头号
      陷阱,窗口阶梯的合法域与 window.derive_window_frame 同一条断言);
    - SB 两档:flags=0(生产位)+ flags=CALIB_CB_ACCURACY(精化位,语义
      实测见模块 docstring);
    - full 档追加 SB_EXHAUSTIVE 候选(穷举搜索位)。

    每个候选带 method_label(classic_w3 / sb / sb_acc / sb_exh),返回字典
    列表:{"kind": "classic", "label": …, "win": w} 或
         {"kind": "sb", "label": …, "flags": int}。
    """
    ladder = [int(w) for w in cfg.classic_win_ladder]
    candidates: list[dict] = []
    for w in sorted(ladder):
        if w >= 1 and 2 * w + 1 < float(sq_px):
            candidates.append({"kind": "classic", "label": f"classic_w{w}",
                               "win": w})
    candidates.append({"kind": "sb", "label": "sb", "flags": 0})
    candidates.append(
        {"kind": "sb", "label": "sb_acc", "flags": int(cv2.CALIB_CB_ACCURACY)})
    if profile == "full":
        candidates.append(
            {"kind": "sb", "label": "sb_exh", "flags": int(cv2.CALIB_CB_EXHAUSTIVE)})
    return candidates


def detect_with(candidate: dict, gray: np.ndarray, cols: int, rows: int,
                corners_cfg: CornersConfig) -> dict | None:
    """按候选配置调度检测:只走 detect_classic / detect_sb 两个入口。

    classic 候选构造指定窗口的 SubpixConfig 变体(force_win = 阶梯档,
    入口内仍断言 2w+1 < 本帧方格像素边长);SB 候选构造 sb_flags 变体。
    SB 结果绝不进入二次亚像素精化路径(本模块结构上不引用 _subpix_refine)。
    未检出返回 None(该候选该帧未检出,是数据不是事故)。
    """
    if candidate["kind"] == "classic":
        variant = replace(
            corners_cfg,
            subpix=replace(corners_cfg.subpix, force_win=int(candidate["win"])))
        return detect_classic(gray, cols, rows, variant)
    if candidate["kind"] == "sb":
        variant = replace(corners_cfg, sb_flags=int(candidate["flags"]))
        return detect_sb(gray, cols, rows, variant)
    raise ValueError(
        f"未知候选类型 {candidate.get('kind')!r}(支持 classic/sb)。"
        f"候选只能由 build_method_candidates 构造。")


def _require_delivery_params(evaluate_report: dict, model_id: str) -> dict:
    """从 evaluate.json 取 r 归一化基准(delivery 的交付参数,与选型同源)。

    只需要 fx/fy/cx/cy 四个键(归一化半径公式);缺键当场中文报错,
    体例照 residual_structure 的取数防线。
    """
    delivery = (evaluate_report or {}).get("delivery") or {}
    models = delivery.get("models") or {}
    final = (models.get(model_id) or {}).get("final_params")
    if not isinstance(final, dict):
        raise AssertionError(  # noqa: TRY004  项目统一中文契约报错(体例照 artifacts.py),不按类型换 TypeError
            f"上游 evaluate.json 缺 delivery.models[{model_id}].final_params,"
            f"角点寻优的 r 归一化基准没有着落。请先跑 calib evaluate 再来,"
            f"或核对 evaluate 轮次是否串号。")
    for name in ("fx", "fy", "cx", "cy"):
        if name not in final:
            raise AssertionError(
                f"上游 evaluate.json 的 delivery final_params 缺键 {name},"
                f"r 归一化半径算不了。请核对 evaluate 轮次后重跑 calib evaluate。")
    return {name: float(final[name]) for name in ("fx", "fy", "cx", "cy")}


def _identity_label(family: str, level: float) -> bool:
    """该档是否恒等档(不改变图像):noise 0 / blur 0 / contrast 1.0。"""
    if family == "noise":
        return float(level) == 0.0
    if family == "blur":
        return float(level) <= 0.0
    return float(level) == 1.0


def _unit_label(family: str, level: float) -> str:
    """退化单元标签:str(浮点)保证 8.0/0.35/1.0 这类档位原样可读。"""
    return f"{family}_{float(level)}"


def build_units(cfg: AnalyzeConfig, profile: str = "quick") -> list[dict]:
    """枚举退化单元(quick = 单因素渐进档;full 追加一个最坏交叉单元)。

    单因素档按配置三族各自扫档;恒等档里只保留 noise_0.0 作校准档
    (blur_0.0 / contrast_1.0 与它逐位重复,跳过——"跳过重复的恒等组合")。
    full 档追加最坏噪 × 最坏糊的交叉单元(应用顺序 = 先模糊后加噪,照
    真实成像顺序:镜头模糊在前、传感器噪声在后;标签仍按 噪声+模糊 读)。
    返回 [{"label": …, "ops": (("blur", 4.0), ("noise", 64.0))}, …],
    ops 按应用顺序排列。
    """
    units: list[dict] = []
    for family, levels in (
            ("noise", cfg.noise_levels),
            ("blur", cfg.blur_levels),
            ("contrast", cfg.contrast_gains)):
        for level in levels:
            if family != "noise" and _identity_label(family, float(level)):
                continue  # 恒等档与校准档 noise_0.0 重复,不重复占单元
            units.append({
                "label": _unit_label(family, float(level)),
                "ops": ((family, float(level)),),
            })
    if profile == "full":
        max_noise = max(float(x) for x in cfg.noise_levels)
        max_blur = max(float(x) for x in cfg.blur_levels)
        if max_noise > 0.0 and max_blur > 0.0:
            units.append({
                "label": f"noise_{max_noise}+blur_{max_blur}",
                "ops": (("blur", max_blur), ("noise", max_noise)),
            })
    return units


def _stratified_frame_sample(frame_r_max: np.ndarray, n_frames: int,
                             seed: int) -> list[int]:
    """按各帧角点最大 r 分层抽帧(角点寻优关心边缘,外环帧必须有代表)。

    把帧按最大 r 升序排成 n 个等分层,每层用种子随机抽一帧——抽哪几帧
    种子说了算(同种子重跑抽到同一批),层覆盖小 r 到大 r。返回升序帧号。
    """
    order = np.argsort(np.asarray(frame_r_max, dtype=np.float64), kind="stable")
    strata = np.array_split(order, int(n_frames))
    rng = np.random.default_rng(int(seed))
    picked = [int(stratum[int(rng.integers(len(stratum)))])
              for stratum in strata if len(stratum)]
    return sorted(picked)


def _corner_opt_worker(task: tuple) -> dict:
    """单个(帧 × 方法 × 重复)的全部退化单元:读图一次,逐单元退化重检。

    子进程入口(顶层函数,可序列化):参数打包成小元组(全部可 pickle),
    返回字典(numpy 数组,可 pickle)。worker 内首行锁 OpenCV 线程(任务
    携带值)——并行子进程是全新进程,父进程的设置管不到(04-02 教训)。
    检测失败/未检出该单元记 NaN 并在 src 留 "not_found",是数据不是事故。
    """
    (image_path, uv_self, uv_prod, units, method, corners_cfg, cv2_threads,
     base_seed, rep, image_id, cols, rows) = task
    cv2.setNumThreads(int(cv2_threads))
    gray = imread_checked(Path(image_path))
    n_units = len(units)
    n_points = (uv_self if uv_self is not None else uv_prod).shape[0]
    disp_self = np.full((n_units, n_points, 2), np.nan, dtype=np.float64)
    disp_prod = np.full((n_units, n_points, 2), np.nan, dtype=np.float64)
    src = np.full(n_units, "", dtype="<U16")
    n_fail = 0
    for u, unit in enumerate(units):
        rng = np.random.default_rng(_seed_for(
            image_id, unit["label"], method["label"], rep, int(base_seed)))
        degraded = gray
        for family, level in unit["ops"]:
            degraded = apply_degradation(degraded, family, float(level), rng)
        result = detect_with(method, degraded, cols, rows, corners_cfg)
        if result is None:
            src[u] = "not_found"
            n_fail += 1
            continue
        src[u] = "ok"
        if uv_self is not None:
            disp_self[u] = result["uv"] - uv_self
        if uv_prod is not None:
            disp_prod[u] = result["uv"] - uv_prod
    return {"disp_self": disp_self, "disp_prodsb": disp_prod,
            "src": src, "n_fail": n_fail}


def _cell_stats(disp_block: np.ndarray, r_block: np.ndarray,
                edges: tuple[float, ...]) -> dict:
    """一格(口径 × 方法 × 单元)的位移统计 + r 分带切片。

    disp_block (K,P,2) 与 r_block (K,P) 为跨帧/重复堆叠的逐角点位移与
    归一化半径;NaN(未检出/非候选)行不计入。散布统计照 band_stats 同
    一把尺:RMS/中位/最大/95 分位用欧氏长度,u/v 分量均值单独给
    ("整体偏"不掺进散布)。全空时数值字段置 None,不冒充 0。
    """
    d = np.asarray(disp_block, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(r_block, dtype=np.float64).ravel()
    finite = np.isfinite(d).all(axis=1) & np.isfinite(r)
    entry: dict = {"n": int(np.count_nonzero(finite)),
                   "rms_px": None, "median_px": None, "max_px": None,
                   "p95_px": None, "mean_du": None, "mean_dv": None,
                   "bands": []}
    if not entry["n"]:
        return entry
    dv = d[finite]
    norms = np.linalg.norm(dv, axis=1)
    entry.update(
        rms_px=float(np.sqrt(np.mean(norms**2))),
        median_px=float(np.median(norms)),
        max_px=float(np.max(norms)),
        p95_px=float(np.percentile(norms, 95)),
        mean_du=float(np.mean(dv[:, 0])),
        mean_dv=float(np.mean(dv[:, 1])),
    )
    band_ids, _n_beyond = assign_bands(r[finite], edges)
    bands: list[dict] = []
    for i in range(len(edges) - 1):
        m = band_ids == i
        n = int(np.count_nonzero(m))
        row: dict = {"band_lo": float(edges[i]), "band_hi": float(edges[i + 1]),
                     "n": n, "rms_px": None}
        if n:
            row["rms_px"] = float(
                np.sqrt(np.mean(np.linalg.norm(dv[m], axis=1) ** 2)))
        bands.append(row)
    entry["bands"] = bands
    return entry


def _build_verdict_note(stats_self: dict, stats_prod: dict, units: list[str],
                        calibration_unit: str | None, threshold_px: float,
                        sigma_detector: float) -> str:
    """判读文案(中文,双口径都写,裁定①;结论由数据出,不预设)。"""
    compare_units = [u for u in units if u != calibration_unit]
    # 未显著劣化档位:自参照 RMS 低于阈值(10 × σ_detector)的单元
    surviving: dict[str, list[float]] = {}
    for method, per_unit in stats_self.items():
        vals = [per_unit[u]["rms_px"] for u in compare_units
                if per_unit.get(u, {}).get("rms_px") is not None]
        kept = [v for v in vals if v < threshold_px]
        if vals:
            surviving[method] = kept
    n_ok = max((len(v) for v in surviving.values()), default=0)
    lines = [
        (
            f"双口径判读(裁定①):主口径 _self 是各方法对自家干净检出的偏移"
            f"(量退化敏感度,主曲线);旁证 _prod_sb 是对生产 SB 干净检出的绝对"
            f"偏移(跨方法可比)。未显著劣化判据:自参照 RMS < {threshold_px:.4f}px"
            f"(= 10 × σ_detector {sigma_detector:.5f}px,阈值随本段落盘);"
            f"共 {len(compare_units)} 个非校准单元,最多 "
            f"{n_ok} 个未显著劣化。"
        ),
    ]
    scored = {m: (float(np.mean(v)) if v else float("inf"))
              for m, v in surviving.items()}
    if not scored:
        lines.append("所有方法在全部非校准单元都显著劣化:本批退化档位下没有"
                     "可用窗口,结论以曲线全貌为准,不单点下结论。")
        return " ".join(lines)
    best = min(scored, key=lambda m: (scored[m], m))
    prod = scored.get(PRODUCTION_LABEL)
    prod_txt = ("未进可比集" if prod is None or not np.isfinite(prod)
                else f"{prod:.4f}px")
    if best == PRODUCTION_LABEL:
        lines.append(
            f"数据显示:生产配置({PRODUCTION_LABEL},flags=0)在未显著劣化档位内"
            f"自参照平均误差最小({prod_txt}),当前角点配置有数据背书;"
            f"两列证据见 stats._self 与 stats._prod_sb。")
    else:
        lines.append(
            f"数据显示 {best} 档在未显著劣化单元更稳(自参照平均 "
            f"{scored[best]:.4f}px,生产配置 {prod_txt};旁证绝对偏移两列见 "
            f"stats._prod_sb),可考虑切换——只提示方向,不自动执行任何改动。")
    return " ".join(lines)


def run_corner_opt(
    model_id: str,
    evaluate_report: dict,
    corners_train: ObservationSet,
    image_records: list[dict],
    board,
    cfg: CalibrationConfig,
    sigma_detector_corner: float,
    progress=None,
) -> tuple[dict, dict]:
    """COR-02 主编排:帧抽样 → 阶段 A 参照检出 → 阶段 B 退化网格 → 统计落账。

    参数:
        model_id:               目标模型短名(evaluate.json 的 selection.selected),
                                r 归一化基准取它 delivery 的交付参数;
        evaluate_report:        evaluate.json 读回的完整报告;
        corners_train:          train 观测集(load_observation_set 装载,行序契约
                                的唯一承载;基线角点 = 生产 SB 干净检出的记录);
        image_records:          数据清单的照片记录(manifest["images"],提供
                                image_id → 文件路径;图像只读,退化图不落盘);
        board:                  棋盘规格(cols/rows 由它来);
        cfg:                    全量配置(寻优参数在 cfg.analyze,检测配置在
                                cfg.corners,并行上限 cfg.analyze.analyze_max_workers);
        sigma_detector_corner:  检测器级 σ(口径铁律唯一入口 noise.json 顶层键,
                                由 CLI 传入;未显著劣化阈值 = 10 × 它);
        progress:               进度回调(库层不依赖 typer,播报由 CLI 注入)。

    返回 (section, detail):section 进 analyze.json 的 corner_opt 段
    (schema analyze-corner-opt-v1,双口径 stats 并排);detail 是拍平的
    明细大数组(编号列 + 位移数组),交给 _corner_opt_detail 走 npz 旁车。
    """
    a_cfg = cfg.analyze
    profile = a_cfg.profile
    if profile not in ("quick", "full"):
        raise AssertionError(
            f"analyze.profile 只认 quick/full,拿到 {profile!r}。"
            f"请检查 config/calibration.yaml 的 analyze 段。")
    cols, rows = int(board.cols), int(board.rows)
    n_points = cols * rows
    edges = tuple(float(x) for x in a_cfg.radial_bin_edges)
    sigma = float(sigma_detector_corner)
    threshold_px = 10.0 * sigma

    # ---- 0) r 归一化基准(evaluate delivery 交付参数,与选型同源)----
    fx, fy, cx, cy = _require_delivery_params(evaluate_report, model_id).values()

    uv_all = np.asarray(corners_train.image_points, dtype=np.float64)
    n_all = uv_all.shape[0]
    if n_all == 0:
        raise AssertionError(
            "train 观测集为空,角点寻优没有可抽的帧。请先跑 calib corners。")
    id_to_path: dict[str, str] = {}
    for rec in image_records:
        if isinstance(rec, dict) and "image_id" in rec and "path" in rec:
            id_to_path[str(rec["image_id"])] = str(rec["path"])

    # ---- 1) 帧抽样:按各帧角点最大 r 分层(外环帧代表性)----
    r_flat, _r4 = radius_footprint(uv_all.reshape(-1, 2), fx, fy, cx, cy)
    frame_r_max = r_flat.reshape(n_all, n_points).max(axis=1)
    want = (a_cfg.quick_corner_opt_frames if profile == "quick"
            else a_cfg.full_corner_opt_frames)
    n_frames = int(min(max(1, want), n_all))
    sampling_seed = _seed_for("frame_sampling", f"{profile}|{n_frames}",
                              "corner_opt", 0, a_cfg.corner_opt_seed)
    sampled = _stratified_frame_sample(frame_r_max, n_frames, sampling_seed)
    frame_ids = [corners_train.image_ids[i] for i in sampled]
    missing = [fid for fid in frame_ids if fid not in id_to_path]
    if missing:
        raise AssertionError(
            f"数据清单里找不到这些照片的文件路径:{missing[:3]}"
            f"(共 {len(missing)} 张)。角点寻优要读原图施加退化,"
            f"请核对数据清单与角点轮次是否同一轮。")
    frame_sq_px = np.asarray(
        [estimate_sq_px(uv_all[i], cols, rows) for i in sampled], dtype=np.float64)

    # ---- 2) 单元与方法候选(逐帧按方格边长过滤 classic 阶梯)----
    units = build_units(a_cfg, profile)
    unit_labels = [u["label"] for u in units]
    calibration_unit = next(
        (lbl for lbl, u in zip(unit_labels, units)
         if len(u["ops"]) == 1 and u["ops"][0][0] == "noise"
         and float(u["ops"][0][1]) == 0.0),
        None)
    per_frame_candidates = [
        build_method_candidates(float(frame_sq_px[f]), a_cfg, profile)
        for f in range(n_frames)
    ]
    method_labels: list[str] = []
    for cands in per_frame_candidates:
        for cand in cands:
            if cand["label"] not in method_labels:
                method_labels.append(cand["label"])
    label_to_idx = {lbl: i for i, lbl in enumerate(method_labels)}
    n_methods = len(method_labels)
    n_units = len(units)
    n_reps = (a_cfg.corner_opt_n_reps if profile == "quick" else _FULL_REPS)
    if calibration_unit is None:
        raise AssertionError(
            "退化单元里没有校准档(noise 0 档):校准档用于逐位校验"
            "「同方法同图必得零位移」,缺了它无法自证对齐。请检查 analyze "
            "配置的 noise_levels 是否含 0.0。")

    # ---- 3) 阶段 A(参照):每 (抽样帧 × 候选方法) 对干净图检出一次 ----
    uv_clean = np.full((n_frames, n_methods, n_points, 2), np.nan)
    uv_prod = np.full((n_frames, n_points, 2), np.nan)
    for f in range(n_frames):
        gray = imread_checked(Path(id_to_path[frame_ids[f]]))
        for cand in per_frame_candidates[f]:
            m = label_to_idx[cand["label"]]
            res = detect_with(cand, gray, cols, rows, cfg.corners)
            if res is None:
                continue
            uv_clean[f, m] = res["uv"]
            if cand["label"] == PRODUCTION_LABEL:
                uv_prod[f] = res["uv"]  # 生产参照 = SB flags=0 干净检出(裁定①)
    n_prod_ok = int(np.isfinite(uv_prod[:, 0, 0]).sum())
    if n_prod_ok == 0:
        raise AssertionError(
            "阶段 A 参照检测全部失败:生产 SB(flags=0)对抽样帧的干净图"
            "一帧都没检出,双口径没有锚点。请检查图像文件与棋盘规格"
            "(board.cols/rows)是否与角点轮次一致。")

    # ---- 4) 阶段 B(退化网格):任务 = (帧 × 方法 × 重复),worker 内扫单元 ----
    tasks: list[tuple] = []
    task_keys: list[tuple[int, int, int]] = []
    for f in range(n_frames):
        for cand in per_frame_candidates[f]:
            m = label_to_idx[cand["label"]]
            uv_self = (uv_clean[f, m] if np.isfinite(uv_clean[f, m, 0, 0])
                       else None)
            uv_prod_f = (uv_prod[f] if np.isfinite(uv_prod[f, 0, 0]) else None)
            for rep in range(n_reps):
                tasks.append((
                    id_to_path[frame_ids[f]], uv_self, uv_prod_f, units, cand,
                    cfg.corners, int(cfg.cv2_threads),
                    int(a_cfg.corner_opt_seed), rep,
                    frame_ids[f], cols, rows))
                task_keys.append((f, m, rep))

    n_rows = n_frames * n_methods * n_reps * n_units
    disp_self = np.full((n_rows, n_points, 2), np.nan, dtype=np.float64)
    disp_prod = np.full((n_rows, n_points, 2), np.nan, dtype=np.float64)
    detected = np.zeros((n_rows, n_units), dtype=bool)
    src_rows = np.full((n_rows, n_units), "", dtype="<U16")
    n_failed_tasks = 0
    n_fail_units = 0

    def _row_of(f: int, m: int, rep: int, u: int) -> int:
        return ((f * n_methods + m) * n_reps + rep) * n_units + u

    emitted: set[tuple[int, int, int]] = set()
    for (f, m, rep), out in zip(
            task_keys, run_parallel(tasks, _corner_opt_worker,
                                    a_cfg.analyze_max_workers, progress)):
        emitted.add((f, m, rep))
        if isinstance(out, TaskFailed):
            # 单任务异常包信封转 NaN 占行 + 计数,任务流不断(T-05-16)
            n_failed_tasks += 1
            continue
        for u in range(n_units):
            row = _row_of(f, m, rep, u)
            disp_self[row] = out["disp_self"][u]
            disp_prod[row] = out["disp_prodsb"][u]
            detected[row, u] = bool(out["src"][u] == "ok")
            src_rows[row, u] = str(out["src"][u])
        n_fail_units += int(out["n_fail"])
    n_missing_tasks = len(task_keys) - len(emitted)
    if n_missing_tasks:
        raise AssertionError(
            f"退化网格任务流不完整:{n_missing_tasks} 个任务没有交回结果"
            f"(共 {len(task_keys)} 个)。run_parallel 的产出数应与任务数一致,"
            f"出现缺口语味着并行后端异常,请重跑。")

    # ---- 5) 统计:双口径并排(_self 主口径 / _prod_sb 旁证)+ r 分带切片 ----
    r_per_point = r_flat.reshape(n_all, n_points)[sampled]
    stats: dict[str, dict] = {"_self": {}, "_prod_sb": {}}
    for ref_key, block in (("_self", disp_self), ("_prod_sb", disp_prod)):
        for m, method in enumerate(method_labels):
            per_unit: dict[str, dict] = {}
            for u, unit_label in enumerate(unit_labels):
                sel_rows = np.asarray([
                    _row_of(f, m, rep, u)
                    for f in range(n_frames) for rep in range(n_reps)])
                disp_block = block[sel_rows]
                # r_block 形状对齐 (n_frames*n_reps, P):每帧重复 n_reps 份
                r_block = np.repeat(r_per_point, n_reps, axis=0)
                per_unit[unit_label] = _cell_stats(disp_block, r_block, edges)
            stats[ref_key][method] = per_unit

    # 跨方法绝对偏移旁证:各方法干净检出 vs 生产 SB 干净检出(方法间互比)
    cross_clean: dict[str, dict] = {}
    for m, method in enumerate(method_labels):
        d = uv_clean[:, m] - uv_prod[:, 0]
        finite = np.isfinite(d).all(axis=2)
        entry: dict = {"n_frames": int(np.count_nonzero(finite)),
                       "rms_px": None, "median_px": None, "max_px": None}
        if entry["n_frames"]:
            dv = d[finite]
            norms = np.linalg.norm(dv, axis=1)
            entry.update(rms_px=float(np.sqrt(np.mean(norms**2))),
                         median_px=float(np.median(norms)),
                         max_px=float(np.max(norms)))
        cross_clean[method] = entry
        # 干净偏移不做 r 分带(样本就是帧数),只报整体三个统计量

    # ---- 6) 判读文案(中文,双口径都给,不预设结论)----
    verdict_note = _build_verdict_note(
        stats["_self"], stats["_prod_sb"], unit_labels, calibration_unit,
        threshold_px, sigma)

    section = {
        "schema": SCHEMA_CORNER_OPT,
        "model_id": model_id,
        "profile": profile,
        "n_frames": n_frames,
        "n_units": n_units,
        "n_methods": n_methods,
        "n_reps": int(n_reps),
        "units": unit_labels,
        "unit_ops": {u["label"]: [[fam, float(lv)] for fam, lv in u["ops"]]
                     for u in units},
        "methods": method_labels,
        "production_method": PRODUCTION_LABEL,
        "calibration_unit": calibration_unit,
        "sigma_detector_corner": sigma,
        "threshold_px": float(threshold_px),
        "stats": stats,
        "cross_clean": cross_clean,
        "verdict_note": verdict_note,
        "sb_accuracy_verdict": SB_ACCURACY_VERDICT,
        "n_failed_tasks": int(n_failed_tasks),
        "n_not_found_units": int(n_fail_units),
        "base_seed": int(a_cfg.corner_opt_seed),
        "sampling": {
            "method": "按帧最大归一化半径 r 分层抽样(每层种子随机抽一帧,"
                      "覆盖从小 r 到大 r;角点寻优关心边缘)",
            "n_requested": int(want),
            "seed": int(sampling_seed),
        },
        "sources": {
            "frames": "corners_train_sb.npz 的 image_ids(经 load_observation_set 装载)",
            "images": "input_data 训练划分原图(只读;退化图不落盘,种子可复现)",
            "reference_prod_sb": ("生产 SB(flags=0)干净检出,本实验现场重检"
                                  "(与角点轮次同一 cv2 构建,保证同版本可比)"),
            "sigma": "noise.json 顶层键 sigma_detector_corner(CLI 传入)",
            "r_basis": f"evaluate delivery final_params(模型 {model_id})",
        },
    }
    detail = {
        "frame_idx": np.asarray(
            [f for f in range(n_frames) for _ in method_labels
             for _ in range(n_reps) for _ in range(n_units)], dtype=np.int64),
        "method_idx": np.asarray(
            [m for _ in range(n_frames) for m in range(n_methods)
             for _ in range(n_reps) for _ in range(n_units)], dtype=np.int64),
        "rep_idx": np.asarray(
            [rep for _ in range(n_frames) for _ in range(n_methods)
             for rep in range(n_reps) for _ in range(n_units)], dtype=np.int64),
        "unit_idx": np.asarray(
            [u for _ in range(n_frames) for _ in range(n_methods)
             for _ in range(n_reps) for u in range(n_units)], dtype=np.int64),
        "disp_self": disp_self,
        "disp_prodsb": disp_prod,
        "detected": detected,
        "src": src_rows,
        "uv_clean": uv_clean.reshape(n_frames * n_methods, n_points, 2),
        "uv_clean_frame_idx": np.asarray(
            [f for f in range(n_frames) for _ in range(n_methods)],
            dtype=np.int64),
        "uv_clean_method_idx": np.asarray(
            [m for _ in range(n_frames) for m in range(n_methods)],
            dtype=np.int64),
        "r_per_point": r_per_point,
        "frame_r_max": np.asarray(frame_r_max[sampled], dtype=np.float64),
        "frame_sq_px": frame_sq_px,
        "image_ids": np.asarray(frame_ids),
        "unit_labels": np.asarray(unit_labels),
        "method_labels": np.asarray(method_labels),
    }
    return section, detail
