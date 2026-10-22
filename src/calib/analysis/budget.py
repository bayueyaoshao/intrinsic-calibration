"""ANA-01 误差预算——实际误差 vs CRLB 理论下界的对账(纯消费,零求解)。

白话背景:误差预算表回答"每个内参实际晃动多大、距离这批数据天生能到
的极限(CRLB,克拉美-罗下界,即"只考虑角点检测噪声时参数最好能定到
多准"的理论下界)还有多远"。全部原料在阶段④评估的落盘产物里已经
算好,本模块只做三件事:复核(独立重算与 evaluate.json 已落账的统计
量逐位对账,不一致说明上游被篡改或串了轮次)、组装(逐参数预算行 +
像素当量换算)、判读(实测低于下界当场硬报错,绝不静默通过)。

唯一权威口径(2026-10-20 定案,findings 结论 4)= 检测器级 σ × 内参块
下界 × 雅可比推到逐角点重投影位移。σ 只认调用方从 noise.json 顶层键
sigma_detector_corner 读入的值(本模块不自己读文件、不挑口径);时域
口径(连拍整帧共模偏移,≈0.30px)会把下界虚高约 35 倍,绝不允许混用。

正式判定在像素空间:逐角点实测散布必须全部不低于理论下界,任何一行
低于、或任一统计量比值跌破 evaluate 已落账的硬告警线,都抛中文
AssertionError(与 evaluate 的 crlb_alert 同一条判定线,不重写第二套)。
参数空间表格只作参考:k1–k3(及 p1/p2)的单独下界被参数间补偿空间
撑大,利用率比值置 None 并随表附 crlb_caliber_note 口径声明。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from calib.evaluation.experiments import _px_dispersion

# 段模式版本:预算段的结构约定(加键不改键算兼容,删键必须升版本)
SCHEMA_BUDGET = "analyze-budget-v1"

# 像素空间三统计量的键名(与 evaluate 侧 crlb_alert.per_param 完全同名,
# 复核就是按这三把尺逐位对账)
_STAT_KEYS = ("重投影位移中位", "重投影位移95分位", "重投影位移最大")

# 利用率比值(实测 σ / 理论下界)只对数据定得住的四个参数给出:
# k1–k3 与 p1/p2 的单独下界被参数间补偿空间撑大(不是噪声大),
# 比值无意义,置 None(findings 结论 4 的判定口径)
_UTILIZATION_PARAMS = ("fx", "fy", "cx", "cy")

# 参数空间参考表的口径声明(照抄 evaluation/experiments.py 的现成文案,
# 不重写第二份——两处必须一字不差,改口径要两边同步)
CRLB_CALIBER_NOTE = (
    "crlb_se_mean 是参数空间参考表(σ=检测器级,内参块下界,"
    "位姿理想已知):fx/fy/cx/cy 等数据定得住的参数对照有效;"
    "k1..k6 等强相关参数的单独下界被参数间补偿空间撑大,不是"
    "噪声大,单独比较无意义。正式的「逼近 CRLB」判定在像素空间,"
    "见 crlb_alert(逐角点重投影位移,2026-10-20 定案)。"
)


def _require(mapping, key: str, what: str):
    """从上游报告取键,缺失当场中文报错(体例照 evaluate 的上游取数防线)。"""
    if not isinstance(mapping, dict) or key not in mapping:
        raise AssertionError(
            f"上游报告缺键 {key}({what})。误差预算按契约消费 evaluate.json "
            f"的落盘段,缺键多半是产物与当前代码不是同一套契约,或串了轮次,"
            f"请先重跑 calib evaluate 再来。"
        )
    return mapping[key]


def _load_slices_arrays(model_id: str, slices_npz_path: Path | str) -> dict:
    """读回 slices.npz 里目标模型的前缀键(allow_pickle=False 写死)。

    quick 档产物只含当选模型(Pitfall 6):缺键当场中文报错,提示核对
    model_id 与 evaluate.json 的 selection.selected,不静默返回残缺数据。
    """
    path = Path(slices_npz_path)
    if not path.is_file():
        raise AssertionError(
            f"上游 slices.npz 不在:{path}。分组实验底账是误差预算的取数源,"
            f"请先跑 calib evaluate(或核对 run_<时间戳> 目录名是否传对)。"
        )
    want = {}
    with np.load(path, allow_pickle=False) as data:
        for short in ("group_uv", "group_bound_px", "group_params", "param_names"):
            key = f"{model_id}__{short}"
            if key not in data.files:
                raise AssertionError(
                    f"上游 slices.npz({path})缺键 {key}。quick 档产物只含"
                    f"当选模型,请核对 model_id 与 evaluate.json 的 "
                    f"selection.selected 是否一致;full 档重跑 evaluate 后"
                    f"才有多模型明细。"
                )
            want[short] = data[key]
    return want


def _check_sigma_detector(sigma_detector: float) -> float:
    """校验 σ 是有限正数(口径铁律:σ 是下界的乘数,坏值污染全部结论)。"""
    try:
        value = float(sigma_detector)
    except (TypeError, ValueError) as e:
        raise AssertionError(
            f"sigma_detector 不是数:{sigma_detector!r}。只认从 noise.json "
            f"顶层键 sigma_detector_corner 读入的检测器级 σ 标量。"
        ) from e
    if not math.isfinite(value) or value <= 0:
        raise AssertionError(
            f"sigma_detector(检测器级 σ)必须是正的有限数,拿到 "
            f"{sigma_detector!r}。σ 是 CRLB 下界的乘数,坏值会让"
            f"「实测低于下界」的判定整体失真;口径铁律见 findings 结论 "
            f"4——只认 noise.json 的 sigma_detector_corner(检测器级口径),"
            f"时域口径(≈0.30px)会把下界虚高约 35 倍,绝不允许混用。"
        )
    return value


def run_budget(
    model_id: str,
    evaluate_report: dict,
    slices_npz_path: Path | str,
    sigma_detector: float,
    model,
    object_points: np.ndarray,
    ref_rvec: np.ndarray,
    ref_tvec: np.ndarray,
) -> tuple[dict, dict]:
    """组装 ANA-01 误差预算段:复核 + 预算行 + 像素当量 + 逐行铁律硬报错。

    纯消费、零求解:实测与下界全部取自 evaluate 落盘产物,本函数只复核、
    组装与判读。返回 (section, detail):

    - section 进 analyze.json 的 budget 段(schema analyze-budget-v1);
    - detail 是逐行明细大数组(meas_rows / bound_rows / ratios_rows /
      group_params / param_names),纯数组,交给 _budget_detail 走 npz 旁车。

    参数:
        model_id:        目标模型短名(从 evaluate.json 的 selection.selected
                         读,quick 档只有当选模型在 slices.npz 里);
        evaluate_report: evaluate.json 读回的完整报告(dict);
        slices_npz_path: experiments/slices.npz 路径(分组实验底账);
        sigma_detector:  检测器级 σ(调用方从 noise.json 的
                         sigma_detector_corner 读入,本模块不自己读文件);
        model:           目标模型的 CameraModel 实例(像素当量换算用);
        object_points:   棋盘物点网格 (N,3)(像素当量换算的投影对象);
        ref_rvec/ref_tvec: 参考位姿(与判行表同口径,findings 结论 1 同源;
                         由调用方从分组实验的参照位姿传入并随产物声明)。
    """
    sigma = _check_sigma_detector(sigma_detector)

    arrays = _load_slices_arrays(model_id, slices_npz_path)
    group_uv = np.asarray(arrays["group_uv"], dtype=np.float64)
    group_bound_px = np.asarray(arrays["group_bound_px"], dtype=np.float64)
    group_params = np.asarray(arrays["group_params"], dtype=np.float64)
    param_names = [str(x) for x in np.asarray(arrays["param_names"]).ravel()]

    # 形状契约当场验(slices.npz 由 evaluate 侧落盘,形状漂移说明串了版本)
    if group_uv.ndim != 4 or group_uv.shape[-1] != 2:
        raise AssertionError(
            f"slices.npz 的 {model_id}__group_uv 形状不合:期望 "
            f"(组数, 视图数, 角点数, 2),实际 {group_uv.shape}。"
            f"请核对产物是否出自当前版本的 calib evaluate。"
        )
    n_groups, n_views, n_points, _two = group_uv.shape
    if n_groups < 2:
        raise AssertionError(
            f"分组实验组数 = {n_groups},少于 2 组谈不了组间散布,"
            f"误差预算无从算起。请用全量分组产物(当前数据 8 组)。"
        )
    if group_bound_px.shape != (n_groups, 2 * n_views * n_points):
        raise AssertionError(
            f"slices.npz 的 {model_id}__group_bound_px 形状与 group_uv 对不上:"
            f"期望 {(n_groups, 2 * n_views * n_points)},实际 "
            f"{group_bound_px.shape}。逐行下界必须与逐角点重投影一一对应,"
            f"请核对产物完整性。"
        )
    if group_params.shape != (n_groups, len(param_names)):
        raise AssertionError(
            f"slices.npz 的 {model_id}__group_params 形状不合:期望 "
            f"{(n_groups, len(param_names))}(按 param_names 布局),实际 "
            f"{group_params.shape}。请核对产物完整性。"
        )
    if list(model.param_names) != param_names:
        raise AssertionError(
            f"传入模型的参数布局 {list(model.param_names)} 与 slices.npz 的 "
            f"param_names {param_names} 对不上:模型与产物必须同源"
            f"(例如拿 m2 的模型去读 m3 的底账),请核对 model_id。"
        )

    # ---- 复核(独立重算,对 evaluate.json 已落账的三统计量逐位对账)----
    # 实测 = 各组解重投影同一物理点的组间标准差(与 evaluate 侧同式);
    # 下界 = 各组匹配口径逐行位移下界的均值。
    meas_rows = np.std(group_uv, axis=0, ddof=1).ravel()
    bound_rows = group_bound_px.mean(axis=0).ravel()
    stat_fns = {
        "重投影位移中位": np.median,
        "重投影位移95分位": lambda a: np.percentile(a, 95),
        "重投影位移最大": np.max,
    }
    ratios = {}
    evaluate_ratios = {}
    consistent = True
    slices_section = _require(
        evaluate_report.get("slices", {}), "models", "evaluate.json 的 slices 段"
    )
    model_section = _require(slices_section, model_id, f"slices.models[{model_id}]")
    crlb_alert = _require(model_section, "crlb_alert", "slices 段的 CRLB 告警块")
    per_param = _require(crlb_alert, "per_param", "crlb_alert 的逐统计量落账")
    for name, fn in stat_fns.items():
        meas_stat = float(fn(meas_rows))
        bound_stat = float(fn(bound_rows))
        ours = meas_stat / bound_stat if bound_stat != 0.0 else math.inf
        ratios[name] = ours
        entry = per_param.get(name)
        if not isinstance(entry, dict) or entry.get("ratio") is None:
            raise AssertionError(
                f"evaluate.json 的 crlb_alert.per_param 缺 {name} 的落账比值:"
                f"误差预算的复核必须与 evaluate 同一条判定线,缺账说明产物"
                f"不完整或串了版本,请重跑 calib evaluate。"
            )
        theirs = float(entry["ratio"])
        evaluate_ratios[name] = theirs
        if not math.isclose(ours, theirs, rel_tol=1e-9):
            consistent = False
    if not consistent:
        detail_txt = ";".join(
            f"{name}:重算 {ratios[name]:.6g} vs 落账 {evaluate_ratios[name]:.6g}"
            for name in _STAT_KEYS
        )
        raise AssertionError(
            f"误差预算复核发现口径漂移:从 slices.npz 独立重算的统计量比值"
            f"与 evaluate.json 已落账的 crlb_alert 比值对不上({detail_txt})。"
            f"两处必须出自同一轮同一口径,不一致说明上游被篡改或串了轮次,"
            f"禁止静默通过;请核对 evaluate 轮次后重跑。"
        )

    # ---- 逐行铁律硬报错(与 evaluate 的 crlb_alert 同一条判定线)----
    lines = _require(crlb_alert, "lines", "crlb_alert 的告警线")
    hard_line = float(_require(lines, "hard", "crlb_alert.lines 的硬告警线"))
    n_pointwise = int((meas_rows < bound_rows).sum())
    ratios_rows = meas_rows / np.maximum(bound_rows, 1e-12)
    min_ratio = float(ratios_rows.min())
    below_hard = [name for name in _STAT_KEYS if ratios[name] < hard_line]
    if n_pointwise > 0 or below_hard:
        problems = []
        if n_pointwise > 0:
            problems.append(
                f"{n_pointwise}/{meas_rows.size} 个(视图,角点)的实测重投影"
                f"散布低于理论下界"
            )
        if below_hard:
            problems.append(
                "统计量 " + "、".join(below_hard) + " 的实测/下界比值低于硬告警线"
            )
        raise AssertionError(
            "误差预算硬报错(ANA-01 铁律):" + ";".join(problems)
            + f"(硬告警线 {hard_line:.4f},逐行最小比值 {min_ratio:.3f})。"
            f"实测低于理论下界物理上不可能 = 口径错配或上游数据被污染,"
            f"禁止静默通过。排查顺序:① σ 来源只认 noise.json 的 "
            f"sigma_detector_corner(检测器级口径;时域口径会把下界虚高"
            f"约 35 倍);② slices.npz 与 evaluate.json 是否同一轮产物;"
            f"③ 上游产物有没有被手工改动。"
        )

    # ---- 逐参数预算行(顺序按 param_names;像素当量与判行表同口径)----
    delivery_models = _require(
        evaluate_report.get("delivery", {}), "models", "evaluate.json 的 delivery 段"
    )
    final_params = _require(
        delivery_models[model_id],
        "final_params",
        f"delivery.models[{model_id}] 的交付参数",
    )
    base_vec = np.array(
        [float(final_params[name]) for name in param_names], dtype=np.float64
    )
    bootstrap_models = _require(
        evaluate_report.get("bootstrap", {}), "models", "evaluate.json 的 bootstrap 段"
    )
    param_std = _require(
        bootstrap_models[model_id],
        "param_std",
        f"bootstrap.models[{model_id}] 的重采样标准差",
    )
    separation = _require(
        evaluate_report, "source_separation", "evaluate.json 的来源分离段"
    )
    multistart = _require(
        evaluate_report.get("optimizer", {}), "multistart", "optimizer 段的多起点块"
    )
    multistart_models = _require(multistart, "models", "multistart 的逐模型块")
    multistart_disp_px = float(
        _require(
            multistart_models[model_id],
            "param_disp_px",
            f"multistart.models[{model_id}] 的参数位移当量",
        )
    )
    crlb_se_mean = _require(
        model_section, "crlb_se_mean", f"slices.models[{model_id}] 的参数空间下界"
    )

    rows: list[dict] = []
    for i, name in enumerate(param_names):
        empirical_std = float(np.std(group_params[:, i], ddof=1))
        # 像素当量:单参数 ±empirical_std 扰动 → 参考位姿下逐角点投影位移
        # 的最大值(与判行表稳定判据同一把尺,findings 结论 1 同源)
        unit = np.zeros_like(base_vec)
        unit[i] = 1.0
        empirical_std_px = _px_dispersion(
            [base_vec, base_vec + empirical_std * unit],
            model,
            object_points,
            np.asarray(ref_rvec, dtype=np.float64),
            np.asarray(ref_tvec, dtype=np.float64),
        )
        crlb_se = float(crlb_se_mean[name])
        utilization = (
            empirical_std / crlb_se if name in _UTILIZATION_PARAMS else None
        )
        var_pose = float(
            _require(
                _require(separation, model_id, "source_separation 的逐模型块"),
                name,
                f"source_separation[{model_id}] 的逐参数块",
            )["var_pose_sampling"]
        )
        if not math.isfinite(var_pose) or var_pose < 0:
            raise AssertionError(
                f"source_separation[{model_id}].{name}.var_pose_sampling 不合法:"
                f"{var_pose!r}(方差必须是非负有限数),请重跑 calib evaluate。"
            )
        rows.append(
            {
                "param": name,
                "empirical_std": empirical_std,
                "empirical_std_px": float(empirical_std_px),
                "bootstrap_std": float(param_std[name]),
                "pose_sampling_std": math.sqrt(var_pose),
                "multistart_disp_px": multistart_disp_px,
                "crlb_se": crlb_se,
                "utilization_ratio": utilization,
            }
        )

    lineage_doc = _require(evaluate_report, "lineage", "evaluate.json 的 lineage 段")
    section = {
        "schema": SCHEMA_BUDGET,
        "model_id": model_id,
        "sigma_detector_corner": sigma,
        "lines": {
            "hard": hard_line,
            "soft": float(_require(lines, "soft", "crlb_alert.lines 的软提示线")),
        },
        "recheck": {
            "ratios": ratios,
            "evaluate_ratios": evaluate_ratios,
            "consistent": consistent,
        },
        "n_pointwise_violations": n_pointwise,
        "min_ratio_pointwise": min_ratio,
        "verdict": "pass",
        "rows": rows,
        "crlb_caliber_note": CRLB_CALIBER_NOTE,
        "sources": {
            "evaluate_round": _require(lineage_doc, "round_id", "lineage 的 round_id"),
            "n_groups": int(n_groups),
            "sigma_source": "noise.json:sigma_detector_corner",
        },
    }
    detail = {
        "meas_rows": meas_rows,
        "bound_rows": bound_rows,
        "ratios_rows": ratios_rows,
        "group_params": group_params,
        "param_names": np.asarray(param_names),
    }
    return section, detail
