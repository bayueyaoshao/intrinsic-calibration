"""ANA-03 test 集相机坐标系三分量分解:test 误差到底落在哪个方向。

白话背景:标定解出内参后,"一张没参与标定的照片误差多大"由评估步回答;
本模块回答更细的一层——这些像素误差**对应棋盘在相机坐标系里朝哪个方向
挪**。把逐角点像素残差经投影对相机坐标点的导数(雅可比,即"相机坐标点
动一点、像素动多少"的斜率)做最小范数反解,得到三维位移 dX=(dX, dY, dZ)。

这是阶段五唯一新建数学的模块,但零新导数:投影与雅可比只经 models/ 接口
(D-06 唯一数学源),位姿只经 solve_pose_only(只估位姿小求解,内参当
尺子固定不动);本模块自己写的只有线性代数(反解与特征值)。

判读纪律(RESEARCH Pitfall 7,"Z 分量解释翻车"坑):单视角对深度(Z)
方向天然弱可观测——dZ 的数值被 1/小特征值放大,"Z 大"可能只是数学必然。
所以原始量级必须与可观测性归一(z 分数 = dX ÷ 理论 σ)并排呈现,每帧
报深度方向放大倍数;AᵀA(AᵀA 是"数据对这个方向有多少约束"的信息矩阵)
条件数超过 analyze.xyz_cond_max 的帧判病态,dX 不具备解释力,按 NaN
占行单独计数,绝不冒充数值。

test 集参加过选型(轻微软奥偏差,evaluate.json 的 agreement.note 有先例
文案),本段成绩带轻微乐观偏差,notes 固定声明。
"""

from __future__ import annotations

import numpy as np

from calib.models.base import transform_to_camera
from calib.solve.scoring import solve_pose_only

# 段模式版本:三分量段的结构约定(加键不改键算兼容,删键必须升版本)
SCHEMA_TEST_XYZ = "analyze-test-xyz-v1"

# notes 固定文案(两条,原文落盘——判读纪律与口径声明不许每轮重写)
_NOTE_DEPTH = (
    "Z 分量大 ≠ 误差主要在深度:单视角深度可观测性差,数值被放大倍数抬高,"
    "判读看 z 分数列与放大倍数"
)
_NOTE_OPTIMISTIC = (
    "test 集参加过选型,本段成绩带轻微乐观偏差"
    "(口径现象,与 evaluate agreement.note 同源声明)"
)


class _IllConditionedFrame(ValueError):
    """病态帧(AᵀA 条件数超限):dX 数值不可解释,调用方按 NaN 占行。

    是 ValueError 的子类——plan 契约是"cond 超阈值抛 ValueError 由编排层
    占行";子类化只是让编排层能把"病态帧"与"位姿解不出"两类失败分开
    计数(n_cond_excluded / n_solve_failed),捕获的异常类型不变。
    """


def _require_final_params(evaluate_report: dict, model_id: str) -> dict:
    """从 evaluate.json 读回的报告取交付参数字典,缺键当场中文报错。"""
    delivery = (evaluate_report or {}).get("delivery") or {}
    models = delivery.get("models") or {}
    entry = models.get(model_id)
    if not isinstance(entry, dict) or not isinstance(entry.get("final_params"), dict):
        raise AssertionError(  # noqa: TRY004  项目统一中文契约报错,不按类型换 TypeError
            f"evaluate.json 的 delivery.models[{model_id}].final_params 缺失或不是"
            f"字典。三分量分解要用交付内参当尺子,缺键多半是串了轮次或产物不完整,"
            f"请先重跑 calib evaluate 再来。"
        )
    return entry["final_params"]


def _frame_xyz_core(resid: np.ndarray, A: np.ndarray,
                    cond_max: float) -> dict:
    """反解核心(唯一数学位置):残差 + 堆叠雅可比 → 三分量位移与可观测性。

    resid: (2N,) 逐角点像素残差(投影 − 实测,拍平);
    A:     (2N,3) 堆叠点雅可比(第 2i/2i+1 行 = 第 i 点投影对相机坐标的
           (2,3) 导数块,来自 model.project_point_jacobian);
    返回:dX (3,) / sigma_theory (3,) / zscores (3,) / eigvals (3,) /
          cond / depth_amplification。

    dX = (AᵀA)⁻¹Aᵀ resid 是最小范数反解(残差能被棋盘整体位移解释的最
    小位移);sigma_theory = sqrt(diag(inv(AᵀA))) 是单位像素噪声对应的
    dX 各分量理论标准差;特征值升序(eigvalsh 契约),最小特征值方向 ≈
    深度方向,其放大倍数 = 1/sqrt(最小特征值)。cond 超过 cond_max 判
    病态,抛 _IllConditionedFrame(ValueError 子类),绝不返回半成品。
    """
    ata = A.T @ A
    cond = float(np.linalg.cond(ata))
    if not np.isfinite(cond) or cond > float(cond_max):
        raise _IllConditionedFrame(
            f"病态帧:AᵀA 条件数 {cond:.3e} 超过上限 {cond_max:.3e}"
            f"(或非有限)。该帧的深度方向几乎无约束,反解出的位移被放大得"
            f"不可解释,按 NaN 占行处理,不冒充数值(RESEARCH Pitfall 7)。"
        )
    d_x = np.linalg.solve(ata, A.T @ resid)
    sigma_theory = np.sqrt(np.diag(np.linalg.inv(ata)))
    eigvals = np.linalg.eigvalsh(ata)
    return {
        "dX": d_x,
        "sigma_theory": sigma_theory,
        "zscores": d_x / sigma_theory,
        "eigvals": eigvals,
        "cond": cond,
        "depth_amplification": 1.0 / np.sqrt(eigvals[0]),
    }


def frame_xyz(uv_view: np.ndarray, object_points: np.ndarray,
              params: np.ndarray, model, cond_max: float = 1.0e4) -> dict:
    """单张 test 照片:像素残差 → 相机坐标系位移三分量 + 可观测性。

    步骤(全部既有接口,零手写导数):solve_pose_only 只估位姿(内参固定)
    → transform_to_camera 把物点变换到相机坐标系 → model.project 回代出
    投影角点 → 残差 = 投影 − 实测(拍平)→ model.project_point_jacobian
    给 (N,2,3) 点雅可比,堆叠成 (2N,3) 作反解矩阵 A → _frame_xyz_core
    反解。返回字典含 "dX" / "sigma_theory" / "zscores" / "eigvals" /
    "cond" / "depth_amplification" / "rms"(回代 RMS,像素)。

    dX 的单位 = 物点单位(棋盘方格边长的同一单位,本板为 mm);
    cond 超过 cond_max 抛 ValueError(病态帧的处理方式,见模块头)。
    """
    uv = np.asarray(uv_view, dtype=np.float64)
    rvec, tvec, rms = solve_pose_only(uv, object_points, params, model)
    pts_cam, _pose_block = transform_to_camera(object_points, rvec, tvec)
    uv_pred = model.project(pts_cam, params)
    resid = (uv_pred - uv).ravel()
    j_pt = model.project_point_jacobian(pts_cam, params)
    A = j_pt.reshape(2 * uv.shape[0], 3)
    out = _frame_xyz_core(resid, A, cond_max)
    out["rms"] = float(rms)
    return out


def _median_max(values: np.ndarray) -> dict:
    """一组有限值的 {median, max};全空给 None(不冒充 0)。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"median": None, "max": None}
    return {"median": float(np.median(finite)), "max": float(np.max(finite))}


def run_test_xyz(model_id: str, evaluate_report: dict, corners_test,
                 board, cfg) -> tuple[dict, dict]:
    """编排 ANA-03:test 集逐帧三分量反解(失败帧 NaN 占行,不中断)。

    返回 (section, detail):

    - section 进 analyze.json 的 test_xyz 段(schema analyze-test-xyz-v1):
      逐方向 |dX| 与 z 分数的 median/max 表、可观测性汇总(cond 中位/最大、
      深度放大倍数中位)、帧计数(n_frames/n_valid/n_cond_excluded/
      n_solve_failed)、两条固定 notes、sources;
    - detail 是逐帧明细大数组(dX / sigma_theory / eigvals / cond /
      image_ids),失败帧数值行整行 NaN,交给 _test_xyz_detail 走 npz 旁车。

    参数:
        model_id:        目标模型短名(evaluate.json 的 selection.selected);
        evaluate_report: evaluate.json 读回的完整报告(dX 反解用 delivery
                         的 final_params 当尺子);
        corners_test:    test 观测集(经 load_observation_set 装载,行序
                         契约的唯一承载);
        board:           棋盘规格(物点网格由此构造);
        cfg:             全量配置(病态判定线读 cfg.analyze.xyz_cond_max)。
    """
    from calib.models import create_model
    from calib.solve.baseline import build_object_points

    model = create_model(model_id)
    object_points = build_object_points(board)
    final_params = _require_final_params(evaluate_report, model_id)
    params = np.array(
        [float(final_params[name]) for name in model.param_names],
        dtype=np.float64,
    )
    cond_max = float(cfg.analyze.xyz_cond_max)

    uv_test = np.asarray(corners_test.image_points, dtype=np.float64)
    n_frames = uv_test.shape[0]
    d_x = np.full((n_frames, 3), np.nan, dtype=np.float64)
    sigma_theory = np.full((n_frames, 3), np.nan, dtype=np.float64)
    eigvals = np.full((n_frames, 3), np.nan, dtype=np.float64)
    cond_rows = np.full(n_frames, np.nan, dtype=np.float64)
    rms_rows = np.full(n_frames, np.nan, dtype=np.float64)
    n_cond_excluded = 0
    n_solve_failed = 0
    for i in range(n_frames):
        try:
            out = frame_xyz(uv_test[i], object_points, params, model,
                            cond_max=cond_max)
        except _IllConditionedFrame:
            n_cond_excluded += 1   # 病态帧:数值不可解释,NaN 占行单独计数
            continue
        except (ValueError, AssertionError):
            n_solve_failed += 1    # 位姿解不出/角点太少:同样 NaN 占行
            continue
        d_x[i] = out["dX"]
        sigma_theory[i] = out["sigma_theory"]
        eigvals[i] = out["eigvals"]
        cond_rows[i] = out["cond"]
        rms_rows[i] = out["rms"]

    valid = np.isfinite(d_x).all(axis=1)
    n_valid = int(np.count_nonzero(valid))

    # ---- 聚合:三分量 |dX| 与 |z 分数| 的 median/max、可观测性汇总 ----
    per_direction: dict = {"dX_abs_mm": {}, "abs_zscore": {}}
    for j, comp in enumerate(("x", "y", "z")):
        per_direction["dX_abs_mm"][comp] = _median_max(np.abs(d_x[valid, j]))
        per_direction["abs_zscore"][comp] = _median_max(
            np.abs(d_x[valid, j] / sigma_theory[valid, j]))
    observability = {
        "cond_median": (float(np.median(cond_rows[valid]))
                        if n_valid else None),
        "cond_max": (float(np.max(cond_rows[valid])) if n_valid else None),
        "depth_amplification_median": (
            float(np.median(1.0 / np.sqrt(eigvals[valid, 0])))
            if n_valid else None),
    }
    rms_block = _median_max(rms_rows)

    evaluate_round = str(
        (evaluate_report.get("lineage") or {}).get("round_id") or "未知轮次")
    section = {
        "schema": SCHEMA_TEST_XYZ,
        "model_id": model_id,
        "unit_note": "dX 单位 = 物点单位(棋盘方格边长的同一单位,本板为 mm)",
        "n_frames": int(n_frames),
        "n_valid": n_valid,
        "n_cond_excluded": int(n_cond_excluded),
        "n_solve_failed": int(n_solve_failed),
        "per_direction": per_direction,
        "observability": observability,
        "rms_px": rms_block,
        "notes": [_NOTE_DEPTH, _NOTE_OPTIMISTIC],
        "sources": {
            "evaluate_round": evaluate_round,
            "params_basis": f"evaluate delivery final_params(模型 {model_id})",
            "corners_test": "corners_test_sb.npz(经 load_observation_set 装载,"
                            "由 analyze 命令体传入)",
            "jacobian": "models.project_point_jacobian(D-06 唯一导数来源,"
                        "堆叠 (N,2,3)→(2N,3) 作反解矩阵)",
            "pose_solver": "solve.scoring.solve_pose_only(只估位姿,内参固定)",
            "cond_threshold": f"config analyze.xyz_cond_max = {cond_max:g}",
        },
    }
    detail = {
        "dX": d_x,
        "sigma_theory": sigma_theory,
        "eigvals": eigvals,
        "cond": cond_rows,
        "image_ids": np.asarray(corners_test.image_ids),
    }
    return section, detail
