"""scipy 精化后端:OpenCV 初值 → 稀疏最小二乘联合精化(02-04 Task 2)。

定位与研究结论(02-RESEARCH §Pattern 2):θ = [内参 | 各视图 6 个位姿
参数] 装成**一个**向量,交给 scipy.optimize.least_squares 的 trf(信赖域
反射法,白话:每步只允许参数在当前解附近一个"可信范围"里动)+ lsmr
(解稀疏线性子问题的迭代法)+ **解析稀疏雅可比**(residuals.py 组装,
数值来源锁在模型层,D-08)。不做内参/外参分块交替优化——那等于自研
优化器,收敛性质更差(研究 Don't Hand-Roll)。

实用收敛判定(研究 Pitfall 3,白话):鲁棒核(soft_l1 这类"把大残差
权重压低"的损失)配 trf+lsmr 时,优化器常在**代价已经走平、残差已经
到噪声底**的时候报 status=0(意思是"我没法再改进了",不是"失败")。
一刀切把 status=0 判失败会把好结果扔掉;所以判收敛看三条:
1. status>0:优化器自己确认停机条件,直接算收敛;
2. status=0 且代价尾部(最后约 10 次评估)相对改进 < 1e-6(走平了)
   且 RMS 没有比 linear 核参照差超 5%(鲁棒核没把解压歪):实用收敛;
3. status=-1:优化器报错,绝不判收敛。
判定依据(status、尾部改进、代价)全部进 solver_meta,不许静默。

失败契约与 OpenCV 后端同构:异常或未收敛都返回 make_failure 结构
(结构键全在、数值字段一律 None、中文 failure_note),不向上抛异常。

D-09:本文件只认 CameraModel 协议,不 import 任何具体模型;
D-10:数值差分只在测试里当基准,本文件的雅可比一律解析式(residuals.py)。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from calib.io.config import ScipyConfig
from calib.models.base import CameraModel
from calib.solve.observation import ObservationSet
from calib.solve.opencv import _solve_generic_pnp
from calib.solve.residuals import (
    make_jacobian_fn,
    make_residual_fn,
    make_sparsity,
    pack_theta,
    unpack_theta,
)
from calib.solve.result import CalibrationResult, compute_result_metrics, make_failure

# 代价尾部窗口:看最后 _TAIL_WINDOW 次残差评估(约 10 次)的相对改进,
# 足够滤掉"被信赖域拒绝的尝试步"造成的单点回弹
_TAIL_WINDOW = 10
# 尾部改进的"走平"判定线:相对改进小于它就认为代价不再下降
_PLATEAU_TOL = 1e-6
# 鲁棒核参照比较线:精化 RMS 不得超过 linear 核 RMS 的 1.05 倍
_REF_TOLERANCE = 1.05
# f_scale 自动档的退化兜底:MAD 算出的尺度小于它(残差近零)时按 1 像素
# 当量——把尺度缩成 0 等于把所有残差都当离群,是反着来的
_F_SCALE_FLOOR = 1e-12


def is_practically_converged(status: int,
                             tail_improvement: float,
                             rms_final: float | None,
                             rms_linear_ref: float | None = None) -> bool:
    """实用收敛判定(纯函数,判定依据显式传入,真值表由测试锁定)。

    status:           scipy 返回的状态码(>0 停机条件满足;0 到达评估上限
                      或无法改进;-1 报错);
    tail_improvement: 代价尾部窗口的相对改进(走平 ≈ 0);
    rms_final:        精化后的整体 RMS(像素,回代尺);
    rms_linear_ref:   linear 核同初值精化的 RMS 参照,None 表示没有参照。
    """
    if status > 0:
        return True
    if status <= -1:
        return False
    # status == 0:要"代价走平"且"没比 linear 参照差超 5%"才算
    if tail_improvement >= _PLATEAU_TOL:
        return False
    if rms_linear_ref is None or rms_final is None:
        return True
    return rms_final <= _REF_TOLERANCE * float(rms_linear_ref)


def _auto_f_scale(residuals: np.ndarray) -> float:
    """f_scale 自动档:3 × 1.4826 × MAD(MAD = 中位数绝对偏差)。

    白话:先量"残差的中位波动"(中位数不受个别坏点影响),乘 1.4826 换算
    成高斯噪声下的标准差,再放大 3 倍——残差大过噪声波动 3 倍才开始被
    鲁棒核压权重。残差近零(合成无噪声数据)时 MAD=0,兜底返回 1.0 像素
    当量,不让尺度退化成 0。
    """
    arr = np.asarray(residuals, dtype=np.float64).ravel()
    mad = float(np.median(np.abs(arr - np.median(arr))))
    scale = 3.0 * 1.4826 * mad
    if scale < _F_SCALE_FLOOR:
        return 1.0
    return scale


def _tail_improvement(cost_trace: list[float]) -> float:
    """代价尾部的相对改进:取"运行最小值包络"(只降不升)后,看最后
    _TAIL_WINDOW 次评估相对前一段降了多少比例。

    为什么要包络:trf 的尝试步会被信赖域**拒绝**(那次残差白算、代价
    回弹),原始曲线锯齿状;包络后单调不增,尾部改进才真实反映
    "优化器还能不能压低代价"。评估次数不足 2 次时无从谈趋势,按 0
    (走平)处理。
    """
    if len(cost_trace) < 2:
        return 0.0
    envelope = np.minimum.accumulate(np.asarray(cost_trace, dtype=np.float64))
    tail_n = min(_TAIL_WINDOW, envelope.shape[0] - 1)
    prev = float(envelope[-1 - tail_n])
    last = float(envelope[-1])
    return (prev - last) / max(abs(prev), 1e-30)


def _initial_theta(obs: ObservationSet, model: CameraModel,
                   init: CalibrationResult | None):
    """整理精化起点:给定初值按参数名逐名取;没给就走通用初始化兜底。

    返回 (theta0, init_from 记号):
    - init 给了:必须是**已收敛**结果且模型一致(精化起点对着别的模型/
      没算出来的结果,继续算只会产出垃圾,这里直接中文报错);
    - init 没给:_solve_generic_pnp(default_params 组 K + 逐视图位姿估计)
      内部兜底,记号 internal_generic。
    """
    if init is None:
        generic = _solve_generic_pnp(obs, model)
        params = np.array([generic.params[name] for name in model.param_names],
                          dtype=np.float64)
        return pack_theta(model, params, generic.rvecs, generic.tvecs), \
            "internal_generic"

    if not init.converged:
        raise ValueError(
            f"给 scipy 精化的初值必须已收敛(模型 {model.model_id}),"
            f"实际拿到的是未收敛结果:{init.failure_note}。"
            f"请先用 OpenCV 后端拿到收敛初值,或改走通用初始化(不传 init)。"
        )
    if init.model_id != model.model_id:
        raise ValueError(
            f"初值与精化目标的模型不一致:初值来自 {init.model_id},"
            f"要精化的是 {model.model_id}。参数布局不同,不能混用。"
        )
    missing = [name for name in model.param_names if name not in init.params]
    if missing:
        raise ValueError(
            f"初值参数字典缺 {model.model_id} 的参数名:{missing}。"
            f"请按参数名给全再精化。"
        )
    params = np.array([init.params[name] for name in model.param_names],
                      dtype=np.float64)
    return pack_theta(model, params, init.rvecs, init.tvecs), \
        f"{init.model_id}__{init.solver}"


def solve_with_scipy(obs: ObservationSet,
                     model: CameraModel,
                     cfg: ScipyConfig,
                     init: CalibrationResult | None = None,
                     loss: str = "linear",
                     rms_linear_ref: float | None = None) -> CalibrationResult:
    """scipy 精化入口:观测集 + 任一模型 + (可选)OpenCV 初值 → 统一结果。

    主线:θ 装箱 → trf + lsmr + 解析稀疏雅可比联合优化 → 同一把尺回代
    (compute_result_metrics,float64)→ 实用收敛判定。鲁棒核由 loss 指定
    (linear / soft_l1 / huber…),f_scale 按 cfg(自动档基于初值残差)。
    任何异常都收敛进 make_failure 失败契约,不向上抛。
    """
    n_points = int(obs.object_points.shape[0])
    n_views = len(obs.image_ids)
    try:
        theta0, init_from = _initial_theta(obs, model, init)

        cost_trace: list[float] = []
        residual_fn = make_residual_fn(model, obs)

        def traced_residual(theta: np.ndarray) -> np.ndarray:
            r = residual_fn(theta)
            cost_trace.append(0.5 * float(r @ r))  # 与 scipy 的代价定义一致
            return r

        if cfg.f_scale == "auto":
            f_scale = _auto_f_scale(traced_residual(theta0))
        else:
            f_scale = float(cfg.f_scale)

        sol = least_squares(
            traced_residual, theta0,
            jac=make_jacobian_fn(model, obs),
            jac_sparsity=make_sparsity(model, n_views, n_points),
            method=cfg.method, tr_solver=cfg.tr_solver,
            loss=loss, f_scale=f_scale, x_scale="jac",
            max_nfev=cfg.max_nfev, ftol=cfg.ftol, xtol=cfg.xtol, gtol=cfg.gtol,
        )

        params, rvecs, tvecs = unpack_theta(model, sol.x)
        rms_overall, per_view_rms, _residuals = compute_result_metrics(
            obs, params, rvecs, tvecs, model)

        tail = _tail_improvement(cost_trace)
        meta = {
            "path": "scipy_refine",
            "method": cfg.method, "tr_solver": cfg.tr_solver, "x_scale": "jac",
            "loss": loss, "f_scale": float(f_scale),
            "status": int(sol.status), "nfev": int(sol.nfev), "njev": int(sol.njev),
            "final_cost": float(sol.cost), "tail_improvement": float(tail),
            "n_cost_evals": len(cost_trace),
            "rms_linear_ref": (None if rms_linear_ref is None
                               else float(rms_linear_ref)),
        }
        converged = is_practically_converged(
            status=int(sol.status), tail_improvement=tail,
            rms_final=rms_overall, rms_linear_ref=rms_linear_ref)
        if not converged:
            return make_failure(
                model_id=model.model_id, solver="scipy",
                note=(f"{model.model_id} 模型 scipy 精化未收敛:"
                      f"status={sol.status}(0=到达评估上限或无法改进),"
                      f"代价尾部相对改进 {tail:.3e},回代 RMS {rms_overall:.4f} 像素;"
                      f"判定依据见 solver_meta,数据或初值可能有病。"),
                image_ids=obs.image_ids, image_size=obs.image_size,
                n_points_per_view=n_points, solver_meta=meta)

        return CalibrationResult(
            converged=True, failure_note=None,
            model_id=model.model_id, solver="scipy", solver_meta=meta,
            params={name: float(v) for name, v in zip(model.param_names, params)},
            rvecs=rvecs, tvecs=tvecs,
            rms_overall=rms_overall, per_view_rms=per_view_rms,
            image_ids=obs.image_ids, image_size=obs.image_size,
            n_views=n_views, n_points_per_view=n_points,
            init_from=init_from, lineage={},
        )
    except (ValueError, ArithmeticError, RuntimeError) as e:
        # 数值路径的失败(病态输入让 scipy 报错、线性代数炸掉等)进失败契约,
        # 不中断实验网格;形状/参数错误(AssertionError/TypeError/KeyError)是
        # 代码契约被违反,故意让它大声抛出,不吞。np.linalg.LinAlgError 是
        # ValueError 的子类,被这里一并覆盖。
        return make_failure(
            model_id=model.model_id, solver="scipy",
            note=f"{model.model_id} 模型 scipy 精化未跑成:{type(e).__name__}——{e}",
            image_ids=obs.image_ids, image_size=obs.image_size,
            n_points_per_view=n_points,
            solver_meta={"path": "scipy_refine", "loss": loss})
