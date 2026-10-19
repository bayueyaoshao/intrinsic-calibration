"""优化器对比编排(D-13 优化器先行):16 组合网格 + 每模型推荐配置 + 多起点一致性。

白话:模型选型之前先回答"用什么优化配置去跑"——4 种损失核(linear 与
三种鲁棒核:soft_l1 / huber / cauchy,鲁棒核 = 把离群残差的权重压低、
不让个别坏点拖偏整体的损失函数)× 4 个模型 = 16 个组合,在全量 640 视图
上各解一次,每模型挑出"收敛且重投影误差不劣"的代表配置。下游的交叉
验证 / bootstrap / 留一姿态 / 全量重训全部统一用这份推荐配置(D-13),
避免"选型用默认配置、之后发现更优配置又全部返工"。

初值契约(D-14 锁死,Pitfall 8):16 个组合**共用同一份 OpenCV 解**当
初值——每模型从上一轮标定产物 mX__opencv 的落盘文件读回,不做内存私递
(文件是唯一真相源)。初值不同的话,"哪个核更好"就分不清是核的功劳
还是起点的运气,对比作废。

失败契约:单个组合内部出任何意外(求解器数值失败已在 solve_with_scipy
内转成失败契约,这里兜编排层的意外)都转失败契约占行,绝不中断其余
组合——一个组合病了不该拖死整批网格。status=0 未过"实用收敛"不是
异常也不是失败:照常占一行、converged=False(历史轮 m1-softl1 类,
这本身就是 SOL-02 的结论材料,Pitfall 5)。

可复现承诺:多起点扰动的随机性只从全局种子(config 的 seed)按
(模型, 起点序号)确定性派生——同配置重跑逐位一致。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from math import radians
from pathlib import Path

import cv2
import numpy as np

from calib.evaluation.artifacts import SCHEMA_VERSION
from calib.evaluation.parallel import run_parallel
from calib.evaluation.splitting import _seed_for
from calib.io.config import BoardSpec, CalibrationConfig, EvaluateConfig, ScipyConfig
from calib.models import create_model
from calib.models.base import transform_to_camera
from calib.solve.matrix import _load_init_result, combo_dir_name
from calib.solve.observation import load_observation_set
from calib.solve.opencv import solve_with_opencv
from calib.solve.result import CalibrationResult, make_failure
from calib.solve.scipy_solver import solve_with_scipy

# 推荐规则的"数值并列"线:两个组合的 RMS 相对差小于它(千分之一)视为
# 打平——此时取更简的核(更少非线性加权,结论更可辩护),不为噪声级
# 的差异引入复杂度
_RECOMMEND_TIE_REL = 1e-3

# 核的"简度"优先级:数字越小越简(linear 无非线性加权最简,cauchy 压制
# 最狠最复杂)。D-14 四核之外的名单进不来(配置层白名单已拦),兜底 99
_LOSS_SIMPLICITY = {"linear": 0, "soft_l1": 1, "huber": 2, "cauchy": 3}


def _solve_combo(task: tuple) -> CalibrationResult:
    """单组合求解(worker):解包小元组 → 统一求解路径 solve_with_scipy。

    worker 必须是顶层函数(能被序列化传给并行子进程);入参小元组:
    (模型ID, 损失核, 观测集, scipy配置, 初值结果, linear参照RMS)。
    求解器内部的数值失败已在 solve_with_scipy 里转成失败契约返回;这里
    再兜一层"意料之外的程序性异常"(比如并行框架传参被改坏):转成失败
    契约而不是任它抛出——run_parallel 的消费循环一旦在 worker 里碰上
    异常,整个任务流会在这里断掉,排在后面的组合会被静默吞掉(串行回退
    时尤甚),必须让异常到不了生成器。
    """
    model_id, loss, obs, scipy_cfg, init, rms_linear_ref = task
    try:
        model = create_model(model_id)
        return solve_with_scipy(obs, model, scipy_cfg, init=init, loss=loss,
                                rms_linear_ref=rms_linear_ref)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return make_failure(
            model_id=model_id, solver="scipy",
            note=(f"优化器组合 {model_id} × {loss} 核 未跑成:"
                  f"{type(exc).__name__}——{exc}"),
            image_ids=obs.image_ids, image_size=obs.image_size,
            n_points_per_view=int(obs.object_points.shape[0]),
            solver_meta={"path": "orchestration_fallback", "loss": loss})


def _combo_row(model_id: str, loss: str, result: CalibrationResult,
               init_rms: float) -> dict:
    """组装一行组合汇总(键照 solve/matrix.py 的 _combo_row,另加两列)。

    init_rms:本组合的共同初值(OpenCV 解)的整体 RMS——"精化前"的起点,
    没有它就说不清精化带来了多少提升(Pitfall 8);
    f_scale_actual:鲁棒核实际生效的分界尺度(auto 档按初值残差浮动,
    是口径不是 bug,记下来才可查——Pitfall 4)。
    """
    return {
        "name": combo_dir_name(model_id, "scipy", loss),
        "model": model_id,
        "solver": "scipy",
        "loss": loss,
        "n_views": result.n_views,
        "rms_overall": (float(result.rms_overall)
                        if result.rms_overall is not None else None),
        "converged": bool(result.converged),
        "failure_note": result.failure_note,
        # scipy 的停机状态码(>0 正式收敛 / 0 到评估上限或无法改进 / -1 报错)
        "solver_status": result.solver_meta.get("status"),
        "init_rms": init_rms,
        "f_scale_actual": (float(result.solver_meta["f_scale"])
                           if result.solver_meta.get("f_scale") is not None
                           else None),
    }


def _failure_combo_row(model_id: str, loss: str, exc: Exception,
                       n_views: int, init_rms: float) -> dict:
    """编排层意外的失败契约占行:键一个不少、数值字段 None、中文说明。"""
    return {
        "name": combo_dir_name(model_id, "scipy", loss),
        "model": model_id,
        "solver": "scipy",
        "loss": loss,
        "n_views": n_views,
        "rms_overall": None,
        "converged": False,
        "failure_note": (f"优化器组合({model_id} × {loss} 核)未跑成:"
                         f"{type(exc).__name__}——{exc}"),
        "solver_status": None,
        "init_rms": init_rms,
        "f_scale_actual": None,
    }


def _consume_combo(iterator, model_id: str, loss: str, n_views: int,
                   init_rms: float, rows_by_key: dict) -> None:
    """取回一个组合的结果并落行:异常转失败契约占行,绝不中断网格。"""
    try:
        result = next(iterator)
    except StopIteration:  # pragma: no cover - 任务数与键数恒相等,防御性兜底
        raise AssertionError(
            f"组合({model_id} × {loss})的结果缺失:并行产出数与任务数对不上,"
            f"这是编排层的代码契约被破坏,请检查 run_optimizer_grid。")
    except Exception as e:  # noqa: BLE001  # 故意宽接:单组合任何异常都转失败契约占行
        rows_by_key[(model_id, loss)] = _failure_combo_row(
            model_id, loss, e, n_views, init_rms)
        return
    rows_by_key[(model_id, loss)] = _combo_row(model_id, loss, result,
                                               init_rms)


def run_optimizer_grid(corners_npz_path: Path | str,
                       manifest_path: Path | str,
                       board: BoardSpec,
                       scipy_cfg: ScipyConfig,
                       eval_cfg: EvaluateConfig,
                       calibration_run_dir: Path | str,
                       exclude_images: tuple[str, ...] = (),
                       progress=None) -> dict:
    """优化器对比网格(D-13/D-14):全量视图 × 4 核 × 参选模型,逐组合落行。

    参数:
        corners_npz_path:   sb 路线角点文件(640 全帧正式口径,与阶段二
                            标定同一份观测集);
        manifest_path:      数据清单(装载时按行序契约对齐);
        board:              棋盘规格(装配核对:角点数须与棋盘一致);
        scipy_cfg:          scipy 精化参数(f_scale 默认 auto,D-14);
        eval_cfg:           评估配置(参选模型、核名单、并行上限从这读);
        calibration_run_dir:上一轮标定的运行目录——每模型的共同初值从
                            mX__opencv 子目录读回(D-14 锁死);
        exclude_images:     人工剔除清单(逻辑剔除,原数据不动);
        progress:           每完成一个组合回调一次(收汇总行;CLI 用它播报)。

    返回 {"schema_version", "recommended_solver", "combos", "multistart",
    "meta"} 的报告字典(multistart 由 run_multistart 产出后另行合并;
    落盘交 save_optimizer_report,本函数不写盘)。

    初值契约(D-14 锁死,白话):16 个组合从**同一份** OpenCV 解出发——
    初值不同,则"哪个核收敛得更好"分不清是核的功劳还是起点的运气
    (Pitfall 8);每行都带 init_rms(起点 RMS),提升才可归因到核。
    逐模型先跑 linear 再跑鲁棒核:linear 的 RMS 是鲁棒核"实用收敛"
    判定的参照(rms_linear_ref),漏传会让收敛判定失真。
    """
    obs = load_observation_set(corners_npz_path, manifest_path, board,
                               split="train", exclude_images=exclude_images)
    n_views = len(obs.image_ids)
    expected_points = board.cols * board.rows
    if obs.object_points.shape[0] != expected_points:
        raise ValueError(
            f"观测集角点数({obs.object_points.shape[0]})与棋盘规格"
            f"({board.cols}x{board.rows} = {expected_points} 个内角点)对不上:"
            f"多半是 board.yaml 与角点文件不是同一套配置,请核对后重跑。")

    models = tuple(eval_cfg.models)
    losses = tuple(eval_cfg.optimizer_losses)
    if "linear" not in losses:
        raise ValueError(
            f"优化器核名单里没有 linear:{losses}。linear 是主线核、必须包含"
            f"——它是「其他核有没有更好」的归一参照(D-14),"
            f"请检查 evaluate.optimizer_losses。")

    # ——共同初值(D-14):每模型从落盘产物读回同一份 OpenCV 解——
    inits: dict[str, CalibrationResult] = {}
    init_rms: dict[str, float] = {}
    for model_id in models:
        init_dir = Path(calibration_run_dir) / combo_dir_name(model_id, "opencv")
        init = _load_init_result(init_dir, model_id)
        if init is None or init.rms_overall is None:
            raise ValueError(
                f"初值产物不可用:{init_dir}(result.json 缺失、未收敛或"
                f"模型对不上)。D-14 锁死 16 组合从同一份 OpenCV 解出发,"
                f"初值读不回来就只能重跑 calib calibrate,不许换别的初值"
                f"凑数——初值不一致的对比表不可信(Pitfall 8)。")
        inits[model_id] = init
        init_rms[model_id] = float(init.rms_overall)

    max_workers = eval_cfg.evaluate_max_workers
    rows_by_key: dict[tuple[str, str], dict] = {}
    linear_ref: dict[str, float] = {}

    # ——第一阶段:每模型先跑 linear 核(鲁棒核的实用收敛参照从这来)——
    linear_tasks = [(m, "linear", obs, scipy_cfg, inits[m], None)
                    for m in models]
    linear_iter = run_parallel(linear_tasks, _solve_combo, max_workers)
    for model_id in models:
        _consume_combo(linear_iter, model_id, "linear", n_views,
                       init_rms[model_id], rows_by_key)
        row = rows_by_key[(model_id, "linear")]
        if progress:
            progress(row)
        if row["converged"] and row["rms_overall"] is not None:
            linear_ref[model_id] = float(row["rms_overall"])

    # ——第二阶段:鲁棒核按配置名单逐个跑,必带 rms_linear_ref(同模型
    #   linear 的回代 RMS;漏传会让 is_practically_converged 判定失真)——
    robust_keys: list[tuple[str, str]] = []
    robust_tasks: list[tuple] = []
    for m in models:
        for loss in losses:
            if loss == "linear":
                continue
            robust_keys.append((m, loss))
            robust_tasks.append((m, loss, obs, scipy_cfg, inits[m],
                                 linear_ref.get(m)))
    robust_iter = run_parallel(robust_tasks, _solve_combo, max_workers)
    for model_id, loss in robust_keys:
        _consume_combo(robust_iter, model_id, loss, n_views,
                       init_rms[model_id], rows_by_key)
        if progress:
            progress(rows_by_key[(model_id, loss)])

    # 汇总行按"模型分组、核按配置名单顺序"排(major = 模型),读表的人
    # 看到的是"同一模型的四个核并排相邻"
    combos = [rows_by_key[(m, loss)] for m in models for loss in losses]

    # 返回值直接组齐 optimizer.json 的 8 键契约(created / failure_note
    # 也由这里给:组合级失败照常占行,整轮失败只在调用方走
    # make_failure_optimizer 的失败契约,正常返回即 success 轮)
    return {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "recommended_solver": recommend_optimization_config(combos, eval_cfg),
        "combos": combos,
        "multistart": None,   # 由 run_multistart 产出后合并进同一份报告
        "failure_note": None,
        "meta": {
            "n_combos": len(combos),
            "n_failed": sum(1 for row in combos if not row["converged"]),
            "n_views": n_views,
            "models": list(models),
            "losses": list(losses),
            "calibration_run": Path(calibration_run_dir).name,
            "init_rms": init_rms,
            "rms_linear_ref": linear_ref,
            "recommend_tie_rel": _RECOMMEND_TIE_REL,
        },
    }


def recommend_optimization_config(combos: list[dict],
                                  eval_cfg: EvaluateConfig) -> dict:
    """每模型的推荐优化配置(D-13):收敛组合里 RMS 最小者;并列取更简核。

    规则:只在 converged=True 的组合里挑;整体 RMS 最小者胜;多个组合
    与最小者的相对差小于 _RECOMMEND_TIE_REL(千分之一,数值并列)时,
    取核更简的那个(linear > soft_l1 > huber > cauchy——更少非线性加权,
    结论更可辩护,不为噪声级的差异引入复杂度)。

    返回 {"mX": {"loss": ..., "f_scale_px": ...}}(linear 主线无
    f_scale_px 键——它没有非线性加权的分界尺度);某模型没有任何收敛
    组合时记 {"loss": None, "note": 中文说明},不静默。
    """
    recommended: dict[str, dict] = {}
    for model_id in eval_cfg.models:
        candidates = [row for row in combos
                      if row["model"] == model_id and row["converged"]
                      and row["rms_overall"] is not None]
        if not candidates:
            recommended[model_id] = {
                "loss": None,
                "note": (f"{model_id} 没有任何收敛的组合,给不出推荐配置;"
                         f"失败与未收敛原因见 combos 逐行 failure_note,"
                         f"请先排查数据或初值再重跑。"),
            }
            continue
        best = min(row["rms_overall"] for row in candidates)
        if best > 0:
            # 数值并列:与最小者的相对差 < 千分之一(RMS 恒非负,差值小
            # 于 best×1e-3 即相对差达标)
            tied = [row for row in candidates
                    if (row["rms_overall"] - best) < _RECOMMEND_TIE_REL * best]
        else:
            # 最小者是 0(理论完美拟合)时相对差无从谈起,只认精确相等
            tied = [row for row in candidates if row["rms_overall"] == best]
        winner = min(tied,
                     key=lambda row: _LOSS_SIMPLICITY.get(row["loss"], 99))
        entry = {"loss": winner["loss"]}
        if winner["loss"] != "linear" and winner.get("f_scale_actual") is not None:
            entry["f_scale_px"] = float(winner["f_scale_actual"])
        recommended[model_id] = entry
    return recommended


# ---------------------------------------------------------------------------
# 多起点扰动一致性检验(D-15)
# ---------------------------------------------------------------------------

# 内参四参数的名单(扰动幅度按参数名分组:内参 ±2%、畸变 ±10%)
_INTRINSIC_NAMES = ("fx", "fy", "cx", "cy")


def _multistart_n_for(model, eval_cfg: EvaluateConfig) -> int:
    """该模型的多起点次数:鱼眼模型(m4,初值敏感、易卡坏解)加倍。"""
    adapter = model.opencv_adapter()
    is_fisheye = adapter is not None and adapter.kind == "fisheye"
    return (eval_cfg.multistart_n_fisheye if is_fisheye
            else eval_cfg.multistart_n)


def _perturbed_init(init: CalibrationResult, eval_cfg: EvaluateConfig,
                    rng: np.random.Generator) -> CalibrationResult:
    """从 OpenCV 解构造一个受控扰动的起点(D-15 扰动幅度表,全部 config 化)。

    幅度为经验起点(04-RESEARCH 假设 A2:±2% 内参 / ±10% 畸变 / 约 1° 旋转 /
    ±2% 平移——只在"足以离开初值邻域、又不至于落入荒谬解"的宽窗内敏感),
    要调幅度改 evaluate 段配置,不改代码。

    - 内参(fx/fy/cx/cy)与畸变(k/p 全族):相对乘性扰动,符号与幅度
      均匀抽样(畸变系数数值本身小,相对幅度要大才起作用);
    - 旋转:单位随机轴 × 固定转角(旋转向量直接加,不手写旋转矩阵);
    - 平移:按平移向量的范数做相对扰动。
    """
    params = dict(init.params)
    for name, value in init.params.items():
        if name in _INTRINSIC_NAMES:
            rel = eval_cfg.multistart_intr_rel
        elif name.startswith(("k", "p")):   # 畸变族按名字判断(k1..k6 / p1 p2)
            rel = eval_cfg.multistart_dist_rel
        else:
            raise AssertionError(
                f"多起点扰动不认识参数名 {name!r}(模型 {init.model_id})。"
                f"扰动幅度按名字分内参/畸变两组,新参数进来要回 "
                f"_perturbed_init 补分组规则,不许静默跳过。")
        sign = float(rng.choice([-1.0, 1.0]))
        factor = 1.0 + sign * float(rng.uniform(0.0, rel))
        params[name] = float(value) * factor

    rvecs = np.array(init.rvecs, dtype=np.float64, copy=True)
    angle = radians(eval_cfg.multistart_rot_deg)
    for i in range(rvecs.shape[0]):
        axis = rng.normal(size=3)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:   # 三维标准正态抽到零向量的概率可忽略,兜底防呆
            axis, norm = np.array([1.0, 0.0, 0.0]), 1.0
        rvecs[i] = rvecs[i] + (axis / norm) * angle

    tvecs = np.array(init.tvecs, dtype=np.float64, copy=True)
    for i in range(tvecs.shape[0]):
        sign = float(rng.choice([-1.0, 1.0]))
        factor = 1.0 + sign * float(rng.uniform(0.0, eval_cfg.multistart_trans_rel))
        tvecs[i] = tvecs[i] * factor

    return replace(init, params=params, rvecs=rvecs, tvecs=tvecs)


def _solve_multistart_start(task: tuple) -> tuple:
    """单起点求解(worker):解包小元组 → scipy 精化,返回 (起点号, 种子, 结果)。

    用 linear 主线核:多起点检验的是"初值敏感性"(换个起点还收敛到同一
    解吗),核间谁好谁坏已在 16 组合网格里比过,这里不混变量。程序性异常
    同样在 worker 层转失败契约(_solve_combo 同理):不让异常穿过
    run_parallel 的生成器,后续起点才不会静默丢失。
    """
    model_id, start_index, seed, perturbed, obs, scipy_cfg = task
    try:
        model = create_model(model_id)
        result = solve_with_scipy(obs, model, scipy_cfg, init=perturbed,
                                  loss="linear")
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        result = make_failure(
            model_id=model_id, solver="scipy",
            note=(f"多起点 {model_id} 起点号 {start_index} 未跑成:"
                  f"{type(exc).__name__}——{exc}"),
            image_ids=obs.image_ids, image_size=obs.image_size,
            n_points_per_view=int(obs.object_points.shape[0]),
            solver_meta={"path": "orchestration_fallback", "loss": "linear"})
    return start_index, seed, result


def _param_dispersion_px(model, obs, ref_rvec: np.ndarray, ref_tvec: np.ndarray,
                         param_dicts: list[dict]) -> float:
    """参数一致性换算成"像素位移当量":同一参考位姿下,每对起点的两组
    内参分别投影棋盘物点网格,逐角点位移的最大值(纯几何,像素说话——
    参数数值差多少没有直观意义,投影到画面上挪了几个像素才有)。
    """
    pts_cam, _pose_deriv = transform_to_camera(obs.object_points, ref_rvec,
                                               ref_tvec)
    projections = []
    for params in param_dicts:
        arr = np.array([params[name] for name in model.param_names],
                       dtype=np.float64)
        projections.append(model.project(pts_cam, arr))
    worst = 0.0
    for a in range(len(projections)):
        for b in range(a + 1, len(projections)):
            disp = np.linalg.norm(projections[a] - projections[b], axis=1)
            worst = max(worst, float(disp.max()))
    return worst


def _judge_multistart_consistency(model, obs, ref_rvec: np.ndarray,
                                  ref_tvec: np.ndarray, starts: list[dict],
                                  eval_cfg: EvaluateConfig) -> dict:
    """双容差判定(纯函数,缺一不可):RMS 极差 + 参数位移像素当量。

    只看 RMS 一致是不够的:参数不同但 RMS 相同 = 可辨识性问题本身
    (Anti-Patterns 第 6 条),两条容差同时满足才判"收敛到同一解"。
    判定式与阈值随结论落盘,不凭空下结论(体例照 fisher 的 verdict_note)。
    参考位姿取基线解的某一视图位姿(ref_rvec/ref_tvec 各是单个 (3,) 向量)——
    参数一致性必须在同一个位姿下比投影,位姿一换投影就不可比。
    """
    converged = [s for s in starts
                 if s["converged"] and s["rms_overall"] is not None]
    if len(converged) < 2:
        failed = [s for s in starts if not s["converged"]]
        return {
            "verdict": "inconsistent",
            "rms_spread_px": None,
            "param_disp_px": None,
            "note": (f"有效起点不足 2 个(收敛 {len(converged)}/"
                     f"{len(starts)},失败 {len(failed)}),无从谈一致性;"
                     f"失败原因见 bad_solutions。"),
        }
    rms_values = [float(s["rms_overall"]) for s in converged]
    rms_spread = max(rms_values) - min(rms_values)
    param_disp = _param_dispersion_px(
        model, obs, ref_rvec, ref_tvec,
        [s["params"] for s in converged])
    rms_tol = eval_cfg.multistart_rms_tol_px
    disp_tol = eval_cfg.multistart_param_tol_px
    consistent = rms_spread < rms_tol and param_disp < disp_tol
    verdict = "consistent" if consistent else "inconsistent"
    reasons = []
    if rms_spread >= rms_tol:
        reasons.append(f"RMS 极差 {rms_spread:.4f} px ≥ 容差 {rms_tol} px")
    if param_disp >= disp_tol:
        reasons.append(f"参数位移当量 {param_disp:.4f} px ≥ 容差 {disp_tol} px")
    note = (f"硬门槛:RMS 极差 < {rms_tol} px 且参数位移当量 < {disp_tol} px"
            f"(双容差缺一不可)——实际 RMS 极差 {rms_spread:.6f} px、"
            f"位移当量 {param_disp:.6f} px,判 {verdict}"
            + (f";原因:{';'.join(reasons)}" if reasons else "。"))
    return {"verdict": verdict, "rms_spread_px": float(rms_spread),
            "param_disp_px": float(param_disp), "note": note}


def run_multistart(obs, board: BoardSpec, cfg: CalibrationConfig,
                   progress=None) -> dict:
    """多起点扰动一致性检验(D-15):扰动初值 × N 次,验"解是不是唯一"。

    白话:从 OpenCV 解出发,给内参/畸变/位姿各加一点随机扰动再解一次,
    重复 N 次——好的解应该"条条大路通罗马":各起点的重投影 RMS 与参数
    (换算成像素位移当量)都落在双容差内;卡到坏解的(尤其 m4 鱼眼,
    初值不当时易发生)会显形为 RMS 或参数的明显偏离——**卡坏解是要观察
    的现象不是事故**,照实记进 bad_solutions,绝不静默(D-15)。

    参数:
        obs:      观测集(全量正式口径);
        board:    棋盘规格(装配核对 + 位移当量的投影网格);
        cfg:      全量配置——随机性从 cfg.seed 按 (模型, 起点序号) 确定性
                  派生(_seed_for,同配置重跑逐位一致);次数与扰动幅度、
                  双容差、并行上限都从 cfg.evaluate 读;
        progress: 每完成一个起点回调一次(库层不依赖命令行框架)。

    返回 {"models": {mX: {"n_starts", "n_converged", "verdict",
    "rms_spread_px", "param_disp_px", "note", "starts", "bad_solutions"}}}。
    落盘交调用方(合并进 optimizer.json 的 multistart 键)。
    """
    eval_cfg = cfg.evaluate
    scipy_cfg = cfg.calibrate.scipy
    cv2.setNumThreads(cfg.cv2_threads)   # D-12:线程数从配置读;关掉 OpenCV
    # 内部多线程同时换来基线求解逐位确定(多线程归约顺序不定会末位漂移,
    # "同种子重跑逐位一致"的承诺就守不住)
    expected_points = board.cols * board.rows
    if obs.object_points.shape[0] != expected_points:
        raise ValueError(
            f"观测集角点数({obs.object_points.shape[0]})与棋盘规格"
            f"({board.cols}x{board.rows} = {expected_points} 个内角点)对不上:"
            f"多半是 board.yaml 与角点文件不是同一套配置,请核对后重跑。")

    models_out: dict[str, dict] = {}
    for model_id in eval_cfg.models:
        model = create_model(model_id)
        n_starts = _multistart_n_for(model, eval_cfg)
        # 扰动基线 = OpenCV 解(D-15:从 OpenCV 初始化出发)
        baseline = solve_with_opencv(obs, model)
        if not baseline.converged or baseline.rms_overall is None:
            models_out[model_id] = {
                "n_starts": 0, "n_converged": 0,
                "verdict": "inconsistent", "rms_spread_px": None,
                "param_disp_px": None,
                "note": (f"{model_id} 的 OpenCV 基线没解出来,扰动无从谈起:"
                         f"{baseline.failure_note}。先排查基线,再重跑本检验。"),
                "starts": [],
                "bad_solutions": [{"start": "baseline", "seed": None,
                                   "failure_note": baseline.failure_note}],
            }
            continue

        # 逐起点:种子确定性派生 → 受控扰动 → 任务小元组(可序列化)
        tasks: list[tuple] = []
        seeds: list[int] = []
        for i in range(n_starts):
            seed = _seed_for(f"mstart|{model_id}|{i}", cfg.seed)
            seeds.append(seed)
            perturbed = _perturbed_init(baseline, eval_cfg,
                                        np.random.default_rng(seed))
            tasks.append((model_id, i, seed, perturbed, obs, scipy_cfg))

        starts: list[dict] = []
        iterator = run_parallel(tasks, _solve_multistart_start,
                                eval_cfg.evaluate_max_workers)
        for i in range(n_starts):
            try:
                start_index, seed, result = next(iterator)
            except Exception as e:  # noqa: BLE001  单起点异常转契约记录,不中断
                starts.append({"start": i, "seed": seeds[i], "converged": False,
                               "rms_overall": None, "params": {},
                               "failure_note": (f"起点 {i} 求解未跑成:"
                                                f"{type(e).__name__}——{e}")})
            else:
                starts.append({
                    "start": int(start_index),
                    "seed": int(seed),
                    "converged": bool(result.converged),
                    "rms_overall": (float(result.rms_overall)
                                    if result.rms_overall is not None else None),
                    "params": (dict(result.params) if result.converged else {}),
                    "failure_note": result.failure_note,
                })
            if progress:
                progress({"model": model_id, "done": i + 1,
                          "total": n_starts})

        judgment = _judge_multistart_consistency(
            model, obs, baseline.rvecs[0], baseline.tvecs[0], starts,
            eval_cfg)
        converged_rows = [s for s in starts if s["converged"]]
        models_out[model_id] = {
            "n_starts": n_starts,
            "n_converged": len(converged_rows),
            "verdict": judgment["verdict"],
            "rms_spread_px": judgment["rms_spread_px"],
            "param_disp_px": judgment["param_disp_px"],
            "note": judgment["note"],
            "starts": starts,
            # 卡坏解照实点名(D-15 要观察的现象,m4 重点):起点号 + 中文原因
            "bad_solutions": [{"start": s["start"], "seed": s["seed"],
                               "failure_note": s["failure_note"]}
                              for s in starts if not s["converged"]],
        }

    return {"models": models_out}
