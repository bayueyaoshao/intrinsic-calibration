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

from pathlib import Path

from calib.evaluation.artifacts import SCHEMA_VERSION
from calib.evaluation.parallel import run_parallel
from calib.io.config import BoardSpec, EvaluateConfig, ScipyConfig
from calib.models import create_model
from calib.solve.matrix import _load_init_result, combo_dir_name
from calib.solve.observation import load_observation_set
from calib.solve.result import CalibrationResult
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
    求解器内部的数值失败已在 solve_with_scipy 里转成失败契约返回,
    这里不做二次兜底。
    """
    model_id, loss, obs, scipy_cfg, init, rms_linear_ref = task
    model = create_model(model_id)
    return solve_with_scipy(obs, model, scipy_cfg, init=init, loss=loss,
                            rms_linear_ref=rms_linear_ref)


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

    return {
        "schema_version": SCHEMA_VERSION,
        "recommended_solver": recommend_optimization_config(combos, eval_cfg),
        "combos": combos,
        "multistart": None,   # 由 run_multistart 产出后合并进同一份报告
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
