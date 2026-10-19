"""四类防作弊实验的编排:交叉验证 / 留一姿态 / 重复张分组 / bootstrap 重采样。

白话:阶段四的模型选型要给每个候选模型发一张"防作弊成绩单"——换一批
没见过的姿态还准不准(交叉验证、留一姿态)、参数稳不稳(重复张分组、
bootstrap 重采样)。四类实验全部走同一条求解路径:OpenCV 初始化 →
scipy 精化(优化器配置用 04-02 优化器对比选出的推荐配置,D-13),全部
随机性只从全局种子按标签确定性派生——同种子重跑逐位一致,结论可复查。

铁律(04-01 切分层已内置,这里负责消费):
- 一切切分只操作姿态 ID,同一姿态的全部连拍同进同出(零重叠断言在
  splitting.py 函数体内,这里不再复述);
- 验证侧外参绝不参与训练:验证姿态一律用 solve_pose_only(内参当尺子、
  只挪棋盘位姿的小求解)评分。这是防"第二层泄漏"的关键——让验证姿态
  的外参也进训练,等于考试前把答案塞给学生,held-out 误差失真
  (04-RESEARCH Pitfall 6)。

失败契约:单次求解失败转失败契约占行,绝不中断整批(m3/m4 在小训练集
上不收敛是"数据现象"不是"程序事故");失败率进汇总,逐条带中文说明。
worker 层 try/except 兜底 + 消费循环宽接双保险(串行回退时 worker 异常
发生在 run_parallel 生成器体内会杀死整个任务流、后续任务被静默吞掉,
体例照 optimizer.py 的既定教训)。

并行与任务打包(Pitfall 7):任务元组只带姿态集合等小对象;整份观测集
随任务传递,由并行框架对 numpy 大数组自动内存映射共享,不逐任务复制。
正式口径先跑 CV(100 次)校准耗时,再放 bootstrap 的大批量(800 次)。

落盘边界:本模块只算不写。返回值里私有 _detail 键(明细数组)交
04-05 的产物层转 npz 旁车,json/下游只消费汇总键(产物保真:逐姿态、
逐次重采样的原始数值全保留,不只在源头汇总完就丢)。
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from calib.evaluation.parallel import run_parallel
from calib.evaluation.splitting import (
    _pose_groups,
    assign_cv_folds,
    obs_rows,
    pose_row_mask,
    unique_poses,
)
from calib.io.config import (
    _ALLOWED_EVAL_LOSSES,
    BoardSpec,
    CalibrationConfig,
    ScipyConfig,
)
from calib.models import create_model
from calib.models.base import CameraModel, transform_to_camera
from calib.solve.observation import ObservationSet
from calib.solve.opencv import solve_with_opencv
from calib.solve.result import CalibrationResult
from calib.solve.scipy_solver import solve_with_scipy
from calib.solve.scoring import score_views, solve_pose_only

# ---------------------------------------------------------------------------
# 共享胶水:装配核对 / 推荐配置 / 统一求解路径 / 验证评分 / 像素位移当量
# ---------------------------------------------------------------------------


def _check_obs_board(obs: ObservationSet, board: BoardSpec) -> None:
    """观测集与棋盘规格的装配核对:角点数对不上当场报错,绝不带病进求解。"""
    expected = board.cols * board.rows
    if obs.object_points.shape[0] != expected:
        raise ValueError(
            f"观测集角点数({obs.object_points.shape[0]})与棋盘规格"
            f"({board.cols}x{board.rows} = {expected} 个内角点)对不上:"
            f"多半是 board.yaml 与角点文件不是同一套配置,请核对后重跑。"
        )


def _recommended_kernel(optimizer_config: dict | None, model_id: str) -> tuple[str, float | None]:
    """从推荐优化配置取该模型的 (损失核, f_scale)(D-13)。

    推荐配置来自 optimizer.json 的 recommended_solver(读取由调用方负责,
    形如 {"mX": {"loss": "linear"}} 或 {"loss": "soft_l1", "f_scale_px": ...})
    ——四类实验统一用它,否则"选型用默认配置、之后发现更优配置又全部
    返工"。条目缺失或核不在名单内当场中文报错,绝不静默回退默认配置
    (那会让四实验与推荐口径悄悄分叉,D-13 作废)。
    """
    entry = (optimizer_config or {}).get(model_id)
    if not isinstance(entry, dict):
        raise ValueError(  # noqa: TRY004  项目统一中文 ValueError(体例照 io/config.py),不按类型换异常
            f"推荐优化配置里没有模型 {model_id} 的条目(拿到的是 {entry!r})。"
            f"四类实验统一用 optimizer.json 的 recommended_solver(D-13),"
            f"请先把该产物读进来再调实验函数,不要手工拼配置。"
        )
    loss = entry.get("loss")
    if loss not in _ALLOWED_EVAL_LOSSES:
        raise ValueError(
            f"模型 {model_id} 的推荐核 {loss!r} 不在白名单 "
            f"{sorted(_ALLOWED_EVAL_LOSSES)} 内。推荐配置应来自 "
            f"optimizer.json 的 recommended_solver,请核对该产物。"
        )
    f_scale = entry.get("f_scale_px")
    return loss, (float(f_scale) if f_scale is not None else None)


def make_calibrate_fn(
    model: CameraModel, scipy_cfg: ScipyConfig, loss: str, f_scale: float | None = None
):
    """统一求解路径工厂:返回 calibrate_fn(train_obs) -> CalibrationResult。

    白话:折内/组内/次次重采样的每次训练标定都走同一套两步——
    1. OpenCV 原生求解拿初值(棋盘角点上的经典解法,几秒钟);
    2. scipy 最小二乘精化(把内参和各照片的位姿放在一起慢慢调到最优),
       用推荐核 loss;鲁棒核必带 rms_linear_ref = 本次训练子集 OpenCV
       初值的回代 RMS——没有参照,鲁棒核"实用收敛"的判定就失真
       (Anti-Pattern 5)。

    f_scale(鲁棒核的"大残差从多大开始被压低"的分界尺度)只在推荐配置
    显式给出(f_scale_px 键)时覆盖配置的 auto 档;没给就保持 auto。
    """
    scipy_cfg_eff = scipy_cfg if f_scale is None else replace(scipy_cfg, f_scale=f_scale)

    def calibrate_fn(train_obs: ObservationSet) -> CalibrationResult:
        init = solve_with_opencv(train_obs, model)
        if not init.converged or init.rms_overall is None:
            return init  # OpenCV 初值失败本身已是失败契约,精化无从谈起
        return solve_with_scipy(
            train_obs,
            model,
            scipy_cfg_eff,
            init=init,
            loss=loss,
            rms_linear_ref=float(init.rms_overall),
        )

    return calibrate_fn


def _holdout_pose_errors(
    val_obs: ObservationSet, params: np.ndarray, model: CameraModel
) -> list[dict]:
    """验证侧逐姿态评分(防"第二层泄漏"的关键执行位)。

    每个验证姿态:该姿态每张照片各用 solve_pose_only 只估位姿后回代 RMS,
    姿态级误差 = 各张 RMS 的平均(D-02:先按姿态平均、再进折级;不按
    角点数加权——同一姿态的连拍近似同一张照片,按角点数加权等于给同一
    姿态重复计票)。验证姿态的外参从不参与训练。

    单张评分失败记中文备注不中断;姿态内仍有存活视图时用存活视图平均,
    全部失败时 rms_px = None(交由调用方按失败口径汇总)。
    """
    rows: list[dict] = []
    for pose, view_rows in _pose_groups(val_obs).items():
        view_rms: list[float] = []
        notes: list[str] = []
        for r in view_rows:
            try:
                _rvec, _tvec, rms = solve_pose_only(
                    val_obs.image_points[r], val_obs.object_points, params, model
                )
                view_rms.append(float(rms))
            except (ValueError, AssertionError) as e:
                notes.append(f"视图 {val_obs.image_ids[r]} 评分失败:{e}")
        rows.append(
            {
                "pose_id": pose,
                "rms_px": (float(np.mean(view_rms)) if view_rms else None),
                "n_views": len(view_rms),
                "failure_note": (";".join(notes) if notes else None),
            }
        )
    return rows


def _px_dispersion(
    params_list: list,
    model: CameraModel,
    object_points: np.ndarray,
    ref_rvec: np.ndarray,
    ref_tvec: np.ndarray,
) -> float:
    """参数向量组换算成"像素位移当量":同一参考位姿下,每两组参数分别
    投影棋盘物点网格,逐角点位移的最大值(两两取最坏)。

    为什么用像素当量而不用相对百分比:参数数值差多少没有直观意义
    (fx 差 5 个单位是大是小,要看焦距本身多大);相对百分比对近零的
    畸变参数更是无意义(0.001 与 0.002 差了 100%,画面上挪不到 0.1
    像素)。投影位移是唯一能把不同量纲的参数放上天平的公共货币——
    判行表稳定判据(stability_disp_px_max)就按它计。

    参数接受按参数名的字典或按 param_names 布局的向量;单组参数无从谈
    "两两之间"的位移,返回 0.0。
    """
    pts_cam, _pose_deriv = transform_to_camera(object_points, ref_rvec, ref_tvec)
    projections: list[np.ndarray] = []
    for params in params_list:
        if isinstance(params, dict):
            arr = np.array([params[name] for name in model.param_names], dtype=np.float64)
        else:
            arr = np.asarray(params, dtype=np.float64).ravel()
        projections.append(model.project(pts_cam, arr))
    worst = 0.0
    for a in range(len(projections)):
        for b in range(a + 1, len(projections)):
            disp = np.linalg.norm(projections[a] - projections[b], axis=1)
            worst = max(worst, float(disp.max()))
    return worst


# ---------------------------------------------------------------------------
# 交叉验证(cv_pose_grouped,D-01/D-02,EVA-01 实验侧)
# ---------------------------------------------------------------------------


def _cv_fold_row(
    model_id: str,
    round_idx: int,
    fold_idx: int,
    n_train_poses: int,
    n_val_poses: int,
    n_train_views: int | None = None,
    n_val_views: int | None = None,
    train_rms_px: float | None = None,
    heldout_px: float | None = None,
    f_scale_actual: float | None = None,
    failure_note: str | None = None,
) -> dict:
    """一行折级汇总(键一个不少;失败时数值字段 None、中文说明非空)。"""
    return {
        "round": int(round_idx),
        "fold": int(fold_idx),
        "model_id": model_id,
        "train_rms_px": train_rms_px,
        "heldout_px": heldout_px,
        "n_train_poses": int(n_train_poses),
        "n_val_poses": int(n_val_poses),
        "n_train_views": (None if n_train_views is None else int(n_train_views)),
        "n_val_views": (None if n_val_views is None else int(n_val_views)),
        "f_scale_actual": f_scale_actual,
        "failure_note": failure_note,
    }


def _cv_fold_worker(task: tuple) -> dict:
    """单折单模型求解(worker,顶层函数才能被序列化传给并行子进程)。

    返回 {"row": 折级行, "pose_rows": 逐姿态明细行}。求解器内部的数值
    失败已在 solve_with_scipy 里转成失败契约;这里再兜一层编排层的意外
    (串行回退时 worker 异常发生在 run_parallel 生成器体内会杀死整个
    任务流、后续折被静默吞掉——异常必须到不了生成器)。
    """
    (model_id, round_idx, fold_idx, train_poses, val_poses, obs, scipy_cfg, loss, f_scale) = task
    try:
        model = create_model(model_id)
        train_obs = obs_rows(obs, pose_row_mask(obs, train_poses))
        val_obs = obs_rows(obs, pose_row_mask(obs, val_poses))
        calibrate_fn = make_calibrate_fn(model, scipy_cfg, loss, f_scale)
        result = calibrate_fn(train_obs)
        n_train_views = len(train_obs.image_ids)
        n_val_views = len(val_obs.image_ids)
        if not result.converged or result.rms_overall is None:
            return {
                "row": _cv_fold_row(
                    model_id,
                    round_idx,
                    fold_idx,
                    n_train_poses=len(train_poses),
                    n_val_poses=len(val_poses),
                    n_train_views=n_train_views,
                    n_val_views=n_val_views,
                    failure_note=result.failure_note,
                ),
                "pose_rows": [],
            }

        params_vec = np.array([result.params[name] for name in model.param_names], dtype=np.float64)
        train_rms = float(
            np.mean(score_views(train_obs, params_vec, result.rvecs, result.tvecs, model))
        )
        pose_rows = _holdout_pose_errors(val_obs, params_vec, model)
        heldout_vals = [r["rms_px"] for r in pose_rows if r["rms_px"] is not None]
        heldout = float(np.mean(heldout_vals)) if heldout_vals else None
        failed_notes = [r["failure_note"] for r in pose_rows if r["failure_note"]]
        if heldout is None:
            note = "验证姿态全部评分失败:" + ";".join(failed_notes)
        elif failed_notes:
            note = "部分验证视图评分失败(姿态级取存活视图平均):" + ";".join(failed_notes)
        else:
            note = None
        return {
            "row": _cv_fold_row(
                model_id,
                round_idx,
                fold_idx,
                n_train_poses=len(train_poses),
                n_val_poses=len(val_poses),
                n_train_views=n_train_views,
                n_val_views=n_val_views,
                train_rms_px=train_rms,
                heldout_px=heldout,
                f_scale_actual=result.solver_meta.get("f_scale"),
                failure_note=note,
            ),
            "pose_rows": [
                {
                    "model_id": model_id,
                    "round": round_idx,
                    "fold": fold_idx,
                    "pose_id": r["pose_id"],
                    "rms_px": r["rms_px"],
                    "n_views": r["n_views"],
                    "failure_note": r["failure_note"],
                }
                for r in pose_rows
            ],
        }
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        n_train_views = int(pose_row_mask(obs, train_poses).sum())
        n_val_views = int(pose_row_mask(obs, val_poses).sum())
        return {
            "row": _cv_fold_row(
                model_id,
                round_idx,
                fold_idx,
                n_train_poses=len(train_poses),
                n_val_poses=len(val_poses),
                n_train_views=n_train_views,
                n_val_views=n_val_views,
                failure_note=(
                    f"{model_id} 第 {round_idx} 轮第 {fold_idx} 折"
                    f"未跑成:{type(exc).__name__}——{exc}"
                ),
            ),
            "pose_rows": [],
        }


def cv_pose_grouped(
    obs: ObservationSet,
    board: BoardSpec,
    cfg: CalibrationConfig,
    optimizer_config: dict,
    progress=None,
) -> dict:
    """按姿态分组 K 折交叉验证(D-01/D-02,EVA-01 实验侧)。

    白话:每轮把全体姿态随机分成 K 折,每折轮流当考题——训练侧拿其余
    姿态的**全部连拍**重新标定一次(D-02 全视图口径),验证侧的考题姿态
    逐个"只估位姿"打分,先平均成姿态级误差、再平均成折级 held-out。
    正式口径 5 折 × 5 轮 = 25 个折级样本/模型,供选型统计(04-03)做
    同折配对比较(折级样本必须逐折对齐同一轮切分)。

    签名说明:取整个 CalibrationConfig(要 cfg.seed 派生折分配随机性、
    cfg.evaluate 读折数/轮数/模型/并行上限、cfg.calibrate.scipy 读精化
    参数)——照 04-01/04-02 既定体例(要种子的实验函数取全量配置)。

    返回 {"schema": "cv-v1", "models": {mX: {"fold_heldout", "fold_train",
    "mean_heldout_px", "mean_train_px", "n_failed_folds", "loss_used",
    "fold_rows"}}, "per_pose_rows": [...], "meta", "_cv_detail"}——汇总键
    名对齐 04-03 判行表的 cv_summary 消费面;_cv_detail 明细数组(折分配
    表 + 逐姿态 RMS,NaN 记失败)交 04-05 产物层落 npz,本模块不写盘。
    """
    eval_cfg = cfg.evaluate
    scipy_cfg = cfg.calibrate.scipy
    cv2.setNumThreads(cfg.cv2_threads)  # D-12:线程数配置化;多线程归约顺序
    # 不定会末位漂移,"同种子重跑逐位一致"的承诺守不住(04-02 教训)
    _check_obs_board(obs, board)

    poses = unique_poses(obs)
    assignment = assign_cv_folds(poses, cfg)
    models = tuple(eval_cfg.models)
    kernels = {m: _recommended_kernel(optimizer_config, m) for m in models}

    # 任务小元组(姿态集合 + 共享观测集,大数组由并行框架内存映射,Pitfall 7)
    tasks = [
        (
            m,
            e["round"],
            e["fold"],
            tuple(e["train_poses"]),
            tuple(e["val_poses"]),
            obs,
            scipy_cfg,
            kernels[m][0],
            kernels[m][1],
        )
        for m in models
        for e in assignment
    ]

    results: dict[tuple[str, int, int], dict] = {}
    iterator = run_parallel(tasks, _cv_fold_worker, eval_cfg.evaluate_max_workers)
    for done, task in enumerate(tasks, start=1):
        model_id, round_idx, fold_idx = task[0], task[1], task[2]
        try:
            payload = next(iterator)
        except StopIteration:  # pragma: no cover - 任务数与产出数恒相等
            raise AssertionError(
                f"并行产出数与任务数对不上(CV 任务 {len(tasks)} 个),"
                f"编排层代码契约被破坏,请检查 cv_pose_grouped。"
            )
        except Exception as e:  # noqa: BLE001  双保险:worker 漏网异常转失败契约
            payload = {
                "row": _cv_fold_row(
                    model_id,
                    round_idx,
                    fold_idx,
                    n_train_poses=len(task[3]),
                    n_val_poses=len(task[4]),
                    n_train_views=int(pose_row_mask(obs, task[3]).sum()),
                    n_val_views=int(pose_row_mask(obs, task[4]).sum()),
                    failure_note=(
                        f"{model_id} 第 {round_idx} 轮第 {fold_idx} 折"
                        f"未跑成:{type(e).__name__}——{e}"
                    ),
                ),
                "pose_rows": [],
            }
        results[(model_id, round_idx, fold_idx)] = payload
        if progress:
            progress({"done": done, "total": len(tasks)})

    # 汇总:每模型折级样本按分配表顺序排(04-03 同折配对比较依赖对齐)
    models_out: dict[str, dict] = {}
    per_pose_rows: list[dict] = []
    for m in models:
        loss, _f_scale = kernels[m]
        fold_rows: list[dict] = []
        for e in assignment:
            payload = results[(m, e["round"], e["fold"])]
            fold_rows.append(payload["row"])
            per_pose_rows.extend(payload["pose_rows"])
        fold_heldout = [row["heldout_px"] for row in fold_rows]
        fold_train = [row["train_rms_px"] for row in fold_rows]
        valid_h = [v for v in fold_heldout if v is not None]
        valid_t = [v for v in fold_train if v is not None]
        models_out[m] = {
            "fold_heldout": fold_heldout,
            "fold_train": fold_train,
            "mean_heldout_px": (float(np.mean(valid_h)) if valid_h else None),
            "mean_train_px": (float(np.mean(valid_t)) if valid_t else None),
            "n_failed_folds": int(sum(1 for v in fold_heldout if v is None)),
            "loss_used": loss,
            "fold_rows": fold_rows,
        }

    detail = {
        "assignment_round": np.array([e["round"] for e in assignment], dtype=np.int64),
        "assignment_fold": np.array([e["fold"] for e in assignment], dtype=np.int64),
        "assignment_train_poses": np.array([",".join(e["train_poses"]) for e in assignment]),
        "assignment_val_poses": np.array([",".join(e["val_poses"]) for e in assignment]),
        "detail_model": np.array([r["model_id"] for r in per_pose_rows]),
        "detail_round": np.array([r["round"] for r in per_pose_rows], dtype=np.int64),
        "detail_fold": np.array([r["fold"] for r in per_pose_rows], dtype=np.int64),
        "detail_pose": np.array([r["pose_id"] for r in per_pose_rows]),
        "detail_rms": np.array(
            [np.nan if r["rms_px"] is None else r["rms_px"] for r in per_pose_rows],
            dtype=np.float64,
        ),
    }
    return {
        "schema": "cv-v1",
        "models": models_out,
        "per_pose_rows": per_pose_rows,
        "meta": {
            "n_models": len(models),
            "n_rounds": int(eval_cfg.cv_n_rounds),
            "n_folds": int(eval_cfg.cv_n_folds),
            "n_solves": len(tasks),
            "n_failed_folds_total": int(sum(m["n_failed_folds"] for m in models_out.values())),
            "seed": int(cfg.seed),
            "max_workers": int(eval_cfg.evaluate_max_workers),
        },
        "_cv_detail": detail,
    }
