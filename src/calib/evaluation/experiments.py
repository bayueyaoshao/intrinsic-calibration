"""四类防作弊实验的编排:交叉验证 / 留一姿态 / 重复张分组 / bootstrap 重采样,
外加选型交付(run_delivery:全量重训 + test 独立照片逐张并排评分,D-16)。

白话:阶段四的模型选型要给每个候选模型发一张"防作弊成绩单"——换一批
没见过的姿态还准不准(交叉验证、留一姿态)、参数稳不稳(重复张分组、
bootstrap 重采样)。四类实验全部走同一条求解路径:OpenCV 初始化 →
scipy 精化(优化器配置用 04-02 优化器对比选出的推荐配置,D-13),全部
随机性只从全局种子按标签确定性派生——同种子重跑逐位一致,结论可复查。
选型定案后,run_delivery 用全量数据给每个候选模型重训一遍最终内参
(CV 只用于选型,交付一律全量重训——既定算法惯例),再让全部候选模型
在 test 独立照片上逐张打分并排(D-16:评分只是投影计算几乎不费时,
真实泛化差距一目了然)。

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
线程锁(OpenCV 线程数)在入口和每个 worker 里各上一次:并行子进程是
全新进程,父进程的设置传不过去,子进程多线程归约顺序不定会末位漂移,
"同种子重跑逐位一致 / 并行与串行同结果"就守不住(04-02 教训的进程级
补全)。正式口径先跑 CV(100 次)校准耗时,再放 bootstrap 的大批量
(800 次)。

落盘边界:本模块只算不写。返回值里私有 _detail 键(明细数组)交
04-05 的产物层转 npz 旁车,json/下游只消费汇总键(产物保真:逐姿态、
逐次重采样的原始数值全保留,不只在源头汇总完就丢)。
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from calib.evaluation.parallel import TaskFailed, run_parallel
from calib.evaluation.selection import bic_full_fit, crlb_alert_check, mad_outlier_poses
from calib.evaluation.splitting import (
    _pose_groups,
    assign_cv_folds,
    bootstrap_resample,
    leave_one_pose_out,
    obs_rows,
    pose_row_mask,
    slice_groups_by_index,
    split_by_pose_ids,
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
from calib.noise.fisher import crlb_se, marginalized_fisher
from calib.solve.observation import ObservationSet
from calib.solve.opencv import solve_with_opencv
from calib.solve.result import CalibrationResult, compute_result_metrics
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
    (
        model_id,
        round_idx,
        fold_idx,
        train_poses,
        val_poses,
        obs,
        scipy_cfg,
        loss,
        f_scale,
        cv2_threads,
    ) = task
    # 线程锁必须在 worker 里再上一次:并行子进程是全新进程,入口处
    # (父进程)的 setNumThreads 管不到它们——子进程多线程归约顺序不定
    # 会末位漂移,"并行与串行同结果"守不住(04-02 教训的进程级补全)
    cv2.setNumThreads(cv2_threads)
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
            cfg.cv2_threads,
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
        except Exception as e:  # noqa: BLE001  双保险:异常已由并行助手包信封(正常到不了这里)
            payload = TaskFailed(e)
        if isinstance(payload, TaskFailed):
            e = payload.error
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


# ---------------------------------------------------------------------------
# 留一姿态(lopo,D-08,EVA-05 实验侧)
# ---------------------------------------------------------------------------


def _lopo_row(
    model_id: str,
    val_pose: str,
    n_train_poses: int,
    n_train_views: int | None = None,
    train_rms_px: float | None = None,
    heldout_px: float | None = None,
    f_scale_actual: float | None = None,
    failure_note: str | None = None,
) -> dict:
    """一行留一姿态汇总(键一个不少;失败时数值字段 None、中文说明非空)。"""
    return {
        "model_id": model_id,
        "val_pose": val_pose,
        "heldout_px": heldout_px,
        "train_rms_px": train_rms_px,
        "n_train_poses": int(n_train_poses),
        "n_train_views": (None if n_train_views is None else int(n_train_views)),
        "f_scale_actual": f_scale_actual,
        "failure_note": failure_note,
    }


def _lopo_worker(task: tuple) -> dict:
    """单姿态求解与评分(worker,顶层函数才能被序列化传给并行子进程)。

    训练侧 = 其余全部姿态(split_by_pose_ids 内置零重叠硬门禁),验证侧 =
    唯一考题姿态只估位姿评分(外参不训练)。异常兜底与 CV 折 worker 同体例:
    失败转契约占行,异常到不了生成器。
    """
    (model_id, val_pose, train_poses, obs, scipy_cfg, loss, f_scale, cv2_threads) = task
    # 线程锁必须在 worker 里再上一次(并行子进程是全新进程,见 _cv_fold_worker)
    cv2.setNumThreads(cv2_threads)
    try:
        model = create_model(model_id)
        train_obs, val_obs = split_by_pose_ids(obs, train_poses)
        calibrate_fn = make_calibrate_fn(model, scipy_cfg, loss, f_scale)
        result = calibrate_fn(train_obs)
        n_train_views = len(train_obs.image_ids)
        if not result.converged or result.rms_overall is None:
            return _lopo_row(
                model_id,
                val_pose,
                n_train_poses=len(train_poses),
                n_train_views=n_train_views,
                failure_note=result.failure_note,
            )
        params_vec = np.array([result.params[name] for name in model.param_names], dtype=np.float64)
        train_rms = float(
            np.mean(score_views(train_obs, params_vec, result.rvecs, result.tvecs, model))
        )
        pose_rows = _holdout_pose_errors(val_obs, params_vec, model)
        # 验证侧恰 1 个姿态:取它那行的姿态级误差与备注
        heldout = pose_rows[0]["rms_px"]
        return _lopo_row(
            model_id,
            val_pose,
            n_train_poses=len(train_poses),
            n_train_views=n_train_views,
            train_rms_px=train_rms,
            heldout_px=heldout,
            f_scale_actual=result.solver_meta.get("f_scale"),
            failure_note=pose_rows[0]["failure_note"],
        )
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return _lopo_row(
            model_id,
            val_pose,
            n_train_poses=len(train_poses),
            n_train_views=int(pose_row_mask(obs, train_poses).sum()),
            failure_note=(f"{model_id} 留一姿态 {val_pose} 未跑成:{type(exc).__name__}——{exc}"),
        )


def lopo(
    obs: ObservationSet,
    board: BoardSpec,
    cfg: CalibrationConfig,
    optimizer_config: dict,
    progress=None,
) -> dict:
    """留一姿态实验(D-08,EVA-05 实验侧):每次留 1 个姿态当考题。

    白话:正式口径 80 个姿态出 80 道题 × 每模型——每次拿 79 个姿态训练、
    留 1 个姿态只估位姿打分。逐姿态误差表回答"是不是某几张照片拖累了
    整体":稳健 z 分数(中位数绝对偏差法,离群点自己撑大标准差躲不过
    点名)超过阈值的姿态被点名"问题姿态",供人工复核与选型参考。

    签名说明:取整个 CalibrationConfig(体例照 cv_pose_grouped——要
    cfg.evaluate 的开关/阈值/模型/并行上限与 cfg.calibrate.scipy)。

    返回 {"schema": "lopo-v1", "models": {mX: {"per_pose", "mean_lopo_px",
    "mad_z", "outlier_poses", "n_failed_poses", "loss_used", "rows"}},
    "meta", "_lopo_detail"}。mad_z 是全体有效姿态的 z 全表(点名用配置
    阈值另过一遍);求解失败的姿态记 None 不参与点名(m3/m4 小训练集
    不收敛是预期内现象,失败率进 n_failed_poses 不静默)。
    """
    eval_cfg = cfg.evaluate
    scipy_cfg = cfg.calibrate.scipy
    if not eval_cfg.lopo_enabled:
        raise ValueError(
            "evaluate.lopo_enabled 为 false,留一姿态实验已被配置关闭;"
            "请不要调用 lopo,或把开关改回 true。"
        )
    cv2.setNumThreads(cfg.cv2_threads)  # 线程锁 = 求解逐位确定(04-02 教训)
    _check_obs_board(obs, board)

    poses = unique_poses(obs)
    assignment = leave_one_pose_out(poses)
    models = tuple(eval_cfg.models)
    kernels = {m: _recommended_kernel(optimizer_config, m) for m in models}

    tasks = [
        (
            m,
            e["val_pose"],
            tuple(e["train_poses"]),
            obs,
            scipy_cfg,
            kernels[m][0],
            kernels[m][1],
            cfg.cv2_threads,
        )
        for m in models
        for e in assignment
    ]
    results: dict[tuple[str, str], dict] = {}
    iterator = run_parallel(tasks, _lopo_worker, eval_cfg.evaluate_max_workers)
    for done, task in enumerate(tasks, start=1):
        model_id, val_pose = task[0], task[1]
        try:
            row = next(iterator)
        except StopIteration:  # pragma: no cover - 任务数与产出数恒相等
            raise AssertionError(
                f"并行产出数与任务数对不上(留一姿态任务 {len(tasks)} 个),"
                f"编排层代码契约被破坏,请检查 lopo。"
            )
        except Exception as e:  # noqa: BLE001  双保险:异常已由并行助手包信封(正常到不了这里)
            row = TaskFailed(e)
        if isinstance(row, TaskFailed):
            e = row.error
            row = _lopo_row(
                model_id,
                val_pose,
                n_train_poses=len(task[2]),
                n_train_views=int(pose_row_mask(obs, task[2]).sum()),
                failure_note=(f"{model_id} 留一姿态 {val_pose} 未跑成:{type(e).__name__}——{e}"),
            )
        results[(model_id, val_pose)] = row
        if progress:
            progress({"done": done, "total": len(tasks)})

    models_out: dict[str, dict] = {}
    all_rows: list[dict] = []
    for m in models:
        loss, _f_scale = kernels[m]
        rows = [results[(m, e["val_pose"])] for e in assignment]
        all_rows.extend(rows)
        per_pose = {row["val_pose"]: row["heldout_px"] for row in rows}
        valid = {p: v for p, v in per_pose.items() if v is not None}
        if valid:
            # 阈值给 -1 取 z 全表(z 恒非负):复用 selection 的同一套 MAD
            # 数学,不写第二份实现;真正的点名按配置阈值再过一遍
            full_z = mad_outlier_poses(valid, threshold=-1.0)
            outliers = mad_outlier_poses(valid, threshold=eval_cfg.mad_z_threshold)
        else:
            full_z, outliers = [], []
        models_out[m] = {
            "per_pose": per_pose,
            "mean_lopo_px": (float(np.mean(list(valid.values()))) if valid else None),
            "mad_z": {p: z for p, z in full_z},
            "outlier_poses": [{"pose_id": p, "mad_z": z} for p, z in outliers],
            "n_failed_poses": int(len(per_pose) - len(valid)),
            "loss_used": loss,
            "rows": rows,
        }

    detail = {
        "detail_model": np.array([r["model_id"] for r in all_rows]),
        "detail_val_pose": np.array([r["val_pose"] for r in all_rows]),
        "detail_rms": np.array(
            [np.nan if r["heldout_px"] is None else r["heldout_px"] for r in all_rows],
            dtype=np.float64,
        ),
    }
    return {
        "schema": "lopo-v1",
        "models": models_out,
        "meta": {
            "n_models": len(models),
            "n_poses": len(poses),
            "n_solves": len(tasks),
            "n_failed_total": int(sum(m["n_failed_poses"] for m in models_out.values())),
            "mad_z_threshold": float(eval_cfg.mad_z_threshold),
            "seed": int(cfg.seed),
            "max_workers": int(eval_cfg.evaluate_max_workers),
        },
        "_lopo_detail": detail,
    }


# ---------------------------------------------------------------------------
# 重复张分组(slices_by_index,D-05/D-06,EVA-03 实验侧)
# ---------------------------------------------------------------------------


def _sigma_temporal_from_noise(noise_report: dict) -> float:
    """从 noise.json 报告读时域口径 σ(白话:同一姿态的连拍互相之间量出来
    的噪声水平,分组实验的对照基准用它)。

    只认时域口径一个键——分组实验每组是"每姿态单张",对照的 σ 必须是
    同口径的单张噪声;拿错口径(比如检测器口径,或连拍取均值后按
    根号张数折减过的聚合口径)会让对照失真,造成假告警或漏告警
    (T-04-04-01,Pitfall 2 的执行位)。缺键/失败轮/非正有限值当场中文
    报错(读取体例照 noise/fisher.py 的 _sigma_scalars_from_noise)。
    """
    if noise_report.get("failure_note"):
        raise AssertionError(
            f"上游 noise.json 是失败轮(failure_note 非空:"
            f"{noise_report['failure_note']}),没有可用的 σ。请先重跑 calib noise。"
        )
    if "sigma_temporal_corner" not in noise_report:
        raise AssertionError(
            "上游 noise.json 缺键 sigma_temporal_corner(时域口径 σ 的契约键)。"
            "分组实验的 CRLB 对照必须用同口径的单张噪声,缺键多半是 noise.json"
            "与当前代码版本不是同一套契约,请重跑 calib noise 后再跑分组实验。"
        )
    value = noise_report["sigma_temporal_corner"]
    if value is None or not np.isfinite(float(value)) or float(value) <= 0:
        raise AssertionError(
            f"上游 noise.json 的 sigma_temporal_corner(时域口径 σ)不是正的"
            f"有限数:{value!r}。σ 是 CRLB 对照的乘数,坏值会污染全部组的结果,"
            f"请重跑 calib noise。"
        )
    return float(value)


def _slice_group_record(
    model_id: str,
    group_idx: int,
    n_poses: int,
    params: dict | None = None,
    rms_overall: float | None = None,
    se: dict | None = None,
    ref_rvec=None,
    ref_tvec=None,
    failure_note: str | None = None,
    crlb_note: str | None = None,
) -> dict:
    """一组的记录(键一个不少):参数向量 + 同口径 CRLB 对照 + 口径标识。

    observation_set 标识语义 = "每姿态单张"(小样按组内实际姿态数如实
    记,正式口径 80 个姿态时即为 views80_single_frame);sigma_source
    锁死 "temporal"——口径标识随记录落盘,下游不可能拿错口径对账
    (T-04-04-01)。
    """
    return {
        "model_id": model_id,
        "group": int(group_idx),
        "n_poses": int(n_poses),
        "observation_set": f"views{int(n_poses)}_single_frame",
        "sigma_source": "temporal",
        "converged": params is not None,
        "params": params if params is not None else {},
        "rms_overall": rms_overall,
        "se": se,
        "ref_rvec": (None if ref_rvec is None else [float(v) for v in ref_rvec]),
        "ref_tvec": (None if ref_tvec is None else [float(v) for v in ref_tvec]),
        "failure_note": failure_note,
        "crlb_note": crlb_note,
    }


def _slice_group_worker(task: tuple) -> dict:
    """单组单模型求解 + 组内同口径 CRLB(worker,顶层函数可序列化)。

    每组独立完整标定(与 CV 同一条求解路径);CRLB 对照**每组新算**:
    线性化点 = 该组该模型自己的收敛解(白话:算理论下界时,把"当前
    解附近误差怎么变"展开的基准点就用这次解本身),σ = noise.json 的
    时域口径,Fisher 数学原样复用 marginalized_fisher / crlb_se(唯一
    数学源约束)。Fisher 算不成不拖垮参数记录(参数照常留账,对照记
    crlb_note 缺席说明)。
    """
    (model_id, group_idx, row_indices, obs, scipy_cfg, loss, f_scale, sigma, cv2_threads) = task
    # 线程锁必须在 worker 里再上一次(并行子进程是全新进程,见 _cv_fold_worker)
    cv2.setNumThreads(cv2_threads)
    try:
        model = create_model(model_id)
        mask = np.zeros(len(obs.image_ids), dtype=bool)
        mask[list(row_indices)] = True
        group_obs = obs_rows(obs, mask)
        result = make_calibrate_fn(model, scipy_cfg, loss, f_scale)(group_obs)
        if not result.converged or result.rms_overall is None:
            return _slice_group_record(
                model_id,
                group_idx,
                n_poses=len(row_indices),
                failure_note=result.failure_note,
            )
        params_vec = np.array([result.params[name] for name in model.param_names], dtype=np.float64)
        record = _slice_group_record(
            model_id,
            group_idx,
            n_poses=len(row_indices),
            params=dict(result.params),
            rms_overall=float(result.rms_overall),
            ref_rvec=result.rvecs[0],
            ref_tvec=result.tvecs[0],
        )
        try:
            f_marg, _f_full, _diag = marginalized_fisher(
                model, group_obs, params_vec, result.rvecs, result.tvecs
            )
            se = crlb_se(f_marg, sigma)
            record["se"] = {name: float(v) for name, v in zip(model.param_names, se)}
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            record["crlb_note"] = f"第 {group_idx} 组的 CRLB 对照未算成:{type(exc).__name__}——{exc}"
        return record
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return _slice_group_record(
            model_id,
            group_idx,
            n_poses=len(row_indices),
            failure_note=(f"{model_id} 第 {group_idx} 组未跑成:{type(exc).__name__}——{exc}"),
        )


def slices_by_index(
    obs: ObservationSet,
    board: BoardSpec,
    cfg: CalibrationConfig,
    optimizer_config: dict,
    noise_report: dict,
    progress=None,
) -> dict:
    """重复张分组实验(D-05/D-06,EVA-03 实验侧):每组每姿态取第 i 张。

    白话:同一姿态拍了多张连拍——第 1 组全取"每姿态的第 1 张"、第 2 组
    全取"第 2 张"……每组独立完整标定一遍。组与组之间姿态集合一样、
    只有噪声不同,所以组间参数的抖动就是"纯噪声"对参数的影响;每组再
    新算一份同口径 CRLB(克拉美-罗下界,白话:"这批数据理论上能把参数
    定到多准"的下界)作对照——经验抖动若显著低于理论下界,物理上不可能,
    说明对照口径错了,触发 D-06 硬告警(经 crlb_alert_check,双档显著性
    从 cfg.evaluate 读,判 FAIL 的说明带参数名不静默)。

    签名说明:取整个 CalibrationConfig(体例照 cv_pose_grouped);
    noise_report 是读回的 noise.json 字典(σ 只从 sigma_temporal_corner
    读,缺键中文报错——T-04-04-01)。

    返回 {"schema": "slices-v1", "models": {mX: {"group_params",
    "empirical_std", "crlb_se_mean", "crlb_alert", "slice_max_disp_px",
    "n_failed_groups", "n_groups", "sigma_used", "loss_used",
    "group_records"}}, "meta", "_slice_detail"}——slice_max_disp_px 键名
    对齐 04-03 判行表 stability_summary 消费面;各组原始 SE 全在明细数组
    (8 份原始全落盘由 04-05 产物层执行,Open Q2)。
    """
    eval_cfg = cfg.evaluate
    scipy_cfg = cfg.calibrate.scipy
    cv2.setNumThreads(cfg.cv2_threads)  # 线程锁 = 求解逐位确定(04-02 教训)
    _check_obs_board(obs, board)
    sigma = _sigma_temporal_from_noise(noise_report)
    groups = slice_groups_by_index(obs, cfg)  # 开关/组内断言在 splitting 层内置
    models = tuple(eval_cfg.models)
    kernels = {m: _recommended_kernel(optimizer_config, m) for m in models}

    tasks = [
        (
            m,
            g["group"],
            tuple(int(r) for r in g["pose_rows"].values()),
            obs,
            scipy_cfg,
            kernels[m][0],
            kernels[m][1],
            sigma,
            cfg.cv2_threads,
        )
        for m in models
        for g in groups
    ]
    results: dict[tuple[str, int], dict] = {}
    iterator = run_parallel(tasks, _slice_group_worker, eval_cfg.evaluate_max_workers)
    for done, task in enumerate(tasks, start=1):
        model_id, group_idx = task[0], task[1]
        try:
            record = next(iterator)
        except StopIteration:  # pragma: no cover - 任务数与产出数恒相等
            raise AssertionError(
                f"并行产出数与任务数对不上(分组任务 {len(tasks)} 个),"
                f"编排层代码契约被破坏,请检查 slices_by_index。"
            )
        except Exception as e:  # noqa: BLE001  双保险:异常已由并行助手包信封(正常到不了这里)
            record = TaskFailed(e)
        if isinstance(record, TaskFailed):
            e = record.error
            record = _slice_group_record(
                model_id,
                group_idx,
                n_poses=len(task[2]),
                failure_note=(f"{model_id} 第 {group_idx} 组未跑成:{type(e).__name__}——{e}"),
            )
        results[(model_id, group_idx)] = record
        if progress:
            progress({"done": done, "total": len(tasks)})

    models_out: dict[str, dict] = {}
    detail: dict[str, dict] = {}
    for m in models:
        loss, _f_scale = kernels[m]
        model = create_model(m)
        names = tuple(model.param_names)
        records = [results[(m, g["group"])] for g in groups]
        converged = [r for r in records if r["converged"]]
        with_se = [r for r in converged if r["se"] is not None]

        group_params = [r["params"] for r in converged]
        empirical_std: dict[str, float] | None = None
        crlb_se_mean: dict[str, float] | None = None
        crlb_alert: dict | None = None
        alert_note: str | None = None
        slice_max_disp_px: float | None = None

        if converged:
            params_arr = np.array([[gp[n] for n in names] for gp in group_params], dtype=np.float64)
            ref0 = converged[0]
            slice_max_disp_px = _px_dispersion(
                group_params,
                model,
                obs.object_points,
                np.asarray(ref0["ref_rvec"], dtype=np.float64),
                np.asarray(ref0["ref_tvec"], dtype=np.float64),
            )
            if len(converged) >= 2:
                empirical_std = {
                    n: float(np.std(params_arr[:, i], ddof=1)) for i, n in enumerate(names)
                }
        if with_se:
            se_arr = np.array([[r["se"][n] for n in names] for r in with_se], dtype=np.float64)
            crlb_se_mean = {n: float(np.mean(se_arr[:, i])) for i, n in enumerate(names)}
            if empirical_std is not None and len(with_se) >= 2:
                crlb_alert = crlb_alert_check(
                    empirical_std,
                    crlb_se_mean,
                    n_groups=len(with_se),
                    alpha_hard=eval_cfg.crlb_alert_alpha_hard,
                    alpha_soft=eval_cfg.crlb_alert_alpha_soft,
                )
            elif empirical_std is None:
                alert_note = "收敛组不足 2 组,经验抖动无从谈起,CRLB 对照告警跳过(不静默)"
        else:
            alert_note = "没有任何一组的 CRLB 对照算成,告警检查跳过(不静默)"
        if crlb_alert is None and alert_note is None and len(with_se) < 2:
            alert_note = "CRLB 对照的有效组数不足 2(卡方自由度 = 组数 − 1),告警检查跳过"

        models_out[m] = {
            "group_params": group_params,
            "empirical_std": empirical_std,
            "crlb_se_mean": crlb_se_mean,
            "crlb_alert": crlb_alert,
            "crlb_note": alert_note,
            "slice_max_disp_px": slice_max_disp_px,
            "n_failed_groups": int(len(records) - len(converged)),
            "n_groups": len(groups),
            "sigma_used": sigma,
            "loss_used": loss,
            "group_records": records,
        }
        detail[m] = {
            "group_index": np.array([r["group"] for r in records], dtype=np.int64),
            # 每组口径标识随明细旁车落盘(组记录本身不进 json,D-13):
            # 没有这两列,下游核对 CRLB 对照口径时只能信汇总、查无原据
            "observation_set": np.array([r["observation_set"] for r in records]),
            "sigma_source": np.array([r["sigma_source"] for r in records]),
            "group_params": np.array(
                [
                    [np.nan if not r["converged"] else r["params"][n] for n in names]
                    for r in records
                ],
                dtype=np.float64,
            ),
            "group_se": np.array(
                [[np.nan if r["se"] is None else r["se"][n] for n in names] for r in records],
                dtype=np.float64,
            ),
            "sigma_used": sigma,
            "param_names": np.array(names),
        }

    return {
        "schema": "slices-v1",
        "models": models_out,
        "meta": {
            "n_models": len(models),
            "n_groups": len(groups),
            "n_solves": len(tasks),
            "n_failed_total": int(sum(m["n_failed_groups"] for m in models_out.values())),
            "sigma_source": "temporal",
            "alpha_hard": float(eval_cfg.crlb_alert_alpha_hard),
            "alpha_soft": float(eval_cfg.crlb_alert_alpha_soft),
            "seed": int(cfg.seed),
            "max_workers": int(eval_cfg.evaluate_max_workers),
        },
        "_slice_detail": detail,
    }


# ---------------------------------------------------------------------------
# bootstrap 重采样(bootstrap_pose,D-04/D-07,EVA-04 实验侧)
# ---------------------------------------------------------------------------


def _bootstrap_record(
    model_id: str,
    rep: int,
    params: dict | None = None,
    rms_overall: float | None = None,
    ref_rvec=None,
    ref_tvec=None,
    failure_note: str | None = None,
) -> dict:
    """一次重采样的记录(键一个不少):整条参数向量 + 回代 RMS + 参考位姿。

    参考位姿取该次解自己的第 0 个视图——像素位移当量(max_disp_px)的
    投影基准;失败转契约占行(params 空、说明带异常类型)。
    """
    return {
        "model_id": model_id,
        "rep": int(rep),
        "converged": params is not None,
        "params": params if params is not None else {},
        "rms_overall": rms_overall,
        "ref_rvec": (None if ref_rvec is None else [float(v) for v in ref_rvec]),
        "ref_tvec": (None if ref_tvec is None else [float(v) for v in ref_tvec]),
        "failure_note": failure_note,
    }


def _bootstrap_worker(task: tuple) -> dict:
    """单次重采样单模型求解(worker,顶层函数才能被序列化传给并行子进程)。

    训练观测集 = 抽中姿态**去重**后带全部连拍:同一姿态抽中 3 次只算一份
    ——bootstrap 考的是"换一批姿态",不是"某个姿态多考几遍",重复计权
    会人为放大被偏爱姿态的话语权。求解与 CV / 分组同一条路径(OpenCV
    初始化 → scipy 精化)。异常兜底同体例:失败转契约占行,异常到不了
    生成器(串行回退不死任务流,体例照 optimizer.py 的既定教训)。
    """
    (model_id, rep, picked_poses, obs, scipy_cfg, loss, f_scale, cv2_threads) = task
    # 线程锁必须在 worker 里再上一次(并行子进程是全新进程,见 _cv_fold_worker)
    cv2.setNumThreads(cv2_threads)
    try:
        model = create_model(model_id)
        train_obs = obs_rows(obs, pose_row_mask(obs, set(picked_poses)))
        result = make_calibrate_fn(model, scipy_cfg, loss, f_scale)(train_obs)
        if not result.converged or result.rms_overall is None:
            return _bootstrap_record(model_id, rep, failure_note=result.failure_note)
        params = {name: float(result.params[name]) for name in model.param_names}
        return _bootstrap_record(
            model_id,
            rep,
            params=params,
            rms_overall=float(result.rms_overall),
            ref_rvec=result.rvecs[0],
            ref_tvec=result.tvecs[0],
        )
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return _bootstrap_record(
            model_id,
            rep,
            failure_note=f"{model_id} 重采样 {rep} 未跑成:{type(exc).__name__}——{exc}",
        )


def bootstrap_pose(
    obs: ObservationSet,
    board: BoardSpec,
    cfg: CalibrationConfig,
    optimizer_config: dict,
    progress=None,
) -> dict:
    """bootstrap 重采样实验(D-04,EVA-04 实验侧):按姿态有放回重抽一批
    姿态重新标定,重复 cfg.evaluate.bootstrap_n 次。

    白话:把全体姿态当纸条放进箱子,抽 n 次(抽了再放回,同一姿态可能被
    抽中多次),用抽中的这一批重新标一遍内参;重复很多次,看参数每次挪多
    少——量的是"换一批姿态,内参会跟着动多少"。

    4 个模型**共用同一批分配表**(先产分配再派任务):bootstrap 的用途是
    跨模型配对比较方差,各模型面对同一组考题,差值才是"模型之间的差"而
    不是"考题难度的差"(D-04)。分配表纯 str/int,可落盘复查。

    抽中姿态带其全部连拍(切分只操作姿态 ID 的铁律消费位);去重语义在
    _bootstrap_worker。

    签名说明:取整个 CalibrationConfig(体例照 cv_pose_grouped——要
    cfg.evaluate 的重采样次数/模型/并行上限与 cfg.calibrate.scipy、cfg.seed)。

    返回 {"schema": "bootstrap-v1", "models": {mX: {"param_std", "param_ci",
    "var_bootstrap", "var_slice", "pose_sampling_var", "var_slice_note",
    "max_disp_px", "n_failed", "loss_used"}}, "meta", "_bootstrap_detail"}。
    var_slice / pose_sampling_var 是占位键:本任务只产出 bootstrap 侧方差,
    分组侧方差由 04-05/04-06 组装侧填入后经 pose_sampling_variance 相减
    (接口在此定死,D-07)。max_disp_px 用像素位移当量(键名对齐判行表
    stability_summary);明细数组(逐次参数向量)挂 _bootstrap_detail 交
    04-05 转旁车文件(产物保真:逐次全存,不只留汇总)。
    """
    eval_cfg = cfg.evaluate
    scipy_cfg = cfg.calibrate.scipy
    cv2.setNumThreads(cfg.cv2_threads)  # 线程锁 = 求解逐位确定(04-02 教训)
    _check_obs_board(obs, board)

    poses = unique_poses(obs)
    assignment = bootstrap_resample(poses, cfg)  # 一批分配,全部模型共用(D-04)
    models = tuple(eval_cfg.models)
    kernels = {m: _recommended_kernel(optimizer_config, m) for m in models}

    tasks = [
        (
            m,
            e["rep"],
            tuple(e["pose_ids"]),
            obs,
            scipy_cfg,
            kernels[m][0],
            kernels[m][1],
            cfg.cv2_threads,
        )
        for e in assignment
        for m in models
    ]
    results: dict[tuple[str, int], dict] = {}
    iterator = run_parallel(tasks, _bootstrap_worker, eval_cfg.evaluate_max_workers)
    for done, task in enumerate(tasks, start=1):
        model_id, rep = task[0], task[1]
        try:
            record = next(iterator)
        except StopIteration:  # pragma: no cover - 任务数与产出数恒相等
            raise AssertionError(
                f"并行产出数与任务数对不上(bootstrap 任务 {len(tasks)} 个),"
                f"编排层代码契约被破坏,请检查 bootstrap_pose。"
            )
        except Exception as e:  # noqa: BLE001  双保险:异常已由并行助手包信封(正常到不了这里)
            record = TaskFailed(e)
        if isinstance(record, TaskFailed):
            e = record.error
            record = _bootstrap_record(
                model_id,
                rep,
                failure_note=f"{model_id} 重采样 {rep} 未跑成:{type(e).__name__}——{e}",
            )
        results[(model_id, rep)] = record
        if progress:
            progress({"done": done, "total": len(tasks)})

    models_out: dict[str, dict] = {}
    detail: dict[str, dict] = {}
    for m in models:
        loss, _f_scale = kernels[m]
        model_obj = create_model(m)
        names = model_obj.param_names
        records = [results[(m, e["rep"])] for e in assignment]
        # 明细矩阵:失败重采样记 NaN 行(产物保真:逐次全存)
        matrix = np.array(
            [[np.nan if not r["converged"] else r["params"][n] for n in names] for r in records],
            dtype=np.float64,
        )
        ok = ~np.isnan(matrix).any(axis=1)
        if int(ok.sum()) >= 2:
            param_std = {n: float(np.std(matrix[ok, i], ddof=1)) for i, n in enumerate(names)}
            param_ci = {
                n: [
                    float(np.percentile(matrix[ok, i], 2.5)),
                    float(np.percentile(matrix[ok, i], 97.5)),
                ]
                for i, n in enumerate(names)
            }
            var_bootstrap = {n: float(np.var(matrix[ok, i], ddof=1)) for i, n in enumerate(names)}
            ref0 = next(r for r in records if r["converged"])
            max_disp_px = _px_dispersion(
                [r["params"] for r in records if r["converged"]],
                model_obj,
                obs.object_points,
                np.asarray(ref0["ref_rvec"], dtype=np.float64),
                np.asarray(ref0["ref_tvec"], dtype=np.float64),
            )
        else:
            # 有效样本不足 2 个,方差/分位数无从谈起:记 None 不冒充数值
            param_std = {n: None for n in names}
            param_ci = {n: None for n in names}
            var_bootstrap = {n: None for n in names}
            max_disp_px = None
        models_out[m] = {
            "param_std": param_std,
            "param_ci": param_ci,
            "var_bootstrap": var_bootstrap,
            "var_slice": None,
            "pose_sampling_var": None,
            "var_slice_note": (
                "分组侧方差由 04-05/04-06 组装时填入;两侧就位后经"
                " pose_sampling_variance 相减得位姿抽样方差(D-07,接口在此定死)"
            ),
            "max_disp_px": max_disp_px,
            "n_failed": int(len(records) - int(ok.sum())),
            "loss_used": loss,
        }
        detail[m] = {
            "rep": np.array([r["rep"] for r in records], dtype=np.int64),
            "params": matrix,
            "rms": np.array(
                [np.nan if r["rms_overall"] is None else r["rms_overall"] for r in records],
                dtype=np.float64,
            ),
            "param_names": np.array(names),
        }

    return {
        "schema": "bootstrap-v1",
        "models": models_out,
        "meta": {
            "n_models": len(models),
            "n_resamples": len(assignment),
            "n_solves": len(tasks),
            "n_failed_total": int(sum(m["n_failed"] for m in models_out.values())),
            "bootstrap_n": int(eval_cfg.bootstrap_n),
            "seed": int(cfg.seed),
            "max_workers": int(eval_cfg.evaluate_max_workers),
        },
        "_bootstrap_detail": detail,
    }


def pose_sampling_variance(var_bootstrap: dict, var_slice: dict) -> dict:
    """来源分离(D-07):逐参数"总方差 − 纯噪声方差 = 位姿抽样方差"。

    白话:重复张分组实验把同一批姿态固定住、只换噪声,量出来的是"纯噪声
    能晃动参数多少"(var_noise);bootstrap 连姿态都在换(换一批姿态重新
    标定),量出来的是"噪声 + 换姿态"合计的晃动(var_total)。两者之差
    就是"换姿态本身"带来的不确定度——即"这批姿态抽得偏不偏"的贡献。
    注意相减的是**方差**(标准差的平方),不是标准差:独立来源的方差才
    可加,标准差不能直接相减。

    负值处置:分组侧方差理论上不应超过 bootstrap 侧(前者只是后者的噪声
    分量),但小样本估计有抖动,真出现了就把位姿抽样方差钳 0,并在该参数
    的 note 里说明(不静默)。参数集合不一致或非有限值当场中文报错。

    返回 {参数: {"var_total", "var_noise", "var_pose_sampling", "pose_share",
    "note"}};pose_share = 位姿抽样方差占总方差的比例(总方差为 0 时记 0)。
    """
    if set(var_bootstrap) != set(var_slice):
        only_boot = sorted(set(var_bootstrap) - set(var_slice))
        only_slice = sorted(set(var_slice) - set(var_bootstrap))
        raise ValueError(
            f"两份方差的参数集合对不上(bootstrap 独有:{only_boot},"
            f"分组独有:{only_slice})。来源分离必须逐参数同集合相减,"
            f"请检查两侧是否来自同一套参数布局。"
        )
    out: dict[str, dict] = {}
    for name, total_raw in var_bootstrap.items():
        total = float(total_raw)
        noise = float(var_slice[name])
        if not np.isfinite(total) or not np.isfinite(noise):
            raise ValueError(
                f"参数 {name} 的方差不是有限数(bootstrap={total!r},分组={noise!r}),"
                f"无法做来源分离;请先检查上游实验是否全部收敛。"
            )
        pose_var = total - noise
        note = None
        if pose_var < 0.0:
            pose_var = 0.0
            note = (
                f"参数 {name} 的分组方差大于 bootstrap 方差,位姿抽样方差按 0 计:"
                "理论上分组(只换噪声)是 bootstrap(换噪声 + 换姿态)的一部分,"
                "倒挂多半是小样本估计的抖动,钳 0 保守处理,不静默。"
            )
        out[name] = {
            "var_total": total,
            "var_noise": noise,
            "var_pose_sampling": pose_var,
            "pose_share": (pose_var / total if total > 0.0 else 0.0),
            "note": note,
        }
    return out


# ---------------------------------------------------------------------------
# 选型交付(run_delivery,D-16/EVA-02 收口):全量重训 + test 逐张并排评分
# ---------------------------------------------------------------------------


def _delivery_model_entry(
    model_id: str,
    n_intrinsic_params: int,
    n_views: int,
    final_params: dict | None = None,
    rms_px: float | None = None,
    rss: float | None = None,
    bic: float | None = None,
    loss_used: str | None = None,
    failure_note: str | None = None,
) -> dict:
    """一个模型的交付记录(键一个不少;失败时数值字段 None、中文说明非空)。

    final_params = 全量重训的最终内参(交付物);rss = 全部重投影残差平方和
    (BIC 的输入,D-10——RSS 取自全量解,这正是"先重训、后选型"的顺序
    理由);bic 在编排内用 bic_full_fit 算好,04-03 的 select_model 只消费
    现成值,不写第二份 BIC 数学。
    """
    return {
        "model_id": model_id,
        "converged": final_params is not None,
        "final_params": final_params if final_params is not None else {},
        "rms_px": rms_px,
        "rss": rss,
        "n_intrinsic_params": int(n_intrinsic_params),
        "n_views": int(n_views),
        "bic": bic,
        "loss_used": loss_used,
        "failure_note": failure_note,
    }


def run_delivery(
    obs_train: ObservationSet,
    obs_test: ObservationSet,
    board: BoardSpec,
    cfg: CalibrationConfig,
    optimizer_config: dict,
    progress=None,
) -> dict:
    """选型交付(D-16,EVA-02/SOL-02 收口):全量重训 + test 逐张并排评分。

    白话:交叉验证只用来"选模型",真正交付的内参要用**全部**训练照片
    重新训一遍(全量 640 视图口径,与选型口径一致,D-03 裁决);重训完
    不急着下结论——四个候选模型的最终内参都在 test 独立照片上逐张打分
    并排(评分 = 内参当尺子、只挪棋盘位姿的投影计算,几乎不费时),
    "选定模型 vs 其他模型"的真实泛化差距一目了然,顺带交叉验证选型结论。

    签名说明:取整个 CalibrationConfig(要 cfg.evaluate 的模型名单与
    bic_n_poses、cfg.calibrate.scipy 读精化参数、cfg.cv2_threads 锁线程)
    ——照 04-01/04-02/04-04 既定体例(实验函数取全量配置),计划文本的
    (models, scipy_cfg, eval_cfg) 散传由此收拢。obs_test 由调用方经
    load_observation_set(split="test") 装载传入(行序契约入口内置断言,
    T-04-05-01);评分行带 image_id 可逐行对账。

    返回 {"schema": "delivery-v1", "models": {mX: 交付记录}, "test_rows":
    [...], "meta", "_delivery_detail"}——models 每项含 final_params / rms_px /
    rss / n_intrinsic_params / n_views / bic(BIC 在此算好:rss 取自全量解
    的全部重投影残差平方和,k = 内参数 + 6 × 视图数,n = cfg.evaluate.
    bic_n_poses,04-03 select_model 只消费现成值,D-10);test_rows 每行
    {"model_id", "image_id", "rms_px", "failure_note"},逐张逐模型全存
    (产物保真,D-16)。单模型重训失败转失败契约占行,不中断其余模型,
    其 test 评分行记 rms_px=None + 中文说明。
    """
    eval_cfg = cfg.evaluate
    scipy_cfg = cfg.calibrate.scipy
    cv2.setNumThreads(cfg.cv2_threads)  # 线程锁 = 求解逐位确定(04-02 教训)
    _check_obs_board(obs_train, board)
    _check_obs_board(obs_test, board)

    models = tuple(eval_cfg.models)
    n_views = len(obs_train.image_ids)
    # 推荐配置在循环外先核一遍:条目缺失/核越界当场报错(不静默回退,D-13),
    # 不让错误拖到半个模型重训完才炸
    kernels = {m: _recommended_kernel(optimizer_config, m) for m in models}

    models_out: dict[str, dict] = {}
    test_rows: list[dict] = []
    detail_model: list[str] = []
    detail_image: list[str] = []
    detail_rms: list[float] = []

    for done, model_id in enumerate(models, start=1):
        loss, f_scale = kernels[model_id]
        model = create_model(model_id)
        names = tuple(model.param_names)
        try:
            # 与 04-02 网格同一接口的两步求解:OpenCV 初值 → scipy 精化
            result = make_calibrate_fn(model, scipy_cfg, loss, f_scale)(obs_train)
            if not result.converged or result.rms_overall is None:
                raise RuntimeError(result.failure_note or "全量重训未收敛(无失败说明)")
            params_vec = np.array(
                [result.params[name] for name in names], dtype=np.float64
            )
            _rms_chk, _per_view, residuals = compute_result_metrics(
                obs_train, params_vec, result.rvecs, result.tvecs, model
            )
            rss = float(np.sum(residuals * residuals))
            k = len(names) + 6 * n_views  # BIC 的 k = 全部被估参数(内参 + 6×视图外参)
            bic = float(bic_full_fit(rss, k, int(eval_cfg.bic_n_poses)))
            entry = _delivery_model_entry(
                model_id,
                n_intrinsic_params=len(names),
                n_views=n_views,
                final_params={name: float(result.params[name]) for name in names},
                rms_px=float(result.rms_overall),
                rss=rss,
                bic=bic,
                loss_used=loss,
            )
            failure_note = None
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            entry = _delivery_model_entry(
                model_id,
                n_intrinsic_params=len(names),
                n_views=n_views,
                loss_used=loss,
                failure_note=(
                    f"{model_id} 全量重训未跑成:{type(exc).__name__}——{exc}"
                ),
            )
            failure_note = entry["failure_note"]
        models_out[model_id] = entry

        # test 逐张评分(D-16 产物保真:逐张逐模型全存,失败记占行)
        for i, image_id in enumerate(obs_test.image_ids):
            if not entry["converged"]:
                row_rms: float | None = None
                row_note = f"模型重训失败,无内参可评分:{failure_note}"
            else:
                try:
                    _rvec, _tvec, rms = solve_pose_only(
                        obs_test.image_points[i], obs_test.object_points, params_vec, model
                    )
                    row_rms = float(rms)
                    row_note = None
                except (ValueError, AssertionError) as e:
                    row_rms = None
                    row_note = f"只估位姿评分失败:{e}"
            test_rows.append(
                {
                    "model_id": model_id,
                    "image_id": image_id,
                    "rms_px": row_rms,
                    "failure_note": row_note,
                }
            )
            detail_model.append(model_id)
            detail_image.append(image_id)
            detail_rms.append(np.nan if row_rms is None else row_rms)
        if progress:
            progress({"done": done, "total": len(models)})

    return {
        "schema": "delivery-v1",
        "models": models_out,
        "test_rows": test_rows,
        "meta": {
            "n_models": len(models),
            "n_views": n_views,
            "n_test_images": len(obs_test.image_ids),
            "n_test_rows": len(test_rows),
            "bic_n_poses": int(eval_cfg.bic_n_poses),
            "optimizer_source": "optimizer.json recommended_solver(D-13 统一配置)",
            "seed": int(cfg.seed),
        },
        "_delivery_detail": {
            "test_model": np.array(detail_model),
            "test_image": np.array(detail_image),
            "test_rms": np.array(detail_rms, dtype=np.float64),
        },
    }
