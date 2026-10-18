"""全矩阵标定编排:读配置矩阵 → 逐组合求解 → 带血缘落盘 → 返回汇总行(D-01)。

白话:这里把"模型 × 求解路径 × 损失核"的清单(全部来自 calibration.yaml,
CLI 零参数,D-12)变成一排目录——每个组合一个目录,里面躺着三件套
(result.json / extrinsics.npz / residuals.npz)与完整血缘。组合顺序按
模型分组:每个模型先跑 OpenCV(它给 scipy 当初值),再跑 scipy 的各条
损失核路径(linear 主线 + soft_l1 鲁棒核对照),所以汇总表里"精化前后"
永远并排相邻——比较是**同一个模型**内的事(Pitfall 8)。

血缘链(成功标准 4):每个组合的 upstream 记两样——消费的那份角点 npz
(名字 + 路径 + sha256),scipy 组合再加同轮 <模型>__opencv/result.json
(初值产物,谁给的头点拿的数,事后可对账)。scipy 的初值从**落盘产物**
读回(load_result + load_extrinsics),不是内存里顺手递——文件是唯一
真相源,血缘里指向的那份文件与实际消费的那份数值严格是同一份。

失败契约的编排级延伸:单个组合内部出任何意外异常(求解器自身的数值
失败已在各自文件里转成失败契约,这里兜的是编排层的意外),都转
make_failure 落盘(结构键全 / 数值 null / 中文 failure_note),绝不中断
其余组合——一个模型病了不该拖死整批实验。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from calib.io.config import BoardSpec, CalibrationConfig, ScipyConfig
from calib.models import create_model
from calib.solve.observation import ObservationSet
from calib.solve.opencv import solve_with_opencv
from calib.solve.result import (
    CalibrationResult,
    compute_result_metrics,
    load_extrinsics,
    load_result,
    make_failure,
    save_result,
)
from calib.solve.scipy_solver import solve_with_scipy
from calib.util.lineage import make_lineage

# 配置文件的默认定位:与 CLI 的 BOARD_PATH / CALIBRATION_PATH 同一套约定
# (相对当前工作目录)——"换数据集只改配置不改代码"的地基,两处必须一致
_DEFAULT_CONFIG_FILES: dict[str, Path] = {
    "board.yaml": Path("config") / "board.yaml",
    "calibration.yaml": Path("config") / "calibration.yaml",
}

# 每完成一个组合的回调:收一行汇总(库层不依赖 typer,播报由 CLI 层做)
ProgressFn = Callable[[dict], None]


def combo_dir_name(model_id: str, solver: str, loss: str | None = None) -> str:
    """组合目录名:m2__opencv;m2__scipy(linear 主线)与 m2__scipy_softl1(其余损失)。

    非主线的损失名去掉下划线拼在后面(soft_l1 → softl1、huber → huber),
    与研究 §Pattern 2 的示例一致;阶段三(Fisher 线性化点)与阶段四(评估)
    按名字引用产物,命名规则必须一处定死、全链只用本函数。
    """
    if solver == "opencv":
        return f"{model_id}__opencv"
    if loss == "linear":
        return f"{model_id}__scipy"
    return f"{model_id}__scipy_{(loss or 'linear').replace('_', '')}"


def _combo_row(name: str, model_id: str, solver: str, loss: str | None,
               result: CalibrationResult) -> dict:
    """组装一行汇总(进 comparison;数值字段失败时为 None,键一个不少)。"""
    return {
        "name": name,
        "model": model_id,
        "solver": solver,
        "loss": loss,
        "n_views": len(result.image_ids),
        "rms_overall": (float(result.rms_overall)
                        if result.rms_overall is not None else None),
        "converged": bool(result.converged),
        "failure_note": result.failure_note,
        # scipy 的停机状态码(>0 正式收敛 / 0 评估上限或无法改进 / -1 报错);
        # OpenCV 路径没有状态码,保持 None
        "solver_status": result.solver_meta.get("status"),
    }


def _residuals_for(obs: ObservationSet, model, result: CalibrationResult) -> np.ndarray:
    """回代重算残差 (V,P,2),给 save_result 的三件套契约配套。

    与求解时用的是同一条代码路径(compute_result_metrics → models.project,
    float64 同一把尺),确定性输入下数值逐位一致,不存在"两把尺"问题;
    求解器把残差算完就丢(只留 RMS),编排层落盘前再要一次。
    """
    params = np.array([result.params[name] for name in model.param_names],
                      dtype=np.float64)
    _rms, _per_view, residuals = compute_result_metrics(
        obs, params, result.rvecs, result.tvecs, model)
    return residuals


def _load_init_result(init_dir: Path, model_id: str) -> CalibrationResult | None:
    """从同轮 <模型>__opencv 产物目录读回初值(文件是唯一真相源)。

    返回 None 的两种情况都由调用方走"通用初始化"路径并留痕:
    result.json 不在(本轮没跑 OpenCV 组合)或未收敛(初值给不出数);
    收敛则连同 extrinsics.npz 的逐视图位姿一起装回 CalibrationResult。
    位姿与 result.json 的 image_ids 逐项核对(行序契约:错位即抛,不带病精化)。
    """
    if not (init_dir / "result.json").is_file():
        return None
    doc = load_result(init_dir)
    if not doc.get("converged") or doc.get("model_id") != model_id:
        return None
    ext = load_extrinsics(init_dir)
    if tuple(ext["image_ids"]) != tuple(doc.get("image_ids", ())):
        raise AssertionError(
            f"初值产物 {init_dir.name} 内部位姿与 result.json 的照片编号对不上"
            f"(行序契约被破坏),这份初值不可信,请重跑该组合。"
        )
    return CalibrationResult(
        converged=True,
        failure_note=None,
        model_id=str(doc["model_id"]),
        solver=str(doc["solver"]),
        solver_meta=dict(doc.get("solver_meta", {})),
        params={str(k): float(v) for k, v in doc["params"].items()},
        rvecs=ext["rvecs"],
        tvecs=ext["tvecs"],
        rms_overall=(float(doc["rms_overall"])
                     if doc.get("rms_overall") is not None else None),
        per_view_rms=None,   # 初值路径用不到,不给占位数
        image_ids=tuple(str(x) for x in doc["image_ids"]),
        image_size=(int(doc["image_size"][0]), int(doc["image_size"][1])),
        n_views=int(doc["n_views"]),
        n_points_per_view=int(doc["n_points_per_view"]),
        init_from=doc.get("init_from"),
        lineage=dict(doc.get("lineage", {})),
    )


def _lineage_for(cfg_files: dict[str, Path], corners_npz: Path,
                 init_dir: Path | None, excluded_images: tuple[str, ...],
                 round_id: str) -> dict:
    """组装一个组合的血缘:配置指纹 + 角点 npz(+ scipy 时的初值产物)+ 剔除名单。

    初值产物路径只在"本轮跑过该模型的 OpenCV 组合"时进 upstream——
    没跑过就没有那份文件,记一个指向不存在文件的指纹等于伪造证据。
    人工剔除名单(T-03-02 体例):谁剔了什么,每个产物的血缘里都可审计。
    """
    upstream: dict[str, Path] = {corners_npz.name: corners_npz}
    if init_dir is not None:
        upstream["init_result.json"] = init_dir / "result.json"
    lineage = make_lineage(config_files=cfg_files, upstream=upstream,
                           round_id=round_id)
    lineage["excluded_images"] = list(excluded_images)
    return lineage


def _run_opencv_combo(model, obs: ObservationSet, out_dir: Path,
                      corners_npz: Path, round_id: str, cfg_files: dict[str, Path],
                      excluded_images: tuple[str, ...]) -> tuple[dict, CalibrationResult]:
    """跑一个 OpenCV 组合并落盘,返回 (汇总行, 结果)。

    求解器内部的数值失败(cv2.error 等)已被 solve_with_opencv 转成失败契约;
    这里的兜底 except 接的是编排层意外(配置组合 wired 出错、读写失败等)——
    单组合任何异常都转 make_failure 落盘,绝不中断循环。
    """
    name = combo_dir_name(model.model_id, "opencv")
    combo_dir = out_dir / name
    lineage = _lineage_for(cfg_files, corners_npz, None, excluded_images, round_id)
    try:
        result = replace(solve_with_opencv(obs, model), lineage=lineage)
        if result.converged:
            save_result(combo_dir, result,
                        residuals=_residuals_for(obs, model, result))
        else:
            save_result(combo_dir, result)
    except Exception as e:  # noqa: BLE001  # 故意宽接:单组合任何异常都转失败契约落盘,绝不中断其余组合(计划明文)
        result = make_failure(
            model_id=model.model_id, solver="opencv",
            note=f"{model.model_id} 模型 OpenCV 组合未跑成:{type(e).__name__}——{e}",
            image_ids=obs.image_ids, image_size=obs.image_size,
            n_points_per_view=int(obs.object_points.shape[0]),
            solver_meta={"path": "orchestration_fallback"}, lineage=lineage)
        save_result(combo_dir, result)
    return _combo_row(name, model.model_id, "opencv", None, result), result


def _run_scipy_combo(model, obs: ObservationSet, scipy_cfg: ScipyConfig, loss: str,
                     out_dir: Path, corners_npz: Path, round_id: str,
                     cfg_files: dict[str, Path],
                     excluded_images: tuple[str, ...],
                     init_dir: Path | None,
                     rms_linear_ref: float | None) -> tuple[dict, CalibrationResult]:
    """跑一个 scipy 精化组合(单条损失核)并落盘,返回 (汇总行, 结果)。

    初值从同轮 <模型>__opencv 的落盘产物读回;该模型本轮没有(或未收敛的)
    OpenCV 结果时传 init=None 走通用初始化,solver_meta.init_source 记下
    实际用了哪条路,不静默。rms_linear_ref 是同模型 linear 核的回代 RMS
    (同轮先跑才有),供"实用收敛"判定做"鲁棒核没把解压歪"的对照——
    损失核清单里 linear 排前面是默认与推荐顺序。
    """
    name = combo_dir_name(model.model_id, "scipy", loss)
    combo_dir = out_dir / name
    lineage = _lineage_for(cfg_files, corners_npz, init_dir, excluded_images,
                           round_id)
    try:
        init = _load_init_result(init_dir, model.model_id) if init_dir else None
        result = solve_with_scipy(obs, model, scipy_cfg, init=init, loss=loss,
                                  rms_linear_ref=rms_linear_ref)
        if init is not None and result.converged:
            # init_from 升级为初值产物的相对路径(血缘一环:"数从哪份文件来");
            # 通用初始化路径保留 internal_generic 记号,诚实反映没有初值产物
            result = replace(result, init_from=f"{init_dir.name}/result.json")
        meta = dict(result.solver_meta)
        meta["init_source"] = "same_round_opencv" if init is not None \
            else "generic_default"
        result = replace(result, solver_meta=meta, lineage=lineage)
        if result.converged:
            save_result(combo_dir, result,
                        residuals=_residuals_for(obs, model, result))
        else:
            save_result(combo_dir, result)
    except Exception as e:  # noqa: BLE001  # 故意宽接:单组合任何异常都转失败契约落盘,绝不中断其余组合(计划明文)
        result = make_failure(
            model_id=model.model_id, solver="scipy",
            note=f"{model.model_id} 模型 scipy 精化(loss={loss})未跑成:"
                 f"{type(e).__name__}——{e}",
            image_ids=obs.image_ids, image_size=obs.image_size,
            n_points_per_view=int(obs.object_points.shape[0]),
            solver_meta={"path": "orchestration_fallback", "loss": loss},
            lineage=lineage)
        save_result(combo_dir, result)
    return _combo_row(name, model.model_id, "scipy", loss, result), result


def run_calibration_matrix(cfg: CalibrationConfig, board: BoardSpec,
                           obs: ObservationSet, out_dir: Path,
                           corners_npz: Path, round_id: str,
                           excluded_images: tuple[str, ...] = (),
                           progress: ProgressFn | None = None,
                           config_files: Mapping[str, Path] | None = None) -> list[dict]:
    """跑完配置矩阵的全部组合,逐组合三件套落盘,返回汇总行列表。

    参数:
        cfg:             全量配置(矩阵清单与 scipy 参数从 cfg.calibrate 读)
        board:           棋盘规格(只做一处装配核对:观测集角点数须与棋盘一致,
                         防"板子配错角点文件"的接线事故)
        obs:             观测集(split=train,剔除清单已生效)
        out_dir:         本轮运行目录 output/calibration/run_<时间戳>
        corners_npz:     本轮消费的角点文件(进每个组合的血缘 upstream)
        round_id:        本轮运行标识 run_<时间戳>
        excluded_images: 本轮实际命中的人工剔除名单(进血缘留痕)
        progress:        每完成一个组合回调一次(收汇总行;CLI 用它播报)
        config_files:    血缘要记指纹的配置文件;缺省用 config/ 相对路径约定

    组合顺序 = 按模型分组:每模型先 opencv(若在 solvers)再 scipy(若在
    solvers,对 scipy.losses 逐条),保证"精化前后"在汇总表里并排相邻;
    同模型 linear 核先跑出的 RMS 作为后续鲁棒核的对照参照(有才传)。
    """
    cv2.setNumThreads(cfg.cv2_threads)   # D-12:线程数从配置读,批量求解不占满机器
    expected_points = board.cols * board.rows
    if obs.object_points.shape[0] != expected_points:
        raise ValueError(
            f"观测集角点数({obs.object_points.shape[0]})与棋盘规格"
            f"({board.cols}x{board.rows} = {expected_points} 个内角点)对不上:"
            f"多半是 board.yaml 与角点文件不是同一套配置,请核对后重跑。"
        )

    cal = cfg.calibrate
    cfg_files = dict(config_files) if config_files else dict(_DEFAULT_CONFIG_FILES)
    out_dir = Path(out_dir)
    corners_npz = Path(corners_npz)

    comparison: list[dict] = []
    # 每模型 linear 核(同轮)的回代 RMS,给后续鲁棒核当"没压歪"的对照
    linear_rms: dict[str, float] = {}

    for model_id in cal.models:
        model = create_model(model_id)
        # 本轮跑过 OpenCV 组合才有初值产物目录可指(scipy 的血缘与初值都认它)
        init_dir = (out_dir / combo_dir_name(model_id, "opencv")
                    if "opencv" in cal.solvers else None)

        if "opencv" in cal.solvers:
            row, _result = _run_opencv_combo(
                model, obs, out_dir, corners_npz, round_id, cfg_files,
                excluded_images)
            comparison.append(row)
            if progress:
                progress(row)

        if "scipy" in cal.solvers:
            for loss in cal.scipy.losses:
                # linear 核自己不做"对照自己"的判定,参照只给鲁棒核路径
                ref = linear_rms.get(model_id) if loss != "linear" else None
                row, result = _run_scipy_combo(
                    model, obs, cal.scipy, loss, out_dir, corners_npz, round_id,
                    cfg_files, excluded_images, init_dir, ref)
                comparison.append(row)
                if progress:
                    progress(row)
                if loss == "linear" and result.converged and result.rms_overall is not None:
                    linear_rms[model_id] = float(result.rms_overall)

    return comparison
