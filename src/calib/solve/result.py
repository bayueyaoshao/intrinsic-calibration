"""标定结果的统一结构与落盘三件套:双求解后端(OpenCV / scipy)共用的输出侧。

白话背景:两个后端解出来的东西形态不同(OpenCV 给 K 矩阵 + D 向量,
scipy 给一个参数向量),下游(对比表、存疑帧复核、报告)不该关心这些
差别。CalibrationResult 把"一次求解的全部结论"装进一个统一结构;落盘
三件套指:

- result.json:   结构化结论(参数按名字给、血缘、逐帧 RMS;失败时数值为 null);
- extrinsics.npz:逐视图位姿(旋转向量/平移向量),给阶段三的线性化点与
                 阶段四复评用;
- residuals.npz: 逐帧 (V, 88, 2) 残差与逐帧 RMS,误差诊断与存疑复核用。

同一把尺(研究 Pitfall 9,跨后端比较的地基):OpenCV 内部的 RMS 基于
float32 角点算(它的 Python 绑定只收 32 位浮点),与本项目用 float64
角点、经模型层投影函数回代重算的 RMS 刻度差约 1e-4 像素——各用各的尺,
跨后端比较就会被精度噪声污染。所以 CalibrationResult 的 rms_overall /
per_view_rms 一律由 compute_result_metrics 用 models.project(float64)
回代计算,OpenCV 返回的 RMS 只进 solver_meta 留参考。

失败契约(与阶段一 save_calibration 同构):converged=False 时结构键
一个不少、数值字段一律 None(落盘成 JSON null)、failure_note 非空中文
——上游读取逻辑一视同仁,不必先判断成败再挑键。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calib.models.base import CameraModel, transform_to_camera
from calib.solve.observation import ObservationSet


@dataclass
class CalibrationResult:
    """一次标定求解的统一结果(双后端同构;失败契约见模块 docstring)。

    字段语义:
        converged:            是否收敛;False 时标"(失败时 None)"的字段一律 None
        failure_note:         未收敛时的中文失败说明(收敛时为 None)
        model_id / solver:    模型短名("m1".."m4")与求解路径("opencv" / "scipy";
                              OpenCV 初值 → scipy 精化的路径记 "scipy")
        solver_meta:          后端专属信息。OpenCV:path(适配器分派或通用初底)、
                              flags_used、opencv_rms(内部 RMS,仅供参考);
                              scipy:method/loss/f_scale/status/nfev/njev/cost/
                              optimality 与实用收敛依据——判定依据全在这里,不静默
        params:               按 param_names 逐名给出的内参字典,下游按名取、
                              不猜位置(失败时是空字典 {} 占位,不是 None)
        rvecs / tvecs:        逐视图位姿 (V,3) float64(失败时 None)
        rms_overall:          整体 RMS,像素(models.project 回代;失败时 None)
        per_view_rms:         逐帧 RMS (V,)(同一把尺;失败时 None)
        image_ids:            视图编号;行序契约:与观测集逐行一致
        image_size:           (宽, 高)
        n_views / n_points_per_view: 视图数与每视图角点数
        init_from:            scipy 精化指向的 OpenCV 初值产物路径(血缘的一环;
                              OpenCV 直接求解时为 None)
        lineage:              make_lineage 血缘字典(由调用方组装后传入)
    """

    converged: bool
    failure_note: str | None
    model_id: str
    solver: str
    solver_meta: dict
    params: dict[str, float]
    rvecs: np.ndarray | None
    tvecs: np.ndarray | None
    rms_overall: float | None
    per_view_rms: np.ndarray | None
    image_ids: tuple[str, ...]
    image_size: tuple[int, int]
    n_views: int
    n_points_per_view: int
    init_from: str | None
    lineage: dict


def make_failure(model_id: str, solver: str, note: str,
                 image_ids: tuple[str, ...], image_size: tuple[int, int],
                 n_points_per_view: int,
                 solver_meta: dict | None = None,
                 lineage: dict | None = None) -> CalibrationResult:
    """失败契约的工厂:数值字段一律 None,结构键全在。

    上游对成功/失败结果的读取逻辑一视同仁(键一个不少)——失败只是
    "数值还没算出来",不是换一种结构。params 用空字典 {} 占位(结构键在、
    无任何数值);rvecs / tvecs / rms_overall / per_view_rms / init_from
    一律 None;failure_note 必须是非空中文,写清楚"哪个模型、什么原因"。
    """
    ids = tuple(image_ids)
    return CalibrationResult(
        converged=False,
        failure_note=str(note),
        model_id=model_id,
        solver=solver,
        solver_meta=dict(solver_meta) if solver_meta else {},
        params={},
        rvecs=None,
        tvecs=None,
        rms_overall=None,
        per_view_rms=None,
        image_ids=ids,
        image_size=(int(image_size[0]), int(image_size[1])),
        n_views=len(ids),
        n_points_per_view=int(n_points_per_view),
        init_from=None,
        lineage=dict(lineage) if lineage else {},
    )


def compute_result_metrics(obs: ObservationSet,
                           params: np.ndarray,
                           rvecs: np.ndarray,
                           tvecs: np.ndarray,
                           model: CameraModel) -> tuple[float, np.ndarray, np.ndarray]:
    """统一回代尺:用模型层自己的投影函数重算残差与 RMS(全 float64)。

    残差 = model.project(物点经 transform_to_camera 变换到相机坐标系)
    − 实测角点;逐帧 RMS = sqrt(该帧全部角点"平方欧氏残差"的平均);
    整体 RMS = sqrt(全部视图全部角点平方残差的平均)。

    为什么不用求解器自带的 RMS(Pitfall 9,白话):OpenCV 的整体 RMS 是
    基于 float32 角点算的(绑定层限制),与 float64 回代的刻度差约
    1e-4 像素——两把尺子各量各的,跨后端比较就会被精度噪声污染,所以
    rms_overall / per_view_rms 一律由本函数算,OpenCV 的 RMS 只进
    solver_meta 留参考。本函数只经 model.project 与 base.transform_to_camera
    (单一数学源 D-06),不出现任何 cv2 投影调用。

    返回:(rms_overall float, per_view_rms (V,), residuals (V,P,2) float64)。
    """
    p = np.asarray(params, dtype=np.float64).ravel()
    rvecs_arr = np.asarray(rvecs, dtype=np.float64)
    tvecs_arr = np.asarray(tvecs, dtype=np.float64)
    n_views = len(obs.image_ids)
    n_points = obs.object_points.shape[0]
    if rvecs_arr.shape != (n_views, 3) or tvecs_arr.shape != (n_views, 3):
        raise AssertionError(
            f"回代位姿形状不合:期望 rvecs/tvecs 各 ({n_views},3),"
            f"实际是 {rvecs_arr.shape} / {tvecs_arr.shape}。"
            f"位姿必须与观测集的 {n_views} 个视图逐行对齐(行序契约)。"
        )

    residuals = np.empty((n_views, n_points, 2), dtype=np.float64)
    for i in range(n_views):
        pts_cam, _pose_deriv = transform_to_camera(
            obs.object_points, rvecs_arr[i], tvecs_arr[i])
        residuals[i] = model.project(pts_cam, p) - obs.image_points[i]

    squared = np.sum(residuals * residuals, axis=2)   # 每角点的平方欧氏残差 (V,P)
    per_view_rms = np.sqrt(squared.mean(axis=1))
    rms_overall = float(np.sqrt(squared.mean()))
    return rms_overall, per_view_rms, residuals


def save_result(out_dir: Path | str, result: CalibrationResult,
                residuals: np.ndarray | None = None) -> Path:
    """把一个求解结果落盘成三件套(体例照阶段一 save_calibration)。

    result.json   必写(失败组合照样落盘:数值键全 null、结构键一个不少);
    extrinsics.npz / residuals.npz 只在收敛时写——没有位姿与残差可写时
    硬写只会撒谎。收敛结果必须同时给 residuals (V,P,2) 数组(三件套是
    一体的,缺残差直接中文报错,不静默漏写)。

    npz 只放纯数值数组与定长 unicode 的编号列(image_ids),读写两侧都
    强制 allow_pickle=False(pickle 模式的 npz 可执行任意代码,关死)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.converged:
        if residuals is None:
            raise ValueError(
                f"收敛结果({result.model_id}__{result.solver})落盘缺残差:"
                f"save_result 需要 residuals (V,P,2) 数组一并写 residuals.npz"
                f"(三件套一体,残差由 compute_result_metrics 回代给出)。"
            )
        residuals_arr = np.asarray(residuals, dtype=np.float64)
        if residuals_arr.shape != (result.n_views, result.n_points_per_view, 2):
            raise AssertionError(
                f"残差数组形状不合:期望 ({result.n_views},{result.n_points_per_view},2)"
                f"(与结果的视图数/角点数对齐),实际是 {residuals_arr.shape}。"
                f"请确认残差与结果来自同一次求解。"
            )
        per_view_dict = {image_id: float(rms) for image_id, rms
                         in zip(result.image_ids, result.per_view_rms)}
        payload = {"rms_overall": float(result.rms_overall),
                   "per_view_rms": per_view_dict}
    else:
        payload = {"rms_overall": None, "per_view_rms": None}

    document = {
        "lineage": result.lineage,
        "model_id": result.model_id,
        "solver": result.solver,
        "solver_meta": result.solver_meta,
        "params": result.params,
        "image_ids": list(result.image_ids),
        "image_size": [int(result.image_size[0]), int(result.image_size[1])],
        "n_views": int(result.n_views),
        "n_points_per_view": int(result.n_points_per_view),
        "init_from": result.init_from,
        "converged": bool(result.converged),
        "failure_note": result.failure_note,
        **payload,
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    if result.converged:
        ids = np.asarray(result.image_ids)   # 元组的 str → 定长 unicode 数组
        np.savez(out_dir / "extrinsics.npz",
                 image_ids=ids,
                 rvecs=np.asarray(result.rvecs, dtype=np.float64),
                 tvecs=np.asarray(result.tvecs, dtype=np.float64))
        np.savez(out_dir / "residuals.npz",
                 image_ids=ids,
                 residuals=residuals_arr,
                 per_view_rms=np.asarray(result.per_view_rms, dtype=np.float64))
    return out_dir


def load_result(out_dir: Path | str) -> dict:
    """读回 result.json(npz 不在本函数职责内,另见 load_extrinsics/load_residuals)。"""
    with open(Path(out_dir) / "result.json", encoding="utf-8") as f:
        return json.load(f)


def load_extrinsics(out_dir: Path | str) -> dict:
    """读 extrinsics.npz:返回 {image_ids (V,) 元组, rvecs (V,3), tvecs (V,3)}。

    强制 allow_pickle=False:npz 的 pickle 载荷可以执行任意代码,
    遇到直接抛错,绝不执行(阶段一 load_corners 的防线,此处延续)。
    """
    with np.load(Path(out_dir) / "extrinsics.npz", allow_pickle=False) as data:
        return {
            "image_ids": tuple(str(x) for x in np.asarray(data["image_ids"]).ravel()),
            "rvecs": np.asarray(data["rvecs"], dtype=np.float64),
            "tvecs": np.asarray(data["tvecs"], dtype=np.float64),
        }


def load_residuals(out_dir: Path | str) -> dict:
    """读 residuals.npz:返回 {image_ids (V,) 元组, residuals (V,P,2), per_view_rms (V,)}。

    防线与 load_extrinsics 相同(allow_pickle=False)。
    """
    with np.load(Path(out_dir) / "residuals.npz", allow_pickle=False) as data:
        return {
            "image_ids": tuple(str(x) for x in np.asarray(data["image_ids"]).ravel()),
            "residuals": np.asarray(data["residuals"], dtype=np.float64),
            "per_view_rms": np.asarray(data["per_view_rms"], dtype=np.float64),
        }
