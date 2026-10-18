"""OpenCV 求解后端:对任意 CameraModel 出一份"初值版"标定结果。

分派规则(本文件不 import 任何具体模型,只见 CameraModel 协议,D-09):
- 模型带 OpenCV 适配器(opencv_adapter() 非 None)→ 按适配器的 kind 分派:
  "calibrateCamera"(针孔族 M1–M3 走 cv2.calibrateCameraExtended,flags 由
  适配器自带:M1=128 固定 k3 / M2=0 / M3=16384 有理模型)或
  "fisheye"(M4 走 cv2.fisheye.calibrate,flags=10 = 每轮重算位姿 +
  skew 固定 0);
- 模型不带适配器(玩具模型等)→ 通用初始化兜底:default_params 组 K、
  逐视图 cv2.solvePnP 估位姿,拼成"初值版"结果——新模型只要实现协议
  就能被本后端使用,这就是注册机制零侵入(D-09)的底座。

与阶段一 baseline 的唯一偏离(Pitfall 9,白话):rms_overall / per_view_rms
不再用 OpenCV 返回的 RMS,一律由 compute_result_metrics 用模型层自己的
投影函数以 float64 回代重算——OpenCV 内部 RMS 基于 float32 角点(绑定层
只收 32 位浮点),两把尺子刻度差约 1e-4 像素,跨后端比较必须同一把尺;
OpenCV 的 RMS 只进 solver_meta(键 opencv_rms)留参考。

失败契约:求解抛 cv2.error(病态输入让方程组解不动、鱼眼对非鱼眼数据
报病态矩阵等)时返回 make_failure 结构——结构键全在、数值字段一律 None、
中文 failure_note,不抛异常、不中断其余组合。
"""

from __future__ import annotations

import cv2
import numpy as np

from calib.models.base import CameraModel
from calib.solve.observation import ObservationSet
from calib.solve.result import CalibrationResult, compute_result_metrics, make_failure

# 鱼眼迭代的停止条件:沿用阶段一 FisheyeConfig 的实测默认(迭代上限 100、
# 收敛容差 1e-6)。阶段二 M4 的 flags 由模型适配器自带,不再读配置段;
# 迭代上限属于 OpenCV 内部细节,固定用同一组默认即可复现阶段一行为。
_FISHEYE_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)


def _prepare_view(points: np.ndarray, n_points: int, dim: int, what: str) -> np.ndarray:
    """把单视图数组整成 OpenCV 要求的 (n,1,dim) float64,进求解器前先断言形状。

    体例照阶段一 baseline._prepare_view:fisheye 系函数对输入形状极挑剔,
    传 (n,2) 而非 (n,1,2) 会报难懂的错——在入口把形状查清楚,报错一眼看懂。
    """
    arr = np.asarray(points)
    if arr.ndim == 2 and arr.shape == (n_points, dim):
        arr = arr.reshape(n_points, 1, dim)
    if arr.shape != (n_points, 1, dim):
        raise AssertionError(
            f"{what} 形状不合:期望每视图 ({n_points},{dim}) 或 ({n_points},1,{dim}),"
            f"实际是 {arr.shape}。角点数组必须与棋盘的 {n_points} 个内角点一一对齐,"
            f"请检查角点文件的规范化顺序是否正确。"
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def solve_with_opencv(obs: ObservationSet, model: CameraModel) -> CalibrationResult:
    """OpenCV 后端入口:观测集 + 任一注册模型 → 统一 CalibrationResult。

    分派与兜底见模块 docstring;失败(cv2.error)走 make_failure 中文契约,
    solver_meta 里保留走到哪条路径(path)与当时的 flags(flags_used),
    排障时不用猜。
    """
    adapter = model.opencv_adapter()
    n_points = int(obs.object_points.shape[0])
    if adapter is None:
        path, flags_used = "generic_pnp", None
    else:
        path, flags_used = f"adapter:{adapter.kind}", int(adapter.flags)
    try:
        if adapter is None:
            return _solve_generic_pnp(obs, model)
        if adapter.kind == "calibrateCamera":
            return _solve_via_calibrate_camera(obs, model, adapter)
        if adapter.kind == "fisheye":
            return _solve_via_fisheye(obs, model, adapter)
        raise AssertionError(
            f"模型 {model.model_id} 的 OpenCV 适配器类型不认识:{adapter.kind!r}。"
            f"可选:calibrateCamera 或 fisheye;请检查该模型的适配器实现。"
        )
    except cv2.error as e:
        meta: dict = {"path": path}
        if flags_used is not None:
            meta["flags_used"] = flags_used
        return make_failure(
            model_id=model.model_id,
            solver="opencv",
            note=f"{model.model_id} 模型 OpenCV 求解未收敛:OpenCV 报错——{e}",
            image_ids=obs.image_ids,
            image_size=obs.image_size,
            n_points_per_view=n_points,
            solver_meta=meta,
        )


def _pack_success(obs: ObservationSet, model: CameraModel,
                  params: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray,
                  solver_meta: dict) -> CalibrationResult:
    """三条成功路径的公共收尾:同一把尺回代算 RMS,params 按 param_names 逐名给。"""
    rms_overall, per_view_rms, _residuals = compute_result_metrics(
        obs, params, rvecs, tvecs, model)
    return CalibrationResult(
        converged=True,
        failure_note=None,
        model_id=model.model_id,
        solver="opencv",
        solver_meta=solver_meta,
        params={name: float(v) for name, v in zip(model.param_names, params)},
        rvecs=rvecs,
        tvecs=tvecs,
        rms_overall=rms_overall,
        per_view_rms=per_view_rms,
        image_ids=obs.image_ids,
        image_size=obs.image_size,
        n_views=len(obs.image_ids),
        n_points_per_view=int(obs.object_points.shape[0]),
        init_from=None,
        lineage={},
    )


def _solve_via_calibrate_camera(obs: ObservationSet, model: CameraModel,
                                adapter) -> CalibrationResult:
    """针孔族路径(M1–M3):cv2.calibrateCameraExtended,flags 与 D 切片由适配器管。"""
    n = int(obs.object_points.shape[0])
    n_views = len(obs.image_ids)
    objs = [_prepare_view(obs.object_points, n, 3, "物点(针孔族)") for _ in range(n_views)]
    imgs = [_prepare_view(p, n, 2, "角点(针孔族)") for p in obs.image_points]

    # OpenCV 4.14 的 calibrateCamera 绑定只收 32 位浮点(Point2f/Point3f),
    # 传 64 位会报 "Unsupported format"——这里显式降精度再调用:像素坐标
    # 最大几千,float32 约有 1e-4 像素的表达精度,远小于 0.5px 的断言线。
    # 回代统一走 float64(同一把尺),这 1e-4px 的刻度差由 solver_meta 里
    # 的 opencv_rms 与回代 RMS 对账可见(Pitfall 9)。
    objs32 = [p.astype(np.float32) for p in objs]
    imgs32 = [p.astype(np.float32) for p in imgs]

    flags = int(adapter.flags)
    rms, K, D, rvecs, tvecs, _std_int, _std_ext, _per_view = cv2.calibrateCameraExtended(
        objs32, imgs32, tuple(obs.image_size), None, None, flags=flags)

    # 解包交适配器(D 长度随 flags 变:Pitfall 2——M1 取前 4、M2 全 5、M3 从
    # 14 长里取前 8),物点尺度/位姿解包照阶段一:float64 化、收 (V,3)
    params = adapter.params_from_cv(np.asarray(K, dtype=np.float64),
                                    np.asarray(D, dtype=np.float64).ravel())
    rvecs_arr = np.asarray([np.asarray(r, dtype=np.float64).ravel() for r in rvecs])
    tvecs_arr = np.asarray([np.asarray(t, dtype=np.float64).ravel() for t in tvecs])
    return _pack_success(
        obs, model, params, rvecs_arr, tvecs_arr,
        {"path": f"adapter:{adapter.kind}", "flags_used": flags,
         "opencv_rms": float(rms)})


def _solve_via_fisheye(obs: ObservationSet, model: CameraModel,
                       adapter) -> CalibrationResult:
    """鱼眼路径(M4):cv2.fisheye.calibrate,float64 与 (N,1,x) 形状(与针孔不同)。"""
    n = int(obs.object_points.shape[0])
    n_views = len(obs.image_ids)
    objs = [_prepare_view(obs.object_points, n, 3, "物点(鱼眼)") for _ in range(n_views)]
    imgs = [_prepare_view(p, n, 2, "角点(鱼眼)") for p in obs.image_points]

    flags = int(adapter.flags)
    # K 传单位阵、D 传零向量,交 OpenCV 内部初始化(主点置画面中心、焦距按
    # max(宽,高)/π 最小二乘起步)——阶段一实测形态;skew 由 flags 里的
    # FIX_SKEW 锁 0,与自实现模型的 8 参数定义两侧严格一致(Pitfall 7)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objs, imgs, tuple(obs.image_size),
        np.eye(3, dtype=np.float64), np.zeros(4, dtype=np.float64),
        flags=flags, criteria=_FISHEYE_CRITERIA)

    params = adapter.params_from_cv(np.asarray(K, dtype=np.float64),
                                    np.asarray(D, dtype=np.float64).ravel())
    rvecs_arr = np.asarray([np.asarray(r, dtype=np.float64).ravel() for r in rvecs])
    tvecs_arr = np.asarray([np.asarray(t, dtype=np.float64).ravel() for t in tvecs])
    return _pack_success(
        obs, model, params, rvecs_arr, tvecs_arr,
        {"path": f"adapter:{adapter.kind}", "flags_used": flags,
         "opencv_rms": float(rms)})


def _solve_generic_pnp(obs: ObservationSet, model: CameraModel) -> CalibrationResult:
    """通用初始化兜底(D-09):模型无 OpenCV 适配器时的"初值版"结果。

    做法:default_params(image_size) 组 K(各模型自己的初值规则,畸变清零);
    逐视图 cv2.solvePnP 估位姿(物点/角点 float64, solvePnP 接受 float64)。
    畸变系数传**空向量** = 无畸变——与 default_params 的"畸变清零"严格一致,
    且不依赖 OpenCV 的畸变向量布局(4/5/8 长度那套是针孔族的私有约定,
    任意参数个数的玩具模型不该被它绑架)。内参不动、只估位姿,所以这是
    "初值版"结果:给 scipy 精化路径(02-04)与新模型测试兜底用的。
    """
    n = int(obs.object_points.shape[0])
    n_views = len(obs.image_ids)
    params = np.asarray(model.default_params(obs.image_size), dtype=np.float64).ravel()
    K = np.asarray(model.intrinsics_matrix(params), dtype=np.float64)
    objp = obs.object_points.reshape(n, 1, 3)
    zero_dist = np.zeros(0, dtype=np.float64)  # 空向量 = 无畸变(初值畸变全 0)

    rvecs = np.empty((n_views, 3), dtype=np.float64)
    tvecs = np.empty((n_views, 3), dtype=np.float64)
    for i in range(n_views):
        img = _prepare_view(obs.image_points[i], n, 2, "角点(通用初始化)")
        _ok, rvec, tvec = cv2.solvePnP(objp, img, K, zero_dist)
        rvecs[i] = np.asarray(rvec, dtype=np.float64).ravel()
        tvecs[i] = np.asarray(tvec, dtype=np.float64).ravel()
    return _pack_success(obs, model, params, rvecs, tvecs, {"path": "generic_pnp"})
