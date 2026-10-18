"""逐帧回代评分与只估位姿小求解:held-out 评分的地基(D-04 复核的引擎)。

白话:标定把内参算出来之后,"一张没参与标定的照片拍得准不准"怎么量?
把标定得到的内参**当尺子用**,只挪棋盘的位姿(6 个数:3 个旋转 + 3 个
平移),让棋盘投影到画面上的位置尽量贴近检测到的角点——挪完之后还剩下
的偏差就是这张照片的重投影误差。内参固定、只估位姿,解的是个 6 参数的
小最小二乘,每张毫秒级。任何注册模型都适用(玩具模型也不例外,D-09):
投影只走 model.project,给 solvePnP 的初值 K 只走 model.intrinsics_matrix,
本文件不出现任何具体模型的名字,也不出现 OpenCV 的投影函数。

两个入口:
- score_views:整批回代评分——位姿已知(train 帧标定时解过)时逐帧算
  RMS。实现直接委托 compute_result_metrics(同一把尺只此一份,严禁分叉);
- solve_pose_only:单视角只估位姿小求解——位姿未知(test 帧没参与标定)
  时先把位姿估出来再算 RMS。初值用 cv2.solvePnP(线性方法,模型无关:
  只要拿得到标准 K 就能调),精化用 scipy 的 least_squares。

失败语义:输入形状不对当场中文报错(不带病进求解);solvePnP 解不出
(点太少或退化)抛中文 ValueError,调用方决定单帧记 NaN 还是中断。
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from calib.models.base import CameraModel, as_params, transform_to_camera
from calib.solve.observation import ObservationSet
from calib.solve.result import compute_result_metrics

# solvePnP 迭代解法要求的最少点数(4 点以下方程组欠定,硬算必错)
_MIN_PNP_POINTS = 4


def score_views(obs: ObservationSet,
                params: np.ndarray,
                rvecs: np.ndarray,
                tvecs: np.ndarray,
                model: CameraModel) -> np.ndarray:
    """整批回代评分:逐帧重投影 RMS,形状 (V,),单位像素。

    与 result.compute_result_metrics 是**同一把尺**(本函数直接委托它,
    不存在第二份实现——行为测试锁定逐位一致):残差 =
    model.project(物点经 transform_to_camera 变换到相机坐标系) − 实测
    角点,逐帧 RMS = sqrt(该帧全部角点平方欧氏残差的平均)。

    参数:
        obs:    观测集(物点网格 + 实测角点,行序契约见 observation.py)
        params: (P,) 内参参数向量(按 model.param_names 布局)
        rvecs / tvecs: (V,3) 逐视图位姿,与 obs 的视图逐行对齐
        model:  任一注册的 CameraModel 实例
    """
    _rms_overall, per_view_rms, _residuals = compute_result_metrics(
        obs, params, rvecs, tvecs, model)
    return per_view_rms


def solve_pose_only(uv_view: np.ndarray,
                    object_points: np.ndarray,
                    params: np.ndarray,
                    model: CameraModel) -> tuple[np.ndarray, np.ndarray, float]:
    """内参当尺子用、只挪棋盘的位姿,给没参与标定的照片打分。

    白话:给定一张照片的 88 个检测角点、棋盘物点网格、标定得到的内参,
    求这张照片里棋盘摆在哪(6 个位姿数),再看贴得有多紧(剩余 RMS)。
    内参全程固定不动——它是"尺子",不是被量对象。

    步骤(两段,全部只经 CameraModel 接口):
    1. 初值 = cv2.solvePnP(物点, 角点, model.intrinsics_matrix(params),
       零畸变)——线性方法,模型无关的通用钥匙;初值天然带偏差
       (PnP 不知道畸变),正好当"扰动的初值"喂给下一段;
    2. 精化 = scipy least_squares(6 参数,残差仅
       model.project(transform_to_camera(物点, rvec, tvec)) − 角点,
       method="trf",x_scale="jac")。

    参数:
        uv_view:       (P,2) 单视角实测角点(像素),P ≥ 4;
        object_points: (P,3) 物点网格(与 uv_view 逐行对齐,Z=0 棋盘平面);
        params:        (P,) 标定得到的内参参数向量(固定不动);
        model:         任一注册的 CameraModel 实例。

    返回:(rvec (3,), tvec (3,), rms float)——精化后的位姿与回代 RMS(像素)。
    """
    uv = np.asarray(uv_view, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise AssertionError(
            f"uv_view 形状不合:期望 (P,2) 单视角实测角点,实际是 {uv.shape}。"
            f"只估位姿小求解一次处理一张照片,请检查调用方传入的数据排布。"
        )
    objp = np.asarray(object_points, dtype=np.float64)
    if objp.shape != (uv.shape[0], 3):
        raise AssertionError(
            f"object_points 形状不合:期望 ({uv.shape[0]},3) 与角点逐行对齐,"
            f"实际是 {objp.shape}。物点网格与检测角点必须一一对齐,请检查行序。"
        )
    if uv.shape[0] < _MIN_PNP_POINTS:
        raise ValueError(
            f"角点数 {uv.shape[0]} 少于位姿求解下限 {_MIN_PNP_POINTS} 个:"
            f"方程组欠定,位姿解不出来,请检查该帧的角点检出质量。"
        )
    p = as_params(params, model.model_id, model.param_names)

    # ---- 初值:solvePnP(K 由模型组出、畸变按零处理)----
    ok, rvec0, tvec0 = cv2.solvePnP(
        objp, uv, model.intrinsics_matrix(p), np.zeros(5, dtype=np.float64))
    if not ok:
        raise ValueError(
            f"模型 {model.model_id} 的位姿初值求解失败(solvePnP 解不出):"
            f"该帧角点可能退化(近共线/分布过窄),复核时按单帧失败记录,不影响其余帧。"
        )

    # ---- 精化:6 参数最小二乘,内参加锁 ----
    def residual(theta: np.ndarray) -> np.ndarray:
        pts_cam, _pose_block = transform_to_camera(objp, theta[:3], theta[3:])
        return (model.project(pts_cam, p) - uv).ravel()

    theta0 = np.concatenate([np.asarray(rvec0, dtype=np.float64).ravel(),
                             np.asarray(tvec0, dtype=np.float64).ravel()])
    sol = least_squares(residual, theta0, method="trf", x_scale="jac")

    rvec = sol.x[:3].copy()
    tvec = sol.x[3:].copy()
    rms = float(np.sqrt(np.mean(np.sum(sol.fun.reshape(-1, 2) ** 2, axis=1))))
    return rvec, tvec, rms
