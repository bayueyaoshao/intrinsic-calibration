"""模型层地基:CameraModel 抽象基类 + 两个共享导数工具。

四类候选模型(M1/M2/M3 针孔 Brown-Conrady、M4 鱼眼 KB4)全部继承这里的
CameraModel——投影、反投影、雅可比的**公式**各模型自己写(模型间数学确实
不同),但**接口形状**在本文件一次定死:输入输出全部 float64、数组形状
在 docstring 里逐一声明、非法输入抛中文 AssertionError(绝不返回半成品)。

为什么接口要统一到这个程度:理论界(Fisher 信息/CRLB,阶段三)与实际
求解(双求解后端,阶段二后半)都必须 import 这同一份投影与雅可比——
接口统一了,"理论算的"和"求解用的"才严格是同一套数学,结果才可比
(项目根决策 D-06:models/ 是全项目唯一数学源)。

名词白话:
- 投影(project):已知相机内参与一个三维点在相机坐标系里的位置,
  算它落在照片上的像素位置;
- 反投影(unproject):投影的逆运算——像素位置反推"这个像素看向哪个
  方向"的三维射线方向;畸变是非线性的,逆解用牛顿迭代(牛顿迭代 =
  反复用当前猜测处的斜率修正猜测,几步就收敛到精确解);
- 雅可比(project_jacobian):投影结果对参数的偏导数矩阵——优化器与
  理论界都要靠它知道"参数动一点,像素动多少"。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class CameraModel(ABC):
    """相机模型的统一抽象:四类模型实现同一组方法,求解器/理论界不分叉。

    类属性(子类必须覆写):
        model_id:     短名,如 "m1".."m4"(与全部规划文档一致);
        description:  中文名 + 一句白话(报告/CLI 汇总直接用);
        param_names:  参数向量布局,如 ("fx","fy","cx","cy","k1",...)——
                      参数在向量里的位置由此定死,任何代码不得按位置猜参数。
    """

    model_id: str
    description: str
    param_names: tuple[str, ...]

    @abstractmethod
    def project(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        """投影:相机坐标系三维点 → 像素坐标。

        pts_cam: (N,3) float64,相机坐标系点(Z 轴朝前,Z>0 才有意义);
        params:  (P,) float64,P = len(param_names),布局见子类;
        返回:    (N,2) float64 像素坐标 (u,v)。
        """

    @abstractmethod
    def project_jacobian(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        """投影对**内参**的解析雅可比(只含内参列,外参链式见 transform_to_camera)。

        pts_cam: (N,3) float64;params: (P,) float64;
        返回:    (N,2,P) float64——第 (i,2*k) 元是第 i 点的投影坐标之一
                 对第 k 个内参的偏导数。手写解析式,对数值差分基准
                 逐列相对误差 < 1e-6(测试锁定,阶段二 02-01 Task 3)。
        """

    @abstractmethod
    def project_point_jacobian(self, pts_cam: np.ndarray,
                               params: np.ndarray) -> np.ndarray:
        """投影对**相机坐标系点**的解析雅可比(外参链式的前半截)。

        pts_cam: (N,3) float64;params: (P,) float64;
        返回:    (N,2,3) float64——第 (i,k,j) 元是第 i 点的投影坐标 k
                 对相机坐标分量 j 的偏导数。

        为什么放模型层(02-04 Rule 2 补充的接口):残差对位姿的导数 =
        d像素/d相机点(本方法,各模型公式不同)× ∂相机点/∂外参
        (base.transform_to_camera,全模型共享)。前半截若不进模型层,
        求解侧就只剩数值差分一条路——研究 Don't Hand-Roll 明令禁止,
        且会把第二份数学搬进 solve/,破坏 D-06 唯一数学源。

        硬契约:光轴附近的奇异点(如 KB4 的 r→0)必须用解析极限顶上,
        绝不返回 NaN/Inf(0/0 会毒化整个求解,Pitfall 4,测试锁定)。
        """

    @abstractmethod
    def unproject(self, uv: np.ndarray, params: np.ndarray) -> np.ndarray:
        """反投影:像素坐标 → 归一化射线方向(真逆解,牛顿迭代,无占位)。

        uv:     (N,2) float64 像素坐标;
        params: (P,) float64;
        返回:   (N,3) float64 单位向量——该像素发出的视线方向(模长为 1)。

        往返契约(测试锁定):project → unproject → project 的二次投影与
        首次逐点差 < 1e-9 像素。把返回的射线直接喂回 project 即可验证:
        project 对射线方向不敏感于缩放,归一化后投影值不变。
        """

    @abstractmethod
    def default_params(self, image_size: tuple[int, int]) -> np.ndarray:
        """按图像尺寸给一组靠谱初值:主点在画面中心、焦距取经验值、畸变清零。

        image_size: (宽, 高) 像素;
        返回:       (P,) float64,全部有限值且焦距 > 0。
        """

    @abstractmethod
    def intrinsics_matrix(self, params: np.ndarray) -> np.ndarray:
        """参数向量 → 标准 3×3 内参矩阵 K([[fx,0,cx],[0,fy,cy],[0,0,1]])。

        所有模型族都给得出标准 K(skew 倾斜因子一律 0):后续"只估位姿、
        内参固定"的小求解(D-04 存疑帧回代、阶段四 held-out 评分)拿到
        K 就能调 OpenCV 的位姿求解,不必知道模型细节。
        """

    def opencv_adapter(self):
        """OpenCV 适配器(可选):返回 None 表示本模型不提供。

        适配器把"参数叫什么、在 cv2 数组里排哪、flags 是多少"封装成
        唯一入口(kind / flags / params_to_cv / params_from_cv),
        OpenCV 求解后端只跟适配器打交道。**玩具模型不覆写本方法**
        → 求解后端自动走通用初始化路径(default_params 组 K + 逐视图
        solvePnP 估位姿),注册一处即可被两个求解器使用(D-09)。
        """
        return


def as_points3d(pts: np.ndarray, what: str = "pts_cam") -> np.ndarray:
    """校验并统一三维点数组为 (N,3) float64;形状不合抛中文 AssertionError。

    why: 模型层所有函数的形状契约第一道闸——错误在入口点名,不带病进公式。
    """
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise AssertionError(
            f"{what} 形状不合:期望 (N,3),实际是 {np.asarray(pts).shape}。"
            f"相机坐标系三维点必须逐点给 (x,y,z) 三个数,请检查调用方的数据排布。"
        )
    return arr


def as_params(params: np.ndarray, model_id: str, param_names: tuple[str, ...]) -> np.ndarray:
    """校验参数向量长度并统一为 (P,) float64;不合抛中文 AssertionError。"""
    p = np.asarray(params, dtype=np.float64).ravel()
    expected = len(param_names)
    if p.shape != (expected,):
        raise AssertionError(
            f"模型 {model_id} 的参数向量形状不合:期望 ({expected},)"
            f"(布局 {param_names}),实际拍平后是 {p.shape}。"
            f"请按 param_names 的顺序传参,不要按位置猜。"
        )
    return p


def transform_to_camera(pts_board: np.ndarray,
                        rvec: np.ndarray,
                        tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """棋盘平面点 → 相机坐标系点,同时给出对 6 个外参的解析导数块。

    这是"外参链式"的共享工具:求解残差对位姿(ω,t)的雅可比 =
    d像素/d相机坐标点 × d相机坐标点/d外参,后半截全模型通用,写在这里一份。

    pts_board: (N,3) float64,棋盘平面上的物点(齐次变换前的坐标);
    rvec:      (3,) float64 旋转向量(Rodrigues 语义:方向=转轴、模长=转角,
               与 OpenCV 全兼容,求解器直接互通);
    tvec:      (3,) float64 平移向量;
    返回:      (pts_cam (N,3) float64, dXc_d_pose (N,3,6) float64)——
               dXc_d_pose 的 6 列依序是 ∂Xc/∂ω(3 列)与 ∂Xc/∂t(3 列)。

    公式:Xc = R(ω)·Xb + t,R 由 cv2.Rodrigues 算出;∂Xc/∂ω 直接用同一
    次调用顺手返回的雅可比(布局 rot_jac[k, 3i+j] = ∂R_ij/∂ω_k,实测对
    中心差分吻合到 ~1e-10,任意大小转角都成立):
        ∂Xc_i/∂ω_k = Σ_j ∂R_ij/∂ω_k · Xb_j(平移不随 ω 动,逐点 einsum 一次算完)
        ∂Xc/∂t  =  I(平移逐分量直接搬)

    历史教训(02-04 修复,Rule 1):早先版本写 ∂Xc/∂ω = −[Xc]×——那只是
    ω=0 处的极限形式,一般转角下不对(0.1 rad 的小转角就已偏 70%)。
    手推旋转导数是研究里点名的坑,此处一律用 cv2 同源的现成雅可比。
    """
    pts = as_points3d(pts_board, "pts_board")
    r = np.asarray(rvec, dtype=np.float64).ravel()
    t = np.asarray(tvec, dtype=np.float64).ravel()
    if r.shape != (3,) or t.shape != (3,):
        raise AssertionError(
            f"rvec/tvec 形状不合:期望各 (3,),实际是 {r.shape} / {t.shape}。"
            f"旋转向量与平移向量必须各给 3 个数(Rodrigues 语义)。"
        )
    rot, rot_jac = cv2.Rodrigues(r)  # 旋转不自己推公式:cv2 现成,语义与 OpenCV 全兼容
    pts_cam = pts @ rot.T + t

    dXc_d_pose = np.empty((pts.shape[0], 3, 6), dtype=np.float64)
    dXc_d_pose[:, :, 0:3] = np.einsum("kij,nj->nik",
                                      rot_jac.reshape(3, 3, 3), pts)
    dXc_d_pose[:, :, 3:6] = np.eye(3, dtype=np.float64)     # ∂Xc/∂t = I
    return pts_cam, dXc_d_pose
