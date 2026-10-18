"""鱼眼等距模型 M4(KB4,OpenCV 的 cv2.fisheye 同族):8 参,无 skew。

白话定义:鱼眼镜头不满足"直线过小孔成像"的针孔假设,光线是按**入射角**
在像面上拉开的。KB4 的规则:三维点相对光轴的入射角 θ(白话:视线与
镜头正前方夹角,arctan2 算出)先过一个四系数多项式"弯折"成畸变角
θd = θ·(1+k1θ²+k2θ⁴+k3θ⁶+k4θ⁸),再把半角 θd 当圆心角展开到像面:
像素半径 = 焦距 × θd,方向保持不变。"等距"指 k 全 0 时像面半径与
入射角成正比(角度均匀拉开)。

无 skew(倾斜因子,成像平面与镜头不垂直的偏离)是**实测定夺**:
cv2.fisheye.projectPoints 完全忽略 K[0][1](skew=0.5 的夸张值下投影
逐位不变)——带 skew 参数而投影不用它,参数与数学就脱节了,所以 M4
两侧(自实现与 OpenCV 适配器)都定死 α=0,共 8 个参数。

形状与 dtype 硬规则(适配器 docstring 再强调):fisheye 系函数要求
(N,1,x) 形状与 float64;本模块数学函数一律 (N,·) float64 进出。
"""

from __future__ import annotations

import cv2
import numpy as np

from calib.models import register_model
from calib.models.base import CameraModel, as_params, as_points3d

# 近轴保护线:归一化半径 r 小于它时,θd/r 的 0/0 用解析极限 1/z 顶上
# (θ≈r/z,故 x·θd/r → x/z;极限值实测推导,不猜)。
_NEAR_AXIS_EPS = 1e-12


@register_model
class FisheyeKB4(CameraModel):
    """M4:鱼眼等距 KB4(fx,fy,cx,cy,k1..k4,共 8 参,无 skew)。"""

    model_id = "m4"
    description = "鱼眼等距 KB4(k1..k4;skew 实测为死参数,定死 0)"
    param_names = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")

    def project(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        """投影:(N,3) 相机坐标系点 + (P,) 参数 → (N,2) 像素,float64。

        θ = arctan2(r,z) 对任意 z 符号都有定义(鱼眼视场可超 90°),
        近轴 r<1e-12 时用解析极限防除零(Pitfall:0/0 会出 NaN 毒化求解)。
        """
        pts = as_points3d(pts_cam)
        p = as_params(params, self.model_id, self.param_names)
        fx, fy, cx, cy = p[:4]
        k = p[4:]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        t2 = theta * theta
        # θd 多项式:θ·(1+k1θ²+k2θ⁴+k3θ⁶+k4θ⁸)
        theta_d = theta * (1.0 + k[0] * t2 + k[1] * t2**2
                           + k[2] * t2**3 + k[3] * t2**4)
        # 近轴保护:r→0 时 θd/r → 1/z(θ≈r/z),直接除会得 0/0=NaN;
        # np.divide 的 where 只在安全处做除法,不做"先全除再挑"(那会先产生 NaN)
        scale = np.divide(theta_d, r, out=1.0 / z, where=r >= _NEAR_AXIS_EPS)
        return np.stack([fx * x * scale + cx, fy * y * scale + cy], axis=1)

    def project_jacobian(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        """投影对内参的解析雅可比:(N,3)+(P,) → (N,2,P),float64,只含内参列。

        像素 = 焦距 ×(方向单位化坐标)× θd + 主点,链式:
          ∂u/∂fx = x·θd/r(近轴 → x/z),∂u/∂kj = fx·x·θ^(2j+1)/r(j 从 1 数:
          k1 配 θ³、k4 配 θ⁹;近轴 → 0,因为 θ 的奇次幂趋 0 的速度远快于
          x/r 的有界性)。
        """
        pts = as_points3d(pts_cam)
        p = as_params(params, self.model_id, self.param_names)
        fx, fy = p[0], p[1]
        k = p[4:]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        t2 = theta * theta
        theta_d = theta * (1.0 + k[0] * t2 + k[1] * t2**2
                           + k[2] * t2**3 + k[3] * t2**4)

        near = r < _NEAR_AXIS_EPS
        safe = ~near  # 安全区:r 足够大,除法不遇 0/0
        su = np.divide(x * theta_d, r, out=x / z, where=safe)  # ∂u/∂fx 原料(近轴 → x/z)
        sv = np.divide(y * theta_d, r, out=y / z, where=safe)  # ∂v/∂fy 原料(近轴 → y/z)

        n = pts.shape[0]
        jac = np.zeros((n, 2, len(self.param_names)), dtype=np.float64)
        jac[:, 0, 0] = su
        jac[:, 1, 1] = sv
        jac[:, 0, 2] = 1.0
        jac[:, 1, 3] = 1.0
        # ∂θd/∂kj = θ^(2j+1)(j 从 1 数:k1 配 θ³、k4 配 θ⁹;多项式逐项求导);
        # 近轴极限为 0(θ 的奇次幂趋 0 远快于 x/r 的有界性)
        for j in range(4):
            theta_odd = theta ** (2 * j + 3)
            jac[:, 0, 4 + j] = np.divide(fx * x * theta_odd, r,
                                         out=np.zeros(n), where=safe)
            jac[:, 1, 4 + j] = np.divide(fy * y * theta_odd, r,
                                         out=np.zeros(n), where=safe)
        return jac

    def unproject(self, uv: np.ndarray, params: np.ndarray) -> np.ndarray:
        """反投影:像素 → 单位射线方向 (N,3),真逆解(θ 反解牛顿迭代)。

        做法:像素减主点除焦距得畸变角向量,θd = hypot;方位角 φ 保持
        不变(畸变只改半径不改方向);对 θd = θ·(1+k1θ²+...+k4θ⁸) 用牛顿
        迭代反解 θ(初值 θ=θd,畸变小则几乎一步收敛);射线方向 =
        (sinθ·cosφ, sinθ·sinφ, cosθ)。往返契约 < 1e-9 像素,测试锁定。
        """
        uv_arr = np.asarray(uv, dtype=np.float64)
        if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
            raise AssertionError(
                f"模型 {self.model_id} 的 unproject 收到形状 {uv_arr.shape} 的像素坐标,"
                f"期望 (N,2)。请检查调用方传的是不是逐点 (u,v)。"
            )
        p = as_params(params, self.model_id, self.param_names)
        fx, fy, cx, cy = p[:4]
        k = p[4:]
        xd = (uv_arr[:, 0] - cx) / fx
        yd = (uv_arr[:, 1] - cy) / fy
        theta_d = np.hypot(xd, yd)
        phi = np.arctan2(yd, xd)  # 方位角:畸变前后不变

        # 牛顿迭代解 f(θ) = θ·P(θ) − θd = 0,
        # f'(θ) = 1 + 3k1θ² + 5k2θ⁴ + 7k3θ⁶ + 9k4θ⁸(积的求导法则)
        theta = theta_d.copy()
        for _ in range(50):
            t2 = theta * theta
            poly = 1.0 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4
            f = theta * poly - theta_d
            f_prime = (1.0 + 3.0 * k[0] * t2 + 5.0 * k[1] * t2**2
                       + 7.0 * k[2] * t2**3 + 9.0 * k[3] * t2**4)
            step = f / f_prime
            theta = theta - step
            if float(np.abs(step).max(initial=0.0)) < 1e-15:
                break

        return np.stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)], axis=1)

    def default_params(self, image_size: tuple[int, int]) -> np.ndarray:
        """初值(OpenCV 鱼眼惯例):fx=fy=max(宽,高)/π——等距投影下
        180° 视场正好铺满最大边;主点在中心,k 全 0。"""
        width, height = image_size
        p = np.zeros(len(self.param_names), dtype=np.float64)
        p[0] = p[1] = max(width, height) / np.pi
        p[2] = width / 2.0
        p[3] = height / 2.0
        return p

    def intrinsics_matrix(self, params: np.ndarray) -> np.ndarray:
        """参数 → 标准 3×3 K(K[0,1]=0:skew 是投影用不上的死参数,定死 0)。"""
        p = as_params(params, self.model_id, self.param_names)
        return np.array([[p[0], 0.0, p[2]],
                         [0.0, p[1], p[3]],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def opencv_adapter(self) -> "_KB4OpenCVAdapter":
        return _KB4OpenCVAdapter(self)


class _KB4OpenCVAdapter:
    """M4 的 OpenCV 适配器:cv2.fisheye 调用的 flags 与 K/D 装包解包唯一入口。

    形状与 dtype 硬规则(fisheye 系函数实测行为,调用方必须遵守):
    - 输入点集/像素一律 (N,1,3)/(N,1,2) 形状,传 (N,x) 会报难懂断言;
    - fisheye 系接受 float64(projectPoints/校准),与 calibrateCamera 只收
      float32 不同——本适配器进出全 float64,不做降精度。
    """

    kind = "fisheye"
    # CALIB_RECOMPUTE_EXTRINSIC(=2):每轮迭代重算位姿,不带它会有假收敛(阶段一实测);
    # CALIB_FIX_SKEW(=8):skew 固定 0,与自实现模型两侧严格一致(Pitfall 7)
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW  # = 10

    def __init__(self, model: FisheyeKB4):
        self._model = model

    def params_to_cv(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """params (8,) → (K 3×3 float64 且 K[0,1]=0, D (4,) float64)。"""
        p = as_params(params, self._model.model_id, self._model.param_names)
        return self._model.intrinsics_matrix(p), p[4:].copy()

    def params_from_cv(self, k: np.ndarray, d: np.ndarray) -> np.ndarray:
        """cv2.fisheye 返回的 (K, D) → params;D 长 4 全取(skew 由 flags 锁 0)。"""
        k = np.asarray(k, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64).ravel()
        if d.shape[0] != 4:
            raise AssertionError(
                f"模型 {self._model.model_id} 的 OpenCV 鱼眼畸变向量长度应为 4,"
                f"实际 {d.shape[0]}。请确认调用的是 cv2.fisheye 系函数。"
            )
        p = np.empty(len(self._model.param_names), dtype=np.float64)
        p[0], p[1], p[2], p[3] = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        # K[0,1](skew)即使非 0 也不进参数表:projectPoints 忽略它,读回来只会撒谎
        p[4:] = d
        return p
