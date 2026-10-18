"""针孔 Brown-Conrady 模型族:M1 / M2 / M3(同一个畸变核,系数个数不同)。

白话定义:针孔模型假设光线过一个小孔直线成像——三维点先除以深度得到
"归一化坐标"(以光心为原点、焦距为 1 的像面坐标),再乘焦距加主点得到
像素。真实镜头把直线拍弯(畸变):径向畸变(离画面中心越远弯得越狠,
k 系数描述)与切向畸变(镜头与成像面没装正,p1/p2 描述)叠加在归一化
坐标上,然后才乘焦距。

三个模型的差别只有畸变系数个数(研究实测对齐 cv2 4.14.0,公式嵌套方式
经数值判定——**切向项加在有理分母之外**:先算径向缩放,除完分母,
再加切向偏移):
- M1(8 参):fx,fy,cx,cy + k1,k2,p1,p2——k3 恒 0(轻畸变镜头够用);
- M2(9 参):M1 + k3——即阶段一针孔基线,最常用的五系数;
- M3(12 参):M2 + k4,k5,k6——有理模型:径向缩放从多项式升级成
  分式 num/den(num、den 各一个三次多项式),强畸变(广角/鱼眼镜头
  硬套针孔模型)时拟合能力强得多,代价是参数更多、更易过拟合
  (是否过拟合由阶段四 held-out 评估裁决,不在本文件)。

dtype 与形状硬规则:全函数 float64 进出;OpenCV 侧的 float32 装包/
解包规则只出现在本文件底部的适配器里,是唯一入口(防 dtype 静默劣化)。
"""

from __future__ import annotations

import cv2
import numpy as np

from calib.models import register_model
from calib.models.base import CameraModel, as_params, as_points3d

# 径向 k 系数在 dist 向量里的下标(dist 布局 [k1,k2,p1,p2,(k3),(k4,k5,k6)]):
# 4 长(M1)→ k1,k2;5 长(M2)→ k1,k2,k3;8 长(M3)→ k1..k6。
# 切向 p1,p2 恒在 (2,3)。
_RADIAL_SLOTS: dict[int, tuple[int, ...]] = {
    4: (0, 1),
    5: (0, 1, 4),
    8: (0, 1, 4, 5, 6, 7),
}


def _radial(r2: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """径向缩放系数 rad(r2)(标量因子,乘在归一化坐标上)。"""
    r4 = r2 * r2
    if len(dist) == 8:  # M3 有理:分子分母各一个三次多项式
        num = 1.0 + dist[0] * r2 + dist[1] * r4 + dist[4] * r4 * r2
        den = 1.0 + dist[5] * r2 + dist[6] * r4 + dist[7] * r4 * r2
        return num / den
    k3 = dist[4] if len(dist) == 5 else 0.0  # M1 没有 k3,恒 0
    return 1.0 + dist[0] * r2 + dist[1] * r4 + k3 * r4 * r2


def _radial_and_derivs(r2: np.ndarray,
                       dist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """径向缩放及其全部导数原料(投影雅可比与畸变逆解共用)。

    返回 (rad (N,), drad_dr2 (N,), drad_dk (N, K)):
      rad        径向缩放系数本身;
      drad_dr2   对 r2 的导数;
      drad_dk    对 _RADIAL_SLOTS[len(dist)] 里那 K 个 k 系数的导数,列序同 slots。

    M3 有理分母的导数用商法则(链式照抄会漏 den 的依赖):
      ∂rad/∂k1 = r2/den,∂rad/∂k4 = −num·r2/den²(其余幂次类推);
      ∂rad/∂(r2) = (num′ − rad·den′)/den,
      其中 num′ = k1+2k2·r2+3k3·r4、den′ = k4+2k5·r2+3k6·r4。
    """
    r4 = r2 * r2
    r6 = r4 * r2
    slots = _RADIAL_SLOTS[len(dist)]
    drad_dk = np.empty((r2.shape[0], len(slots)), dtype=np.float64)
    if len(dist) == 8:
        k1, k2, _p1, _p2, k3, k4, k5, k6 = dist
        num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        rad = num / den
        # 分子侧:k1,k2,k3(幂次 r2,r4,r6);分母侧:k4,k5,k6(带 −num/den²)
        drad_dk[:, 0] = r2 / den
        drad_dk[:, 1] = r4 / den
        drad_dk[:, 2] = r6 / den
        drad_dk[:, 3] = -num * r2 / (den * den)
        drad_dk[:, 4] = -num * r4 / (den * den)
        drad_dk[:, 5] = -num * r6 / (den * den)
        num_prime = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4
        den_prime = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4
        drad_dr2 = (num_prime - rad * den_prime) / den
    else:
        k3 = dist[4] if len(dist) == 5 else 0.0
        rad = 1.0 + dist[0] * r2 + dist[1] * r4 + k3 * r6
        drad_dk[:, 0] = r2
        drad_dk[:, 1] = r4
        if len(dist) == 5:
            drad_dk[:, 2] = r6
        drad_dr2 = dist[0] + 2.0 * dist[1] * r2 + 3.0 * k3 * r4
    return rad, drad_dr2, drad_dk


def _distort(x: np.ndarray, y: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """归一化无畸变坐标 (x,y) → 归一化畸变坐标 (xd,yd)(畸变核本体)。

    切向项加在有理分母之外(先除分母,再加切向)——嵌套方式经研究实测
    对齐 cv2 4.14.0 的 projectPoints,parity 测试锁死在 1e-9 像素内。
    """
    r2 = x * x + y * y
    rad = _radial(r2, dist)
    p1, p2 = dist[2], dist[3]
    xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


class _PinholeBrown(CameraModel):
    """针孔族共享实现(不注册、不实例化):三个子类只差畸变系数个数。"""

    _n_dist: int = 0  # 子类定:4(M1)/ 5(M2)/ 8(M3)

    def project(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        """投影:(N,3) 相机坐标系点 + (P,) 参数 → (N,2) 像素,float64。"""
        pts = as_points3d(pts_cam)
        p = as_params(params, self.model_id, self.param_names)
        if not np.all(pts[:, 2] > 0.0):
            raise AssertionError(
                f"模型 {self.model_id} 的 project 收到 Z<=0 的点:针孔投影要除以 Z,"
                f"Z 必须全为正(相机坐标系 Z 轴朝前)。请检查点的坐标系。"
            )
        z = pts[:, 2]
        x = pts[:, 0] / z
        y = pts[:, 1] / z
        xd, yd = _distort(x, y, p[4:])
        return np.stack([p[0] * xd + p[2], p[1] * yd + p[3]], axis=1)

    def project_jacobian(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        """投影对内参的解析雅可比:(N,3)+(P,) → (N,2,P),float64。

        只含内参列;对位姿的链式块(d像素/d相机点 × ∂Xc/∂外参)由
        base.transform_to_camera 与后续 solve/residuals.py 组装——
        每个模型只写自己那份内参导数,外参导数全模型共享一份。
        """
        pts = as_points3d(pts_cam)
        p = as_params(params, self.model_id, self.param_names)
        fx, fy = p[0], p[1]
        z = pts[:, 2]
        x = pts[:, 0] / z
        y = pts[:, 1] / z
        r2 = x * x + y * y
        dist = p[4:]
        rad, _drad_dr2, drad_dk = _radial_and_derivs(r2, dist)
        p1, p2 = dist[2], dist[3]
        xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        n = pts.shape[0]
        jac = np.zeros((n, 2, len(self.param_names)), dtype=np.float64)
        # 焦距与主点:线性项,直接是畸变坐标/常数 1
        jac[:, 0, 0] = xd
        jac[:, 1, 1] = yd
        jac[:, 0, 2] = 1.0
        jac[:, 1, 3] = 1.0
        # 径向 k 系数:d u/d k = fx·x·drad/dk,d v/d k = fy·y·drad/dk
        for j, slot in enumerate(_RADIAL_SLOTS[len(dist)]):
            col = 4 + slot  # dist 下标 → 参数向量下标(前面 4 个是 fx,fy,cx,cy)
            jac[:, 0, col] = fx * x * drad_dk[:, j]
            jac[:, 1, col] = fy * y * drad_dk[:, j]
        # 切向 p1,p2(对归一化坐标是多项式,对像素再乘焦距)
        jac[:, 0, 6] = fx * 2.0 * x * y
        jac[:, 1, 6] = fy * (r2 + 2.0 * y * y)
        jac[:, 0, 7] = fx * (r2 + 2.0 * x * x)
        jac[:, 1, 7] = fy * 2.0 * x * y
        return jac

    def project_point_jacobian(self, pts_cam: np.ndarray,
                               params: np.ndarray) -> np.ndarray:
        """投影对相机坐标系点的解析雅可比:(N,3)+(P,) → (N,2,3),float64。

        链式 = 畸变核对归一化坐标的 2×2 导数(与 unproject 牛顿迭代里的
        那份 2×2 公式逐字相同——同一份公式两处用,保证正逆解导数一致)
        乘上 归一化坐标对相机点的导数(∂x/∂X=1/z、∂x/∂Z=−x/z 这组除法链)。
        针孔族在光轴上没有奇异点(z>0 已由 project 断言),无需特判。
        """
        pts = as_points3d(pts_cam)
        p = as_params(params, self.model_id, self.param_names)
        fx, fy = p[0], p[1]
        z = pts[:, 2]
        if not np.all(z > 0.0):
            raise AssertionError(
                f"模型 {self.model_id} 的 project_point_jacobian 收到 Z<=0 的点:"
                f"针孔投影要除以 Z,Z 必须全为正。请检查点的坐标系。"
            )
        x = pts[:, 0] / z
        y = pts[:, 1] / z
        r2 = x * x + y * y
        dist = p[4:]
        rad, drad_dr2, _drad_dk = _radial_and_derivs(r2, dist)
        p1, p2 = dist[2], dist[3]

        # 2×2 畸变核导数(与 unproject 牛顿迭代的雅可比同式):a12 对称共用
        a11 = rad + 2.0 * x * x * drad_dr2 + 2.0 * p1 * y + 6.0 * p2 * x
        a12 = 2.0 * x * y * drad_dr2 + 2.0 * p1 * x + 2.0 * p2 * y
        a22 = rad + 2.0 * y * y * drad_dr2 + 6.0 * p1 * y + 2.0 * p2 * x

        jac = np.empty((pts.shape[0], 2, 3), dtype=np.float64)
        inv_z = 1.0 / z
        jac[:, 0, 0] = fx * a11 * inv_z
        jac[:, 0, 1] = fx * a12 * inv_z
        jac[:, 0, 2] = -fx * (a11 * x + a12 * y) * inv_z
        jac[:, 1, 0] = fy * a12 * inv_z
        jac[:, 1, 1] = fy * a22 * inv_z
        jac[:, 1, 2] = -fy * (a12 * x + a22 * y) * inv_z
        return jac

    def unproject(self, uv: np.ndarray, params: np.ndarray) -> np.ndarray:
        """反投影:像素 → 归一化射线方向 (N,3),真逆解(畸变逆解牛顿迭代)。

        做法:像素减主点除焦距得归一化畸变坐标 (xd,yd);解方程组
        "xd = x·rad(r2)+切向、yd = y·rad(r2)+切向"(畸变核正向公式)
        反求无畸变坐标 (x,y)——牛顿迭代:初值取 (xd,yd)(畸变小则几乎
        一步收敛),每步解 2×2 线性方程修正,迭代到修正量 < 1e-15。
        返回 (x,y,1)/‖(x,y,1)‖(单位射线;project 对射线缩放不敏感,
        往返契约 project → unproject → project 误差 < 1e-9 像素,测试锁定)。
        """
        uv_arr = np.asarray(uv, dtype=np.float64)
        if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
            raise AssertionError(
                f"模型 {self.model_id} 的 unproject 收到形状 {uv_arr.shape} 的像素坐标,"
                f"期望 (N,2)。请检查调用方传的是不是逐点 (u,v)。"
            )
        p = as_params(params, self.model_id, self.param_names)
        fx, fy, cx, cy = p[:4]
        dist = p[4:]
        xd = (uv_arr[:, 0] - cx) / fx
        yd = (uv_arr[:, 1] - cy) / fy

        # 牛顿迭代解畸变正映射的逆:残差 = 当前猜测的畸变结果 − 观测畸变坐标
        x, y = xd.copy(), yd.copy()
        p1, p2 = dist[2], dist[3]
        for _ in range(50):
            r2 = x * x + y * y
            rad, drad_dr2, _drad_dk = _radial_and_derivs(r2, dist)
            x_pred = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_pred = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            res_x = x_pred - xd
            res_y = y_pred - yd
            # 2×2 雅可比(畸变坐标对无畸变坐标),解析式:
            a11 = rad + 2.0 * x * x * drad_dr2 + 2.0 * p1 * y + 6.0 * p2 * x
            a12 = 2.0 * x * y * drad_dr2 + 2.0 * p1 * x + 2.0 * p2 * y
            a22 = rad + 2.0 * y * y * drad_dr2 + 6.0 * p1 * y + 2.0 * p2 * x
            det = a11 * a22 - a12 * a12
            dx = (a22 * res_x - a12 * res_y) / det
            dy = (a11 * res_y - a12 * res_x) / det
            x = x - dx
            y = y - dy
            if max(float(np.abs(dx).max()), float(np.abs(dy).max())) < 1e-15:
                break

        rays = np.stack([x, y, np.ones_like(x)], axis=1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def default_params(self, image_size: tuple[int, int]) -> np.ndarray:
        """初值:fx=fy=0.75×宽(本数据量级 fx≈1872/宽 2560,0.75 是贴近的经验值)、
        主点在画面中心、畸变全 0——全部有限值、焦距为正。"""
        width, height = image_size
        p = np.zeros(len(self.param_names), dtype=np.float64)
        p[0] = p[1] = 0.75 * float(width)
        p[2] = width / 2.0
        p[3] = height / 2.0
        return p

    def intrinsics_matrix(self, params: np.ndarray) -> np.ndarray:
        """参数 → 标准 3×3 K(skew 恒 0:针孔族参数表里本来就没有 skew)。"""
        p = as_params(params, self.model_id, self.param_names)
        return np.array([[p[0], 0.0, p[2]],
                         [0.0, p[1], p[3]],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


class _BrownOpenCVAdapter:
    """针孔族 OpenCV 适配器:flags 与 K/D 装包解包的唯一入口。

    OpenCV 的畸变向量长度随 flags 变(Pitfall 实测):FIX_K3 返回 5 长
    (k3=0)、默认 5 长、RATIONAL_MODEL 返回 **14 长**(尾部 6 个是未启用
    的 thin-prism/tilted 系数恒 0)——解包一律按 take 切片,装包→解包
    往返逐位相等有测试锁定。float32 装包(calibrateCamera 绑定只收
    32 位浮点)由求解器在调用处做,本适配器保持 float64 数学不降精度。
    """

    kind = "calibrateCamera"

    def __init__(self, model: _PinholeBrown, flags: int, cv_d_len: int, take: int):
        self._model = model
        # 常量表达式:cv2.CALIB_FIX_K3=128 / 0 / cv2.CALIB_RATIONAL_MODEL=16384
        self.flags = int(flags)
        self._cv_d_len = cv_d_len
        self._take = take

    def params_to_cv(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """params (P,) → (K 3×3, D (cv_d_len,)),float64。"""
        p = as_params(params, self._model.model_id, self._model.param_names)
        k = self._model.intrinsics_matrix(p)
        d = np.zeros(self._cv_d_len, dtype=np.float64)
        d[:self._take] = p[4:]
        return k, d

    def params_from_cv(self, k: np.ndarray, d: np.ndarray) -> np.ndarray:
        """cv2 返回的 (K, D) → params;D 超长按 take 切片(如 M3 的 14 长取前 8)。"""
        k = np.asarray(k, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64).ravel()
        if d.shape[0] < self._take:
            raise AssertionError(
                f"模型 {self._model.model_id} 的 OpenCV 畸变向量太短:"
                f"至少需要 {self._take} 个系数,实际 {d.shape[0]}。"
                f"请确认 flags 与模型是否匹配(flags={self.flags})。"
            )
        p = np.empty(len(self._model.param_names), dtype=np.float64)
        p[0], p[1], p[2], p[3] = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        p[4:] = d[:self._take]
        return p


@register_model
class PinholeM1(_PinholeBrown):
    """M1:针孔 + k1,k2,p1,p2(8 参,k3 恒 0)——轻畸变镜头的省参数选择。"""

    model_id = "m1"
    description = "针孔+Brown 四系数(k1,k2,p1,p2;k3 固定 0)"
    param_names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")
    _n_dist = 4

    def opencv_adapter(self) -> _BrownOpenCVAdapter:
        # CALIB_FIX_K3(=128):k3 固定为传入值 0;OpenCV 返回 D 长 5,取前 4
        return _BrownOpenCVAdapter(self, flags=cv2.CALIB_FIX_K3, cv_d_len=5, take=4)


@register_model
class PinholeM2(_PinholeBrown):
    """M2:针孔 + k1,k2,p1,p2,k3(9 参)——最常用的五系数,即阶段一针孔基线(D-07)。"""

    model_id = "m2"
    description = "针孔+Brown 五系数(k1,k2,p1,p2,k3),阶段一针孔基线"
    param_names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    _n_dist = 5

    def opencv_adapter(self) -> _BrownOpenCVAdapter:
        # flags=0:估计全部 5 个系数;D 长 5 全取
        return _BrownOpenCVAdapter(self, flags=0, cv_d_len=5, take=5)


@register_model
class PinholeM3(_PinholeBrown):
    """M3:针孔有理模型 k1–k6+p1p2(12 参)——径向缩放升级成 num/den 分式。"""

    model_id = "m3"
    description = "针孔+Brown 有理模型(k1..k6,p1,p2;径向为 num/den 分式)"
    param_names = ("fx", "fy", "cx", "cy",
                   "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    _n_dist = 8

    def opencv_adapter(self) -> _BrownOpenCVAdapter:
        # CALIB_RATIONAL_MODEL(=16384):启用 k4..k6;OpenCV 返回 D 长 14
        # (尾部 6 个是未启用的 thin-prism/tilted 恒 0),取前 8 才是 M3
        return _BrownOpenCVAdapter(self, flags=cv2.CALIB_RATIONAL_MODEL,
                                   cv_d_len=8, take=8)
