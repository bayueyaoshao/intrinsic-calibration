"""θ 装箱、残差与解析稀疏雅可比的组装:scipy 精化后端的数学供给侧(D-08)。

行序契约(为什么要在文件开头先讲它):残差向量按 **视图→点→(u,v) 两行
交错** 排列,雅可比的行必须与之逐行对齐。这个约定一旦在残差与雅可比两侧
各自为政(比如一侧按视图分块、一侧按坐标轴分块),least_squares 拿到的
梯度就是错的——**不报任何异常,只是收敛奇慢或原地不动**,等发现时已经
烧掉几小时的实验网格(研究 Pitfall 6)。防线有两条:本文件里残差与雅可
比共用同一个行号函数 row_index(结构性防线);测试用双数值基准逐项比对
(数值防线,见 tests/test_solve_scipy_residuals.py)。

θ 布局(联合单向量,研究 §Pattern 2 实测结论):θ = [内参 (P,) |
视图 0 的 (ω,t) 6 个 | 视图 1 的 6 个 | …],总长 P + 6V(640 视图的 M2
是 9 + 3840 = 3849)。不做"内参/外参分块交替优化"——那等于自研优化器,
收敛性质更差且实测收益为零(Don't Hand-Roll)。

稀疏结构的硬前提:每个残差行只碰 P 个内参列 + 本视图 6 个外参列
(640 视图 M2 的非零元约 169 万),而稠密雅可比等价 3.47 GB 内存——
先稠密跑通再优化在真实规模直接不可行,所以从第一行代码起就走
scipy.sparse(不自建稀疏索引,研究 Don't Hand-Roll)。

数学来源锁定(D-08):残差 = model.project(经 base.transform_to_camera
变换到相机坐标系的物点) − 实测角点——投影只经模型层,本文件不出现任何
OpenCV 的投影函数,monkeypatch 与源码静态断言双锁(02-04 Task 3)。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from calib.models.base import CameraModel, as_params, transform_to_camera
from calib.solve.observation import ObservationSet


def row_index(view: int, point: int, uv_axis: int, n_points: int) -> int:
    """残差行号:视图 → 点 → (u,v) 两行交错(残差与雅可比共用的唯一行号函数)。

    白话:第 view 个视图第 point 个角点的 u 残差住偶数行,v 残差住紧随的
    奇数行。行号算错 = 梯度接错行(Pitfall 6),所以两侧只准调本函数,
    不准各自手算下标。
    """
    if uv_axis not in (0, 1):
        raise AssertionError(
            f"uv_axis 只能是 0(u 分量)或 1(v 分量),实际是 {uv_axis!r}。"
            f"残差按 (u,v) 两行交错排列,行号函数只认识这两个分量。"
        )
    return 2 * (view * n_points + point) + uv_axis


def pack_theta(model: CameraModel, params, rvecs, tvecs) -> np.ndarray:
    """内参 + 逐视图位姿 → 联合单向量 θ = [内参 (P,) | 各视图 (ω,t) 每 6 个]。

    params 接受 (P,) 数组,或"参数名 → 数值"的字典(按 model.param_names
    顺序取值;缺名直接中文报错,不猜位置);rvecs / tvecs 各 (V,3)。
    返回 (P + 6V,) float64。
    """
    if isinstance(params, Mapping):
        missing = [name for name in model.param_names if name not in params]
        if missing:
            raise AssertionError(
                f"参数字典缺 {model.model_id} 的参数名:{missing}"
                f"(全部应为 {list(model.param_names)})。请按参数名给全,"
                f"装箱不猜位置。"
            )
        arr = np.array([float(params[name]) for name in model.param_names],
                       dtype=np.float64)
    else:
        arr = np.asarray(params, dtype=np.float64).ravel()
    arr = as_params(arr, model.model_id, model.param_names)

    rv = np.asarray(rvecs, dtype=np.float64)
    tv = np.asarray(tvecs, dtype=np.float64)
    if rv.ndim != 2 or rv.shape[1] != 3 or rv.shape[0] < 1 or tv.shape != rv.shape:
        raise AssertionError(
            f"rvecs/tvecs 形状不合:期望各 (V,3) 且 V≥1(逐视图位姿),"
            f"实际是 {rv.shape} / {tv.shape}。请检查位姿是否按视图逐行给出。"
        )

    pieces = [arr]
    for i in range(rv.shape[0]):
        pieces.append(np.concatenate([rv[i], tv[i]]))  # 视图 i 的 (ω,t) 6 个连排
    return np.concatenate(pieces)


def unpack_theta(model: CameraModel,
                 theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """θ (P+6V,) → (params (P,), rvecs (V,3), tvecs (V,3));与 pack_theta 互逆。

    长度不是 P + 6 的整数倍(即对不上任何视图数)时直接中文报错——
    装箱与解箱必须同一布局,错位在这里点名,不带病进优化器。
    """
    th = np.asarray(theta, dtype=np.float64).ravel()
    n_params = len(model.param_names)
    if th.shape[0] < n_params + 6 or (th.shape[0] - n_params) % 6 != 0:
        raise AssertionError(
            f"θ 向量长度不合:模型 {model.model_id} 的长度应为 内参{n_params} + "
            f"6×视图数,实际 {th.shape[0]}(对不上任何视图数)。"
            f"请检查 θ 是否由 pack_theta 按同一布局装出。"
        )
    n_views = (th.shape[0] - n_params) // 6
    params = th[:n_params].copy()
    rvecs = np.empty((n_views, 3), dtype=np.float64)
    tvecs = np.empty((n_views, 3), dtype=np.float64)
    for i in range(n_views):
        chunk = th[n_params + 6 * i: n_params + 6 * (i + 1)]
        rvecs[i] = chunk[:3]   # 前 3 个 = 旋转向量 ω(Rodrigues 语义)
        tvecs[i] = chunk[3:]   # 后 3 个 = 平移向量 t
    return params, rvecs, tvecs


def make_residual_fn(model: CameraModel, obs: ObservationSet):
    """残差函数工厂:返回 callable(θ) → (2·V·P,) 残差向量,float64。

    残差 = model.project(transform_to_camera(物点, rvec, tvec) 的相机坐标点,
    params) − 实测角点,逐视图循环、视图内全 numpy——640 视图单次约 0.02
    秒;逐点的 Python 循环(640×88×2 次追加)是被研究 Pitfall 5 明令禁止
    的写法,慢一个数量级以上。
    """
    object_points = obs.object_points
    image_points = obs.image_points
    n_views = len(obs.image_ids)
    n_points = object_points.shape[0]

    def residual_fn(theta: np.ndarray) -> np.ndarray:
        params, rvecs, tvecs = unpack_theta(model, theta)
        out = np.empty(2 * n_views * n_points, dtype=np.float64)
        for i in range(n_views):
            pts_cam, _pose_block = transform_to_camera(
                object_points, rvecs[i], tvecs[i])
            # (n,2) 拉平后天然就是 (u,v) 两行交错,与 row_index 的约定一致
            out[i * 2 * n_points:(i + 1) * 2 * n_points] = (
                model.project(pts_cam, params) - image_points[i]).ravel()
        return out

    return residual_fn


def _structure_positions(n_params: int, n_views: int, n_points: int):
    """全部 (行, 列) 结构位置,按 [视图0 内参块, 视图0 外参块, 视图1 …] 顺序展平。

    结构事实(实测):每残差行恰碰 P 个内参列 + 本视图 6 个外参列;
    nnz = 2·V·P·(P+6)。雅可比组装与结构矩阵共用本函数,保证两者的
    行列排布永远一致(Pitfall 6 的结构性防线)。
    """
    row_template = np.arange(2 * n_points)
    col_int = np.arange(n_params)
    col_pose = np.arange(n_params, n_params + 6)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for i in range(n_views):
        base = i * 2 * n_points
        rows.append(np.broadcast_to(base + row_template[:, None],
                                    (2 * n_points, n_params)))
        cols.append(np.broadcast_to(col_int, (2 * n_points, n_params)))
        rows.append(np.broadcast_to(base + row_template[:, None],
                                    (2 * n_points, 6)))
        cols.append(np.broadcast_to(col_pose + 6 * i, (2 * n_points, 6)))
    return (np.concatenate([r.ravel() for r in rows]),
            np.concatenate([c.ravel() for c in cols]))


def make_jacobian_fn(model: CameraModel, obs: ObservationSet):
    """解析稀疏雅可比工厂:返回 callable(θ) → scipy.sparse.csr_matrix。

    形状 (2·V·P, P+6V)。逐视图组装 (2P, P+6) 块:
    - 内参 P 列 = model.project_jacobian 的 (n,2,P) 直接 reshape (2n,P)
      ——数组内存布局天然就是 (u,v) 交错行序,不做任何重排;
    - 外参 6 列 = 链式乘积 model.project_point_jacobian (n,2,3)
      · base.transform_to_camera 给出的 ∂Xc/∂(ω,t) (n,3,6),einsum 一次
      算完整个视图(公式见 02-RESEARCH §Code Examples"三个组成块")。
    COO 三元组向量化拼接后转 csr(640 视图单次约 0.15 秒,实测参照);
    数值正确性由双数值基准测试逐列逐项锁定。
    """
    object_points = obs.object_points
    n_views = len(obs.image_ids)
    n_points = object_points.shape[0]
    n_params = len(model.param_names)
    shape = (2 * n_views * n_points, n_params + 6 * n_views)
    rows, cols = _structure_positions(n_params, n_views, n_points)

    def jacobian_fn(theta: np.ndarray) -> csr_matrix:
        params, rvecs, tvecs = unpack_theta(model, theta)
        vals: list[np.ndarray] = []
        for i in range(n_views):
            pts_cam, dxc_d_pose = transform_to_camera(
                object_points, rvecs[i], tvecs[i])
            jac_int = model.project_jacobian(pts_cam, params).reshape(
                2 * n_points, n_params)
            jac_pose = np.einsum(
                "nij,njk->nik",                    # (n,2,3)·(n,3,6) → (n,2,6)
                model.project_point_jacobian(pts_cam, params),
                dxc_d_pose).reshape(2 * n_points, 6)
            vals.append(jac_int.ravel())   # 展平顺序与 _structure_positions 一致
            vals.append(jac_pose.ravel())
        return coo_matrix((np.concatenate(vals), (rows, cols)),
                          shape=shape).tocsr()

    return jacobian_fn


def make_sparsity(model: CameraModel, n_views: int, n_points: int) -> csr_matrix:
    """0/1 稀疏结构矩阵(least_squares 的 jac_sparsity 用),csr。

    结构 = "允许非零"的位置清单:每残差行恰 P 个内参列 + 本视图 6 个外参
    列,别视图的外参列结构为零。个别位置的实际导数可能恰为 0(如针孔族
    ∂u/∂fy 恒 0)——结构照给、值为 0 不碍事;反过来,结构外的位置必须
    永远为 0,scipy 按结构分组解子问题,这是它对结构清单的信赖。
    """
    n_params = len(model.param_names)
    rows, cols = _structure_positions(n_params, n_views, n_points)
    return coo_matrix((np.ones(rows.shape[0], dtype=np.int8), (rows, cols)),
                      shape=(2 * n_views * n_points, n_params + 6 * n_views)).tocsr()
