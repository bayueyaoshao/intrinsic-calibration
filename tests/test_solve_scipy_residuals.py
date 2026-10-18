"""scipy 精化后端的残差与解析稀疏雅可比测试(θ 装箱 + 行序契约 + 双数值基准)。

锁定清单(对应 02-04 Task 1 的四条行为):
- θ 装箱往返:pack_theta → unpack_theta 对参数与逐视图位姿逐位还原,
  参数允许数组或"按参数名"的字典两种形式;
- 解析稀疏雅可比的双重数值锁定(研究 Pitfall 6):在随机真值 θ 上,
  ① 对 numdifftools 基准(理查森外推,精度最高)逐列相对误差 < 1e-6;
  ② 对 scipy least_squares 默认采用的 2-point 前向差分方案
    (rel_step = √机器精度,diff_step 缺省语义)在随机抽取的 60 个
    行列位置上逐项一致——残差与雅可比的行序错位会在这里现形
    (行序错位无报错,但梯度错、优化收敛不动);
- 行序契约:残差按 视图→点→(u,v) 两行交错,总长 = 2·V·P;改一个视图
  的 tvec 只影响该视图的行(残差与雅可比都是,稀疏结构的局部性);
- 稀疏结构:make_sparsity 的 0/1 结构覆盖雅可比的全部实际非零位置,
  每个残差行只碰 P 个内参列 + 本视图 6 个外参列——640 视图的稠密
  雅可比等价 3.47 GB 内存,稀疏结构是硬前提不是优化项;
- 性能行为代理(研究 Pitfall 5,禁逐点 Python 循环):8 视图单次雅可
  比调用 < 1 秒(逐点循环组装是 1 秒量级起步,向量化应为毫秒级);
- KB4 近轴保护:光轴附近 r<1e-12 处像素对相机点的导数用解析极限顶上,
  绝不出现 NaN(0/0 不加保护会毒化整个求解,研究 Pitfall 4)。

测试数据全部是合成真值(随机真值 θ,角点由 models.project 精确投影生成,
无噪声),秒级;真实 640 视图的耗时与收敛由 02-05 试车验证。
"""

from __future__ import annotations

import time
from pathlib import Path

import numdifftools as nd  # D-10:数值差分基准,只进测试,绝不进 src/calib
import numpy as np
import pytest

from calib.io.config import BoardSpec
from calib.models import create_model
from calib.models.base import transform_to_camera
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet
from calib.solve.residuals import (
    make_jacobian_fn,
    make_residual_fn,
    make_sparsity,
    pack_theta,
    row_index,
    unpack_theta,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _random_truth_obs(model_id: str, n_views: int = 2, seed: int = 13):
    """构造"随机真值 θ + 精确投影角点"的合成观测集。

    真值内参取 default_params 加随机畸变(典型量级),真值位姿在小转角
    范围随机;角点由 models.project 精确投影生成(无噪声)。雅可比数值
    校验要的就是"残差接近零"的环境:残差大时前向差分的截断误差会盖过
    1e-6 的判定线,残差为零时差分精度只受机器精度限制。
    """
    rng = np.random.default_rng(seed)
    model = create_model(model_id)
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    objp = build_object_points(board)
    params = np.asarray(model.default_params((640, 480)), dtype=np.float64).copy()
    params[4:] += rng.normal(0.0, 0.03, size=params[4:].shape[0])
    rvecs = rng.uniform(-0.10, 0.10, size=(n_views, 3))
    tvecs = np.stack([rng.uniform(-30.0, 30.0, size=n_views),
                      rng.uniform(-30.0, 30.0, size=n_views),
                      np.full(n_views, 800.0)], axis=1)
    image_points = np.empty((n_views, objp.shape[0], 2), dtype=np.float64)
    for i in range(n_views):
        pts_cam, _pose_block = transform_to_camera(objp, rvecs[i], tvecs[i])
        image_points[i] = model.project(pts_cam, params)
    obs = ObservationSet(
        image_ids=tuple(f"img_{k:02d}" for k in range(n_views)),
        pose_ids=tuple(f"pose_{k:02d}" for k in range(n_views)),
        object_points=objp,
        image_points=image_points,
        image_size=(640, 480),
        corner_source=("sb",) * n_views,
    )
    return obs, model, params, rvecs, tvecs


# ---- 行为 1:θ 装箱往返 ----

@pytest.mark.parametrize("model_id", ["m1", "m2", "m3", "m4"])
def test_pack_unpack_roundtrip_bitwise(model_id):
    """pack_theta → unpack_theta 对参数与位姿逐位还原;参数也接受字典形式。

    θ 布局 = [内参 (P,) | 视图 0 的 (ω,t) 6 个 | 视图 1 的 6 个 | …],
    4 个模型参数个数不同(8/9/12/8),装箱长度随之变化,逐模型验证。
    """
    _obs, model, params, rvecs, tvecs = _random_truth_obs(model_id, n_views=3, seed=5)
    theta = pack_theta(model, params, rvecs, tvecs)
    assert theta.shape == (len(model.param_names) + 6 * 3,), \
        f"{model_id} 装箱长度应为 内参{len(model.param_names)} + 6×3视图 = " \
        f"{len(model.param_names) + 18},实际 {theta.shape}"
    unpacked_params, unpacked_rvecs, unpacked_tvecs = unpack_theta(model, theta)
    np.testing.assert_array_equal(unpacked_params, params)
    np.testing.assert_array_equal(unpacked_rvecs, rvecs)
    np.testing.assert_array_equal(unpacked_tvecs, tvecs)

    theta_from_dict = pack_theta(
        model, {name: float(v) for name, v in zip(model.param_names, params)},
        rvecs, tvecs)
    np.testing.assert_array_equal(theta_from_dict, theta)


# ---- 行为 2:解析稀疏雅可比对双数值基准 ----

@pytest.mark.parametrize("model_id", ["m1", "m2", "m3", "m4"])
def test_sparse_jacobian_matches_numeric_oracles(model_id):
    """随机真值 θ 上,解析稀疏雅可比同时过两道数值基准:
    numdifftools 逐列相对误差 < 1e-6;scipy least_squares 默认的 2-point
    前向差分(rel_step = √机器精度)在 60 个随机行列位置逐项一致。"""
    obs, model, params, rvecs, tvecs = _random_truth_obs(model_id, n_views=2, seed=13)
    residual_fn = make_residual_fn(model, obs)
    jac_fn = make_jacobian_fn(model, obs)
    theta = pack_theta(model, params, rvecs, tvecs)

    jac_ana = jac_fn(theta).toarray()
    n_rows, n_cols = jac_ana.shape
    n_points = obs.object_points.shape[0]
    assert n_rows == 2 * 2 * n_points, \
        f"{model_id} 雅可比行数应为 2·V·P = {2 * 2 * n_points},实际 {n_rows}"
    assert n_cols == len(model.param_names) + 6 * 2

    # 基准一:numdifftools(理查森外推)——逐列相对误差 < 1e-6
    jac_nd = nd.Jacobian(residual_fn)(theta)
    for j in range(n_cols):
        denom = max(float(np.linalg.norm(jac_nd[:, j])), 1e-12)
        rel = float(np.linalg.norm(jac_ana[:, j] - jac_nd[:, j])) / denom
        assert rel < 1e-6, \
            (f"{model_id} 雅可比第 {j}/{n_cols} 列对 numdifftools 基准的"
             f"列相对误差 {rel:.2e} 超线 1e-6(公式或列序有误)")

    # 基准二:2-point 前向差分(scipy least_squares 的 jac='2-point' 缺省方案,
    # rel_step = √机器精度),随机抽 60 个行列位置逐项比对——行序错位现形处。
    # 容差说明:前向差分自带截断误差 ~(h/2)·f''(h≈1.5e-8),畸变列的 f''
    # 可到几百,实测偏差约 2e-6;行序错位则是"拿到别行的值",偏差 O(1)。
    # 两者差 4 个量级,1e-4 的线既放行截断误差又不放过任何错位。
    rng = np.random.default_rng(99)
    f0 = residual_fn(theta)
    machine_eps = np.finfo(np.float64).eps
    for _ in range(60):
        rr = int(rng.integers(n_rows))
        cc = int(rng.integers(n_cols))
        step = machine_eps ** 0.5 * max(1.0, abs(theta[cc]))
        bumped = theta.copy()
        bumped[cc] += step
        numeric = (residual_fn(bumped) - f0)[rr] / step
        analytic = jac_ana[rr, cc]
        assert abs(analytic - numeric) <= 1e-4 * max(abs(analytic), abs(numeric), 1.0), \
            (f"{model_id} 雅可比位置(行 {rr},列 {cc})解析值 {analytic:.6e} 与 "
             f"2-point 差分 {numeric:.6e} 不一致——行/列序错位或公式有误")


# ---- 行为 3:行序契约与单视图局部性 ----

def test_residual_row_order_and_single_view_locality():
    """残差按 视图→点→(u,v) 两行交错;改一个视图的 tvec 只影响该视图的行
    (残差与雅可比都逐位锁定)。"""
    obs, model, params, rvecs, tvecs = _random_truth_obs("m1", n_views=2, seed=17)
    n_points = obs.object_points.shape[0]
    residual_fn = make_residual_fn(model, obs)
    jac_fn = make_jacobian_fn(model, obs)

    theta = pack_theta(model, params, rvecs, tvecs)
    residual = residual_fn(theta)
    assert residual.shape == (2 * 2 * n_points,)

    # 与逐视图手工投影逐位一致(交错 = (P,2) 数组直接拉平)
    for i in range(2):
        pts_cam, _pose_block = transform_to_camera(
            obs.object_points, rvecs[i], tvecs[i])
        manual = (model.project(pts_cam, params) - obs.image_points[i]).ravel()
        np.testing.assert_array_equal(
            residual[i * 2 * n_points:(i + 1) * 2 * n_points], manual)

    # row_index 函数与实际排布一致:u 行 = 偶行,v 行 = 奇行
    assert row_index(0, 5, 0, n_points) == 2 * 5
    assert row_index(0, 5, 1, n_points) == 2 * 5 + 1
    assert row_index(1, 3, 1, n_points) == 2 * n_points + 2 * 3 + 1

    # 改视图 1 的 tvec:视图 0 的残差行与雅可比行逐位不动,视图 1 的行变
    tvecs_moved = tvecs.copy()
    tvecs_moved[1] = tvecs[1] + np.array([0.5, -0.3, 1.0])
    theta_moved = pack_theta(model, params, rvecs, tvecs_moved)
    residual_moved = residual_fn(theta_moved)
    jac_before = jac_fn(theta).toarray()
    jac_after = jac_fn(theta_moved).toarray()
    view0 = slice(0, 2 * n_points)
    np.testing.assert_array_equal(residual_moved[view0], residual[view0])
    np.testing.assert_array_equal(jac_after[view0], jac_before[view0])
    assert not np.array_equal(residual_moved[2 * n_points:], residual[2 * n_points:]), \
        "改视图 1 的 tvec 理应改变视图 1 的残差行"
    assert not np.array_equal(jac_after[2 * n_points:], jac_before[2 * n_points:]), \
        "改视图 1 的 tvec 理应改变视图 1 的雅可比行"


# ---- 行为 4:稀疏结构吻合 ----

def test_sparsity_structure_matches_jacobian_nonzeros():
    """make_sparsity 的 0/1 结构覆盖雅可比全部实际非零位置;每个残差行
    恰好 P 个内参列 + 本视图 6 个外参列;别的视图外参列结构为零。"""
    obs, model, params, rvecs, tvecs = _random_truth_obs("m2", n_views=2, seed=23)
    n_points = obs.object_points.shape[0]
    n_params = len(model.param_names)
    sparsity = make_sparsity(model, 2, n_points)
    jac = make_jacobian_fn(model, obs)(
        pack_theta(model, params, rvecs, tvecs))

    dense_structure = sparsity.toarray()
    assert set(np.unique(dense_structure)) <= {0, 1}, "结构矩阵只能是 0/1"

    # 实际非零 ⊆ 结构(结构 = "允许非零"的位置清单)
    pattern = jac != 0
    covered = pattern.multiply(sparsity)
    assert covered.nnz == pattern.nnz, \
        f"雅可比有 {pattern.nnz - covered.nnz} 个非零元素落在结构矩阵之外"

    # 每残差行恰好 P + 6 个结构非零
    row_nnz = np.asarray(sparsity.sum(axis=1)).ravel()
    assert (row_nnz == n_params + 6).all(), \
        f"结构行非零数应全为 {n_params + 6}(P 内参 + 6 外参),实际出现别的值"

    # 视图 i 的行里,视图 (1-i) 的外参列必须结构为零
    for i in range(2):
        other_view_cols = slice(n_params + 6 * (1 - i), n_params + 6 * (2 - i))
        rows_of_view = slice(2 * n_points * i, 2 * n_points * (i + 1))
        assert not dense_structure[rows_of_view, other_view_cols].any(), \
            f"视图 {i} 的残差行里出现了视图 {1 - i} 的外参列(结构串视图)"


# ---- 性能行为代理(禁逐点循环) ----

def test_jacobian_single_call_fast_on_8_views():
    """8 视图单次雅可比 < 1 秒:逐点 Python 循环组装(640×88×30 次追加的
    写法)单次要 1 秒量级,向量化组装应为毫秒级——这是 Pitfall 5 的回归线。"""
    obs, model, params, rvecs, tvecs = _random_truth_obs("m2", n_views=8, seed=21)
    jac_fn = make_jacobian_fn(model, obs)
    theta = pack_theta(model, params, rvecs, tvecs)

    start = time.perf_counter()
    jac = jac_fn(theta)
    elapsed = time.perf_counter() - start

    assert jac.shape == (2 * 8 * 88, len(model.param_names) + 6 * 8)
    assert elapsed < 1.0, \
        f"8 视图单次雅可比耗时 {elapsed:.3f}s 超线 1s(疑似逐点 Python 循环)"


# ---- KB4 近轴保护的导数行为 ----

def test_kb4_point_jacobian_near_axis_limit():
    """KB4 像素对相机点导数在光轴附近(r<1e-12)用解析极限(du/dX=fx/z
    等针孔形态)顶上,与中心差分一致且无 NaN(研究 Pitfall 4)。"""
    model = create_model("m4")
    params = np.asarray(model.default_params((640, 480)), dtype=np.float64)
    params[4:] = [0.05, -0.02, 0.005, 0.001]
    pts = np.array([[0.0, 0.0, 800.0],
                    [1e-13, 0.0, 800.0],
                    [0.0, 1e-13, 800.0],
                    [3e-13, -2e-13, 800.0]])

    jac = model.project_point_jacobian(pts, params)
    assert np.isfinite(jac).all(), "近轴点的导数出现 NaN/Inf(0/0 未加保护)"

    step = 1e-6  # 差分点 r≈1e-6 停在正常分支,极限连续,两侧应一致
    for k, pt in enumerate(pts):
        for axis in range(3):
            bumped = np.repeat(pt[None, :], 2, axis=0)
            bumped[0, axis] += step
            bumped[1, axis] -= step
            proj = model.project(bumped, params)
            numeric = (proj[0] - proj[1]) / (2.0 * step)  # (2,) = [du/d轴, dv/d轴]
            np.testing.assert_allclose(
                jac[k, :, axis], numeric, rtol=1e-6, atol=1e-9,
                err_msg=f"KB4 近轴点 {pt} 的像素导数(第 {axis} 轴)对中心差分不一致")


# ---- D-08:残差单一来源(投影只经模型层)----

def test_residual_single_source_monkeypatch_lock(monkeypatch):
    """D-08 行为锁:给 model.project 的返回值整体加 +0.25 像素(u 分量),
    残差的全部 u 行(偶数行)必须恰好平移 0.25、v 行不动——证明残差
    逐点调用的投影就是模型层的 project,别处没有第二份投影数学。
    测试末尾显式还原并复核残差回到原值,保证没留下污染。"""
    obs, model, params, rvecs, tvecs = _random_truth_obs("m2", n_views=2, seed=29)
    residual_fn = make_residual_fn(model, obs)
    theta = pack_theta(model, params, rvecs, tvecs)
    before = residual_fn(theta)

    original_project = model.project

    def shifted_project(pts_cam, params_):
        uv = original_project(pts_cam, params_)
        return uv + np.array([0.25, 0.0])  # 只动 u,留 v 当对照

    monkeypatch.setattr(model, "project", shifted_project)
    after = residual_fn(theta)

    diff = after - before
    np.testing.assert_allclose(diff[0::2], 0.25, rtol=0, atol=1e-12,
                               err_msg="投影加 0.25px,u 行残差应恰好平移 0.25")
    np.testing.assert_allclose(diff[1::2], 0.0, atol=1e-12,
                               err_msg="投影只动了 u 分量,v 行残差不该动")

    # 显式还原后复核:残差回到原值(替身没有留下任何痕迹)
    monkeypatch.undo()
    np.testing.assert_array_equal(residual_fn(theta), before)


def test_scipy_side_sources_have_no_opencv_projection_calls():
    """D-08 源码级锁定:residuals.py 与 scipy_solver.py 的源码里不出现
    OpenCV 投影函数的名字——scipy 侧的投影只准经模型层,第二份投影
    数学(哪怕是借 OpenCV 的)都不许出现,出现即打破"理论算的和求解
    用的同一套数学"。"""
    for name in ("residuals.py", "scipy_solver.py"):
        source = (REPO_ROOT / "src" / "calib" / "solve" / name).read_text(
            encoding="utf-8")
        assert "projectPoints" not in source, \
            f"{name} 出现了 OpenCV 投影函数名——投影必须只经模型层(D-08)"


def test_numdifftools_never_in_production_sources():
    """D-10 源码级锁定:numdifftools(数值差分基准)只准进 tests/,solve/
    与 models/ 的产品代码里出现即违规——产品路径的雅可比必须是解析式,
    差分只配当测试里的对照答案。"""
    for module in ("solve", "models"):
        for py_file in sorted((REPO_ROOT / "src" / "calib" / module).rglob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            assert "numdifftools" not in source, \
                f"{py_file.relative_to(REPO_ROOT)} 出现 numdifftools——" \
                f"数值差分只准进测试,产品路径雅可比必须解析式(D-10)"
