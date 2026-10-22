"""ANA-03 test 三分量分解的行为测试:反解恒等式 + 真值场景 + 病态帧占行。

测的是什么(PLAN 05-04 Task 1 的四条 behavior + notes 断言):
1. 反解恒等式:合成列满秩 A、已知 dX0 生成残差 → 反解恢复 dX0
   (atol=1e-9);理论 σ 与手算 sqrt(diag(inv(AᵀA))) 一致。
2. 真值场景:合成真值视角 + 真值内参 → 逐帧 |dX| 各分量 < 1e-3(物点
   单位 mm)、回代 rms < 0.01 px(数学没造假的最直接证据)。
3. 病态帧:近平共线观测(AᵀA 条件数 > 1e4)→ 该帧 NaN 占行 +
   n_cond_excluded 计数,dX 不冒充数值;好帧照常有值。
4. z 分数恒等:zscores == dX / sigma_theory 逐位一致;
   depth_amplification == 1/sqrt(最小特征值) 一致。
5. notes 键存在且点名「乐观偏差」与「深度」(两条固定文案)。
"""

from __future__ import annotations

import numpy as np
import pytest

from calib.analysis.test_xyz import _frame_xyz_core, frame_xyz, run_test_xyz
from calib.io.config import BoardSpec, CalibrationConfig
from calib.models import create_model
from calib.solve.observation import ObservationSet

# 与 conftest.make_true_views 的工厂完全同款:11×8 = 88 角点、方格物理
# 边长 20mm(square_px=20 × depth 800 ÷ fx 800 反推所得)、棋盘中心对齐光轴
_COLS, _ROWS, _SQUARE_MM = 11, 8, 20.0


def _centered_objp() -> np.ndarray:
    """make_chessboard_image 内部使用的居中物点网格(行主序,与工厂逐位一致)。"""
    xs = (np.arange(_COLS) - (_COLS - 1) / 2.0) * _SQUARE_MM
    ys = (np.arange(_ROWS) - (_ROWS - 1) / 2.0) * _SQUARE_MM
    objp = np.zeros((_ROWS * _COLS, 3), dtype=np.float64)
    objp[:, 0] = np.tile(xs, _ROWS)
    objp[:, 1] = np.repeat(ys, _COLS)
    return objp


def _m1_true_params() -> dict:
    """m1 真值内参(与 make_true_views 的 K_true 一致)+ 畸变清零。"""
    return {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}


def _true_views_at(board_z: float, n_views: int, seed: int) -> list[np.ndarray]:
    """按真值内参直接投影出合成视角(零畸变;board_z 控制棋盘距离,
    距离越远画面占比越小,AᵀA 对深度的条件数越差——病态帧测试用)。"""
    model = create_model("m1")
    params = model_params_vec(_m1_true_params(), model)
    objp = _centered_objp()
    from calib.models.base import transform_to_camera

    rng = np.random.default_rng(seed)
    views = []
    for _ in range(n_views):
        rvec = np.array([rng.uniform(-0.15, 0.15),
                         rng.uniform(-0.10, 0.20),
                         rng.uniform(-0.15, 0.15)])
        tvec = np.array([rng.uniform(-30.0, 30.0),
                         rng.uniform(-30.0, 30.0), board_z])
        pts_cam, _ = transform_to_camera(objp, rvec, tvec)
        views.append(model.project(pts_cam, params))
    return views


def model_params_vec(params_by_name: dict, model) -> np.ndarray:
    """按 param_names 布局把参数字典拼成向量(测试助手)。"""
    return np.array([float(params_by_name[name]) for name in model.param_names])


# ---------------------------------------------------------------------------
# Test 1:反解恒等式(合成 A,数学正确性的执行位)
# ---------------------------------------------------------------------------

def test_inversion_identity_recovers_dx0():
    """合成列满秩 A (176,3)、已知 dX0:resid = A@dX0 → 反解恢复 dX0。

    atol=1e-9 是线性代数恒等式的门槛(残差就是 A@dX0 本身,反解必须
    逐位找回,不允许"差不多");理论 σ 与手算 sqrt(diag(inv(AᵀA))) 一致。
    """
    rng = np.random.default_rng(2026)
    # 列各自缩放,让 AᵀA 条件数拉开(但不越 1e4 线,病态分支由下一例专测)
    scales = np.array([1.0, 4.0, 16.0])
    A = rng.normal(size=(176, 3)) * scales
    d_x0 = np.array([0.7, -1.2, 3.0])
    resid = A @ d_x0

    out = _frame_xyz_core(resid, A, cond_max=1.0e4)
    assert out["cond"] < 1.0e4, "本例构造的是良态矩阵(病态分支由下一例专测)"

    assert np.allclose(out["dX"], d_x0, atol=1e-9), \
        f"反解恒等式失败:恢复 {out['dX']} vs 真值 {d_x0}"
    ata = A.T @ A
    expected_sigma = np.sqrt(np.diag(np.linalg.inv(ata)))
    assert np.allclose(out["sigma_theory"], expected_sigma, rtol=1e-12)
    assert np.allclose(out["eigvals"], np.linalg.eigvalsh(ata), rtol=1e-12)
    assert out["cond"] == pytest.approx(float(np.linalg.cond(ata)))


def test_ill_conditioned_frame_raises_value_error():
    """病态帧的处理方式:cond 超阈值的 A 抛 ValueError(由编排层 NaN 占行),
    异常必须是 ValueError 家族(plan 契约:抛 ValueError("病态帧"))。"""
    rng = np.random.default_rng(7)
    near = rng.normal(size=(176, 3))
    near[:, 2] = near[:, 0] * (1.0 + 1e-9)  # 第三列近乎共线 → 条件数爆炸
    resid = near @ np.array([1.0, 0.5, 2.0])
    with pytest.raises(ValueError) as exc_info:
        _frame_xyz_core(resid, near, cond_max=1.0e4)
    assert "病态" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 2 + Test 4:frame_xyz 真值场景与 z 分数恒等
# ---------------------------------------------------------------------------

@pytest.fixture
def truth_setup(make_true_views):
    """真值视角 + 真值 m1 内参 + 居中物点网格(三方同源,零畸变)。"""
    views, _k = make_true_views(n_views=3)
    model = create_model("m1")
    params = model_params_vec(_m1_true_params(), model)
    return {"views": views, "model": model, "params": params,
            "objp": _centered_objp()}


def test_frame_xyz_truth_scene_near_zero(truth_setup):
    """真值场景:内参取真值 → 逐帧 |dX| 各分量 < 1e-3 mm、rms < 0.01 px。"""
    setup = truth_setup
    for view in setup["views"]:
        out = frame_xyz(view, setup["objp"], setup["params"], setup["model"])
        assert np.max(np.abs(out["dX"])) < 1e-3, \
            f"真值内参下反解位移应近零,实际 {out['dX']}"
        assert out["rms"] < 0.01, f"真值内参下回代 RMS 应 < 0.01px,实际 {out['rms']}"


def test_frame_xyz_zscore_and_amplification_identity(truth_setup):
    """z 分数恒等:zscores == dX / sigma_theory 逐位一致;
    depth_amplification == 1/sqrt(最小特征值) 一致(eigvalsh 升序,最小在首位)。"""
    setup = truth_setup
    out = frame_xyz(setup["views"][0], setup["objp"], setup["params"],
                    setup["model"])
    assert np.array_equal(out["zscores"], out["dX"] / out["sigma_theory"]), \
        "zscores 必须恰为 dX / sigma_theory(逐位,不许另算第二份)"
    assert out["depth_amplification"] == 1.0 / np.sqrt(out["eigvals"][0]), \
        "深度放大倍数必须恰为 1/sqrt(最小特征值)"
    assert np.all(np.diff(out["eigvals"]) >= 0), "特征值应升序(eigvalsh 契约)"


# ---------------------------------------------------------------------------
# Test 3 + Test 5:run_test_xyz 编排(病态帧占行 + notes 固定文案)
# ---------------------------------------------------------------------------

def _make_obs(views: list[np.ndarray], objp: np.ndarray) -> ObservationSet:
    return ObservationSet(
        image_ids=tuple(f"img{i:03d}" for i in range(len(views))),
        pose_ids=tuple(f"pose{i}" for i in range(len(views))),
        object_points=objp,
        image_points=np.asarray(views, dtype=np.float64),
        image_size=(640, 480),
        corner_source=("sb",) * len(views),
    )


def _evaluate_report_m1() -> dict:
    return {"delivery": {"models": {"m1": {"final_params": _m1_true_params()}}}}


def test_run_test_xyz_ill_conditioned_frame_placeholder():
    """病态帧占行:两张好帧(近距)+ 一张远距帧(AᵀA 条件数远超 1e4)→
    病态帧整行 NaN + n_cond_excluded 计数,dX 不冒充数值;好帧照常有值。"""
    board = BoardSpec(cols=_COLS, rows=_ROWS, square_size_mm=_SQUARE_MM)
    objp = _centered_objp()
    good = _true_views_at(800.0, 2, seed=11)
    bad = _true_views_at(8.0e5, 1, seed=12)   # 远距:画面占比极小 → cond ~1e8
    obs = _make_obs(good + bad, objp)

    section, detail = run_test_xyz(
        "m1", _evaluate_report_m1(), obs, board, CalibrationConfig())

    assert section["n_frames"] == 3
    assert section["n_cond_excluded"] >= 1, "远距帧应被判病态并单独计数"
    assert section["n_solve_failed"] == 0
    assert section["n_valid"] == 2
    assert detail["dX"].shape == (3, 3)
    assert np.all(np.isnan(detail["dX"][2])), "病态帧 dX 必须整行 NaN(不冒充数值)"
    assert np.isnan(detail["cond"][2]), "病态帧条件数也占 NaN(数值不可解释)"
    assert np.all(np.isfinite(detail["dX"][:2])), "好帧 dX 必须有值"
    # 聚合统计只来自有效帧,且各键有值
    assert section["observability"]["depth_amplification_median"] is not None
    for comp in ("x", "y", "z"):
        assert section["per_direction"]["dX_abs_mm"][comp]["median"] is not None


def test_run_test_xyz_notes_present():
    """notes 两条固定文案在盘:一条讲「深度」可观测性判读,一条声明
    test 集「乐观偏差」(选型参加过,成绩口径现象如实声明)。"""
    board = BoardSpec(cols=_COLS, rows=_ROWS, square_size_mm=_SQUARE_MM)
    obs = _make_obs(_true_views_at(800.0, 2, seed=21), _centered_objp())
    section, _detail = run_test_xyz(
        "m1", _evaluate_report_m1(), obs, board, CalibrationConfig())
    notes = " ".join(section["notes"])
    assert "乐观偏差" in notes, "notes 必须声明 test 集参加选型的乐观偏差"
    assert "深度" in notes, "notes 必须点名 Z 分量的可观测性判读"
    assert section["schema"] == "analyze-test-xyz-v1"


def test_source_single_math_source_red_line():
    """源码红线(D-06):雅可比只来自 project_point_jacobian,本模块
    自己写的只有 np.linalg 线性代数;绝不 import OpenCV、绝不手写
    d(投影)/d(相机点) 公式。"""
    import inspect

    import calib.analysis.test_xyz as mod

    source = inspect.getsource(mod)
    assert "project_point_jacobian" in source, "点雅可比必须来自 models/ 接口"
    assert "import cv2" not in source, "analysis 层绝不 import OpenCV"
    for banned in ("projectPoints", "Rodrigues", "solvePnP"):
        assert banned not in source, f"analysis 层不许出现 {banned}(位姿/投影只走既有接口)"
