"""双求解后端的结构与落盘测试(第一批:CalibrationResult + 三件套 + 统一回代尺;
第二批:OpenCVSolver 适配器分派 + 通用初始化兜底 + 合成真值恢复)。

锁定清单:
- 失败契约:converged=False 时结构键一个不少、数值字段一律 None(落盘成
  JSON null)、failure_note 非空中文——上游读取逻辑一视同仁;
- 同一把尺(Pitfall 9):rms_overall / per_view_rms 一律由 models.project
  回代计算(float64),OpenCV 内部 RMS 只进 solver_meta——两把尺子刻度差
  约 1e-4 像素(float32 输入 vs float64 回代),跨后端必须同一把尺;
- 落盘三件套:result.json / extrinsics.npz / residuals.npz 可写可读,
  失败组合照样落盘且 JSON 数值键为 null;
- OpenCV 后端(D-09):求解器只见 CameraModel 协议不 import 任何具体模型;
  无适配器的玩具模型走通用初始化路径(default_params 组 K + 逐视图
  solvePnP)同样能出结果;
- 适配器 flags 对账:M1=128(FIX_K3)/ M2=0 / M3=16384(RATIONAL)/ M4=10
  (每轮重算位姿 + skew 固定 0),记进 solver_meta。

全部用例跑在合成数据上(模型参数/位姿全已知),秒级。02-04 再扩展
scipy 侧(ScipySolver 与 D-08 残差锁定)。
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import pytest
from calib.solve.opencv import solve_with_opencv

from calib.io.config import BoardSpec
from calib.models import create_model
from calib.models.base import CameraModel, transform_to_camera
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet
from calib.solve.result import (
    CalibrationResult,
    compute_result_metrics,
    load_extrinsics,
    load_residuals,
    load_result,
    make_failure,
    save_result,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# CalibrationResult 的字段名单契约(与 02-RESEARCH §Pattern 2 清单逐一对齐;
# 增删字段都要先过这条测试——它保护"上游读取逻辑一视同仁"的失败契约)
_RESULT_FIELDS = (
    "converged", "failure_note", "model_id", "solver", "solver_meta",
    "params", "rvecs", "tvecs", "rms_overall", "per_view_rms",
    "image_ids", "image_size", "n_views", "n_points_per_view",
    "init_from", "lineage",
)

# result.json 的结构键契约:失败时也要一个不少(数值键全 null)。位姿与残差
# 不在 JSON 里——它们是 npz 三件套的另外两件(阶段一 calibration.json 体例 +
# 新增 model_id/solver/solver_meta/params/init_from/lineage)
_RESULT_JSON_KEYS = (
    "lineage", "model_id", "solver", "solver_meta", "params",
    "image_ids", "image_size", "n_views", "n_points_per_view",
    "init_from", "converged", "failure_note", "rms_overall", "per_view_rms",
)


def _synthetic_obs(n_views: int = 3, noise_px: float = 0.05, seed: int = 5):
    """合成小观测集:已知 M1 参数与逐视图位姿,投影生成角点(可加微小噪声)。

    返回 (观测集, 模型实例, 参数向量, rvecs (V,3), tvecs (V,3))。
    """
    rng = np.random.default_rng(seed)
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    objp = build_object_points(board)
    model = create_model("m1")
    params = np.array([800.0, 810.0, 330.0, 250.0, -0.05, 0.02, 0.001, -0.001])
    rvecs = np.empty((n_views, 3), dtype=np.float64)
    tvecs = np.empty((n_views, 3), dtype=np.float64)
    pts = []
    for i in range(n_views):
        rvecs[i] = rng.uniform(-0.1, 0.1, 3)
        tvecs[i] = [*rng.uniform(-30.0, 30.0, 2), 800.0]
        pts_cam, _ = transform_to_camera(objp, rvecs[i], tvecs[i])
        pts.append(model.project(pts_cam, params))
    image_points = (np.asarray(pts)
                    + rng.normal(0.0, noise_px, (n_views, objp.shape[0], 2)))
    obs = ObservationSet(
        image_ids=tuple(f"img_{k:02d}" for k in range(n_views)),
        pose_ids=tuple(f"pose_{k:02d}" for k in range(n_views)),
        object_points=objp,
        image_points=image_points,
        image_size=(640, 480),
        corner_source=("sb",) * n_views,
    )
    return obs, model, params, rvecs, tvecs


def _converged_result(obs, model, params, rvecs, tvecs, rms, per_view):
    """按收敛契约拼一个 CalibrationResult(手工造,不经求解器)。"""
    return CalibrationResult(
        converged=True, failure_note=None, model_id=model.model_id, solver="opencv",
        solver_meta={"path": "adapter:calibrateCamera", "flags_used": 128,
                     "opencv_rms": rms + 1e-4},
        params={name: float(v) for name, v in zip(model.param_names, params)},
        rvecs=rvecs, tvecs=tvecs, rms_overall=rms, per_view_rms=per_view,
        image_ids=obs.image_ids, image_size=obs.image_size,
        n_views=len(obs.image_ids), n_points_per_view=obs.object_points.shape[0],
        init_from=None, lineage={"round_id": "run_test"},
    )


# ---- 字段契约与失败契约 ----

def test_result_field_names_contract():
    """字段名单与 02-RESEARCH §Pattern 2 清单逐一对齐(dataclasses.fields 断言)。"""
    assert tuple(f.name for f in fields(CalibrationResult)) == _RESULT_FIELDS


def test_make_failure_all_numeric_fields_none():
    """make_failure:数值字段一律 None、结构键全在、note 非空。"""
    failure = make_failure("m2", "opencv", "m2 模型 OpenCV 求解未收敛:OpenCV 报错——测试注入",
                           image_ids=("a", "b"), image_size=(640, 480),
                           n_points_per_view=88)
    assert failure.converged is False
    assert failure.failure_note and failure.failure_note.strip()
    for attr in ("rvecs", "tvecs", "rms_overall", "per_view_rms", "init_from"):
        assert getattr(failure, attr) is None, f"失败时 {attr} 必须为 None"
    assert failure.params == {}          # 结构键在、无任何数值
    assert failure.solver_meta == {} and failure.lineage == {}
    assert failure.n_views == 2 and failure.n_points_per_view == 88


# ---- 落盘三件套:写读往返 ----

def test_save_load_roundtrip_bitwise(tmp_path):
    """行为 1:收敛结果落盘后,load_result 读回的 params/rms/per_view/image_ids
    与写前逐位一致(Python 的 json 浮点往返是精确的)。"""
    obs, model, params, rvecs, tvecs = _synthetic_obs()
    rms, per_view, residuals = compute_result_metrics(obs, params, rvecs, tvecs, model)
    result = _converged_result(obs, model, params, rvecs, tvecs, rms, per_view)

    out = save_result(tmp_path, result, residuals=residuals)
    loaded = load_result(out)

    assert loaded["converged"] is True
    assert loaded["params"] == result.params
    assert loaded["rms_overall"] == result.rms_overall
    assert loaded["per_view_rms"] == {i: float(r) for i, r in zip(obs.image_ids, per_view)}
    assert tuple(loaded["image_ids"]) == obs.image_ids
    assert loaded["model_id"] == "m1" and loaded["solver"] == "opencv"
    assert loaded["solver_meta"]["flags_used"] == 128


def test_failure_saves_json_null_numeric_keys(tmp_path):
    """行为 2:失败结果照样落盘——JSON 结构键一个不少、数值键为 null、
    failure_note 非空中文;extrinsics/residuals 两个 npz 不写(无数值可写)。"""
    failure = make_failure("m2", "opencv", "m2 模型 OpenCV 求解未收敛:OpenCV 报错——测试注入",
                           image_ids=("a", "b"), image_size=(640, 480),
                           n_points_per_view=88)
    out = save_result(tmp_path, failure)

    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    for key in _RESULT_JSON_KEYS:
        assert key in doc, f"失败落盘缺结构键:{key}"
    assert doc["converged"] is False
    assert doc["rms_overall"] is None and doc["per_view_rms"] is None
    assert isinstance(doc["failure_note"], str) and doc["failure_note"].strip()
    assert not (out / "extrinsics.npz").exists()
    assert not (out / "residuals.npz").exists()


def test_save_converged_without_residuals_raises(tmp_path):
    """三件套契约:收敛结果落盘必须带上残差数组(缺了直接中文报错,不静默漏写)。"""
    obs, model, params, rvecs, tvecs = _synthetic_obs()
    rms, per_view, _residuals = compute_result_metrics(obs, params, rvecs, tvecs, model)
    result = _converged_result(obs, model, params, rvecs, tvecs, rms, per_view)
    with pytest.raises(ValueError, match="残差"):
        save_result(tmp_path, result)


def test_residuals_npz_shape_and_ids(tmp_path):
    """行为 4:residuals.npz 的数组形状 (V,P,2) 且 image_ids 数组随行;
    extrinsics.npz 同样带齐 rvecs/tvecs/image_ids。"""
    obs, model, params, rvecs, tvecs = _synthetic_obs()
    rms, per_view, residuals = compute_result_metrics(obs, params, rvecs, tvecs, model)
    result = _converged_result(obs, model, params, rvecs, tvecs, rms, per_view)
    out = save_result(tmp_path, result, residuals=residuals)

    res = load_residuals(out)
    v, p = len(obs.image_ids), obs.object_points.shape[0]
    assert res["residuals"].shape == (v, p, 2)
    assert tuple(res["image_ids"]) == obs.image_ids
    np.testing.assert_allclose(res["per_view_rms"], per_view, rtol=0, atol=0)

    ext = load_extrinsics(out)
    assert ext["rvecs"].shape == (v, 3) and ext["tvecs"].shape == (v, 3)
    assert tuple(ext["image_ids"]) == obs.image_ids
    np.testing.assert_allclose(ext["rvecs"], rvecs, rtol=0, atol=0)


# ---- 同一把尺(统一回代) ----

def test_compute_metrics_matches_manual_recomputation():
    """行为 3:compute_result_metrics 的逐帧 RMS 与测试里独立复算(直接调
    model.project + transform_to_camera)的值一致;整体 RMS = 全部角点平方
    残差均值的开方。"""
    obs, model, params, rvecs, tvecs = _synthetic_obs(noise_px=0.2, seed=9)
    rms, per_view, residuals = compute_result_metrics(obs, params, rvecs, tvecs, model)

    manual = np.empty_like(residuals)
    for i in range(len(obs.image_ids)):
        pts_cam, _ = transform_to_camera(obs.object_points, rvecs[i], tvecs[i])
        manual[i] = model.project(pts_cam, params) - obs.image_points[i]
    assert np.allclose(residuals, manual, rtol=0, atol=1e-12)

    manual_per_view = np.sqrt(np.mean(np.sum(manual ** 2, axis=2), axis=1))
    assert np.allclose(per_view, manual_per_view, rtol=0, atol=1e-12)
    assert rms == pytest.approx(float(np.sqrt(np.mean(np.sum(manual ** 2, axis=2)))),
                                abs=1e-12)


def test_result_metrics_source_uses_only_models_project():
    """同一把尺的源码级锁定:compute_result_metrics 只经 model.project 与
    base.transform_to_camera 计算残差,源码里不得出现 cv2.projectPoints(Pitfall 9)。"""
    source = (REPO_ROOT / "src" / "calib" / "solve" / "result.py").read_text(encoding="utf-8")
    assert "projectPoints" not in source
    assert "transform_to_camera" in source
    assert "model.project" in source


# ---- OpenCVSolver:适配器分派 + 合成真值恢复(02-03 Task 3)----

# 四个内置模型的适配器 flags 数值契约(02-RESEARCH §Pattern 1 实测表,
# solver_meta.flags_used 必须与之逐个相等,防止 flags 悄悄漂移)
_OPENCV_FLAGS = {"m1": 128, "m2": 0, "m3": 16384, "m4": 10}


def _obs_from_views(views, image_size=(640, 480)) -> ObservationSet:
    """把合成角点列表装成观测集(编号/姿态编号按序生成,行序契约自带)。"""
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    n = len(views)
    return ObservationSet(
        image_ids=tuple(f"img_{k:02d}" for k in range(n)),
        pose_ids=tuple(f"pose_{k:02d}" for k in range(n)),
        object_points=build_object_points(board),
        image_points=np.asarray(views, dtype=np.float64),
        image_size=image_size,
        corner_source=("sb",) * n,
    )


@pytest.mark.parametrize("model_id", ["m1", "m2", "m3"])
def test_pinhole_models_recover_known_truth(make_true_views, model_id):
    """行为 1:M1/M2/M3 在 8 个真值视角上经 OpenCVSolver 求解:回代
    RMS < 0.5px 且 fx/fy/cx/cy 对真值误差 < 1%(三条各自独立用例)。"""
    views, _k_true = make_true_views()
    obs = _obs_from_views(views)

    result = solve_with_opencv(obs, create_model(model_id))

    assert result.converged is True, f"{model_id} 应收敛(failure_note={result.failure_note})"
    assert result.failure_note is None
    assert result.model_id == model_id and result.solver == "opencv"
    assert result.rms_overall < 0.5, \
        f"{model_id} 回代整体 RMS {result.rms_overall:.3f}px 超线"
    assert result.per_view_rms.shape == (len(views),)
    for name, true_value in (("fx", 800.0), ("fy", 800.0), ("cx", 320.0), ("cy", 240.0)):
        assert abs(result.params[name] - true_value) / true_value < 0.01, \
            f"{model_id} 的 {name} 对真值误差超 1%:真值 {true_value}," \
            f"估计 {result.params[name]:.3f}"
    assert result.solver_meta["flags_used"] == _OPENCV_FLAGS[model_id], \
        f"{model_id} 的 flags_used 与适配器契约不一致"


def test_kb4_synthetic_converges_or_fails_gracefully():
    """行为 2:M4 在合成 KB4 视角(焦距 204、深度 250)上求解:收敛则回代
    RMS < 0.5px;不收敛则失败契约完整——两种都算通过(KB4 初值敏感属预期)。"""
    k_true = np.array([[204.0, 0.0, 320.0],
                       [0.0, 204.0, 240.0],
                       [0.0, 0.0, 1.0]])
    d_true = np.array([0.05, -0.02, 0.005, 0.001])
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    objp = build_object_points(board).reshape(-1, 1, 3)
    rng = np.random.default_rng(23)
    views = []
    for _k in range(8):
        rvec = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.25),
                         rng.uniform(-0.2, 0.2)])
        tvec = np.array([rng.uniform(-15.0, 15.0), rng.uniform(-15.0, 15.0), 250.0])
        uv, _jac = cv2.fisheye.projectPoints(objp, rvec, tvec, k_true, d_true)
        assert np.isfinite(uv).all() and np.abs(uv - [320, 240]).max() < 300, \
            "合成鱼眼角点跑出画面,测试数据构造有误"
        views.append(uv.reshape(-1, 2))
    obs = _obs_from_views(views)

    result = solve_with_opencv(obs, create_model("m4"))

    assert result.solver_meta["flags_used"] == 10, "M4 应带 FIX_SKEW(=10)求解"
    if result.converged:
        assert result.failure_note is None
        assert result.rms_overall < 0.5, \
            f"KB4 合成回归回代 RMS {result.rms_overall:.3f}px 超线"
        assert result.per_view_rms.shape == (8,)
    else:
        note = result.failure_note
        assert isinstance(note, str) and note.strip(), "未收敛时必须有中文失败说明"
        for attr in ("rms_overall", "per_view_rms", "rvecs", "tvecs"):
            assert getattr(result, attr) is None, f"失败时 {attr} 必须为 None"
        assert result.params == {}


def test_cv2_error_returns_failure_contract(make_true_views, monkeypatch):
    """行为 3:cv2.calibrateCameraExtended 抛 cv2.error → 返回 make_failure
    结构(converged=False、中文 failure_note 含「OpenCV 报错」),不抛异常。

    替身用空输入调用原函数:空点集让 OpenCV 抛货真价实的 cv2.error
    (体例照 test_solve_baseline.py 的 WR-03 用例)。
    """
    original = cv2.calibrateCameraExtended

    def _explode(*args, **kwargs):
        return original([], [], (640, 480), None, None, flags=0)

    monkeypatch.setattr(cv2, "calibrateCameraExtended", _explode)

    views, _k_true = make_true_views(n_views=2)
    obs = _obs_from_views(views)
    result = solve_with_opencv(obs, create_model("m2"))

    assert result.converged is False
    assert isinstance(result.failure_note, str) and "OpenCV 报错" in result.failure_note
    for attr in ("rms_overall", "per_view_rms", "rvecs", "tvecs"):
        assert getattr(result, attr) is None, f"失败时 {attr} 必须为 None"
    assert result.params == {}
    assert result.image_ids == obs.image_ids
    assert result.solver_meta["flags_used"] == 0


def test_backsubstituted_rms_matches_opencv_internal(make_true_views):
    """行为 4(锁 Pitfall 9 行为):回代 RMS 与 OpenCV 内部 RMS(解包自返回值、
    记进 solver_meta)差 < 1e-3px——差值应只有 float32 装包的刻度差。"""
    views, _k_true = make_true_views()
    obs = _obs_from_views(views)
    result = solve_with_opencv(obs, create_model("m2"))

    assert result.converged is True
    internal = result.solver_meta["opencv_rms"]
    assert abs(result.rms_overall - internal) < 1e-3, \
        f"回代 RMS {result.rms_overall:.6f} 与 OpenCV 内部 {internal:.6f} 刻度差超 1e-3px"


class _ToyNoAdapterModel(CameraModel):
    """玩具模型:纯针孔无畸变,不覆写 opencv_adapter(D-09 零侵入验证用)。

    刻意不注册进模型表——solve_with_opencv 收模型实例、不查注册表,
    本类只证明"求解器对任意满足协议的模型都能工作"。
    """

    model_id = "toy_pinhole"
    description = "测试玩具模型:纯针孔无畸变,不提供 OpenCV 适配器"
    param_names = ("fx", "fy", "cx", "cy")

    def project(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_cam, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64).ravel()
        return np.stack([p[0] * pts[:, 0] / pts[:, 2] + p[2],
                         p[1] * pts[:, 1] / pts[:, 2] + p[3]], axis=1)

    def project_jacobian(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_cam, dtype=np.float64)
        jac = np.zeros((pts.shape[0], 2, 4), dtype=np.float64)
        jac[:, 0, 0] = pts[:, 0] / pts[:, 2]
        jac[:, 1, 1] = pts[:, 1] / pts[:, 2]
        jac[:, 0, 2] = 1.0
        jac[:, 1, 3] = 1.0
        return jac

    def unproject(self, uv: np.ndarray, params: np.ndarray) -> np.ndarray:
        arr = np.asarray(uv, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64).ravel()
        rays = np.stack([(arr[:, 0] - p[2]) / p[0],
                         (arr[:, 1] - p[3]) / p[1],
                         np.ones(arr.shape[0])], axis=1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def default_params(self, image_size: tuple[int, int]) -> np.ndarray:
        width, height = image_size
        return np.array([0.75 * width, 0.75 * width, width / 2.0, height / 2.0])

    def intrinsics_matrix(self, params: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=np.float64).ravel()
        return np.array([[p[0], 0.0, p[2]],
                         [0.0, p[1], p[3]],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


def test_generic_pnp_path_for_model_without_adapter(make_chessboard_image):
    """D-09 兜底:无适配器的玩具模型经通用初始化路径(default_params 组 K +
    逐视图 solvePnP 估位姿)同样能被 OpenCVSolver 求出初值结果。

    合成数据按玩具模型自己的初值规则造(K 与 default_params(640,480)
    完全一致,焦距 0.75×宽 = 480、主点居中),位姿仍由 solvePnP 估计——
    验证的是"兜底路径的数学正确",不是初值策略的优劣。
    """
    k_true = np.array([[480.0, 0.0, 320.0],
                       [0.0, 480.0, 240.0],
                       [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(3)
    views = []
    for _k in range(6):
        rvec = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.10, 0.20),
                         rng.uniform(-0.15, 0.15)])
        tvec = np.array([rng.uniform(-30.0, 30.0), rng.uniform(-30.0, 30.0), 800.0])
        _img, corners, _sq = make_chessboard_image(K=k_true, rvec=rvec, tvec=tvec)
        views.append(corners)
    obs = _obs_from_views(views)

    result = solve_with_opencv(obs, _ToyNoAdapterModel())

    assert result.converged is True, f"兜底路径应出结果:{result.failure_note}"
    assert result.solver_meta["path"] == "generic_pnp"
    assert result.rms_overall < 0.5, \
        f"玩具模型初值路径回代 RMS {result.rms_overall:.3f}px 超线"
    assert result.rvecs.shape == (len(views), 3)
    assert result.tvecs.shape == (len(views), 3)
    assert set(result.params) == {"fx", "fy", "cx", "cy"}


def test_opencv_solver_source_has_no_specific_model_imports():
    """D-09 源码级锁定:求解器不见任何具体模型——只允许 CameraModel 协议,
    绝不 import pinhole_brown / kb4,也不经注册表模块引入具体模型。"""
    source = (REPO_ROOT / "src" / "calib" / "solve" / "opencv.py").read_text(encoding="utf-8")
    assert "from calib.models.pinhole_brown" not in source
    assert "from calib.models.kb4" not in source
    assert "from calib.models import" not in source
    assert "CameraModel" in source
