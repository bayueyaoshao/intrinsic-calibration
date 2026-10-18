"""scoring 层(逐帧回代评分 + 只估位姿小求解)的合成数据回归测试。

全部用例跑在合成数据上(内参/位姿全已知,角点由模型自己的投影函数生成,
带"标准答案")。锁定的行为清单:
- 位姿恢复:已知真值内参与位姿的合成视角,只估位姿的小求解从
  solvePnP 初值(天然偏离真值,即"扰动的初值")出发精化——回代
  RMS < 0.5px、平移恢复误差 < 深度量的 1%;
- 同一把尺:score_views 的逐帧 RMS 与 result.compute_result_metrics
  对同一输入**逐位**一致——同一把尺的两处实现不许分叉;
- 模型无关:玩具模型同样能解——残差与初值都只经 CameraModel 接口
  (scoring.py 源码无任何具体模型 import,也不出现 projectPoints);
- 毫秒级:65 个视角(真实数据 test 切分的规模)合计 < 5 秒。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from calib.io.config import BoardSpec
from calib.models import (
    _REGISTRY,
    BUILTIN_MODEL_IDS,
    create_model,
    register_model,
    registered_model_ids,
)
from calib.models.base import CameraModel, transform_to_camera
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet
from calib.solve.scoring import score_views, solve_pose_only

REPO_ROOT = Path(__file__).resolve().parents[1]

_DEPTH = 800.0  # 合成位姿的统一深度(平移误差按它的 1% 量级断言)


class ToyScoringPinhole(CameraModel):
    """玩具模型:纯针孔无畸变,定义同 test_models_registry 的 ToyPinhole
    (换短名避免与那边的注册冲突),证明只估位姿小求解对任何注册模型成立。"""

    model_id = "toy_scoring_pinhole"
    description = "scoring 测试玩具:纯针孔无畸变"
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

    def project_point_jacobian(self, pts_cam: np.ndarray,
                               params: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_cam, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64).ravel()
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        jac = np.empty((pts.shape[0], 2, 3), dtype=np.float64)
        jac[:, 0, 0] = p[0] / z
        jac[:, 0, 2] = -p[0] * x / (z * z)
        jac[:, 1, 1] = p[1] / z
        jac[:, 1, 2] = -p[1] * y / (z * z)
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


def _m2_truth_params() -> np.ndarray:
    """M2 真值内参:fx=fy=800、主点 (320,240)、带一小组真实感畸变。

    畸变非零是刻意的:PnP 初值按零畸变算,天然偏离真值——正是
    "从扰动的初值出发"的扰动来源(初值由 solvePnP 在函数内部给出)。
    """
    return np.array([800.0, 800.0, 320.0, 240.0,
                     0.05, -0.02, 0.001, -0.001, 0.003], dtype=np.float64)


def _truth_pose(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """一组真值位姿:小转角 + 深度 800(与 make_true_views 夹具同量级)。"""
    rvec = np.array([rng.uniform(-0.15, 0.15),
                     rng.uniform(-0.10, 0.20),
                     rng.uniform(-0.15, 0.15)])
    tvec = np.array([rng.uniform(-30.0, 30.0),
                     rng.uniform(-30.0, 30.0), _DEPTH])
    return rvec, tvec


def _project_view(objp: np.ndarray, params: np.ndarray,
                  rvec: np.ndarray, tvec: np.ndarray,
                  model: CameraModel,
                  noise_px: float = 0.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """用模型自己的投影函数造一帧"标准答案"角点(可加高斯噪声)。"""
    pts_cam, _pose_block = transform_to_camera(objp, rvec, tvec)
    uv = model.project(pts_cam, params)
    if noise_px > 0.0:
        assert rng is not None, "加噪声必须同时给随机数发生器"
        uv = uv + rng.normal(0.0, noise_px, size=uv.shape)
    return uv


def _synthetic_obs(model: CameraModel, params: np.ndarray,
                   n_views: int, seed: int = 5) -> ObservationSet:
    """造一个多视角合成观测集(角点全部由模型投影生成,无噪声)。"""
    rng = np.random.default_rng(seed)
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    objp = build_object_points(board)
    rvecs = np.empty((n_views, 3), dtype=np.float64)
    tvecs = np.empty((n_views, 3), dtype=np.float64)
    image_points = np.empty((n_views, objp.shape[0], 2), dtype=np.float64)
    for i in range(n_views):
        rvecs[i], tvecs[i] = _truth_pose(rng)
        image_points[i] = _project_view(objp, params, rvecs[i], tvecs[i], model)
    ids = tuple(f"syn_{k:02d}" for k in range(n_views))
    return ObservationSet(
        image_ids=ids,
        pose_ids=tuple(f"pose_{k:02d}" for k in range(n_views)),
        object_points=objp,
        image_points=image_points,
        image_size=(640, 480),
        corner_source=("sb",) * n_views,
    ), objp, rvecs, tvecs


# ---- 行为 1:位姿恢复(从 solvePnP 扰动初值出发) ----

def test_solve_pose_only_recovers_known_truth():
    """真值内参 + 真值位姿的合成视角:回代 RMS < 0.5px,
    平移恢复误差 < 深度的 1%(800 × 1% = 8 个单位)。"""
    model = create_model("m2")
    params = _m2_truth_params()
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    objp = build_object_points(board)
    rng = np.random.default_rng(17)
    rvec_true, tvec_true = _truth_pose(rng)
    uv = _project_view(objp, params, rvec_true, tvec_true, model,
                       noise_px=0.1, rng=rng)  # 0.1px 噪声:贴近实测角点质量

    rvec, tvec, rms = solve_pose_only(uv, objp, params, model)

    assert rms < 0.5, f"回代 RMS {rms:.4f}px 超线 0.5px(角点带 0.1px 噪声)"
    rot_err = float(np.linalg.norm(rvec - rvec_true))
    assert rot_err < 5e-3, f"旋转恢复误差 {rot_err:.2e} rad 超线 5e-3"
    t_err = float(np.abs(tvec - tvec_true).max())
    assert t_err < 0.01 * _DEPTH, \
        f"平移恢复误差 {t_err:.3f} 超深度量的 1%({0.01 * _DEPTH:.1f})"


def test_solve_pose_only_noise_free_hits_machine_precision():
    """无噪声合成视角:解应回到机器精度(RMS 远小于 1px),
    旋转恢复误差缩到 1e-6 rad 级——证明小求解本身没有系统偏差。"""
    model = create_model("m2")
    params = _m2_truth_params()
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    objp = build_object_points(board)
    rng = np.random.default_rng(23)
    rvec_true, tvec_true = _truth_pose(rng)
    uv = _project_view(objp, params, rvec_true, tvec_true, model)

    rvec, tvec, rms = solve_pose_only(uv, objp, params, model)

    assert rms < 1e-6, f"无噪声回代 RMS {rms:.2e}px 应到机器精度级"
    assert float(np.linalg.norm(rvec - rvec_true)) < 1e-6
    assert float(np.abs(tvec - tvec_true).max()) < 1e-3


# ---- 行为 2:score_views 与 compute_result_metrics 逐位一致 ----

def test_score_views_bit_identical_with_compute_result_metrics():
    """同一把尺的两处实现不许分叉:score_views 的逐帧 RMS 与
    compute_result_metrics 对同一输入**逐位**相等(array_equal)。"""
    model = create_model("m2")
    params = _m2_truth_params()
    obs, _objp, rvecs, tvecs = _synthetic_obs(model, params, n_views=4)

    per_view = score_views(obs, params, rvecs, tvecs, model)

    from calib.solve.result import compute_result_metrics
    _rms_all, reference, _res = compute_result_metrics(
        obs, params, rvecs, tvecs, model)
    assert per_view.shape == (len(obs.image_ids),)
    assert np.array_equal(per_view, reference), \
        "score_views 与 compute_result_metrics 的逐帧 RMS 必须逐位一致" \
        "(同一把尺不许有两份实现)"


# ---- 行为 3:玩具模型同样可用(残差与初值都只经 CameraModel 接口) ----

def test_solve_pose_only_works_for_registered_toy_model():
    """玩具模型注册进全局表后,只估位姿小求解不改一行代码即可用:
    角点由玩具模型投影生成,恢复的位姿与回代 RMS 都到机器精度。

    注册放在测试体内(不用装饰器):模块导入即注册会让玩具模型
    提前污染全局注册表,踩中 test_models_registry 的泄漏检查——
    用完即摘,卫生条款照那边的体例。"""
    register_model(ToyScoringPinhole)
    try:
        model = create_model("toy_scoring_pinhole")
        params = np.asarray(model.default_params((640, 480)), dtype=np.float64)
        board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
        objp = build_object_points(board)
        rng = np.random.default_rng(29)
        rvec_true, tvec_true = _truth_pose(rng)
        uv = _project_view(objp, params, rvec_true, tvec_true, model)

        rvec, tvec, rms = solve_pose_only(uv, objp, params, model)
    finally:
        _REGISTRY.pop("toy_scoring_pinhole", None)

    assert rms < 1e-6, f"玩具模型回代 RMS {rms:.2e}px 应到机器精度级"
    assert float(np.linalg.norm(rvec - rvec_true)) < 1e-6
    assert float(np.abs(tvec - tvec_true).max()) < 1e-3
    # 卫生:玩具模型已摘除,注册表回到只剩四个内置模型
    assert set(registered_model_ids()) == set(BUILTIN_MODEL_IDS)


def test_scoring_source_has_no_specific_models_or_projectpoints():
    """源码级锁定(D-09 延伸到评分层):scoring.py 不 import 任何具体
    模型、不出现 projectPoints——投影与初值的 K 全部只经
    model.project 与 model.intrinsics_matrix 唯一路径。"""
    source = (REPO_ROOT / "src/calib/solve/scoring.py").read_text(encoding="utf-8")
    assert "from calib.models.pinhole_brown" not in source
    assert "from calib.models.kb4" not in source
    assert "from calib.models import" not in source
    assert "projectPoints" not in source, \
        "scoring.py 不得出现 projectPoints:投影只准经 model.project"
    assert "intrinsics_matrix" in source, \
        "solvePnP 的初值 K 必须经 model.intrinsics_matrix 组出"
    assert "CameraModel" in source


# ---- 行为 4:毫秒级(65 视角合计 < 5 秒) ----

def test_65_views_pose_only_under_five_seconds():
    """真实数据 test 切分的规模(65 张)合计 < 5 秒——每帧毫秒级的粗界。"""
    model = create_model("m2")
    params = _m2_truth_params()
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    objp = build_object_points(board)
    rng = np.random.default_rng(31)

    views = []
    truths = []
    for _k in range(65):
        rvec_true, tvec_true = _truth_pose(rng)
        views.append(_project_view(objp, params, rvec_true, tvec_true, model,
                                   noise_px=0.1, rng=rng))
        truths.append((rvec_true, tvec_true))

    start = time.perf_counter()
    rms_list = []
    for uv, (rvec_true, _t_true) in zip(views, truths):
        _rvec, _tvec, rms = solve_pose_only(uv, objp, params, model)
        rms_list.append(rms)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"65 视角合计耗时 {elapsed:.2f} 秒,超 5 秒粗界"
    assert max(rms_list) < 0.5, \
        f"65 视角里最大回代 RMS {max(rms_list):.4f}px 超线 0.5px"
