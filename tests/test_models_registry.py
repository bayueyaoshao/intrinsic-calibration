"""注册机制零侵入验证(D-09):一个"玩具模型"注册进全局表,两个求解
后端**不改一行代码**就能用它求解。

做法:定义 ToyPinhole(纯针孔无畸变,教科书公式,**不提供** OpenCV 适配器)
并在类头登记 @register_model——之后 solve_with_opencv 与 solve_with_scipy
都只通过 create_model("toy_pinhole") 拿实例:
- OpenCV 后端自动落进通用初始化兜底(default_params 组内参 K + 逐视图
  位姿估计),不需要玩具模型迁就任何 OpenCV 约定;
- scipy 后端同样从通用初始化起步,解析雅可比直接用玩具模型自己写的
  导数——"一处注册、两个求解器都能用"就是注册机制的全部意义。

卫生条款:玩具模型只在**本文件**的测试期间存在于注册表里——模块级
夹具在收尾时把它从表里摘除,文件最后一个测试先显式验证"摘除后注册表
回到只剩四个内置模型",保证测试不污染全局状态(别的测试文件看得到
同一张表)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calib.io.config import BoardSpec, ScipyConfig
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
from calib.solve.opencv import solve_with_opencv
from calib.solve.scipy_solver import solve_with_scipy

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="module")
def _restore_registry_after_module():
    """本模块收尾兜底:把玩具模型从全局注册表摘除(幂等;最后一个测试
    通常已经摘过,这里再兜一次,保证无论测试怎么跳过都不留污染)。"""
    yield
    _REGISTRY.pop("toy_pinhole", None)


@register_model
class ToyPinhole(CameraModel):
    """玩具模型:纯针孔无畸变(教科书公式),刻意不提供 OpenCV 适配器。

    只为验证注册机制的零侵入(D-09):求解器代码不认识任何具体模型,
    新模型实现 CameraModel 协议并登记,就能被两个后端直接使用。
    """

    model_id = "toy_pinhole"
    description = "测试玩具:纯针孔无畸变(不提供 OpenCV 适配器)"
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
        # 纯针孔无畸变:du/dX=fx/z、du/dZ=−fx·x/z²(v 对应 fy),教科书写照
        pts = np.asarray(pts_cam, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64).ravel()
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        jac = np.empty((pts.shape[0], 2, 3), dtype=np.float64)
        jac[:, 0, 0] = p[0] / z
        jac[:, 0, 1] = 0.0
        jac[:, 0, 2] = -p[0] * x / (z * z)
        jac[:, 1, 0] = 0.0
        jac[:, 1, 1] = p[1] / z
        jac[:, 1, 2] = -p[1] * y / (z * z)
        return jac

    def unproject(self, uv: np.ndarray, params: np.ndarray) -> np.ndarray:
        # 无畸变针孔的逆是闭式解:像素减主点除焦距,归一化成单位射线
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


def _toy_obs(n_views: int = 5, seed: int = 11):
    """合成玩具观测集:真值内参 = 玩具模型自己的 default_params((640,480))
    (fx=fy=480、主点居中),位姿小转角随机,角点由玩具模型精确投影生成
    (无噪声)——数据与模型完全同构,回代 RMS 理应近零。"""
    rng = np.random.default_rng(seed)
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    objp = build_object_points(board)
    model = create_model("toy_pinhole")
    params = np.asarray(model.default_params((640, 480)), dtype=np.float64)
    rvecs = rng.uniform(-0.12, 0.12, size=(n_views, 3))
    tvecs = np.stack([rng.uniform(-30.0, 30.0, size=n_views),
                      rng.uniform(-30.0, 30.0, size=n_views),
                      np.full(n_views, 800.0)], axis=1)
    image_points = np.empty((n_views, objp.shape[0], 2), dtype=np.float64)
    for i in range(n_views):
        pts_cam, _pose_block = transform_to_camera(objp, rvecs[i], tvecs[i])
        image_points[i] = model.project(pts_cam, params)
    obs = ObservationSet(
        image_ids=tuple(f"toy_{k:02d}" for k in range(n_views)),
        pose_ids=tuple(f"toy_pose_{k:02d}" for k in range(n_views)),
        object_points=objp,
        image_points=image_points,
        image_size=(640, 480),
        corner_source=("sb",) * n_views,
    )
    return obs, model, params


# ---- 注册机制本体 ----

def test_toy_model_registered_and_instantiated():
    """登记即可查:create_model("toy_pinhole") 拿到实例,描述与参数名
    逐字保留;四个内置模型仍在,注册只添不删。"""
    assert "toy_pinhole" in registered_model_ids()
    assert set(BUILTIN_MODEL_IDS) <= set(registered_model_ids())

    model = create_model("toy_pinhole")
    assert model.model_id == "toy_pinhole"
    assert model.description == "测试玩具:纯针孔无畸变(不提供 OpenCV 适配器)"
    assert model.param_names == ("fx", "fy", "cx", "cy")
    assert model.opencv_adapter() is None, "玩具模型刻意不提供 OpenCV 适配器"


def test_duplicate_registration_raises_chinese_error():
    """同名模型重复注册:当场中文报错拦下(两个类抢一个短名,后续按名
    查表会拿到说不清的那一个)。"""
    with pytest.raises(ValueError, match="重复注册"):
        @register_model
        class _ToyAgain(ToyPinhole):
            pass


# ---- 双求解器零侵入(D-09 的最终证明)----

def test_opencv_backend_solves_registered_toy_model():
    """玩具模型注册后,OpenCV 后端不改一行代码即可用:无适配器自动落
    通用初始化兜底,default_params 恰是真值 → 位姿估计机器精度 →
    回代 RMS < 1px。"""
    obs, model, _params = _toy_obs()

    result = solve_with_opencv(obs, model)

    assert result.converged is True, f"玩具模型应能求解:{result.failure_note}"
    assert result.solver_meta["path"] == "generic_pnp"
    assert result.rms_overall < 1.0, \
        f"玩具模型通用初始化 RMS {result.rms_overall:.6f}px 超线 1px"
    assert set(result.params) == {"fx", "fy", "cx", "cy"}
    assert result.rvecs.shape == (len(obs.image_ids), 3)


def test_scipy_backend_solves_registered_toy_model():
    """同一玩具模型,scipy 后端同样零侵入可用:不给初值走通用初始化,
    解析雅可比用的是玩具模型自己写的导数;init_from 留痕 internal_generic。"""
    obs, model, _params = _toy_obs()

    result = solve_with_scipy(obs, model, ScipyConfig())

    assert result.converged is True, f"玩具模型 scipy 精化应实用收敛:{result.failure_note}"
    assert result.init_from == "internal_generic"
    assert result.solver == "scipy"
    assert result.rms_overall < 1.0, \
        f"玩具模型 scipy 精化 RMS {result.rms_overall:.6f}px 超线 1px"


# ---- 求解器源码不见具体模型(与 test_solve_solvers.py 的锁定互补)----

@pytest.mark.parametrize("module_name", ["opencv.py", "scipy_solver.py", "residuals.py"])
def test_solver_sources_have_no_specific_model_imports(module_name):
    """D-09 源码级锁定(补 scipy 侧):三个求解侧文件都不 import 任何
    具体模型(pinhole_brown / kb4),也不经注册表模块引入具体模型——
    只允许 CameraModel 协议,玩具模型能跑通靠的就是这条。"""
    source = (REPO_ROOT / "src" / "calib" / "solve" / module_name).read_text(
        encoding="utf-8")
    assert "from calib.models.pinhole_brown" not in source
    assert "from calib.models.kb4" not in source
    assert "from calib.models import" not in source
    assert "CameraModel" in source or module_name == "residuals.py", \
        f"{module_name} 应引用 CameraModel 协议(残差文件经类型注解引用)"


# ---- 收尾卫生:本文件必须是最后一个动注册表的测试 ----

def test_registry_restored_after_toy_module():
    """泄漏检查(本模块最后一个测试):玩具模型从注册表摘除后,表回到
    只剩四个内置模型——测试没给全局状态留下任何痕迹。"""
    _REGISTRY.pop("toy_pinhole", None)
    assert set(registered_model_ids()) == set(BUILTIN_MODEL_IDS), \
        f"摘除玩具模型后注册表应为四个内置模型,实际 {registered_model_ids()}"
