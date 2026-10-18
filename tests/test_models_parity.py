"""模型层与 OpenCV 的一致性比对测试(容差矩阵):四模型投影逐点对齐。

锁定的决策与量级(D-06/D-07/D-10,容差依据):
- D-06:models/ 是全项目唯一数学源——本测试就是"数学源与 OpenCV 对齐"
  的锁,先于任何求解代码存在(阶段二 02-01);
- D-07:M1/M2/M3 对 cv2.projectPoints、M4 对 cv2.fisheye.projectPoints;
- D-10:numdifftools 只进雅可比测试文件(见 test_models_jacobian.py),
  本文件不涉及;
- 容差:实测偏差 ~5e-13 像素(机器精度量级),断言线 1e-9 留 3 个
  数量级余量——走样到 1e-4 量级即说明混进了 float32 路径(Pitfall 1);
- float32 路径单独锁定:OpenCV 绑定层 32 位输入与 64 位差 ~1e-4px,
  sanity 断言 1e-3(把"有差但有限"这一事实钉死,不是精度要求);
- 参数采样:典型(本数据量级 fx≈1872/宽 2560)/ 边界(M3 的 k4~k6≠0、
  p1p2≠0、M4 大 θ)/ 随机(种子 ≥3);
- 点集:随机 1000 点 + 近轴点(归一化 r≈1e-6,锁 KB4 除零保护)+
  大半径点(针孔族归一化 r≈1.2、KB4 θ≈1.3 rad);
- 适配器装包→解包往返逐位相等(锁 Pitfall 2 的 D 长度切片);
- 反投影往返:project → unproject → project 误差 < 1e-9 像素
  (研究遗留问题 4 的定夺:真逆解,不是占位)。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calib.models import BUILTIN_MODEL_IDS, create_model

# ---------- 参数采样:三组(典型 / 边界 / 随机种子) ----------

_TYPICAL = {
    "m1": (1872.0, 1872.5, 1280.2, 721.8, -0.13, 0.028, 0.0004, -0.0003),
    "m2": (1872.0, 1872.5, 1280.2, 721.8, -0.13, 0.028, 0.0004, -0.0003, -0.004),
    "m3": (1872.0, 1872.5, 1280.2, 721.8, -0.13, 0.028, 0.0004, -0.0003, -0.004,
           0.012, -0.021, 0.009),
    "m4": (410.0, 410.5, 1279.6, 720.9, 0.05, -0.02, 0.005, 0.001),
}

_BOUNDARY = {
    "m1": (1872.0, 1872.5, 1280.2, 721.8, -0.35, 0.18, 0.003, -0.0025),
    "m2": (1872.0, 1872.5, 1280.2, 721.8, -0.35, 0.18, 0.003, -0.0025, 0.06),
    "m3": (1872.0, 1872.5, 1280.2, 721.8, -0.35, 0.18, 0.003, -0.0025, 0.06,
           0.05, -0.08, 0.03),  # k4~k6 显著非 0:有理分母路径的边界
    "m4": (410.0, 410.5, 1279.6, 720.9, 0.35, -0.28, 0.16, -0.05),  # 强畸变大 θ 配套
}


def _random_params(model_id: str, seed: int) -> np.ndarray:
    """随机参数组:量级参考本数据(针孔 fx~1872/2560 宽、鱼眼 fx~410)。"""
    rng = np.random.default_rng(seed)
    p = np.zeros(len(create_model(model_id).param_names), dtype=np.float64)
    if model_id == "m4":
        p[0], p[1] = rng.uniform(350, 450, 2)
        p[2], p[3] = rng.uniform(1200, 1360), rng.uniform(650, 790)
        p[4:] = rng.uniform(-0.35, 0.25, 4)
    else:
        p[0], p[1] = rng.uniform(1500, 2200, 2)
        p[2], p[3] = rng.uniform(1200, 1360), rng.uniform(650, 790)
        p[4:] = rng.uniform(-0.4, 0.25, len(p) - 4)
        p[6:8] = rng.uniform(-0.01, 0.01, 2)  # 切向系数量级远小于径向
    return p


def _params(model_id: str, kind: str) -> np.ndarray:
    if kind == "typical":
        return np.array(_TYPICAL[model_id], dtype=np.float64)
    if kind == "boundary":
        return np.array(_BOUNDARY[model_id], dtype=np.float64)
    raise ValueError(kind)


# ---------- 点集:随机 + 近轴 + 大半径(形状统一 (N,3) float64) ----------

def _make_points(model_id: str, seed: int = 2026, n_random: int = 1000) -> np.ndarray:
    """随机 n_random 点 + 近轴 r≈1e-6(含正原点)+ 大半径(针孔 r≈1.2 / KB4 θ≈1.3)。"""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, size=n_random)
    if model_id == "m4":
        theta = rng.uniform(0.0, 1.2, size=n_random)  # 入射角均匀铺
        pts = np.stack([np.sin(theta) * np.cos(ang), np.sin(theta) * np.sin(ang),
                        np.cos(theta)], axis=1)
        near_r = 1e-6
        near = np.array([[near_r, 0.0, np.sqrt(1.0 - near_r**2)],
                         [0.0, -near_r, np.sqrt(1.0 - near_r**2)],
                         [0.0, 0.0, 1.0]])  # 正原点:除零保护的最狠考题
        big = np.array([[np.sin(1.3) * np.cos(a), np.sin(1.3) * np.sin(a), np.cos(1.3)]
                        for a in (0.3, 1.1, 2.2, 4.0)])
    else:
        rad = rng.uniform(0.0, 1.2, size=n_random)
        z = rng.uniform(0.3, 5.0, size=n_random)
        pts = np.stack([rad * np.cos(ang) * z, rad * np.sin(ang) * z, z], axis=1)
        near = np.array([[1e-6, 0.0, 1.0], [0.0, -1e-6, 1.0], [7e-7, 7e-7, 1.0]])
        big = np.array([[1.2 * np.cos(a), 1.2 * np.sin(a), 1.0]
                        for a in (0.4, 1.3, 2.6, 4.4)])
    return np.vstack([pts, near, big])


def _cv_project(model_id: str, pts: np.ndarray, params: np.ndarray,
                dtype=np.float64) -> np.ndarray:
    """OpenCV 侧投影:一律 (N,1,3) 形状进、(N,2) 出,float64 为准。"""
    model = create_model(model_id)
    if model_id == "m4":
        k = model.intrinsics_matrix(params).astype(dtype)
        d = params[4:].astype(dtype)
        uv, _ = cv2.fisheye.projectPoints(
            pts.astype(dtype).reshape(-1, 1, 3),
            np.zeros(3), np.zeros(3), k, d)
    else:
        n_dist = {"m1": 4, "m2": 5, "m3": 8}[model_id]
        d = np.zeros(8, dtype=dtype)
        d[:n_dist] = params[4:]
        uv, _ = cv2.projectPoints(
            pts.astype(dtype).reshape(-1, 1, 3),
            np.zeros(3), np.zeros(3),
            model.intrinsics_matrix(params).astype(dtype), d)
    return uv.reshape(-1, 2).astype(np.float64)


# ---------- parity:逐模型逐参数组 ----------

@pytest.mark.parametrize("model_id", BUILTIN_MODEL_IDS)
@pytest.mark.parametrize("kind", ("typical", "boundary"))
def test_parity_vs_opencv_float64(model_id: str, kind: str):
    """自实现投影与 OpenCV 逐点一致:float64 下 max|Δ| < 1e-9 像素。

    实测 ~5e-13(机器精度),断言线留 3 个数量级余量;偏差一旦到 1e-4
    量级,基本可以断定走了 float32 路径或公式嵌套错了(Pitfall 1/11)。
    """
    params = _params(model_id, kind)
    pts = _make_points(model_id)
    ours = create_model(model_id).project(pts, params)
    ref = _cv_project(model_id, pts, params)
    assert ours.shape == (pts.shape[0], 2)
    err = float(np.abs(ours - ref).max())
    assert err < 1e-9, \
        (f"模型 {model_id}({kind} 参数)与 OpenCV 投影偏差 {err:.2e}px 超线"
         f"(断言线 1e-9px;若在 1e-4px 量级,先查 float32 混入)")


@pytest.mark.parametrize("model_id", BUILTIN_MODEL_IDS)
@pytest.mark.parametrize("seed", (1, 2, 3))
def test_parity_vs_opencv_random_params(model_id: str, seed: int):
    """随机参数组(种子 ≥3)同样逐点一致:防"只在手挑参数上对齐"的假阳性。"""
    params = _random_params(model_id, seed)
    pts = _make_points(model_id, seed=seed + 100)
    ours = create_model(model_id).project(pts, params)
    ref = _cv_project(model_id, pts, params)
    err = float(np.abs(ours - ref).max())
    assert err < 1e-9, \
        f"模型 {model_id}(随机种子 {seed})与 OpenCV 偏差 {err:.2e}px 超线 1e-9px"


@pytest.mark.parametrize("model_id", ("m1", "m2", "m3"))
def test_float32_binding_sanity(model_id: str):
    """float32 输入路径的 sanity:OpenCV 绑定层 32 位行为差 ~1e-4px,锁 < 1e-3。

    这不是精度要求,是把"32 位输入确实有 1e-4px 量级差异"这一实测事实
    钉进测试(Pitfall 1):差异一旦超出 1e-3,说明绑定层行为变了。
    fisheye 系函数项目规则只走 float64,不在此列。
    """
    params = _params(model_id, "boundary")
    pts = _make_points(model_id)
    ours = create_model(model_id).project(pts, params)
    ref32 = _cv_project(model_id, pts, params, dtype=np.float32)
    err = float(np.abs(ours - ref32).max())
    assert err < 1e-3, \
        f"模型 {model_id} 的 float32 路径偏差 {err:.2e}px 超线 1e-3px(绑定层行为变化?)"


@pytest.mark.parametrize("model_id", BUILTIN_MODEL_IDS)
def test_adapter_roundtrip_bitexact(model_id: str):
    """适配器装包→解包往返逐位相等(锁 Pitfall 2:D 长度切片)。

    M3 特意模拟 OpenCV 的真实返回:RATIONAL_MODEL 下 D 是 14 长
    (尾部 6 个是未启用的 thin-prism/tilted 恒 0),解包必须切前 8。
    """
    model = create_model(model_id)
    adapter = model.opencv_adapter()
    assert adapter is not None, f"{model_id} 应提供 OpenCV 适配器"
    for kind in ("typical", "boundary"):
        params = _params(model_id, kind)
        k, d = adapter.params_to_cv(params)
        if model_id == "m3":
            d = np.pad(d, (0, 6))  # 模拟 OpenCV 返回的 14 长畸变向量
        back = adapter.params_from_cv(k, d)
        assert np.array_equal(params, back), \
            (f"模型 {model_id}({kind})适配器往返不相等:装包前 {params},"
             f"解包后 {back}——D 长度切片规则被破坏")


@pytest.mark.parametrize("model_id", BUILTIN_MODEL_IDS)
@pytest.mark.parametrize("kind", ("typical", "boundary"))
def test_unproject_roundtrip(model_id: str, kind: str):
    """反投影往返:project → unproject → project 二次投影逐点差 < 1e-9 像素。

    研究遗留问题 4 的定夺落地:四模型 unproject 都是真逆解(牛顿迭代),
    不是占位。往返通过 = 逆解在测试参数域内收敛到机器精度。
    """
    model = create_model(model_id)
    params = _params(model_id, kind)
    pts = _make_points(model_id)
    uv1 = model.project(pts, params)
    rays = model.unproject(uv1, params)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-12), \
        f"模型 {model_id} 反投影结果不是单位射线"
    uv2 = model.project(rays, params)
    err = float(np.abs(uv1 - uv2).max())
    assert err < 1e-9, \
        f"模型 {model_id}({kind})反投影往返误差 {err:.2e}px 超线 1e-9px"


def test_kb4_near_axis_points_finite():
    """KB4 近轴保护:光轴上的点(r=0 与 r≈1e-6)投影必须是有限值。

    不加 np.divide(out=1/z, where=...) 保护时,光轴点是 0/0=NaN,
    会无声毒化整个求解——本用例把它锁死(Pitfall 4 / 威胁 T-02-01-02)。
    """
    model = create_model("m4")
    params = _params("m4", "boundary")
    pts = np.array([[0.0, 0.0, 1.0], [1e-6, 0.0, 1.0], [0.0, -1e-6, 1.0],
                    [5e-13, -5e-13, 1.0]])
    uv = model.project(pts, params)
    assert uv.shape == (4, 2)
    assert bool(np.isfinite(uv).all()), f"KB4 近轴投影出了非有限值:{uv}"


def test_registry_error_lists_options():
    """create_model 未命中报中文错,含可选清单与配置段提示(MOD-01 注册基座)。"""
    with pytest.raises(ValueError, match="不认识的模型名") as exc_info:
        create_model("m0")
    msg = str(exc_info.value)
    assert "m1" in msg and "m4" in msg, f"报错应列出全部可选模型名:{msg}"
    assert "可选" in msg and "calibrate.models" in msg, \
        f"报错应含可选清单与 calibration.yaml 的 models 段提示:{msg}"
