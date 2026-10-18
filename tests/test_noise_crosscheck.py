"""OpenCV stdDev 交叉验证协议的锁定测试(合成 20 视图 × 88 角点,注入 σ=0.1)。

锁定的决策与量级(对应 03-04 计划的行为清单):
- 恒等参数映射:M1/M2/M3 的参数名与 OpenCV 内参标准差 18 元向量的前 P 位
  同名同序(静态锁定,防映射悄悄漂移);
- 结构比对(主判据):我们的 σ=1 精度界与 OpenCV 的 stdDeviationsIntrinsics
  拟合单一公共尺度 k 后逐参数相对偏差 < 5%;且 k ∈ (0.05, 0.2)——
  σ=0.1 注入下 k 就是 OpenCV 内部自估的噪声尺度(量级对账位);
- M1 固定位:被固定的 k3 位标准差恒 0,比对前按前 8 位对齐,无 NaN、
  无除零(M1 同样吃 < 5% 硬门槛);
- M3 分级(D-07):尽力对齐——测试只要求"有结果",不对上不失败,
  真实数据上的结论由 03-05 按发现记录;
- M4 分级(D-07):鱼眼无可比输出 → 记中文结论即算完成,绝不调用任何
  cv2.fisheye 函数(测试装监视桩证明一个都没被碰);
- 失败契约:OpenCV 报错 → status=failed + 中文说明,不抛异常——交叉验证
  是 CRLB 主产物的附加件,它失败不阻塞主产物。

头号坑(03-RESEARCH §Pattern 4 / Pitfall 1)的判读规则也锁在这里:比值
"所有参数几乎同一个数" = 尺度因子问题(数学是对的);逐参数散乱 =
数学真有问题。这就是协议要先做尺度归一的原因。

场景说明:conftest 的 make_true_views 造 20 个已知真值视角(无畸变、
fx=fy=800),角点注入 σ=0.1 的高斯噪声后装成观测集;交叉验证在 OpenCV
自己解出的参数与位姿上线性化,不需要真值位姿,所以渲染夹具够用。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from calib.io.config import BoardSpec
from calib.models import create_model
from calib.noise.crosscheck import (
    _OPENCV_INTRINSIC_ORDER,
    compare_structural,
    crosscheck_model,
    opencv_stddev_intrinsics,
)
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet

REPO_ROOT = Path(__file__).resolve().parents[1]

# 注入噪声 σ=0.1:研究实测该量级下 OpenCV 内部自估噪声尺度 ≈ 0.1,
# 拟合出的公共尺度 k 应落在 (0.05, 0.2)(量级对账区间)
_NOISE_PX = 0.1
_BOARD = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
_N_POINTS = _BOARD.cols * _BOARD.rows


def _noisy_obs(make_true_views, n_views: int = 20,
               noise_px: float = _NOISE_PX, seed: int = 11) -> ObservationSet:
    """合成带噪观测集:make_true_views 造真值角点,注入高斯噪声后装观测集。

    噪声注入用固定种子的随机数(可复现);角点加噪是 float 域直接加
    (这里喂的是"角点已提好"的观测集,不经过图像与检测器,与 03-01
    注入实验的 uint8 量化口径无关——本文件锁的是比对协议,不是检测器)。
    """
    views, _k_true = make_true_views(n_views)
    rng = np.random.default_rng(seed)
    uv = (np.asarray(views, dtype=np.float64)
          + rng.normal(0.0, noise_px, (n_views, _N_POINTS, 2)))
    n = len(views)
    return ObservationSet(
        image_ids=tuple(f"img_{k:02d}" for k in range(n)),
        pose_ids=tuple(f"pose_{k:02d}" for k in range(n)),
        object_points=build_object_points(_BOARD),
        image_points=uv,
        image_size=(640, 480),
        corner_source=("sb",) * n,
    )


# ---- 行为 3(映射部分):恒等参数映射,M1/M2/M3 各断言一条 ----

@pytest.mark.parametrize("model_id", ["m1", "m2", "m3"])
def test_identity_param_mapping_pinned(model_id):
    """M1/M2/M3 的参数名与 OpenCV stdDev 18 元布局的前 P 位同名同序。

    这是"取前 P 位就是对齐"这一协议前提的根:映射若漂移,后面的结构
    比对会拿 fx 去 比 k1,错得悄无声息。M1 的 8 位里没有 k3(第 8 位
    被它的固定选项锁死,标准差恒 0,不取)。
    """
    model = create_model(model_id)
    p = len(model.param_names)
    assert model.param_names == _OPENCV_INTRINSIC_ORDER[:p], (
        f"{model_id} 的参数名 {model.param_names} 与 OpenCV stdDev 布局"
        f"前 {p} 位 {_OPENCV_INTRINSIC_ORDER[:p]} 不同名不同序,"
        f"取前 {p} 位对齐的前提被破坏,请核对模型参数表或布局常量。"
    )


# ---- 行为 1 + 行为 2:M2 上 stdDev 形状/正值 + 结构比对 < 5% + k 量级 ----

def test_m2_stddev_shape_and_structural_agreement(make_true_views):
    """M2(9 参数)硬门槛:OpenCV stdDev 形状与正值(行为 1);拟合单一
    公共尺度 k 后逐参数相对偏差 < 5%、k ∈ (0.05, 0.2)(行为 2)。

    k 是"opencv ≈ k·ours"的最小二乘公共倍数——σ=0.1 注入下它应接近
    OpenCV 从残差自估的噪声尺度(≈0.1),量级跑偏说明输入没对齐。
    线性化点传 OpenCV 自己的解,顺带验证"同点重算零差异"。
    """
    obs = _noisy_obs(make_true_views, n_views=20)
    model = create_model("m2")

    # 行为 1:stdDev 向量长度 == 参数个数,全部为正有限值
    base = opencv_stddev_intrinsics(model, obs)
    assert base["status"] == "ok", f"Extended 标定应跑成:{base['failure_note']}"
    se = base["opencv_se"]
    assert se.shape == (len(model.param_names),)
    assert np.isfinite(se).all() and (se > 0).all(), \
        f"OpenCV stdDev 应全为正有限值,实际:{se}"
    assert base["params"].shape == (len(model.param_names),)
    assert np.isfinite(base["rms"]) and base["rms"] > 0
    assert base["per_view_errors"].shape == (20,)
    assert base["rvecs"].shape == (20, 3) and base["tvecs"].shape == (20, 3)

    # 行为 2:协议三步全跑,主判据 < 5%,k 量级对账
    report = crosscheck_model(
        model, obs,
        linearization=(base["params"], base["rvecs"], base["tvecs"]))
    assert report["status"] == "ok", f"交叉验证应跑成:{report['failure_note']}"
    cmp = report["compare"]
    assert cmp["n_active"] == 9
    assert 0.05 < cmp["scale_k"] < 0.2, (
        f"公共尺度 k={cmp['scale_k']:.4f} 跑出 (0.05, 0.2):σ=0.1 注入下它"
        f"应接近 OpenCV 自估噪声尺度,量级不对说明两边输入没对齐")
    assert cmp["max_rel_dev"] < 0.05, (
        f"尺度归一后逐参数最大相对偏差 {cmp['max_rel_dev']:.4f} 超过 5%"
        f"(M2 硬门槛):按 Pitfall 1 判读,若各参数偏差几乎同一个数,"
        f"是尺度因子问题;散乱则数学真有问题。逐参数偏差:{cmp['per_param_rel_dev']}")
    active_devs = cmp["per_param_rel_dev"][:9]
    assert np.isfinite(active_devs).all() and (active_devs < 0.05).all()

    # 同一个点当线性化点重算一遍:换点不敏感的对照应零差异
    assert report["point_sensitivity_max_rel"] < 1e-12

    # 副报告:σ_detector 传入时绝对比对并排存档(不设门槛,只验结构)
    sec = report["secondary_report"]
    assert sec["ours_se_scaled"] is None  # 未传 σ_detector 时占位 None
    report2 = crosscheck_model(model, obs, sigma_detector=_NOISE_PX)
    scaled = report2["secondary_report"]["ours_se_scaled"]
    assert scaled is not None and scaled.shape == se.shape
    assert np.isfinite(scaled).all()
    assert report2["fisher_diagnostics"]["n_views"] == 20


# ---- 行为 3:M1 固定位剔除——对齐后无 NaN、无除零,硬门槛同样成立 ----

def test_m1_fixed_k3_slot_aligned_without_nan(make_true_views):
    """M1(8 参数,k3 被固定):OpenCV stdDev 的第 8 位恒 0(天然对账位),
    比对按前 8 位对齐后无 NaN、无除零,结构比对同样过 < 5% 硬门槛;
    未参与的固定位在逐参数数组里留 NaN 占位(按参数名读数时一眼可辨)。"""
    obs = _noisy_obs(make_true_views, n_views=20)
    model = create_model("m1")

    base = opencv_stddev_intrinsics(model, obs)
    assert base["status"] == "ok", f"Extended 标定应跑成:{base['failure_note']}"
    # 固定的 k3 位(stdDev 布局第 8 位)恒 0——OpenCV 自己承认"这位没估"
    assert base["opencv_se_raw"][8] == 0.0, (
        f"M1 的 k3 位 stdDev 应恒 0,实际 {base['opencv_se_raw'][8]!r}:"
        f"固定选项没生效或布局漂移")
    # 切齐后的 8 位全为正有限:固定位已剔除,后续除法无零除、无 NaN
    se = base["opencv_se"]
    assert se.shape == (8,)
    assert np.isfinite(se).all() and (se > 0).all()

    report = crosscheck_model(model, obs)
    assert report["status"] == "ok", f"交叉验证应跑成:{report['failure_note']}"
    cmp = report["compare"]
    assert cmp["n_active"] == 8
    assert 0.05 < cmp["scale_k"] < 0.2, \
        f"M1 公共尺度 k={cmp['scale_k']:.4f} 跑出量级对账区间"
    assert cmp["max_rel_dev"] < 0.05, \
        f"M1 尺度归一后最大相对偏差 {cmp['max_rel_dev']:.4f} 超过 5%(硬门槛)"
    assert np.isnan(cmp["per_param_rel_dev"][8]), \
        "M1 的固定位(第 8 位)应留 NaN 占位,表示'不参与比对'"
    assert np.isfinite(cmp["per_param_rel_dev"][:8]).all(), \
        "M1 参与比对的 8 位不得出现 NaN(除零或固定值混入)"


# ---- 行为 3(M3 部分,D-07 尽力档)+ 恒等映射在真实调用路径上成立 ----

def test_m3_best_effort_results_present(make_true_views):
    """M3(12 参数有理模型,D-07 尽力档):协议全程跑通、结果齐全即可,
    不对齐不失败——OpenCV 有理模型内部做法未实测,对不上按发现记录,
    真实数据的结论由 03-05 落 crlb.json 时给出。"""
    obs = _noisy_obs(make_true_views, n_views=20)
    model = create_model("m3")

    report = crosscheck_model(model, obs)
    assert report["status"] == "ok", f"M3 交叉验证应跑成:{report['failure_note']}"
    cmp = report["compare"]
    assert cmp["n_active"] == 12
    assert np.isfinite(cmp["scale_k"]) and cmp["scale_k"] > 0
    assert np.isfinite(cmp["max_rel_dev"]) and cmp["max_rel_dev"] >= 0
    assert np.isfinite(cmp["per_param_rel_dev"]).all()
    # 我们的精度界与 OpenCV 解都齐全(结果"有",分级判读留给 03-05)
    assert report["se_ours_opencv_point"].shape == (12,)
    assert np.isfinite(report["se_ours_opencv_point"]).all()
    assert (report["se_ours_opencv_point"] > 0).all()


# ---- 行为 4:M4 分支——中文结论记录,且绝不调用 cv2.fisheye ----

def test_m4_unavailable_fisheye_records_conclusion(make_true_views, monkeypatch):
    """M4(鱼眼):OpenCV 4.14 的鱼眼模块没有带标准差输出的标定函数(实测),
    分支必须直接记中文结论返回,不硬调任何 cv2.fisheye 函数——装监视桩
    把鱼眼标定相关函数全换成"被调即炸"的桩,分支若硬调当场红。"""
    calls: list[str] = []

    def _guard(name):
        def _no_call(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"M4 分支不得调用 cv2.fisheye.{name}")
        return _no_call

    for fname in dir(cv2.fisheye):
        if fname.startswith("calibrate"):
            monkeypatch.setattr(cv2.fisheye, fname, _guard(fname))

    obs = _noisy_obs(make_true_views, n_views=3)
    report = crosscheck_model(create_model("m4"), obs)

    assert calls == [], f"鱼眼函数被调用:{calls}"
    assert report["status"] == "unavailable_fisheye"
    note = report["failure_note"]
    assert isinstance(note, str) and "无可比" in note and note.strip(), \
        f"M4 结论必须是可读中文:{note!r}"
    for key in ("params", "opencv_se", "rms", "per_view_errors"):
        assert report[key] is None, f"不可用状态下 {key} 必须为 None"


# ---- 行为 5:OpenCV 报错 → status=failed + 中文说明,不抛异常 ----

def test_cv2_error_becomes_failed_status(make_true_views, monkeypatch):
    """Extended 标定抛 cv2.error → 顶层拿到 status=failed + 中文 failure_note,
    不抛异常(交叉验证是附加产物,失败不阻塞 CRLB 主产物)。

    替身用空输入调原函数,让 OpenCV 抛货真价实的 cv2.error
    (体例照 test_solve_solvers.py 的同名用例)。"""
    original = cv2.calibrateCameraExtended

    def _explode(*args, **kwargs):
        return original([], [], (640, 480), None, None, flags=0)

    monkeypatch.setattr(cv2, "calibrateCameraExtended", _explode)

    obs = _noisy_obs(make_true_views, n_views=3)
    model = create_model("m2")

    # 桥接函数单测:同样不抛、契约完整
    base = opencv_stddev_intrinsics(model, obs)
    assert base["status"] == "failed"
    assert isinstance(base["failure_note"], str) and "OpenCV 报错" in base["failure_note"]

    # 顶层编排:原样透传失败契约,数值键全 None
    report = crosscheck_model(model, obs)
    assert report["status"] == "failed"
    note = report["failure_note"]
    assert isinstance(note, str) and "OpenCV 报错" in note and note.strip()
    for key in ("params", "opencv_se", "rms", "per_view_errors",
                "se_ours_opencv_point", "compare"):
        assert report[key] is None, f"失败状态下 {key} 必须为 None"


# ---- compare_structural 单元行为:掩码、占位、判读输入校验 ----

def test_compare_structural_scale_fit_and_mask():
    """手算对表:构造 ours = [1, 2]、opencv = [10, 20](精确 10 倍),
    k 应精确等于 0.1(σ 视角:OpenCV 自估噪声 0.1)、归一后逐参数偏差 0;
    掩码剔除的位留 NaN 占位;长度不齐/含零值中文报错。"""
    ours = np.array([1.0, 2.0, 3.0])
    opencv = np.array([10.0, 20.0, 0.0])  # 第 3 位模拟"固定参数恒 0"
    mask = np.array([True, True, False])

    out = compare_structural(ours, opencv, active_mask=mask)

    assert out["n_active"] == 2
    assert out["scale_k"] == pytest.approx(0.1)
    assert out["max_rel_dev"] < 1e-12
    assert np.isnan(out["per_param_rel_dev"][2])
    assert out["per_param_rel_dev"][:2] == pytest.approx(np.zeros(2), abs=1e-12)

    # 缺省掩码 = 全部参与;输入含 0 或负值必须中文报错(防除零 NaN)
    with pytest.raises(AssertionError, match="正的有限值"):
        compare_structural(ours, opencv)
    with pytest.raises(AssertionError, match="长度"):
        compare_structural(ours, opencv[:2])


# ---- 源码级锁定:flags 只走适配器(T-03-04-02 的缓解措施) ----

def test_crosscheck_source_has_no_handmade_flags():
    """crosscheck.py 源码(含注释)不得出现任何手拼的 OpenCV 标定选项
    字面量——flags 与 K/D 装包解包只走 model.opencv_adapter() 唯一入口,
    M1/M3 的选项差异已在适配器里封装并有对账测试。"""
    source = (REPO_ROOT / "src" / "calib" / "noise" / "crosscheck.py").read_text(
        encoding="utf-8")
    assert "CALIB" not in source, \
        "crosscheck.py 出现了手拼的标定选项字面量,必须改走 opencv_adapter()"
    assert "opencv_adapter" in source
    assert "calibrateCameraExtended" in source
