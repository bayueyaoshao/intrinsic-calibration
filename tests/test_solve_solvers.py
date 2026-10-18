"""双求解后端的结构与落盘测试(第一批:CalibrationResult + 三件套 + 统一回代尺)。

锁定清单(02-03 Task 2 起步,02-04 再扩展 scipy 侧):
- 失败契约:converged=False 时结构键一个不少、数值字段一律 None(落盘成
  JSON null)、failure_note 非空中文——上游读取逻辑一视同仁;
- 同一把尺(Pitfall 9):rms_overall / per_view_rms 一律由 models.project
  回代计算(float64),OpenCV 内部 RMS 只进 solver_meta——两把尺子刻度差
  约 1e-4 像素(float32 输入 vs float64 回代),跨后端必须同一把尺;
- 落盘三件套:result.json / extrinsics.npz / residuals.npz 可写可读,
  失败组合照样落盘且 JSON 数值键为 null。

全部用例跑在合成数据上(模型参数/位姿全已知),秒级。
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from calib.io.config import BoardSpec
from calib.models import create_model
from calib.models.base import transform_to_camera
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
    for key in _RESULT_FIELDS:
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
