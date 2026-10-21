"""residual_structure 库层的行为测试:分带/足迹/外推折算 + train/test 双侧统计。

白话:这层是 ANA-02 残差方向分解的数学与编排,测试锁四件事——
1. 归一化半径 r 与 r⁴ 信息权重的公式本身(3-4-5 三角形可手算);
2. 分带器的左闭右开边界与"超界不丢、单独计数";
3. band_stats 把系统偏移(mean_du/mean_dv)与散布(rms)分开——整带
   同向挪动不改 rms;
4. run_residual_structure 端到端:用零畸变的合成 m2"真值"造 train/test
   观测,test 侧只估位姿后回代,残差应贴到 0.01px 以下;失败帧 NaN
   占行不中断;外推披露的数值块与 k2_wobble_px 公式逐位自洽。

红线断言:本模块绝不 import OpenCV、绝不手写投影公式(D-06)——
源码级检查兜底。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace as dc_replace

import numpy as np
import pytest

from calib.analysis.residual_structure import (
    SCHEMA_RESIDUAL,
    assign_bands,
    band_stats,
    coverage_summary,
    k2_wobble_px,
    radius_footprint,
    run_residual_structure,
)
from calib.io.config import AnalyzeConfig, BoardSpec, CalibrationConfig
from calib.models import create_model
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet

# 合成"真值"内参:零畸变 m2,主点按 640×480 画面中心
_TRUTH = {
    "fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0,
    "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
}
_BOARD = BoardSpec(cols=11, rows=8, square_size_mm=25.0, unit="mm")
_EDGES = (0.0, 0.08, 0.16)  # 造小的分带边界:合成画面 r 约 0.19,能切出 3 带


def _project_views(poses: list[tuple]) -> np.ndarray:
    """用"真值"内参 + 零畸变把物点网格投到像素平面:(V, 88, 2)。"""
    model = create_model("m2")
    objp = build_object_points(_BOARD)
    from calib.models.base import transform_to_camera

    uv = np.stack(
        [model.project(transform_to_camera(objp, np.array(r), np.array(t))[0],
                       np.array([_TRUTH[n] for n in model.param_names]))
         for r, t in poses]
    )
    return uv


def _make_obs(image_ids, uv, image_size=(640, 480)) -> ObservationSet:
    return ObservationSet(
        image_ids=tuple(image_ids),
        pose_ids=tuple(f"p{i}" for i in range(len(image_ids))),
        object_points=build_object_points(_BOARD),
        image_points=np.asarray(uv, dtype=np.float64),
        image_size=image_size,
        corner_source=("sb",) * len(image_ids),
    )


def _write_calib_run(tmp_path, image_ids, uv_train):
    """tmp 下造假标定轮三件套:residuals 全零(合成数据零畸变零残差)。"""
    combo = tmp_path / "run_synthetic_calib" / "m2__scipy"
    combo.mkdir(parents=True)
    n = len(image_ids)
    np.savez(combo / "residuals.npz",
             image_ids=np.array(list(image_ids)),
             residuals=np.zeros((n, 88, 2)),
             per_view_rms=np.zeros(n))
    np.savez(combo / "extrinsics.npz",
             image_ids=np.array(list(image_ids)),
             rvecs=np.array([p[0] for p in _POSES]),
             tvecs=np.array([p[1] for p in _POSES]))
    (combo / "result.json").write_text(
        json.dumps({"model_id": "m2", "params": dict(_TRUTH)}), encoding="utf-8")
    return tmp_path / "run_synthetic_calib"


# 三组合成位姿:(rvec, tvec),距离约 0.8 m、姿态各异,投影铺满画面中区
_POSES = [
    ((0.0, 0.0, 0.0), (0.0, 0.0, 800.0)),
    ((0.02, -0.01, 0.005), (30.0, -20.0, 780.0)),
    ((-0.015, 0.02, -0.004), (-25.0, 15.0, 820.0)),
]


# ---------------------------------------------------------------------------
# 1) 归一化半径公式(3-4-5 三角形,手算基准)
# ---------------------------------------------------------------------------

def test_radius_footprint_three_four_five():
    """r = √(((u−cx)/fx)² + ((v−cy)/fy)²):横向 3、纵向 4、焦距 5 → r=5/5=1。"""
    r, r4 = radius_footprint(
        np.array([[320.0 + 30.0, 240.0 + 40.0]]), 50.0, 50.0, 320.0, 240.0)
    assert r[0] == pytest.approx(1.0)   # 30/50=0.6、40/50=0.8、√(0.36+0.64)=1
    assert r4[0] == pytest.approx(1.0)
    # 焦距改 10:同样偏移 r 缩一半,r⁴ 缩 16 倍(四次方律的直观呈现)
    r2, r4b = radius_footprint(
        np.array([[320.0 + 30.0, 240.0 + 40.0]]), 100.0, 100.0, 320.0, 240.0)
    assert r2[0] == pytest.approx(0.5)
    assert r4b[0] == pytest.approx(0.5**4)


# ---------------------------------------------------------------------------
# 2) 分带器:左闭右开 + 超界不丢
# ---------------------------------------------------------------------------

def test_assign_bands_left_closed_and_beyond_counted():
    """边界 [0, 0.2, 0.4]:r=0.2 恰好落在边界上归右带(左闭右开);
    r≥0.4 的观测(边界点 0.4 与 0.9)超出最大边界,归最外带并单独计数
    = 2(不丢,它们是最外环信息)。"""
    r = np.array([0.05, 0.199, 0.2, 0.4, 0.9])
    band_ids, n_beyond = assign_bands(r, (0.0, 0.2, 0.4))
    assert list(band_ids) == [0, 0, 1, 1, 1]  # 只有内部边界 0.2 一条分割线
    assert n_beyond == 2


def test_assign_bands_rejects_bad_edges():
    """边界不足 2 个 / 非有限 / 倒序:当场中文报错(配置错不许静默吞)。"""
    with pytest.raises(AssertionError, match="至少要 2 个"):
        assign_bands(np.array([0.1]), (0.5,))
    with pytest.raises(AssertionError, match="有限数"):
        assign_bands(np.array([0.1]), (0.0, float("nan"), 0.4))
    with pytest.raises(AssertionError, match="严格递增"):
        assign_bands(np.array([0.1]), (0.0, 0.4, 0.2))


# ---------------------------------------------------------------------------
# 3) band_stats:系统偏移与散布分开
# ---------------------------------------------------------------------------

def test_band_stats_mean_shift_keeps_rms_and_reports_beyond():
    """整带同向挪一个常向量:每条残差的长度不变 → rms/median/max 不变;
    mean_du/mean_dv 恰好平移同量。空带记 None(不冒充 0);最外带带
    n_beyond_hi(带上界外的观测数)。"""
    base = np.array([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]])  # 长度 5/0/10
    shifted = base + np.array([100.0, -50.0])
    r = np.array([0.05, 0.06, 0.07])          # 全在第一带
    edges = (0.0, 0.1, 0.2)                   # 第二带(0.1–0.2)空
    rows_base = band_stats(base, r, edges)
    rows_shift = band_stats(shifted, r, edges)

    for a, b in zip(rows_base, rows_shift):
        assert b["rms_px"] == pytest.approx(a["rms_px"])
        assert b["max_px"] == pytest.approx(a["max_px"])
    assert rows_shift[0]["mean_du"] == pytest.approx(rows_base[0]["mean_du"] + 100.0)
    assert rows_shift[0]["mean_dv"] == pytest.approx(rows_base[0]["mean_dv"] - 50.0)
    assert rows_shift[0]["n"] == 3
    assert rows_shift[1]["n"] == 0 and rows_shift[1]["rms_px"] is None
    assert rows_shift[-1]["n_beyond_hi"] == 0  # 全体 r < 0.2,无超界


# ---------------------------------------------------------------------------
# 4) coverage_summary:覆盖足迹与 k2 信息份额(手算)
# ---------------------------------------------------------------------------

def test_coverage_summary_info_share_hand_computed():
    """三点半手算:两个点(r=0.6 与 r=0.4)加一个中心点(r=0),
    r>0.5 的占比 1/3,却承载 r⁴ 总量的 0.1296/0.1552 ≈ 83.5%。"""
    uv = np.array([[110.0, 50.0], [90.0, 50.0], [50.0, 50.0]])
    cov = coverage_summary(uv, 100.0, 100.0, 50.0, 50.0, (101, 101))
    # r: 60/100=0.6、40/100=0.4、0
    assert cov["r_max_obs"] == pytest.approx(0.6)
    # 画面四角 (100,100):(50/100, 50/100) → r=√0.5
    assert cov["r_corner_frame"] == pytest.approx(np.sqrt(0.5))
    assert cov["extrapolation_interval"] == pytest.approx([0.6, np.sqrt(0.5)])
    assert cov["frac_beyond_05"] == pytest.approx(1.0 / 3.0)
    assert cov["info_share_beyond_05"] == pytest.approx(0.1296 / 0.1552)


def test_coverage_summary_full_coverage_no_extrapolation():
    """观测到画面正四角:r_max == r_corner → 外推区间为 None(无外推区)。"""
    uv = np.array([[100.0, 100.0]])  # 恰在画面角上
    cov = coverage_summary(uv, 100.0, 100.0, 50.0, 50.0, (101, 101))
    assert cov["extrapolation_interval"] is None
    assert cov["r_max_obs"] == pytest.approx(cov["r_corner_frame"])


# ---------------------------------------------------------------------------
# 5) k2 晃动 r⁴ 律折算(findings 结论 5 的数字)
# ---------------------------------------------------------------------------

def test_k2_wobble_r4_law_findings_numbers():
    """0.135px 当量在 r=0.685 处,折到 r=0.80 ≈ 0.25px(结论 5 的 ±10%);
    折到 r=0.4 约 0.016px(中区现状够用);律单调递增。"""
    w_corner = k2_wobble_px(0.135, 0.80, 0.685)
    assert w_corner == pytest.approx(0.25, rel=0.10)
    assert w_corner == pytest.approx(0.135 * (0.80 / 0.685) ** 4, rel=1e-12)
    assert k2_wobble_px(0.135, 0.4, 0.685) < 0.02
    assert (k2_wobble_px(0.135, 0.4, 0.685)
            < k2_wobble_px(0.135, 0.5, 0.685)
            < k2_wobble_px(0.135, 0.8, 0.685))


# ---------------------------------------------------------------------------
# 6) run_residual_structure 端到端(合成零畸变"真值")
# ---------------------------------------------------------------------------

def _synthetic_evaluate_report() -> dict:
    return {
        "lineage": {"round_id": "run_synthetic"},
        "delivery": {"models": {"m2": {"final_params": dict(_TRUTH)}}},
        "slices": {"models": {"m2": {"empirical_std": {"k2": 1e-4}}}},
    }


def _synthetic_cfg():
    cfg = CalibrationConfig()
    return dc_replace(cfg, analyze=dc_replace(AnalyzeConfig(), radial_bin_edges=_EDGES))


def test_run_residual_structure_end_to_end(tmp_path):
    """零畸变真值端到端:基准声明/逐位一致、test 侧回代残差 <0.01px、
    失败帧 NaN 占行 + 计数、明细形状与分带、外推披露与公式自洽。"""
    uv_train = _project_views(_POSES)
    calib_run = _write_calib_run(tmp_path, ("t0", "t1", "t2"), uv_train)

    # test 侧:一帧好帧 + 一帧只有 3 个可用角点的坏帧
    good = _project_views([((0.01, 0.008, 0.0), (10.0, 5.0, 790.0))])[0]
    bad = good.copy()
    bad[3:] = np.nan
    obs_train = _make_obs(("t0", "t1", "t2"), uv_train)
    obs_test = _make_obs(("s0", "s1"), np.stack([good, bad]))

    section, detail = run_residual_structure(
        "m2", _synthetic_evaluate_report(), calib_run, obs_train, obs_test,
        _BOARD, _synthetic_cfg())

    assert section["schema"] == SCHEMA_RESIDUAL == "analyze-residual-v1"
    assert "delivery" in section["r_normalization_basis"]
    assert "run_synthetic" in section["r_normalization_basis"]
    check = section["r_normalization_baseline_check"]
    assert check["status"] == "consistent" and check["max_rel_diff"] < 1e-6
    assert section["band_edges"] == [0.0, 0.08, 0.16]

    # train 侧:零畸变零残差 → 逐视图 RMS 恰好 0;分带统计在盘
    assert section["train"]["per_view"]["rms_max_px"] == pytest.approx(0.0)
    assert len(section["train"]["bands"]) == len(_EDGES) - 1
    assert section["train"]["n_beyond_last_edge"] > 0  # r≈0.19 超出边界 0.16

    # test 侧:内参当尺子、只估位姿后回代,零畸变真值下残差贴 0
    test_blk = section["test"]
    assert test_blk["n_frames"] == 2 and test_blk["n_failed"] == 1
    assert 0.0 < test_blk["per_frame_rms"]["median_px"] < 0.01
    resid_test = detail["resid_test"]
    assert np.isfinite(resid_test[0]).all()
    assert np.isnan(resid_test[1]).all()  # 失败帧整行 NaN 占行,不冒充

    # 明细形状:逐点大数组与观测集逐行对齐
    assert detail["r_train"].shape == (3, 88)
    assert detail["band_ids_train"].shape == (3, 88)
    assert detail["uv_train"].shape == (3, 88, 2)
    assert detail["resid_test"].shape == (2, 88, 2)
    assert detail["r_test"].shape == (2, 88)
    assert list(detail["image_ids_train"]) == ["t0", "t1", "t2"]
    assert list(detail["image_ids_test"]) == ["s0", "s1"]

    # 外推披露:r_max_obs ≈ 0.19 < 画面四角 r(640×480 @ fx=800 → ≈0.498)
    ext = section["extrapolation"]
    assert ext["applicable"] is True
    cov = section["train"]["coverage"]
    assert ext["r_max_obs"] == pytest.approx(cov["r_max_obs"])
    corner_r = cov["r_corner_frame"]
    assert ext["k2_wobble_at_corner_frame"] == pytest.approx(
        k2_wobble_px(ext["k2_sigma_px_at_r_max_obs"], corner_r, ext["r_max_obs"]))
    assert ext["criteria_table"][2]["band"].startswith("r >")
    note = section["extrapolation_note"]
    assert "外推" in note and "k2" in note
    # 参考位姿 = extrinsics 首视图(rvecs[0]),findings 结论 1 惯例
    assert json.loads((calib_run / "m2__scipy" / "result.json").read_text(
        encoding="utf-8"))["params"]["fx"] == _TRUTH["fx"]


def test_run_residual_structure_row_order_mismatch(tmp_path):
    """train 观测集与标定 residuals.npz 的照片编号对不上:当场中文报错
    (行序契约是残差对齐的唯一保证,错了绝不能算)。"""
    uv_train = _project_views(_POSES)
    calib_run = _write_calib_run(tmp_path, ("t0", "t1", "t2"), uv_train)
    obs_train = _make_obs(("t0", "t1", "WRONG"), uv_train)
    obs_test = _make_obs(("s0",), _project_views(_POSES[:1]))
    with pytest.raises(AssertionError, match=" residuals.npz 对不上"):
        run_residual_structure(
            "m2", _synthetic_evaluate_report(), calib_run, obs_train, obs_test,
            _BOARD, _synthetic_cfg())


# ---------------------------------------------------------------------------
# 7) 红线:单一数学源(绝不 import OpenCV、绝不手写投影)
# ---------------------------------------------------------------------------

def test_source_has_no_cv2_and_uses_existing_interfaces():
    """D-06 源码断言:不出现 cv2/Rodrigues,位姿只走 solve_pose_only,
    投影只走 model.project——公式只许在 models/ 存在一份。"""
    import calib.analysis.residual_structure as mod

    source = inspect.getsource(mod)
    assert "import cv2" not in source and "Rodrigues" not in source
    assert "solve_pose_only" in source, "test 侧必须复用既有位姿评分接口"
    assert "model.project" in source, "回代必须走模型统一投影接口"
