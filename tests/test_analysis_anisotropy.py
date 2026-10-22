"""COR-03 连拍各向异性测试(05-06:库层 temporal 过滤/防零除/分带复用 + 出图)。

全部用合成 npz(tmp_path 落盘再读回,走与生产一致的 allow_pickle=False
读回路径)+ 合成 train 观测集:σ 场已知比值、来源可控、NaN 可注入,
每条判读都能手算对账。另含两道源码防线:分带器必须 import
residual_structure 的 assign_bands(严禁第二份实现),以及源码里不得
出现任何检测/投影字样(纯消费块红线)。出图函数四分支(有/无数据 ×
地图/直方图)逐一落 PNG 验证。
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import calib.analysis.anisotropy as anisotropy_module
from calib.analysis.anisotropy import SCHEMA_ANISOTROPY, run_anisotropy
from calib.io.config import BoardSpec, CalibrationConfig
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet

# 合成棋盘:4 列 × 3 行 = 12 个角点(够切分带与象限,手算不累)
_BOARD = BoardSpec(cols=4, rows=3, square_size_mm=25.0, unit="mm")
_N_POINTS = _BOARD.rows * _BOARD.cols
_EDGES = (0.0, 0.05, 0.10, 0.15)
# r 归一化基准(delivery final_params 口径)
_FX = _FY = 800.0
_CX, _CY = 320.0, 240.0


def _write_npz(tmp_path: Path, sigma_all: np.ndarray, sources: list[str],
               pose_ids: list[str] | None = None,
               sigma_detector: np.ndarray | None = None) -> Path:
    """合成阶段三逐角点重复性 npz(键结构与生产端逐键一致)。"""
    path = tmp_path / "temporal_sigma_per_corner.npz"
    payload = {
        "pose_ids": np.array(pose_ids
                             or [f"p{i}" for i in range(sigma_all.shape[0])]),
        "sigma_per_corner_all": sigma_all,
        "sources": np.array(sources),
    }
    if sigma_detector is not None:
        payload["sigma_detector"] = sigma_detector
    np.savez(path, **payload)
    return path


def _make_obs(uv_per_point: np.ndarray, n_views: int = 3) -> ObservationSet:
    """合成 train 观测集:每个角点固定在给定像素位置(各照片相同)。"""
    uv = np.broadcast_to(np.asarray(uv_per_point, dtype=np.float64),
                         (n_views, _N_POINTS, 2)).copy()
    return ObservationSet(
        image_ids=tuple(f"img_{i:02d}" for i in range(n_views)),
        pose_ids=tuple(f"pose_{i}" for i in range(n_views)),
        object_points=build_object_points(_BOARD),
        image_points=uv,
        image_size=(640, 480),
        corner_source=("sb",) * n_views,
    )


def _evaluate_report() -> dict:
    """最小 evaluate 报告:delivery 里给 r 归一化基准四参数。"""
    return {
        "lineage": {"round_id": "run_eval_test"},
        "delivery": {"models": {"m2": {"final_params": {
            "fx": _FX, "fy": _FY, "cx": _CX, "cy": _CY,
        }}}},
    }


def _cfg():
    base = CalibrationConfig()
    from dataclasses import replace
    return replace(base, analyze=replace(base.analyze,
                                         radial_bin_edges=_EDGES))


def _uv_at_r(r_values: list[float]) -> np.ndarray:
    """把每个角点摆在指定 r 处(沿 u 正方向,v 压在主点上),r 手算可控。"""
    uv = np.empty((_N_POINTS, 2), dtype=np.float64)
    for i, r in enumerate(r_values):
        uv[i] = (_CX + r * _FX, _CY)
    return uv


# ---------------------------------------------------------------------------
# 1) 合成 σu = 2σv 常数场:逐点 aniso == 2
# ---------------------------------------------------------------------------

def test_constant_ratio_field_aniso_equals_two(tmp_path):
    """σ 场 σu = 2σv(常数):逐点 aniso == 2(容差 1e-9),summary 与
    σu/σv 中位同值;防零除下限不应干扰正常量级。"""
    sigma_v = np.full((4, _N_POINTS), 0.01)
    sigma_all = np.stack([sigma_v * 2.0, sigma_v], axis=2)  # (4,P,2) σu,σv
    path = _write_npz(tmp_path, sigma_all, ["temporal"] * 4)
    section, detail = run_anisotropy(
        "m2", _evaluate_report(), path, _make_obs(_uv_at_r([0.02] * _N_POINTS)),
        _BOARD, _cfg(), sigma_detector_floor=0.008)

    assert section["schema"] == SCHEMA_ANISOTROPY
    assert np.allclose(detail["aniso"], 2.0, atol=1e-9)
    assert section["summary"]["median"] == pytest.approx(2.0, abs=1e-9)
    assert section["summary"]["p90"] == pytest.approx(2.0, abs=1e-9)
    assert section["summary"]["max"] == pytest.approx(2.0, abs=1e-9)
    assert section["summary"]["sigma_u_median"] == pytest.approx(0.02, abs=1e-12)
    assert section["summary"]["sigma_v_median"] == pytest.approx(0.01, abs=1e-12)
    assert section["n_temporal_poses"] == 4
    assert section["n_excluded_sources"] == 0
    assert section["n_excluded_nan"] == 0
    assert section["summary"]["n_valid_values"] == 4 * _N_POINTS


# ---------------------------------------------------------------------------
# 2) 来源过滤(Pitfall 9):immerkaer_transfer 行被排除并计数
# ---------------------------------------------------------------------------

def test_transfer_source_rows_excluded_and_counted(tmp_path):
    """混入 immerkaer_transfer 折算姿态:被硬过滤、n_excluded_sources 计数
    正确、明细 pose_ids 只剩 temporal 行;floor 缺省时退回 npz 自带
    sigma_detector 列中位且来源落账。"""
    sigma_v = np.full((3, _N_POINTS), 0.01)
    sigma_all = np.stack([sigma_v * 2.0, sigma_v], axis=2)
    path = _write_npz(
        tmp_path, sigma_all, ["temporal", "immerkaer_transfer", "temporal"],
        pose_ids=["p0", "p1", "p2"],
        sigma_detector=np.full(3, 0.008))
    section, detail = run_anisotropy(
        "m2", _evaluate_report(), path, _make_obs(_uv_at_r([0.02] * _N_POINTS)),
        _BOARD, _cfg())  # 不传 floor:退回 npz 列中位

    assert section["n_poses_total"] == 3
    assert section["n_temporal_poses"] == 2
    assert section["n_excluded_sources"] == 1
    assert section["excluded_source_list"] == ["immerkaer_transfer"]
    assert section["excluded_pose_ids"] == ["p1"]
    assert [str(x) for x in detail["pose_ids"]] == ["p0", "p2"]
    assert detail["aniso"].shape[0] == 2
    assert section["sigma_detector_floor_px"] == pytest.approx(0.008)
    assert "npz 自带" in section["sigma_detector_floor_source"]


# ---------------------------------------------------------------------------
# 3) 防零除:σv = 0 行 clamp 到下限,无 inf 泄漏(T-05-21)
# ---------------------------------------------------------------------------

def test_zero_sigma_v_clamped_finite_no_inf(tmp_path):
    """σv = 0 的角点:分母 clamp 到检测器 σ 下限,aniso == σu/floor
    (有限值),全明细无 inf/NaN 泄漏(全部行有效时)。"""
    sigma_all = np.full((2, _N_POINTS, 2), 0.01)
    sigma_all[:, 5, 1] = 0.0     # 角点 5:σv = 0
    sigma_all[:, 5, 0] = 0.05    # σu = 0.05
    path = _write_npz(tmp_path, sigma_all, ["temporal"] * 2)
    section, detail = run_anisotropy(
        "m2", _evaluate_report(), path, _make_obs(_uv_at_r([0.02] * _N_POINTS)),
        _BOARD, _cfg(), sigma_detector_floor=0.008)

    assert np.isfinite(detail["aniso"]).all(), "全有效场不允许 inf/NaN 泄漏"
    expected = 0.05 / 0.008
    assert detail["aniso"][:, 5] == pytest.approx(expected, rel=1e-12)
    assert np.max(detail["aniso"]) == pytest.approx(expected, rel=1e-12)
    assert section["summary"]["max"] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 4) NaN 行排除与计数(n_excluded_nan + valid_mask)
# ---------------------------------------------------------------------------

def test_nan_rows_excluded_counted_and_stats_skip_them(tmp_path):
    """temporal 姿态内的 NaN 角点行:输出 NaN、n_excluded_nan 计数、
    valid_mask 占位;中位/分位只算有效值(注入大 NaN 场不改变中位)。"""
    sigma_all = np.full((3, _N_POINTS, 2), 0.01)
    sigma_all[0, :, 0] = 0.02   # pose0:σu = 2σv
    sigma_all[1, :, 0] = 0.06   # pose1:σu = 6σv
    sigma_all[2, :, 0] = 0.02
    sigma_all[1, 0, :] = np.nan  # pose1 角点 0 整行 NaN
    sigma_all[2, 3, 0] = np.nan  # pose2 角点 3 的 σu 分量 NaN
    path = _write_npz(tmp_path, sigma_all, ["temporal"] * 3)
    section, detail = run_anisotropy(
        "m2", _evaluate_report(), path, _make_obs(_uv_at_r([0.02] * _N_POINTS)),
        _BOARD, _cfg(), sigma_detector_floor=0.008)

    assert section["n_excluded_nan"] == 2
    assert not detail["valid_mask"][1, 0]
    assert not detail["valid_mask"][2, 3]
    assert np.isnan(detail["aniso"][1, 0])
    assert np.isnan(detail["aniso"][2, 3])
    assert section["summary"]["n_valid_values"] == 3 * _N_POINTS - 2
    # NaN 不掺统计:有效值只有 2 与 6 两档,中位必为 2 的占比决定(2 多)
    assert section["summary"]["median"] == pytest.approx(2.0, abs=1e-9)
    # 逐角点中位:角点 0 只有 pose0/pose2 有效(2 与 2)→ 2.0
    assert section["per_corner_median_aniso"][0] == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5) 分带/象限切片:已知 r 网格的分带中位手算一致(共用同一把尺)
# ---------------------------------------------------------------------------

def test_by_band_and_quadrant_hand_computed(tmp_path):
    """12 个角点按 r 摆进 3 条带(0.02/0.07/0.12),两姿态 aniso 1.0 与
    3.0:每带 n_points = 4、中位 = 2.0;象限以 cx/cy 分界,命名与
    residual_structure 同一套;band_ids / quadrant_ids 落进明细。"""
    r_values = ([0.02] * 4) + ([0.07] * 4) + ([0.12] * 4)
    sigma_all = np.empty((2, _N_POINTS, 2))
    sigma_all[0, :, 0], sigma_all[0, :, 1] = 0.01, 0.01   # aniso 1.0
    sigma_all[1, :, 0], sigma_all[1, :, 1] = 0.03, 0.01   # aniso 3.0
    path = _write_npz(tmp_path, sigma_all, ["temporal"] * 2)
    # 角点横向摆开:一半在 cx 左、一半在 cx 右(v 全压主点线以下 20px)
    uv = _uv_at_r(r_values)
    uv[:, 1] = _CY + 20.0
    uv[::2, 0] -= 2.0 * np.asarray(r_values)[::2] * _FX + 10.0  # 偶数号点挪左
    section, detail = run_anisotropy(
        "m2", _evaluate_report(), path, _make_obs(uv),
        _BOARD, _cfg(), sigma_detector_floor=0.008)

    assert section["band_edges"] == list(_EDGES)
    assert len(section["by_band"]) == 3
    for band in section["by_band"]:
        assert band["n_points"] == 4
        assert band["n_values"] == 8
        assert band["median_aniso"] == pytest.approx(2.0, abs=1e-9)
    assert [b["band_lo"] for b in section["by_band"]] == [0.0, 0.05, 0.10]
    # 分带器一致:同一 uv 下 band_ids 与 residual_structure.assign_bands 逐位同
    from calib.analysis.residual_structure import assign_bands, radius_footprint
    r_per_corner, _r4 = radius_footprint(uv, _FX, _FY, _CX, _CY)
    expected_ids, _ = assign_bands(r_per_corner, _EDGES)
    assert np.array_equal(detail["band_ids"], expected_ids)
    assert detail["band_ids"].shape == (_N_POINTS,)

    # 象限:左侧 6 点 + 右侧 6 点,v 全在主点下 → 左下 6 / 右下 6,其余 0
    quads = {q["quadrant"]: q for q in section["by_quadrant"]}
    assert quads["左下(u<cx, v>=cy)"]["n_points"] == 6
    assert quads["右下(u>=cx, v>=cy)"]["n_points"] == 6
    assert quads["左上(u<cx, v<cy)"]["n_points"] == 0
    assert quads["左上(u<cx, v<cy)"]["median_aniso"] is None
    assert quads["右下(u>=cx, v>=cy)"]["median_aniso"] == pytest.approx(2.0, abs=1e-9)
    # 判读行:分带/象限最高都点名且是数据现算的 2.0
    assert "2.00" in section["reading"]
    assert "只报数据" in section["reading"]


# ---------------------------------------------------------------------------
# 6) 互证行:corner_opt 段在盘引其 u/v 均值方向;缺席注明
# ---------------------------------------------------------------------------

def test_cross_ref_with_and_without_corner_opt(tmp_path):
    """corner_opt 段在盘:互证行引生产配置 mean_du/mean_dv 的中位方向;
    段缺席:如实注明「本轮无退化实验数据」,不冒充。"""
    sigma_v = np.full((1, _N_POINTS), 0.01)
    sigma_all = np.stack([sigma_v * 2.0, sigma_v], axis=2)
    path = _write_npz(tmp_path, sigma_all, ["temporal"])
    obs = _make_obs(_uv_at_r([0.02] * _N_POINTS))

    section_none, _ = run_anisotropy(
        "m2", _evaluate_report(), path, obs, _BOARD, _cfg(),
        sigma_detector_floor=0.008, corner_opt_section=None)
    assert "本轮无退化实验数据" in section_none["cross_ref"]

    corner_opt = {
        "production_method": "sb",
        "stats": {"_self": {"sb": {
            "noise_2.0": {"mean_du": 0.02, "mean_dv": -0.01},
            "blur_1.0": {"mean_du": 0.04, "mean_dv": -0.03},
        }}},
    }
    section_ref, _ = run_anisotropy(
        "m2", _evaluate_report(), path, obs, _BOARD, _cfg(),
        sigma_detector_floor=0.008, corner_opt_section=corner_opt)
    assert "sb" in section_ref["cross_ref"]
    assert "+0.0300" in section_ref["cross_ref"]     # u 中位 +0.03
    assert "-0.0200" in section_ref["cross_ref"]     # v 中位 -0.02
    assert "水平(u)" in section_ref["cross_ref"]     # |u| > |v| 的方向点名
    assert "只报方向" in section_ref["cross_ref"]


# ---------------------------------------------------------------------------
# 7) 白话文案与显著位置清单(物理量级全部现算)
# ---------------------------------------------------------------------------

def test_plain_language_and_significant_positions(tmp_path):
    """σu=4σv 场:plain_language 含检测器倍数与发丝参照;>1.5 显著值的
    (行,列) 位置清单覆盖全部角点且行/列号在棋盘网格内。"""
    sigma_v = np.full((2, _N_POINTS), 0.01)
    sigma_all = np.stack([sigma_v * 4.0, sigma_v], axis=2)  # aniso = 4 > 1.5
    path = _write_npz(tmp_path, sigma_all, ["temporal"] * 2)
    # 角点沿 u 摆开(间距 30px):毫米折算的分母(方格像素边长)量得出
    uv = np.column_stack([
        np.arange(_N_POINTS, dtype=np.float64) * 30.0 + 60.0,
        np.full(_N_POINTS, 100.0),
    ])
    section, _detail = run_anisotropy(
        "m2", _evaluate_report(), path, _make_obs(uv),
        _BOARD, _cfg(), sigma_detector_floor=0.008)

    plain = section["plain_language"]
    assert "横向" in plain and "纵向" in plain
    assert "倍" in plain and "发丝" in plain and "mm" in plain
    assert section["summary"]["n_aniso_gt_1p5"] == 2 * _N_POINTS
    positions = section["summary"]["aniso_gt_1p5_positions"]
    assert len(positions) == _N_POINTS
    for pos in positions:
        assert 0 <= pos["row"] < _BOARD.rows
        assert 0 <= pos["col"] < _BOARD.cols
        assert pos["median_aniso"] == pytest.approx(4.0, abs=1e-9)
    # 直方图块:全域 40 桶、计数守恒(不截尾)
    hist = section["aniso_hist"]
    assert len(hist["counts"]) == 40 and len(hist["edges"]) == 41
    assert sum(hist["counts"]) == hist["n_values"] == 2 * _N_POINTS

    # 退化布局(角点全部重合,方格间距量不出):毫米折算如实缺席,不硬除
    section_degen, _ = run_anisotropy(
        "m2", _evaluate_report(), path,
        _make_obs(_uv_at_r([0.02] * _N_POINTS)),
        _BOARD, _cfg(), sigma_detector_floor=0.008)
    assert "缺席" in section_degen["plain_language"]


# ---------------------------------------------------------------------------
# 8) 源码防线:分带器一处实现两处用 + 纯消费块红线
# ---------------------------------------------------------------------------

def test_source_bander_reused_not_copied():
    """源码断言:anisotropy.py 从 residual_structure import assign_bands /
    radius_footprint(无第二份分带器),且不出现任何检测/投影/求解字样
    (纯消费块红线,D-06/D-10)。"""
    source = inspect.getsource(anisotropy_module)
    assert "from calib.analysis.residual_structure import assign_bands" in source
    assert "def assign_bands" not in source, "分带器必须复用,不许第二份实现"
    assert "def radius_footprint" not in source
    for banned in ("cornerSubPix", "findChessboardCorners", "calibrateCamera",
                   "solvePnP", "model.project", "project_point_jacobian"):
        assert banned not in source, f"纯消费块不许出现 {banned}"


def test_empty_temporal_poses_chinese_error(tmp_path):
    """全部姿态都是折算来源(零真连拍):当场中文报错,不拿假数据出表。"""
    sigma_all = np.zeros((2, _N_POINTS, 2))
    path = _write_npz(tmp_path, sigma_all,
                      ["immerkaer_transfer", "immerkaer_transfer"])
    with pytest.raises(AssertionError, match="temporal"):
        run_anisotropy(
            "m2", _evaluate_report(), path,
            _make_obs(_uv_at_r([0.02] * _N_POINTS)), _BOARD, _cfg(),
            sigma_detector_floor=0.008)


def test_missing_floor_and_sigma_detector_column_chinese_error(tmp_path):
    """floor 未传且 npz 无 sigma_detector 列:中文报错点名唯一口径入口。"""
    sigma_v = np.full((1, _N_POINTS), 0.01)
    sigma_all = np.stack([sigma_v * 2.0, sigma_v], axis=2)
    path = _write_npz(tmp_path, sigma_all, ["temporal"])  # 无 sigma_detector 键
    with pytest.raises(AssertionError, match="sigma_detector_corner"):
        run_anisotropy(
            "m2", _evaluate_report(), path,
            _make_obs(_uv_at_r([0.02] * _N_POINTS)), _BOARD, _cfg())


def test_corner_count_mismatch_chinese_error(tmp_path):
    """train 观测集角点数与时域 npz 对不上:中文报错点名"同一套角点编号"。"""
    sigma_v = np.full((1, _N_POINTS), 0.01)
    sigma_all = np.stack([sigma_v * 2.0, sigma_v], axis=2)
    path = _write_npz(tmp_path, sigma_all, ["temporal"])
    obs = ObservationSet(
        image_ids=("img_00",),
        pose_ids=("pose_0",),
        object_points=build_object_points(_BOARD),
        image_points=np.zeros((1, 6, 2)),   # 6 个角点 ≠ 12
        image_size=(640, 480),
        corner_source=("sb",))
    with pytest.raises(AssertionError, match="角点"):
        run_anisotropy(
            "m2", _evaluate_report(), path, obs, _BOARD, _cfg(),
            sigma_detector_floor=0.008)


# ---------------------------------------------------------------------------
# 9) 出图函数:四分支(有数据/无数据 × 地图/直方图)都出 PNG
# ---------------------------------------------------------------------------

def _mini_section() -> dict:
    """最小 anisotropy 段:8 个角点、两张照片,键结构与 run_anisotropy 一致。"""
    per_corner = [1.0, 1.2, 0.9, 1.6, 1.1, 0.8, 1.3, None]
    uv = np.empty((2, 8, 2), dtype=np.float64)
    for i in range(8):
        uv[:, i, 0] = 120.0 + 60.0 * (i % 4)
        uv[:, i, 1] = 120.0 + 120.0 * (i // 4)
    return {
        "per_corner_median_aniso": per_corner,
        "r_normalization_params": {"fx": 800.0, "fy": 800.0,
                                   "cx": 240.0, "cy": 180.0},
        "r_normalization_basis": "delivery final_params(与 ANA-02 同口径)",
        "image_size": [480, 360],
        "aniso_hist": {
            "edges": list(np.linspace(0.5, 2.0, 41)),
            "counts": [3, 5, 4, 2, 1] + [0] * 35,
            "n_values": 15,
        },
        "summary": {"median": 1.1, "significant_line": 1.5,
                    "n_aniso_gt_1p5": 1},
    }, uv


def test_plot_anisotropy_map_and_hist_with_and_without_data(tmp_path):
    """地图与直方图各有"有数据/无数据"两分支:四张都落 PNG,文件非空。"""
    import matplotlib
    matplotlib.use("Agg")
    from calib.visualization.analyze_figs import plot_anisotropy_hist, plot_anisotropy_map

    section, uv = _mini_section()
    p_map = plot_anisotropy_map(section, uv, tmp_path / "map.png")
    p_hist = plot_anisotropy_hist(section, tmp_path / "hist.png")
    p_map_none = plot_anisotropy_map(None, None, tmp_path / "map_none.png")
    p_hist_none = plot_anisotropy_hist({}, tmp_path / "hist_none.png")
    for p in (p_map, p_hist, p_map_none, p_hist_none):
        assert p.exists() and p.stat().st_size > 0
