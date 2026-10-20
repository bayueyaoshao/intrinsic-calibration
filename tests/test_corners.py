"""corners 模块测试:窗口推导(Task 1)+ 规范化/连拍断言/双检测器(Task 2)。

全部用合成棋盘投影夹具(conftest 的 make_chessboard_image):它自带真值角点,
检测对不对、误差几个像素,都有"标准答案"可对照——不靠肉眼,靠断言。
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

from calib.corners.canonical import burst_consistency, canonicalize
from calib.corners.detect import detect_classic, detect_sb, grid_defects
from calib.corners.window import (
    derive_window_frame,
    derive_window_per_corner,
    estimate_sq_px,
)
from calib.io.config import CornersConfig

# ---------- Task 1:方格像素边长估计 ----------

def test_estimate_sq_px_close_to_truth(make_chessboard_image):
    """合成图上估计的方格像素边长,与生成时给定的值误差 < 5%。

    位姿接近正对(夹具默认只带极轻微侧转),方格在画面里几乎等大,
    中位数估出的边长应当紧贴真值。
    """
    img, corners, sq_truth = make_chessboard_image(square_px=20.0)
    assert img.shape == (480, 640)
    sq_est = estimate_sq_px(corners, cols=11, rows=8)
    assert abs(sq_est - sq_truth) / sq_truth < 0.05


def test_estimate_sq_px_conservative_under_perspective(make_chessboard_image):
    """透视下取保守小值:侧转角加大后,估计值应接近"较小那一侧"的边长,
    且依然与给定值相差在 10% 内(不至于把窗口推导带偏)。"""
    _img, corners, sq_truth = make_chessboard_image(
        square_px=24.0, rvec=np.array([0.0, 0.35, 0.0]))
    sq_est = estimate_sq_px(corners, cols=11, rows=8)
    assert 0.9 * sq_truth < sq_est <= sq_truth * 1.001  # 保守取小:不应超过真值太多


# ---------- Task 1:帧级窗口推导 ----------

@pytest.mark.parametrize("sq_px", [15.0, 20.0, 25.0, 40.0])
def test_derive_window_frame_inequality(sq_px):
    """四个典型方格边长逐一验证:搜索区 2w+1 必须严格小于方格边长。

    例:sq=20 → w=9,搜索区 19 < 20;sq=15 → w=6,搜索区 13 < 15。
    """
    w = derive_window_frame(sq_px)
    assert w == int(sq_px // 2) - 1
    assert 2 * w + 1 < sq_px


def test_derive_window_frame_min_win_floor():
    """min_win=3 下限生效:极小方格不给小于 3 的窗口。

    sq=7.5:floor(3.75)−1 = 2,被下限托到 3,且 7 < 7.5 仍成立(合法);
    sq=5:下限托到 3 后 7 ≥ 5,不等式违反——按设计必须当场抛断言错误
    (这是正确行为,用测试锁定,绝不允许带病窗口往下流)。
    """
    assert derive_window_frame(7.5, min_win=3) == 3
    with pytest.raises(AssertionError):
        derive_window_frame(5.0, min_win=3)


# ---------- Task 1:逐角点窗口推导 ----------

def test_derive_window_per_corner(make_chessboard_image):
    """逐角点模式:返回长度 88 的数组,且每点都满足同一不等式。"""
    _img, corners, _sq = make_chessboard_image(square_px=20.0)
    wins = derive_window_per_corner(corners, cols=11, rows=8, min_win=3)
    assert isinstance(wins, np.ndarray)
    assert wins.shape == (11 * 8,)
    assert (wins >= 3).all()
    # 帧级保守边长约束下逐点检查同一不等式(合成图方格近乎等大,局部≈帧级)
    sq_frame = estimate_sq_px(corners, 11, 8)
    assert (2 * wins + 1 < sq_frame).all()


def test_derive_window_per_corner_flags_tiny_squares(make_chessboard_image):
    """局部方格过小、连下限 3 都违反不等式时:逐点断言必须报错(锁定行为)。"""
    _img, corners, _ = make_chessboard_image(square_px=5.0)
    with pytest.raises(AssertionError):
        derive_window_per_corner(corners, cols=11, rows=8, min_win=3)


# ---------- Task 2:排序规范化 ----------

def _four_variants(corners: np.ndarray, cols: int, rows: int) -> dict[str, np.ndarray]:
    """把同一份规范角点序列变换成四种排列变体(恒等/水平/垂直/旋转180°)。"""
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    return {
        "恒等": g.copy(),
        "水平翻转": g[:, ::-1, :],
        "垂直翻转": g[::-1, :, :],
        "旋转180": g[::-1, ::-1, :],
    }


def test_canonicalize_maps_all_variants_to_same_sequence(make_chessboard_image):
    """四种翻转变体经规范化后逐点相等:输出唯一形态(合成图行朝右、列朝下,
    夹具真值本身就是规范形态)。"""
    _img, corners, _ = make_chessboard_image()
    base = canonicalize(corners, cols=11, rows=8)
    assert np.allclose(base, corners)  # 规范输入原样通过(不动手)
    for name, variant in _four_variants(corners, 11, 8).items():
        out = canonicalize(variant.reshape(-1, 2), cols=11, rows=8)
        assert np.allclose(out, base), f"变体「{name}」没有被规范化到唯一形态"


# ---------- Task 2:双检测器封装 ----------

# 检测器测试用配置:直接用数据类默认值(与仓库 calibration.yaml 一致)
_CFG = CornersConfig()


@pytest.mark.parametrize("detector", ["classic", "sb"])
def test_detector_accuracy_within_half_pixel(make_chessboard_image, detector):
    """合成图上两颗检测器的角点对真值的逐点最大定位误差 < 0.5px。"""
    img, corners, _ = make_chessboard_image(degrade="blur")
    detect = detect_classic if detector == "classic" else detect_sb
    result = detect(img, 11, 8, _CFG)
    assert result is not None, f"{detector} 检测器未检出合成棋盘"
    assert result["detector"] == detector
    assert result["uv"].shape == (88, 2)
    assert result["uv"].dtype == np.float64
    assert result["recovery"] is None, "干净合成图不该触发缺陷闸门恢复"
    truth = canonicalize(corners, 11, 8)
    max_err = float(np.linalg.norm(result["uv"] - truth, axis=1).max())
    assert max_err < 0.5, f"{detector} 最大定位误差 {max_err:.3f}px 超过 0.5px"


def test_detector_window_rule_on_synthetic(make_chessboard_image):
    """合成图上两颗检测器返回的窗口都满足 2*win+1 < sq_px(结构防线延伸)。"""
    img, _, _ = make_chessboard_image(square_px=24.0, degrade="noise")
    for detect in (detect_classic, detect_sb):
        result = detect(img, 11, 8, _CFG)
        assert result is not None
        assert 2 * result["win_size"] + 1 < result["sq_px"]


def test_detect_returns_none_on_pure_noise():
    """纯噪声图:两颗检测器都未检出,统一返回 None(不返回半成品)。"""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)
    assert detect_classic(noise, 11, 8, _CFG) is None
    assert detect_sb(noise, 11, 8, _CFG) is None


def test_detect_sb_source_guard_single_refinement_call():
    """源码防线:二次亚像素精化在 detect.py 里恰好只出现 1 次(只在经典分支);
    SB 分支的注释用「二次亚像素精化」中文措辞,不会误撞这个断言。"""
    source = pathlib.Path("src/calib/corners/detect.py").read_text(encoding="utf-8")
    assert source.count("cv2.cornerSubPix(") == 1, (
        "二次亚像素精化的调用形式必须在全文件恰好出现 1 次(只许经典分支调用);"
        "SB 路线再精化属于头号污染错误"
    )


# ---------- 缺陷闸门与窗口阶梯恢复(158294 事故的修复) ----------

def test_grid_defects_clean_grid_has_zero(make_chessboard_image):
    """干净网格(强侧转真值角点,间距本来就渐变)必须数出零缺陷:
    闸门只抓硬伤,不许对正常透视误报。"""
    _img, corners, _ = make_chessboard_image(
        square_px=24.0, rvec=np.array([0.0, 0.35, 0.0]))
    d = grid_defects(corners, 11, 8)
    assert d["total"] == 0, f"干净网格被误报缺陷:{d}"
    assert d["n_dup"] == 0 and d["n_zigzag"] == 0


def test_grid_defects_flags_duplicate_and_zigzag(make_chessboard_image):
    """两类硬伤都必须被抓到:两点叠同处(重复点)+ 中间一点被拉走(锯齿)。"""
    _img, corners, _ = make_chessboard_image(square_px=24.0)
    bad = corners.copy()
    g = bad.reshape(8, 11, 2)
    g[1, 10] = g[1, 9]                # 重复点:相邻两角点叠在同处
    g[4, 5] += np.array([0.0, -8.0])  # 锯齿:列方向中间一点被拉离原位
    d = grid_defects(bad, 11, 8)
    assert d["n_dup"] >= 1, f"重复点没被抓到:{d}"
    assert d["n_zigzag"] >= 1, f"锯齿没被抓到:{d}"
    assert d["total"] == d["n_dup"] + d["n_zigzag"]


@pytest.mark.integration
def test_detect_classic_recovers_incident_image_158294():
    """真实事故帧回归:SB 检不出的 20260722_111500_158294,帧级窗口 18
    越过右缘局部小方格(25~29px),初次精化产生重复点/锯齿——缺陷闸门
    必须触发窗口阶梯恢复,恢复后网格零缺陷、窗口对齐头号陷阱不等式。"""
    path = pathlib.Path("input_data/test/20260722_111500_158294.jpg")
    if not path.exists():
        pytest.skip("本机没有真实数据 input_data,跳过事故帧回归")
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    result = detect_classic(gray, 11, 8, _CFG)
    assert result is not None, "事故帧 classic 应当能粗检出"
    rec = result["recovery"]
    assert rec is not None, "事故帧初次精化必有缺陷,闸门必须触发恢复"
    assert rec["defects_before"] >= 1
    assert rec["defects_after"] == 0, f"恢复后仍有缺陷:{rec}"
    assert rec["used_win"] < rec["initial_win"], "恢复窗口必须小于闯祸的帧级窗口"
    assert 2 * result["win_size"] + 1 < result["sq_px"], "恢复后仍须满足头号陷阱不等式"
    # 恢复后网格的相邻间距全部健康:没有任何"叠点"(最小间距远离 0)
    assert grid_defects(result["uv"], 11, 8)["total"] == 0
    min_spacing = float(np.linalg.norm(
        result["uv"].reshape(8, 11, 2)[:, 1:, :] - result["uv"].reshape(8, 11, 2)[:, :-1, :],
        axis=2).min())
    assert min_spacing > 0.25 * result["sq_px"], f"恢复后仍有叠点:最小间距 {min_spacing:.2f}px"


# ---------- Task 2:连拍一致性断言 ----------

def test_burst_consistency_passes_on_quiet_burst(make_chessboard_image):
    """8 张同姿态图(真值 + 小幅噪声):断言通过,返回 dict 带齐四个键。"""
    _img, corners, sq = make_chessboard_image(square_px=20.0)
    base = canonicalize(corners, 11, 8)
    rng = np.random.default_rng(0)
    uvs = np.stack([base + rng.normal(0.0, 0.05, base.shape) for _ in range(8)])
    result = burst_consistency(uvs, sq,
                               median_tol=_CFG.burst_check.median_tol,
                               max_tol=_CFG.burst_check.max_tol)
    assert set(result) >= {"ok", "spread_median_px", "spread_max_px", "threshold_px"}
    assert result["ok"] is True
    assert result["spread_max_px"] < result["threshold_px"][1]


def test_burst_consistency_catches_flip_pollution(make_chessboard_image):
    """连拍里一帧被水平翻转污染:同名角点跳到棋盘对侧,断言必须不通过。"""
    _img, corners, sq = make_chessboard_image(square_px=20.0)
    base = canonicalize(corners, 11, 8)
    rng = np.random.default_rng(1)
    uvs = np.stack([base + rng.normal(0.0, 0.05, base.shape) for _ in range(8)])
    polluted = uvs.copy()
    g = polluted[3].reshape(8, 11, 2)
    polluted[3] = g[:, ::-1, :].reshape(-1, 2)  # 第 3 帧水平翻转
    result = burst_consistency(polluted, sq,
                               median_tol=_CFG.burst_check.median_tol,
                               max_tol=_CFG.burst_check.max_tol)
    assert result["ok"] is False
    assert result["spread_max_px"] > 100.0  # 翻转让同名角点跨到对侧,量级完全不同

