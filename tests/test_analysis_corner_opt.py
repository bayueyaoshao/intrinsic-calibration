"""COR-02 角点退化寻优库层测试(05-05 Task 1:退化原语/种子四元组/方法候选)。

全部用合成棋盘投影夹具(conftest 的 make_chessboard_image):图像自带真值
角点,退化是否恒等、候选窗口是否合规、检测是否找得准都能就地量出来。
另含一道源码防线测试:corner_opt.py 是调度与统计层,检测只能走
calib.corners.detect 两个入口——源码里绝不出现 cv2 角点函数字样。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from calib.analysis.corner_opt import (
    _seed_for,
    apply_degradation,
    build_method_candidates,
    detect_with,
)
from calib.corners.canonical import canonicalize
from calib.corners.detect import detect_classic
from calib.corners.window import estimate_sq_px
from calib.io.config import AnalyzeConfig, CornersConfig

# 检测器测试用配置:直接用数据类默认值(与仓库 calibration.yaml 一致)
_CFG = CornersConfig()


# ---------------------------------------------------------------------------
# 退化原语 apply_degradation
# ---------------------------------------------------------------------------


def test_apply_degradation_deterministic_same_seed():
    """同种子两次施加同档加噪,退化副本逐位一致(可复现承诺的根基)。"""
    rng0 = np.random.default_rng(7)
    img = rng0.integers(0, 256, size=(120, 160), dtype=np.uint8)
    out_a = apply_degradation(img, "noise", 8.0, np.random.default_rng(12345))
    out_b = apply_degradation(img, "noise", 8.0, np.random.default_rng(12345))
    assert np.array_equal(out_a, out_b)
    # 换种子必须变(否则"随机"是假的)
    out_c = apply_degradation(img, "noise", 8.0, np.random.default_rng(54321))
    assert not np.array_equal(out_a, out_c)


def test_apply_degradation_identity_levels_bitwise_equal_input():
    """恒等档(噪 0 / 糊 0 / 对比 1.0)输出逐位等于输入——校准档的零位移前提。"""
    img = np.random.default_rng(3).integers(0, 256, size=(96, 128), dtype=np.uint8)
    for family, level in (("noise", 0.0), ("blur", 0.0), ("contrast", 1.0)):
        out = apply_degradation(img, family, level, np.random.default_rng(0))
        assert np.array_equal(out, img), f"{family} 恒等档 {level} 改动了图像"
        assert out.dtype == np.uint8


def test_apply_degradation_noise64_changes_image_and_stays_in_range():
    """σ=64 强加噪:图像确实变了,且收尾量化后仍是 uint8、值域 [0, 255]。"""
    img = np.full((80, 100), 128, dtype=np.uint8)
    out = apply_degradation(img, "noise", 64.0, np.random.default_rng(9))
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img), "σ=64 的加噪不可能不改动图像"
    assert int(out.min()) >= 0 and int(out.max()) <= 255


def test_apply_degradation_unknown_family_chinese_error():
    """未知退化族当场中文报错(T-05-19:坏配置不许静默当恒等处理)。"""
    img = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="退化族"):
        apply_degradation(img, "jpeg", 5.0, np.random.default_rng(0))


def test_apply_degradation_invalid_level_chinese_error():
    """负数 / 非有限强度当场中文报错(强度没有负值,NaN 不许混进档位)。"""
    img = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(AssertionError, match="退化强度不合法"):
        apply_degradation(img, "noise", -1.0, np.random.default_rng(0))
    with pytest.raises(AssertionError, match="退化强度不合法"):
        apply_degradation(img, "blur", float("nan"), np.random.default_rng(0))


def test_apply_degradation_rejects_non_2d_input():
    """非二维输入(比如误传彩色图)当场中文报错,不喂进检测器。"""
    color = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(AssertionError, match="二维灰度图"):
        apply_degradation(color, "noise", 1.0, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# 种子四元组 _seed_for
# ---------------------------------------------------------------------------


def test_seed_quadruplet_deterministic_and_discriminating():
    """四元组任一元不同种子就不同;同四元组永远同种子;值域落在 32 位内。"""
    base = 20261021
    kw = {"image_id": "img_01", "unit_label": "noise_8.0",
          "method_label": "classic_w5", "rep": 0, "base_seed": base}
    ref = _seed_for(**kw)
    assert _seed_for(**kw) == ref, "同四元组必须永远同种子"
    assert 0 <= ref < 2**32
    for tweak in (
            {"image_id": "img_02"}, {"unit_label": "noise_16.0"},
            {"method_label": "sb"}, {"rep": 1}, {"base_seed": base + 1}):
        assert _seed_for(**{**kw, **tweak}) != ref, f"改 {tweak} 种子必须变"


# ---------------------------------------------------------------------------
# 方法候选 build_method_candidates / 检测调度 detect_with
# ---------------------------------------------------------------------------


def test_build_method_candidates_filters_ladder_by_sq_px():
    """窗口阶梯按 2w+1 < sq_px 过滤:大方格容纳大窗口,小方格自动缩档不报错。"""
    cfg = AnalyzeConfig()
    big = build_method_candidates(40.0, cfg)
    big_labels = [c["label"] for c in big]
    assert "classic_w13" in big_labels, "2×13+1=27 < 40,最大档必须进候选"
    assert big_labels.count("sb") == 1 and "sb_acc" in big_labels

    tiny = build_method_candidates(9.0, cfg)
    tiny_classic = [c["label"] for c in tiny if c["kind"] == "classic"]
    assert tiny_classic == ["classic_w3"], (
        f"sq_px=9 时只有 w3 合法(2×3+1=7<9),实际 {tiny_classic}")
    assert "sb" in [c["label"] for c in tiny], "SB 两档不受方格边长过滤"


def test_build_method_candidates_full_profile_adds_exhaustive():
    """full 档追加穷举搜索候选 sb_exh,quick 档没有。"""
    quick = [c["label"] for c in build_method_candidates(40.0, AnalyzeConfig())]
    full = [c["label"] for c in
            build_method_candidates(40.0, AnalyzeConfig(), profile="full")]
    assert "sb_exh" not in quick
    assert "sb_exh" in full


def test_detect_with_candidates_find_truth_under_1px(make_chessboard_image):
    """每个合法候选在干净合成图上都检出 88 个角点,对真值最大位移 < 1px。

    这是"方法 = 检测入口参数,不新增实现"的行为锁:候选经 detect_with
    调度后必须真的能检出——而不是返回 None 或错位点。
    """
    img, corners, _ = make_chessboard_image()
    sq_px = estimate_sq_px(corners, 11, 8)
    truth = canonicalize(corners, 11, 8)
    candidates = build_method_candidates(float(sq_px), AnalyzeConfig())
    assert candidates, "合成图方格边长下至少应有一个合法候选"
    for cand in candidates:
        result = detect_with(cand, img, 11, 8, _CFG)
        assert result is not None, f"候选 {cand['label']} 未检出合成棋盘"
        assert result["uv"].shape == (88, 2)
        max_err = float(np.linalg.norm(result["uv"] - truth, axis=1).max())
        assert max_err < 1.0, (
            f"候选 {cand['label']} 最大定位误差 {max_err:.3f}px 超过 1px")


def test_force_win_pins_window_and_unsafe_window_rejected(make_chessboard_image):
    """force_win 指定窗口生效(win_size 钉在指定档),违反安全断言当场报错。"""
    img, corners, _ = make_chessboard_image(square_px=20.0)
    sq_px = estimate_sq_px(corners, 11, 8)

    pinned = replace(_CFG, subpix=replace(_CFG.subpix, force_win=5))
    result = detect_classic(img, 11, 8, pinned)
    assert result is not None
    assert result["win_size"] == 5, "指定窗口必须被逐字采用,不被常规推导顶掉"

    # 2×12+1 = 25 不小于本帧方格像素边长(约 20):头号陷阱防线必须在场
    unsafe = replace(_CFG, subpix=replace(_CFG.subpix, force_win=12))
    with pytest.raises(AssertionError, match="指定窗口断言失败"):
        detect_classic(img, 11, 8, unsafe)
    assert 2 * 12 + 1 >= sq_px * 0.99  # 前提自检:这一档在本图上确实不安全


def test_corner_opt_source_has_no_cv2_corner_calls():
    """源码防线:corner_opt.py 里绝不出现 cv2 角点函数字样(检测唯一入口)。"""
    source = Path(__file__).resolve().parents[1].joinpath(
        "src", "calib", "analysis", "corner_opt.py").read_text(encoding="utf-8")
    assert "cornerSubPix" not in source, "二次亚像素精化只允许在 corners/detect.py"
    assert "findChessboardCorners" not in source, (
        "粗定位只允许在 corners/detect.py;corner_opt 只走 detect_classic/detect_sb")
