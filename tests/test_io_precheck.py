"""入库体检测试(IO-02):指标方向正确性 + 判定逻辑 + 永不自动剔除的行为锁定。

本文件前半是三个体检指标的方向性验证(清晰 vs 模糊/过曝/暗图/无棋盘),
后半(随 Task 2/3 补齐)是双来源存疑判定、manifest 全字段、存疑原图副本
与跨轮对比表的行为锁定。
"""

from __future__ import annotations

import numpy as np

from calib.io.config import CornersConfig
from calib.io.precheck import board_roi, compute_metrics, exposure_clip, visibility_probe

# 测试统一用真实棋盘规格(11×8 内角点),夹具默认 640×480 画布
COLS, ROWS = 11, 8
CORNERS_CFG = CornersConfig()


# ---------------------------------------------------------------- 指标方向性

def test_blur_metric_direction_clear_vs_blurred(make_chessboard_image):
    """清晰棋盘 vs 高斯模糊版:blur_raw 与 blur_norm 都要显著更低(比值 < 1/3)。

    模糊破坏边缘锐度,拉普拉斯(二阶差分)响应随之大幅下降——
    这是"值越小越糊"这一方向的直接验证。
    """
    clear, _, sq_px = make_chessboard_image(square_px=20.0)
    blurred, _, _ = make_chessboard_image(square_px=20.0, degrade="blur")

    m_clear = compute_metrics(clear, COLS, ROWS, CORNERS_CFG)
    m_blur = compute_metrics(blurred, COLS, ROWS, CORNERS_CFG)

    assert m_clear["roi_found"] and m_blur["roi_found"]  # 两张都应框出棋盘
    assert m_blur["blur_raw"] < m_clear["blur_raw"] / 3.0
    assert m_blur["blur_norm"] < m_clear["blur_norm"] / 3.0
    # 归一值确实除掉了方格尺度:与原始值不是同一个数
    assert m_clear["blur_norm"] != m_clear["blur_raw"]
    assert sq_px > 0


def test_blur_norm_normalization_formula(make_chessboard_image):
    """比例化公式验证:blur_norm 恰等于 blur_raw / 实测方格像素边长²。

    归一的含义:边缘锐度能量随方格像素边长的平方缩放,把实测尺度除掉之后,
    不同拍摄距离(方格在画面里变大变小)的模糊度才能放在一起比。
    原始值与归一值必须明显不同——证明归一不是摆设。
    """
    img, _, expected_sq = make_chessboard_image(square_px=20.0)
    m = compute_metrics(img, COLS, ROWS, CORNERS_CFG)

    assert m["roi_found"] is True
    assert abs(m["sq_px"] - expected_sq) / expected_sq < 0.05  # 实测方格边长贴近期望
    assert abs(m["blur_norm"] - m["blur_raw"] / m["sq_px"] ** 2) < 1e-9
    assert abs(m["blur_norm"] - m["blur_raw"]) > 1.0  # 除以 ~400 后两者必然拉开


def test_overexpose_high_clip_ratio(make_chessboard_image):
    """过曝图(像素截断到 250 以上):高光裁剪占比 > 0.3。"""
    over, _, _ = make_chessboard_image(degrade="overexpose")
    hi, lo = exposure_clip(over)
    assert hi > 0.3
    assert lo < 0.3  # 过曝图的死黑不应多


def test_dark_image_shadow_clip_ratio():
    """暗图(像素压到 5 以下):阴影裁剪占比 > 0.3。"""
    dark = np.full((240, 320), 3, dtype=np.uint8)
    hi, lo = exposure_clip(dark)
    assert lo > 0.3
    assert hi == 0.0


def test_pure_noise_visibility_probe_fails():
    """纯噪声图(无棋盘):双检测器探针都失败,角点数检查也为否。"""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    probe = visibility_probe(noise, COLS, ROWS, CORNERS_CFG)
    assert probe["classic_ok"] is False
    assert probe["sb_ok"] is False
    assert probe["corners_count_ok"] is False


def test_board_roi_falls_back_to_full_image_on_noise():
    """无棋盘图:board_roi 回退整张图,不抛异常,roi_found=False。"""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    roi, corners, found = board_roi(noise, COLS, ROWS)
    assert found is False
    assert corners is None
    assert roi.shape == noise.shape  # 回退整张图


def test_compute_metrics_returns_all_documented_keys(make_chessboard_image):
    """指标函数返回 dict 含全部约定键(blur_raw/blur_norm/expo_hi/expo_lo/
    classic_ok/sb_ok/box_area_ratio/corners_count_ok),纯函数不抛异常。"""
    img, _, _ = make_chessboard_image(square_px=20.0)
    m = compute_metrics(img, COLS, ROWS, CORNERS_CFG)
    for key in ("blur_raw", "blur_norm", "expo_hi", "expo_lo", "classic_ok",
                "sb_ok", "box_area_ratio", "corners_count_ok", "roi_found", "sq_px"):
        assert key in m, f"缺少约定键 {key}"
    assert 0.0 <= m["expo_hi"] <= 1.0 and 0.0 <= m["expo_lo"] <= 1.0
    assert 0.0 < m["box_area_ratio"] <= 1.0
    assert m["classic_ok"] is True or m["sb_ok"] is True
