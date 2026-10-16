"""corners 模块测试:窗口推导(Task 1)+ 规范化/连拍断言/双检测器(Task 2)。

全部用合成棋盘投影夹具(conftest 的 make_chessboard_image):它自带真值角点,
检测对不对、误差几个像素,都有"标准答案"可对照——不靠肉眼,靠断言。
"""

from __future__ import annotations

import numpy as np
import pytest

from calib.corners.window import (
    derive_window_frame,
    derive_window_per_corner,
    estimate_sq_px,
)

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
