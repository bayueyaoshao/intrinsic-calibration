"""Immerkaer 单帧像素噪声估计的回收测试(研究假设 A2 的常数锁定)。

**常数锁定测试(5×5 手算对照)对应研究假设 A2**:公式常数(sqrt(π/2) 系数、
核放大因子 6、中位数换算系数 1.4826)出自 1996 年论文的训练记忆,未能当场
查原文——公式常数写错了,这里先红,实现根本进不了主干。

**已知噪声回收测试(σ = 2.0 / 5.0 / 12.0 三档)**:往纯色底图注入已知强度的
高斯噪声,两种估计(标准式 / 稳健中位数式)都必须把该噪声强度收回来
(相对误差 < 5%)——这是 NOI-01"Immerkaer 单帧退化路径"的数学地基。
"""

from __future__ import annotations

import numpy as np
import pytest

from calib.noise.immerkaer import immerkaer_sigma


# ---- 行为 1:已知噪声回收(两式 × 三档,共 6 条断言,相对误差 < 5%)----

@pytest.mark.parametrize("sigma_true", [2.0, 5.0, 12.0])
def test_known_noise_recovered_std(sigma_true):
    """标准式(整图平均):纯色底图 + 已知高斯噪声,σ 回收相对误差 < 5%。"""
    rng = np.random.default_rng(20261018)
    flat = np.full((480, 640), 128.0)
    noisy = flat + rng.normal(0.0, sigma_true, size=flat.shape)
    out = immerkaer_sigma(noisy)
    rel = abs(out["sigma_std"] - sigma_true) / sigma_true
    assert rel < 0.05, (
        f"标准式回收失败:σ_true={sigma_true},估计 {out['sigma_std']:.4f},"
        f"相对误差 {rel:.2%}(超 5% 线,公式常数可能错了,查 A2)"
    )


@pytest.mark.parametrize("sigma_true", [2.0, 5.0, 12.0])
def test_known_noise_recovered_robust(sigma_true):
    """稳健式(中位数):纯色底图 + 已知高斯噪声,σ 回收相对误差 < 5%。"""
    rng = np.random.default_rng(20261018)
    flat = np.full((480, 640), 128.0)
    noisy = flat + rng.normal(0.0, sigma_true, size=flat.shape)
    out = immerkaer_sigma(noisy)
    rel = abs(out["sigma_robust"] - sigma_true) / sigma_true
    assert rel < 0.05, (
        f"稳健式回收失败:σ_true={sigma_true},估计 {out['sigma_robust']:.4f},"
        f"相对误差 {rel:.2%}(超 5% 线,公式常数可能错了,查 A2)"
    )


# ---- 行为 2:零噪声不虚报 ----

def test_zero_noise_reports_near_zero():
    """零噪声纯色图:两种估计都 < 0.5(纯平图的卷积响应应全为 0)。"""
    flat = np.full((480, 640), 128, dtype=np.uint8)
    out = immerkaer_sigma(flat)
    assert out["sigma_std"] < 0.5, f"零噪声图标准式虚报 {out['sigma_std']:.4f}"
    assert out["sigma_robust"] < 0.5, f"零噪声图稳健式虚报 {out['sigma_robust']:.4f}"


# ---- 行为 3:纹理图上稳健式 ≤ 标准式(方向性)----

def test_robust_le_std_on_texture(make_chessboard_image):
    """含棋盘纹理的合成图:稳健式 ≤ 标准式。

    中位数天然压低结构区域(黑白格边缘)的权重,标准式把纹理响应
    也平均进去——这个方向性成立是"主线默认用稳健式"的依据(研究 Pitfall 5)。
    """
    img, _corners, _sq = make_chessboard_image()
    out = immerkaer_sigma(img)
    assert out["sigma_robust"] <= out["sigma_std"], (
        f"纹理图上稳健式({out['sigma_robust']:.4f})反而大于"
        f"标准式({out['sigma_std']:.4f}),方向性不成立"
    )


# ---- 行为 4:5×5 手算对照,公式常数逐位锁定(atol 1e-12)----

def test_hand_computed_5x5_exact():
    """5×5 手算对照:输出与独立 numpy 实现逐位一致(atol 1e-12)。

    卷积响应 N 只在内圈 3×3 上逐像素手算(核系数上下左右对称,
    "相关"与"卷积"同结果):
    N[i,j] = G[i-1,j-1] - 2·G[i-1,j] + G[i-1,j+1]
           - 2·G[i,j-1]  + 4·G[i,j]   - 2·G[i,j+1]
           + G[i+1,j-1]  - 2·G[i+1,j] + G[i+1,j+1]
    """
    gray = np.array([
        [7.0, 23.0, 41.0, 59.0, 67.0],
        [13.0, 29.0, 47.0, 61.0, 71.0],
        [17.0, 31.0, 53.0, 73.0, 79.0],
        [19.0, 37.0, 43.0, 83.0, 89.0],
        [11.0, 5.0, 3.0, 97.0, 2.0],
    ])
    coeff = {
        (-1, -1): 1.0, (-1, 0): -2.0, (-1, 1): 1.0,
        (0, -1): -2.0, (0, 0): 4.0, (0, 1): -2.0,
        (1, -1): 1.0, (1, 0): -2.0, (1, 1): 1.0,
    }
    n_map = np.empty((3, 3))
    for i in range(1, 4):
        for j in range(1, 4):
            n_map[i - 1, j - 1] = sum(
                coeff[di, dj] * gray[i + di, j + dj]
                for di in (-1, 0, 1) for dj in (-1, 0, 1))

    h, w = n_map.shape
    expected_std = float(np.sqrt(np.pi / 2.0) / (6.0 * h * w) * np.abs(n_map).sum())
    expected_robust = float(1.4826 * np.median(np.abs(n_map)) / 6.0)

    out = immerkaer_sigma(gray)
    assert out["sigma_std"] == pytest.approx(expected_std, abs=1e-12), (
        f"标准式与手算不一致:输出 {out['sigma_std']!r},手算 {expected_std!r}"
        f"(A2 公式常数错了)"
    )
    assert out["sigma_robust"] == pytest.approx(expected_robust, abs=1e-12), (
        f"稳健式与手算不一致:输出 {out['sigma_robust']!r},手算 {expected_robust!r}"
        f"(A2 公式常数错了)"
    )


# ---- 入口断言:非二维 / 过小的图给中文报错 ----

def test_rejects_non_2d_input():
    """一维输入:中文 AssertionError 点名"必须二维灰度图"。"""
    with pytest.raises(AssertionError, match="二维"):
        immerkaer_sigma(np.zeros(10))


def test_rejects_tiny_image():
    """小于 3×3 的图:卷积核就是 3×3,中文 AssertionError 点名。"""
    with pytest.raises(AssertionError, match="3×3"):
        immerkaer_sigma(np.zeros((2, 2)))
