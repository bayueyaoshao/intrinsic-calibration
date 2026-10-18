"""Immerkaer 方法:用一个固定的 3×3 卷积核从单张照片估计噪声强度,不需要连拍。

白话原理:这颗卷积核的九个系数加起来是 0,所以在平坦区域(没有纹理、
没有边缘的地方)卷积响应全是 0——只有噪声会留下响应;而响应的波动幅度
与噪声强度成正比,比例系数可以精确算出来。于是不靠连拍对比,单张照片
就能估出像素噪声的标准差 σ(单位:灰阶)。

两种估计并报(研究 Pitfall 5):
- sigma_std:标准式,全部卷积响应取绝对值后求平均——整图口径,纹理重的
  照片会被纹理与 JPEG 压缩痕迹抬高;
- sigma_robust:稳健式,取响应绝对值的中位数——中位数不受个别大片纹理
  影响,真实 JPEG 图上更接近真实噪声底,是配置里的默认主线口径。

公式出处:Immerkaer, "Fast Noise Variance Estimation", CVIU 1996。
公式常数由 tests/test_noise_immerkaer.py 的"已知噪声回收 + 5×5 手算对照"
测试当场锁死(研究假设 A2:常数出自论文记忆,常数错了那里先红)。

本模块是纯数值函数:不 import 本项目任何其他模块、不改任何文件——
给它一张灰度图,它还回两个数,没有副作用。
"""

from __future__ import annotations

import cv2
import numpy as np

# Immerkaer 卷积核 L = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]。
# 它的噪声放大因子 = sqrt(ΣL²) = sqrt(1+4+1+4+16+4+1+4+1) = 6:
# 白噪声经过这颗核,响应的标准差恰是原噪声 σ 的 6 倍——
# 所以下面两个公式最后都要除以 6 才是噪声本身。
# 1.4826 = 中位绝对偏差到标准差的换算系数(正态分布下 |X| 的中位数 ≈ 0.6745σ,
# 1/0.6745 ≈ 1.4826)。
_KERNEL = np.array([[1.0, -2.0, 1.0],
                    [-2.0, 4.0, -2.0],
                    [1.0, -2.0, 1.0]])


def immerkaer_sigma(gray: np.ndarray) -> dict:
    """单帧像素噪声 σ 估计:整图标准式 + 稳健中位数式,两值并报。

    参数:
        gray: 二维灰度图(uint8 或 float 数组都行,内部统一转 float64)。

    返回:
        {"sigma_std": float, "sigma_robust": float},单位灰阶。
        sigma_std = sqrt(π/2) / (6·h·w) × Σ|N|(N 为卷积响应图);
        sigma_robust = 1.4826 × median(|N|) / 6。

    异常:
        输入不是二维数组、或任一边短于 3 像素:抛中文 AssertionError 点名
        (卷积核就是 3×3,更小的图没有可计算的内圈)。
    """
    arr = np.asarray(gray)
    if arr.ndim != 2:
        raise AssertionError(
            f"immerkaer_sigma 的输入必须是二维灰度图;实际是 {arr.ndim} 维,"
            f"形状 {arr.shape}。请先转灰度(cv2.imread 用 IMREAD_GRAYSCALE 读入即可)。"
        )
    h, w = arr.shape
    if h < 3 or w < 3:
        raise AssertionError(
            f"immerkaer_sigma 的输入至少要 3×3 像素(卷积核就是 3×3,"
            f"更小的图没有内圈可算);实际形状是 {arr.shape}。请检查图像是否读全。"
        )

    # cv2.filter2D 的默认补边方式只影响最外一圈的响应;随后裁掉最外圈,
    # 保留下来的响应全部由原图真实像素算出,不受补边值污染
    n_map = cv2.filter2D(arr.astype(np.float64), -1, _KERNEL)[1:-1, 1:-1]
    h, w = n_map.shape
    sigma_std = float(np.sqrt(np.pi / 2.0) / (6.0 * h * w) * np.abs(n_map).sum())
    sigma_robust = float(1.4826 * np.median(np.abs(n_map)) / 6.0)
    return {"sigma_std": sigma_std, "sigma_robust": sigma_robust}
