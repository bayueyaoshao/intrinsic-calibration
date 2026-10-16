"""双检测器统一封装(D-07:两颗都跑、各存一份;同一返回结构,下游不分叉)。

经典路线:cv2.findChessboardCorners(自适应阈值 + 归一化)先粗定位,
再做二次亚像素精化(把角点从整数像素细化到小数像素的迭代搜索),
精化窗口从本帧实测的方格像素边长推导(D-08,见 window.py),绝不写死像素。

新款路线(SB):cv2.findChessboardCornersSB 一次到位,**输出已是亚像素精度,
严禁再做二次亚像素精化**——两条路线的精化用的是不同的角点模型,叠在一起
互相污染、精度反而下降(Pitfall 4)。因此本模块里二次精化的调用全文件
只允许出现在经典分支。

两颗检测器的角点**排列顺序可能不同**(同块棋盘,起点/方向不一样的已知行为),
所以两条路线的输出都统一过规范化层(canonicalize)再返回——
下游拿到的两份结果,同一编号永远指向棋盘上同一个物理点(D-09)。

两颗检测器失败都返回 None(本帧该检测器未检出),绝不返回半成品。
"""

from __future__ import annotations

import cv2
import numpy as np

from calib.corners.canonical import canonicalize
from calib.corners.window import derive_window_frame, derive_window_per_corner, estimate_sq_px
from calib.io.config import CornersConfig

# calibration.yaml 里经典检测器的选项名 → cv2 常量(名字在配置层已校验过)
_CLASSIC_FLAG_VALUES = {
    "adaptive_thresh": cv2.CALIB_CB_ADAPTIVE_THRESH,
    "normalize_image": cv2.CALIB_CB_NORMALIZE_IMAGE,
    "fast_check": cv2.CALIB_CB_FAST_CHECK,
}


def _classic_flag_value(flag_names: tuple[str, ...] | list[str]) -> int:
    """把配置里的选项名组合成 cv2 的选项位掩码。"""
    value = 0
    for name in flag_names:
        value |= _CLASSIC_FLAG_VALUES[name]
    return value


def detect_classic(gray: np.ndarray, cols: int, rows: int, cfg: CornersConfig) -> dict | None:
    """经典检测器:粗定位 → 二次亚像素精化(窗口按实测方格边长)→ 规范化。

    返回 dict(uv (rows*cols, 2) float64, sq_px, win_size, detector="classic"),
    未检出返回 None。
    """
    found, raw = cv2.findChessboardCorners(
        gray, (cols, rows), _classic_flag_value(cfg.classic_flags))
    if not found:
        return None
    corners = raw.reshape(-1, 2).astype(np.float64)

    sq_px = estimate_sq_px(corners, cols, rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                cfg.subpix.max_iter, cfg.subpix.epsilon)

    # 两种窗口粒度统一成"逐组精化":帧级模式下所有角点共用一个窗口值,
    # 逐角点模式下各点用自己的窗口值;同窗口的角点分一组、一组调用一次
    # (精化函数一次只收一个窗口)。二次精化的调用因此全文件只有这一处,
    # 且只挂在经典分支——SB 路线严禁经过这里(见 detect_sb 的说明)。
    if cfg.subpix.window_granularity == "corner":
        wins = derive_window_per_corner(corners, cols, rows, cfg.subpix.min_win)
    else:
        wins = np.full(corners.shape[0],
                       derive_window_frame(sq_px, cfg.subpix.min_win), dtype=int)
    win_size = int(wins.max())
    refined = corners.astype(np.float32)  # 精化函数只收 32 位浮点点坐标
    for w in np.unique(wins):
        group = refined[wins == w].reshape(-1, 1, 2)
        refined[wins == w] = cv2.cornerSubPix(
            gray, group, (int(w), int(w)), (-1, -1), criteria).reshape(-1, 2)
    corners = refined.reshape(-1, 2).astype(np.float64)

    return {
        "uv": canonicalize(corners, cols, rows),
        "sq_px": sq_px,
        "win_size": win_size,
        "detector": "classic",
    }


def detect_sb(gray: np.ndarray, cols: int, rows: int, cfg: CornersConfig) -> dict | None:
    """新款(SB)检测器:一次检测直接输出亚像素角点 → 规范化。

    SB 输出已是亚像素精度,**绝不再做二次亚像素精化**——它内部的角点模型
    与经典路线的精化模型不是一回事,二次处理会互相污染、精度反降(Pitfall 4)。

    返回与 detect_classic 完全同构的 dict(detector="sb"),未检出返回 None。
    win_size 字段只为产物结构对齐而按同一公式推导入档(SB 路线并不使用窗口),
    下游审查两颗检测器的方格尺度时口径一致。
    """
    found, raw = cv2.findChessboardCornersSB(gray, (cols, rows), cfg.sb_flags)
    if not found:
        return None
    corners = raw.reshape(-1, 2).astype(np.float64)

    sq_px = estimate_sq_px(corners, cols, rows)
    win_size = derive_window_frame(sq_px, cfg.subpix.min_win)  # 仅记录,不参与任何精化

    return {
        "uv": canonicalize(corners, cols, rows),
        "sq_px": sq_px,
        "win_size": win_size,
        "detector": "sb",
    }
