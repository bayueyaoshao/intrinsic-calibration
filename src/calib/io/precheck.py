"""入库体检的三项指标(IO-02):模糊度、曝光、棋盘可见性——全部天然比例化。

比例化的意思:指标的数值不依赖"图片多大、棋盘占多少像素"这类绝对尺度,
换一批数据(分辨率不同、棋盘远近不同)指标依然可比,阈值不用重调。三个指标
各自的比例化方式:

1. 模糊度——拉普拉斯方差(拉普拉斯 = 对图像做二阶差分,边缘越锐利响应越大,
   数值越小说明画面越糊),只在棋盘检测框内计算,再除以"方格像素边长"的平方。
   整图算会把背景内容与分辨率混进来,棋盘框内计算把度量锚定在标定真正在乎的
   区域;锐度能量随方格像素边长的平方缩放,除掉它之后不同拍摄距离也可比。
2. 曝光——灰度直方图裁剪比例:高光裁剪 = 灰度 >= 250 的像素占比,
   阴影裁剪 = 灰度 <= 5 的像素占比。占比本身就是比例,天然与分辨率无关
   (JPEG 是 8 位灰度,范围固定 0–255,这两个界限是全行业通用的经验值)。
3. 棋盘可见性——两颗检测器(经典 findChessboardCorners 与新款
   findChessboardCornersSB)各问一次"看没看见",记录各自的成败布尔、
   棋盘检测框面积占全图的比例、检出角点数是否恰好对上棋盘规格。

探针的克制的(务必维持):体检只问"看没看见",拿到原始检测结果就收手——
不做把角点从整数像素细化到小数像素的二次搜索,也不做角点排列顺序的统一整理,
那些都是角点阶段(②)的事。体检阶段若重复做,既浪费算力,也会让
"体检没看见、角点阶段却检出了"这类有价值的差异无从对比。

本模块全部是纯函数:读图、算数、返回字典,不修改任何输入,也没有任何
删除文件的动作——体检只标记、不剔除(D-01)。
"""

from __future__ import annotations

import cv2
import numpy as np

from calib.corners.window import estimate_sq_px
from calib.io.config import CornersConfig

# 曝光裁剪的灰度界限(8 位灰度 0–255 的通用经验值,不是"像素多少"那种尺度依赖)
HIGH_LIGHT_LIMIT = 250  # 灰度 >= 250 视为高光裁剪(烧白,层次丢失)
SHADOW_LIMIT = 5        # 灰度 <= 5 视为阴影裁剪(死黑,层次丢失)

# calibration.yaml 里经典检测器的选项名 → cv2 常量(与 corners/detect.py 同一套名字;
# 这里独立翻译是为了不引入角点阶段的精化与整理逻辑,体检只碰最原始的检测调用)
_CLASSIC_FLAG_VALUES = {
    "adaptive_thresh": cv2.CALIB_CB_ADAPTIVE_THRESH,
    "normalize_image": cv2.CALIB_CB_NORMALIZE_IMAGE,
    "fast_check": cv2.CALIB_CB_FAST_CHECK,
}


def exposure_clip(gray: np.ndarray) -> tuple[float, float]:
    """曝光裁剪比例:返回 (高光裁剪占比, 阴影裁剪占比)。

    高光裁剪 = 灰度 >= 250 的像素占比(画面烧白的部分);
    阴影裁剪 = 灰度 <= 5 的像素占比(画面死黑的部分)。
    在整张图上计算——占比本身就是比例,分辨率无关。
    """
    total = gray.size
    hi = float((gray >= 250).sum()) / total
    lo = float((gray <= 5).sum()) / total
    return hi, lo


def board_roi(gray: np.ndarray, cols: int, rows: int,
              corners_found: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """定位棋盘检测框,供模糊度在棋盘区域内计算(比例化的第一层)。

    查找顺序:
    1. 调用方已带进一次成功检测的角点(corners_found,复用可见性探针的结果,
       免去同一条检测在同一家检测器上白跑第二遍);
    2. 快速预筛(选项 FAST_CHECK:先快速判断图里"像不像有棋盘",
       像才肯花完整查找的时间——只作预筛加速,不作判决);
    3. 完整查找(自适应阈值 + 亮度归一化)。
    都失败:回退用整张图,roi_found=False(此时没有角点,归一值无从谈起,
    由调用方把归一值记为"缺失")。

    返回 (框内图像区域, 检出的角点或 None, 是否成功框出棋盘)。
    """
    corners: np.ndarray | None = None
    if corners_found is not None:
        corners = np.asarray(corners_found, dtype=np.float64).reshape(-1, 2)
    else:
        # 快速预筛:失败不代表真没有,必须回退完整查找再试一次
        found, raw = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_FAST_CHECK)
        if not found:
            found, raw = cv2.findChessboardCorners(
                gray, (cols, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            corners = raw.reshape(-1, 2).astype(np.float64)

    if corners is None:
        return gray, None, False

    # 检测框向外扩一圈再裁:检出的角点只落在外圈角点位置上,
    # 棋盘最外侧还有一圈黑白格,把模糊度计算的窗口补完整。
    square_px = estimate_sq_px(corners, cols, rows)
    pad = int(np.ceil(square_px))
    height, width = gray.shape[:2]
    x0 = max(int(np.floor(corners[:, 0].min())) - pad, 0)
    y0 = max(int(np.floor(corners[:, 1].min())) - pad, 0)
    x1 = min(int(np.ceil(corners[:, 0].max())) + pad, width)
    y1 = min(int(np.ceil(corners[:, 1].max())) + pad, height)
    return gray[y0:y1, x0:x1], corners, True


def visibility_probe(gray: np.ndarray, cols: int, rows: int, cfg: CornersConfig) -> dict:
    """棋盘可见性探针:两颗检测器各问一次"看没看见"(只探成败,不做精化)。

    经典检测器按配置里的选项调用 findChessboardCorners,
    新款(SB)按配置的选项位掩码调用 findChessboardCornersSB,各跑一次。
    返回 dict:
        classic_ok / sb_ok   两颗各自的成败布尔
        box_area_ratio       棋盘检测框面积 / 全图面积(谁检出用谁,经典优先)
        corners_count_ok     是否至少一颗检出、且角点数恰好等于 cols×rows
        classic_corners / sb_corners   各自的原始角点(未精化未整理,体检内部复用,
                             不入盘;键名带 corners 是为了与产物里的坐标区分)
    """
    classic_flags = 0
    for name in cfg.classic_flags:
        classic_flags |= _CLASSIC_FLAG_VALUES[name]

    ok_c, raw_c = cv2.findChessboardCorners(gray, (cols, rows), classic_flags)
    ok_s, raw_s = cv2.findChessboardCornersSB(gray, (cols, rows), cfg.sb_flags)

    height, width = gray.shape[:2]
    box_area_ratio = float("nan")
    if ok_c or ok_s:
        pts = (raw_c if ok_c else raw_s).reshape(-1, 2).astype(np.float64)
        box_w = float(pts[:, 0].max() - pts[:, 0].min())
        box_h = float(pts[:, 1].max() - pts[:, 1].min())
        box_area_ratio = box_w * box_h / (width * height)

    corners_count_ok = False
    for ok, raw in ((ok_c, raw_c), (ok_s, raw_s)):
        if ok and raw.reshape(-1, 2).shape[0] == cols * rows:
            corners_count_ok = True

    return {
        "classic_ok": bool(ok_c),
        "sb_ok": bool(ok_s),
        "box_area_ratio": box_area_ratio,
        "corners_count_ok": corners_count_ok,
        "classic_corners": raw_c.reshape(-1, 2).astype(np.float64) if ok_c else None,
        "sb_corners": raw_s.reshape(-1, 2).astype(np.float64) if ok_s else None,
    }


def compute_metrics(gray: np.ndarray, cols: int, rows: int, cfg: CornersConfig) -> dict:
    """对一张图算全部体检指标,返回扁平 dict(manifest 每张照片都带全这些数字)。

    返回键:
        blur_raw        棋盘框内拉普拉斯方差(原始值,越小越糊)
        blur_norm       blur_raw / 方格像素边长²(归一值,跨拍摄距离可比;
                        没框出棋盘时为 NaN——缺失,判定阶段会跳过它)
        expo_hi/lo      高光/阴影裁剪占比(全图)
        classic_ok/sb_ok          双检测器各自成败
        box_area_ratio  棋盘检测框面积占比
        corners_count_ok          角点数恰好对上棋盘规格
        roi_found       是否成功框出棋盘(没有时 blur_norm 为缺失)
        sq_px           实测方格像素边长(没有时为 NaN)
    """
    probe = visibility_probe(gray, cols, rows, cfg)

    # 模糊度:复用探针的检测结果定位棋盘框,免得同一颗检测器白跑第二遍
    roi, corners, roi_found = board_roi(
        gray, cols, rows,
        corners_found=probe["classic_corners"] if probe["classic_ok"] else None)
    blur_raw = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    if corners is not None:
        sq_px = estimate_sq_px(corners, cols, rows)
        blur_norm = blur_raw / (sq_px ** 2)
    else:
        sq_px = float("nan")
        blur_norm = float("nan")  # 归一值缺失:由可见性结论体现,判定阶段自动跳过

    expo_hi, expo_lo = exposure_clip(gray)

    return {
        "blur_raw": blur_raw,
        "blur_norm": blur_norm,
        "expo_hi": expo_hi,
        "expo_lo": expo_lo,
        "classic_ok": probe["classic_ok"],
        "sb_ok": probe["sb_ok"],
        "box_area_ratio": probe["box_area_ratio"],
        "corners_count_ok": probe["corners_count_ok"],
        "roi_found": roi_found,
        "sq_px": sq_px,
    }
