"""环境冒烟测试:三件套版本锁定验证 + 研究假设(A1/A2/A6)本地事实确认。

研究阶段(01-RESEARCH.md「Assumptions Log」)因沙箱网络受限,无法本地验证
三个 OpenCV 接口在 4.14 的可用性;本文件就是"首日确认"的落点:
- 接口存在 → 断言通过,下游计划可依赖;
- 接口不存在 → pytest.skip 并写明理由,结论记入 SUMMARY,下游走回退路径。
"""

import cv2
import numpy
import pytest
import scipy


def test_opencv_locked_to_4x():
    """OpenCV 必须是 4.x:锁定 <5.0(5.0 刚发布,calib3d 行为易变,与可审计目标冲突)。"""
    assert cv2.__version__.startswith("4."), f"期望 OpenCV 4.x,实际 {cv2.__version__}"
    assert int(cv2.__version__.split(".")[0]) < 5


def test_numpy_scipy_pair():
    """numpy 锁 2.5 系、scipy 锁 1.18 系:两者是成对验证过的组合,不能单独升级。"""
    assert numpy.__version__.startswith("2.5"), f"期望 numpy 2.5.x,实际 {numpy.__version__}"
    assert scipy.__version__.startswith("1.18"), f"期望 scipy 1.18.x,实际 {scipy.__version__}"


def test_hypothesis_a6_calibrate_camera_extended():
    """研究假设 A6:calibrateCameraExtended 绑定是否可用。

    标定需要逐视图(每张照片)的重投影误差,Extended 版本直接返回;
    若不可用,回退路径是 calibrateCamera + projectPoints 自算(多几行)。
    """
    if not hasattr(cv2, "calibrateCameraExtended"):
        pytest.skip(
            "A6 不成立:本 OpenCV 构建无 calibrateCameraExtended 绑定,"
            "下游标定改用 calibrateCamera + projectPoints 自算逐视图误差"
        )
    assert callable(cv2.calibrateCameraExtended)


def test_hypothesis_a2_find_chessboard_corners_sb():
    """研究假设 A2:findChessboardCornersSB 是否在主模块可用。

    SB 是基于鞍点检测的新款棋盘角点检测器(自带亚像素精度);
    双检测器并跑是既定决策,若 SB 缺失则只剩经典路线,需重新评估。
    """
    if not hasattr(cv2, "findChessboardCornersSB"):
        pytest.skip(
            "A2 不成立:本 OpenCV 构建无 findChessboardCornersSB,"
            "角点检测只剩经典路线,双检测器对比需要重新评估"
        )
    assert callable(cv2.findChessboardCornersSB)


def test_hypothesis_a1_estimate_chessboard_sharpness():
    """研究假设 A1:estimateChessboardSharpness 是否可用。

    它按角点边缘陡度算棋盘局部清晰度,是模糊度指标的候选增强项;
    主方案(拉普拉斯方差)不依赖它,缺失只影响增强项。
    """
    if not hasattr(cv2, "estimateChessboardSharpness"):
        pytest.skip(
            "A1 不成立:本 OpenCV 构建无 estimateChessboardSharpness,"
            "体检模糊度指标用拉普拉斯方差方案(主方案),不受影响"
        )
    assert callable(cv2.estimateChessboardSharpness)
