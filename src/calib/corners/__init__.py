"""角点提取层:棋盘角点检测、亚像素精化、排序规范化。

流水线第②步「找棋盘角点」的承载模块:
经典检测器(findChessboardCorners + cornerSubPix)与新款检测器
(findChessboardCornersSB)双路线并跑、各存一份,供后续迭代回环对比;
亚像素窗口从实测方格像素边长按比例推导,绝不写死像素值。
"""
