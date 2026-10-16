"""角点排序规范化 + 同姿态连拍一致性断言(D-09 的两道防线)。

为什么需要规范化:两颗检测器对同一块棋盘给出的角点序列虽然都是"行主序"
(先遍历一行内的列、再换下一行),但**起点和方向可以不同**——同一块板,
一份序列从左上角开始、另一份可能从右下角开始(OpenCV 的已知行为)。
角点编号的意义完全取决于这个起点:不做统一,下游"第 37 号角点"在两份
产物里指向棋盘上两个不同的物理点,一切对比、断言、统计全部作废。

规范化规则(纯几何、确定性、逐帧独立,不依赖任何跨帧参照):
计算行主方向 e_row(相邻列角点差向量的平均)与列主方向 e_col(相邻行
角点差向量的平均),要求最终形态满足「行方向朝画面右(e_row.x > 0)、
列方向朝画面下(e_col.y > 0)」。水平翻转控制 e_row.x 的符号、垂直翻转
控制 e_col.y 的符号,两个约束相互独立,四种排列变体
(恒等 / 旋转180° / 水平翻转 / 垂直翻转)中**恰有一个**被映到唯一形态。

连拍一致性断言:同一姿态的连拍几乎同一姿态(只差手抖与噪声),规范化后
同名角点的位置应当在亚像素量级内重合。若某帧的排序被镜像/翻转污染,
同编号角点会跳到棋盘对侧(距离约为棋盘尺寸),断言立刻爆掉——
这就是"翻转污染入盘前被点名"的机制。断言失败不抛异常,返回结果字典,
由调用方记入检测质量摘要并标记该帧存疑(只标记,不剔除,D-01)。
"""

from __future__ import annotations

import numpy as np


def canonicalize(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """把行主序角点序列规范化为唯一形态:行方向朝画面右、列方向朝画面下。

    corners: (rows*cols, 2) 任意变体的行主序角点;
    返回:  (rows*cols, 2) 规范化后的新数组(不修改输入)。

    步骤:排成 (rows, cols, 2) 网格 → 看行主方向朝不朝右,不朝右就在每行内
    反转顺序(水平翻转)→ 再看列主方向朝不朝下,不朝下就反转行顺序(垂直
    翻转)。先检查的量在翻转前算,后检查的量在翻转后的网格上重算,
    保证两个判断各自基于当前形态、互不干扰。
    """
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)

    # 行主方向(相邻列差向量的平均).x < 0 → 行方向朝左 → 每行内反转
    e_row_x = float((g[:, 1:, :] - g[:, :-1, :])[..., 0].mean())
    if e_row_x < 0:
        g = g[:, ::-1, :]

    # 列主方向(相邻行差向量的平均).y < 0 → 列方向朝上 → 反转行顺序
    e_col_y = float((g[1:, :, :] - g[:-1, :, :])[..., 1].mean())
    if e_col_y < 0:
        g = g[::-1, :, :]

    return g.reshape(-1, 2).copy()


def burst_consistency(uvs: np.ndarray, sq_px: float,
                      median_tol: float = 0.05, max_tol: float = 0.3) -> dict:
    """同姿态连拍的一致性断言:同名角点在连拍内的位置波动必须极小。

    uvs: (B, 点数, 2) 同姿态 B 张连拍**各自规范化后**的角点数组;
    sq_px: 本组连拍的方格像素边长(容差以它的倍数表达,不写死像素,Pitfall 9);
    median_tol / max_tol: 波动中位数与最大值的容差倍数(默认 0.05 / 0.3)。

    判定:对每个角点编号算波动 spread = max − min(逐坐标取极差再求范数),
    通过条件 median(spread) < median_tol×sq_px 且 max(spread) < max_tol×sq_px。
    连拍几乎同姿态,预期波动在亚像素级;若某帧排序被镜像/翻转污染,
    同编号角点跳到棋盘对侧,spread 直接放大几百倍,断言立刻不通过。

    返回 dict(ok, spread_median_px, spread_max_px, threshold_px);
    ok=False **不抛异常**——由调用方记入质量摘要并标记该帧存疑(D-01 只标记)。
    """
    uvs = np.asarray(uvs, dtype=np.float64)
    spread = np.linalg.norm(uvs.max(axis=0) - uvs.min(axis=0), axis=1)
    spread_median = float(np.median(spread))
    spread_max = float(spread.max())
    thr_median = float(median_tol) * float(sq_px)
    thr_max = float(max_tol) * float(sq_px)
    return {
        "ok": bool(spread_median < thr_median and spread_max < thr_max),
        "spread_median_px": spread_median,
        "spread_max_px": spread_max,
        "threshold_px": [thr_median, thr_max],
    }
