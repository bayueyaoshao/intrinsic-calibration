"""全量带角点标记的副本(D-06/D-07):每张照片、每颗检测器各留一份标记副本。

为什么要有它:"角点是否可信"最终要靠人眼核查——汇总拼图只是索引,
这里的全量副本才是逐张核查的证据本体。经典 classic 与新款 sb 各存一套,
人工可以并排看同一张照片哪个检测器的角点贴得更准(D-07 对比的直接红利)。

真实性承诺(T-05-01):原图只读进内存,标记全部画在副本上,源文件自始至终
不被改写——测试锁定这一点。未检出的照片也留痕:整幅画红边框 + 左上角
ASCII 短标 "MISS"(cv2.putText 不支持中文字形,图内文字只能用 ASCII;
中文说明写在 marked_copies/summary.json 与控制台输出里)。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from calib.corners.artifacts import load_corners
from calib.io.config import BoardSpec, CalibrationConfig

# 两颗检测器的名字(与角点 npz 文件名 corners_<划分>_<检测器>.npz 对应)
DETECTORS = ("classic", "sb")


def render_marked(image_path: Path | str,
                  uv_or_none: np.ndarray | None,
                  cols: int, rows: int,
                  out_path: Path | str,
                  quality: int = 90) -> Path:
    """把一张原图的副本画上角点标记后另存,返回副本路径。

    参数:
        image_path:  原图路径(只读,绝不修改)
        uv_or_none:  角点像素坐标 (cols*rows, 2);None 表示该检测器未检出,
                     改画整幅红边框并在左上角写 ASCII 短标 "MISS" 留痕
        cols / rows: 内角点列数 / 行数(如 11×8 = 88 个角点)
        out_path:    副本输出路径(JPEG)
        quality:     JPEG 质量 1–100(来自 config 的 visualization.jpeg_quality)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f"照片读取失败:{image_path},无法生成标记副本,请人工检查这张图。")

    if uv_or_none is not None:
        # OpenCV 的标准角点叠加:格子交替着色 + 角点连线,一行代码出图;
        # 自己画圆圈既费工又不标准(研究结论:Don't Hand Roll)
        corners = np.asarray(uv_or_none, dtype=np.float32).reshape(-1, 1, 2)
        cv2.drawChessboardCorners(img, (int(cols), int(rows)), corners, True)
    else:
        height, width = img.shape[:2]
        # 边框粗细/字高按画面尺寸比例推导,不写死像素(换分辨率不失效)
        thickness = max(4, min(height, width) // 80)
        cv2.rectangle(img, (0, 0), (width - 1, height - 1),
                      color=(0, 0, 255), thickness=thickness)
        font_scale = max(1.0, min(height, width) / 480)
        cv2.putText(img, "MISS",
                    org=(thickness * 2, thickness * 2 + int(font_scale * 22)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                    color=(0, 0, 255), thickness=max(2, thickness // 2),
                    lineType=cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), img,
                       [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]):
        raise ValueError(f"标记副本写盘失败:{out_path}")
    return out_path


def render_all(corners_run_dir: Path | str,
               manifest_path: Path | str,
               board: BoardSpec,
               cfg: CalibrationConfig) -> dict:
    """对数据清单里的全部照片(train + test 一张不落)生成标记副本(D-06)。

    每颗检测器各一套:marked_copies/<检测器>/<image_id>.jpg。
    角点来自 corners_run_dir 下的 4 份 npz(train/test × classic/sb),
    按 image_id 逐张对应;检出传角点画标准叠加,未检出画 MISS 红框留痕。
    JPEG 质量读 config 的 visualization.jpeg_quality(默认 90,可读性足够
    且磁盘约省一半,见研究 Pitfall 11);重跑 corners 即整目录重新生成,幂等。

    结束后写 marked_copies/summary.json(两套各多少张、多少张 MISS),
    返回同一份 summary 字典供控制台汇总。
    """
    corners_run_dir = Path(corners_run_dir)
    with open(manifest_path, encoding="utf-8") as f:
        images = json.load(f)["images"]

    quality = int(cfg.visualization.jpeg_quality)

    # 每颗检测器:把它两份 npz(train/test)合成 image_id → (角点, 是否检出) 对照表
    tables: dict[str, dict[str, tuple]] = {}
    for detector in DETECTORS:
        lookup: dict[str, tuple] = {}
        for split in ("train", "test"):
            npz_path = corners_run_dir / f"corners_{split}_{detector}.npz"
            if not npz_path.is_file():
                continue  # 某个划分缺文件时按"全部未检出"处理,不让副本中断
            data = load_corners(npz_path)
            for i, image_id in enumerate(data["image_ids"]):
                lookup[str(image_id)] = (data["uv"][i], bool(data["detected"][i]))
        tables[detector] = lookup

    stats: dict[str, dict[str, int]] = {
        d: {"total": 0, "marked": 0, "miss": 0} for d in tables}
    for img in images:
        for detector, lookup in tables.items():
            uv, detected = lookup.get(img["image_id"], (None, False))
            render_marked(
                img["path"], uv if detected else None,
                board.cols, board.rows,
                corners_run_dir / "marked_copies" / detector / f"{img['image_id']}.jpg",
                quality=quality)
            stats[detector]["total"] += 1
            stats[detector]["marked" if detected else "miss"] += 1

    summary = {
        "note": "全量标记副本:每张照片每颗检测器各一份;检出画角点叠加,"
                "未检出画 MISS 红框(只标记不剔除,D-06/D-07)。",
        "jpeg_quality": quality,
        **stats,
    }
    summary_path = corners_run_dir / "marked_copies" / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
