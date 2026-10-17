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
import shutil
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
                  quality: int = 90,
                  note: str | None = None) -> Path:
    """把一张原图的副本画上角点标记后另存,返回副本路径。

    参数:
        image_path:  原图路径(只读,绝不修改)
        uv_or_none:  角点像素坐标 (cols*rows, 2);None 表示该检测器未检出,
                     改画整幅红边框并在左上角写 ASCII 短标 "MISS" 留痕
        cols / rows: 内角点列数 / 行数(如 11×8 = 88 个角点)
        out_path:    副本输出路径(JPEG)
        quality:     JPEG 质量 1–100(来自 config 的 visualization.jpeg_quality)
        note:        可选的红色 ASCII 短标(画在左上角 MISS 短标下方),
                     给"SB 未检出、classic 兜底"这类逐帧证据加一行说明。
                     cv2.putText 不支持中文字形,note 只能是 ASCII;
                     完整中文说明写在 summary.json 与控制台输出里
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

    if note is not None:
        # 附加短标:字号/线宽沿用 MISS 分支那套"按画面尺寸比例推导"的写法,
        # 位置放在 MISS 短标正下方一行(没画 MISS 时同样落左上角区域)
        height, width = img.shape[:2]
        thickness = max(4, min(height, width) // 80)
        font_scale = max(1.0, min(height, width) / 480)
        cv2.putText(img, note,
                    org=(thickness * 2,
                         thickness * 2 + int(font_scale * (22 + 26))),
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


def render_sb_failed_review(corners_run_dir: Path | str,
                            frames: list[dict],
                            board: BoardSpec,
                            quality: int = 90) -> dict | None:
    """SB 检测器本体未检出的帧单独留证据:原图副本 + 标记图 + 中文汇总。

    设计约定"提取效果不好的图像,原图 + 标记图都单独存"的落地产物:
    这些帧混在全量 marked_copies/sb/ 里找起来费劲,单独归拢一份,
    人工逐张核查兜底角点贴得准不准(证据本体,拼图只是索引)。

    参数:
        corners_run_dir: 本轮 corners 运行目录(产物写到其下 review_sb_failed/)
        frames: "SB 本体检出失败"的帧清单,每项含
                image_id(照片编号)、path(原图路径)、
                uv_or_none(classic 兜底角点;两法都未检出时为 None)
        board: 棋盘规格(角点叠加要按列×行画)
        quality: JPEG 质量 1–100

    清单为空直接返回 None 且不建目录(没有失败帧就没有这个文件夹)。
    非空时:original/ 存原图副本(shutil.copy2,保持原文件名与格式,
    源文件只读不改写);marked/ 存标记图(兜底帧画角点叠加 + ASCII 短标
    "SB MISS - CLASSIC FALLBACK",双失败帧画 MISS 红框 + "SB+CLASSIC MISS");
    最后写 summary.json(通俗中文说明 + 计数),并返回同一份字典供控制台汇总。
    """
    if not frames:
        return None

    review_dir = Path(corners_run_dir) / "review_sb_failed"
    originals_dir = review_dir / "original"
    marked_dir = review_dir / "marked"

    n_fallback = 0
    for frame in frames:
        src_path = Path(frame["path"])
        # 原图副本:逐字节复制(保留原文件名与格式),源文件绝不改写
        originals_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, originals_dir / src_path.name)

        uv_or_none = frame.get("uv_or_none")
        if uv_or_none is not None:
            n_fallback += 1
            note = "SB MISS - CLASSIC FALLBACK"  # SB 没检出,classic 兜底
        else:
            note = "SB+CLASSIC MISS"             # 两颗检测器都没检出
        render_marked(src_path, uv_or_none, board.cols, board.rows,
                      marked_dir / f"{frame['image_id']}.jpg",
                      quality=quality, note=note)

    summary = {
        "note": "本文件夹由 calib corners 自动生成,汇总 SB 检测器本体未检出的帧:"
                "标记图是逐帧证据——'SB 未检出、已用经典检测器(classic)兜底'的帧"
                "画兜底角点叠加并在左上角标 'SB MISS - CLASSIC FALLBACK';"
                "经典检测器也未检出的帧画 MISS 红框并标 'SB+CLASSIC MISS'。"
                "original/ 存这些帧的原图副本(逐字节复制,源文件未改动)。",
        "n_total": len(frames),
        "n_classic_fallback": n_fallback,
        "n_both_missed": len(frames) - n_fallback,
        "jpeg_quality": quality,
    }
    with open(review_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
