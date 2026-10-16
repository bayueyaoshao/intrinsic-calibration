"""汇总拼图(D-06):把几十到几百张照片缩略拼成一张大图,人工一眼扫全局。

定位必须说清楚:**拼图只是索引,不是全量标记副本的替代**(D-06)。
缩略图看不清角点细节,逐张核查永远回到 marked_copies/ 的单张副本;
拼图回答的是"整体长什么样、该先看哪几张"。

两个入口(corners 命令收尾自动调用,都落在 corners run 目录 mosaic/ 子目录):
- 总览拼图 overview_mosaic.png:每姿态第 1 张 + 模糊度最差 N 张 + 全部存疑,
  覆盖"常态、最糟、可疑"三种画面;
- 存疑拼图 suspect_mosaic.png:只拼体检判为存疑的照片,并且看的是
  suspect_originals/ 里的**未加标记原貌**(与角点标记副本互相印证)。

逐格清单 mosaic_index.json 记录格号 → 照片编号/标签/全部指标值——
拼图上认出哪格不对劲,按格号来这里查编号,再去单张副本放大细看。

图内标注只用 ASCII(cv2.putText 不支持中文;pose_id 截断 8 字符 +
连拍序号 + 判定缩写 OK/SUS/MISS),中文说明都在索引文件与控制台。
不引入任何新依赖:缩放用 cv2.resize、拼贴用 numpy 数组拼接。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from calib.corners.artifacts import load_corners
from calib.io.config import CalibrationConfig, MosaicConfig

# 总览拼图的格数上限:超出时按优先级(存疑 > 最差 > 姿态抽样)截断,
# 还超(极端数据存疑极多)就均匀抽样。当前数据(80 姿态 + 89 存疑)远够不着。
MAX_OVERVIEW_CELLS = 200


def _detected_map(corners_run_dir: Path, detector: str = "classic") -> dict[str, bool]:
    """从角点 npz 读"每张照片该检测器有没有检出",按 image_id 查。

    用经典检测器的检出状态给拼图标签定 MISS(经典是默认标定路线;
    两颗检测器各自的完整检出信息在 detection_summary.json 与索引里)。
    """
    out: dict[str, bool] = {}
    for split in ("train", "test"):
        npz_path = corners_run_dir / f"corners_{split}_{detector}.npz"
        if not npz_path.is_file():
            continue
        data = load_corners(npz_path)
        for i, image_id in enumerate(data["image_ids"]):
            out[str(image_id)] = bool(data["detected"][i])
    return out


def _verdict_abbreviation(img: dict, detected_classic: bool) -> str:
    """判定缩写:未检出 MISS 最要紧 > 存疑 SUS > 正常 OK。"""
    if not detected_classic:
        return "MISS"
    if img.get("precheck", {}).get("verdict") == "suspect":
        return "SUS"
    return "OK"


def _cell_label(img: dict, detected_classic: bool) -> str:
    """一格的 ASCII 标签:pose_id 截 8 字符 + 连拍序号 + 判定缩写。"""
    pose = str(img.get("pose_id") or "")[:8]
    return f"{pose}#{img.get('repeat_idx', 0)} {_verdict_abbreviation(img, detected_classic)}"


def _cell_record(cell: int, img: dict, label: str, detected_classic: bool) -> dict:
    """一格的索引记录:格号 + 照片编号 + 标签 + 全部体检数字(指标值可排序对比)。"""
    pre = img.get("precheck", {})
    return {
        "cell": cell,
        "image_id": img["image_id"],
        "label": label,
        "split": img.get("split"),
        "pose_id": img.get("pose_id"),
        "repeat_idx": img.get("repeat_idx"),
        "verdict": pre.get("verdict", "ok"),
        "suspect_sources": list(pre.get("suspect_sources", [])),
        "blur_norm": pre.get("blur_norm"),
        "expo_hi": pre.get("expo_hi"),
        "expo_lo": pre.get("expo_lo"),
        "detected_classic": detected_classic,
    }


def build_mosaic(items: list[tuple[Path | str, str]],
                 thumb_width: int, cols: int,
                 out_path: Path | str) -> Path:
    """把 (图片路径, ASCII 标签) 清单拼成一张大图 PNG,返回图片路径。

    每格:上方一条深色标注条(写标签),下方是等比缩放到 thumb_width 宽的
    缩略图;按 cols 列从左到右、从上到下排布,末行空位补黑块。
    读不出的图按纯黑格子处理,不让整张拼图中断(坏图的格子本身就是信息)。
    """
    thumb_width, cols = int(thumb_width), int(cols)
    if not items:
        raise ValueError("拼图清单为空:没有可拼的照片,请先检查上游产物。")

    label_h = max(24, thumb_width // 12)          # 标注条高度随缩略宽比例走
    font_scale = max(0.45, label_h / 34.0)
    char_budget = max(8, int(thumb_width / (font_scale * 15)))  # 一格装得下的字符数

    cells: list[np.ndarray] = []
    for image_path, label in items:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((thumb_width, thumb_width, 3), dtype=np.uint8)
        height, width = img.shape[:2]
        thumb_h = max(1, round(height * thumb_width / width))
        thumb = cv2.resize(img, (thumb_width, thumb_h),
                           interpolation=cv2.INTER_AREA)
        band = np.full((label_h, thumb_width, 3), 40, dtype=np.uint8)
        cv2.putText(band, label[:char_budget], org=(4, label_h - 7),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                    color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cells.append(np.vstack([band, thumb]))

    cell_h = max(cell.shape[0] for cell in cells)   # 各格高度取最大,矮的补黑
    n_rows = math.ceil(len(cells) / cols)
    canvas = np.zeros((n_rows * cell_h, cols * thumb_width, 3), dtype=np.uint8)
    for k, cell in enumerate(cells):
        r, c = divmod(k, cols)
        canvas[r * cell_h:(r + 1) * cell_h,
               c * thumb_width:(c + 1) * thumb_width][:cell.shape[0]] = cell

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise ValueError(f"拼图写盘失败:{out_path}")
    return out_path


def _select_overview(images: list[dict], mcfg: MosaicConfig) -> list[dict]:
    """总览拼图的抽样规则(保持清单原顺序,去重):
    1. 全部存疑(人工回环最优先看的);
    2. 每姿态前 per_pose_sample 张(train 全部姿态覆盖一遍常态);
    3. 模糊度归一值最小的 worst_n 张(值越小越糊,全数据排名)。
    超过 MAX_OVERVIEW_CELLS:按上面优先级保留,再把姿态抽样尾部均匀抽样截到上限。
    """
    picked: list[dict] = []
    seen: set[str] = set()

    def add(img: dict) -> None:
        if img["image_id"] not in seen:
            seen.add(img["image_id"])
            picked.append(img)

    suspects = [img for img in images
                if img.get("precheck", {}).get("verdict") == "suspect"]
    for img in suspects:
        add(img)

    per_pose: dict[str, int] = {}
    for img in images:
        if img.get("split") != "train":
            continue
        pose = img.get("pose_id", "")
        if per_pose.get(pose, 0) < mcfg.per_pose_sample:
            per_pose[pose] = per_pose.get(pose, 0) + 1
            add(img)

    def blur_key(img: dict) -> float:
        value = (img.get("precheck", {}) or {}).get("blur_norm")
        return float("inf") if value is None else float(value)

    ranked = sorted(images, key=blur_key)  # 缺失值排最后(无穷大)
    for img in ranked[:mcfg.worst_n]:
        add(img)

    if len(picked) > MAX_OVERVIEW_CELLS:
        # 超上限:存疑全部保留,存疑之外的部分在原顺序里均匀抽样截到上限
        suspect_ids = {img["image_id"] for img in suspects}
        rest = [img for img in picked if img["image_id"] not in suspect_ids]
        budget = max(0, MAX_OVERVIEW_CELLS - len(suspects))
        if len(rest) > budget > 0:
            idx = np.unique(np.linspace(0, len(rest) - 1, budget)
                            .round().astype(int))
            rest = [rest[i] for i in idx]
        picked = suspects + rest
    return picked


def _as_manifest(manifest: dict | Path | str) -> dict:
    """清单参数兼容两种给法:已读进来的字典,或 manifest.json 的路径。"""
    if isinstance(manifest, (str, Path)):
        with open(manifest, encoding="utf-8") as f:
            return json.load(f)
    return manifest


def overview_mosaic(corners_run_dir: Path | str,
                    manifest: dict,
                    cfg: CalibrationConfig) -> dict:
    """总览拼图:常态(每姿态抽样)+ 最差若干 + 全部存疑,写 overview_mosaic.png。

    返回 {"path": 图片路径或 None, "n_cells": 格数, "cells": 逐格清单}。
    """
    corners_run_dir = Path(corners_run_dir)
    images: list[dict] = _as_manifest(manifest)["images"]
    mcfg = cfg.visualization.mosaic
    detected = _detected_map(corners_run_dir, "classic")

    selected = _select_overview(images, mcfg)
    items = [(img["path"], _cell_label(img, detected.get(img["image_id"], False)))
             for img in selected]
    cells = [_cell_record(k, img, label, detected.get(img["image_id"], False))
             for k, (img, label) in enumerate(zip(selected, [l for _, l in items]))]
    out_path = corners_run_dir / "mosaic" / "overview_mosaic.png"
    if items:
        build_mosaic(items, mcfg.thumb_width, mcfg.cols, out_path)
        return {"path": str(out_path), "n_cells": len(cells), "cells": cells}
    return {"path": None, "n_cells": 0, "cells": []}


def suspect_mosaic(corners_run_dir: Path | str,
                   manifest: dict,
                   cfg: CalibrationConfig) -> dict:
    """存疑拼图:只拼体检存疑的照片,看 suspect_originals/ 的未标记原貌。

    写 suspect_mosaic.png;没有存疑照片时不产出文件(返回 n_cells=0),
    mosaic_index.json 里如实记"无存疑"。
    """
    corners_run_dir = Path(corners_run_dir)
    images: list[dict] = _as_manifest(manifest)["images"]
    mcfg = cfg.visualization.mosaic
    detected = _detected_map(corners_run_dir, "classic")

    suspects = [img for img in images
                if img.get("precheck", {}).get("verdict") == "suspect"]
    items, cells = [], []
    for img in suspects:
        # 优先用存疑原图副本(未加标记的原貌,拼图缩略图看不清细节才存的);
        # 副本不在时退回原始路径,不让拼图中断
        original_copy = corners_run_dir / "suspect_originals" / f"{img['image_id']}.jpg"
        image_path = original_copy if original_copy.is_file() else img["path"]
        label = _cell_label(img, detected.get(img["image_id"], False))
        items.append((image_path, label))
        cells.append(_cell_record(len(cells), img, label,
                                  detected.get(img["image_id"], False)))
    out_path = corners_run_dir / "mosaic" / "suspect_mosaic.png"
    if items:
        build_mosaic(items, mcfg.thumb_width, mcfg.cols, out_path)
        return {"path": str(out_path), "n_cells": len(cells), "cells": cells}
    return {"path": None, "n_cells": 0, "cells": []}


def write_mosaics(corners_run_dir: Path | str,
                  manifest: dict,
                  cfg: CalibrationConfig) -> dict:
    """两个拼图入口 + 逐格索引一次写齐(corners 命令的收尾调用)。

    写 mosaic/mosaic_index.json(格号 → image_id/标签/指标值)并返回
    控制台汇总用的一行信息。拼图只是索引,不是全量副本的替代(D-06)。
    """
    corners_run_dir = Path(corners_run_dir)
    overview = overview_mosaic(corners_run_dir, manifest, cfg)
    suspect = suspect_mosaic(corners_run_dir, manifest, cfg)

    index = {
        "note": "拼图只是索引,不是替代(D-06):在这里认格子,去 marked_copies/"
                "看单张标记副本,存疑照片的未标记原貌在 suspect_originals/。"
                "标签为 ASCII(pose_id 前 8 字符#连拍序号 判定缩写),"
                "判定缩写:OK=正常,SUS=体检存疑,MISS=经典检测器未检出。",
        "overview": {"n_cells": overview["n_cells"], "path": overview["path"],
                     "cells": overview["cells"]},
        "suspect": {"n_cells": suspect["n_cells"], "path": suspect["path"],
                    "cells": suspect["cells"]},
    }
    index_path = corners_run_dir / "mosaic" / "mosaic_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return {
        "index_path": index_path,
        "overview_path": overview["path"],
        "overview_cells": overview["n_cells"],
        "suspect_path": suspect["path"],
        "suspect_cells": suspect["n_cells"],
    }
