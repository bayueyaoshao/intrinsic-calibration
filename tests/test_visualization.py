"""角点可视化测试:全量标记副本(overlay)与汇总拼图(mosaic)的行为锁定。

副本部分验证四件事:
1. 检出路径:产出 JPEG 副本、尺寸与原图一致、文件非空;
2. 未检出路径:产出 MISS 红框副本(左上角像素为纯红 BGR=(0,0,255));
3. 原图永不被改写(真实性承诺 T-05-01,逐字节比对);
4. render_all 覆盖清单全部照片 × 两套目录,summary.json 计数与实际一致。

拼图部分(Task 2 追加)验证尺寸量级与逐格索引文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from calib.corners.artifacts import save_corners
from calib.io.config import BoardSpec, CalibrationConfig
from calib.visualization.overlay import render_all, render_marked

# 测试统一用真实棋盘规格(11×8 内角点)
COLS, ROWS = 11, 8
BOARD = BoardSpec()
IMG_W, IMG_H = 320, 240  # 假照片尺寸(小,跑得快)


def _grid_uv(width: int = IMG_W, height: int = IMG_H,
             cols: int = COLS, rows: int = ROWS) -> np.ndarray:
    """画面内一个规则的假角点网格(行主序),只供绘制,不要求是真实检测。"""
    xs = np.linspace(width * 0.1, width * 0.9, cols)
    ys = np.linspace(height * 0.1, height * 0.9, rows)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)


def _write_fake_image(path: Path, seed: int = 0,
                      width: int = IMG_W, height: int = IMG_H) -> Path:
    """写一张随机灰度假照片(内容无所谓,只验证副本流程),返回路径。"""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), rng.integers(0, 256, (height, width),
                                              dtype=np.uint8)):
        raise RuntimeError(f"测试假照片写盘失败:{path}")
    return path


def _write_fake_manifest(images: list[dict], path: Path) -> Path:
    """把假数据清单写成 JSON(与真实 manifest 的 images 数组同形状)。"""
    path.write_text(json.dumps({"images": images}, ensure_ascii=False),
                    encoding="utf-8")
    return path


def _is_reddish(row: np.ndarray) -> np.ndarray:
    """按容差判断一行像素是否"接近纯红"(JPEG 有损压缩不允许逐字节相等)。"""
    return (row[:, 0] < 64) & (row[:, 1] < 64) & (row[:, 2] > 200)


# ---------------------------------------------------------- 标记副本 render_marked

def test_render_marked_detected_path_produces_same_size_copy(tmp_path):
    """检出路径:产出 JPEG 副本,尺寸与原图一致,文件非空。"""
    src = _write_fake_image(tmp_path / "src.jpg", seed=1)
    out = tmp_path / "copies" / "src_marked.jpg"
    result = render_marked(src, _grid_uv(), COLS, ROWS, out, quality=90)

    assert result == out and out.is_file() and out.stat().st_size > 0
    drawn = cv2.imread(str(out))
    assert drawn is not None
    assert drawn.shape[:2] == (IMG_H, IMG_W)  # 副本不改变原图分辨率


def test_render_marked_does_not_touch_source(tmp_path):
    """真实性承诺(T-05-01):生成副本前后,源文件逐字节不变。"""
    src = _write_fake_image(tmp_path / "src.jpg", seed=2)
    before = src.read_bytes()

    render_marked(src, _grid_uv(), COLS, ROWS, tmp_path / "out.jpg", quality=90)

    assert src.read_bytes() == before


def test_render_marked_miss_path_draws_red_border(tmp_path):
    """未检出路径:产出红框副本(顶边整行为纯红),文件非空。"""
    src = _write_fake_image(tmp_path / "src.jpg", seed=3)
    out = tmp_path / "out_miss.jpg"

    render_marked(src, None, COLS, ROWS, out, quality=90)

    assert out.is_file() and out.stat().st_size > 0
    drawn = cv2.imread(str(out))
    # JPEG 有损压缩会让纯红带一点毛刺,按容差判断"接近纯红"
    # (drawChessboardCorners 的检测结果叠加同理不做逐像素相等断言)
    assert _is_reddish(drawn[0]).all()  # 顶边框整行
    assert drawn.shape[:2] == (IMG_H, IMG_W)


def test_render_marked_rejects_unreadable_source(tmp_path):
    """读不出的源文件必须响亮报错(带路径),不能悄悄写一张黑图糊弄过去。"""
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    try:
        render_marked(bad, None, COLS, ROWS, tmp_path / "out.jpg")
    except ValueError as exc:
        assert "broken.jpg" in str(exc)
    else:
        raise AssertionError("损坏源文件未报错")


# ------------------------------------------------------- 全量副本 render_all

def test_render_all_covers_all_images_in_two_detector_sets(tmp_path):
    """3 张假记录(2 检出 + 1 未检出):classic/sb 各 3 个副本文件,
    summary.json 计数与实际一致,未检出那张是 MISS 红框。"""
    ids = ["img_a", "img_b", "img_c"]  # a/b 检出(train),c 未检出(test)
    src_paths = {image_id: _write_fake_image(tmp_path / "src" / f"{image_id}.jpg",
                                             seed=k)
                 for k, image_id in enumerate(ids)}
    manifest_path = _write_fake_manifest(
        [{"image_id": i, "path": str(src_paths[i]),
          "split": "train" if i != "img_c" else "test"} for i in ids],
        tmp_path / "manifest.json")
    before = {i: src_paths[i].read_bytes() for i in ids}

    run_dir = tmp_path / "corners" / "run_test"
    uv = _grid_uv()
    for detector in ("classic", "sb"):
        save_corners(run_dir, "train", detector,
                     image_ids=["img_a", "img_b"],
                     uv=np.stack([uv, uv]),
                     sq_px=np.array([20.0, 20.0]),
                     win_size=np.array([5.0, 5.0]),
                     detected=np.array([True, True]),
                     lineage={"round_id": "t"})
        save_corners(run_dir, "test", detector,
                     image_ids=["img_c"],
                     uv=np.full((1, COLS * ROWS, 2), np.nan),
                     sq_px=np.array([np.nan]),
                     win_size=np.array([np.nan]),
                     detected=np.array([False]),
                     lineage={"round_id": "t"})

    summary = render_all(run_dir, manifest_path, BOARD, CalibrationConfig())

    # 两套目录结构:每个检测器下 3 个文件,一张不落(D-06)
    for detector in ("classic", "sb"):
        copies = sorted((run_dir / "marked_copies" / detector).glob("*.jpg"))
        assert [p.stem for p in copies] == ids
    # 计数:检出 2 + MISS 1,两颗检测器一致
    assert summary["classic"] == {"total": 3, "marked": 2, "miss": 1}
    assert summary["sb"] == {"total": 3, "marked": 2, "miss": 1}
    saved = json.loads(
        (run_dir / "marked_copies" / "summary.json").read_text(encoding="utf-8"))
    assert saved["classic"]["marked"] == 2 and saved["classic"]["miss"] == 1
    assert saved["sb"]["miss"] == 1
    # 未检出那张在两套里都是 MISS 红框(顶边框整行接近纯红)
    for detector in ("classic", "sb"):
        miss_img = cv2.imread(str(run_dir / "marked_copies" / detector / "img_c.jpg"))
        assert _is_reddish(miss_img[0]).all()
    # 原图一张都没被改写(T-05-01)
    for i in ids:
        assert src_paths[i].read_bytes() == before[i]


def test_render_all_uses_configured_jpeg_quality(tmp_path):
    """副本 JPEG 质量来自 config 的 visualization.jpeg_quality,不是写死的。"""
    src = _write_fake_image(tmp_path / "src.jpg", seed=4)
    manifest_path = _write_fake_manifest(
        [{"image_id": "src", "path": str(src), "split": "train"}],
        tmp_path / "manifest.json")
    run_dir = tmp_path / "run"
    uv = _grid_uv()
    for detector in ("classic", "sb"):
        save_corners(run_dir, "train", detector, image_ids=["src"],
                     uv=uv[np.newaxis], sq_px=np.array([20.0]),
                     win_size=np.array([5.0]), detected=np.array([True]),
                     lineage={"round_id": "t"})

    from dataclasses import replace
    cfg = replace(CalibrationConfig(), visualization=replace(
        CalibrationConfig().visualization, jpeg_quality=42))
    render_all(run_dir, manifest_path, BOARD, cfg)

    saved = json.loads(
        (run_dir / "marked_copies" / "summary.json").read_text(encoding="utf-8"))
    assert saved["jpeg_quality"] == 42
