"""角点可视化测试:全量标记副本(overlay)与汇总拼图(mosaic)的行为锁定。

副本部分验证四件事:
1. 检出路径:产出 JPEG 副本、尺寸与原图一致、文件非空;
2. 未检出路径:产出 MISS 红框副本(左上角像素为纯红 BGR=(0,0,255));
3. 原图永不被改写(真实性承诺 T-05-01,逐字节比对);
4. render_all 覆盖清单全部照片 × 两套目录,summary.json 计数与实际一致。

拼图部分验证:缩略拼贴尺寸量级正确;总览抽样规则(存疑全收 + 每姿态抽样
+ 最差若干)并集进逐格索引;零存疑时不产存疑拼图、索引如实记 0 格。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from calib.corners.artifacts import save_corners
from calib.io.config import BoardSpec, CalibrationConfig
from calib.visualization.mosaic import build_mosaic, write_mosaics
from calib.visualization.overlay import (
    render_all,
    render_flagged_review,
    render_marked,
    render_sb_failed_review,
)

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


def test_render_marked_with_ascii_note(tmp_path):
    """可选 ASCII 短标(note):副本正常生成、尺寸与原图一致、源文件不动。

    note 用于 SB 失败帧的逐帧证据(如 "SB MISS - CLASSIC FALLBACK");
    默认不传 note 时走的还是原来的路径(由上方既有 render_marked 测试锁定)。
    """
    src = _write_fake_image(tmp_path / "src.jpg", seed=5)
    before = src.read_bytes()
    out = tmp_path / "out_note.jpg"

    result = render_marked(src, _grid_uv(), COLS, ROWS, out, quality=90,
                           note="SB MISS - CLASSIC FALLBACK")

    assert result == out and out.is_file() and out.stat().st_size > 0
    drawn = cv2.imread(str(out))
    assert drawn is not None
    assert drawn.shape[:2] == (IMG_H, IMG_W)  # 副本不改变原图分辨率
    assert src.read_bytes() == before          # 源文件自始至终不被改写


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

    cfg = replace(CalibrationConfig(), visualization=replace(
        CalibrationConfig().visualization, jpeg_quality=42))
    render_all(run_dir, manifest_path, BOARD, cfg)

    saved = json.loads(
        (run_dir / "marked_copies" / "summary.json").read_text(encoding="utf-8"))
    assert saved["jpeg_quality"] == 42


# ------------------------------------------- SB 失败帧独立产物 render_sb_failed_review

def test_render_sb_failed_review_two_frames(tmp_path, make_chessboard_image):
    """2 帧失败清单(1 帧 classic 兜底带角点 + 1 帧两法都未检出):

    original/ 存 2 张原图副本(保持原文件名,逐字节一致);marked/ 存 2 张
    标记图(兜底帧 = 角点叠加 + ASCII 短标,双失败帧 = MISS 红框);
    summary.json 计数正确且含通俗中文说明;源文件一张都不被改写。
    """
    # 兜底帧:合成棋盘当原图,夹具自带的真值角点当 classic 兜底角点
    board_img, truth_uv, _sq = make_chessboard_image()
    fb_src = tmp_path / "src" / "fb_frame.jpg"
    fb_src.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(fb_src), board_img):
        raise RuntimeError("测试夹具写棋盘原图失败")
    # 双失败帧:随机噪声图
    miss_src = _write_fake_image(tmp_path / "src" / "miss_frame.jpg", seed=6)
    before = {p.name: p.read_bytes() for p in (fb_src, miss_src)}

    run_dir = tmp_path / "corners" / "run_t"
    frames = [
        {"image_id": "fb_frame", "path": str(fb_src), "uv_or_none": truth_uv},
        {"image_id": "miss_frame", "path": str(miss_src), "uv_or_none": None},
    ]
    summary = render_sb_failed_review(run_dir, frames, BOARD, quality=90)

    review = run_dir / "review_sb_failed"
    # 原图副本:保持原文件名与格式,内容逐字节一致(只读复制)
    assert sorted(p.name for p in (review / "original").iterdir()) == \
        ["fb_frame.jpg", "miss_frame.jpg"]
    assert (review / "original" / "fb_frame.jpg").read_bytes() == before["fb_frame.jpg"]
    assert (review / "original" / "miss_frame.jpg").read_bytes() == before["miss_frame.jpg"]
    # 标记图:两张都在,尺寸与各自原图一致
    fb_marked = cv2.imread(str(review / "marked" / "fb_frame.jpg"))
    miss_marked = cv2.imread(str(review / "marked" / "miss_frame.jpg"))
    assert fb_marked is not None and fb_marked.shape[:2] == board_img.shape[:2]
    assert miss_marked is not None and miss_marked.shape[:2] == (IMG_H, IMG_W)
    # 双失败帧 = MISS 红框(顶边整行接近纯红);兜底帧 = 角点叠加(非红框)
    assert _is_reddish(miss_marked[0]).all()
    assert not _is_reddish(fb_marked[0]).all()
    # summary 计数:总数 2、classic 兜底 1、两法都未检出 1,说明是通俗中文
    assert summary["n_total"] == 2
    assert summary["n_classic_fallback"] == 1 and summary["n_both_missed"] == 1
    saved = json.loads((review / "summary.json").read_text(encoding="utf-8"))
    assert saved["n_total"] == 2 and saved["n_classic_fallback"] == 1
    assert "兜底" in saved["note"] and "calib corners" in saved["note"]
    # 源文件自始至终不被改写(真实性承诺 T-05-01)
    assert fb_src.read_bytes() == before["fb_frame.jpg"]
    assert miss_src.read_bytes() == before["miss_frame.jpg"]


def test_render_sb_failed_review_empty_list_creates_nothing(tmp_path):
    """空失败清单:返回 None,且不创建 review_sb_failed 目录(没有失败帧不建目录)。"""
    run_dir = tmp_path / "corners" / "run_t"

    result = render_sb_failed_review(run_dir, [], BOARD, quality=90)

    assert result is None
    assert not (run_dir / "review_sb_failed").exists()


# ------------------------------------- 通用复核文件夹 render_flagged_review
#
# review_sb_failed 只管"SB 未检出"一类帧;存疑帧/连拍点名帧需要同一套
# 文件夹结构但原因各不相同,所以抽出通用函数:文件夹名、中文说明、
# 附加计数、逐帧 info 全部由调用方传,函数本身只负责"归拢成一个文件夹"。

def test_render_flagged_review_two_frames(tmp_path, make_chessboard_image):
    """2 帧复核清单(1 帧两颗检测器都检出 + 1 帧两颗都未检出):

    original/ 存 2 张原图副本(保持原文件名,逐字节一致);marked_classic/
    与 marked_sb/ 各 2 张标记图(未检出帧 = MISS 红框,检出帧 = 角点叠加);
    summary.json:n_total 与按检测器的 marked/miss 计数正确、frames 数组
    每条含 image_id 加调用方传入的 info(中文值原样落盘)、顶层含调用方
    传入的 note 中文说明与 extra 计数字段;源文件逐字节不变。
    """
    # 检出帧:合成棋盘当原图,夹具自带的真值角点当两颗检测器的角点
    board_img, truth_uv, _sq = make_chessboard_image()
    ok_src = tmp_path / "src" / "ok_frame.jpg"
    ok_src.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(ok_src), board_img):
        raise RuntimeError("测试夹具写棋盘原图失败")
    # 未检出帧:随机噪声图(两颗检测器都没有角点)
    miss_src = _write_fake_image(tmp_path / "src" / "miss_frame.jpg", seed=7)
    before = {p.name: p.read_bytes() for p in (ok_src, miss_src)}

    run_dir = tmp_path / "corners" / "run_t"
    frames = [
        {"image_id": "ok_frame", "path": str(ok_src),
         "uv_classic": truth_uv, "uv_sb": truth_uv, "note": "SUS EXPO",
         "info": {"pose_id": "poseA", "repeat_idx": 0, "split": "train",
                  "reason": "存疑原因:同姿态相对比较:曝光"}},
        {"image_id": "miss_frame", "path": str(miss_src),
         "uv_classic": None, "uv_sb": None, "note": "BURST FLAG",
         "info": {"pose_id": "poseA", "repeat_idx": 1, "split": "train",
                  "reason": "连拍断言失败:组内波动中位数 4.00px 超限"}},
    ]
    summary = render_flagged_review(
        run_dir, "review_suspects", frames, BOARD, quality=90,
        note="本文件夹由 calib corners 自动生成,汇总需人工复核的帧。",
        extra={"pose_relative": 1, "global": 1, "suspect_total": 2})

    review = run_dir / "review_suspects"
    # 原图副本:保持原文件名与格式,内容逐字节一致(只读复制)
    assert sorted(p.name for p in (review / "original").iterdir()) == \
        ["miss_frame.jpg", "ok_frame.jpg"]
    assert (review / "original" / "ok_frame.jpg").read_bytes() == before["ok_frame.jpg"]
    assert (review / "original" / "miss_frame.jpg").read_bytes() == before["miss_frame.jpg"]
    # 两套标记图:每颗检测器各 2 张,尺寸与各自原图一致
    for detector in ("classic", "sb"):
        marked = {}
        for name in ("ok_frame", "miss_frame"):
            img = cv2.imread(str(review / f"marked_{detector}" / f"{name}.jpg"))
            assert img is not None and img.size > 0
            marked[name] = img
        assert marked["ok_frame"].shape[:2] == board_img.shape[:2]
        assert marked["miss_frame"].shape[:2] == (IMG_H, IMG_W)
        # 未检出帧 = MISS 红框(顶边整行接近纯红);检出帧 = 角点叠加(非红框)
        assert _is_reddish(marked["miss_frame"][0]).all()
        assert not _is_reddish(marked["ok_frame"][0]).all()
    # summary:计数、调用方传入的 note 与 extra、逐帧 info 全部对上
    assert summary["n_total"] == 2
    assert summary["marked"] == {"classic": 1, "sb": 1}
    assert summary["miss"] == {"classic": 1, "sb": 1}
    assert summary["pose_relative"] == 1 and summary["suspect_total"] == 2
    assert "人工复核" in summary["note"] and "calib corners" in summary["note"]
    assert summary["jpeg_quality"] == 90
    by_id = {f["image_id"]: f for f in summary["frames"]}
    assert len(summary["frames"]) == 2
    assert by_id["ok_frame"]["reason"] == "存疑原因:同姿态相对比较:曝光"
    assert by_id["miss_frame"]["pose_id"] == "poseA"
    assert by_id["miss_frame"]["repeat_idx"] == 1
    assert by_id["ok_frame"]["split"] == "train"
    # 落盘的 summary.json 与返回值是同一份内容(键名英文、中文值原样)
    saved = json.loads((review / "summary.json").read_text(encoding="utf-8"))
    assert saved["n_total"] == 2
    assert saved["marked"] == {"classic": 1, "sb": 1}
    assert saved["frames"][0]["reason"].startswith("存疑原因")
    # 源文件自始至终不被改写(真实性承诺 T-05-01)
    assert ok_src.read_bytes() == before["ok_frame.jpg"]
    assert miss_src.read_bytes() == before["miss_frame.jpg"]


def test_render_flagged_review_empty_list_creates_nothing(tmp_path):
    """空清单:返回 None,且不创建以 folder 参数命名的目录(没帧就没文件夹)。"""
    run_dir = tmp_path / "corners" / "run_t"

    result = render_flagged_review(run_dir, "review_suspects", [], BOARD,
                                   quality=90, note="说明", extra={"n": 0})

    assert result is None
    assert not (run_dir / "review_suspects").exists()


# ---------------------------------------------------------------- 汇总拼图 mosaic

def test_build_mosaic_size_and_file(tmp_path):
    """3 张假图 × 每行 2 格:PNG 宽在 [cols×thumb, cols×(thumb+40)) 量级,
    高 > 0,文件非空(精确值以标注条实现为准,只锁量级)。"""
    items = [(_write_fake_image(tmp_path / f"m{k}.jpg", seed=10 + k),
              f"pose{k:04d}#0 OK") for k in range(3)]
    out = tmp_path / "mosaic.png"

    build_mosaic(items, thumb_width=64, cols=2, out_path=out)

    assert out.is_file() and out.stat().st_size > 0
    img = cv2.imread(str(out))
    assert img is not None
    assert 2 * 64 <= img.shape[1] < 2 * (64 + 40)
    assert img.shape[0] > 0


def _mosaic_fixtures(tmp_path: Path):
    """假数据场景:poseA 两张(img1 正常 / img2 存疑)+ poseB 一张(img3 正常)
    + test 一张(img4 正常),全部检出,存疑原图副本就位。返回 (run_dir, 清单路径)。"""
    paths = {f"img{i}": _write_fake_image(tmp_path / "src" / f"img{i}.jpg",
                                          seed=20 + i, width=160, height=120)
             for i in range(1, 5)}
    images = [
        {"image_id": "img1", "path": str(paths["img1"]), "split": "train",
         "pose_id": "poseA", "repeat_idx": 0, "n_repeats": 2,
         "precheck": {"verdict": "ok", "blur_norm": 1.0, "expo_hi": 0.0,
                      "expo_lo": 0.0, "suspect_sources": []}},
        {"image_id": "img2", "path": str(paths["img2"]), "split": "train",
         "pose_id": "poseA", "repeat_idx": 1, "n_repeats": 2,
         "precheck": {"verdict": "suspect", "blur_norm": 0.1, "expo_hi": 0.3,
                      "expo_lo": 0.0, "suspect_sources": ["pose_relative:blur"]}},
        {"image_id": "img3", "path": str(paths["img3"]), "split": "train",
         "pose_id": "poseB", "repeat_idx": 0, "n_repeats": 1,
         "precheck": {"verdict": "ok", "blur_norm": 1.2, "expo_hi": 0.0,
                      "expo_lo": 0.0, "suspect_sources": []}},
        {"image_id": "img4", "path": str(paths["img4"]), "split": "test",
         "pose_id": "img4", "repeat_idx": 0, "n_repeats": 1,
         "precheck": {"verdict": "ok", "blur_norm": 0.9, "expo_hi": 0.0,
                      "expo_lo": 0.0, "suspect_sources": []}},
    ]
    manifest_path = _write_fake_manifest(images, tmp_path / "manifest.json")

    run_dir = tmp_path / "corners" / "run_t"
    uv = _grid_uv(160, 120)
    for detector in ("classic", "sb"):
        save_corners(run_dir, "train", detector, image_ids=["img1", "img2", "img3"],
                     uv=np.stack([uv, uv, uv]),
                     sq_px=np.full(3, 15.0), win_size=np.full(3, 4.0),
                     detected=np.array([True, True, True]),
                     lineage={"round_id": "t"})
        save_corners(run_dir, "test", detector, image_ids=["img4"],
                     uv=uv[np.newaxis], sq_px=np.array([15.0]),
                     win_size=np.array([4.0]), detected=np.array([True]),
                     lineage={"round_id": "t"})
    # 存疑原图副本(suspect 拼图优先看这里的未标记原貌)
    _write_fake_image(run_dir / "suspect_originals" / "img2.jpg", seed=99,
                      width=160, height=120)
    return run_dir, manifest_path


def test_write_mosaics_overview_sampling_and_index(tmp_path):
    """总览抽样(存疑全收 + 每姿态 1 张,worst_n=0 关掉最差通道):
    并集 = {img1, img2, img3},干净的 test 单张不混入;
    存疑拼图只含 img2;逐格索引含全部标签与指标值。"""
    run_dir, manifest_path = _mosaic_fixtures(tmp_path)
    cfg = replace(CalibrationConfig(), visualization=replace(
        CalibrationConfig().visualization,
        mosaic=replace(CalibrationConfig().visualization.mosaic, worst_n=0)))

    info = write_mosaics(run_dir, manifest_path, cfg)

    assert info["overview_cells"] == 3 and info["suspect_cells"] == 1
    index = json.loads(
        (run_dir / "mosaic" / "mosaic_index.json").read_text(encoding="utf-8"))
    overview_ids = [c["image_id"] for c in index["overview"]["cells"]]
    assert sorted(overview_ids) == ["img1", "img2", "img3"]
    suspect_ids = [c["image_id"] for c in index["suspect"]["cells"]]
    assert suspect_ids == ["img2"]
    # 标签是 ASCII 缩写:存疑格 SUS、正常格 OK;索引带指标值可排序
    by_id = {c["image_id"]: c for c in index["overview"]["cells"]}
    assert by_id["img2"]["label"].endswith("SUS")
    assert by_id["img1"]["label"].endswith("OK")
    assert by_id["img2"]["blur_norm"] == 0.1
    assert index["overview"]["n_cells"] == 3
    # 两张拼图都在盘上且非空
    for name in ("overview_mosaic.png", "suspect_mosaic.png"):
        p = run_dir / "mosaic" / name
        assert p.is_file() and p.stat().st_size > 0


def test_write_mosaics_zero_suspect_writes_no_suspect_mosaic(tmp_path):
    """零存疑数据:总览照常产出,存疑拼图不产文件、索引如实记 0 格(不崩)。"""
    run_dir, manifest_path = _mosaic_fixtures(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for img in manifest["images"]:  # 全部改判正常
        img["precheck"]["verdict"] = "ok"
        img["precheck"]["suspect_sources"] = []
    manifest_path = _write_fake_manifest(manifest["images"], manifest_path)

    info = write_mosaics(run_dir, manifest_path, CalibrationConfig())

    assert info["overview_cells"] > 0
    assert info["suspect_cells"] == 0 and info["suspect_path"] is None
    assert not (run_dir / "mosaic" / "suspect_mosaic.png").exists()
    index = json.loads(
        (run_dir / "mosaic" / "mosaic_index.json").read_text(encoding="utf-8"))
    assert index["suspect"]["n_cells"] == 0


def test_overview_fills_all_when_worst_n_covers_everything(tmp_path):
    """默认 worst_n=20 覆盖全部 4 张假照片时,总览 = 全体(一个不落)。"""
    run_dir, manifest_path = _mosaic_fixtures(tmp_path)

    info = write_mosaics(run_dir, manifest_path, CalibrationConfig())

    assert info["overview_cells"] == 4
    index = json.loads(
        (run_dir / "mosaic" / "mosaic_index.json").read_text(encoding="utf-8"))
    assert sorted(c["image_id"] for c in index["overview"]["cells"]) == \
        ["img1", "img2", "img3", "img4"]
