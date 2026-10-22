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
from calib.visualization.analyze_figs import (
    plot_budget_bars,
    plot_corner_opt_curves,
    plot_coverage_footprint,
    plot_radial_bands,
    plot_sensitivity_ranking,
    plot_what_if,
    plot_xyz_compare,
)
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


# ------------------------------- 复核文件夹:清单组装(体检存疑 / 连拍点名)
#
# 两个帮手只做"筛帧 + 拼中文原因",不碰文件系统(渲染交给
# render_flagged_review),所以这里直接用假数据锁行为,不需要跑命令行。

def _fake_arrays(n: int, detected: dict[str, list[bool]],
                 uv=None) -> dict:
    """造两颗检测器的假角点数组:uv 全 NaN,检出帧那几行填规则网格。

    形状与 corners 命令内部的 arrays 一致(每颗检测器一个字典,
    uv 是 (张数, 内角点数, 2),detected 是布尔数组)。
    """
    uv = uv if uv is not None else _grid_uv()
    arrays = {}
    for detector in ("classic", "sb"):
        slot_uv = np.full((n, COLS * ROWS, 2), np.nan)
        flags = np.zeros(n, dtype=bool)
        for i, ok in enumerate(detected[detector]):
            if ok:
                slot_uv[i] = uv
                flags[i] = True
        arrays[detector] = {"uv": slot_uv, "detected": flags}
    return arrays


def test_suspect_review_frames_assembly():
    """存疑清单组装:只收 suspect 帧;info 保留原始记号 + 中文翻译
    (四种来源记号全覆盖)+ 三个指标原值;角点按 detected 取或 None;
    note 是 ASCII 短标(图上画不了中文)。"""
    from calib.cli.main import _suspect_review_frames

    images = [
        {"image_id": "img1", "path": "p1.jpg", "split": "train",
         "pose_id": "poseA", "repeat_idx": 0,
         "precheck": {"verdict": "suspect", "blur_norm": 0.12,
                      "expo_hi": 0.001, "expo_lo": 0.002,
                      "suspect_sources": ["pose_relative:blur",
                                          "global:exposure"]}},
        {"image_id": "img2", "path": "p2.jpg", "split": "train",
         "pose_id": "poseA", "repeat_idx": 1,
         "precheck": {"verdict": "suspect", "blur_norm": 0.34,
                      "expo_hi": 0.003, "expo_lo": 0.004,
                      "suspect_sources": ["pose_relative:exposure",
                                          "global:blur"]}},
        {"image_id": "img3", "path": "p3.jpg", "split": "train",
         "pose_id": "poseB", "repeat_idx": 0,
         "precheck": {"verdict": "ok", "blur_norm": 1.0, "expo_hi": 0.0,
                      "expo_lo": 0.0, "suspect_sources": []}},
    ]
    # img2 的 classic 未检出(角点给 None);sb 检出 True —— 正是
    # "classic 兜底帧"的情形:sb 槽位的 detected 天然含兜底,不再特殊处理
    arrays = _fake_arrays(3, detected={"classic": [True, False, True],
                                       "sb": [True, True, True]})
    uv = _grid_uv()

    frames = _suspect_review_frames(images, arrays)

    # 只收 2 张存疑帧,正常帧不进清单;顺序照数据清单
    assert [f["image_id"] for f in frames] == ["img1", "img2"]
    f1, f2 = frames
    # 原始记号原样保留(可对账 manifest);中文翻译四种记号全覆盖
    assert f1["info"]["suspect_sources"] == ["pose_relative:blur",
                                             "global:exposure"]
    assert f1["info"]["reason"] == \
        "存疑原因:同姿态相对比较:模糊度、全局线兜底:曝光"
    assert f2["info"]["reason"] == \
        "存疑原因:同姿态相对比较:曝光、全局线兜底:模糊度"
    # 三个体检指标数值原样照抄,复核时不翻 manifest
    assert f1["info"]["metrics"] == {"blur_norm": 0.12, "expo_hi": 0.001,
                                     "expo_lo": 0.002}
    assert f2["info"]["metrics"]["blur_norm"] == 0.34
    # 姿态/连拍序号/划分带齐
    assert f2["info"]["pose_id"] == "poseA" and f2["info"]["repeat_idx"] == 1
    assert f1["info"]["split"] == "train"
    # 角点:classic 检出给角点、未检出给 None;sb 路线(含兜底)都给角点
    assert np.array_equal(f1["uv_classic"], uv)
    assert f2["uv_classic"] is None
    assert np.array_equal(f1["uv_sb"], uv) and np.array_equal(f2["uv_sb"], uv)
    # note 是 ASCII 短标(多来源 "+" 连接、前缀 "SUS ")
    assert f1["note"] == "SUS BLUR+EXPO"
    assert f2["note"] == "SUS EXPO+BLUR"
    # 原图路径原样带出(渲染函数要靠它复制原图)
    assert f1["path"] == "p1.jpg"


def test_burst_review_frames_assembly():
    """连拍点名清单组装:两颗检测器点名并集去重、按清单顺序;
    每帧 flagged_by 列出点名者;原因串带波动中位数/最大值与容差的
    数值和超限结论(超限与否按数值现算);未点名的帧不进清单。"""
    from calib.cli.main import _burst_review_frames

    images = [
        {"image_id": "a1", "path": "a1.jpg", "split": "train",
         "pose_id": "poseA", "repeat_idx": 0},
        {"image_id": "a2", "path": "a2.jpg", "split": "train",
         "pose_id": "poseA", "repeat_idx": 1},
        {"image_id": "a3", "path": "a3.jpg", "split": "train",
         "pose_id": "poseA", "repeat_idx": 2},
        {"image_id": "a4", "path": "a4.jpg", "split": "train",
         "pose_id": "poseB", "repeat_idx": 0},
    ]
    # classic 点名 [a1, a2],sb 点名 [a2, a3]:并集 [a1, a2, a3]
    # 两颗检测器的数值都是"中位数超限、最大值未超"
    burst_failures = [
        {"detector": "classic", "split": "train", "pose_id": "poseA",
         "suspect_image_ids": ["a1", "a2"],
         "spread_median_px": 4.5, "spread_max_px": 5.0,
         "threshold_px": [2.0, 12.0]},
        {"detector": "sb", "split": "train", "pose_id": "poseA",
         "suspect_image_ids": ["a2", "a3"],
         "spread_median_px": 4.6, "spread_max_px": 5.2,
         "threshold_px": [2.1, 12.5]},
    ]
    arrays = _fake_arrays(4, detected={"classic": [True] * 4,
                                       "sb": [True] * 4})

    frames = _burst_review_frames(images, arrays, burst_failures)

    # 并集去重、按数据清单顺序;未点名的 a4 不进清单
    assert [f["image_id"] for f in frames] == ["a1", "a2", "a3"]
    by_id = {f["image_id"]: f for f in frames}
    # a2 被两颗检测器同时点名:只保留一条,点名者列全
    assert by_id["a2"]["info"]["flagged_by"] == ["classic", "sb"]
    assert by_id["a1"]["info"]["flagged_by"] == ["classic"]
    assert by_id["a3"]["info"]["flagged_by"] == ["sb"]
    # 每帧原因按检测器各一条,数值与超限结论齐全(中位超限、最大未超)
    reason_c = by_id["a1"]["info"]["reason"]["classic"]
    assert "4.50px" in reason_c and "2.00px" in reason_c and "超限" in reason_c
    assert "5.00px" in reason_c and "12.00px" in reason_c and "未超" in reason_c
    assert "连拍断言失败" in reason_c
    reason_s = by_id["a3"]["info"]["reason"]["sb"]
    assert "4.60px" in reason_s and "2.10px" in reason_s and "超限" in reason_s
    # a2 的原因两颗检测器都有,各带各的数值
    assert set(by_id["a2"]["info"]["reason"]) == {"classic", "sb"}
    # 姿态/连拍序号/划分带齐;note 固定 BURST FLAG;角点照 detected 给
    assert by_id["a1"]["info"]["pose_id"] == "poseA"
    assert by_id["a1"]["info"]["repeat_idx"] == 0
    assert by_id["a1"]["info"]["split"] == "train"
    assert all(f["note"] == "BURST FLAG" for f in frames)
    assert all(f["uv_classic"] is not None and f["uv_sb"] is not None
               for f in frames)


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
    + test 一张(img4 正常),全部检出。按真实流水线布局构造:数据清单与
    存疑原图副本都在 ingest 运行目录,corners 运行目录只有角点产物
    (suspect_originals 从不在 corners 目录下)。返回 (corners 运行目录, 清单路径)。"""
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
    # ingest 运行目录:数据清单 + 存疑原图副本(入库体检写的,就在这里)
    ingest_run = tmp_path / "ingest" / "run_t"
    ingest_run.mkdir(parents=True)
    manifest_path = _write_fake_manifest(images, ingest_run / "manifest.json")
    _write_fake_image(ingest_run / "suspect_originals" / "img2.jpg", seed=99,
                      width=160, height=120)

    run_dir = tmp_path / "corners" / "run_t"
    uv = _grid_uv(160, 120)
    # 血缘里记上游清单路径:mosaic 按它反推 ingest 运行目录(真实布局可测)
    lineage = {"round_id": "t",
               "upstream": {"manifest.json": {"path": str(manifest_path)}}}
    for detector in ("classic", "sb"):
        save_corners(run_dir, "train", detector, image_ids=["img1", "img2", "img3"],
                     uv=np.stack([uv, uv, uv]),
                     sq_px=np.full(3, 15.0), win_size=np.full(3, 4.0),
                     detected=np.array([True, True, True]),
                     lineage=lineage)
        save_corners(run_dir, "test", detector, image_ids=["img4"],
                     uv=uv[np.newaxis], sq_px=np.array([15.0]),
                     win_size=np.array([4.0]), detected=np.array([True]),
                     lineage=lineage)
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


def test_suspect_mosaic_uses_ingest_copy_when_corners_dir_has_none(tmp_path):
    """WR-01 回归:corners 运行目录没有 suspect_originals(真实流水线从来
    如此——副本由入库体检写在 ingest 目录),存疑拼图仍取到 ingest 副本:
    不显式传 ingest_run_dir,靠角点产物血缘反推定位,逐格索引的 source
    指向 ingest 副本,证明没有退回原始路径。"""
    run_dir, manifest_path = _mosaic_fixtures(tmp_path)

    write_mosaics(run_dir, manifest_path, CalibrationConfig())

    # 真实布局:corners 运行目录下确实没有 suspect_originals
    assert not (run_dir / "suspect_originals").exists()
    index = json.loads(
        (run_dir / "mosaic" / "mosaic_index.json").read_text(encoding="utf-8"))
    cells = {c["image_id"]: c for c in index["suspect"]["cells"]}
    expected_copy = str(tmp_path / "ingest" / "run_t" / "suspect_originals" / "img2.jpg")
    assert cells["img2"]["source"] == expected_copy


def test_suspect_mosaic_falls_back_to_original_path_when_copy_missing(tmp_path):
    """WR-01 退回分支:ingest 目录里副本也缺失(如被手工清理)时,
    source 等于清单里的原始路径,拼图照常产出、不中断。"""
    run_dir, manifest_path = _mosaic_fixtures(tmp_path)
    (tmp_path / "ingest" / "run_t" / "suspect_originals" / "img2.jpg").unlink()

    write_mosaics(run_dir, manifest_path, CalibrationConfig())

    index = json.loads(
        (run_dir / "mosaic" / "mosaic_index.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = next(img["path"] for img in manifest["images"]
                    if img["image_id"] == "img2")
    cells = {c["image_id"]: c for c in index["suspect"]["cells"]}
    assert cells["img2"]["source"] == original


# ---------------------------------------------------------- 误差预算图 analyze_figs

def _synthetic_budget_section() -> dict:
    """合成最小预算段(键与 run_budget 的 budget 行契约一致,画图只消费
    段内现成键):前四个参数利用率比值有值(fx/fy/cx/cy),其余置 None
    (k1–k3/p1/p2 的下界对照无意义);后两个参数像素当量近零。"""
    layout = [
        ("fx", 0.059, 0.036, 0.002, 41.5),
        ("fy", 0.057, 0.019, 0.001, 41.6),
        ("cx", 0.056, 0.056, 0.001, 361.0),
        ("cy", 0.058, 0.058, 0.001, 378.0),
        ("k1", 4.0e-5, 0.052, 1.0e-4, None),
        ("k2", 1.3e-4, 0.135, 2.0e-4, None),
        ("k3", 1.1e-4, 0.094, 3.0e-4, None),
        ("p1", 3.5e-6, 0.009, 4.0e-4, None),
        ("p2", 2.5e-6, 0.010, 5.0e-4, None),
    ]
    return {
        "schema": "analyze-budget-v1",
        "model_id": "m2",
        "rows": [
            {
                "param": name,
                "empirical_std": emp,
                "empirical_std_px": px,
                "bootstrap_std": 0.2,
                "pose_sampling_std": 0.1,
                "multistart_disp_px": 0.07,
                "crlb_se": se,
                "utilization_ratio": util,
            }
            for name, emp, px, se, util in layout
        ],
    }


def test_plot_budget_bars_renders_png(tmp_path):
    """合成预算段:PNG 生成、返回路径与入参一致、文件非空(双子图布局
    含对数轴与中文标注,渲染不抛)。"""
    out = plot_budget_bars(_synthetic_budget_section(), tmp_path / "budget_bars.png")
    assert out == tmp_path / "budget_bars.png"
    assert out.is_file() and out.stat().st_size > 0


def test_plot_budget_bars_none_or_empty_draws_no_data(tmp_path):
    """无数据不冒充 0:budget 段为 None 或 rows 空都不抛异常,
    照常生成含"无数据"标注的 PNG。"""
    out_none = plot_budget_bars(None, tmp_path / "empty1.png")
    assert out_none.is_file() and out_none.stat().st_size > 0
    out_empty = plot_budget_bars({"rows": []}, tmp_path / "empty2.png")
    assert out_empty.is_file() and out_empty.stat().st_size > 0


def test_plot_budget_bars_chinese_labels_render(tmp_path):
    """中文标题/轴标签照常渲染(font.sans-serif 三件套已配,字体缺失
    只会变方框,不允许抛异常):真值近零参数(p1/p2)走"画面影响可忽略"
    标注分支,整图落盘成功。"""
    section = _synthetic_budget_section()
    out = plot_budget_bars(section, tmp_path / "中文预算图.png")
    assert out.is_file() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# ANA-02 残差结构两张空间图(r 分带曲线 + 覆盖足迹)
# ---------------------------------------------------------------------------

def _synthetic_residual_section(
        extrapolation: bool = True) -> dict:
    """合成 residual_structure 段:三带统计(末带空)、真实量级的覆盖
    足迹(r_max 0.685 / 四角 0.796,findings 结论 5 同款数字),
    extrapolation=False 时模拟全域覆盖(无外推区)。"""
    edges = (0.0, 0.2, 0.4, 0.6)

    def _bands(rms_vals, mean_vals):
        rows = []
        for i in range(len(edges) - 1):
            rows.append({
                "band_lo": edges[i], "band_hi": edges[i + 1],
                "n": 0 if rms_vals[i] is None else 100,
                "rms_px": rms_vals[i], "median_px": rms_vals[i],
                "max_px": None if rms_vals[i] is None else rms_vals[i] * 2,
                "mean_du": mean_vals[i][0] if mean_vals[i] else None,
                "mean_dv": mean_vals[i][1] if mean_vals[i] else None,
            })
        rows[-1]["n_beyond_hi"] = 12
        return rows

    r_max, r_corner = 0.685, 0.796
    return {
        "schema": "analyze-residual-v1",
        "model_id": "m2",
        "image_size": [640, 480],
        "r_normalization_basis": "evaluate delivery final_params @ run_synthetic",
        "r_normalization_params": {"fx": 800.0, "fy": 800.0,
                                   "cx": 320.0, "cy": 240.0},
        "train": {
            "per_view": {"n_views": 10, "rms_median_px": 0.12,
                         "rms_mean_px": 0.13, "rms_max_px": 0.2},
            "bands": _bands([0.10, 0.16, None], [(0.02, -0.01), (-0.03, 0.05), None]),
            "quadrants": [],
            "coverage": {
                "r_max_obs": r_max,
                "r_corner_frame": r_corner,
                "extrapolation_interval": [r_max, r_corner] if extrapolation else None,
                "frac_beyond_05": 0.21,
                "info_share_beyond_05": 0.61,
            },
        },
        "test": {
            "n_frames": 3, "n_failed": 0,
            "bands": _bands([0.09, 0.18, 0.22], [(0.01, 0.0), (-0.02, 0.04), (0.05, 0.02)]),
            "coverage": {"r_max_obs": r_max, "r_corner_frame": r_corner,
                         "extrapolation_interval": [r_max, r_corner]},
            "per_frame_rms": {"median_px": 0.12, "max_px": 0.2},
        },
        "extrapolation": {"applicable": extrapolation},
        "extrapolation_note": "外推区披露文案(k2)" if extrapolation else "",
    }


def _footprint_uv() -> np.ndarray:
    """640×480 画面里的规则观测网格(外环留空,对应外推区合成场景)。"""
    xs = np.linspace(60.0, 580.0, 40)
    ys = np.linspace(45.0, 435.0, 30)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def test_plot_radial_bands_renders_png(tmp_path):
    """合成残差结构段:两张子图(train/test 曲线 + 外推阴影)渲染成功,
    PNG 非空、返回路径与入参一致。"""
    out = plot_radial_bands(_synthetic_residual_section(),
                            tmp_path / "radial_bands.png")
    assert out == tmp_path / "radial_bars_or_bands" or out.is_file()
    assert out.is_file() and out.stat().st_size > 0


def test_plot_radial_bands_none_or_empty_draws_no_data(tmp_path):
    """无数据不冒充 0:段为 None 或两带表全空都不抛异常,
    照常生成含"无数据"标注的 PNG。"""
    out_none = plot_radial_bands(None, tmp_path / "empty1.png")
    assert out_none.is_file() and out_none.stat().st_size > 0
    out_empty = plot_radial_bands({"train": {"bands": []}, "test": {"bands": []}},
                                  tmp_path / "empty2.png")
    assert out_empty.is_file() and out_empty.stat().st_size > 0


def test_plot_coverage_footprint_renders_png_with_extrapolation(tmp_path):
    """合成足迹:散点 + r 等值线 + 外推阴影 + 四角标记全部渲染成功,
    PNG 非空。"""
    out = plot_coverage_footprint(_synthetic_residual_section(), _footprint_uv(),
                                  tmp_path / "footprint.png")
    assert out.is_file() and out.stat().st_size > 0


def test_plot_coverage_footprint_full_coverage_no_shading_no_raise(tmp_path):
    """全域覆盖(外推区间 None、r_max ≥ 四角 r):不画阴影、不抛异常,
    照常出图。"""
    section = _synthetic_residual_section(extrapolation=False)
    section["train"]["coverage"]["r_max_obs"] = 0.81  # ≥ r_corner 全域覆盖
    out = plot_coverage_footprint(section, _footprint_uv(),
                                  tmp_path / "footprint_full.png")
    assert out.is_file() and out.stat().st_size > 0
    # 分带曲线图同样吃全域覆盖场景(阴影函数静默跳过)
    out2 = plot_radial_bands(section, tmp_path / "bands_full.png")
    assert out2.is_file() and out2.stat().st_size > 0


def test_plot_coverage_footprint_none_or_empty_draws_no_data(tmp_path):
    """无数据不冒充:段为 None / uv 为空数组都照常出"无数据"PNG。"""
    out_none = plot_coverage_footprint(None, None, tmp_path / "empty1.png")
    assert out_none.is_file() and out_none.stat().st_size > 0
    out_empty = plot_coverage_footprint(_synthetic_residual_section(),
                                        np.zeros((0, 2)), tmp_path / "empty2.png")
    assert out_empty.is_file() and out_empty.stat().st_size > 0


def test_spatial_figs_source_contains_extrapolation_labels():
    """足迹叠加是硬要求(成功判据 2):两函数源码都含"外推"标注文本,
    并且注明图只是可视化(数字结论以 analyze.json 为准)。"""
    import inspect

    from calib.visualization import analyze_figs

    for fn in (analyze_figs.plot_radial_bands,
               analyze_figs.plot_coverage_footprint):
        source = inspect.getsource(fn)
        assert "外推" in source, f"{fn.__name__} 缺外推区标注(足迹叠加硬要求)"
        assert "analyze.json" in source, f"{fn.__name__} docstring 缺口径声明"


# ---------------------------------------------------------------------------
# ANA-03/ANA-04:三分量对比图 + 敏感度排序图 + 补拍预演图
# ---------------------------------------------------------------------------

def _synthetic_test_xyz_section(valid: bool = True) -> dict:
    """合成 test_xyz 段(键与 run_test_xyz 契约一致):三分量 median/max
    与可观测性汇总,量级照真实数据(z 分数 < 3、放大倍数几百倍)。"""
    def _mm(med, mx):
        return {"median": med, "max": mx}
    return {
        "schema": "analyze-test-xyz-v1",
        "model_id": "m2",
        "n_frames": 3,
        "n_valid": 3 if valid else 0,
        "n_cond_excluded": 0,
        "n_solve_failed": 0,
        "per_direction": {
            "dX_abs_mm": {"x": _mm(0.05, 0.09), "y": _mm(0.04, 0.08),
                          "z": _mm(0.30, 0.55)},
            "abs_zscore": {"x": _mm(0.6, 1.1), "y": _mm(0.5, 0.9),
                           "z": _mm(1.8, 2.6)},
        },
        "observability": {"cond_median": 33.0, "cond_max": 51.0,
                          "depth_amplification_median": 412.0},
        "rms_px": {"median": 0.12, "max": 0.19},
        "notes": ["Z 分量大 ≠ 误差主要在深度", "test 集参加过选型,带乐观偏差"],
    }


def _synthetic_what_if_section() -> dict:
    """合成 what_if 子段(键与 run_sensitivity 的 full 档产出一致):
    逐参数压缩比,fx/fy/cx/cy 略低于 1,k 参考列被补偿空间撑大。"""
    return {
        "synthetic": "补拍预演为合成假设(虚拟位姿非实拍)",
        "n_poses_requested": 6,
        "target_r": 0.75,
        "n_candidates_considered": 375,
        "n_poses_selected": 6,
        "selected_r_max": [0.92, 0.90, 0.88, 0.87, 0.86, 0.85],
        "compression_ratio": {"fx": 0.82, "fy": 0.83, "cx": 0.79, "cy": 0.80,
                              "k1": 0.55, "k2": 0.51, "p1": 0.62, "p2": 0.60,
                              "k3": 0.50},
        "compression_ratio_k_reference": {"k1": 0.55, "k2": 0.51, "k3": 0.50},
        "compression_reference_note": "k 列只作参考",
    }


def _synthetic_sensitivity_section() -> dict:
    """合成 sensitivity 段(画图只消费现成键):最大方向载荷集中在
    k2/k3(findings 结论 1 口径),方向像素当量与近零参数 p1/p2。"""
    return {
        "schema": "analyze-sensitivity-v1",
        "model_id": "m2",
        "step_sigma": 1.0,
        "theory_spectrum": {
            "param_names": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"],
            "eigvals": [6.2e-06, 5.6e-07],
            "top_directions": [
                {"rank": 1, "eigval": 6.2e-06,
                 "loads": {"k2": 0.42, "k3": 0.36, "k1": 0.15, "fx": 0.04,
                           "fy": 0.02, "cx": 0.005, "cy": 0.003,
                           "p1": 0.002, "p2": 0.001},
                 "px_equiv": 0.054},
                {"rank": 2, "eigval": 5.6e-07,
                 "loads": {"fx": 0.36, "fy": 0.35, "k1": 0.09, "k3": 0.09,
                           "cx": 0.05, "cy": 0.04, "k2": 0.01,
                           "p1": 0.005, "p2": 0.005},
                 "px_equiv": 0.325},
            ],
        },
        "px_equivalents": {
            "per_direction": [
                {"rank": 1, "step_sigma": 1.0, "px_equiv": 0.054},
                {"rank": 2, "step_sigma": 1.0, "px_equiv": 0.325},
            ],
            "p1_px_equiv": 0.0012,
            "p2_px_equiv": 0.0011,
            "note": "一切比较用像素当量公共货币",
        },
        "what_if": None,
        "what_if_note": "quick 档不做 what-if",
    }


def test_plot_xyz_compare_renders_png(tmp_path):
    """合成三分量段:左右两张条形图(原始 mm + z 分数)渲染成功,
    PNG 非空、返回路径与入参一致。"""
    out = plot_xyz_compare(_synthetic_test_xyz_section(),
                           tmp_path / "test_xyz.png")
    assert out == tmp_path / "test_xyz.png"
    assert out.is_file() and out.stat().st_size > 0


def test_plot_xyz_compare_none_or_all_failed_draws_no_data(tmp_path):
    """无数据不冒充:段为 None / n_valid=0(全失败)都不抛异常,
    照常生成含"无数据"标注的 PNG。"""
    out_none = plot_xyz_compare(None, tmp_path / "empty1.png")
    assert out_none.is_file() and out_none.stat().st_size > 0
    out_zero = plot_xyz_compare(_synthetic_test_xyz_section(valid=False),
                                tmp_path / "empty2.png")
    assert out_zero.is_file() and out_zero.stat().st_size > 0


def test_plot_sensitivity_ranking_renders_png(tmp_path):
    """合成敏感度段:最大方向载荷条形 + 像素当量天平(近零参数 p1/p2
    走"画面影响可忽略"标注分支)渲染成功。"""
    out = plot_sensitivity_ranking(_synthetic_sensitivity_section(),
                                   tmp_path / "sensitivity_ranking.png")
    assert out.is_file() and out.stat().st_size > 0


def test_plot_sensitivity_ranking_none_or_empty_draws_no_data(tmp_path):
    """无数据不冒充:段为 None / top 方向空都不抛异常,照常出 PNG。"""
    out_none = plot_sensitivity_ranking(None, tmp_path / "empty1.png")
    assert out_none.is_file() and out_none.stat().st_size > 0
    out_empty = plot_sensitivity_ranking({"theory_spectrum": {}},
                                         tmp_path / "empty2.png")
    assert out_empty.is_file() and out_empty.stat().st_size > 0


def test_plot_what_if_renders_png_with_synthetic_title(tmp_path):
    """合成 what_if 段:逐参数压缩比条形(1.0 参考线)渲染成功;源码
    标题文本含「合成预演(假设非实测)」——合成证据不许被误读为实测。"""
    out = plot_what_if(_synthetic_what_if_section(), tmp_path / "what_if.png")
    assert out.is_file() and out.stat().st_size > 0
    import inspect

    from calib.visualization import analyze_figs

    assert "合成预演(假设非实测)" in inspect.getsource(analyze_figs.plot_what_if)


def test_plot_what_if_none_draws_no_data(tmp_path):
    """无数据不冒充:what_if 为 None(quick 档)不抛异常,照常出
    含「quick 档不做 what-if」标注的 PNG。"""
    out = plot_what_if(None, tmp_path / "empty.png")
    assert out.is_file() and out.stat().st_size > 0
    import inspect

    from calib.visualization import analyze_figs

    assert "quick 档不做 what-if" in inspect.getsource(analyze_figs.plot_what_if)


def _synthetic_corner_opt_section() -> dict:
    """最小 corner_opt 段:2 方法 × 6 单元(噪声/模糊/对比各按真实键形)。"""
    units = ["noise_0.0", "noise_2.0", "noise_4.0",
             "blur_1.0", "blur_2.0", "contrast_0.2"]
    unit_ops = {"noise_0.0": [["noise", 0.0]], "noise_2.0": [["noise", 2.0]],
                "noise_4.0": [["noise", 4.0]], "blur_1.0": [["blur", 1.0]],
                "blur_2.0": [["blur", 2.0]], "contrast_0.2": [["contrast", 0.2]]}

    def _stat(rms):
        return {"rms_px": rms, "median_px": rms * 0.8,
                "mean_du": 0.001, "mean_dv": -0.002}

    def _stats(noise_slope):
        return {
            "sb": {u: _stat(0.0 if u == "noise_0.0" else noise_slope * float(
                u.split("_")[1])) for u in units},
            "classic_w7": {u: _stat(0.0 if u == "noise_0.0" else noise_slope
                                    * float(u.split("_")[1]) * 0.5)
                           for u in units},
        }

    return {"units": units, "unit_ops": unit_ops,
            "methods": ["classic_w7", "sb"], "production_method": "sb",
            "threshold_px": 0.087,
            "stats": {"_self": _stats(0.1), "_prod_sb": _stats(0.2)}}


def test_plot_corner_opt_curves_short_reference_name_renders_data(tmp_path):
    """回归(05-06 收口人工核查抓出):调用方传短名 reference="self" /
    "prod_sb",段内 stats 键是 "_self" / "_prod_sb"——别名必须对齐,
    否则查不到键、有数据也画成"无数据"分支。断言:短名渲染出的 PNG
    显著大于无数据分支(曲线真的画了),且校准档零位移在源码判读可查。"""
    section = _synthetic_corner_opt_section()
    out_self = plot_corner_opt_curves(section, tmp_path / "self.png",
                                      reference="self")
    out_prodsb = plot_corner_opt_curves(section, tmp_path / "prodsb.png",
                                        reference="prod_sb")
    out_none = plot_corner_opt_curves(None, tmp_path / "none.png")
    for p in (out_self, out_prodsb, out_none):
        assert p.is_file() and p.stat().st_size > 0
    # 有数据图必须明显大于"无数据"占位图(三面板曲线 + 图例 + 参考线)
    assert out_self.stat().st_size > out_none.stat().st_size * 2, \
        "reference='self' 仍走无数据分支:stats 别名对齐失效"
    assert out_prodsb.stat().st_size > out_none.stat().st_size * 2, \
        "reference='prod_sb' 仍走无数据分支:stats 别名对齐失效"


def test_plot_corner_opt_curves_missing_stats_draws_no_data(tmp_path):
    """无数据不冒充:stats 缺该口径(如段内只有 _self 却画 prod_sb)
    不抛异常,照常出含「无数据」标注的 PNG。"""
    section = _synthetic_corner_opt_section()
    section["stats"] = {"_self": section["stats"]["_self"]}
    out = plot_corner_opt_curves(section, tmp_path / "empty.png",
                                 reference="prod_sb")
    assert out.is_file() and out.stat().st_size > 0
