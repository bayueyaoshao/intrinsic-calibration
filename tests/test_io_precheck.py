"""入库体检测试(IO-02):指标方向正确性 + 判定逻辑 + 永不自动剔除的行为锁定。

本文件前半是三个体检指标的方向性验证(清晰 vs 模糊/过曝/暗图/无棋盘),
后半(随 Task 2/3 补齐)是双来源存疑判定、manifest 全字段、存疑原图副本
与跨轮对比表的行为锁定。
"""

from __future__ import annotations

import json
import os

import numpy as np

from calib.io.config import CornersConfig, PrecheckConfig
from calib.io.dataset import scan_dataset
from calib.io.manifest import write_manifest
from calib.io.precheck import (
    board_roi,
    compute_metrics,
    exposure_clip,
    judge_suspects,
    visibility_probe,
)
from calib.visualization.suspects import write_suspect_originals

# 测试统一用真实棋盘规格(11×8 内角点),夹具默认 640×480 画布
COLS, ROWS = 11, 8
CORNERS_CFG = CornersConfig()


# ---------------------------------------------------------------- 指标方向性

def test_blur_metric_direction_clear_vs_blurred(make_chessboard_image):
    """清晰棋盘 vs 高斯模糊版:blur_raw 与 blur_norm 都要显著更低(比值 < 1/3)。

    模糊破坏边缘锐度,拉普拉斯(二阶差分)响应随之大幅下降——
    这是"值越小越糊"这一方向的直接验证。
    """
    clear, _, sq_px = make_chessboard_image(square_px=20.0)
    blurred, _, _ = make_chessboard_image(square_px=20.0, degrade="blur")

    m_clear = compute_metrics(clear, COLS, ROWS, CORNERS_CFG)
    m_blur = compute_metrics(blurred, COLS, ROWS, CORNERS_CFG)

    assert m_clear["roi_found"] and m_blur["roi_found"]  # 两张都应框出棋盘
    assert m_blur["blur_raw"] < m_clear["blur_raw"] / 3.0
    assert m_blur["blur_norm"] < m_clear["blur_norm"] / 3.0
    # 归一值确实除掉了方格尺度:与原始值不是同一个数
    assert m_clear["blur_norm"] != m_clear["blur_raw"]
    assert sq_px > 0


def test_blur_norm_normalization_formula(make_chessboard_image):
    """比例化公式验证:blur_norm 恰等于 blur_raw / 实测方格像素边长²。

    归一的含义:边缘锐度能量随方格像素边长的平方缩放,把实测尺度除掉之后,
    不同拍摄距离(方格在画面里变大变小)的模糊度才能放在一起比。
    原始值与归一值必须明显不同——证明归一不是摆设。
    """
    img, _, expected_sq = make_chessboard_image(square_px=20.0)
    m = compute_metrics(img, COLS, ROWS, CORNERS_CFG)

    assert m["roi_found"] is True
    assert abs(m["sq_px"] - expected_sq) / expected_sq < 0.05  # 实测方格边长贴近期望
    assert abs(m["blur_norm"] - m["blur_raw"] / m["sq_px"] ** 2) < 1e-9
    assert abs(m["blur_norm"] - m["blur_raw"]) > 1.0  # 除以 ~400 后两者必然拉开


def test_overexpose_high_clip_ratio(make_chessboard_image):
    """过曝图(像素截断到 250 以上):高光裁剪占比 > 0.3。"""
    over, _, _ = make_chessboard_image(degrade="overexpose")
    hi, lo = exposure_clip(over)
    assert hi > 0.3
    assert lo < 0.3  # 过曝图的死黑不应多


def test_dark_image_shadow_clip_ratio():
    """暗图(像素压到 5 以下):阴影裁剪占比 > 0.3。"""
    dark = np.full((240, 320), 3, dtype=np.uint8)
    hi, lo = exposure_clip(dark)
    assert lo > 0.3
    assert hi == 0.0


def test_pure_noise_visibility_probe_fails():
    """纯噪声图(无棋盘):双检测器探针都失败,角点数检查也为否。"""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    probe = visibility_probe(noise, COLS, ROWS, CORNERS_CFG)
    assert probe["classic_ok"] is False
    assert probe["sb_ok"] is False
    assert probe["corners_count_ok"] is False


def test_board_roi_falls_back_to_full_image_on_noise():
    """无棋盘图:board_roi 回退整张图,不抛异常,roi_found=False。"""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    roi, corners, found = board_roi(noise, COLS, ROWS)
    assert found is False
    assert corners is None
    assert roi.shape == noise.shape  # 回退整张图


def test_compute_metrics_returns_all_documented_keys(make_chessboard_image):
    """指标函数返回 dict 含全部约定键(blur_raw/blur_norm/expo_hi/expo_lo/
    classic_ok/sb_ok/box_area_ratio/corners_count_ok),纯函数不抛异常。"""
    img, _, _ = make_chessboard_image(square_px=20.0)
    m = compute_metrics(img, COLS, ROWS, CORNERS_CFG)
    for key in ("blur_raw", "blur_norm", "expo_hi", "expo_lo", "classic_ok",
                "sb_ok", "box_area_ratio", "corners_count_ok", "roi_found", "sq_px"):
        assert key in m, f"缺少约定键 {key}"
    assert 0.0 <= m["expo_hi"] <= 1.0 and 0.0 <= m["expo_lo"] <= 1.0
    assert 0.0 < m["box_area_ratio"] <= 1.0
    assert m["classic_ok"] is True or m["sb_ok"] is True


# ------------------------------------------------- 双来源判定 + 只标记不剔除

# 判定阈值用默认配置(与 calibration.yaml 一致:z=3.5,分位线 1%/99%)
PRECHECK_CFG = PrecheckConfig()


def _burst_metrics(blur_values: list[float], pose_id: str = "P1") -> list[dict]:
    """构造一个同姿态连拍的指标序列:可见性全通过、曝光全相同、模糊度按给定值。"""
    return [
        {
            "image_id": f"{pose_id}_{i}",
            "split": "train",
            "pose_id": pose_id,
            "n_repeats": len(blur_values),
            "blur_norm": blur_values[i],
            "expo_hi": 0.0,
            "expo_lo": 0.0,
            "classic_ok": True,
            "sb_ok": True,
        }
        for i in range(len(blur_values))
    ]


def test_pose_relative_flags_one_blurry_in_burst():
    """8 连拍里 1 张模糊度异常低:该张带 pose_relative:blur,其余 7 张 ok(D-02 主来源)。"""
    # 7 张正常值带轻微起伏;顶部两值并列,避免全局 99% 分位线误伤最高张;
    # 异常糊的那张(10.0)同时会踩全局 1% 线,但断言只要求主来源必须出现。
    values = [100.0, 100.5, 99.5, 100.2, 99.8, 100.3, 100.5, 10.0]
    judged = judge_suspects(_burst_metrics(values), PRECHECK_CFG)

    assert len(judged) == 8  # 输入几张输出几张(D-01)
    outlier = judged[7]
    assert "pose_relative:blur" in outlier["suspect_sources"]
    assert outlier["verdict"] == "suspect"
    for i in range(7):
        assert judged[i]["verdict"] == "ok", f"第 {i} 张不该被判存疑:{judged[i]}"


def test_single_image_pose_only_global_line():
    """单张姿态(n_repeats=1)没有连拍可比:只在踩全局分位线时存疑(D-02 兜底)。"""
    metrics = _burst_metrics([100.0, 100.5, 99.5, 100.2, 99.8, 100.3, 100.5, 100.1])
    lone = {
        "image_id": "LONE_0", "split": "test", "pose_id": "LONE_0", "n_repeats": 1,
        "blur_norm": 1.0, "expo_hi": 0.0, "expo_lo": 0.0,
        "classic_ok": True, "sb_ok": True,
    }
    judged = judge_suspects(metrics + [lone], PRECHECK_CFG)

    lone_verdict = judged[-1]
    assert "global:blur" in lone_verdict["suspect_sources"]
    assert not any(s.startswith("pose_relative:") for s in lone_verdict["suspect_sources"])
    assert lone_verdict["verdict"] == "suspect"
    # 同姿态 8 张互相差异细微,不应有人踩线(顶部并列设计,99% 线不误伤)
    assert all(j["verdict"] == "ok" for j in judged[:-1])


def test_mad_zero_group_no_false_positive():
    """组内指标完全相同(MAD=0):没有"组内偏离"可言,不应误报。"""
    judged = judge_suspects(_burst_metrics([100.0] * 8), PRECHECK_CFG)
    assert all(j["verdict"] == "ok" for j in judged)
    assert all(j["suspect_sources"] == [] for j in judged)


def test_judge_never_reduces_count_and_only_marks():
    """D-01 行为锁定:8 张输入 3 张异常,输出仍是 8 条、image_id 一个不少——
    判定只是标记,不存在"剔除"这个动作。"""
    values = [100.0, 10.0, 100.2, 10.0, 99.8, 100.2, 10.0, 100.2]  # 顶部并列,全局 99% 线不误伤
    judged = judge_suspects(_burst_metrics(values), PRECHECK_CFG)

    assert len(judged) == 8
    assert [j["image_id"] for j in judged] == [f"P1_{i}" for i in range(8)]
    assert sum(1 for j in judged if j["verdict"] == "suspect") == 3
    assert all("suspect_sources" in j and "verdict" in j for j in judged)


def test_global_visibility_source_when_both_detectors_fail():
    """双检测器都看不见棋盘:来源是 global:visibility(全局兜底的一种)。"""
    metrics = _burst_metrics([100.0] * 2)
    metrics[0]["classic_ok"] = False
    metrics[0]["sb_ok"] = False
    judged = judge_suspects(metrics, PRECHECK_CFG)
    assert "global:visibility" in judged[0]["suspect_sources"]
    assert judged[0]["verdict"] == "suspect"


# ------------------------------------------------------- manifest 与副本行为

def test_manifest_carries_full_precheck_fields(tmp_path, make_tiny_jpg):
    """manifest 每条 image 记录带 precheck 全字段(D-06:manifest 存全部数字)。"""
    make_tiny_jpg("input_data/train/P1/P1_0.jpg", 32, 24)
    make_tiny_jpg("input_data/train/P1/P1_1.jpg", 32, 24)
    records = scan_dataset(make_tiny_jpg.input_root)
    precheck = [
        {"image_id": "P1_0", "blur_raw": 1.5, "blur_norm": 0.25, "expo_hi": 0.01, "expo_lo": 0.02,
         "classic_ok": True, "sb_ok": False, "box_area_ratio": 0.4, "corners_count_ok": True,
         "suspect_sources": ["pose_relative:blur"], "verdict": "suspect"},
        {"image_id": "P1_1", "blur_raw": 1.6, "blur_norm": 0.26, "expo_hi": 0.01, "expo_lo": 0.02,
         "classic_ok": True, "sb_ok": True, "box_area_ratio": 0.4, "corners_count_ok": True,
         "suspect_sources": [], "verdict": "ok"},
    ]
    cfg_file = tmp_path / "calibration.yaml"
    cfg_file.write_text("data: {train: t, test: s}\n", encoding="utf-8")
    run_dir = write_manifest(records, (24, 32), (), tmp_path / "ingest",
                             {"calibration.yaml": cfg_file},
                             precheck_results=precheck)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    by_id = {img["image_id"]: img for img in manifest["images"]}

    p0 = by_id["P1_0"]["precheck"]
    assert p0["suspect_sources"] == ["pose_relative:blur"]
    assert p0["verdict"] == "suspect"
    assert p0["blur_raw"] == 1.5 and p0["expo_hi"] == 0.01
    assert p0["visibility"] == {"classic_ok": True, "sb_ok": False,
                                "box_area_ratio": 0.4, "corners_count_ok": True}
    assert by_id["P1_1"]["precheck"]["verdict"] == "ok"

    # 汇总:按来源前缀分桶计数 + 分划分双检测器成功率
    counts = manifest["summary"]["suspect_counts"]
    assert counts == {"pose_relative": 1, "global": 0, "suspect_total": 1}
    rates = manifest["summary"]["detector_success_rates"]["train"]
    assert rates["classic"] == 1.0 and rates["sb"] == 0.5


def test_manifest_without_precheck_keeps_schema_note(tmp_path, make_tiny_jpg):
    """不带体检结果调用(兼容旧入口):行为与 01-01 版一致,images 无 precheck 键。"""
    make_tiny_jpg("input_data/train/P1/P1_0.jpg", 32, 24)
    records = scan_dataset(make_tiny_jpg.input_root)
    cfg_file = tmp_path / "calibration.yaml"
    cfg_file.write_text("data: {train: t, test: s}\n", encoding="utf-8")
    run_dir = write_manifest(records, (24, 32), (), tmp_path / "ingest",
                             {"calibration.yaml": cfg_file})
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert all("precheck" not in img for img in manifest["images"])
    assert "suspect_counts" not in manifest["summary"]


def test_manifest_rejects_mismatched_precheck_length(tmp_path, make_tiny_jpg):
    """体检结果与照片记录条数对不上:硬报错,绝不静默错位配对。"""
    make_tiny_jpg("input_data/train/P1/P1_0.jpg", 32, 24)
    records = scan_dataset(make_tiny_jpg.input_root)
    cfg_file = tmp_path / "calibration.yaml"
    cfg_file.write_text("data: {train: t, test: s}\n", encoding="utf-8")
    try:
        write_manifest(records, (24, 32), (), tmp_path / "ingest",
                       {"calibration.yaml": cfg_file},
                       precheck_results=[{"verdict": "ok"}, {"verdict": "ok"}])
    except ValueError as exc:
        assert "不一致" in str(exc)
    else:
        raise AssertionError("条数不一致应当报错")


def test_write_suspect_originals_copies_only_suspects_and_keeps_source(tmp_path, make_tiny_jpg):
    """存疑原图副本(D-06):只复制 verdict==suspect 的照片,源文件一个字节都不动。"""
    paths = []
    items = []
    verdicts = ["suspect", "ok", "suspect", "ok", "suspect", "ok", "ok", "ok"]
    for i in range(8):
        p = make_tiny_jpg(f"input_data/train/P1/P1_{i}.jpg", 32, 24)
        paths.append(p)
        items.append({"image_id": f"P1_{i}", "path": p,
                      "precheck": {"verdict": verdicts[i]}})

    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    run_dir = tmp_path / "ingest" / "run_20260101_000000"
    run_dir.mkdir(parents=True)
    copied = write_suspect_originals(items, run_dir)

    assert sorted(p.name for p in copied) == ["P1_0.jpg", "P1_2.jpg", "P1_4.jpg"]
    copy_dir = run_dir / "suspect_originals"
    assert sorted(os.listdir(copy_dir)) == ["P1_0.jpg", "P1_2.jpg", "P1_4.jpg"]
    for p, (content, mtime) in zip(paths, before):
        assert p.read_bytes() == content      # 源内容不变
        assert p.stat().st_mtime_ns == mtime  # 源修改时间不变(只读源,只复制)
