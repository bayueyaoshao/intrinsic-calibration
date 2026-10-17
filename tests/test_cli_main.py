"""CLI 命令入口的行为测试(区别于各模块的单元测试)。

typer 的命令函数本来就是普通函数,这里直接进程内调用;工作目录用
monkeypatch.chdir 切到临时目录树——命令里的路径全部相对当前工作目录解析,
产物落在 tmp_path 下,真实 output/ 目录不受影响。上游产物(如 ingest 的
数据清单)按需手工构造,不真跑上游命令,测试保持秒级。
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from calib.cli.main import corners


def test_corners_empty_test_split_produces_all_npz_and_null_rate(
        tmp_path, monkeypatch, make_chessboard_image):
    """CR-01 锁定:"只有 train 没有 test"是数据层承诺的合法输入,
    corners 命令必须照常完成——4 份角点文件齐全(空划分 0 行)、检测质量
    摘要里该划分 success_rate 如实记 null,而不是在 numpy 索引上崩溃。
    """
    # 1. 两张同姿态 640×480 合成棋盘图(夹具自带真值角点,检出稳定)
    train_dir = tmp_path / "input_data" / "train" / "poseA"
    train_dir.mkdir(parents=True)
    for idx in (0, 1):
        img, _uv, _sq = make_chessboard_image()
        target = train_dir / f"poseA_{idx:02d}.jpg"
        if not cv2.imwrite(str(target), img):
            raise RuntimeError(f"测试夹具写图失败:{target}")

    # 2. 最小合法配置:棋盘规格按夹具默认 11×8;calibration.yaml 只写
    #    data 段,其余键全有默认值
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "board.yaml").write_text(
        "board:\n  board_type: chessboard\n  cols: 11\n  rows: 8\n"
        "  square_size_mm: 25.0\n  unit: mm\n", encoding="utf-8")
    (config_dir / "calibration.yaml").write_text(
        "data:\n  train: input_data/train\n  test: input_data/test\n",
        encoding="utf-8")

    # 3. 手工构造 ingest 产物:清单里只有 train 记录,没有任何 test 记录
    ingest_run = tmp_path / "output" / "ingest" / "run_20260101_000000"
    ingest_run.mkdir(parents=True)
    images = [
        {
            "image_id": f"poseA_{idx:02d}",
            "path": str(train_dir / f"poseA_{idx:02d}.jpg"),
            "split": "train",
            "pose_id": "poseA",
            "repeat_idx": idx,
            "n_repeats": 2,
            "precheck": {"verdict": "ok", "blur_norm": 1.0, "expo_hi": 0.0,
                         "expo_lo": 0.0, "suspect_sources": []},
        }
        for idx in (0, 1)
    ]
    (ingest_run / "manifest.json").write_text(
        json.dumps({"images": images}, ensure_ascii=False), encoding="utf-8")

    # 4. 进程内调用 corners 命令(工作目录已在临时树,产物落 tmp)
    monkeypatch.chdir(tmp_path)
    corners()  # 修复前:空 test 划分在检测质量摘要处抛 IndexError

    # 5. 断言:4 份 npz 齐全;test 两份是 0 行;摘要如实记 success_rate=null
    corners_runs = sorted((tmp_path / "output" / "corners").glob("run_*"))
    assert len(corners_runs) == 1, "corners 应恰好产出一个运行目录"
    out_dir = corners_runs[0]
    for split in ("train", "test"):
        for detector in ("classic", "sb"):
            assert (out_dir / f"corners_{split}_{detector}.npz").is_file(), \
                f"缺角点文件 corners_{split}_{detector}.npz"
    for detector in ("classic", "sb"):
        with np.load(out_dir / f"corners_test_{detector}.npz",
                     allow_pickle=False) as data:
            assert data["image_ids"].shape == (0,)

    summary = json.loads(
        (out_dir / "detection_summary.json").read_text(encoding="utf-8"))
    for detector in ("classic", "sb"):
        stat = summary["detection"]["test"][detector]
        assert stat["success_rate"] is None
        assert stat["n_images"] == 0
