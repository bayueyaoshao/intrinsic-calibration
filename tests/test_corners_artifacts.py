"""角点产物落盘与读取的测试:save_corners → load_corners 往返一致性 + 安全防线。

夹具造 3 张图:2 张合成棋盘(应检出)+ 1 张纯噪声(应未检出),
走完整落盘/读回链路,验证:未检出帧照常占一行(全 NaN)、
掩码语义正确、血缘与检测配置留痕齐全、pickle 反序列化被关死。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from calib.corners.artifacts import load_corners, save_corners
from calib.corners.detect import detect_classic
from calib.io.config import CornersConfig
from calib.util.lineage import make_lineage

_CFG = CornersConfig()


@pytest.fixture
def burst_of_three(make_chessboard_image, tmp_path):
    """3 张图(2 张棋盘 + 1 张纯噪声)的检测结果 + 血缘字典 + 上游假 manifest。"""
    image_ids: list[str] = []
    uv_rows: list[np.ndarray | None] = []
    sq_rows: list[float] = []
    win_rows: list[float] = []
    detected: list[bool] = []

    for idx in range(2):
        img, _truth, _sq = make_chessboard_image(
            rvec=np.array([0.0, 0.02 * idx + 0.03, 0.0]))
        result = detect_classic(img, 11, 8, _CFG)
        assert result is not None
        image_ids.append(f"20260717_17000{idx}_000001")
        uv_rows.append(result["uv"])
        sq_rows.append(result["sq_px"])
        win_rows.append(float(result["win_size"]))
        detected.append(True)

    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)
    assert detect_classic(noise, 11, 8, _CFG) is None
    image_ids.append("20260717_170002_000001")
    uv_rows.append(None)   # 未检出:整行 NaN
    sq_rows.append(np.nan)
    win_rows.append(np.nan)
    detected.append(False)

    uv = np.full((3, 88, 2), np.nan)
    for i, row in enumerate(uv_rows):
        if row is not None:
            uv[i] = row

    # 上游假 manifest + 配置文件,让血缘指纹有真实哈希可算
    fake_manifest = tmp_path / "manifest.json"
    fake_manifest.write_text("{\"images\": []}", encoding="utf-8")
    fake_config = tmp_path / "calibration.yaml"
    fake_config.write_text("seed: 42\n", encoding="utf-8")
    lineage = make_lineage(
        config_files={"calibration.yaml": fake_config},
        upstream={"manifest.json": fake_manifest},
        round_id="run_test",
    )
    return {
        "image_ids": image_ids, "uv": uv, "sq_px": np.array(sq_rows),
        "win_size": np.array(win_rows), "detected": np.array(detected),
        "lineage": lineage, "out_dir": tmp_path / "corners_run",
    }


def test_save_and_load_roundtrip(burst_of_three):
    """落盘 → 读回:image_ids 长度 3、检出掩码 [True, True, False]、
    未检出整行全 NaN、检出行数值逐点一致。"""
    data = burst_of_three
    npz_path = save_corners(
        data["out_dir"], "train", "classic",
        image_ids=data["image_ids"], uv=data["uv"], sq_px=data["sq_px"],
        win_size=data["win_size"], detected=data["detected"],
        lineage=data["lineage"],
        meta_extra={"classic_flags": ["adaptive_thresh", "normalize_image"]},
    )
    assert npz_path.name == "corners_train_classic.npz"
    loaded = load_corners(npz_path)

    assert len(loaded["image_ids"]) == 3
    assert [str(x) for x in loaded["image_ids"]] == data["image_ids"]
    assert loaded["detected"].dtype == bool
    assert list(loaded["detected"]) == [True, True, False]
    assert not np.isfinite(loaded["uv"][2]).any()      # 未检出:整行 NaN
    assert np.isfinite(loaded["uv"][0]).all()          # 检出:无 NaN
    assert np.allclose(loaded["uv"][0], data["uv"][0])  # 往返逐点一致
    assert np.isfinite(loaded["sq_px"][:2]).all()
    assert not np.isfinite(loaded["sq_px"][2])
    assert (2 * loaded["win_size"][:2] + 1 < loaded["sq_px"][:2]).all()  # 窗口断言留痕可复核


def test_meta_sidecar_records_lineage_and_detector_config(burst_of_three):
    """同名 meta.json 旁车:血缘哈希(config/上游 manifest)与检测配置齐全。"""
    data = burst_of_three
    npz_path = save_corners(
        data["out_dir"], "train", "classic",
        image_ids=data["image_ids"], uv=data["uv"], sq_px=data["sq_px"],
        win_size=data["win_size"], detected=data["detected"],
        lineage=data["lineage"],
        meta_extra={"classic_flags": ["adaptive_thresh"], "sb_flags": 0},
    )
    meta_path = npz_path.with_name("corners_train_classic.meta.json")
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "config_sha256" in meta["lineage"]
    assert "calibration.yaml" in meta["lineage"]["config_sha256"]
    assert meta["lineage"]["upstream"]["manifest.json"]["sha256"]
    assert meta["classic_flags"] == ["adaptive_thresh"]
    assert meta["canonical_rule"]  # 规范化规则版本留痕


def test_save_writes_four_files_for_all_splits_and_detectors(burst_of_three):
    """train/test × classic/sb 四个文件名精确符合命名契约(D-07 + 划分)。"""
    data = burst_of_three
    paths = [
        save_corners(data["out_dir"], split, det,
                     image_ids=data["image_ids"], uv=data["uv"],
                     sq_px=data["sq_px"], win_size=data["win_size"],
                     detected=data["detected"], lineage=data["lineage"])
        for split in ("train", "test") for det in ("classic", "sb")
    ]
    names = sorted(p.name for p in paths)
    assert names == [
        "corners_test_classic.npz", "corners_test_sb.npz",
        "corners_train_classic.npz", "corners_train_sb.npz",
    ]


def test_save_and_load_source_roundtrip(burst_of_three):
    """SB 生产路线带 source 数组:三种来源值落盘读回逐位一致。

    source 记录每帧角点来源:"sb" = SB 本体检出,"classic_fallback" =
    SB 未检出、由经典检测器兜底填入,"" = 两法都未检出。夹具的 3 帧
    (2 检出 + 1 未检出)正好对应三种值。定长 unicode 数组不需要 pickle,
    读取仍走 load_corners 的 allow_pickle=False 安全路径。
    """
    data = burst_of_three
    source = np.array(["sb", "classic_fallback", ""])
    npz_path = save_corners(
        data["out_dir"], "train", "sb",
        image_ids=data["image_ids"], uv=data["uv"], sq_px=data["sq_px"],
        win_size=data["win_size"], detected=data["detected"],
        source=source, lineage=data["lineage"],
    )
    loaded = load_corners(npz_path)  # 内部强制 allow_pickle=False

    assert "source" in loaded
    assert loaded["source"].dtype.kind == "U"  # 纯文字数组,不含对象/pickle
    assert [str(x) for x in loaded["source"]] == ["sb", "classic_fallback", ""]


def test_save_without_source_writes_no_source_key(burst_of_three):
    """经典路线(不传 source):npz 里没有 source 键,产物保持纯净。

    经典路线没有"兜底"概念,调用签名与文件内容都与现状一字不差;
    下游按 'source' in data.files 即可区分两种产物。
    """
    data = burst_of_three
    npz_path = save_corners(
        data["out_dir"], "train", "classic",
        image_ids=data["image_ids"], uv=data["uv"], sq_px=data["sq_px"],
        win_size=data["win_size"], detected=data["detected"],
        lineage=data["lineage"],
    )
    loaded = load_corners(npz_path)

    assert "source" not in loaded
    assert loaded["detected"].tolist() == [True, True, False]  # 其余内容照常


def test_load_corners_rejects_pickle(tmp_path):
    """安全防线:含 pickle 对象数组的 npz,读取必须抛错而不是执行(T-02-01)。"""
    evil = tmp_path / "evil.npz"
    np.savez(evil, payload=np.array([{"命令": "rm -rf /"}], dtype=object))
    with pytest.raises(ValueError):
        load_corners(evil)


def test_load_corners_source_guard():
    """源码防线:load_corners 内的 np.load 必须显式 allow_pickle=False。"""
    source = pathlib.Path("src/calib/corners/artifacts.py").read_text(encoding="utf-8")
    assert "allow_pickle=False" in source
