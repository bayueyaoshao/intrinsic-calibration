"""观测集装载契约测试:行序/剔除/未检出/NaN/视图数下限/来源留痕,全用合成数据锁死。

锁定的行为清单(对应研究 Pitfall 10 行序漂移、D-02 sb 路线来源留痕):
1. 全检出装载:image_ids / pose_ids / corner_source 与写入顺序逐项相等,
   object_points 形状 (88,3) 且逐值等于 baseline.build_object_points,
   image_points 形状 (V,88,2) float64;
2. 剔除清单命中的帧不进观测集(逻辑剔除,原数据不动),名单经
   excluded_image_ids 可查,清单里写了不存在的照片不算命中;
3. detected=False 的帧被跳过,其 NaN 角点不触发坏数断言;
4. 检出标记为真但角点含 NaN 的帧抛 AssertionError,消息点名该照片;
5. 过滤后只剩 1 个可用视图时抛 ValueError(消息含「可用视图」);
6. npz 的 image_ids 与清单顺序不一致时抛 AssertionError(行序锁);
7. split="test" 同样可装载(D-04 存疑帧复核的输入);非法 split 抛 ValueError;
8. 没有 source 键的 npz 不是 sb 路线(D-02),中文报错拒绝装载。

全部用例跑在 tmp_path 里的合成 npz + 最小合法清单上(images 含
image_id/pose_id/split,summary 含 resolution),不引用真实 output/ 产物,
秒级完成;npz 写入只用纯数值与定长 unicode 数组(禁 object dtype,
与 allow_pickle=False 读取防线一致)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from calib.io.config import BoardSpec
from calib.solve.baseline import build_object_points
from calib.solve.observation import excluded_image_ids, load_observation_set

# 与真实棋盘同规格:11×8 = 88 个内角点,角点数组的形状契约由此锚定
BOARD = BoardSpec(cols=11, rows=8, square_size_mm=25.0)
IMAGE_SIZE = (640, 480)  # (宽, 高),写进清单 summary.resolution


def _write_manifest(path: Path, entries: list[tuple[str, str, str]],
                    resolution: tuple[int, int] = IMAGE_SIZE) -> None:
    """写最小合法数据清单:images 每条含 image_id/split/pose_id,summary 含 resolution。"""
    document = {
        "summary": {"resolution": [resolution[0], resolution[1]]},
        "images": [
            {"image_id": image_id, "split": split, "pose_id": pose_id}
            for image_id, split, pose_id in entries
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)


def _write_corners(path: Path, image_ids: list[str], uv: np.ndarray,
                   detected: list[bool], source: list[str] | None = None) -> None:
    """写合成角点 npz:纯数值 + 定长 unicode 数组,绝不用 object dtype(pickle 防线不破)。"""
    arrays = {
        "image_ids": np.asarray(image_ids),
        "uv": np.asarray(uv, dtype=np.float64),
        "detected": np.asarray(detected, dtype=bool),
    }
    if source is not None:
        arrays["source"] = np.asarray(source)
    np.savez(path, **arrays)


def _train_entries(n: int = 8) -> list[tuple[str, str, str]]:
    """n 张 train 照片的清单条目:img_00..img_{n-1},一照一姿态。"""
    return [(f"img_{k:02d}", "train", f"pose_{k:02d}") for k in range(n)]


def _finite_uv(n: int = 8, seed: int = 5) -> np.ndarray:
    """n 帧有限随机角点 (n, 88, 2),量级贴真实画面像素坐标。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(50.0, 590.0, size=(n, BOARD.cols * BOARD.rows, 2))


def test_load_train_preserves_row_order_and_shapes(tmp_path):
    """行为 1:全检出装载后,三个逐行字段与写入顺序逐项相等,数组形状符合契约。"""
    n = 8
    entries = _train_entries(n)
    ids = [e[0] for e in entries]
    poses = [e[2] for e in entries]
    sources = ["classic_fallback" if k == 3 else "sb" for k in range(n)]
    uv = _finite_uv(n)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, entries)
    npz_path = tmp_path / "corners_train_sb.npz"
    _write_corners(npz_path, ids, uv, [True] * n, sources)

    obs = load_observation_set(npz_path, manifest_path, BOARD)

    assert obs.image_ids == tuple(ids), \
        f"image_ids 应与清单逐项相等:期望 {tuple(ids)},实际 {obs.image_ids}"
    assert obs.pose_ids == tuple(poses), \
        f"pose_ids 应与清单逐项相等:期望 {tuple(poses)},实际 {obs.pose_ids}"
    assert obs.corner_source == tuple(sources), \
        f"corner_source 应逐帧留痕:期望 {tuple(sources)},实际 {obs.corner_source}"
    assert obs.object_points.shape == (88, 3), \
        f"object_points 形状应为 (88,3),实际 {obs.object_points.shape}"
    assert np.array_equal(obs.object_points, build_object_points(BOARD)), \
        "物点网格应直接复用 baseline.build_object_points,数值必须逐项一致"
    assert obs.image_points.shape == (n, 88, 2), \
        f"image_points 形状应为 ({n},88,2),实际 {obs.image_points.shape}"
    assert obs.image_points.dtype == np.float64, \
        f"image_points 应为 float64,实际 {obs.image_points.dtype}"
    assert np.array_equal(obs.image_points, uv), "image_points 应原样保留写入的角点数值"
    assert obs.image_size == IMAGE_SIZE, \
        f"image_size 应取清单 resolution(宽,高):期望 {IMAGE_SIZE},实际 {obs.image_size}"


def test_exclude_images_removed_and_names_visible(tmp_path):
    """行为 2:剔除清单命中的帧不进观测集,名单经 excluded_image_ids 可查;
    清单里写了不存在的照片不算命中(谁剔了什么可审计)。"""
    entries = _train_entries(8)
    ids = [e[0] for e in entries]
    uv = _finite_uv(8)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, entries)
    npz_path = tmp_path / "corners_train_sb.npz"
    _write_corners(npz_path, ids, uv, [True] * 8, ["sb"] * 8)

    exclude = ("img_02", "img_05", "not_in_split")
    obs = load_observation_set(npz_path, manifest_path, BOARD,
                               exclude_images=exclude)

    expected = tuple(i for i in ids if i not in ("img_02", "img_05"))
    assert obs.image_ids == expected, \
        f"剔除后 image_ids:期望 {expected},实际 {obs.image_ids}"
    assert obs.image_points.shape == (6, 88, 2), \
        f"剔除 2 帧后应剩 6 个视图,实际形状 {obs.image_points.shape}"

    seen = excluded_image_ids(npz_path, manifest_path, exclude_images=exclude)
    assert seen == ("img_02", "img_05"), \
        f"实际命中名单应为 ('img_02','img_05')(按字典序、不含清单外的名字),实际 {seen}"


def test_undetected_frames_skipped(tmp_path):
    """行为 3(前半):detected=False 的帧被跳过,其 NaN 角点不触发坏数断言。"""
    n = 8
    entries = _train_entries(n)
    ids = [e[0] for e in entries]
    uv = _finite_uv(n)
    uv[1] = np.nan  # 未检出的整行是 NaN(与真实产物一致)
    uv[4] = np.nan
    detected = [k not in (1, 4) for k in range(n)]
    sources = ["" if k in (1, 4) else "sb" for k in range(n)]
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, entries)
    npz_path = tmp_path / "corners_train_sb.npz"
    _write_corners(npz_path, ids, uv, detected, sources)

    obs = load_observation_set(npz_path, manifest_path, BOARD)

    expected = tuple(ids[k] for k in range(n) if k not in (1, 4))
    assert obs.image_ids == expected, \
        f"未检出帧应被跳过:期望 {expected},实际 {obs.image_ids}"
    assert obs.corner_source == ("sb",) * 6, \
        f"跳过后只剩 6 帧且来源全为 sb,实际 {obs.corner_source}"
    assert np.isfinite(obs.image_points).all(), "进标定的角点必须全部是有限值"


def test_detected_but_nan_raises_with_image_id(tmp_path):
    """行为 3(后半,独立用例):检出标记为真但角点含 NaN → AssertionError,
    消息点名该照片,绝不带着坏数进求解。"""
    entries = _train_entries(8)
    ids = [e[0] for e in entries]
    uv = _finite_uv(8)
    uv[3, 40, 1] = np.nan  # img_03 第 40 个角点的 v 坐标被污染
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, entries)
    npz_path = tmp_path / "corners_train_sb.npz"
    _write_corners(npz_path, ids, uv, [True] * 8, ["sb"] * 8)

    with pytest.raises(AssertionError) as excinfo:
        load_observation_set(npz_path, manifest_path, BOARD)
    message = str(excinfo.value)
    assert "img_03" in message, f"报错应点名坏照片编号,实际消息:{message}"
    assert "非数值" in message, f"报错应说明是 NaN/无穷坏数,实际消息:{message}"


def test_too_few_usable_views_raises(tmp_path):
    """行为 4:剔除后只剩 1 个可用视图 → ValueError,消息含「可用视图」。"""
    entries = _train_entries(3)
    ids = [e[0] for e in entries]
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, entries)
    npz_path = tmp_path / "corners_train_sb.npz"
    _write_corners(npz_path, ids, _finite_uv(3), [True] * 3, ["sb"] * 3)

    with pytest.raises(ValueError, match="可用视图"):
        load_observation_set(npz_path, manifest_path, BOARD,
                             exclude_images=("img_01", "img_02"))


def test_row_order_mismatch_raises(tmp_path):
    """行为 5(行序锁,Pitfall 10):npz 与清单的照片编号顺序不一致 → AssertionError,
    报错指出第一处不一致的行号与两侧的照片编号。"""
    entries = _train_entries(8)
    ids = [e[0] for e in entries]
    shuffled = list(ids)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]  # 只交换前两行
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, entries)
    npz_path = tmp_path / "corners_train_sb.npz"
    _write_corners(npz_path, shuffled, _finite_uv(8), [True] * 8, ["sb"] * 8)

    with pytest.raises(AssertionError) as excinfo:
        load_observation_set(npz_path, manifest_path, BOARD)
    message = str(excinfo.value)
    assert "对不上" in message, f"报错应说明清单与角点文件对不上,实际消息:{message}"
    assert "第 0 行" in message, f"报错应指出第一处不一致在第 0 行,实际消息:{message}"
    assert "img_00" in message and "img_01" in message, \
        f"报错应同时给出清单侧与文件侧的照片编号,实际消息:{message}"


def test_split_test_loads_test_npz(tmp_path):
    """行为 6:test 切分同样可装载(D-04 存疑帧复核只估位姿评分的输入),
    装的是 test 那份 npz 与清单里 test 的行。"""
    entries = (_train_entries(6)
               + [(f"tst_{k:02d}", "test", f"test_pose_{k:02d}") for k in range(4)])
    test_ids = [e[0] for e in entries if e[1] == "test"]
    test_poses = [e[2] for e in entries if e[1] == "test"]
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, entries)
    npz_path = tmp_path / "corners_test_sb.npz"
    _write_corners(npz_path, test_ids, _finite_uv(4, seed=9), [True] * 4, ["sb"] * 4)

    obs = load_observation_set(npz_path, manifest_path, BOARD, split="test")

    assert obs.image_ids == tuple(test_ids), \
        f"test 切分应装载清单里 test 的照片:期望 {tuple(test_ids)},实际 {obs.image_ids}"
    assert obs.pose_ids == tuple(test_poses), \
        f"pose_ids 应取 test 各照片的姿态编号,实际 {obs.pose_ids}"
    assert obs.image_points.shape == (4, 88, 2), \
        f"image_points 形状应为 (4,88,2),实际 {obs.image_points.shape}"


def test_invalid_split_raises_chinese_error(tmp_path):
    """行为 6:split 白名单只有 train/test,非法值报中文 ValueError(含合法取值与实际值)。"""
    entries = _train_entries(8)
    ids = [e[0] for e in entries]
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, entries)
    npz_path = tmp_path / "corners_train_sb.npz"
    _write_corners(npz_path, ids, _finite_uv(8), [True] * 8, ["sb"] * 8)

    with pytest.raises(ValueError) as excinfo:
        load_observation_set(npz_path, manifest_path, BOARD, split="val")
    message = str(excinfo.value)
    assert "train" in message and "test" in message, \
        f"报错应写明合法取值 train/test,实际消息:{message}"
    assert "val" in message, f"报错应带上传入的实际值,实际消息:{message}"


def test_missing_source_key_rejected_as_non_sb_route(tmp_path):
    """契约:没有 source 键的 npz 不是 sb 路线(D-02),报中文错误拒绝装载。"""
    entries = _train_entries(8)
    ids = [e[0] for e in entries]
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, entries)
    npz_path = tmp_path / "corners_train_sb.npz"
    _write_corners(npz_path, ids, _finite_uv(8), [True] * 8, source=None)  # 不写 source 键

    with pytest.raises(ValueError) as excinfo:
        load_observation_set(npz_path, manifest_path, BOARD)
    message = str(excinfo.value)
    assert "source" in message, f"报错应说明缺的是 source 键,实际消息:{message}"
    assert "sb" in message, f"报错应提示改用 sb 路线角点文件,实际消息:{message}"
