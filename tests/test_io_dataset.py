"""姿态数据模型与目录扫描测试(IO-01):张数不齐/单张姿态/空目录/坏文件名的韧性。"""

from __future__ import annotations

import pytest

from calib.io.dataset import (
    imread_checked,
    scan_dataset,
    scan_dataset_detailed,
    verify_consistent_resolution,
)

# 与真实数据一致的时间戳文件名模式
TS_A = "20260717_172523"
GOOD_NAMES = [f"{TS_A}_559573", f"{TS_A}_728240", f"{TS_A}_876880"]


def test_two_poses_parse_pose_repeat_and_count(make_tiny_jpg):
    """2 个姿态(3 张 + 1 张):pose_id=目录名,repeat_idx 连续,n_repeats=实际张数。"""
    for name in GOOD_NAMES:
        make_tiny_jpg(f"input_data/train/{TS_A}/{name}.jpg", 32, 24)
    make_tiny_jpg("input_data/train/20260717_180101/20260717_180101_111111.jpg", 32, 24)

    records = scan_dataset(make_tiny_jpg.input_root)
    by_id = {r.image_id: r for r in records}
    pose = [r for r in records if r.pose_id == TS_A]
    assert len(pose) == 3
    assert sorted(r.repeat_idx for r in pose) == [0, 1, 2]
    assert all(r.n_repeats == 3 for r in pose)  # 张数数据驱动,不写死 8
    single = by_id["20260717_180101_111111"]
    assert (single.pose_id, single.repeat_idx, single.n_repeats) == ("20260717_180101", 0, 1)
    assert all(r.split == "train" for r in records)


def test_test_split_pose_id_is_image_id(make_tiny_jpg):
    """test 平铺图:pose_id == image_id,repeat_idx=0,n_repeats=1。"""
    make_tiny_jpg("input_data/test/20260721_142928_428148.jpg", 32, 24)
    make_tiny_jpg("input_data/test/20260721_143004_754969.jpg", 32, 24)

    records = [r for r in scan_dataset(make_tiny_jpg.input_root) if r.split == "test"]
    assert len(records) == 2
    for r in records:
        assert r.pose_id == r.image_id
        assert r.repeat_idx == 0
        assert r.n_repeats == 1


def test_single_image_pose_survives(make_tiny_jpg):
    """单张姿态(目录里只有 1 张):合法存续,不报错、不警告。"""
    make_tiny_jpg("input_data/train/20260717_180101/20260717_180101_111111.jpg", 32, 24)
    (make_tiny_jpg.input_root / "test").mkdir(parents=True, exist_ok=True)  # test 划分就位但为空

    scan = scan_dataset_detailed(make_tiny_jpg.input_root)
    assert len(scan.records) == 1
    assert scan.records[0].n_repeats == 1
    assert scan.warnings == ()  # 单张姿态是合法数据,不算警告


def test_empty_pose_dir_skipped_with_warning(make_tiny_jpg):
    """空姿态目录:跳过不崩,警告数 +1,警告文字说清了目录。"""
    make_tiny_jpg(f"input_data/train/{TS_A}/{TS_A}_559573.jpg", 32, 24)
    (make_tiny_jpg.input_root / "train" / "20260718_000000").mkdir(parents=True, exist_ok=True)
    (make_tiny_jpg.input_root / "test").mkdir(parents=True, exist_ok=True)

    scan = scan_dataset_detailed(make_tiny_jpg.input_root)
    assert [r.pose_id for r in scan.records] == [TS_A]
    assert len(scan.warnings) == 1
    assert "20260718_000000" in scan.warnings[0]


def test_non_timestamp_name_warns_but_parses(make_tiny_jpg):
    """文件名不符合时间戳模式:仍按自然序解析(repeat_idx 连续),且有中文警告。"""
    make_tiny_jpg(f"input_data/train/{TS_A}/bbb.jpg", 32, 24)
    make_tiny_jpg(f"input_data/train/{TS_A}/aaa.jpg", 32, 24)
    (make_tiny_jpg.input_root / "test").mkdir(parents=True, exist_ok=True)

    scan = scan_dataset_detailed(make_tiny_jpg.input_root)
    assert len(scan.records) == 2
    by_id = {r.image_id: r for r in scan.records}
    # 字典序 aaa < bbb,连拍序号按文件名排序(回退路径,不崩)
    assert by_id["aaa"].repeat_idx == 0
    assert by_id["bbb"].repeat_idx == 1
    assert len(scan.warnings) == 2  # 每张不合模式的照片各一条警告
    assert "时间戳" in scan.warnings[0]


def test_inconsistent_resolution_hard_error(make_tiny_jpg):
    """分辨率不一致:硬报错(单相机契约被破坏),消息含两张图的尺寸。"""
    make_tiny_jpg(f"input_data/train/{TS_A}/{TS_A}_559573.jpg", 32, 24)
    make_tiny_jpg("input_data/test/20260721_142928_428148.jpg", 64, 48)

    records = scan_dataset(make_tiny_jpg.input_root)
    with pytest.raises(ValueError) as exc_info:
        verify_consistent_resolution(records)
    msg = str(exc_info.value)
    assert "分辨率不一致" in msg
    assert "32x24" in msg and "64x48" in msg  # 两张图的尺寸都在消息里


def test_verify_resolution_returns_size(make_tiny_jpg):
    """分辨率一致:返回 (宽, 高) 供 manifest 记录。"""
    make_tiny_jpg(f"input_data/train/{TS_A}/{TS_A}_559573.jpg", 32, 24)
    make_tiny_jpg(f"input_data/train/{TS_A}/{TS_A}_728240.jpg", 32, 24)

    assert verify_consistent_resolution(scan_dataset(make_tiny_jpg.input_root)) == (32, 24)


def test_imread_checked_raises_on_corrupt_file(make_tiny_jpg):
    """坏文件读图:cv2.imread 返回 None 不报错的老坑,封装后必须立刻给中文错误(Pitfall 7)。"""
    bad = make_tiny_jpg("input_data/test/bad.jpg", 8, 8)
    bad.write_bytes(b"not a jpeg at all")  # 故意写坏
    with pytest.raises(ValueError, match="bad.jpg"):
        imread_checked(bad)


def test_missing_data_dir_error(tmp_path):
    """数据目录整个不存在:给中文错误提示检查配置,而不是 FileNotFoundError 裸异常。"""
    with pytest.raises(ValueError, match="input_data"):
        scan_dataset(tmp_path / "input_data")
