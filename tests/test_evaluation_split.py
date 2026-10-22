"""姿态切分纯函数的单元测试(小样合成数据,防泄漏门禁在这里盯死)。

小样证明什么:
- 零重叠硬门禁(ROADMAP 成功标准 1):任意折/任意轮/任意训练子集下,
  训练侧与验证侧的姿态交集必须为空——按姿态切分是结构性防数据泄漏,
  切错一行整场考试作废;
- 确定性(D-03/D-04/D-15 的复现根基):同种子下分折 / bootstrap 重采样 /
  取第 i 张分组 / 留一姿态的分配逐位一致(seed=42);
- 分组构造(D-05/EVA-03):组数 = 全体姿态最少重复张数、姿态内按文件名
  排序取第 i 张、组间姿态集合单调不增。

合成 ObservationSet 直接构造(字段契约见 solve/observation.py,数值随意,
切分只看姿态与照片编号);配置对象直构造绕过 yaml(体例照
test_noise_injection.py)。真实 80 姿态全量实验由 calib evaluate 闭环与
集成测试覆盖。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calib.evaluation.splitting import (
    _seed_for,
    assign_cv_folds,
    bootstrap_resample,
    exposures_per_pose_mask,
    leave_one_pose_out,
    obs_rows,
    pose_row_mask,
    sample_pose_ids,
    slice_groups_by_index,
    split_by_pose_ids,
    unique_poses,
)
from calib.io.config import CalibrationConfig
from calib.solve.observation import ObservationSet

# 默认配置:seed=42 + evaluate 默认(5 折 × 5 轮、bootstrap 200)
CFG = CalibrationConfig()


def _pose_list(n: int) -> list[str]:
    """n 个去重姿态 ID(pose_000 … pose_0xx)。"""
    return [f"pose_{k:03d}" for k in range(n)]


def _make_obs(pose_images: dict[str, list[str]]) -> ObservationSet:
    """从 {姿态: [照片编号]} 合成观测集(数值随意,角点形状合规即可)。"""
    image_ids: list[str] = []
    pose_ids: list[str] = []
    for pose, imgs in pose_images.items():
        for img in imgs:
            image_ids.append(img)
            pose_ids.append(pose)
    n, n_points = len(image_ids), 88  # 与真实棋盘同规格(11×8 内角点)
    return ObservationSet(
        image_ids=tuple(image_ids),
        pose_ids=tuple(pose_ids),
        object_points=np.zeros((n_points, 3), dtype=np.float64),
        image_points=np.arange(n * n_points * 2, dtype=np.float64).reshape(n, n_points, 2),
        image_size=(640, 480),
        corner_source=tuple(["sb"] * n),
    )


def _six_pose_obs() -> ObservationSet:
    """6 姿态 × 每姿态 2 张的小样观测集(共 12 行)。"""
    return _make_obs({f"pose_{k}": [f"IMG_{k}a", f"IMG_{k}b"] for k in range(6)})


# ---- 行为 1:split_by_pose_ids——行数、零交集、连拍同进同出 ----

def test_split_by_pose_ids_row_counts_and_zero_overlap():
    """行为 1:两侧行数 = 各侧姿态数 × 每姿态张数;姿态零交集;连拍同进同出。"""
    obs = _six_pose_obs()
    train, val = split_by_pose_ids(obs, {"pose_0", "pose_3", "pose_5"})
    assert len(train.image_ids) == 6  # 3 姿态 × 2 张
    assert len(val.image_ids) == 6
    assert set(train.pose_ids) == {"pose_0", "pose_3", "pose_5"}
    assert set(train.pose_ids).isdisjoint(set(val.pose_ids))
    assert set(train.image_ids) == {"IMG_0a", "IMG_0b", "IMG_3a", "IMG_3b",
                                    "IMG_5a", "IMG_5b"}
    # 其余字段跟着走:物点网格与图像尺寸原样传递,来源逐行保留
    assert train.object_points.shape == obs.object_points.shape
    assert train.image_size == obs.image_size
    assert train.corner_source == tuple(["sb"] * 6)


def test_split_by_pose_ids_empty_train_side_chinese_error():
    """行为 1:训练侧为空(没留考题也治不了学)——中文报错。"""
    with pytest.raises(ValueError, match="训练"):
        split_by_pose_ids(_six_pose_obs(), set())


def test_split_by_pose_ids_empty_val_side_chinese_error():
    """行为 1:训练姿态占满全体,验证侧为空(留不出考题)——中文报错。"""
    obs = _six_pose_obs()
    all_poses = set(unique_poses(obs))
    with pytest.raises(ValueError, match="验证"):
        split_by_pose_ids(obs, all_poses)


def test_split_by_pose_ids_unknown_pose_chinese_error():
    """行为 1:传入不存在的姿态 ID——中文报错并点名。"""
    with pytest.raises(ValueError, match="pose_99"):
        split_by_pose_ids(_six_pose_obs(), {"pose_0", "pose_99"})


# ---- 防泄漏硬门禁(ROADMAP 成功标准 1):80 姿态 × 5 轮 × 5 折 ----

def test_hard_gate_zero_overlap_across_all_folds_and_rounds():
    """任意折/任意轮 train∩val = ∅,且按分配真切开观测集后行也不串(显式断言)。"""
    poses = _pose_list(80)
    obs = _make_obs({p: [f"{p}_r{r}" for r in range(2)] for p in poses})
    for entry in assign_cv_folds(poses, CFG):
        train_poses, val_poses = set(entry["train_poses"]), set(entry["val_poses"])
        assert train_poses.isdisjoint(val_poses)  # 姿态级零交集
        train_obs, val_obs = split_by_pose_ids(obs, set(entry["train_poses"]))
        # 行级零交集:同一张照片绝不出现在两侧;行数守恒
        assert set(train_obs.image_ids).isdisjoint(set(val_obs.image_ids))
        assert len(train_obs.image_ids) + len(val_obs.image_ids) == len(obs.image_ids)


# ---- 行为 2:assign_cv_folds——5 轮 × 5 折、每姿态恰 5 次进验证、确定性 ----

def test_assign_cv_folds_structure_and_val_counts():
    """行为 2:5 轮 × 5 折;每轮每姿态恰进一个折;全部轮次累计恰 5 次进验证。"""
    poses = _pose_list(80)
    assignment = assign_cv_folds(poses, CFG)
    assert len(assignment) == 25  # 5 轮 × 5 折
    for r in range(5):
        round_entries = [e for e in assignment if e["round"] == r]
        assert len(round_entries) == 5
        merged = [p for e in round_entries for p in e["val_poses"]]
        # 每轮:折间零交集且并集 = 全体姿态(每姿态恰进一个折)
        assert len(merged) == 80
        assert set(merged) == set(poses)
        for e in round_entries:
            assert len(e["val_poses"]) == 16   # 80 姿态 / 5 折
            assert len(e["train_poses"]) == 64
    # 全部轮次累计:每姿态恰好 5 次进验证
    val_counts: dict[str, int] = {}
    for e in assignment:
        for p in e["val_poses"]:
            val_counts[p] = val_counts.get(p, 0) + 1
    assert val_counts == {p: 5 for p in poses}


def test_assign_cv_folds_deterministic_and_serializable():
    """行为 2:同种子两次调用分配逐位一致;分配表纯 str/int,可 json 落盘复查。"""
    poses = _pose_list(80)
    first = assign_cv_folds(poses, CFG)
    assert assign_cv_folds(poses, CFG) == first  # 同种子逐位一致(D-03)
    payload = json.dumps(first)  # 序列化不炸 = 折分配表可落盘复查
    assert '"round"' in payload and '"train_poses"' in payload


def test_assign_cv_folds_rejects_empty_and_duplicate_poses():
    """非法输入:空姿态清单 / 重复姿态——中文报错。"""
    with pytest.raises(ValueError, match="空"):
        assign_cv_folds([], CFG)
    with pytest.raises(ValueError, match="重复"):
        assign_cv_folds(["pose_a", "pose_b", "pose_a"], CFG)


# ---- 行为 3:bootstrap_resample——有放回、可重复、确定性 ----

def test_bootstrap_resample_with_replacement_and_deterministic():
    """行为 3:200 次有放回重采样,每次恰抽 80 个姿态;同种子逐位一致。"""
    poses = _pose_list(80)
    resamples = bootstrap_resample(poses, CFG)
    assert len(resamples) == 200
    assert {e["rep"] for e in resamples} == set(range(200))
    for e in resamples:
        assert len(e["pose_ids"]) == 80
        assert set(e["pose_ids"]).issubset(set(poses))
        assert sum(e["counts"].values()) == 80
        for p, c in e["counts"].items():  # counts 与 pose_ids 逐一对应
            assert c == e["pose_ids"].count(p)
    # 有放回的实证:存在抽中不止一次的姿态(80 抽 80 全不重复的概率 ~e^-40)
    assert any(c > 1 for e in resamples for c in e["counts"].values())
    assert bootstrap_resample(poses, CFG) == resamples  # 同种子逐位一致(D-04)


# ---- 行为 4:slice_groups_by_index——组数 = 最少重复张数、文件名排序 ----

def test_slice_groups_by_index_min_count_and_sorting():
    """行为 4:组数 = 最少重复张数;每姿态至多 1 张/组;姿态内按文件名排序取第 i 张。"""
    obs = _make_obs({
        "pose_a": ["a_03", "a_01", "a_02"],   # 故意乱序:切分必须按文件名排序
        "pose_b": ["b_02", "b_01", "b_03"],
        "pose_c": ["c_02", "c_01"],           # 只有 2 张 → 决定组数 = 2
    })
    groups = slice_groups_by_index(obs, CFG)
    assert [g["group"] for g in groups] == [0, 1]  # 组数 = min(3,3,2) = 2
    g0, g1 = groups
    # 第 0 组取排序后第 1 张,第 1 组取第 2 张(行号按合成时的行序)
    assert g0["pose_rows"] == {"pose_a": 1, "pose_b": 4, "pose_c": 7}
    assert g1["pose_rows"] == {"pose_a": 2, "pose_b": 3, "pose_c": 6}
    assert g0["image_ids"] == ["a_01", "b_01", "c_01"]
    assert g1["image_ids"] == ["a_02", "b_02", "c_02"]
    for g in groups:  # 组内行号互不重复(每姿态至多 1 张的结构保证)
        assert len(set(g["pose_rows"].values())) == len(g["pose_rows"])
    # 组间姿态集合单调不增;重复数不足的姿态不会出现在更深的组
    assert set(g1["pose_rows"]) <= set(g0["pose_rows"])


# ---- 行为 5:leave_one_pose_out——80 组、验证恰 1 姿态、无随机性 ----

def test_leave_one_pose_out_structure_and_determinism():
    """行为 5:80 组,每组验证侧恰 1 个姿态、训练侧 79 个;两次调用恒等。"""
    poses = _pose_list(80)
    lopo = leave_one_pose_out(poses)
    assert len(lopo) == 80
    assert [e["val_pose"] for e in lopo] == poses  # 顺序保持首现顺序
    for e in lopo:
        assert len(e["train_poses"]) == 79
        assert e["val_pose"] not in e["train_poses"]
        assert len(set(e["train_poses"])) == 79
    assert lopo == leave_one_pose_out(poses)  # 无随机性(两次调用恒等)


def test_leave_one_pose_out_rejects_empty_and_duplicates():
    """非法输入:空清单 / 重复姿态——中文报错。"""
    with pytest.raises(ValueError, match="空"):
        leave_one_pose_out([])
    with pytest.raises(ValueError, match="重复"):
        leave_one_pose_out(["pose_a", "pose_a"])


# ---- 留一姿态抽查:sample_pose_ids(2026-10-21 quick 档 30/80 裁定)----

def test_sample_pose_ids_deterministic_subset():
    """抽查 = 种子确定性抽样:同配置两次恒等、是原子集、个数恰为 n。"""
    poses = _pose_list(80)
    a = sample_pose_ids(poses, 30, CFG)
    b = sample_pose_ids(poses, 30, CFG)
    assert a == b
    assert len(a) == 30
    assert set(a) <= set(poses)
    assert len(set(a)) == 30
    # 原名单顺序保持(读产物的人好对照)
    assert a == [p for p in poses if p in set(a)]


def test_sample_pose_ids_not_prefix_and_seed_sensitive():
    """抽样不是"只看前 30 个"(否则后 50 个姿态永远没人查);换 seed 换样本。"""
    poses = _pose_list(80)
    a = sample_pose_ids(poses, 30, CFG)
    assert set(a) != set(poses[:30])  # 前缀抽样被否决的直接证据
    cfg_other = CalibrationConfig(seed=CFG.seed + 1)
    assert set(sample_pose_ids(poses, 30, cfg_other)) != set(a)


def test_sample_pose_ids_n_ge_len_returns_all_and_zero_rejected():
    """n >= 姿态数 = 全查原样返回;n < 1 中文报错。"""
    poses = _pose_list(80)
    assert sample_pose_ids(poses, 80, CFG) == poses
    assert sample_pose_ids(poses, 200, CFG) == poses
    with pytest.raises(ValueError, match="lopo 抽查数"):
        sample_pose_ids(poses, 0, CFG)


# ---- 行为 6:_seed_for——同标签恒等、异标签互异、base_seed 变则全变 ----

def test_seed_for_identity_and_divergence():
    """行为 6:同标签恒等;不同标签互异;base_seed 变则全变。"""
    s = _seed_for("cv|round0|80", 42)
    assert s == _seed_for("cv|round0|80", 42)
    assert s != _seed_for("cv|round1|80", 42)
    assert s != _seed_for("boot|rep0|80", 42)
    other_base = _seed_for("cv|round0|80", 43)
    assert other_base != s


# ---- 辅助函数:去重姿态清单 / 行掩码 / 掩码选行 ----

def test_unique_poses_keeps_first_occurrence_order():
    """unique_poses 保持首现顺序去重(读产物的人好对照清单顺序)。"""
    obs = _make_obs({"pose_b": ["x", "y"], "pose_a": ["z"]})
    assert unique_poses(obs) == ["pose_b", "pose_a"]


def test_pose_row_mask_marks_only_selected_poses():
    """行掩码:选中姿态的全部行(连拍)为 True,其余为 False。"""
    mask = pose_row_mask(_six_pose_obs(), {"pose_1"})
    assert mask.tolist() == [False, False, True, True] + [False] * 8


def test_obs_rows_mask_shape_mismatch_chinese_error():
    """掩码形状与行数不符:中文报错(行序错位静默选错行的防线,T-04-01-02)。"""
    with pytest.raises(ValueError, match="掩码"):
        obs_rows(_six_pose_obs(), np.ones(5, dtype=bool))


# ---- 行为 7:exposures_per_pose_mask——训练侧组内抽样,不跨训练/考题边界 ----

def test_exposures_per_pose_mask_keeps_file_order_head():
    """行为 7:n=1 只留每姿态文件序第 1 张;n=2 留前 2 张;n 大于重复数全留
    (文件名故意乱序进场,取"前 N"必须按文件名排序,与分组实验同一把尺)。"""
    obs = _make_obs({
        "pose_a": ["a_03", "a_01", "a_02"],   # 行 0/1/2,乱序进场
        "pose_b": ["b_02", "b_01"],           # 行 3/4
    })
    m1 = exposures_per_pose_mask(obs, 1)
    assert m1.tolist() == [False, True, False, False, True], (
        "n=1 应恰留文件序第 1 张:a_01(行 1)与 b_01(行 4)"
    )
    m2 = exposures_per_pose_mask(obs, 2)
    assert m2.tolist() == [False, True, True, True, True], (
        "n=2 应留每姿态文件序前 2 张(a_01/a_02 与 b_01/b_02)"
    )
    m9 = exposures_per_pose_mask(obs, 9)
    assert m9.all(), "n 大于每姿态重复数时全部保留(有多少带多少)"


def test_exposures_per_pose_mask_rejects_bad_n_chinese_error():
    """行为 7:n_exposures 写 0 / true / 1.5——中文报错并点名
    evaluate.exposures_per_pose(先拒 bool 伪装,配置层与函数体双保险)。"""
    obs = _six_pose_obs()
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="exposures_per_pose"):
            exposures_per_pose_mask(obs, bad)


def test_exposures_per_pose_mask_and_pose_mask_keeps_split_disjoint():
    """行为 7(铁律消费位回归):按位与后的掩码只落在训练侧姿态内——
    组内抽样不改变姿态到训练/考题的归属;验证侧不带降载掩码仍带全部连拍。"""
    obs = _six_pose_obs()
    train_poses = {"pose_0", "pose_3", "pose_5"}
    val_poses = set(unique_poses(obs)) - train_poses
    train_mask = pose_row_mask(obs, train_poses) & exposures_per_pose_mask(obs, 1)
    val_mask = pose_row_mask(obs, val_poses)
    train_obs, val_obs = obs_rows(obs, train_mask), obs_rows(obs, val_mask)
    # 姿态级零交集:组内抽样绝不跨训练/考题边界
    assert set(train_obs.pose_ids).isdisjoint(set(val_obs.pose_ids))
    assert len(train_obs.image_ids) == 3, "训练侧 = 3 个训练姿态 × 每姿态 1 张"
    assert len(val_obs.image_ids) == 6, "验证侧 3 个考题姿态 × 2 张全带(不降载)"
