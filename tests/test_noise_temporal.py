"""时域噪声底的单元测试:合成连拍 + 假传递曲线注入(NOI-01 核心)。

测的是什么:temporal_noise 的五条行为——
1. 合成连拍(已知扰动 σ)→ 逐角点 σ 回收 + 姿态级方均根池化手算逐位一致;
2. 检出帧不足 2 张的姿态自动退化为 Immerkaer+传递曲线折算,来源标注切换;
3. 抖动分离:时域方差减检测器方差,减出负数截断为零并打 clamped 标记;
4. 未检出的帧不进波动统计(抹掉 1 帧结果与 3 帧版逐位一致);
5. 全局标量两种池化(rms/median)手算对照 + 离群姿态只点名不剔除。

传递曲线用"假曲线"注入(返回已知常数的函数)——常数÷√2 就是可手算的
检测器口径 σ,折算路径有没有命中假曲线,断言一翻便知。角点 npz 由测试
直接构造(与真实 corners 产物同键名同形状),不真跑检测器。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import calib.noise.artifacts as artifacts_module
from calib.io.config import NoiseConfig
from calib.noise.artifacts import (
    IMMERKAER_NPZ_NAME,
    REPORT_KEYS,
    TEMPORAL_NPZ_NAME,
    load_noise_report,
    make_failure_report,
    save_noise_report,
)
from calib.noise.temporal import load_corners_rows, pose_groups, temporal_noise

N_POINTS = 88  # 与真实棋盘同规格(11×8 内角点)


def _truth() -> np.ndarray:
    """88×2 的固定"真值角点"布局:合成连拍里所有帧共享的角点位置。"""
    base = np.arange(N_POINTS, dtype=np.float64)
    return np.stack([base / 7.0 + 100.0, base / 11.0 + 200.0], axis=1)


def _make_records(spec: list[tuple[str, int, str]]) -> list[dict]:
    """按 [(姿态名, 帧数, 切分)] 造照片记录清单(字典形态,与 manifest 同键)。"""
    records = []
    for pose_id, n_frames, split in spec:
        for r in range(n_frames):
            records.append({
                "image_id": f"{pose_id}_r{r}",
                "path": f"input_data/{split}/{pose_id}/img{r}.jpg",
                "split": split,
                "pose_id": pose_id,
                "repeat_idx": r,
                "n_repeats": n_frames,
            })
    return records


def _write_npz(path, image_ids: list[str], uv: np.ndarray,
               detected: np.ndarray) -> str:
    """按真实 corners 产物的键名与形状写一份合成角点 npz,返回路径文字。"""
    np.savez(path,
             image_ids=np.asarray(image_ids),
             uv=np.asarray(uv, dtype=np.float64),
             detected=np.asarray(detected, dtype=bool),
             source=np.asarray(["sb"] * len(image_ids)))
    return str(path)


def _burst_npz(tmp_path, name: str, spec: list[tuple[str, int, float, int]],
               seed0: int = 100):
    """合成连拍角点 npz:spec 为 [(姿态名, 帧数, 扰动σ, 检出帧数)]。

    每帧角点 = 真值 + 固定种子高斯扰动(σ 按 spec);检出帧数不足的帧整行
    记 NaN + detected=False(与真实产物"未检出整行 NaN"同语义)。
    返回 (npz 路径, records, immerkaer_sigmas)。
    """
    truth = _truth()
    image_ids: list[str] = []
    uv_rows: list[np.ndarray] = []
    det_rows: list[bool] = []
    records: list[dict] = []
    sigmas: dict[str, dict] = {}
    seed = seed0
    for pose_id, n_frames, sigma, n_detected in spec:
        rng = np.random.default_rng(seed)
        seed += 10
        for r in range(n_frames):
            detected = r < n_detected
            # 只给检出的帧抽噪声:这样"4 帧抹掉 1 帧"与"只写 3 帧"两版里,
            # 同样那 3 张检出帧的噪声逐位一致,行为 4 的对照才成立
            if detected:
                row = truth + rng.normal(0.0, sigma, size=(N_POINTS, 2))
            else:
                row = np.full((N_POINTS, 2), np.nan)
            image_ids.append(f"{pose_id}_r{r}")
            uv_rows.append(row)
            det_rows.append(detected)
            records.append({
                "image_id": f"{pose_id}_r{r}",
                "path": f"input_data/train/{pose_id}/img{r}.jpg",
                "split": "train",
                "pose_id": pose_id,
                "repeat_idx": r,
                "n_repeats": n_frames,
            })
            sigmas[f"{pose_id}_r{r}"] = {"sigma_std": 2.0, "sigma_robust": 2.0}
    npz = _write_npz(tmp_path / f"{name}.npz", image_ids, np.stack(uv_rows),
                     det_rows)
    return npz, records, sigmas


def _fake_interp(const: float):
    """假传递曲线:不管对数 σ 是多少都返回同一常数(欧氏 RMS 口径)。

    真插值输出要再除以 √2 才是每轴 σ,所以"命中假曲线"的断言值 = const/√2。
    """
    return lambda _log_sigma: const


def _by_pose(result: dict) -> dict:
    return {p["pose_id"]: p for p in result["poses"]}


def _cfg(pooling: str = "rms", splits=("train",)) -> NoiseConfig:
    return NoiseConfig(temporal_pooling=pooling, noise_splits=splits)


# ---- 行为 1:合成连拍的 σ 回收与姿态级手算一致 ----

def test_behavior1_recovers_known_sigma_and_pose_rms_matches_hand_calc(tmp_path):
    """3 姿态 × 4 帧、已知扰动 σ=0.5:逐姿态 σ_pose/0.5 ∈ [0.8, 1.2],
    且 σ_pose == sqrt(mean(σ_pj²)) 与手算逐位一致。"""
    npz, records, sigmas = _burst_npz(
        tmp_path, "b1", [("pA", 4, 0.5, 4), ("pB", 4, 0.5, 4), ("pC", 4, 0.5, 4)])
    result = temporal_noise(npz, records, sigmas, _fake_interp(0.2), _cfg())

    assert result["n_poses_temporal"] == 3 and result["n_poses_fallback"] == 0
    for pose in result["poses"]:
        assert pose["source"] == "temporal"
        assert pose["sigma_per_corner"].shape == (N_POINTS, 2)
        assert np.isfinite(pose["sigma_per_corner"]).all()
        ratio = pose["sigma_pose"] / 0.5
        assert 0.8 <= ratio <= 1.2, \
            f"姿态 {pose['pose_id']} 的 σ 回收失败:{pose['sigma_pose']:.4f} / 0.5"
        # 手算逐位一致:方均根池化 sqrt(mean(σ_pj²))
        per_corner = pose["sigma_per_corner"]
        hand = float(np.sqrt(np.mean(per_corner ** 2)))
        assert pose["sigma_pose"] == hand
        # 正常姿态:检测器口径来自假曲线(0.2/√2),抖动 = 减法开根
        det = 0.2 / np.sqrt(2.0)
        assert pose["sigma_detector_used"] == pytest.approx(det)
        assert pose["clamped"] is False
        assert pose["sigma_jitter"] == pytest.approx(
            np.sqrt(pose["sigma_pose"] ** 2 - det ** 2), rel=1e-12)


# ---- 行为 2:检出帧不足 2 张的姿态退化为折算路径 ----

def test_behavior2_single_frame_pose_falls_back_to_transfer(tmp_path):
    """某姿态只有 1 张检出帧:source 切 immerkaer_transfer、逐角点表为 null、
    σ_pose 命中假曲线(常数/√2)。"""
    npz, records, sigmas = _burst_npz(
        tmp_path, "b2", [("pA", 4, 0.5, 4), ("pB", 4, 0.5, 1)])
    result = temporal_noise(npz, records, sigmas, _fake_interp(0.2), _cfg())
    poses = _by_pose(result)

    assert poses["pA"]["source"] == "temporal"
    assert poses["pB"]["source"] == "immerkaer_transfer"
    assert poses["pB"]["sigma_per_corner"] is None
    assert poses["pB"]["sigma_pose"] == pytest.approx(0.2 / np.sqrt(2.0))
    assert result["n_poses_temporal"] == 1 and result["n_poses_fallback"] == 1
    assert result["per_source_counts"] == {"temporal": 1, "immerkaer_transfer": 1}


# ---- 行为 3:抖动分离与 clamped 截断标记 ----

def test_behavior3_jitter_separation_with_clamped_flag(tmp_path):
    """σ_detector > σ_temporal 的姿态:σ_jitter = 0 且 clamped = true;
    正常姿态 clamped = false、抖动 = 减法开根。"""
    npz, records, sigmas = _burst_npz(
        tmp_path, "b3", [("calm", 4, 0.5, 4), ("wild", 4, 20.0, 4)])
    # 假曲线常数 10 → 检测器口径 = 10/√2 ≈ 7.07:calm(≈0.5)被压到零,wild(≈20)正常
    result = temporal_noise(npz, records, sigmas, _fake_interp(10.0), _cfg())
    poses = _by_pose(result)

    det = 10.0 / np.sqrt(2.0)
    assert poses["calm"]["clamped"] is True
    assert poses["calm"]["sigma_jitter"] == 0.0
    assert poses["wild"]["clamped"] is False
    assert poses["wild"]["sigma_jitter"] == pytest.approx(
        np.sqrt(poses["wild"]["sigma_pose"] ** 2 - det ** 2), rel=1e-12)
    assert poses["calm"]["sigma_detector_used"] == pytest.approx(det)
    assert poses["wild"]["sigma_detector_used"] == pytest.approx(det)


# ---- 行为 4:未检出帧不进波动统计 ----

def test_behavior4_undetected_frames_excluded_bitwise(tmp_path):
    """同一姿态 4 帧里抹掉 1 帧(整行 NaN + detected=False):
    逐角点 σ 表与"只写那 3 张检出帧"的版本逐位一致。"""
    # 版本 A:4 帧全检出(种子 7,前 3 帧的噪声与版本 B 的检出帧相同)
    npz_a, records_a, sigmas_a = _burst_npz(
        tmp_path, "b4a", [("pX", 4, 0.8, 4)], seed0=7)
    # 版本 B:同种子同 4 帧,但第 1 帧标未检出(检出帧数 = 3)
    npz_b, records_b, sigmas_b = _burst_npz(
        tmp_path, "b4b", [("pX", 4, 0.8, 3)], seed0=7)
    # 版本 C:只有 3 帧、全检出(与 B 的 3 张检出帧同种子同噪声)
    npz_c, records_c, sigmas_c = _burst_npz(
        tmp_path, "b4c", [("pX", 3, 0.8, 3)], seed0=7)

    ra = temporal_noise(npz_a, records_a, sigmas_a, _fake_interp(0.2), _cfg())
    rb = temporal_noise(npz_b, records_b, sigmas_b, _fake_interp(0.2), _cfg())
    rc = temporal_noise(npz_c, records_c, sigmas_c, _fake_interp(0.2), _cfg())
    pa = _by_pose(ra)["pX"]
    pb = _by_pose(rb)["pX"]
    pc = _by_pose(rc)["pX"]

    # 抹掉的帧绝不进波动统计:4 帧版(剔除 1 帧)与 3 帧版逐位一致
    np.testing.assert_array_equal(pb["sigma_per_corner"], pc["sigma_per_corner"])
    assert pb["sigma_pose"] == pc["sigma_pose"]
    assert pa["n_detected"] == 4 and pb["n_detected"] == 3


# ---- 行为 5:全局两种池化 + 离群姿态只点名 ----

def test_behavior5_global_pooling_rms_median_and_outliers(tmp_path):
    """rms 池化 = sqrt(mean(σ_pose²)) 手算对照;median 池化 = 中位数;
    σ_pose 最大的姿态进 outlier_poses 且数值不被剔除。"""
    spec = [("q1", 4, 0.5, 4), ("q2", 4, 0.5, 4), ("q3", 4, 0.5, 4),
            ("q4", 4, 0.5, 4), ("qOUT", 4, 30.0, 4)]
    npz, records, sigmas = _burst_npz(tmp_path, "b5", spec)

    rms_result = temporal_noise(npz, records, sigmas, _fake_interp(0.2),
                                _cfg("rms"))
    pose_sigmas = np.asarray([p["sigma_pose"] for p in rms_result["poses"]])
    assert rms_result["sigma_temporal_corner"] == float(
        np.sqrt(np.mean(pose_sigmas ** 2)))
    assert rms_result["sigma_temporal_rms"] == rms_result["sigma_temporal_corner"]
    assert rms_result["sigma_temporal_median"] == float(np.median(pose_sigmas))

    med_result = temporal_noise(npz, records, sigmas, _fake_interp(0.2),
                                _cfg("median"))
    assert med_result["sigma_temporal_corner"] == float(np.median(pose_sigmas))

    # 离群姿态:排第一、只点名、数值照常在 poses 里(不剔除)
    outliers = rms_result["outlier_poses"]
    assert outliers[0]["pose_id"] == "qOUT"
    assert 1 < len(outliers) <= 3
    by_pose = _by_pose(rms_result)
    assert outliers[0]["sigma_pose"] == by_pose["qOUT"]["sigma_pose"]
    assert rms_result["n_poses_temporal"] == 5
    assert any(p["pose_id"] == "qOUT" for p in rms_result["poses"])


# ---- 双切分:test 独立照逐张走单帧路径 ----

def test_test_split_images_get_per_image_transfer_entries(tmp_path):
    """cfg.noise_splits 含 test 时:test 每张独立照一条折算记录,
    source 全为 immerkaer_transfer、每轴 σ 命中假曲线。"""
    npz, records, sigmas = _burst_npz(tmp_path, "tt", [("pA", 4, 0.5, 4)])
    for k, robust_sigma in enumerate((3.0, 2.5)):
        image_id = f"solo_t{k}"
        records.append({"image_id": image_id, "path": f"x/{image_id}.jpg",
                        "split": "test", "pose_id": image_id,
                        "repeat_idx": 0, "n_repeats": 1})
        # 主线口径默认 robust:两档不同的稳健值用于断言"取的正是主线口径"
        sigmas[image_id] = {"sigma_std": 99.0, "sigma_robust": robust_sigma}

    result = temporal_noise(npz, records, sigmas, _fake_interp(1.0),
                            _cfg(splits=("train", "test")))
    per_image = result["splits"]["test"]["per_image"]
    assert [e["image_id"] for e in per_image] == ["solo_t0", "solo_t1"]
    for entry, main_sigma in zip(per_image, (3.0, 2.5)):
        assert entry["source"] == "immerkaer_transfer"
        assert entry["sigma_immerkaer"] == main_sigma
        assert entry["sigma_corner_per_axis"] == pytest.approx(1.0 / np.sqrt(2.0))


# ---- 辅助函数:pose_groups 与 load_corners_rows ----

def test_pose_groups_groups_train_only_and_accepts_dicts_or_objects():
    """只聚 train 切分;字典与对象两种记录形态都认。"""
    records = [
        {"image_id": "a0", "split": "train", "pose_id": "P1"},
        {"image_id": "b0", "split": "train", "pose_id": "P2"},
        {"image_id": "a1", "split": "train", "pose_id": "P1"},
        {"image_id": "t0", "split": "test", "pose_id": "t0"},
    ]
    groups = pose_groups(records)
    assert groups == {"P1": [0, 2], "P2": [1]}

    obj_records = [SimpleNamespace(split="train", pose_id="P9"),
                   SimpleNamespace(split="test", pose_id="x")]
    assert pose_groups(obj_records) == {"P9": [0]}


def test_load_corners_rows_roundtrip_and_shape_guards(tmp_path):
    """读回键与形状齐全;形状不对给中文报错;含 pickle 对象的 npz 拒载。"""
    image_ids = ["i0", "i1"]
    uv = np.stack([_truth(), _truth() + 1.0])
    detected = np.array([True, False])
    npz = _write_npz(tmp_path / "ok.npz", image_ids, uv, detected)

    rows = load_corners_rows(npz)
    assert rows["image_ids"] == image_ids
    assert rows["uv"].shape == (2, N_POINTS, 2)
    assert rows["detected"].dtype == bool
    assert rows["source"] == ["sb", "sb"]

    # 形状不合:uv 不是 (N, 点数, 2) → 中文 AssertionError 点名
    bad = tmp_path / "bad.npz"
    np.savez(bad, image_ids=np.asarray(image_ids),
             uv=np.zeros((2, 4), dtype=np.float64),
             detected=detected, source=np.asarray(["sb", "sb"]))
    with pytest.raises(AssertionError, match="形状"):
        load_corners_rows(str(bad))

    # 行数互相矛盾:image_ids 两条、uv 一行 → 中文报错
    ragged = tmp_path / "ragged.npz"
    np.savez(ragged, image_ids=np.asarray(image_ids),
             uv=np.zeros((1, N_POINTS, 2), dtype=np.float64),
             detected=np.array([True]), source=np.asarray(["sb"]))
    with pytest.raises(AssertionError):
        load_corners_rows(str(ragged))

    # pickle 防线:对象数组一律拒载(allow_pickle=False)
    obj = tmp_path / "obj.npz"
    np.savez(obj, image_ids=np.asarray([{"k": 1}], dtype=object),
             uv=np.zeros((1, N_POINTS, 2)), detected=np.array([True]),
             source=np.asarray(["sb"]))
    with pytest.raises(ValueError):
        load_corners_rows(str(obj))


def test_temporal_noise_missing_immerkaer_sigma_raises(tmp_path):
    """某张照片没有 Immerkaer 估计:中文报错点名照片,不静默跳过。"""
    npz, records, sigmas = _burst_npz(tmp_path, "mm", [("pA", 4, 0.5, 4)])
    del sigmas["pA_r2"]
    with pytest.raises(ValueError, match="pA_r2"):
        temporal_noise(npz, records, sigmas, _fake_interp(0.2), _cfg())


def test_temporal_noise_npz_row_without_train_record_raises(tmp_path):
    """角点行在照片清单里找不到(多半串了轮):中文报错点名,不硬算。"""
    npz, records, sigmas = _burst_npz(tmp_path, "mm2", [("pA", 4, 0.5, 4)])
    shortlist = [r for r in records if r["image_id"] != "pA_r2"]
    with pytest.raises(ValueError, match="pA_r2"):
        temporal_noise(npz, shortlist, sigmas, _fake_interp(0.2), _cfg())


# ---- 产物契约:noise.json 落盘 / 读回 / 失败报告 ----

def _sample_report() -> dict:
    """一份最小但结构齐全的成功报告(带两个私有明细键)。"""
    return {
        "schema_version": "noise-v1",
        "created": "2026-10-18T00:00:00Z",
        "lineage": {},
        "sigma_detector_corner": 0.12,
        "sigma_temporal_corner": 0.34,
        "n_poses_temporal": 1,
        "n_poses_fallback": 1,
        "per_source_counts": {"temporal": 1, "immerkaer_transfer": 1},
        "outlier_poses": [{"pose_id": "pB", "sigma_pose": 0.5}],
        "splits": {"train": {"n_poses": 2},
                   "test": {"n_images": 1, "per_image": []}},
        "transfer_curve": {"path": "transfer_curve.json"},
        "injection": {"n_images": 2},
        "meta": {},
        "failure_note": None,
        "_temporal_detail": {
            "pose_ids": ["pA", "pB"],
            "sigma_per_corner_all": np.zeros((2, 3, 2)),
            "sources": ["temporal", "immerkaer_transfer"],
            "sigma_pose": np.asarray([0.1, 0.5]),
            "sigma_detector": np.asarray([0.1, 0.1]),
            "sigma_jitter": np.asarray([0.0, 0.0]),
            "clamped": np.asarray([False, True]),
            "n_frames": np.asarray([4, 1]),
            "n_detected": np.asarray([4, 1]),
        },
        "_immerkaer_detail": {
            "image_ids": ["i0", "i1"],
            "sigma_std": np.asarray([1.0, 2.0]),
            "sigma_robust": np.asarray([1.5, 2.5]),
            "split": ["train", "test"],
        },
    }


def test_save_and_load_roundtrip_contract(tmp_path):
    """save → load 往返:契约键一个不少,明细 npz 键名/形状对,血缘旁车在;
    私有明细键绝不进 json,只留路径引用(汇总是索引不是替代,D-13)。"""
    lineage = {"created": "2026-10-18T00:00:00Z", "round_id": "run_x"}
    paths = save_noise_report(tmp_path, _sample_report(), lineage)
    assert paths["noise_json"] == tmp_path / "noise.json"
    assert paths["temporal_npz"] == tmp_path / TEMPORAL_NPZ_NAME
    assert paths["immerkaer_npz"] == tmp_path / IMMERKAER_NPZ_NAME

    loaded = load_noise_report(tmp_path)
    for key in REPORT_KEYS:
        assert key in loaded["report"], f"缺契约键:{key}"
    assert loaded["report"]["splits"]["train"]["detail_npz"] == TEMPORAL_NPZ_NAME
    assert loaded["report"]["meta"]["immerkaer_detail_npz"] == IMMERKAER_NPZ_NAME
    assert loaded["report"]["lineage"] == lineage
    assert loaded["temporal"]["sigma_per_corner_all"].shape == (2, 3, 2)
    assert loaded["temporal"]["clamped"].dtype == bool
    assert loaded["immerkaer"]["image_ids"].tolist() == ["i0", "i1"]
    assert loaded["immerkaer"]["sigma_robust"].tolist() == [1.5, 2.5]
    # 血缘旁车:每份明细 npz 的 meta.json 带同一份指纹(产物来源可考,T-03-03-03)
    for name in (TEMPORAL_NPZ_NAME, IMMERKAER_NPZ_NAME):
        meta = json.loads(
            (tmp_path / f"{Path(name).stem}.meta.json").read_text(encoding="utf-8"))
        assert meta["lineage"] == lineage


def test_make_failure_report_structure_and_empty_note_rejected(tmp_path):
    """失败契约:结构键全在、数值字段一律 None、空说明当场拒;
    失败报告照常落盘(没有明细,npz 路径为 None)。"""
    failure = make_failure_report(
        {"created": "t", "round_id": "r"}, "注入实验失败:读不到角点文件")
    for key in REPORT_KEYS:
        assert key in failure
    assert failure["sigma_detector_corner"] is None
    assert failure["sigma_temporal_corner"] is None
    assert failure["n_poses_temporal"] is None
    assert failure["failure_note"].strip()
    assert failure["per_source_counts"] == {} and failure["outlier_poses"] == []

    paths = save_noise_report(tmp_path / "fail", failure,
                              {"created": "t", "round_id": "r"})
    assert paths["temporal_npz"] is None and paths["immerkaer_npz"] is None
    reloaded = load_noise_report(tmp_path / "fail")
    assert reloaded["temporal"] is None and reloaded["immerkaer"] is None

    with pytest.raises(ValueError, match="failure_note"):
        make_failure_report({}, "   ")


def test_save_noise_report_missing_contract_keys_raises(tmp_path):
    """顶层契约键缺失:写盘前当场中文报错,不许残缺的 json 落盘。"""
    with pytest.raises(AssertionError, match="契约键缺失"):
        save_noise_report(tmp_path / "bad",
                          {"schema_version": "noise-v1"}, {})


def test_artifacts_module_has_no_cli_dependency():
    """库层不依赖 CLI 层:artifacts.py 源码里不许出现 typer(白话:算数的
    模块不许反过来管命令行打印,依赖只能单向)。"""
    source = Path(artifacts_module.__file__).read_text(encoding="utf-8")
    assert "typer" not in source
