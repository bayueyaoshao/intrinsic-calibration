"""评估交付(evaluate.json 主契约 + 合成数据全链冒烟)的测试。

本文件两段(体例照 test_optimizer_comparison.py 与 test_evaluation_experiments.py):
- 契约单元段(Task 2):evaluate.json 13 键写盘前校验、save → load 往返
  逐键还原、明细数组只进 experiments/*.npz 旁车(json 只留 detail_npz
  引用)、失败契约结构键全、npz 旁车 meta 带血缘;
- 全链冒烟段(Task 3):合成 6 姿态小样一条龙(优化器小网格 → 四实验 →
  全量重训交付 → 来源分离组装 → 判行表与选型 → 契约落盘 → 三图),
  04-06 的命令层只是把这里的调用序列接上取数与中文播报。

合成观测集用 conftest 的 make_true_views(已知真值内参的合成棋盘角点),
配置对象直构造绕过 yaml;并行上限写 1 走串行回退,monkeypatch 才能生效。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calib.evaluation.artifacts import (
    EVALUATE_KEYS,
    SCHEMA_VERSION_EVALUATE,
    load_evaluate_report,
    make_failure_evaluate,
    save_evaluate_report,
)

# ---------------------------------------------------------------------------
# Task 2:evaluate.json 契约(合法往返 / 缺键中文报错 / 失败契约 / npz 旁车)
# ---------------------------------------------------------------------------


def _minimal_report() -> dict:
    """一份键全的最小报告(各段都是字典,数值随意,契约测试只看结构)。"""
    return {
        "schema_version": SCHEMA_VERSION_EVALUATE,
        "created": "2026-10-19T00:00:00+00:00",
        "lineage": {},
        "optimizer": {"recommended_solver": {"m1": {"loss": "linear"}}},
        "cv": {"models": {"m1": {"mean_heldout_px": 0.1}}},
        "bootstrap": {"models": {"m1": {"max_disp_px": 0.01}}},
        "slices": {"models": {"m1": {"slice_max_disp_px": 0.02}}},
        "lopo": {"models": {"m1": {"mean_lopo_px": 0.12}}},
        "source_separation": {"m1": {"fx": {"pose_share": 0.5}}},
        "selection": {"selected": "m1"},
        "delivery": {"models": {"m1": {"rms_px": 0.1}}},
        "meta": {},
        "failure_note": None,
    }


def _all_details() -> dict:
    """五份私有明细键(含嵌套 {模型: {数组}} 的分组/bootstrap 形态)。"""
    return {
        "_cv_detail": {
            "assignment_round": np.array([0, 0], dtype=np.int64),
            "detail_rms": np.array([0.1, np.nan], dtype=np.float64),
            "detail_pose": np.array(["pose_00", "pose_01"]),
        },
        "_bootstrap_detail": {
            "m1": {
                "rep": np.array([0, 1], dtype=np.int64),
                "params": np.array([[1.0, 2.0], [3.0, 4.0]]),
            },
        },
        "_slice_detail": {
            "m1": {
                "group_index": np.array([0, 1], dtype=np.int64),
                "sigma_used": 0.30,
            },
        },
        "_lopo_detail": {
            "detail_val_pose": np.array(["pose_00", "pose_01"]),
            "detail_rms": np.array([0.2, 0.3], dtype=np.float64),
        },
        "_delivery_detail": {
            "test_model": np.array(["m1", "m1"]),
            "test_image": np.array(["timg_00", "timg_01"]),
            "test_rms": np.array([0.15, 0.16], dtype=np.float64),
        },
    }


def test_evaluate_contract_has_thirteen_keys():
    """13 键契约本身是红线:多一键少一键都算破坏下游消费面,直接显形。"""
    assert len(EVALUATE_KEYS) == 13
    assert SCHEMA_VERSION_EVALUATE == "evaluate-v1"


def test_evaluate_roundtrip_preserves_all_keys(tmp_path):
    """合法报告(带全部五份明细)save → load:顶层契约键逐键还原,数值
    逐位一致;明细数组在 npz 不在 json(json 各段只剩 detail_npz 引用)。"""
    report = _minimal_report()
    for key, arrays in _all_details().items():
        report[key] = arrays
    lineage = {"round_id": "run_test", "created": report["created"]}
    paths = save_evaluate_report(tmp_path, report, lineage)

    assert paths["evaluate_json"] == tmp_path / "evaluate.json"
    assert set(paths["detail_npz"]) == {"cv", "bootstrap", "slices", "lopo", "delivery"}
    for section, npz_path in paths["detail_npz"].items():
        assert npz_path == tmp_path / "experiments" / f"{section}.npz"
        assert npz_path.is_file()

    loaded = load_evaluate_report(tmp_path)
    doc = loaded["report"]
    for key in EVALUATE_KEYS:
        assert key in doc, f"读回的报告缺契约键:{key}"
    assert doc["schema_version"] == SCHEMA_VERSION_EVALUATE
    assert doc["lineage"]["round_id"] == "run_test"
    assert doc["selection"] == {"selected": "m1"}
    assert doc["failure_note"] is None

    # json 各段只留引用:private 明细键被摘走,段内出现 detail_npz 指路
    for section in ("cv", "bootstrap", "slices", "lopo", "delivery"):
        assert doc[section]["detail_npz"] == f"experiments/{section}.npz"
    raw = json.loads((tmp_path / "evaluate.json").read_text(encoding="utf-8"))
    assert "assignment_round" not in raw["cv"], "明细数组不许进 json(汇总只是索引)"
    assert "_cv_detail" not in raw, "私有明细键不许留在顶层"

    # 明细数组逐键还原(嵌套键拍平成 模型__数组名)
    details = loaded["details"]
    assert set(details) == {"cv", "bootstrap", "slices", "lopo", "delivery"}
    assert np.array_equal(details["cv"]["detail_rms"][0], np.float64(0.1))
    assert np.isnan(details["cv"]["detail_rms"][1])
    assert np.array_equal(details["bootstrap"]["m1__params"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert details["slices"]["m1__sigma_used"] == np.float64(0.30)
    assert np.array_equal(details["delivery"]["test_image"], np.array(["timg_00", "timg_01"]))


def test_evaluate_missing_contract_key_raises_chinese(tmp_path):
    """缺契约键:写盘当场中文报错,指名缺了哪些键(不许静默落半成品)。"""
    report = _minimal_report()
    del report["selection"]
    with pytest.raises(AssertionError, match="契约键缺失"):
        save_evaluate_report(tmp_path, report, {"round_id": "run_test"})
    with pytest.raises(AssertionError, match="selection"):
        save_evaluate_report(tmp_path / "again", report, {"round_id": "r"})


def test_evaluate_detail_without_dict_section_raises(tmp_path):
    """明细键有数组、但对应段不是字典:组装顺序错了当场报错(明细旁车必须
    挂在实验段下,悬空明细读回的人找不到目录)。"""
    report = _minimal_report()
    report["lopo"] = None  # 失败段的形态:None 段不该挂明细
    report["_lopo_detail"] = {"detail_rms": np.array([0.1])}
    with pytest.raises(AssertionError, match="lopo"):
        save_evaluate_report(tmp_path, report, {"round_id": "run_test"})


def test_evaluate_failure_contract_complete_and_note_forced():
    """失败契约:结构键一个不少、各实验段 None、中文说明强制非空。"""
    failure = make_failure_evaluate({"round_id": "run_x"}, "评估中途失败:ValueError——上游缺失")
    for key in EVALUATE_KEYS:
        assert key in failure, f"失败契约缺结构键:{key}"
    assert failure["failure_note"] and "评估中途失败" in failure["failure_note"]
    for section in (
        "optimizer",
        "cv",
        "bootstrap",
        "slices",
        "lopo",
        "source_separation",
        "selection",
        "delivery",
    ):
        assert failure[section] is None, f"失败契约的 {section} 段应为 None"
    # 空说明 = 伪造证据,直接拒绝
    with pytest.raises(ValueError, match="非空的中文失败说明"):
        make_failure_evaluate({}, "   ")


def test_evaluate_npz_sidecar_meta_carries_lineage(tmp_path):
    """每个明细 npz 配同名 .meta.json 且血缘必带(证据链不断,T-04-05-04)。"""
    report = _minimal_report()
    report["_cv_detail"] = {"detail_rms": np.array([0.1, 0.2])}
    save_evaluate_report(tmp_path, report, {"round_id": "run_meta"})
    meta_path = tmp_path / "experiments" / "cv.meta.json"
    assert meta_path.is_file()
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["file"] == "experiments/cv.npz"
    assert meta["section"] == "cv"
    assert meta["lineage"]["round_id"] == "run_meta"


def test_evaluate_missing_json_load_raises_chinese(tmp_path):
    """主 json 缺失:读回直接抛中文错,不静默返回空壳。"""
    with pytest.raises(FileNotFoundError, match="evaluate.json 不在"):
        load_evaluate_report(tmp_path)


def test_evaluate_broken_npz_reference_raises_chinese(tmp_path):
    """json 引用了明细 npz 但文件不在:读回中文报错(引用断裂 = 产物不完整,
    静默吞掉等于撒谎)。"""
    report = _minimal_report()
    report["_cv_detail"] = {"detail_rms": np.array([0.1])}
    save_evaluate_report(tmp_path, report, {"round_id": "run_broken"})
    (tmp_path / "experiments" / "cv.npz").unlink()
    with pytest.raises(FileNotFoundError, match="明细旁车"):
        load_evaluate_report(tmp_path)
