"""优化器对比(16 组合网格 + 推荐配置 + 多起点)的单元测试(小样合成数据)。

小样证明什么(真实数据 640 视图的试车在 tests/test_optimizer_realdata.py):
- optimizer.json 契约(Task 1):合法报告 save → load 逐键还原;缺键当场
  中文报错;失败契约结构键全且中文说明强制非空;私有明细键转 npz 旁车、
  json 只留引用,npz 读回强制 allow_pickle=False;
- 16 组合网格构造(Task 2,D-13/D-14):组合行键一个不少(含 init_rms 与
  f_scale_actual)、单组合失败转契约占行不中断、推荐规则"最小 RMS 胜出、
  1e-3 相对差内并列取更简核";
- 多起点扰动一致性(Task 3,D-15):扰动种子确定性(同种子重跑逐位一致)、
  双容差判定缺一不可(RMS 一致但参数位移超限也判不一致)、卡坏解照实记录。

合成观测集用 conftest 的 make_true_views(已知真值内参的合成棋盘角点,
6 姿态 × 2 张),配置对象直构造绕过 yaml(体例照 test_noise_injection.py);
并行上限写 1 走串行回退,monkeypatch 才能生效。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calib.evaluation.artifacts import (
    OPTIMIZER_KEYS,
    SCHEMA_VERSION,
    load_optimizer_report,
    make_failure_optimizer,
    save_optimizer_report,
)


# ---------------------------------------------------------------------------
# Task 1:optimizer.json 契约(合法往返 / 缺键中文报错 / 失败契约 / npz 旁车)
# ---------------------------------------------------------------------------

def _valid_report() -> dict:
    """一份键全的合法报告(数值随意,契约测试只看结构与往返一致)。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "created": "2026-10-19T00:00:00+00:00",
        "lineage": {"round_id": "run_test"},
        "recommended_solver": {"m1": {"loss": "linear"}},
        "combos": [{"name": "m1__scipy", "model": "m1", "solver": "scipy",
                    "loss": "linear", "n_views": 12, "rms_overall": 0.1,
                    "converged": True, "failure_note": None,
                    "solver_status": 1, "init_rms": 0.2,
                    "f_scale_actual": None}],
        "multistart": None,
        "meta": {"n_combos": 1, "n_failed": 0},
        "failure_note": None,
    }


def test_report_roundtrip_preserves_all_keys(tmp_path):
    """合法报告 save → load:顶层契约键逐键还原,数值逐位一致。"""
    report = _valid_report()
    lineage = {"round_id": "run_test", "created": report["created"]}
    paths = save_optimizer_report(tmp_path, report, lineage)
    assert paths["optimizer_json"] == tmp_path / "optimizer.json"
    assert paths["detail_npz"] is None  # 没给明细键就不该有 npz

    loaded = load_optimizer_report(tmp_path)
    doc = loaded["report"]
    assert loaded["detail"] is None
    for key in OPTIMIZER_KEYS:
        assert key in doc, f"读回的报告缺契约键:{key}"
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["recommended_solver"] == {"m1": {"loss": "linear"}}
    assert doc["combos"][0]["rms_overall"] == 0.1
    assert doc["combos"][0]["init_rms"] == 0.2
    assert doc["meta"]["n_combos"] == 1
    assert doc["failure_note"] is None
    # 血缘由 save 强制落位(参数传入的那份)
    assert doc["lineage"]["round_id"] == "run_test"


def test_missing_contract_key_raises_chinese(tmp_path):
    """缺契约键:写盘当场中文报错,指名缺了哪些键(不许静默落半成品)。"""
    report = _valid_report()
    del report["multistart"]
    with pytest.raises(AssertionError, match="契约键缺失"):
        save_optimizer_report(tmp_path, report, {"round_id": "run_test"})
    # 报错信息里要能直接看到缺的键名
    with pytest.raises(AssertionError, match="multistart"):
        save_optimizer_report(tmp_path / "again", report, {"round_id": "r"})


def test_missing_json_load_raises_chinese(tmp_path):
    """主 json 缺失:读回直接抛中文错,不静默返回空壳。"""
    with pytest.raises(FileNotFoundError, match="optimizer.json 不在"):
        load_optimizer_report(tmp_path)


def test_failure_optimizer_contract_complete_and_note_forced():
    """失败契约:结构键一个不少、数值字段 None、中文说明强制非空。"""
    failure = make_failure_optimizer({"round_id": "run_x"}, "网格中途失败:ValueError——初值缺失")
    for key in OPTIMIZER_KEYS:
        assert key in failure, f"失败契约缺结构键:{key}"
    assert failure["failure_note"] and "网格中途失败" in failure["failure_note"]
    assert failure["recommended_solver"] == {}
    assert failure["combos"] == []
    assert failure["multistart"] is None
    # 空说明 = 伪造证据,直接拒绝
    with pytest.raises(ValueError, match="非空的中文失败说明"):
        make_failure_optimizer({}, "   ")


def test_detail_key_goes_to_npz_sidecar_not_json(tmp_path):
    """私有明细键 _optimizer_detail:写盘转 npz 旁车 + 血缘 meta,json 只留引用。"""
    report = _valid_report()
    report["_optimizer_detail"] = {
        "per_start_rms": np.array([0.1, 0.2, 0.3]),
        "start_labels": np.array(["s0", "s1", "s2"]),
    }
    paths = save_optimizer_report(tmp_path, report, {"round_id": "run_npz"})
    assert paths["detail_npz"] == tmp_path / "optimizer_detail.npz"
    # npz 旁车与血缘 meta 都在
    assert (tmp_path / "optimizer_detail.meta.json").is_file()
    with open(tmp_path / "optimizer_detail.meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["file"] == "optimizer_detail.npz"
    assert meta["n_arrays"] == 2
    assert meta["lineage"]["round_id"] == "run_npz"
    # json 里只留引用,明细键本体绝不进 json
    assert "_optimizer_detail" not in report
    loaded = load_optimizer_report(tmp_path)
    assert loaded["report"]["meta"]["detail_npz"] == "optimizer_detail.npz"
    assert loaded["detail"] is not None
    np.testing.assert_array_equal(loaded["detail"]["per_start_rms"],
                                  [0.1, 0.2, 0.3])
    assert list(loaded["detail"]["start_labels"]) == ["s0", "s1", "s2"]


def test_numpy_scalars_are_jsonable(tmp_path):
    """numpy 标量/数组混进报告也能落盘(转原生类型),读回是纯 Python 值。"""
    report = _valid_report()
    report["combos"][0]["rms_overall"] = np.float64(0.25)
    report["combos"][0]["n_views"] = np.int64(12)
    save_optimizer_report(tmp_path, report, {"round_id": "run_json"})
    with open(tmp_path / "optimizer.json", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["combos"][0]["rms_overall"] == 0.25
    assert raw["combos"][0]["n_views"] == 12
