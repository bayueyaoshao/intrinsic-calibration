"""误差归因 analyze.json 契约的测试(05-01,体例照 test_evaluation_smoke.py)。

覆盖五类要求:契约键先验(11 键红线)、save → load 往返逐位还原、
缺键中文报错、失败契约(结构键全 + 六段 None + 中文说明强制非空)、
明细 npz 旁车(analysis/<段名>.npz + 血缘 meta + 引用断裂中文报错),
外加源码安全断言(禁 yaml.load、npz 读回只许 allow_pickle=False)。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calib.analysis.artifacts import (
    ANALYZE_DETAIL_DIR,
    ANALYZE_KEYS,
    SCHEMA_VERSION_ANALYZE,
    load_analyze_report,
    make_failure_analyze,
    save_analyze_report,
)

# 仓库根目录(测试文件在 <根>/tests/ 下)
REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def _assert_reports_equal(a, b, path="报告") -> None:
    """递归逐位比较(数组用 np.array_equal,浮点 NaN 视为相等)——
    "save → load 逐位一致"的硬断言工具(体例照 test_evaluation_experiments.py,
    测试目录无包结构不跨文件 import,本地复制一份)。"""
    if isinstance(a, dict):
        assert isinstance(b, dict), path
        assert set(a) == set(b), f"{path}: 键集合不一致 {set(a) ^ set(b)}"
        for k in a:
            _assert_reports_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_reports_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.shape == b.shape, (
            f"{path}: 数组形状不一致 {a.shape} vs {getattr(b, 'shape', None)}"
        )
        if a.dtype.kind == "f":
            assert np.array_equal(a, b, equal_nan=True), f"{path}: 数组数值不一致"
        else:
            assert np.array_equal(a, b), f"{path}: 数组数值不一致"
    elif isinstance(a, float):
        if np.isnan(a):
            assert isinstance(b, float) and np.isnan(b), f"{path}: NaN 不对称"
        else:
            assert a == b, f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _minimal_report() -> dict:
    """一份键全的最小报告(六块各自给一小段字典,契约测试只看结构)。"""
    return {
        "schema_version": SCHEMA_VERSION_ANALYZE,
        "created": "2026-10-22T00:00:00+00:00",
        "lineage": {},
        "budget": {"verdict": "pass", "n_pointwise_violations": 0},
        "residual_structure": {"n_bins": 4},
        "test_xyz": {"n_frames": 3},
        "sensitivity": {"top_direction": "k2"},
        "corner_opt": {"n_methods": 2},
        "anisotropy": {"median_ratio": 1.1},
        "meta": {},
        "failure_note": None,
    }


def _all_details() -> dict:
    """六份私有明细键(含嵌套 {模型: {数组}} 的预算块形态)。"""
    return {
        "_budget_detail": {
            "meas_rows": np.array([0.1, 0.2], dtype=np.float64),
            "bound_rows": np.array([0.01, 0.02], dtype=np.float64),
        },
        "_residual_detail": {
            "bin_centers": np.array([0.1, 0.3], dtype=np.float64),
            "bin_rms": np.array([0.3, np.nan], dtype=np.float64),
        },
        "_test_xyz_detail": {
            "dX": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "cond": np.array([10.0, 1.0e5]),
        },
        "_sensitivity_detail": {
            "m2": {
                "eigvals": np.array([0.5, 2.0]),
                "direction_px": np.array([0.01, 0.02]),
            },
        },
        "_corner_opt_detail": {
            "disp": np.array([[[0.1, 0.2]]]),
            "method": np.array(["sb0"]),
        },
        "_anisotropy_detail": {
            "sigma_u": np.array([0.01, 0.02]),
            "sigma_v": np.array([0.01, 0.01]),
        },
    }


def test_analyze_contract_has_eleven_keys():
    """11 键契约本身是红线:多一键少一键都算破坏下游消费面,直接显形。"""
    assert len(ANALYZE_KEYS) == 11
    assert SCHEMA_VERSION_ANALYZE == "analyze-v1"
    for key in ("budget", "residual_structure", "test_xyz", "sensitivity",
                "corner_opt", "anisotropy", "failure_note"):
        assert key in ANALYZE_KEYS, f"契约缺键:{key}"


def test_analyze_failure_contract_complete_and_note_forced():
    """失败契约:结构键一个不少、六块数值段一律 None、中文说明强制非空。"""
    failure = make_failure_analyze({"round_id": "run_x"}, "分析中途失败:上游产物缺失")
    for key in ANALYZE_KEYS:
        assert key in failure, f"失败契约缺结构键:{key}"
    assert failure["failure_note"] and "分析中途失败" in failure["failure_note"]
    for section in ("budget", "residual_structure", "test_xyz",
                    "sensitivity", "corner_opt", "anisotropy"):
        assert failure[section] is None, f"失败契约的 {section} 段应为 None"
    # 空说明 = 伪造证据,直接拒绝(消息点名 failure_note)
    with pytest.raises(ValueError, match="failure_note"):
        make_failure_analyze({"round_id": "r"}, "")
    with pytest.raises(ValueError, match="非空的中文失败说明"):
        make_failure_analyze({}, "   ")


def test_analyze_roundtrip_preserves_all_keys(tmp_path):
    """合法报告(带全部六份明细)save → load:顶层契约键逐键还原,数值
    逐位一致;明细数组在 npz 不在 json(json 各段只剩 detail_npz 引用)。"""
    report = _minimal_report()
    for key, arrays in _all_details().items():
        report[key] = arrays
    lineage = {"round_id": "run_test", "created": report["created"]}
    paths = save_analyze_report(tmp_path, report, lineage)

    assert paths["analyze_json"] == tmp_path / "analyze.json"
    assert set(paths["detail_npz"]) == {
        "budget", "residual_structure", "test_xyz", "sensitivity",
        "corner_opt", "anisotropy",
    }
    for section, npz_path in paths["detail_npz"].items():
        assert npz_path == tmp_path / ANALYZE_DETAIL_DIR / f"{section}.npz"
        assert npz_path.is_file()

    loaded = load_analyze_report(tmp_path)
    doc = loaded["report"]
    for key in ANALYZE_KEYS:
        assert key in doc, f"读回的报告缺契约键:{key}"
    assert doc["schema_version"] == SCHEMA_VERSION_ANALYZE
    assert doc["lineage"]["round_id"] == "run_test"
    assert doc["failure_note"] is None

    # json 各段只留引用:private 明细键被摘走,段内出现 detail_npz 指路
    for section in ("budget", "residual_structure", "test_xyz",
                    "sensitivity", "corner_opt", "anisotropy"):
        assert doc[section]["detail_npz"] == f"{ANALYZE_DETAIL_DIR}/{section}.npz"
    raw = json.loads((tmp_path / "analyze.json").read_text(encoding="utf-8"))
    assert "meas_rows" not in raw["budget"], "明细数组不许进 json(汇总只是索引)"
    assert "_budget_detail" not in raw, "私有明细键不许留在顶层"

    # 明细数组逐键还原(嵌套键拍平成 模型__数组名)
    details = loaded["details"]
    assert set(details) == {"budget", "residual_structure", "test_xyz",
                            "sensitivity", "corner_opt", "anisotropy"}
    assert np.array_equal(details["budget"]["meas_rows"], np.array([0.1, 0.2]))
    assert np.isnan(details["residual_structure"]["bin_rms"][1])
    assert np.array_equal(details["sensitivity"]["m2__eigvals"], np.array([0.5, 2.0]))
    assert np.array_equal(details["corner_opt"]["method"], np.array(["sb0"]))

    # save 后的内存 report 与读回的报告逐位一致(浮点 NaN 视为相等)
    _assert_reports_equal(report, doc)


def test_analyze_missing_contract_key_raises_chinese(tmp_path):
    """缺契约键(如 anisotropy):写盘当场中文报错,指名缺了哪些键。"""
    report = _minimal_report()
    del report["anisotropy"]
    with pytest.raises(AssertionError) as exc_info:
        save_analyze_report(tmp_path, report, {"round_id": "run_test"})
    message = str(exc_info.value)
    assert "契约键缺失" in message
    assert "anisotropy" in message
    # 再跑一次(不同目录):报错照旧,不带残缺文件落地
    with pytest.raises(AssertionError, match="契约键缺失"):
        save_analyze_report(tmp_path / "again", report, {"round_id": "r"})


def test_analyze_detail_without_dict_section_raises(tmp_path):
    """明细键有数组、但对应段不是字典:组装顺序错了当场报错(明细旁车必须
    挂在对应需求块段下,悬空明细读回的人找不到目录)。"""
    report = _minimal_report()
    report["test_xyz"] = None  # 失败段的形态:None 段不该挂明细
    report["_test_xyz_detail"] = {"dX": np.array([[1.0, 2.0, 3.0]])}
    with pytest.raises(AssertionError, match="test_xyz"):
        save_analyze_report(tmp_path, report, {"round_id": "run_test"})


def test_analyze_npz_sidecar_meta_carries_lineage(tmp_path):
    """每个明细 npz 配同名 .meta.json 且血缘必带(证据链不断)。"""
    report = _minimal_report()
    report["_budget_detail"] = {"meas_rows": np.array([0.1, 0.2])}
    save_analyze_report(tmp_path, report, {"round_id": "run_meta"})
    meta_path = tmp_path / ANALYZE_DETAIL_DIR / "budget.meta.json"
    assert meta_path.is_file()
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["file"] == f"{ANALYZE_DETAIL_DIR}/budget.npz"
    assert meta["section"] == "budget"
    assert meta["lineage"]["round_id"] == "run_meta"


def test_analyze_missing_json_load_raises_chinese(tmp_path):
    """主 json 缺失:读回直接抛中文错,不静默返回空壳。"""
    with pytest.raises(FileNotFoundError, match="analyze.json 不在"):
        load_analyze_report(tmp_path)


def test_analyze_broken_npz_reference_raises_chinese(tmp_path):
    """json 引用了明细 npz 但文件不在:读回中文报错(引用断裂 = 产物不完整,
    静默吞掉等于撒谎)。"""
    report = _minimal_report()
    report["_sensitivity_detail"] = {"eigvals": np.array([0.5, 2.0])}
    save_analyze_report(tmp_path, report, {"round_id": "run_broken"})
    (tmp_path / ANALYZE_DETAIL_DIR / "sensitivity.npz").unlink()
    with pytest.raises(FileNotFoundError) as exc_info:
        load_analyze_report(tmp_path)
    assert "明细旁车" in str(exc_info.value)
    assert "sensitivity" in str(exc_info.value)


def test_analyze_artifacts_source_safety():
    """源码安全断言(T-05-01/T-05-03):npz 读回只许 allow_pickle=False,
    绝不许 allow_pickle=True;yaml 解析归 io/config.py 管,这里不许出现
    yaml.load(配置注入的入口只留一处)。"""
    source = (REPO_ROOT / "src" / "calib" / "analysis" / "artifacts.py").read_text(
        encoding="utf-8")
    assert "yaml.load" not in source
    assert "allow_pickle=True" not in source
    assert "allow_pickle=False" in source
