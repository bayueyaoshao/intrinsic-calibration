"""真实数据误差归因对账:`calib analyze` 首跑 × ANA-01(findings 结论 4 数字对账)。

整文件标记 integration(需要真实 input_data 与 corners/noise/calibration/
fisher/evaluate 上游产物,日常开发用 `uv run pytest -m "not integration"`
自动跳过;前置产物缺失或血缘对不上时整文件跳过并写中文理由,不误报失败)。

测的是什么(对应 PLAN 05-02 Task 3;数字基准 = docs/findings.md 结论 4,
出自 evaluate run_20261021_131613 的 slices.models.m2.crlb_alert):
1. 取数体例(先于一切断言):最新 output/analyze/run_*/analyze.json 的
   血缘指纹(配置哈希 + 全部上游文件哈希)与当前上游产物一致才消费;
   对不上 pytest.skip,中文理由「产物轮次已更新,请重跑 calib analyze」
   (analyze 是纯消费的分钟级命令,不重算、不盲目复用串轮产物;
   _latest_analyze_or_skip 供后续计划的分析段测试扩展复用)。
2. 预算复核:recheck.consistent 为真(独立重算与 evaluate 落账逐位对账);
   逐行核查零违例(实测低于理论下界物理上不可能,CRLB 口径铁律)。
3. findings 结论 4 对账:硬告警线与 evaluate.json crlb_alert 逐位一致
   (约 0.4207);三统计量比值在基准(约 102 / 1108 / 1444 倍)±5% 内。
4. 判读:逐行最小比值约 2.8(±0.1);verdict == "pass"。
另:明细旁车 analysis/budget.npz(逐行底账 14080 行)与证据图
figures/budget_bars.png 在盘;CLI 播报层由单元测试覆盖,本文件不重复。

体例照 tests/test_evaluation_realdata.py:模块级夹具只取一轮产物,全部
断言共享。前置:先跑一条 `uv run calib analyze`(分钟级)产生数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from calib.analysis.artifacts import ANALYZE_KEYS, load_analyze_report

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZE_DIR = REPO_ROOT / "output" / "analyze"

# findings 结论 4 的数字基准(evaluate run_20261021_131613 落账;
# 三统计量 = 中位 / 95 分位 / 最大,实测是同口径下界的倍数)
_BASE_RATIOS = (101.77, 1107.60, 1443.84)
_STAT_KEYS = ("重投影位移中位", "重投影位移95分位", "重投影位移最大")
_HARD_LINE_BASE = 0.4207
_MIN_RATIO_BASE = 2.8  # 逐行最小倍数(结论 4:实测全部 ≥ 2.8× 下界、零违例)
_N_ROWS = 14080  # 逐行底账行数 = 80 视图 × 88 角点 × 2 分量


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analyze_runs_newest_first() -> list[Path]:
    if not ANALYZE_DIR.is_dir():
        return []
    return sorted(
        (p for p in ANALYZE_DIR.iterdir()
         if p.is_dir() and p.name.startswith("run_")),
        reverse=True,
    )


def _fingerprint_problem(run_dir: Path) -> str | None:
    """检查一轮 analyze 产物能不能消费:返回 None 表示血缘指纹完全吻合。

    消费条件(从严,体例照 test_evaluation_realdata 的同名助手):analyze.json
    在、非失败轮、配置哈希一致、且登记的**每一个**上游文件都还在且内容
    逐位一致——上游任何一个重跑过,旧轮结论就不能冒充当前数据的结论。
    """
    json_path = run_dir / "analyze.json"
    if not json_path.is_file():
        return "缺 analyze.json"
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("failure_note"):
        return "失败轮(分析段为 null)不可消费"
    lineage = doc.get("lineage") or {}
    for name, sha in (lineage.get("config_sha256") or {}).items():
        path = REPO_ROOT / "config" / name
        if not path.is_file() or _sha256(path) != sha:
            return f"配置 {name} 与当前不一致"
    for name, entry in (lineage.get("upstream") or {}).items():
        path = Path(entry.get("path", ""))
        if not path.is_file():
            return f"上游 {name} 文件不在:{path}"
        if _sha256(path) != entry.get("sha256"):
            return f"上游 {name} 内容已变(上游重跑过,旧轮不能消费)"
    return None


def _latest_analyze_or_skip() -> dict:
    """共享取数助手:最新 analyze 轮指纹对得上才消费,对不上中文跳过。

    返回 {"report": analyze.json 字典, "details": {段名: 明细数组},
    "run_dir": 轮目录}。前置产物缺失或指纹对不上都 pytest.skip(中文
    理由),绝不误报失败——供后续计划的分析段测试扩展复用。
    """
    if not (REPO_ROOT / "input_data" / "train").is_dir():
        pytest.skip("真实 input_data 不在,跳过误差归因集成测试")
    for stage in ("corners", "noise", "calibration", "fisher", "evaluate"):
        if not (REPO_ROOT / "output" / stage).is_dir():
            pytest.skip(f"还没有 output/{stage} 产物,先跑对应上游命令再来")
    if not ANALYZE_DIR.is_dir():
        pytest.skip("还没有 output/analyze 产物,先跑 calib analyze 再来")

    runs = _analyze_runs_newest_first()
    if not runs:
        pytest.skip("output/analyze 下没有任何 run_<时间戳> 轮,请重跑 calib analyze")
    problem = _fingerprint_problem(runs[0])
    if problem is not None:
        pytest.skip(f"产物轮次已更新,请重跑 calib analyze({problem})")

    loaded = load_analyze_report(runs[0])
    report = loaded["report"]
    for key in ANALYZE_KEYS:
        assert key in report, f"analyze.json 缺契约键:{key}"
    assert report["failure_note"] is None, \
        f"本轮是失败轮:{report['failure_note']}"
    print(f"\n[取数] 血缘指纹吻合,消费最新轮:{runs[0].name}")
    return {"report": report, "details": loaded["details"], "run_dir": runs[0]}


@pytest.fixture(scope="module")
def analyze_tree() -> dict:
    return _latest_analyze_or_skip()


def test_analyze_sections_shape_and_selected_model(analyze_tree: dict):
    """契约形状:budget 段有值,其余五段显式缺位(None)并进
    meta.missing_sections;目标模型与 evaluate 选型结论一致(m2)。"""
    report = analyze_tree["report"]
    budget = report["budget"]
    assert isinstance(budget, dict) and budget, "budget 段缺席(误差归因没跑成)"
    for key in ("residual_structure", "test_xyz", "sensitivity",
                "corner_opt", "anisotropy"):
        assert report[key] is None, \
            f"{key} 段应由后续计划接入,当前必须显式为 None"
    assert set(report["meta"]["missing_sections"]) == {
        "residual_structure", "test_xyz", "sensitivity", "corner_opt", "anisotropy"}
    assert report["meta"]["selected_model"] == "m2", \
        f"目标模型应与 evaluate 选型结论一致,实际 {report['meta']['selected_model']!r}"
    assert budget["schema"] == "analyze-budget-v1"
    assert budget["sigma_detector_corner"] == pytest.approx(0.008708, abs=1e-4), \
        "噪声口径必须是检测器级 σ(CRLB 口径铁律,时域口径会虚高约 35 倍)"


def test_analyze_budget_recheck_consistent_and_zero_violations(analyze_tree: dict):
    """复核与铁律:独立重算与 evaluate 落账逐位对账一致;逐行核查零违例
    (实测低于理论下界物理上不可能,出现即口径错配 bug)。"""
    budget = analyze_tree["report"]["budget"]
    assert budget["recheck"]["consistent"] is True, \
        "独立重算与 evaluate 落账的统计量比值对不上 = 口径漂移"
    assert budget["n_pointwise_violations"] == 0, \
        f"逐行核查出现 {budget['n_pointwise_violations']} 个违例" \
        "(实测低于理论下界,物理上不可能)"
    print(f"\n[复核] 三统计量对账一致;逐行零违例,"
          f"最小比值 {budget['min_ratio_pointwise']:.3f}")


def test_analyze_budget_lines_and_ratios_match_findings(analyze_tree: dict):
    """findings 结论 4 对账:硬告警线与 evaluate.json crlb_alert 逐位一致
    (约 0.4207);三统计量比值在基准(约 102 / 1108 / 1444 倍)±5% 内。"""
    report = analyze_tree["report"]
    budget = report["budget"]
    # evaluate.json 从血缘登记的路径取(与 analyze 同一轮的原件,不信转抄)
    evaluate_path = Path(report["lineage"]["upstream"]["evaluate.json"]["path"])
    with open(evaluate_path, encoding="utf-8") as f:
        evaluate_doc = json.load(f)
    alert = evaluate_doc["slices"]["models"]["m2"]["crlb_alert"]

    hard = float(budget["lines"]["hard"])
    assert hard == float(alert["lines"]["hard"]), \
        "budget 硬告警线与 evaluate 落账不一致(判定线必须同一条)"
    assert abs(hard - _HARD_LINE_BASE) < 1e-3, \
        f"硬告警线 {hard} 偏离结论 4 基准 {_HARD_LINE_BASE}"

    ratios = budget["recheck"]["ratios"]
    for name, ours, base in zip(_STAT_KEYS, _BASE_RATIOS,
                                (alert["per_param"][k]["ratio"] for k in _STAT_KEYS)):
        assert ratios[name] == pytest.approx(ours, rel=0.05), \
            f"{name} 比值 {ratios[name]:.1f} 偏离结论 4 基准 {ours} 超过 5%"
        assert float(base) == pytest.approx(ours, rel=0.05), \
            f"{name}:evaluate 落账比值 {base} 与结论 4 基准 {ours} 对不上"
    print("\n[结论 4] 三统计量倍数:"
          + ", ".join(f"{k}={budget['recheck']['ratios'][k]:.1f}" for k in _STAT_KEYS)
          + f";硬线 {hard:.4f}")


def test_analyze_budget_min_ratio_and_verdict(analyze_tree: dict):
    """判读:逐行最小实测/下界比约 2.8(±0.1,结论 4「实测全部 ≥ 2.8×
    下界」);预算判定 pass。"""
    budget = analyze_tree["report"]["budget"]
    assert budget["min_ratio_pointwise"] == pytest.approx(_MIN_RATIO_BASE, abs=0.1), \
        f"逐行最小比值 {budget['min_ratio_pointwise']:.3f} 偏离结论 4 基准 " \
        f"{_MIN_RATIO_BASE}(±0.1)"
    assert budget["verdict"] == "pass"
    # 与 evaluate 落账的同名统计量逐位一致(同一组底账的两次重算)
    evaluate_path = Path(analyze_tree["report"]["lineage"]["upstream"]["evaluate.json"]["path"])
    with open(evaluate_path, encoding="utf-8") as f:
        alert = json.load(f)["slices"]["models"]["m2"]["crlb_alert"]
    assert budget["min_ratio_pointwise"] == pytest.approx(
        float(alert["min_ratio_pointwise"]), rel=1e-9)
    print(f"\n[判读] verdict={budget['verdict']},"
          f"逐行最小比值 {budget['min_ratio_pointwise']:.3f}(基准 2.8)")


def test_analyze_budget_sidecars_and_figure(analyze_tree: dict):
    """产物保真:逐行底账 npz(ratios_rows 14080 行)+ 证据图 PNG 在盘;
    预算行 9 参数齐,利用率比值只有 fx/fy/cx/cy 有值(k1–k3/p1p2 置 None,
    findings 结论 4 口径)。"""
    report = analyze_tree["report"]
    run_dir = analyze_tree["run_dir"]
    detail = analyze_tree["details"]["budget"]
    assert detail["ratios_rows"].shape == (_N_ROWS,), \
        f"逐行底账应 {_N_ROWS} 行(80 视图 × 88 角点 × 2),实际 {detail['ratios_rows'].shape}"
    assert (run_dir / "analysis" / "budget.npz").is_file()
    assert (run_dir / "figures" / "budget_bars.png").is_file()

    rows = report["budget"]["rows"]
    assert len(rows) == 9
    for row in rows:
        for key in ("empirical_std", "empirical_std_px", "bootstrap_std",
                    "pose_sampling_std", "crlb_se"):
            assert row[key] is not None and np.isfinite(row[key]), \
                f"{row['param']} 的 {key} 缺数值"
        if row["param"] in ("fx", "fy", "cx", "cy"):
            assert row["utilization_ratio"] is not None and row["utilization_ratio"] > 1
        else:
            assert row["utilization_ratio"] is None, \
                f"{row['param']} 的利用率比值应置 None(单独下界被补偿空间撑大)"
    print(f"\n[产物] budget.npz 逐行底账 {_N_ROWS} 行 + budget_bars.png 在盘;"
          f"预算行 {len(rows)} 参数齐")
