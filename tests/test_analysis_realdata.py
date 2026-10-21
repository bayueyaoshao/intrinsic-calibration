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
    """契约形状:budget / residual_structure / test_xyz / sensitivity 四段
    有值,块 5–6 显式缺位(None)并进 meta.missing_sections;目标模型与
    evaluate 选型结论一致(m2)。"""
    report = analyze_tree["report"]
    budget = report["budget"]
    assert isinstance(budget, dict) and budget, "budget 段缺席(误差归因没跑成)"
    residual = report["residual_structure"]
    assert isinstance(residual, dict) and residual, \
        "residual_structure 段缺席(残差方向分解没跑成)"
    for key in ("corner_opt", "anisotropy"):
        assert report[key] is None, \
            f"{key} 段应由后续计划接入,当前必须显式为 None"
    assert set(report["meta"]["missing_sections"]) == {
        "corner_opt", "anisotropy"}
    assert report["meta"]["selected_model"] == "m2", \
        f"目标模型应与 evaluate 选型结论一致,实际 {report['meta']['selected_model']!r}"
    assert budget["schema"] == "analyze-budget-v1"
    assert budget["sigma_detector_corner"] == pytest.approx(0.008708, abs=1e-4), \
        "噪声口径必须是检测器级 σ(CRLB 口径铁律,时域口径会虚高约 35 倍)"
    assert residual["schema"] == "analyze-residual-v1"
    assert report["test_xyz"]["schema"] == "analyze-test-xyz-v1"
    assert report["sensitivity"]["schema"] == "analyze-sensitivity-v1"
    assert "delivery" in report["meta"]["r_normalization_basis"], \
        "r 归一化基准必须随 meta 透传(跨轮比较先对齐口径)"
    assert residual["r_normalization_baseline_check"]["status"] == "consistent", \
        "r 归一化基准应与 findings 结论 5 所用标定基准一致(delivery 与 " \
        "标定轮 fx/fy/cx/cy 逐位同源)"


def test_analyze_residual_footprint_matches_findings_5(analyze_tree: dict):
    """findings 结论 5 对账:最远观测 r ≈ 0.685、画面四角 r ≈ 0.796
    (各 ±2%);r>0.5 观测约两成(21.1%)却承载 k2 信息约 61.4%(±10%);
    外推区间恰为 [r_max_obs, r_corner_frame],披露文案点出"外推"与"k2"。"""
    residual = analyze_tree["report"]["residual_structure"]
    cov = residual["train"]["coverage"]
    assert cov["r_max_obs"] == pytest.approx(0.685, rel=0.02), \
        f"最远观测 r {cov['r_max_obs']:.4f} 偏离结论 5 基准 0.685 超过 2%"
    assert cov["r_corner_frame"] == pytest.approx(0.796, rel=0.02), \
        f"画面四角 r {cov['r_corner_frame']:.4f} 偏离结论 5 基准 0.796 超过 2%"
    assert cov["extrapolation_interval"] == pytest.approx(
        [cov["r_max_obs"], cov["r_corner_frame"]]), \
        "外推区间必须恰为 [最远观测 r, 画面四角 r]"
    assert cov["frac_beyond_05"] == pytest.approx(0.211, rel=0.10), \
        f"r>0.5 观测占比 {cov['frac_beyond_05']:.3f} 偏离结论 5 基准 0.211"
    assert cov["info_share_beyond_05"] == pytest.approx(0.614, rel=0.10), \
        f"k2 信息份额 {cov['info_share_beyond_05']:.3f} 偏离结论 5 基准 0.614"
    note = residual["extrapolation_note"]
    assert "外推" in note and "k2" in note, \
        "外推披露文案必须点名「外推」与「k2」(畸变外推风险显式披露)"
    ext = residual["extrapolation"]
    assert ext["applicable"] is True
    assert ext["r_max_obs"] == pytest.approx(cov["r_max_obs"])
    assert ext["k2_wobble_at_corner_frame"] > ext["k2_wobble_at_r_05"] \
        > ext["k2_wobble_at_r_04"], "k2 晃动必须随 r 递增(r⁴ 律)"
    assert len(ext["criteria_table"]) == 3, "判据表应三行(中区/外环/外推区)"
    print(f"\n[结论 5] r_max_obs={cov['r_max_obs']:.3f},四角 "
          f"r={cov['r_corner_frame']:.3f},r>0.5 占 "
          f"{cov['frac_beyond_05']:.1%} 承载 k2 信息 "
          f"{cov['info_share_beyond_05']:.1%}")


def test_analyze_residual_bands_and_test_side(analyze_tree: dict):
    """分带与 test 侧:test 65 张全评分(n_failed 有数值键留痕);两侧
    bands 行数与分带边界一致(默认 0/0.2/0.4/0.6/0.8);实测最远观测
    r≈0.685 没超出最外边界 0.8,train 超界计数应为 0(与最外带
    n_beyond_hi 同源同值)。"""
    residual = analyze_tree["report"]["residual_structure"]
    edges = residual["band_edges"]
    assert edges == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8]), \
        f"分带边界应读配置默认值,实际 {edges}"
    n_bands = len(edges) - 1
    test_blk = residual["test"]
    assert test_blk["n_frames"] == 65, \
        f"test 侧应逐帧评分 65 张,实际 {test_blk['n_frames']}"
    assert isinstance(test_blk["n_failed"], int) and test_blk["n_failed"] >= 0, \
        "失败帧计数键必须在盘(失败留痕,不冒充)"
    assert test_blk["per_frame_rms"]["median_px"] is not None, \
        "test 逐帧 RMS 中位必须有值(内参当尺子的独立考题成绩)"
    for side in ("train", "test"):
        bands = residual[side]["bands"]
        assert len(bands) == n_bands, f"{side} bands 应 {n_bands} 行"
        assert bands[-1]["n_beyond_hi"] >= 0
    assert residual["train"]["n_beyond_last_edge"] == \
        residual["train"]["bands"][-1]["n_beyond_hi"], \
        "train 超界计数与最外带 n_beyond_hi 必须同源同值"
    # 实测最远 r 0.685 < 边界 0.8:超界应为 0(有观测越界才非零)
    assert residual["train"]["bands"][-1]["n_beyond_hi"] == 0, \
        "实测最远 r≈0.685 未超出边界 0.8,超界计数应为 0"
    print("\n[分带] train 每带 RMS:"
          + ", ".join(
              f"[{b['band_lo']:g},{b['band_hi']:g})={b['rms_px']}"
              for b in residual["train"]["bands"])
          + f";test {test_blk['n_frames']} 张失败 {test_blk['n_failed']} 张")


def test_analyze_residual_sidecars_and_figures(analyze_tree: dict):
    """产物保真:residual_structure.npz 旁车(r_train 640×88 逐角点半径)
    与两张空间图(r 分带曲线 + 覆盖足迹)都在盘。"""
    run_dir = analyze_tree["run_dir"]
    detail = analyze_tree["details"]["residual_structure"]
    assert detail["r_train"].shape == (640, 88), \
        f"r_train 应 (640, 88)(640 视图 × 88 角点),实际 {detail['r_train'].shape}"
    assert (run_dir / "analysis" / "residual_structure.npz").is_file()
    assert (run_dir / "figures" / "radial_bands.png").is_file()
    assert (run_dir / "figures" / "coverage_footprint.png").is_file()
    print(f"\n[产物] residual_structure.npz(r_train {detail['r_train'].shape})"
          "+ radial_bands.png + coverage_footprint.png 在盘")


def test_analyze_blocks_234_use_stage_and_budget_stays_direct():
    """源码断言(接线体例):块 2/3/4 包在 _stage 里(单块失败记 None
    照常出报告),块 1 预算依旧直调不套兜底(铁律违例必须退出码 1)——
    两种失败语义不许混。"""
    import inspect

    import calib.cli.main as cli_main

    source = inspect.getsource(cli_main.analyze)
    for call in ('_stage("残差方向分解"', '_stage("test 三分量分解"',
                 '_stage("参数敏感度排序"'):
        assert call in source, f"{call} 缺席:诊断块必须走 _stage 逐块兜底"
    assert "run_residual_structure" in source and "run_budget(" in source
    # budget 直调:run_budget 的调用行不许出现在 _stage 调用行上
    for line in source.splitlines():
        if "run_budget(" in line:
            assert "_stage" not in line, "budget 必须直调,不许套 _stage"


def test_analyze_test_xyz_section(analyze_tree: dict):
    """ANA-03 真实数据:test 65 张全部分解(零失败零病态);逐方向
    median/max 有值;notes 两条固定声明在盘(test 乐观偏差 + 深度判读);
    深度放大倍数中位有值(可观测性归一不缺席)。"""
    section = analyze_tree["report"]["test_xyz"]
    assert section["n_frames"] == 65, \
        f"test 侧应逐帧分解 65 张,实际 {section['n_frames']}"
    assert section["n_valid"] == 65, \
        f"65 张应全部有效(真实数据角点质量足以支撑反解),实际 {section['n_valid']}"
    assert section["n_cond_excluded"] == 0 and section["n_solve_failed"] == 0, \
        "真实数据不该有病态帧/解不出的帧(出现即上游角点或内参出问题)"
    for comp in ("x", "y", "z"):
        for block in ("dX_abs_mm", "abs_zscore"):
            entry = section["per_direction"][block][comp]
            assert entry["median"] is not None and entry["max"] is not None
    assert section["observability"]["depth_amplification_median"] is not None
    notes = " ".join(section["notes"])
    assert "乐观偏差" in notes, "notes 必须声明 test 集参加过选型的乐观偏差"
    assert "深度" in notes, "notes 必须点名 Z 分量的可观测性判读"
    print("\n[ANA-03] 65 帧全解;|dX| 中位 x/y/z = "
          + ", ".join(
              f"{section['per_direction']['dX_abs_mm'][c]['median']:.3f}mm"
              for c in ("x", "y", "z"))
          + f";深度放大倍数中位 "
          f"{section['observability']['depth_amplification_median']:.2f} 倍")


def test_analyze_sensitivity_section(analyze_tree: dict):
    """ANA-04 真实数据(findings 结论 1/2/3 对账):uniqueness 的 bootstrap
    最大位移当量 ≈ 1.9943(±1%),多起点数值键在盘;advice 非空且首条
    点名 k2/k3 软方向、给"补"字手段、带"不自动执行"声明(裁定③);
    近零参数像素当量低于 0.05px 影响线;quick 档 what_if 为 None 且
    what_if_note 说明档位;双谱对账夹角有限在 [0,1]。"""
    section = analyze_tree["report"]["sensitivity"]
    uniq = section["uniqueness"]
    assert uniq["bootstrap_max_disp_px"] == pytest.approx(1.9943, rel=0.01), \
        f"bootstrap 最大位移 {uniq['bootstrap_max_disp_px']:.4f} 偏离结论 2 " \
        f"基准 1.9943 超过 1%"
    assert isinstance(uniq["multistart_disp_px"], float)
    assert uniq["multistart_verdict"] in ("consistent", "inconsistent")
    assert "解不唯一" in uniq["note"]

    advice = section["advice"]
    assert advice, "建议规则表不该为空(最大方向载荷集中在 k2/k3 必触发分支一)"
    first = advice[0]
    assert ("k2" in first["direction"] or "k3" in first["direction"]), \
        f"首条建议应点名 k2/k3 软方向,实际:{first['direction']}"
    assert "补" in first["advice"] and "不自动执行" in first["advice"], \
        "首条建议要给补拍手段且带不自动执行声明(裁定③)"

    px = section["px_equivalents"]
    assert px["p1_px_equiv"] < 0.05 and px["p2_px_equiv"] < 0.05, \
        "p1/p2 的 σ 级像素当量应低于 0.05px 影响线(结论 1 反面教材)"

    assert section["what_if"] is None, "quick 档不做 what-if(裁定②)"
    assert "quick 档" in section["what_if_note"]
    emp = section["empirical_spectrum"]
    for key in ("cosine_theory_vs_bootstrap", "cosine_theory_vs_slices"):
        assert 0.0 <= emp[key] <= 1.0, f"{key} 应是 [0,1] 内的夹角余弦"
    print(f"\n[ANA-04] bootstrap 最大位移 {uniq['bootstrap_max_disp_px']:.4f}px;"
          f"建议 {len(advice)} 条(首条:{first['direction'][:30]});"
          f"理论 vs 经验夹角 cos = {emp['cosine_theory_vs_bootstrap']:.3f} / "
          f"{emp['cosine_theory_vs_slices']:.3f}")


def test_analyze_xyz_sensitivity_sidecars_and_figures(analyze_tree: dict):
    """产物保真:test_xyz.npz(dX 65×3 逐帧矩阵)与 sensitivity.npz 旁车
    在盘;三分量对比图与敏感度排序图在盘;quick 档无补拍预演图。"""
    run_dir = analyze_tree["run_dir"]
    detail = analyze_tree["details"]["test_xyz"]
    assert detail["dX"].shape == (65, 3), \
        f"test_xyz 旁车 dX 应 (65, 3),实际 {detail['dX'].shape}"
    assert np.isfinite(detail["dX"]).all(), "65 帧全有效,dX 不该有 NaN"
    assert (run_dir / "analysis" / "test_xyz.npz").is_file()
    assert (run_dir / "analysis" / "sensitivity.npz").is_file()
    assert (run_dir / "figures" / "test_xyz.png").is_file()
    assert (run_dir / "figures" / "sensitivity_ranking.png").is_file()
    assert not (run_dir / "figures" / "what_if.png").is_file(), \
        "quick 档不做 what-if,补拍预演图不该存在"
    print(f"\n[产物] test_xyz.npz(dX {detail['dX'].shape})+ sensitivity.npz "
          "+ test_xyz.png + sensitivity_ranking.png 在盘(what_if.png 无,quick 档)")


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
