"""报告组装层测试(合成最小上游树,白天快速档,秒级)。

测的是 06-01 落地的报告两层结构:
1. build_report_context:合成七路上游树 → 全中文 context;九节骨架齐、
   判行与缺席披露字段如实、物理换算尺现算(方格毫米 ÷ 方格像素中位)、
   整个 context 可原样 JSON 序列化(= report_data.json 底账的前提);
   直接取数主文件整个缺失时中文报错退出(不静默出假报告)。
2. 双模板冒烟:同一 context 喂 HTML(autoescape 开)/ Markdown(autoescape
   关)两个模板——九节标题出现;报告里的关键数字字符串与 context 逐位
   同源(锁「同源一致」,不锁固定值);缺席段渲染不崩;含 < 和 & 的字段
   值在 HTML 里被自动转义(Markdown 纯文本则原样保留)。
3. calib report 命令体:四件产物齐(report.html / report.md /
   report_data.json / figures/);报告内数字能在 report_data.json 里逐位
   找到同值;新轮落盘成功后删更旧轮、只留最新两轮。

体例照 tests/test_cli_main.py:工作目录用 monkeypatch.chdir 切到临时树,
上游产物手工构造不真跑,真实 output/ 不受影响。
"""

from __future__ import annotations

import json
from pathlib import Path

import jinja2
import pytest

from calib.cli.main import report as report_cmd
from calib.report.build import SECTION_TITLES, TEMPLATES_DIR, build_report_context

# 合成树里的固定输入(断言「同源一致」时在测试里现算对照值,不硬编码结论)
_SQ_PX_MEDIAN = 45.3
_SQUARE_MM = 25.0
_PARAM_NAMES = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
_PARAMS = {"fx": 800.5, "fy": 800.2, "cx": 320.4, "cy": 240.3,
           "k1": -0.21, "k2": 0.05, "p1": 0.001, "p2": -0.0005, "k3": 0.01}


def _w(root: Path, rel: str, obj) -> None:
    """往合成树里写一份 JSON 产物(逐级建目录)。"""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _fake_png(root: Path, rel: str) -> None:
    """放一个只用于「在盘检查」的占位 PNG(报告层只认文件存在,不解析内容)。"""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n placeholder-for-existence-check")


def build_minimal_tree(root: Path) -> None:
    """合成最小七路上游树:七份主文件 + 候选图表 PNG + 一条含 < 和 & 的字段值。

    故意留一个缺席段(analyze 轮 corner_opt 落账为 null)练习缺席披露路径;
    burst 点名里的 pose_id 塞入 < 和 & 练习 HTML 自动转义。
    """
    root.mkdir(parents=True, exist_ok=True)
    cfg = root / "config"
    cfg.mkdir(exist_ok=True)
    (cfg / "board.yaml").write_text(
        "board:\n  board_type: chessboard\n  cols: 11\n  rows: 8\n"
        f"  square_size_mm: {_SQUARE_MM}\n  unit: mm\n", encoding="utf-8")

    _w(root, "output/ingest/run_a/manifest.json", {
        "summary": {
            "n_train": 8, "n_poses": 3, "n_test": 5, "resolution": [640, 480],
            "suspect_counts": {"suspect_total": 1, "pose_relative": 1, "global": 0},
            "detector_success_rates": {"train": {"classic": 0.9, "sb": 1.0},
                                       "test": {"classic": 0.8, "sb": 1.0}},
            "repeats_distribution": {"3": 3}, "warnings": []}})
    _w(root, "output/corners/run_a/detection_summary.json", {
        "detection": {
            "train": {"classic": {"success_rate": 0.9, "n_detected": 9,
                                  "n_images": 10},
                      "sb": {"success_rate": 1.0, "n_detected": 10, "n_images": 10,
                             "sq_px": {"median": _SQ_PX_MEDIAN}}},
            "test": {"classic": {"success_rate": 0.8, "n_detected": 4,
                                 "n_images": 5},
                     "sb": {"success_rate": 1.0, "n_detected": 5, "n_images": 5,
                            "sq_px": {"median": 44.9}}}},
        "burst_check": {
            "n_groups_checked": {"classic": 3, "sb": 3}, "n_failures": 1,
            "failures": [{"detector": "classic", "pose_id": "p<x>&\"q",
                          "spread_median_px": 0.9, "spread_max_px": 1.4,
                          "threshold_px": 1.0}]}})
    _w(root, "output/noise/run_a/noise.json", {
        "sigma_detector_corner": 0.0087, "sigma_temporal_corner": 0.2995,
        "per_source_counts": {"temporal": 3, "immerkaer_transfer": 3},
        "outlier_poses": [{"pose_id": "p02", "sigma_pose": 0.42}]})
    _w(root, "output/calibration/run_a/comparison.json",
       [{"model_id": "m1", "rms_px": 0.55}, {"model_id": "m2", "rms_px": 0.31}])
    _w(root, "output/calibration/run_a/m2__scipy/result.json",
       {"params": _PARAMS})
    _w(root, "output/fisher/run_a/crlb.json", {
        "models": {"m2": {
            "observation_sets": {"views640": {
                "detector": {"status": "ok", "param_names": _PARAM_NAMES,
                             "se": [0.5, 0.5, 0.2, 0.2, 0.01, 0.01,
                                    5.8e-07, 5.8e-07, 0.05],
                             "condition_number": 8850, "flag": ""},
                "temporal": {"status": "ok", "param_names": _PARAM_NAMES,
                             "se": [2.0] * 9, "condition_number": 9000,
                             "flag": ""}}},
            "linearization_check": {"note": "与数值导数对得上", "rel_diff": 1e-06},
            "opencv_crosscheck": {"status": "一致",
                                  "verdict_note": "与 OpenCV 自带标准差同量级"}}}})
    _w(root, "output/evaluate/run_a/evaluate.json", {
        "selection": {
            "selected": "m2",
            "walk": [{"note": "从 m1 起步逐档走查"}],
            "reason": "更复杂一档提升不达门槛,取更简者",
            "verdict_thresholds": {"gen_overfit_ratio_max": 1.2,
                                   "stability_disp_px_max": 2.0},
            "test_rms_px": {"m1": 0.55, "m2": 0.31},
            "verdicts": [{
                "model_id": "m2", "bic": 1234.5, "overall": "FAIL",
                "generalization": {"pass": True, "mean_heldout_px": 0.33,
                                   "mean_train_px": 0.30, "overfit_ratio": 1.1,
                                   "note": "留出成绩贴近训练"},
                "stability": {"pass": False, "max_disp_px_overall": 3.1,
                              "note": "k1 位移超门槛"},
                "crlb": {"pass": True, "note": "全部比值高于告警线"}}],
            "agreement": {"note": "三种晃动来源方向一致"},
            "three_criteria": {"reason": "稳定判据未过,综合 FAIL"}},
        "delivery": {"models": {"m2": {"final_params": _PARAMS,
                                       "rms_px": 0.29, "bic": 1230.1}}},
        "bootstrap": {"meta": {"n_resamples": 30},
                      "models": {"m2": {"param_std": {
                          "fx": 0.6, "fy": 0.6, "cx": 0.3, "cy": 0.3,
                          "k1": 0.02, "k2": 0.02, "p1": 2e-07, "p2": 2e-07,
                          "k3": 0.06}}}},
        "slices": {"meta": {"n_groups": 3},
                   "models": {"m2": {"empirical_std": {
                       "fx": 0.55, "fy": 0.58, "cx": 0.28, "cy": 0.31,
                       "k1": 0.018, "k2": 0.019, "p1": 1.8e-07, "p2": 1.9e-07,
                       "k3": 0.055}}}},
        "cv": {"meta": {"n_folds": 3, "n_rounds": 3, "n_solves": 9},
               "models": {"m2": {"mean_heldout_px": 0.34}}},
        "lopo": {"models": {"m2": {"max_disp_px": 2.5}}},
        "optimizer": {"recommended_solver": {"m2": {"loss": "soft_l1"}},
                      "multistart": {"models": {"m2": {
                          "n_starts": 5, "n_converged": 5,
                          "verdict": "收敛一致"}}}}})
    # corner_opt 落账为 null:缺席披露路径(报告如实记「缺席」,不冒充数值)
    _w(root, "output/analyze/run_a/analyze.json", {
        "meta": {"missing_sections": ["corner_opt"]},
        "budget": {
            "verdict": "fail", "n_pointwise_violations": 0,
            "min_ratio_pointwise": 102.3, "lines": {"hard": 0.42},
            "recheck": {"consistent": True,
                        "ratios": {"重投影位移中位": 102, "重投影位移95分位": 1108,
                                   "重投影位移最大": 1444}},
            "rows": [
                {"param": "fx", "empirical_std_px": 0.55, "bootstrap_std": 0.6,
                 "pose_sampling_std": 0.5, "multistart_disp_px": 0.4,
                 "crlb_se": 0.5, "utilization_ratio": 1.2},
                {"param": "k1", "empirical_std_px": 0.018, "bootstrap_std": 0.02,
                 "pose_sampling_std": 0.02, "multistart_disp_px": 0.03,
                 "crlb_se": 0.01, "utilization_ratio": 2.0}]},
        "residual_structure": {
            "extrapolation_note": "最远观测 r 之外没有照片,纯靠外推",
            "train": {"coverage": {"r_max_obs": 0.48, "r_corner_frame": 0.8,
                                   "frac_beyond_05": 0.02,
                                   "info_share_beyond_05": 0.3},
                      "bands": [{"band_lo": 0.0, "band_hi": 0.2, "n": 1000,
                                 "rms_px": 0.3, "max_px": 0.8}]}},
        "test_xyz": {
            "n_valid": 4, "n_frames": 5, "n_cond_excluded": 1, "n_solve_failed": 0,
            "notes": ["test 集参加过选型,成绩带轻微乐观偏差"],
            "observability": {"depth_amplification_median": 6.0},
            "per_direction": {
                "dX_abs_mm": {"x": {"median": 0.05, "max": 0.1},
                              "y": {"median": 0.04, "max": 0.09},
                              "z": {"median": 0.12, "max": 0.2}},
                "abs_zscore": {"x": {"median": 1.0, "max": 2.0},
                               "y": {"median": 1.1, "max": 2.2},
                               "z": {"median": 3.0, "max": 5.0}}}},
        "sensitivity": {
            "advice": [{"direction": "补拍边角", "evidence": "k2 信息三成来自边角",
                        "advice": "画面四角再补几张"}],
            "uniqueness": {"bootstrap_max_disp_px": 3.1,
                           "multistart_verdict": "收敛", "note": "多起点都收敛"}},
        "corner_opt": None,
        "anisotropy": {"summary": {"median": 1.1, "significant_line": 1.5,
                                   "n_aniso_gt_1p5": 0},
                       "plain_language": "横竖定位基本对称"}})

    # 17 张候选 PNG 全部放盘:图表清单应一张不缺地进 context 并被复制
    _fake_png(root, "output/corners/run_a/mosaic/overview_mosaic.png")
    _fake_png(root, "output/corners/run_a/mosaic/suspect_mosaic.png")
    _fake_png(root, "output/noise/run_a/transfer_curve.png")
    for name in ("corner_opt_self", "corner_opt_prodsb", "budget_bars",
                 "radial_bands", "coverage_footprint", "test_xyz",
                 "sensitivity_ranking", "anisotropy_map", "anisotropy_hist"):
        _fake_png(root, f"output/analyze/run_a/figures/{name}.png")
    for name in ("spectrum_m2_views640_detector", "spectrum_m2_views640_temporal"):
        _fake_png(root, f"output/fisher/run_a/figures/{name}.png")
    for name in ("cv_errorbars", "dispersion_compare", "lopo_pose_errors"):
        _fake_png(root, f"output/evaluate/run_a/figures/{name}.png")


@pytest.fixture()
def synthetic_tree(tmp_path, monkeypatch) -> Path:
    """合成树 + 工作目录切过去(board.yaml 按当前工作目录解析)。"""
    build_minimal_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# =====================================================================
# 1. build_report_context:骨架 / 同源 / 缺席披露 / 可序列化
# =====================================================================
def test_context_nine_sections_same_source_and_serializable(synthetic_tree):
    ctx = build_report_context(synthetic_tree / "output")

    # 九节骨架与可序列化(原样快照落 report_data.json 的前提)
    assert ctx["schema"] == "report-context-v1"
    assert tuple(ctx["sections"]) == SECTION_TITLES
    json.dumps(ctx, ensure_ascii=False)  # 抛异常即失败

    # 判行如实:合成数据稳定判据 FAIL,综合 FAIL,判行串与 context 同源
    assert ctx["s1"]["selected"] == "m2"
    assert ctx["s1"]["verdict_line"] is not None
    assert "泛化 PASS" in ctx["s1"]["verdict_line"]
    assert "稳定 FAIL" in ctx["s1"]["verdict_line"]
    assert "综合 FAIL" in ctx["s1"]["verdict_line"]

    # 物理换算尺现算:方格毫米 ÷ 方格像素中位(两侧都现算,不锁固定值)
    expect_mm_per_px = _SQUARE_MM / _SQ_PX_MEDIAN
    assert ctx["metrics_raw"]["mm_per_px"] == pytest.approx(expect_mm_per_px)
    assert f"{expect_mm_per_px:.3f}" in ctx["s1"]["physical_line"]

    # 内参表与底账同源:交付参数逐位进入取值列
    row_fx = next(r for r in ctx["s1"]["params_table"] if r["param"] == "fx")
    assert row_fx["value"] == f"{_PARAMS['fx']:.4f}"

    # 缺席段如实:corner_opt 落账 null → 定案缺席 + 披露清单有记录
    assert ctx["s3"]["corner_opt_note"] is None
    assert "角点寻优段缺席" in ctx["s3"]["corner_opt_absent_note"]
    assert any("分析段缺席" == d["kind"] for d in ctx["disclosures"])

    # 计划强制的口径键名不改名,白话注解随行
    assert "views640" in ctx["s5"]["obs_note"]


def test_context_missing_main_file_raises_chinese_error(synthetic_tree):
    """直接取数主文件整个缺失 → 中文报错退出,指路先跑对应命令(T-06-02)。"""
    (synthetic_tree / "output/noise/run_a/noise.json").unlink()
    with pytest.raises(ValueError, match="请先运行 calib noise"):
        build_report_context(synthetic_tree / "output")


# =====================================================================
# 2. 双模板冒烟:同源一致 / 缺席渲染 / HTML 自动转义
# =====================================================================
def _render(context: dict, name: str, autoescape: bool) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        autoescape=autoescape,
        undefined=jinja2.StrictUndefined,  # 模板引用不存在的键当场报错
    )
    return env.get_template(name).render(**context)


def test_templates_render_same_source_and_escape(synthetic_tree):
    ctx = build_report_context(synthetic_tree / "output")
    html = _render(ctx, "report.html.j2", autoescape=True)
    md = _render(ctx, "report.md.j2", autoescape=False)

    # 九节标题在两种格式里都出现
    for title in SECTION_TITLES:
        assert title in html, f"HTML 缺节标题:{title}"
        assert title in md, f"Markdown 缺节标题:{title}"

    # 关键数字字符串与 context 逐位同源(锁同源一致,不锁固定值):
    # 从 context 取定稿字符串,原样出现在渲染产物里
    for text in (ctx["s1"]["verdict_line"], ctx["s1"]["test_rms_line"],
                 ctx["s1"]["physical_line"], ctx["s7"]["ratios_line"],
                 ctx["s5"]["selected_lines"][0],
                 next(r for r in ctx["s1"]["params_table"]
                      if r["param"] == "fx")["value"]):
        assert text in html, f"HTML 缺 context 定稿串:{text[:40]}"
        assert text in md, f"Markdown 缺 context 定稿串:{text[:40]}"

    # 缺席段渲染不崩,披露文案出现
    assert "角点寻优段缺席" in html
    assert "角点寻优段缺席" in md

    # HTML 自动转义:含 < 和 & 的字段值不得原样进页面(T-06-01);
    # Markdown 纯文本不开转义,原样保留
    raw_value = "p<x>&\"q"
    assert raw_value not in html, "HTML 出现未转义的 < 和 &"
    assert "p&lt;x&gt;&amp;" in html
    assert raw_value in md


# =====================================================================
# 3. calib report 命令体:四件产物 / 底账同源 / 只留最新两轮
# =====================================================================
def test_report_command_artifacts_snapshot_and_retention(synthetic_tree):
    # 预置两轮更旧的报告轮(名字按 run_<时间戳> 字典序排在任何新轮之前):
    # 新轮落盘成功后应删掉最旧一轮,只留「上一轮 + 本轮」
    report_root = synthetic_tree / "output" / "report"
    for old in ("run_20250101_000000", "run_20250102_000000"):
        (report_root / old).mkdir(parents=True)

    report_cmd()

    runs = sorted(p.name for p in report_root.glob("run_*"))
    assert len(runs) == 2, f"只应保留最新两轮,实际 {len(runs)}:{runs}"
    assert runs[0] == "run_20250102_000000", "上一轮应保留(06-02 跨运行对比的读锚)"
    run_dir = report_root / runs[1]

    # 四件产物齐
    for name in ("report.html", "report.md", "report_data.json"):
        assert (run_dir / name).is_file(), f"缺产物 {name}"
    figs = list((run_dir / "figures").glob("*.png"))
    assert len(figs) == 17, f"figures/ 应有 17 张,实际 {len(figs)}"

    # 底账快照与报告逐位同源(从底账取串,回查报告原文)
    data = json.loads(
        (run_dir / "report_data.json").read_text(encoding="utf-8"))
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert data["schema"] == "report-context-v1"
    assert data["metrics_raw"]["selected"] == "m2"
    for text in (data["s1"]["verdict_line"], data["s1"]["physical_line"],
                 data["s1"]["params_table"][0]["value"],
                 data["s7"]["ratios_line"]):
        assert text in html, f"报告缺底账定稿串:{text[:40]}"


def test_report_command_broadcast_four_elements(synthetic_tree, capsys):
    """收尾播报给零基础人的四要素指引一个不少。"""
    report_cmd()
    out = capsys.readouterr().out
    for keyword in ("看哪个文件", "文件作用", "具体看哪里", "好坏判据"):
        assert keyword in out, f"播报缺四要素之一:{keyword}"
    assert "report.html" in out
