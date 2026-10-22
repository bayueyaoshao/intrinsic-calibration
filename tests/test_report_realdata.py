"""真实数据报告对账:`calib report` 首跑 × REP-01/02(证据链报告落地)。

整文件标记 realdata(要跑真实七路上游的最新轮组装,分钟级;白天快速档用
`pytest -m "not realdata"` 自动跳过)。前置:七路上游最新轮的取数主文件都
在盘(ingest/corners/noise/calibration/fisher/evaluate/analyze)——缺任何
一路整文件跳过并写中文理由,不误报失败;报告命令本身只消费落盘轮,
不重算任何上游。

测的是什么(对应 PLAN 06-01 Task 3):
1. 四件产物齐:output/report/ 最新轮下 report.html、report.md、
   report_data.json、figures/ 一个不少。
2. 数字逐位同源:report.html 里的判行行、当选模型名、内参取值都能在
   report_data.json 里找到同值(报告不私造数字)。
3. 六类图逐一可查(REP-01):角点叠加、残差分带(裁定①替代热图)、
   CV 曲线、CRLB 对比、敏感度排序、摆位覆盖——每类在图表索引有编号
   引用且已复制进 figures/。
4. 报告九节骨架在 HTML 与 Markdown 双格式里都在。

体例照 tests/test_analysis_realdata.py:模块级夹具只跑一次命令,全部
断言共享。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.realdata]

REPO_ROOT = Path(__file__).resolve().parents[1]

# 七路上游:阶段目录 → 取数主文件(缺任何一份就整文件跳过)
_UPSTREAM_MAIN = (
    ("ingest", "manifest.json"),
    ("corners", "detection_summary.json"),
    ("noise", "noise.json"),
    ("calibration", "comparison.json"),
    ("fisher", "crlb.json"),
    ("evaluate", "evaluate.json"),
    ("analyze", "analyze.json"),
)

# 六类图(REP-01):通俗名 → 图表清单里的文件名
# (残差热图按 2026-10-22 裁定① = 残差分带曲线 + 覆盖足迹两件既有图)
_SIX_FIGURES = (
    ("角点叠加", "overview_mosaic.png"),
    ("残差分带(裁定①)", "radial_bands.png"),
    ("CV 曲线", "cv_errorbars.png"),
    ("CRLB 对比", None),  # 文件名含当选模型名,运行时拼
    ("敏感度排序", "sensitivity_ranking.png"),
    ("摆位覆盖", "coverage_footprint.png"),
)


def _newest_run(stage_dir: Path) -> Path | None:
    if not stage_dir.is_dir():
        return None
    runs = sorted((p for p in stage_dir.iterdir()
                   if p.is_dir() and p.name.startswith("run_")), reverse=True)
    return runs[0] if runs else None


def _missing_prerequisites() -> str | None:
    for stage, fname in _UPSTREAM_MAIN:
        run = _newest_run(REPO_ROOT / "output" / stage)
        if run is None:
            return f"output/{stage}/ 下没有任何运行轮"
        if not (run / fname).is_file():
            return f"output/{stage}/ 最新轮缺取数主文件 {fname}"
    return None


@pytest.fixture(scope="module")
def report_round():
    """跑一次 calib report(真实最新轮,分钟级),返回最新报告轮的产物。"""
    problem = _missing_prerequisites()
    if problem:
        pytest.skip(f"前置产物不全,请先按流水线顺序补跑上游:{problem}")

    calib_bin = Path(sys.executable).parent / "calib"
    proc = subprocess.run([str(calib_bin), "report"], check=False,
                          cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"calib report 对真实数据退出码 {proc.returncode}\n"
        f"--- stdout 末尾 ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr 末尾 ---\n{proc.stderr[-2000:]}")

    runs = sorted((REPO_ROOT / "output" / "report").glob("run_*"))
    run_dir = runs[-1]
    return {
        "run_dir": run_dir,
        "data": json.loads(
            (run_dir / "report_data.json").read_text(encoding="utf-8")),
        "html": (run_dir / "report.html").read_text(encoding="utf-8"),
        "md": (run_dir / "report.md").read_text(encoding="utf-8"),
    }


def test_four_artifacts_and_nonempty_figures(report_round):
    run_dir = report_round["run_dir"]
    for name in ("report.html", "report.md", "report_data.json"):
        assert (run_dir / name).is_file(), f"缺产物 {name}"
    figs = list((run_dir / "figures").glob("*.png"))
    assert figs, "figures/ 一张图都没有,报告图表引用会全空"


def test_html_numbers_traceable_to_snapshot(report_round):
    """报告内数字与 report_data.json 逐位同源:判行行、当选模型、内参取值
    都能在底账快照里找到同值。"""
    data, html = report_round["data"], report_round["html"]
    assert data["schema"] == "report-context-v1"

    selected = data["metrics_raw"]["selected"]
    assert selected, "真实数据应有选型结论"
    assert selected in html, "当选模型名没出现在报告里"

    # 判行行与内参取值列(前三个参数足够抽查同源性)
    assert data["s1"]["verdict_line"] in html, "判行行与底账不同源"
    for row in data["s1"]["params_table"][:3]:
        assert row["value"] in html, f"内参 {row['param']} 取值与底账不同源"


def test_nine_sections_in_both_formats(report_round):
    data = report_round["data"]
    for title in data["sections"]:
        assert title in report_round["html"], f"HTML 缺节:{title}"
        assert title in report_round["md"], f"Markdown 缺节:{title}"


def test_six_figure_categories_indexed_and_copied(report_round):
    """六类图逐一可查:图表索引有编号引用,且源图已复制进 figures/。"""
    data = report_round["data"]
    fig_dir = report_round["run_dir"] / "figures"
    indexed = {fig["file"]: fig for fig in data["s9"]["figures"]}

    for label, fname in _SIX_FIGURES:
        if fname is None:  # CRLB 特征谱带当选模型名
            fname = (f"spectrum_{data['metrics_raw']['selected']}"
                     f"_views640_detector.png")
        assert fname in indexed, f"「{label}」没有进图表索引({fname})"
        assert (fig_dir / fname).is_file(), \
            f"「{label}」({fname})没被复制进 figures/"
        # 编号引用(图N)应在正文出现至少一次(索引或节内嵌)
        assert indexed[fname]["idx"] in report_round["html"], \
            f"「{label}」的编号 {indexed[fname]['idx']} 没在 HTML 出现"
