"""calib run-all 编排测试:monkeypatch 八步为记录调用序的假函数,不真跑标定。

白天快速档(秒级):只锁编排逻辑——步骤顺序对齐 docs/pipeline.md、
--from/--until 区间切片、非法步骤名快速失败、失败断点中文指引、命令入口
退出码收口。真实链路(真跑标定的端到端)由 06-04 的变体与人工检查点
统一验收,本文件绝不真跑任何一步。
"""

from __future__ import annotations

import typer
from typer.testing import CliRunner

from calib.cli import main as cli_main
from calib.cli.main import app

runner = CliRunner()

# docs/pipeline.md 的八步顺序(run-all 的调用序必须与它完全一致)
PIPELINE_ORDER = ["ingest", "corners", "noise", "calibrate", "fisher",
                  "evaluate", "analyze", "report"]


def _install_fake_steps(monkeypatch, fail_at: str | None = None,
                        fail_exc: Exception | None = None) -> list[str]:
    """把 main 模块的步骤表换成记录调用序的假函数(不真跑任何一步)。

    fail_at 指定的步骤在记录调用名后抛 fail_exc,模拟该步中途失败
    (与真命令失败时的形态一致:noise/evaluate 等失败契约收口后抛的
    正是 typer.Exit(code=1))。返回调用记录列表,供测试断言调用序。
    """
    calls: list[str] = []

    def _make(name: str):
        def _step() -> None:
            calls.append(name)
            if name == fail_at and fail_exc is not None:
                raise fail_exc
        return _step

    fake_steps = [(name, _make(name)) for name in PIPELINE_ORDER]
    monkeypatch.setattr(cli_main, "STEPS", fake_steps)
    return calls


def test_run_all_full_range_follows_pipeline_order(tmp_path, monkeypatch):
    """全区间:八步调用序与 docs/pipeline.md 完全一致。

    旧版把 calibrate 排在 noise 前面且缺后三步,此例锁死正确顺序防回退。
    """
    monkeypatch.chdir(tmp_path)
    calls = _install_fake_steps(monkeypatch)
    result = runner.invoke(app, ["run-all"])
    assert result.exit_code == 0, result.output
    assert calls == PIPELINE_ORDER


def test_run_all_from_until_runs_only_the_slice(tmp_path, monkeypatch):
    """--from calibrate --until fisher:只调 calibrate、fisher 两步。

    这是"剔除照片后重标"的载体(裁定①):--from/--until 是流程范围
    开关,不是算法参数,不违反 D-12。
    """
    monkeypatch.chdir(tmp_path)
    calls = _install_fake_steps(monkeypatch)
    result = runner.invoke(
        app, ["run-all", "--from", "calibrate", "--until", "fisher"])
    assert result.exit_code == 0, result.output
    assert calls == ["calibrate", "fisher"]


def test_run_all_bad_step_names_fail_fast_in_chinese(tmp_path, monkeypatch):
    """非法步骤名 / 区间倒置:中文报错、退出码 1、一步都不跑。"""
    monkeypatch.chdir(tmp_path)
    calls = _install_fake_steps(monkeypatch)

    unknown = runner.invoke(app, ["run-all", "--from", "bogus"])
    assert unknown.exit_code == 1
    assert "不认识" in unknown.output
    assert "可用步骤名" in unknown.output

    inverted = runner.invoke(
        app, ["run-all", "--from", "report", "--until", "ingest"])
    assert inverted.exit_code == 1
    assert "区间是空的" in inverted.output

    assert calls == [], "参数非法时应快速失败,任何一步都不该被调用"


def test_run_all_stops_at_failed_step_with_recovery_hint(tmp_path, monkeypatch):
    """中途失败即停(T-06-06,直调链 Exit 冒泡即停的兜底验证)。

    假 evaluate 抛 typer.Exit(1)(与真命令失败契约收口后的形态一致):
    run-all 退出码 1、播报含「已完成到」「恢复」、失败步之后的
    analyze/report 不再被调用。
    """
    monkeypatch.chdir(tmp_path)
    calls = _install_fake_steps(monkeypatch, fail_at="evaluate",
                                fail_exc=typer.Exit(code=1))
    result = runner.invoke(app, ["run-all"])
    assert result.exit_code == 1
    assert calls == ["ingest", "corners", "noise", "calibrate", "fisher",
                     "evaluate"], "失败步之后的步骤绝不该被调用"
    assert "已完成到" in result.output
    assert "恢复" in result.output
    assert "--from evaluate" in result.output, "恢复命令应指回失败那一步"


def test_ingest_bad_input_wrapped_as_chinese_error(tmp_path, monkeypatch):
    """退出码收口(T-06-07):坏输入的裸 ValueError 收口为中文一句
    + 退出码 1,不吐 traceback(修复前直接吐堆栈)。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()  # config 目录在但 board.yaml 缺失 → 库层中文 ValueError
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "照片入库体检失败" in result.output
    assert "请先检查" in result.output
    assert "Traceback" not in result.output
