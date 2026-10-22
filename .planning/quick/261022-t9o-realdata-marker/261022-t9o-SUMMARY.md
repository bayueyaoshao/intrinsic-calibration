---
quick: 261022-t9o-realdata-marker
plan: 261022-t9o
type: execute
status: complete
date: "2026-10-22"
started: "2026-10-22T13:09:06Z"
finished: "2026-10-22T13:16:30Z"
duration: "约 7 分钟(纯标记改动 + collect-only 验证,零真跑)"
commits:
  - "5932bdc: test(261022-t9o): pyproject 注册 realdata 标记——真数据慢速档,白天可跳过"
  - "cc57451: test(261022-t9o): 6 个真数据文件叠加 realdata 标记——integration 保留,叠加不替换"
files_modified:
  - pyproject.toml
  - tests/test_analysis_realdata.py
  - tests/test_evaluation_realdata.py
  - tests/test_fisher_realdata.py
  - tests/test_noise_realdata.py
  - tests/test_optimizer_realdata.py
  - tests/test_smoke_realdata.py
---

# 快任务 261022-t9o:测试套件分两档——6 个真数据测试文件打 realdata 标记 执行总结

一句话:6 个真数据测试文件(共 47 个用例,每个文件要跑十几分钟)在保留 integration 标记的同时叠加 realdata 标记,pyproject.toml 注册该标记——白天用 `uv run pytest tests -q -m "not realdata"` 只跑纯单元/快速测试(47 个真数据用例全部跳过),夜间全量档命令不变照跑;`-m "not integration"` 既有工作流逐位不变(叠加前后该口径都是 605 个)。

## 各任务执行结果

### Task 1:pyproject.toml 注册 realdata 标记(提交 5932bdc)

- `[tool.pytest.ini_options].markers` 列表照 integration 条目体例(两行中文注释 + 一句说明)追加:
  - 注释:`# realdata:标记要跑真数据解算的最慢一档(每个文件十几分钟);` / `# 白天快速档用 -m "not realdata" 跳过,夜间全量档命令不变、照跑`
  - 条目:`"realdata: 真数据集成测试,慢,夜间跑"`
- 既有 integration 条目及其注释一个字未动
- 验收:`uv run python -c "import tomllib; ..."` 解析通过,markers 恰 2 条(integration + realdata);`uv run ruff check pyproject.toml` All checks passed

### Task 2:6 个真数据文件叠加 realdata 标记(提交 cc57451)

- 6 个文件(analysis/evaluation/fisher/noise/optimizer/smoke)各改 1 行:
  `pytestmark = pytest.mark.integration` → `pytestmark = [pytest.mark.integration, pytest.mark.realdata]`
- docstring、测试函数、夹具零改动(6 文件均已 import pytest,无新增 import);计划明示 docstring 里 integration 跳过说明仍然成立,不顺手改文案——照办
- 验收:grep 逐文件复核——新列表写法各命中 1 次、旧单标记写法 0 次残留;`uv run ruff check` 6 文件 All checks passed;git diff 只出现上述 6 个文件

### Task 3:collect-only 验证两档数量(无提交——纯只读验证,无文件改动)

全程只用 `--collect-only`(从未真跑任何 realdata 用例,遵守红线),三条统计:

| 口径 | 命令 | 实测 | 计划预期 |
|------|------|------|----------|
| 全量(夜间档) | `pytest tests --collect-only -q` | **656** | 605(见偏差 1) |
| 快速档(白天) | `pytest tests -m "not realdata" --collect-only -q` | **609**(deselected 47) | 558(见偏差 1) |
| realdata 档本身 | `pytest tests -m "realdata" --collect-only -q` | **47** | 47 ✓ |

- 逐文件基数全部对上:analysis 14 / evaluation 11 / fisher 7 / noise 5 / optimizer 6 / smoke 4,合计 47 ✓
- 加和校验:609 + 47 = 656 ✓(无漏打、无写错标记名)
- 警告核查:`-m "realdata"` 收集在 stdout/stderr 两个流均零 PytestUnknownMarkWarning,退出码 0,标记注册生效
- 既有口径回归(计划必达项第 3 条):`-m "not integration"` 收集 605、deselected 51(= 47 真数据 + 4 个函数级 integration),与改动前(ikq 任务 07:04 落账的 605 passed)**逐位一致**——叠加未吞掉 integration

## 与计划的偏差(自动处理)

1. **[口径勘误] 计划预期 605/558 实为混了两个统计口径,实测 656/609**
   - 计划勘察写"全量套件 605,快速档 605 − 47 = 558"。实测全量 656:其中 605 恰是 `not integration` 口径(STATE 落账与 ikq SUMMARY 均可交叉证实),558 = 605 − 47 把"not integration 数"误当全量基数再减 47。realdata 档 47 与逐文件基数(14/11/7/5/6/4)与计划完全一致,结构目标(47 个真数据用例白天全跳过、全量行为不变、双标记叠加)全部达成,无需任何代码补救;仅预期数字本身口径错位。
2. **[任务提交粒度] 计划任务 3 原文含"按项目惯例提交"(设想单提交收口)**,实际按每任务一原子提交执行:任务 1、2 各自提交后,任务 3 为纯只读验证无文件改动,不制造空提交。最终落账提交见下。

## 提交号列表

| 提交 | 内容 |
|------|------|
| 5932bdc | pyproject.toml 注册 realdata 标记(任务 1,1 文件 +3 行) |
| cc57451 | 6 个真数据文件叠加 realdata 标记(任务 2,6 文件各 1 行) |

任务 3 无提交(只读验证)。两提交合计恰好覆盖计划 files_modified 的 7 个文件,零越界改动。

## Self-Check: PASSED

- 7 个计划文件全部在两个提交的 diff 中出现,无多余文件
- `git log` 确认 5932bdc、cc57451 两条提交存在
- 三条 collect-only 统计、逐文件基数、UnknownMark 零警告、not integration 口径回归——全部留存于本文档"各任务执行结果"
