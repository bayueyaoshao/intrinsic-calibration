---
phase: 06-cli
plan: 03
subsystem: cli
tags: [typer, run-all, pipeline-orchestration, exit-code, cli]

requires:
  - phase: 06-01
    provides: calib report 真命令(report/build.py 顶层 run_report,STEP 表直调即可)
provides:
  - run_all 八步编排重写(模块级步骤表常量 STEPS,顺序对齐 docs/pipeline.md)
  - --from/--until 流程范围开关(剔除照片重标的载体,裁定①)
  - 失败断点中文三要素指引(已完成到第几步 / 产物在哪 / 恢复用哪条命令),退出码 1
  - ingest/corners/calibrate 命令入口退出码收口(坏输入中文一句,不吐堆栈)
  - tests/test_cli_run_all.py 合成编排测试五例
affects: [06-04(真链变体验收), 主会话夜间真链验证]

tech-stack:
  added: []
  patterns:
    - "命令入口薄壳 + _run_* 实现体:实现逻辑与报错呈现分离,库层照旧抛 ValueError"
    - "步骤表常量 STEPS:步骤名 + 命令函数的单一事实源,测试 monkeypatch 整表换假函数"

key-files:
  created:
    - tests/test_cli_run_all.py
  modified:
    - src/calib/cli/main.py
    - .planning/phases/06-cli/deferred-items.md

key-decisions:
  - "失败指引刻意不自动续跑/跳过已完成步骤:血缘指纹已保证重跑安全,自动跳过反而有『上游变了下游没跑』的静默风险(T-06-08);--from 手动区间就是断点粒度"
  - "收口只包 CLI 层薄壳,库函数照旧抛 ValueError,库层既有测试零回退"
  - "失败指引的产物位置从落盘现场现读(latest_run_dir),不额外记账;指引自身绝不再失败(读不到就如实写『未找到』)"
  - "播报用全流水线绝对步数(第 i/8 步),--from 切片时用户仍知道自己在整条链的哪一环"

requirements-completed: [REP-03]

duration: 约 2 小时(其中 1 小时 29 分为 not-realdata 快速档回归等待)
completed: 2026-10-23
---

# Phase 6 Plan 03: CLI 全流水线收口 Summary

**run-all 从五步半成品收口为八步全流程一条命令(顺序对齐 docs/pipeline.md,支持 --from/--until 区间,失败播报断点与恢复命令),ingest/corners/calibrate 坏输入不再吐堆栈。**

## 做了什么

### Task 1:run_all 重写(提交 937a220)

- 模块级步骤表常量 `STEPS`:ingest→corners→noise→calibrate→fisher→evaluate→analyze→report,顺序照 docs/pipeline.md;**修正旧版 noise/calibrate 颠倒且缺后三步**(旧版只串到 fisher,还播报"后续阶段由后续版本实现"的过时文案)
- `--from`/`--until` 两个 typer.Option(默认 ingest/report),附通俗中文帮助;非法步骤名或区间倒置 → 中文一句 + 退出码 1,一步都不跑。是流程范围开关,不是算法参数,不违 D-12
- 执行循环:每步前播报"第 i/8 步:<名>(<通俗名>)";仍是直接函数调用
- 失败即停:任一步抛 typer.Exit(非零)或异常 → 三要素中文指引(已完成到第几步 + 已完成步骤的最新产物目录[latest_run_dir 现读] / 恢复命令 `calib run-all --from <失败步>` 或单步 `calib <步>`),退出码 1,绝不静默续跑(T-06-06)
- 新建 `tests/test_cli_run_all.py`:先落顺序锁死 / --from calibrate --until fisher 切片 / 非法名+区间倒置 三例

### Task 2:三命令退出码收口 + 合成测试补齐(提交 cb9df9c)

- ingest/corners/calibrate 实现体改名 `_run_ingest`/`_run_corners`/`_run_calibrate`,命令入口改为薄壳:try/except → 一句中文原因 + "先检查 config/上游产物"指引 + `typer.Exit(code=1) from e`,对齐 noise/fisher/evaluate/analyze 的失败契约体例;库函数照旧抛 ValueError,库层零回退
- `tests/test_cli_run_all.py` 追加两例:假 evaluate 抛 typer.Exit(1) → run-all 退出码 1、播报含"已完成到""恢复""--from evaluate"、失败步之后的 analyze/report 不被调用;CliRunner 调 ingest 坏 config → 退出码 1、输出含中文指引、不含 "Traceback"

## 验证数字

| 检查 | 结果 |
|---|---|
| tests/test_cli_run_all.py | **5 passed**(顺序/切片/非法名+倒置/失败断点/退出码收口) |
| tests/test_cli_main.py | **1 passed**(零回退,与执行前基线一致) |
| `.venv/bin/ruff check src tests` | All checks passed |
| `calib run-all --help` | --from/--until 中文说明正常渲染 |
| `-m "not realdata"` 快速档(绕开当时在建的 test_report_loopback.py) | **615 passed / 4 failed**,4 个失败全部发生在 test_report_build.py(06-02 并行中间态,详见"偏差");06-02 收口后单独复跑该文件 **5 passed 全绿** |
| 此前登记的 test_noise_injection 注入慢测试 | 本轮未再出现失败 |
| 耗时 | 快速档全程 1:29:26(含已知注入慢测试);进程内合成测试 5+1 例合计秒级 |

## 提交

| 提交 | 内容 |
|---|---|
| 937a220 | feat(06-03): run-all 八步收口——顺序对齐 pipeline.md + --from/--until 区间 + 断点中文指引 |
| cb9df9c | feat(06-03): ingest/corners/calibrate 退出码收口——坏输入中文一句 + 退出码 1,不吐堆栈 |
| (本次) | docs(06-03): SUMMARY + deferred-items 落账 |

## 偏差

1. **[Rule 2 - 补齐]收口范围从 ValueError 扩为 (ValueError, OSError)**:计划写"try/except ValueError";实现连配置文件缺失等读盘失败(OSError)一并收口——坏输入走堆栈正是 T-06-07 要堵的口子,库层照旧抛 ValueError 不受影响。
2. **[范围外观察,已登记 deferred-items]** 快速档跑出 4 个 test_report_build.py 失败:06-03 的套件 02:41 收集,06-02 的两个提交 02:42/02:47 才落盘,属并行执行中间态;06-02 收口(d5a75fc)后复跑全绿。**非本计划问题,未越界修**,全过程已登记 `.planning/phases/06-cli/deferred-items.md`。
3. 其余按计划执行,无架构偏差;未触碰 06-02 范围(src/calib/report/*、tests/test_report_*.py 只跑未改)。

## 边界说明

- 本计划只锁编排逻辑(合成测试,不真跑标定);`calib run-all` 真链端到端(含真跑八步)按分工由 06-04 变体与主会话夜间档人工验收,本计划未真跑 run-all。
- REP-03 的 run-all 载体在本计划落地;成功标准 1 的整体验收在 06-04。

## Self-Check: PASSED

- 文件三处全在:tests/test_cli_run_all.py、src/calib/cli/main.py、本 SUMMARY
- 提交两笔全在:937a220(Task 1)、cb9df9c(Task 2)
- 终态复跑:tests/test_cli_run_all.py + tests/test_cli_main.py 合计 6 passed
