---
phase: 06-cli
fixed_at: 2026-10-23T00:00:00+08:00
review_path: .planning/phases/06-cli/06-REVIEW.md
iteration: 1
findings_in_scope: 8
fixed: 8
skipped: 0
status: all_fixed
---

# 阶段六:代码评审修复报告

**Fixed at:** 2026-10-23
**Source review:** .planning/phases/06-cli/06-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 8(全部 4 条 WARNING + 指派顺手修的 3 条 INFO)
- Fixed: 8
- Skipped: 0
- 范围外维持不改:IN-02、IN-04、IN-06、IN-07(按修复指令明确不修;IN-07 评审已确认 report 层未加重,维持登记)

## Fixed Issues

### WR-01: burst `threshold_px` 是二元列表,报告显示退化 + 候选理由丢容差

**Files modified:** `src/calib/report/build.py`、`src/calib/report/diff.py`、`tests/test_report_build.py`、`tests/test_report_diff.py`
**Commit:** ccd0ca9
**Status:** fixed
**Applied fix:** diff.py 新增 `_fmt_threshold`、build.py 新增 `_fmt_threshold_px`:真实形状
`[中位容差, 最大容差]` 二元列表渲染成白话「中位容差 X px / 最大容差 Y px」(报告③节与
候选理由两处;裸标量兜底照读,不再出现裸列表或「容差 —」丢数字)。两处合成测试的
`threshold_px` 从标量 `1.0` 改喂真实二元列表 `[0.35, 1.4]` 锁形状,并加白话渲染断言。

### WR-02: 连拍候选无视生产端已点名的 `suspect_image_ids`,整组稀释人工名单

**Files modified:** `src/calib/report/diff.py`、`tests/test_report_diff.py`
**Commit:** ae0ad1b
**Status:** fixed: requires human verification(候选选取逻辑变化,语义请人工确认)
**Applied fix:** `_burst_candidates` 读 failures 条目的 `suspect_image_ids`(生产端
`_burst_failures_for` 按「偏离超最大容差」点名的跳位帧):有名单逐张点名、理由沿用;
名单缺席(旧轮产物)退回原整组展开。同姿态两颗检测器各点名一次仍按姿态去重收一组。
补「逐帧点名优先」用例;原整组展开用例转为兜底路径用例。

### WR-03: `calib report` 命令漏做退出码收口,坏输入吐全堆栈

**Files modified:** `src/calib/cli/main.py`、`tests/test_cli_run_all.py`
**Commit:** 31797ca
**Status:** fixed
**Applied fix:** 照 06-03 体例(T-06-07):实现体挪进 `_run_report`,`report` 命令入口
包 `except (ValueError, OSError)` → 「汇总证据出报告失败:{原因}」+ 指引「按流水线顺序
补跑对应命令」+ `typer.Exit(code=1)`,坏输入不再吐堆栈;run-all 的失败即停逻辑
(捕 typer.Exit)不受影响。补空目录坏输入收口用例(退出码 1、中文一句、无 Traceback)。

### WR-04: 真实 STEPS 表的八步顺序没有快速档测试锁定

**Files modified:** `tests/test_cli_run_all.py`
**Commit:** a9fdb33
**Status:** fixed
**Applied fix:** 新增 `test_real_steps_table_matches_pipeline_order`:不经假表直接断言
`[name for name, _ in cli_main.STEPS] == PIPELINE_ORDER`,真实表被改乱序/漏一步时白天
快速档当场红;顺带核 `_STEP_CN` 八步键齐、`_STEP_OUTPUT_DIR` 键都在真实表内。

### IN-01 + IN-03 + IN-05(合并一个提交)

**Files modified:** `src/calib/cli/main.py`、`src/calib/report/diff.py`、`tests/test_report_diff.py`
**Commit:** 0706379
**Status:** fixed(IN-05 为候选去重逻辑变化,请人工顺眼确认)
**Applied fix:**
- IN-01:`_run_report` docstring「落盘四件」改「五件」,补 `suggest_exclude.json`
  (与 test_run_all_variant_realdata.py 按五件验收对齐)。
- IN-03:diff.py 两处注释行话「语义」换白话——「方向语义按指标含义写死」改
  「变化方向按指标含义写死」,docstring 同款(_change_word)。
- IN-05:`collect_exclude_candidates` 三路并集按 `image_id` 去重——同一张照片被多路
  同时点名只收先到的一条,同张照片不再重复上榜;`image_id` 缺席的整组候选不参与
  去重照收。补三路重叠去重用例。

## Skipped Issues

无——范围内 8 条全部修复。

范围外(按修复指令不改,评审原文意见留档):
- IN-02(build.py:399 summary 取法):不改。
- IN-04(diff.py higher 方向分支死代码):不改。
- IN-06(build.py `_src` 绝对路径进底账):不改。
- IN-07(同秒两轮覆盖):评审确认 report 层未加重,维持登记不改。

## 验证记录

- 每条修复:`py_compile` 语法检查通过 + 改动区重读确认 + 相关测试文件全绿。
- 最终汇总:`pytest tests/test_report_build.py tests/test_report_loopback.py
  tests/test_report_diff.py tests/test_cli_run_all.py tests/test_cli_main.py -q`
  → **36 passed, 0 failed**。
- 注:worktree 内无独立 .venv,测试经主仓库 .venv + `PYTHONPATH=<worktree>/src`
  运行,确保被测代码是 worktree 的修复版而非 editable 安装指向的主仓库旧码。

---

_Fixed: 2026-10-23_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
