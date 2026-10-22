---
phase: 05-error-attribution-corner-tuning
fixed_at: 2026-10-22T05:20:00Z
review_path: .planning/phases/05-error-attribution-corner-tuning/05-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 05: 代码评审修复报告

**修复时间:** 2026-10-22T05:20:00Z
**来源评审:** .planning/phases/05-error-attribution-corner-tuning/05-REVIEW.md
**迭代:** 1

**摘要:**
- 范围内发现:4(全部 Warning,fix_scope = critical_warning)
- 已修复:4
- 跳过:0

全量验证(在隔离 worktree 内对修复后代码执行):
- `pytest tests -q -m "not integration"`:596 passed, 51 deselected,0 失败
- `ruff check src tests`:All checks passed

## Fixed Issues

### WR-01: 敏感度建议分支三判据用 `min`,证据文案却断言"均低于"

**Files modified:** `src/calib/analysis/sensitivity.py`, `tests/test_analysis_sensitivity.py`
**Commit:** 8d6c7a3
**Status:** fixed: requires human verification(改动了建议分支的判定逻辑,请人工确认分档文案符合预期)
**Applied fix:** 按项目要求先写失败测试再修(红→绿)。分支三触发条件保留 `min(px_known) < _PX_NEGLIGIBLE`,但把结论拆成两档:全部已知当量低于 0.05px 影响线时,维持原文案「均低于/不治」;只要 p1/p2 有一个在线上,证据逐参数如实写「p1 低于影响线、p2 不低于影响线(画面影响不可忽略)」,建议只对低于线的参数判「不治」,线上的明说「不适用近零不治,按像素当量天平的排序正常判读」。新增反例测试 `test_advice_branch3_one_above_line_reports_truthfully`(p1=0.009px / p2=0.300px),锁死「p2 在线上时绝不允许整体断言『均低于』」。

### WR-02: "误差预算先行"名不副实——铁律门禁跑在最后

**Files modified:** `src/calib/cli/main.py`
**Commit:** 5a50c01
**Status:** fixed: requires human verification(控制流挪动,请人工确认执行顺序符合预期)
**Applied fix:** 采用评审的首选方案:把 budget 块(含失败契约 try/except 原文)整体挪到六块执行区最前、`_stage` 定义与块 2 之前,并删除原末尾的旧调用与多余的 `budget_section/budget_detail = None` 初始化(挪动后若保留会把真结果覆盖成 None)。挪动安全性经核实:budget 的输入(evaluate_report/slices.npz/sigma/model/物点/参考位姿)全部在执行区之前就绪,不依赖任何诊断块。铁律违例现在当场失败退出码 1,不再白算 15–40 分钟的角点寻优;docstring「误差预算先行」与实际执行顺序恢复一致。既有 CLI 测试(minimal tree / budget 失败契约 / 块 6 兜底 / 源码断言 budget 直调)全部通过。

### WR-03: budget 段取 evaluate 血缘时裸双下标

**Files modified:** `src/calib/analysis/budget.py`, `tests/test_analysis_budget.py`
**Commit:** 75271bd
**Status:** fixed
**Applied fix:** `evaluate_report["lineage"]["round_id"]` 改走同文件既有的 `_require` 中文报错体例(先 `_require(evaluate_report, "lineage", "evaluate.json 的 lineage 段")`,再 `_require(lineage_doc, "round_id", "lineage 的 round_id")`),缺键时给带白话指引的中文 AssertionError,不再抛裸 KeyError。新增 `test_budget_missing_lineage_raises_chinese`(先红:确认旧行为抛 `KeyError: 'lineage'`;后绿),双反例分别锁「缺 lineage 段」与「lineage 缺 round_id」。

### WR-04: 新代码注释/文案多处使用"语义"一词

**Files modified:** `src/calib/analysis/test_xyz.py`, `src/calib/analysis/residual_structure.py`, `src/calib/analysis/corner_opt.py`, `src/calib/cli/main.py`, `tests/test_analysis_test_xyz.py`, `tests/test_analysis_realdata.py`
**Commit:** 0515032
**Status:** fixed
**Applied fix:** 评审清单 13 处逐一换白话:「失败语义」→「失败时怎么处理」、「病态帧语义」→「病态帧的处理方式」、「捕获语义不变」→「捕获的异常类型不变」、「语义实测」→「作用实测」、「预期语义未获证实」→「预期作用未获证实」、「语义与收尾」→「做法与收尾」、「退出语义不变」→「退出行为不变」、「两种失败语义不许混」→「两种失败的处理方式不许混」。corner_opt 的模块 docstring 与 `SB_ACCURACY_VERDICT` 落盘常量是同一文案的两份,已同步修改保持一字不差(analyze.json 的 `sb_accuracy_verdict` 直达读者的字段随之变白话)。已用 grep 确认 6 个文件「语义」清零;仓库其余「语义」均在 05 阶段之前的旧代码(solve/models/noise/evaluation),不在本次评审范围。

## Skipped Issues

无——范围内 4 个发现全部修复。

---

_修复时间: 2026-10-22T05:20:00Z_
_修复者: Claude (gsd-code-fixer)_
_迭代: 1_
