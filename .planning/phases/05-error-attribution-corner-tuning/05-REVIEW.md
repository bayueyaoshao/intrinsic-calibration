---
phase: 05-error-attribution-corner-tuning
reviewed: 2026-10-22T04:19:01Z
depth: standard
files_reviewed: 24
files_reviewed_list:
  - config/calibration.yaml
  - docs/findings.md
  - docs/pipeline.md
  - src/calib/analysis/anisotropy.py
  - src/calib/analysis/artifacts.py
  - src/calib/analysis/budget.py
  - src/calib/analysis/corner_opt.py
  - src/calib/analysis/residual_structure.py
  - src/calib/analysis/sensitivity.py
  - src/calib/analysis/test_xyz.py
  - src/calib/cli/main.py
  - src/calib/io/config.py
  - src/calib/visualization/analyze_figs.py
  - tests/test_analysis_anisotropy.py
  - tests/test_analysis_artifacts.py
  - tests/test_analysis_budget.py
  - tests/test_analysis_cli.py
  - tests/test_analysis_corner_opt.py
  - tests/test_analysis_realdata.py
  - tests/test_analysis_residual.py
  - tests/test_analysis_sensitivity.py
  - tests/test_analysis_test_xyz.py
  - tests/test_io_config.py
  - tests/test_visualization.py
findings:
  critical: 0
  warning: 4
  info: 2
  total: 6
status: issues_found
---

# Phase 05: Code Review Report

**Reviewed:** 2026-10-22T04:19:01Z
**Depth:** standard
**Files Reviewed:** 24
**Status:** issues_found

## Summary

对第⑦步误差归因(ANA-01~04 + COR-02/03)全链 24 个文件做了逐文件审查:11 个源码/配置文件、2 份文档、11 个测试文件。整体质量高:CRLB 口径铁律在三层落地(budget 的 `_check_sigma_detector` 硬校验、逐行实测/下界零违例硬断言、anisotropy 防零除下限取检测器级 σ),反序列化安全面干净(全部 `np.load(..., allow_pickle=False)`、仅 `yaml.safe_load`,ruff 零告警),失败契约(铁律违例退出码 1 vs 单块失败记 None)有测试双向锁定,并行/串行逐位一致有回归测试。

审查中发现 4 个警告级、2 个信息级问题。最值得修的是敏感度建议分支三的条件与证据文案不一致(可能对高于影响线的参数给出"均低于/不治"的错误结论),以及"误差预算先行"的文档声明与实际执行顺序相反(铁律门禁跑在 15–40 分钟的重计算块之后,违例时把已算完的段落全部丢弃)。无 Critical 级问题:两个初始候选(逐帧异常网太窄、`.values()` 按序解包)经交叉验证均被证伪——`np.linalg.LinAlgError` 继承自 `ValueError`,逐帧捕获网实际覆盖了 numpy 求解器异常面;`_require_delivery_params` 在同模块内以固定键序自建字典,解包顺序当前有保证(仍记 Info 防将来改动)。

## Narrative Findings (AI reviewer)

### Warnings

### WR-01: 敏感度建议分支三判据用 `min`,证据文案却断言"均低于"

**File:** `src/calib/analysis/sensitivity.py:274-283`
**Issue:** 分支三的触发条件是 `min(px_known) < _PX_NEGLIGIBLE`(第 274 行)——只要 p1、p2 中**任意一个**像素当量低于 0.05px 就开火;但落盘的证据文案(第 282-283 行)写的是"均低于 0.05px 的影响线",advice 给出的裁决是"不治:画面影响可忽略"。当 p1=0.009px 而 p2=0.30px 时,分支照样触发,analyze.json 会白纸黑字写下"p1 0.0090px / p2 0.3000px,均低于 0.05px 的影响线"——对 p2 是一句假话,且对一个画面影响 0.30px(6 倍于影响线)的参数给出"不治"结论,直接误导人工判读。现有测试 `test_advice_branch3_nearzero_untreated`(tests/test_analysis_sensitivity.py:206)只喂了"两个都低于"的输入,恰好测不到这条岔路。
**Fix:**

```python
# 判据与文案对齐:要断言"均低于",就必须两个都低于才开火
if visible and px_known and max(px_known) < _PX_NEGLIGIBLE:
    ...
# 或者保留 min 触发、但文案按实情逐参数写:
#   f"p1 {px_p1:.4f}px(低于)/ p2 {px_p2:.4f}px(高于)0.05px 影响线"
```

并补一个"恰好一个高于影响线"的反例测试锁住行为。

### WR-02: "误差预算先行"名不副实——铁律门禁跑在最后,违例时丢弃全部已算段落

**File:** `src/calib/cli/main.py:1822` vs `src/calib/cli/main.py:2089-2099`
**Issue:** analyze 命令的 docstring(第 1822 行)写"误差归因一条龙(误差预算先行,六块全链)",但实际执行顺序是:块 2 残差分解 → 块 3 test 三分量 → 块 4 敏感度 → 块 5 角点寻优(第 2060-2062 行自述 quick 档 15–40 分钟)→ 块 6 各向异性 → **最后**才在第 2090 行调 `run_budget`。后果:① 预算内的铁律核查(逐行实测低于理论下界即硬失败)是整条命令的"口径对不对"门禁,却排在全部重计算之后——门禁一旦违例,块 2–6 已经算完的几十分钟成果被 `make_failure_analyze` 整体置 None 丢弃(第 2094-2099 行失败契约),纯粹浪费;② 文档与行为相反,下一位维护者按 docstring 理解执行顺序会得出错误结论。测试 `test_analyze_blocks_2345_use_stage_and_budget_stays_direct` 只锁了"budget 直调不套 _stage",没锁顺序。
**Fix:** 把 budget 块(或至少 `run_budget` 里廉价的 crlb_alert 复核部分)挪到块 2 之前执行,门禁先过、诊断后跑;若顺序是有意为之(例如 budget 需要块 2 的产物——经查并不需要,其输入只有 evaluate_report/slices.npz/sigma/model/object_points/ref 位姿),则至少把 docstring 的"误差预算先行"改为与实际一致的表述。

### WR-03: budget 段取 evaluate 血缘时裸双下标,绕开本文件 `_require` 中文报错防线

**File:** `src/calib/analysis/budget.py:359`
**Issue:** `"evaluate_round": evaluate_report["lineage"]["round_id"]` 是整个 `run_budget` 里唯一一处裸字典双下标取数;同函数内其余上游取数全部走 `_require(...)`(如第 346、348 行),缺键时给带白话指引的中文 `AssertionError`。一旦上游 evaluate.json 缺 `lineage` 或 `round_id`(旧轮产物、手改文件),这里会抛裸 `KeyError: 'round_id'`——英文堆栈,违反项目"缺上游当场中文报错"的硬性契约,而 `run_budget` 自身入口并不预检 lineage(CLI 层只查了 `failure_note`)。
**Fix:**

```python
lineage_doc = _require(evaluate_report, "lineage", "evaluate.json 的 lineage 段")
"evaluate_round": _require(lineage_doc, "round_id", "lineage 的 round_id"),
```

### WR-04: 新代码注释/文案多处使用"语义"一词,违反 CLAUDE.md 文档用语硬性约束

**File:** `src/calib/analysis/test_xyz.py:49,116`;`src/calib/analysis/residual_structure.py:47`;`src/calib/analysis/corner_opt.py:11,14,71,73,75,87,151`;`tests/test_analysis_test_xyz.py:102`;`tests/test_analysis_realdata.py:259`(另 `src/calib/cli/main.py:1047` 亦有,疑为前期遗留)
**Issue:** CLAUDE.md 把"避免软件工程行话(如「切片/闭包/runner/语义」这类英语直译词)"列为硬性约束。本阶段新增代码的注释与随段落盘文案里"语义"出现了约 15 处,如"失败语义""病态帧语义""捕获语义不变""CALIB_CB_ACCURACY 语义实测""预期语义未获证实"。这些词会原样进入 analyze.json 的 `sb_accuracy_verdict` 等交付字段,最终到达零基础读者面前。注:"空间切片/r 分带切片"中的"切片"是空间分带的物理比喻且上下文有解释,不计入本条。
**Fix:** 逐处改白话:"失败语义"→"失败时怎么处理";"病态帧语义"→"病态帧的处理方式";"捕获语义不变"→"捕获的异常类型不变";"语义实测"→"作用实测/实际效果";"预期语义未获证实"→"预期的作用未获证实"。

### Info

### IN-01: `_median_max_bars` 文档写"缺值跳过",代码却把缺值画成 0 高度条

**File:** `src/calib/visualization/analyze_figs.py:355,366`
**Issue:** docstring(第 355 行)承诺"缺值跳过",第 366 行实际执行 `vals.append(float(v) if v is not None else 0.0)`——缺值不跳过,而是画一根 0 高度的实心条,图例上仍是"中位/最大",读图人会当成"误差为零"。这与本文件其余画图函数(及 tests/test_visualization.py 多处断言)坚守的"无数据不冒充 0"惯例相悖。当前为潜伏问题:真实 `run_test_xyz` 段只在整段走"无数据"分支时才出现 None,此路径尚不可达;但任何未来调用方传入部分缺值的段落就会踩中。
**Fix:** 缺值改画为 NaN 并在轴上注明该档缺值(`vals.append(float(v) if v is not None else np.nan)` + 文本标注),或如实把 docstring 改为"缺值按 0 画"——但前者才符合项目惯例。

### IN-02: `run_corner_opt` 按 `.values()` 顺序解包内参,与取数助手的内部字典键序隐性耦合

**File:** `src/calib/analysis/corner_opt.py:452`
**Issue:** `fx, fy, cx, cy = _require_delivery_params(evaluate_report, model_id).values()` 依赖第 196-215 行助手自建字典时恰好按 `("fx", "fy", "cx", "cy")` 键序插入。当前正确(同模块、键序固定),但这是一条没有类型保护的隐性契约:将来有人给助手的键清单加一个键或调换顺序,这里会把 cx 值静默装进 fy——r 归一化足迹、帧抽样分层全部悄然失真,且没有任何断言会叫。返回值消费点与构造点相隔 250 余行,靠 `.values()` 对齐纯属易碎设计。
**Fix:** 助手改返回四元组 `return (float(final[n]) for n in ("fx","fy","cx","cy"))` 并按元组解包;或调用点显式按键取:`d = _require_delivery_params(...); fx, fy, cx, cy = d["fx"], d["fy"], d["cx"], d["cy"]`。

---

_Reviewed: 2026-10-22T04:19:01Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
