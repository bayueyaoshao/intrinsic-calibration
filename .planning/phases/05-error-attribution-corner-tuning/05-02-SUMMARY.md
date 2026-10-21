---
phase: 05-error-attribution-corner-tuning
plan: 02
subsystem: cli + analysis + visualization + tests
tags: [analyze-command, cli-wiring, lineage, error-budget, evidence-figure, realdata-reconciliation, integration, ana-01]

# Dependency graph
requires:
  - phase: 05-error-attribution-corner-tuning (05-01)
    provides: ANALYZE_KEYS 11 键契约 + make_failure_analyze 失败契约 + save/load_analyze_report(npz 旁车)+ run_budget 复核/预算行/铁律硬报错 + AnalyzeConfig(analyze 段)
  - phase: 04-evaluation-model-selection
    provides: evaluate.json(slices/crlb_alert/selection)+ experiments/slices.npz 分组底账(quick 档只含当选模型 m2)+ 判定线落账口径
  - phase: 01~03
    provides: corners/noise/calibration/fisher 四轮产物 + noise.json:sigma_detector_corner(检测器级 σ 0.008708px)+ models 注册表 + BoardSpec
provides:
  - "calib analyze 命令端到端:上游六轮最新定位(evaluate/corners/noise/calibration/fisher/manifest)→ 失败轮拒绝 → 13 项血缘一次性登记 → budget 直调 → 六块组装落盘 → 证据图 → 中文播报;缺文件当场「上游产物缺失:…请先运行 calib <阶段>」"
  - "budget 块刻意不套 _stage 逐块兜底:铁律违例(AssertionError)冒泡到外层失败契约——typer.Exit 退出码 1 + analyze.json 落 failure_note 非空中文 + 六段全 None,坏结论不许静默出表"
  - "src/calib/visualization/analyze_figs.py:plot_budget_bars 左右双条形图(参数空间经验 σ vs CRLB 下界对数轴 + 画面影响像素当量 0.1px 参考线);只消费 budget 段现成键,图失败不拦主产物"
  - "真实数据首跑对账:run_20261021_165501 verdict=pass、逐行零违例、三统计量 102/1108/1444 倍、逐行最小比值 2.8——findings 结论 4 逐位对账通过(阶段四遗留销账)"
  - "13 条新测试(5 CLI 单测 + 3 图 + 5 集成),全套 523 passed + 42 integration 零回归"
affects: [05-03~05-06(五段分析按同契约落 _stage 接线), Phase 6(report 消费 analyze.json 五段+图)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "命令体六块结构:定位 → 拒失败轮 → 取数 → 各块(_stage 或直调)→ 组装落盘 → 出图 → 播报;血缘在取数前一次性登记全上游(防串轮的原始证据)"
    - "budget 直调不套 _stage:CRLB 口径铁律属硬红线,违例必须走失败契约退出码 1,不是「该段缺位照常出报告」"
    - "图与数据分离:plot_budget_bars 只画 analyze.json budget 段现成键,数字结论以 json 为准;k1–k3/p1/p2 不画下界条(利用率比值为 None 的同口径),近零参数就地标注「画面影响可忽略」"
    - "集成测试血缘指纹门(照 test_evaluation_realdata 体例):最新 analyze 轮的配置哈希 + 全部上游文件 sha256 与当前盘上产物一致才消费,对不上 pytest.skip 中文理由「产物轮次已更新,请重跑 calib analyze」"

key-files:
  created:
    - src/calib/visualization/analyze_figs.py
    - tests/test_analysis_cli.py
    - tests/test_analysis_realdata.py
  modified:
    - src/calib/cli/main.py
    - tests/test_visualization.py

key-decisions:
  - "budget 违例走失败契约而非逐块兜底:analyze.json 六段全 None + failure_note 强制非空中文 + 退出码 1——误差归因是纯消费命令,产出坏结论比不出结论更糟"
  - "上游 13 项文件(双 npz/json/底账/模型三件套)在取数前一次性登记进 make_lineage 的 upstream 字典;配置哈希同批登记——analyze.json 自带完整防串轮证据,集成测试逐位核对"
  - "参考位姿取 extrinsics.npz 首视图 rvecs[0]/tvecs[0](CRLB 雅可比的线性化点口径),与 fisher 轮一致;allow_pickle=False 读 npz"
  - "集成测试数字基准 = findings 结论 4(硬线 0.4207、比值 102/1108/1444±5%、最小比值 2.8±0.1),硬线另与 evaluate.json crlb_alert 逐位一致;逐行底账断言 14080 行(80 视图 × 88 角点 × 2)"

requirements-completed: [ANA-01]

# Metrics
duration: 18min
completed: 2026-10-22
---

# Phase 5 Plan 2: calib analyze 命令骨架 + ANA-01 端到端 Summary

**`calib analyze` 命令端到端落地:上游六轮定位 + 13 项血缘 + budget 直调(违例走失败契约退出码 1)+ 预算条形图;真实数据首跑 run_20261021_165501 零违例、102/1108/1444 倍,findings 结论 4 逐位对账通过,13 条新测试全套 523 passed**

## Performance

- **Duration:** 18 min
- **Started:** 2026-10-21T16:43:38Z
- **Completed:** 2026-10-22T01:01Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- calib analyze 命令体:上游六轮各找最新(corners 双 npz + noise + calibration 三件套 + fisher + evaluate 主契约与双底账 + 数据清单),缺文件当场中文报错点名先跑哪条命令、失败轮当场拒绝;13 项上游 + 配置哈希一次性登记血缘;六块组装(budget 有值 + 其余五段显式 None 进 meta.missing_sections)+ 失败契约 + 中文播报(四要素人工核查指引 + findings 结论 4 基准数)
- plot_budget_bars 证据图:左子图参数空间经验 σ vs CRLB 下界(对数轴,仅 fx/fy/cx/cy 画下界条),右子图像素当量 + 0.1px 参考线 + 近零参数「画面影响可忽略」标注;None/空行画「无数据」不冒充,图失败不拦主产物
- 真实数据首跑:run_20261021_165501(quick 档 evaluate run_20261021_131613 上游)verdict=pass、逐行零违例、三统计量 101.8/1107.6/1443.8 倍、逐行最小比值 2.756——与 findings 结论 4 逐位对账通过,阶段四「匹配口径 CRLB 重跑」遗留销账
- 测试三层:CLI 单测(monkeypatch.chdir 造假上游六轮树,自洽合成底账复核必过,失败契约/缺上游/失败轮/签名零参数断言)+ 图单测 3 例 + 真实数据集成 5 例(血缘指纹门 + 结论 4 数字对账)

## Task Commits

每个任务独立提交(执行顺序:Task 2 → Task 1 → Task 3):

1. **Task 2: plot_budget_bars 证据图** - `a011c0c` (feat)
2. **Task 1: calib analyze 命令骨架** - `06b2511` (feat)
3. **Task 3: 真实数据首跑对账 + 集成测试** - `9da81d1` (test)

**Plan metadata:** 见本计划收尾 docs 提交(SUMMARY + STATE + ROADMAP)

## Files Created/Modified

- `src/calib/visualization/analyze_figs.py` - plot_budget_bars(左右双条形图,Agg + 中文字体三件套,只消费 budget 段现成键)
- `tests/test_visualization.py` - +3 例:合成 budget 段出图/None 与空行「无数据」/中文标签与近零标注分支
- `src/calib/cli/main.py` - analyze() 命令体(上游六轮定位 + 失败轮拒绝 + 13 项血缘 + budget 直调 + 六块组装 + 出图 + 四要素播报)
- `tests/test_analysis_cli.py` - +5 例:最小上游树跑通(11 键契约 + budget pass + 五段 None + 旁车在盘 + 血缘 9 项点名)/铁律违例失败契约/缺上游中文报错/失败轮拒绝/签名零参数 + 血缘源码断言
- `tests/test_analysis_realdata.py` - +5 例集成(整文件 integration 标记):契约形状与 m2 选型/复核一致 + 零违例/硬线与结论 4 对账/最小比值 2.8 与 verdict/旁车 14080 行 + 图在盘

## Decisions Made

- budget 块直调不套 _stage 兜底:CRLB 口径铁律违例必须冒泡到外层失败契约(退出码 1 + 中文 failure_note),坏结论不许静默出表——与后续五段的「单段失败照常出报告」刻意不同
- 血缘 13 项取数前一次性登记(体例照 evaluate):analyze.json 自带完整防串轮证据,集成测试逐位核对 sha256,上游任何一轮重跑过旧轮结论即不可消费
- 集成测试跳过体例照 test_evaluation_realdata:前置产物缺失或指纹对不上 pytest.skip 中文理由,不误报失败;_latest_analyze_or_skip 供 05-03~05-06 的分析段测试复用
- 参考位姿从 extrinsics.npz 首视图取(CRLB 线性化点),不重解位姿;σ 只从 noise.json:sigma_detector_corner 读入

## Deviations from Plan

Task 2(图库)先于 Task 1(命令接线)执行——Task 1 的 CLI 测试断言 figures/budget_bars.png 落盘,图库必须先在位。仅执行顺序调换,两任务内容与提交划分不变,提交记录按计划任务号标注。

其余 None - plan executed exactly as written。

## Known Stubs

- analyze.json 五段(residual_structure / test_xyz / sensitivity / corner_opt / anisotropy)显式为 None——**计划内的刻意留位**,不是缺失:契约要求缺位段必须显式 None 并登记进 meta.missing_sections,分别由 05-03(残差结构 + test 三维)/05-04(敏感度)/05-05(角点寻优)/05-06(各向异性)按 _stage 逐块接线

## Issues Encountered

None - 三任务全绿一次通过(RED 门:集成测试先于数据产生 5 例中文跳过不误报;GREEN 门:calib analyze 首跑 exit 0,523 passed + 42 integration,零回归)。

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 05-03~05-05 接线条件齐备:analyze() 六块结构就位,每段只需实现库函数后挂进对应 None 留位(_stage 体例),明细走 analysis/<段名>.npz 旁车;集成测试的 _latest_analyze_or_skip 直接复用
- 真实数据 analyze 轮已产出(14080 行逐行底账 + 证据图),后续计划重跑 analyze 时血缘指纹门会自动要求重跑(上游未变则直接消费)

## Self-Check: PASSED

- 5 个关键产物文件全部在盘(analyze_figs.py / 两份新测试 / cli/main.py / test_visualization.py)
- 3 个任务提交全部在案:a011c0c / 06b2511 / 9da81d1
- 全套 523 passed(515 基线 + 8 单测)+ 42 integration(37 基线 + 5),ruff check src tests 通过;真实数据首跑 run_20261021_165501 verdict=pass 零违例

---
*Phase: 05-error-attribution-corner-tuning*
*Completed: 2026-10-22*
