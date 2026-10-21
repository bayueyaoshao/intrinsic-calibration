---
phase: 05-error-attribution-corner-tuning
plan: 03
subsystem: analysis + visualization + cli + tests
tags: [residual-structure, radial-bands, coverage-footprint, extrapolation-disclosure, r4-law, findings-5, ana-02]

# Dependency graph
requires:
  - phase: 05-error-attribution-corner-tuning (05-02)
    provides: analyze 命令六块骨架 + _stage 逐块兜底体例 + budget 段(k2 像素当量同源底账)+ analysis/*.npz 旁车契约 + _latest_analyze_or_skip 取数助手
  - phase: 04-evaluation-model-selection
    provides: evaluate.json delivery.final_params(r 归一化基准)+ slices.empirical_std(k2 经验 σ,与 budget 同源)
  - phase: 01~03
    provides: calibration 三件套(residuals/extrinsics/result 行序契约)+ corners 双 npz(load_observation_set 装载)+ solve_pose_only 只估位姿评分 + models 统一投影
provides:
  - "src/calib/analysis/residual_structure.py:radius_footprint(归一化半径 r + r⁴ 信息权重)/assign_bands(左闭右开分带,超界不丢单独计数)/band_stats(系统偏移与散布分离——散布按减本带均值后算,整带同向挪动不改 rms)/coverage_summary(最远观测 r / 画面四角 r / 外推区间 / r>0.5 观测占比与 k2 信息份额)/k2_wobble_px(r⁴ 律折算)/run_residual_structure(train 读落盘残差 + test 只估位姿回代,失败帧 NaN 占行)"
  - "analyze 块 2 接线:residual_structure 段有值(_stage 兜底,budget 仍直调);meta.r_normalization_basis 透传;residual_structure.npz 旁车 + radial_bands.png / coverage_footprint.png 两张空间图;missing_sections 收缩到 4 段"
  - "plot_radial_bands / plot_coverage_footprint:每张空间图统一叠观测足迹 + 外推区阴影 + r 等值线(RESEARCH §Pattern 2 既定样式);分带器一处实现,COR-03(05-06)直接复用"
  - "真实数据对账 run_20261021_173659:r_max_obs 0.6853 / 四角 0.7974 / r>0.5 观测 21.1% 承载 k2 信息 61.4%——findings 结论 5 逐位吻合;归一化基准与标定轮 max_rel_diff 0.0(consistent)"
  - "16 条新测试(10 库层 + 6 图 + 4 集成扩展,其中 1 例源码断言锁两种失败语义不混),全套 539 passed + 46 integration 零回归"
affects: [05-04(test_xyz/sensitivity 两段同体例接线), 05-05 角点寻优, 05-06 anisotropy(复用 assign_bands/band_stats 分带器), Phase 6(report 消费 residual_structure 段+双图)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "同一把尺纪律(D-06):train 侧残差直接读标定落盘 residuals.npz(不重解),test 侧逐帧 solve_pose_only 只估位姿后经 model.project 回代——本模块零投影/导数/位姿公式,绝不 import OpenCV(源码级断言锁死)"
    - "两种失败语义分工:块 1 budget 直调(铁律违例冒泡退出码 1),块 2 起 _stage 逐块兜底(单块失败记 None 照常出报告);块内取数也在兜底里(上游角点装不下只垮这一块)"
    - "外推披露不抄死 findings 数字:判据表三行数值全部由公式从当前数据现算(σ 取 slices.empirical_std 与 budget 段同源 0.0966px),findings 的 0.135px 是最大组间散布口径、两者各说各话不混用"
    - "归一化基准随产物声明:r_normalization_basis = evaluate delivery final_params @ 轮名,并与 findings 结论 5 所用标定轮 fx/fy/cx/cy 做相对差核对(漂移中文披露不静默,非阻塞)"

key-files:
  created:
    - src/calib/analysis/residual_structure.py
    - tests/test_analysis_residual.py
  modified:
    - src/calib/visualization/analyze_figs.py
    - src/calib/cli/main.py
    - tests/test_visualization.py
    - tests/test_analysis_realdata.py

key-decisions:
  - "band_stats 的散布统计按减去本带均值向量后的残差长度计算——计划验收明文「mean_du 系统偏移不改变 rms」,系统偏移记在 mean_du/mean_dv、乱散记在 rms,两套各说各话(实现初版按原始长度算 rms,对照计划验收后纠正)"
  - "residual_structure 段补 image_size 兼容键:覆盖足迹图的画幅范围与画面四角坐标需要它,段是图的唯一数据源(签名三参数不再另传);加键不改键,契约兼容"
  - "k2 晃动折算的 σ 源取 slices.empirical_std(±σ 扰动投影换算当量,与 budget 段完全同源 0.0966px),不取 findings 结论 5 的 0.135px(那是最大组间散布口径);判据表数值现算,findings 数字只做量级对照"
  - "train 侧行序契约三重断言:residuals.npz ↔ extrinsics.npz 逐行同序、观测集 ↔ residuals 照片编号一致、数组形状对齐——残差对齐的唯一保证,错了当场中文报错不算"
  - "test 侧失败帧 NaN 占行不中断(角点 <4 抛 ValueError 转 NaN 行 + n_failed 计数);r_test 照算(r 是观测侧量,不依赖求解成败)"

requirements-completed: [ANA-02]

# Metrics
duration: 30min
completed: 2026-10-22
---

# Phase 5 Plan 3: ANA-02 残差方向分解端到端 Summary

**残差按视图/画面区域/r 分带切方向 + r⁴ 覆盖足迹产品化(findings 结论 5):真实数据首跑 r_max_obs 0.6853、四角 0.7974、r>0.5 观测 21.1% 承载 k2 信息 61.4% 逐位吻合,外推区 [0.69, 0.80] 显式披露(占画面 6.3%,k2 晃动折到四角 0.177px),16 条新测试全套 539 passed**

## Performance

- **Duration:** 30 min
- **Started:** 2026-10-21T17:14:14Z
- **Completed:** 2026-10-21T17:44Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- residual_structure 库层:radius_footprint(r 与 r⁴ 信息权重)、assign_bands(左闭右开,超最外边界观测归最外带并单独计数不丢)、band_stats(系统偏移 mean_du/mean_dv 与散布 rms/median/max 分开,散布按减本带均值后算)、coverage_summary(最远观测 r、画面四角 r、外推区间、r>0.5 观测占比与 k2 信息份额)、k2_wobble_px(r⁴ 律折算)、run_residual_structure 编排(train/test 双侧统计 + 外推披露数值块与判据表)
- analyze 块 2 接线:_stage 逐块兜底(块内含观测集取数),明细走 residual_structure.npz 旁车,meta 透传 r_normalization_basis;radial_bands.png(分带散布 + 系统偏移双图,外推阴影)与 coverage_footprint.png(足迹散点 + r=0.4/0.5/0.685/0.75 等值线 + 外推环带 + 画面四角标记,标题注明归一化基准轮名)两张空间图,四要素播报追加 residual_structure 段指引
- 真实数据首跑 run_20261021_173659:最远观测 r=0.6853、画面四角 r=0.7974、外推区间 [0.6853, 0.7974](占画面像素 6.3%)、r>0.5 观测 21.1% 承载 k2 信息 61.4%——findings 结论 5 全部数字逐位吻合;归一化基准(evaluate delivery final_params @ run_20261021_131613)与标定轮 fx/fy/cx/cy 逐位一致(max_rel_diff 0.0);test 65 张全评分零失败(逐帧 RMS 中位 0.119px);外推披露 k2 当量 0.0966px 与 budget 段同源,折到四角 0.177px、中区 r≤0.4 约 0.011px 现状够用
- 测试三层:库层 10 例(3-4-5 手算、左闭右开与超界计数、平移不变散布、信息份额手算、r⁴ 律 findings 数字、端到端合成真值含失败帧占行、行序契约报错、单一数学源源码断言)+ 图 6 例(出图/无数据/全域覆盖不画阴影/源码含「外推」)+ 集成扩展 4 例(足迹与信息份额对账、分带与 test 侧、旁车双图、块 2 走 _stage 而 budget 直调的源码断言)

## Task Commits

每个任务独立提交(另有一例 Task 2 图库的后续一行修正):

1. **Task 1: 残差方向分解库层** - `7f8bf88` (feat)
2. **Task 2: 空间图表两张(r 分带曲线 + 覆盖足迹图)** - `2604094` (feat)
3. **Task 3: analyze 接块 2 + 真实数据对账** - `1196bed` (test→feat 实质接线)
4. **Task 2 后续修正: 足迹图 aspect 消警告** - `a08cf34` (fix)

**Plan metadata:** 见本计划收尾 docs 提交(SUMMARY + STATE + ROADMAP)

## Files Created/Modified

- `src/calib/analysis/residual_structure.py` - ANA-02 全部数学与编排(分带器一处实现,供 05-06 anisotropy 复用;零 OpenCV、零手写投影)
- `tests/test_analysis_residual.py` - +10 例库层行为测试(合成零畸变真值端到端 + 手算对账 + 源码红线)
- `src/calib/visualization/analyze_figs.py` - +plot_radial_bands / plot_coverage_footprint(空间图统一叠足迹 + 外推阴影的既定样式)
- `tests/test_visualization.py` - +6 例:两图出图/无数据/全域覆盖/源码「外推」断言
- `src/calib/cli/main.py` - analyze 块 2 _stage 接线 + 明细旁车挂载 + meta 基准透传 + 双图 + 播报扩展
- `tests/test_analysis_realdata.py` - missing_sections 收缩为 4 段 + residual 段断言,+4 例集成(结论 5 对账/分带与 test 侧/旁车双图/失败语义源码断言)

## Decisions Made

- 散布与偏移分离按计划验收口径落地:band_stats 先减本带均值再取长度算 rms/median/max,整带同向挪动只动 mean_du/mean_dv——「整体偏」与「乱散」在数字上不可混淆
- 外推披露 σ 源与 budget 同源(empirical_std ±σ 扰动,0.0966px),不用 findings 的 0.135px(最大组间散布口径);判据表数值全部现算,findings 数字只做量级对照——同一批产物里同一参数只有一个 σ 口径
- r 归一化基准声明式披露:basis 字符串随段与 meta 双落账,与 findings 所用标定基准做相对差核对(当前 0.0 一致);上游换轮漂移只披露不阻塞,跨轮比较的口径责任交给读者
- 覆盖足迹图所需画面尺寸以 residual_structure 段新增 image_size 兼容键承载(图签名只有三参数,段是唯一数据源)

## Deviations from Plan

**1. [Rule 1 - Bug] band_stats 散布口径纠正(实现初版与计划验收不符)**
- **Found during:** Task 1
- **Issue:** 初版 rms/median/max 按原始残差长度算,整带同向偏移会推高 rms,与计划验收「mean_du 系统偏移不改变 rms」直接冲突(测试当场抓住)
- **Fix:** 散布统计改为减去本带均值向量后计算
- **Files modified:** src/calib/analysis/residual_structure.py
- **Commit:** 7f8bf88(随 Task 1 一并落库)

**2. [Rule 3 - 阻塞修复] residual_structure 段补 image_size 键**
- **Found during:** Task 2
- **Issue:** 覆盖足迹图要画画面四角标记与画幅范围,但段契约没有画面尺寸,函数签名(plan 定死三参数)也没有位置再传
- **Fix:** 段新增 image_size 兼容键(加键不改键),Task 2 提交一并带上
- **Files modified:** src/calib/analysis/residual_structure.py
- **Commit:** 2604094

**3. [Rule 1 - Bug] 覆盖足迹图 aspect 与手动限轴冲突**
- **Found during:** Task 3(真实数据首跑刷「Ignoring fixed x/y limits」警告)
- **Issue:** set_aspect("equal", adjustable="datalim") 会改写调用方设定的数据范围
- **Fix:** 改 adjustable="box"(像素等比由轴框形状吸收),修正后重跑 analyze 使最新轮出自最终代码
- **Files modified:** src/calib/visualization/analyze_figs.py
- **Commit:** a08cf34

其余 None - plan executed exactly as written。

## Known Stubs

None - 本计划五段留位中的 residual_structure 段已接线为真实数据,余下四段(test_xyz/sensitivity/corner_opt/anisotropy)仍是计划内的显式 None 留位(逐计划接入,meta.missing_sections 如实登记)。

## Issues Encountered

None - 三任务全绿一次通过(库层测试在实现后就地抓住散布口径偏差,端到端合成真值 test 侧回代残差 <0.01px;真实数据首跑 exit 0 警告清零,539 passed + 46 integration 零回归)。

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 05-04 接线条件齐备:test_xyz 与 sensitivity 两段按同体例挂 _stage,集成测试复用 _latest_analyze_or_skip(血缘指纹门会因本轮 analyze 已重跑而自动消费最新轮 run_20261021_173659)
- 分带器(assign_bands/band_stats)已就位,05-06 连拍各向异性直接复用同一把尺(严禁第二份实现的源码红线已写进模块头)
- 覆盖足迹 + 外推阴影的空间图样式已定型,后续空间图照此叠加;真实信号提示:test 侧最外带 RMS 0.39px 显著高于 train 0.14px,可供 05-04 test 三维分解对照

## Self-Check: PASSED

- 6 个关键产物文件全部在盘(residual_structure.py / test_analysis_residual.py / analyze_figs.py / cli/main.py / 两份既有测试的扩展)
- 4 个任务提交全部在案:7f8bf88 / 2604094 / 1196bed / a08cf34
- 全套 539 passed(523 基线 + 16 新)+ 46 integration(42 基线 + 4),ruff check src tests 通过;真实数据首跑 run_20261021_173659 residual_structure 段有值、结论 5 逐位吻合、旧轮 run_20261021_165501 已按及时清理规则删除

---
*Phase: 05-error-attribution-corner-tuning*
*Completed: 2026-10-22*
