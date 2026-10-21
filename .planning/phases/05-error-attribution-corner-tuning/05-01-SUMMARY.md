---
phase: 05-error-attribution-corner-tuning
plan: 01
subsystem: analysis + config + tests
tags: [analyze-contract, error-budget, crlb, detector-sigma, npz-sidecar, failure-contract, ana-01]

# Dependency graph
requires:
  - phase: 04-evaluation-model-selection
    provides: evaluate.json 13 键产物(slices/delivery/bootstrap/source_separation/optimizer 段)+ experiments/slices.npz 分组底账(mX__group_uv / mX__group_bound_px / mX__group_params)+ crlb_alert 落账口径(硬 0.4207/软 0.5564 双线)与 _px_dispersion 像素当量同尺
  - phase: 01~03
    provides: noise.json 顶层键 sigma_detector_corner(检测器级 σ 实测 0.008708px)、models 注册表(create_model)、BoardSpec 与棋盘物点网格
provides:
  - "AnalyzeConfig(analyze 段 16 键,frozen dataclass + 逐字段中文校验;config/calibration.yaml 落段,what_if 键声明仅 full 档生效=裁定②)"
  - "src/calib/analysis/artifacts.py:ANALYZE_KEYS 11 键契约 + make_failure_analyze 失败契约(六段 None + 中文说明强制非空)+ save/load_analyze_report(六份明细转 analysis/<段名>.npz 旁车 + .meta.json 血缘,json 只留 detail_npz 引用;读回 allow_pickle=False)"
  - "src/calib/analysis/budget.py:run_budget 纯消费零求解——三统计量独立重算与 evaluate 落账 isclose(1e-9) 复核、逐参数预算行(经验标准差 + 像素当量 + bootstrap/位姿采样/多起点列)、逐行铁律硬报错(实测低于下界或比值跌破硬线当场中文 AssertionError 附 σ 口径排查指引)"
  - "k1–k3 与 p1/p2 利用率比值置 None + crlb_caliber_note 口径声明随表(findings 结论 4:单独下界被参数间补偿空间撑大,比值无意义)"
  - "26 条新测试(test_io_config 8 + test_analysis_artifacts 9 + test_analysis_budget 9),全套 515 passed 零回归"
affects: [05-02(calib analyze 命令接线 ANA-01 端到端), 05-03~05-05(五段分析同契约落盘), Phase 6(report 消费 analyze.json)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "预算判定与 evaluate 同一条判定线:统计量从 slices.npz 独立重算与 crlb_alert.per_param 落账逐位对账(isclose rel_tol=1e-9),漂移即「口径漂移」报错——不重写第二套判据"
    - "σ 来源单点化:run_budget 不自己读文件,只认调用方从 noise.json:sigma_detector_corner 读入的值,坏值在进任何取数前拒绝"
    - "analysis 侧产物契约整文件照 evaluation/artifacts.py 体例,不跨模块借私有件(各阶段契约各自成文);测试同样本地复制对照工具(_assert_reports_equal/统计量键名)"
    - "「自洽合成底账」测试法:slices.npz 与 evaluate 落账比值从同一组数组算出使复核必过,8 组等差构造让标准差可手算(步长×√6),再按篡改方向逐个击破"

key-files:
  created:
    - src/calib/analysis/artifacts.py
    - src/calib/analysis/budget.py
    - tests/test_analysis_artifacts.py
    - tests/test_analysis_budget.py
  modified:
    - src/calib/io/config.py
    - config/calibration.yaml
    - tests/test_io_config.py

key-decisions:
  - "复核 = 独立重算对账,不是复读:三统计量比值从 slices.npz 现算,与 evaluate.json 已落账的 crlb_alert 比值 isclose(1e-9) 对账,不一致当场「口径漂移」报错——两处必须出自同一轮同一口径,串轮/篡改不静默通过"
  - "利用率比值只给 fx/fy/cx/cy;k1/k2/k3/p1/p2 置 None 并随表带 crlb_caliber_note(文案照抄 experiments.py 现成版,改口径要两边同步)——参数空间单独下界被补偿空间撑大,正式判定在像素空间"
  - "测试不消费真实 evaluate 轮产物:合成自洽底账单测库层,真实数据端到端留给 05-02(与 evaluate 命令接线同批验收),单测保持秒级"
  - "AnalyzeConfig 逐字段校验用 math.isfinite + itertools.pairwise(严格递增),布尔先于整数判(isinstance(True, int) 为真的陷阱);报错全部中文点名键名与合法范围"

patterns-established:
  - "配置/契约/库三层先行、CLI 后接线:05-01 每层独立可测(86 例定向 + 515 全套),calib analyze 命令体在 05-02 才出现——冒烟不依赖 CLI"

requirements-completed: [ANA-01]

# Metrics
duration: 20min
completed: 2026-10-22
---

# Phase 5 Plan 1: 阶段⑦地基(AnalyzeConfig + analyze 契约层 + ANA-01 误差预算库)Summary

**analyze 产物契约层(11 键 + 六段 npz 旁车 + 失败契约)与 ANA-01 误差预算库(与 evaluate 同一条判定线的复核 + 逐参数预算行 + 铁律硬报错)+ AnalyzeConfig 配置段,26 条新测试全套 515 passed**

## Performance

- **Duration:** 20 min
- **Started:** 2026-10-21T16:13:48Z
- **Completed:** 2026-10-21T16:34Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- AnalyzeConfig 落地:analyze 段 16 键(档位/网格/分带/what-if),frozen dataclass + 逐字段中文校验,config/calibration.yaml 同步落段(what_if 键注明「仅 full 档生效,合成证据产物里必须声明"假设非实测"」)
- analyze 产物契约层:11 键主契约写盘前校验、失败契约六段全 None 且中文说明强制非空、六份明细转 analysis/*.npz 旁车(血缘 meta 必带,json 各段只留 detail_npz 引用,读回 allow_pickle=False,引用断裂当场中文报错)
- run_budget 误差预算库:复核(独立重算 vs 落账逐位对账)+ 组装(逐参数预算行 + 像素当量)+ 判读(逐行实测低于下界或统计量跌破硬线当场硬报错,附 σ 口径排查指引);σ 坏值在进任何取数前拒绝

## Task Commits

每个任务独立提交:

1. **Task 1: AnalyzeConfig 配置段** - `63f1815` (feat)
2. **Task 2: analyze 产物契约层** - `a215641` (feat)
3. **Task 3: budget.py 误差预算库层** - `7cf3004` (feat)

**Plan metadata:** 见本计划收尾 docs 提交(SUMMARY + STATE + ROADMAP + REQUIREMENTS)

## Files Created/Modified

- `src/calib/io/config.py` - AnalyzeConfig + _parse_analyze(16 键中文校验,挂进 CalibrationConfig.analyze)
- `config/calibration.yaml` - analyze 段(每键中文注释,xyz_cond_max 写作 1.0e+4)
- `tests/test_io_config.py` - +8 例:缺段默认/仓库 yaml 对账/档位切换/非法值四类拒绝(含 bool 当数字的陷阱)
- `src/calib/analysis/artifacts.py` - analyze.json 契约与落盘(11 键 + 失败契约 + npz 旁车 + 读回防线)
- `tests/test_analysis_artifacts.py` - +9 例:契约键红线/往返逐位还原/缺键与引用断裂中文报错/源码安全断言(禁 yaml.load、只许 allow_pickle=False)
- `src/calib/analysis/budget.py` - run_budget(复核 + 预算行 + 像素当量 + 铁律硬报错,纯消费零求解)
- `tests/test_analysis_budget.py` - +9 例:自洽合成底账取值手算核对 + 三种篡改方向击破 + σ/缺键/形状契约报错

## Decisions Made

- 复核比值对账用 math.isclose(rel_tol=1e-9):两次独立重算允许浮点末位抖动,但 1% 级篡改必被拦下(测试实证)
- 测试用「自洽合成底账」:底账与落账比值同源使复核必过,篡改方向(改落账比值 / 抬下界 1.5 倍只触逐行铁律 / 抬 10 倍连统计量一起跌破硬线)逐一单独验证报错文案
- crlb_se_mean 取值路径 = slices.models[model_id](与 crlb_alert 同层),bootstrap/source_separation/multistart/final_params 全部按 evaluate.json 真实键路径取数,缺键中文报错体例照 evaluate 侧上游取数防线

## Deviations from Plan

None - plan executed exactly as written。(执行内修正两处,均发生在提交前:budget.py 初稿报错文案的引号嵌套笔误由 py_compile 拦下;两处测试断言初稿过严——位级相等改为与被测代码同松紧的 approx——不构成计划偏差)

## Issues Encountered

- gsd-tools 的 state.advance-plan / state.update-progress 误读 STATE.md 现有格式(把 Performance Metrics 表里的「Plans this phase | 6(planned)」当成当前计划号,且因缺「Last Activity」字段中止写入)——按项目既有惯例改为手写维护 STATE.md
- roadmap.update-plan-progress 在 SUMMARY 落盘前运行会把阶段五刷成 0/6 Planned,已在其后重跑;顺手修正阶段四进度表陈旧行(4/6 In Progress → 6/6 Complete,与 04-06 收口一致)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 05-02 接线条件齐备:run_budget 的全部入参(model_id/evaluate_report/slices_npz_path/sigma/model/object_points/ref 位姿)都有明确取数来源;失败契约 make_failure_analyze 就位,命令体按「先验键、再取数」消费
- 阶段四遗留对账(匹配口径 CRLB 重跑,findings 结论 4 销账)仍待 quick 档 evaluate 重跑,与本计划无阻塞关系;05-02 端到端跑真实数据时顺路对账

## Self-Check: PASSED

- 6 个关键产物文件全部在盘(artifacts.py / budget.py / 两份测试 / SUMMARY / yaml)
- 3 个任务提交全部在案:63f1815 / a215641 / 7cf3004
- 全套 515 passed(489 基线 + 26 新增),ruff check src tests 通过

---
*Phase: 05-error-attribution-corner-tuning*
*Completed: 2026-10-22*
