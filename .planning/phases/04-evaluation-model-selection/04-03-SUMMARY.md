---
phase: 04-evaluation-model-selection
plan: 03
subsystem: evaluation
tags: [model-selection, paired-test, confidence-interval, bic, chi-square-alert, mad-outlier, verdict-table]

# Dependency graph
requires:
  - phase: 04-evaluation-model-selection (04-01)
    provides: EvaluateConfig 阈值键(tie_band_px/tie_band_rel/bic_n_poses/mad_z_threshold/crlb_alert_alpha_*/gen_overfit_ratio_max/stability_disp_px_max)、CalibrationConfig 全量配置体例
  - phase: 02-m1-m4
    provides: 模型注册表参数布局(_MODEL_PARAM_COUNTS 的出处:m1=8/m2=9/m3=12/m4=8)
  - phase: 03-fisher-crlb
    provides: verdict/verdict_note 判定式随结论落盘体例(noise/fisher.py 648–659 行)
provides:
  - paired_comparison 三分支平局判定(等价取简/统计平局交 BIC/真实差异,ttest_rel 95% CI 与 Wilcoxon 兜底,方法与 p 值记 meta)
  - bic_full_fit(n·ln(RSS/n)+k·ln(n),n=80 有效姿态、k=内参+6×视图外参)
  - crlb_alert_check(卡方双告警线,df=组数−1 自适应,硬告警不静默,n_no_bound 跳过计数)
  - build_verdict_table(schema evaluate-verdict-v1,三判据判行表,接口先行定死输入形状)
  - select_model(全 PASS 才当选/held-out 最小/等价取简/平局 BIC 裁决,selection_rule 随结论落盘)
  - mad_outlier_poses(MAD 稳健 z 分数问题姿态点名,MAD=0 零误报兜底)
affects: [04-04-experiments, 04-05-experiments, 04-06-refit-test-scoring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "三分支判定不看 p 值:配对差值的 95% 置信区间对等价带 ±δ 三分支互斥且穷尽(带内→等价/含 0→平局/其余→真实差异)"
    - "Wilcoxon 符号秩置信区间的标准近似:Walsh 平均排序 + 正态近似深度截尾,exact/approx 标签按 scipy method='auto' 同款规则自判记 meta"
    - "卡方告警线数学:s < sqrt(chi2.ppf(alpha, df)/df) × σ_CRLB,df=组数−1;实测值锁进测试(df=7 硬线 0.4207/软线 0.5564,df=3 硬线 0.1956)"
    - "判定式与阈值随结论落盘(verdict/verdict_note + note 全带数值),选型结论每一步可辩护"

key-files:
  created:
    - src/calib/evaluation/selection.py
    - tests/test_selection_stats.py
  modified: []

key-decisions:
  - "select_model 的 BIC 数值从判行表行内可选 bic 键读:计划文本'按 cfg 传入的 bic 值'含糊,落为'调用方在全量重训解上用 bic_full_fit 算好写进行 dict'(接口先行);行内无 BIC 时退回 D-09 取简并在理由里说明缺席,不静默冒充"
  - "select_model 取 CalibrationConfig(读 cfg.evaluate.tie_band_px/rel):照 04-01/04-02 既定体例(实验函数取全量配置),测试直接 CalibrationConfig() 构造"
  - "三分支落为互斥穷尽 partition:区间整体在带内→equivalent、含 0→tie、其余→different;docstring 写明这覆盖了'区间不含 0 但 |均值差| ≤ δ'的夹缝情形(方向已定,不判平局)"
  - "crlb_alert_check 对'实测值非有限'的参数同样跳过计 n_no_bound:测不出来 ≠ 好于下界,与 inf σ_CRLB 同一处置"
  - "任务按 tdd=true 走双提交:两个任务各 RED(ImportError/ModuleNotFoundError 实证)+ GREEN,共 4 提交"

patterns-established:
  - "合成样本锁分支:每个判定分支一条固定种子合成数据测试,期望值(BIC -70.4358、卡方线 0.4207/0.5564/0.1956、MAD z 8.2213)手工核算写死断言"
  - "平局裁决可检验性:tie 分支测试让更复杂模型凭更小 BIC 胜出(m2 9 参 BIC 1000 < m1 8 参 1010),与取简兜底严格区分"
  - "非有限值防线全模块统一:_require_finite_1d 中文报错,统计判定拒绝带病输入(T-04-03-01)"

requirements-completed: [EVA-02, EVA-05]

# Metrics
duration: 18min
completed: 2026-10-19
---

# Phase 4 Plan 3: 选型统计纯函数库(配对检验 + BIC + 卡方告警线 + 判行表)Summary

**选型判定全链成为独立可测纯函数:配对检验三分支(等价取简/统计平局交 BIC/真实差异)在合成样本上逐分支锁死,BIC 手算值与卡方双告警线数学写死断言,三判据判行表与选型规则(全 PASS 才当选、held-out 最小者胜、平局取简)接口先行定死——04-04/04-05/04-06 照此供给数据,EVA-02/EVA-05 的判定核心闭环。**

## Performance

- **Duration:** 18 min
- **Started:** 2026-10-19T11:41:33Z
- **Completed:** 2026-10-19T11:59:56Z
- **Tasks:** 2(均按 tdd=true 拆 RED/GREEN 双提交,共 4 提交)
- **Files modified:** 2(2 新建)

## Accomplishments
- **paired_comparison 三分支(D-09/D-10)**:同折配对差值的 95% 置信区间对等价带 δ=max(tie_band_px, tie_band_rel×较简模型均值)判定——区间整体在带内判"实际等价"取简、含 0 但超带判"统计平局"交 BIC、不含 0 判"真实差异"取 held-out 小者;不看 p 值判平局(Anti-Pattern 3 兑现)。Shapiro-Wilk p≥0.05 走 ttest_rel,非正态走 Wilcoxon 符号秩区间(标准 Walsh 平均正态近似深度),所用方法、exact/approx 标签与正态性 p 值全记 meta
- **bic_full_fit(D-10)**:n·ln(RSS/n)+k·ln(n),n 锁 80 有效姿态(640 会把复杂模型惩罚弱化 8 倍)、k 写内参+6×视图外参(各模型同加常数不影响比较,写全可辩护);rss=0 返回 inf 不崩,非法输入中文报错
- **crlb_alert_check(D-05/D-06)**:经验 std 对同口径 CRLB 的卡方双线——df=组数−1 自适应,alpha=0.01 硬告警 FAIL 不静默(说明含参数名+实测比值+线值),alpha=0.05 边缘只计数;inf σ_CRLB/非有限实测跳过计 n_no_bound,不进 FAIL 也不进 PASS 分母
- **build_verdict_table(D-12/D-11)**:schema evaluate-verdict-v1 逐模型一行,泛化(过拟合比)/稳定(位移当量)/逼近 CRLB(零硬告警)各记数值+PASS/FAIL+判定式;阈值实际用值随表落盘(T-04-03-02);Fisher 预警原样进 notes、失败折数说明进 notes(D-11)
- **select_model(D-12/D-09/D-10)**:三判据全 PASS 才当选(不搞加权打分);held-out 最小者胜,前两名配对比较——等价取更简者(_MODEL_PARAM_COUNTS 注明注册表出处)、统计平局按 BIC 更小者裁决、无 BIC 退回取简不静默;无全 PASS 候选 selected=null 且 reasons 逐模型可读;selection_rule 中文规则说明随结论落盘(T-04-03-03)
- **mad_outlier_poses(D-08)**:MAD 稳健 z 分数>3.5 点名问题姿态,按 z 降序返回;MAD=0 零误报兜底写明理由
- **测试 32 例全绿**:三分支/告警线/判行表/选型规则各用固定种子合成数据锁死,期望值(BIC -70.4358、卡方线 0.4207/0.5564/0.1956、MAD z 8.2213)手工核算写死

## Task Commits

Each task was committed atomically:

1. **Task 1: 配对检验三分支失败测试(RED)** - `5014919` (test)
2. **Task 1: 配对检验三分支 + BIC + MAD 离群判据(GREEN)** - `fb7261a` (feat)
3. **Task 2: 卡方告警线/判行表/选型规则失败测试(RED)** - `c5a24c3` (test)
4. **Task 2: 卡方硬告警线 + 三判据判行表组装与选型规则(GREEN)** - `f47a6ac` (feat)

_Note: 两任务均 tdd="true",按协议各拆 test(RED)→feat(GREEN)双提交_

## Files Created/Modified
- `src/calib/evaluation/selection.py` - paired_comparison(三分支+meta)/bic_full_fit/crlb_alert_check/build_verdict_table/select_model/mad_outlier_poses + _wilcoxon_ci/_require_finite_1d 私有件 + _MODEL_PARAM_COUNTS(注册表出处注释)
- `tests/test_selection_stats.py` - 32 例:三分支各固定种子合成样本、BIC 手算值、卡方线 df=7/df=3 数学、MAD 点名与零误报、判行表行结构与 notes、选型四走向(唯一候选/等价取简/平局 BIC/无当选)、非有限值与缺摘要中文报错

## Decisions Made
- **BIC 数值走判行表行内可选 bic 键**:计划文本"tie 时按 cfg 传入的 bic 值更小者裁决"未定数值来源;落为"调用方在全量重训解上用 bic_full_fit 算好写进行 dict 的 bic 键"(判行表示例 04-RESEARCH Code Examples 本就带 "bic" 键);行内无 BIC 时 select_model 退回 D-09 取简并在理由里写明 BIC 缺席,绝不静默冒充
- **select_model 签名取 CalibrationConfig**:读 cfg.evaluate.tie_band_px/tie_band_rel,照 04-01(切分函数)/04-02(run_multistart)既定体例;测试直接 CalibrationConfig() 构造,不读 yaml
- **三分支实现为互斥穷尽 partition**:计划文本三分支有一处夹缝("CI 不含 0 但 |均值差| ≤ δ"时谁也接不住);落为先判带内、再判含 0、其余必为 different(方向已定),docstring 写明论证
- **crlb_alert_check 的"实测非有限"同 inf σ_CRLB 处置**:跳过计 n_no_bound——测不出来不等于好于理论下界,不能让它推进 PASS 也不该误判 FAIL
- **Wilcoxon 置信区间自实现标准近似**:scipy 无现成符号秩区间 API;按教科书口径(Walsh 平均排序+正态近似深度)实现,exact/approx 标签按 scipy method='auto' 同款规则自判,连同 wilcoxon p 值记 meta(04-RESEARCH Pattern 3 的"所用方法记 meta"兑现)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 测试自身缺陷:次强离群样本 4.0 的 z 分数 ≈ 2.97,过不了 3.5 阈值**
- **Found during:** Task 1(GREEN 阶段 test_results_sorted_by_z_descending 失败)
- **Issue:** 测试造数时假定 z(4.0) > 3.5,实际注入离群后 MAD 变化,z(4.0)=2.97 不会被点名——是测试期望算错,不是实现错
- **Fix:** 次强离群改 5.0(z ≈ 3.95,过线),注释写明两个数值的核算
- **Files modified:** tests/test_selection_stats.py
- **Commit:** fb7261a

其余:计划 <output> 草案 2 条提交、实际 4 条,属 tdd="true" 协议规定动作(RED/GREEN 双提交 × 2),非偏差。

## Issues Encountered
None。两个 RED 门禁均以正确原因失败(任务 1 ModuleNotFoundError、任务 2 ImportError: cannot import name 'build_verdict_table'),GREEN 均一次转绿(任务 1 的 1 例失败为上记测试自身缺陷,当场修复)。

## Known Stubs
None——selection.py 六个公开函数全部实现,无占位、无 TODO;select_model 任务 1 期间的 NotImplementedError 占位已在任务 2 GREEN(f47a6ac)替换为完整实现。

## TDD Gate Compliance

- RED 门禁 ×2:`5014919`(ModuleNotFoundError: calib.evaluation.selection,失败原因正确)、`c5a24c3`(ImportError: cannot import name 'build_verdict_table',失败原因正确)
- GREEN 门禁 ×2:`fb7261a`(17/17)、`f47a6ac`(32/32);全套 428 passed(-m "not integration")
- REFACTOR:无需要,实现一次成型;两处 ruff format 整批套用不属重构

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- **04-04(实验编排)消费面已定死**:cv_summary[mX]={"mean_heldout_px","mean_train_px","fold_heldout","n_failed_folds"}、stability_summary[mX]={"max_disp_px","slice_max_disp_px"}、crlb_summary[mX]=crlb_alert_check 返回值、fisher_warnings[mX]=str;判行表行内可选 bic 键由 04-06 全量重训后补
- select_model 前两名比较用 fold_heldout 同折对齐数组(25 个,LOPO 场景换 80 个姿态级样本同法);折级样本必须逐折对齐同一轮切分
- 无阻塞项;注意 BIC 的 rss 来自全量重训解的全部重投影残差平方和,k 传全部被估参数 = 内参数 + 6×视图数(bic_full_fit(rss, n_params=len(model.param_names) + 6*n_views, n_poses=cfg.evaluate.bic_n_poses),口径见函数 docstring)

---
*Phase: 04-evaluation-model-selection*
*Completed: 2026-10-19*

## Self-Check: PASSED

- 2 个新建文件存在(src/calib/evaluation/selection.py / tests/test_selection_stats.py)
- 4 个任务提交在案(5014919 test RED / fb7261a feat GREEN / c5a24c3 test RED / f47a6ac feat GREEN)
- 验证证据:tests/test_selection_stats.py 32/32 绿;全套 428 passed(-m "not integration");ruff check + format 干净
