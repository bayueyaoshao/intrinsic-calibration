---
phase: 04-evaluation-model-selection
plan: 06
subsystem: cli + evaluation + integration-tests
tags: [evaluate-cli, integration-test, selection-refactor, quick-full-profile, warm-start-rejection, speed, phase-close]

# Dependency graph
requires:
  - phase: 04-evaluation-model-selection (04-01~04-05)
    provides: 切分纯函数/优化器网格/选型统计/四实验函数/交付与契约层(evaluate 命令只做薄编排:取数 → 编排调用 → 中文播报 → 落盘)
  - phase: 01~03
    provides: corners sb npz + manifest、noise.json 双口径 σ、mX__scipy/mX__opencv 标定解、crlb.json(Fisher 预警来源)
provides:
  - "`calib evaluate` 一条命令闭环(src/calib/cli/main.py):上游四轮定位 + 血缘登记(含 crlb.json)→ 优化器网格 → 多起点一致性 → 全量重训交付 + test 逐张评分 → test 整体 RMS 选型 → 四类实验 → 判行表与第二意见 → 契约落盘 + 三图;失败契约落盘 + 退出码 1,图失败不拦主产物"
  - 选型规则重构(selection.py select_by_test_rms):test 整体 RMS 简单优先为主判据(复杂度逐档走查、提升 ≥ selection_improve_min=0.10 才升级,非相邻好模型不被中间坏模型埋没),旧三判据降为第二意见,agreement 打架当场播报
  - quick/full 两档(EvaluateConfig):quick(默认)交付仍全模型、四类实验只考当选模型(cv 3 折 × 3 轮、bootstrap 50);full 全模型全量口径;选型失败(None)自动回落 full 行为
  - cv_pose_grouped / bootstrap_pose / slices_by_index 增 models 参数(必须是 evaluate.models 子集,中文报错),当选模型-only 实验量落地
  - 真实数据全链集成测试(tests/test_evaluation_realdata.py,档位感知):成功标准 1–5 逐条断言 + 血缘复用
  - 工程结论落账 docs/findings.md 结论 1(连拍无收益 + m2 全内参对账)/结论 2(test RMS 选型 m2 + hold-out 稳定性)/结论 3(温启动否决)/结论 5(画面半径覆盖判据)
affects: [Phase 5(复用 evaluate 实验函数与选型结论), Phase 6(report 消费 evaluate.json)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CLI 编排体例:数据classes.replace 派生生效配置(eff_cfg)给实验层,原 cfg 只用于交付与选型——档位缩水不污染落盘 meta 的来源记录"
    - "实验函数 models 参数校验铁律:传入模型名单必须是 evaluate.models 的非空子集,未知名/空表当场中文报错(防静默跑错模型集)"
    - "pytest 命名陷阱:模块级导入的 test_ 前缀函数会被收集成用例——库函数严禁 test_ 开头(overall_test_rms_by_model 由此改名)"
    - "joblib 单任务异常隔离:worker 生成器内 try/except 转失败信封,防一票异常断流整批(run_parallel 修复)"

key-files:
  created:
    - tests/test_evaluation_realdata.py
    - .planning/phases/04-evaluation-model-selection/04-06-SUMMARY.md
  modified:
    - src/calib/cli/main.py
    - src/calib/evaluation/{selection,experiments}.py
    - src/calib/io/config.py
    - config/calibration.yaml
    - tests/{test_selection_stats,test_evaluation_experiments,test_io_config,test_calibrate_matrix}.py
    - docs/{pipeline,findings}.md
    - .planning/{REQUIREMENTS,ROADMAP,STATE}.md

key-decisions:
  - "选型规则重构(74935b8,用户方法论):test 整体 RMS 简单优先替代三判据主判据——用户可立刻读懂的指标(整体/中位/最大 RMS)+ '复杂度只买 ≥10% 提升'的简单优先原则;三判据降为第二意见并做 agreement 对账,两处打架当场播报;声明 test 参与选型带轻微乐观偏差,由 hold-out 第二意见对冲"
  - "quick/full 两档(74935b8,用户要求'初步验证少、稳定了少、最后全面'):交付一律全模型(选型前提),四类实验才按档位缩水;quick 档实验只考当选模型——'等后期有问题必要的话再分析所有'的直接实现"
  - "温启动否决(用户 C 方案):4 模型 × 16 真实子集冷/温对账,RMS 差全部 <0.001 px 但 6/62 对位移超容差、且温启动中位 404s 反比冷启动 332s 慢——根因是数据在 Fisher 退化方向解不唯一(多起点位移 0.2046 px / RMS 极差 2.8e-7 px),任何求解路径差异都会落进这个漂移;默认 warm_start_experiments: false,findings 结论 3 落账"
  - "提速两板斧(cd5a066,用户 A/B 方案):A worker 内限 BLAS 单线程 + workers 8→12(修并行失效);B bootstrap 200→砍半 + lopo 只跑候选模型——两者都不改统计口径"
  - "匹配口径 CRLB(任务遗留):检测器级 σ 的 CRLB 代码已落地,但既有 run 的 CRLB 对照栏仍是旧口径数值不可引用(findings 结论 4 待办);quick 档重跑落地后销账,重跑期间本阶段先行收口(成功标准 1–5 已由集成测试断言,不依赖该对账)"
  - "集成测试体例升级:血缘指纹一致复用历史轮不重跑;档位感知断言(quick 档断言实验模型 = (selection.selected,),cv 折数 = n_rounds × n_folds)"

patterns-established:
  - "收口期发现的问题先立结论再改码:findings.md 结论(带血缘与证据数字)先行,代码与文档随后对齐——引用数字前先核对出自最新正确一轮"
  - "历史轮清理规则:被替代的旧 run/临时脚本用完即删,只留最新正确一轮;结论写进 findings.md,不靠留旧文件传信息"

requirements-completed: [EVA-01, EVA-02, EVA-03, EVA-04, EVA-05, SOL-02]

# Metrics
duration: 跨 2 日(2026-10-20 ~ 10-21,含执行期用户追加的提速/温启动/选型重构三轮)
completed: 2026-10-21
commits: [d47dcf2, 8fdb12a, 5ac256f, 05fc97f, 2106d54, cd5a066, 74935b8, 2c1d4b5, 8428d83]
---

# Phase 4 Plan 6:`calib evaluate` 命令闭环 + 阶段收口 Summary

**一条 `calib evaluate` 从上游产物直达判行表与选型结论;执行期依用户三轮追加(提速/温启动/选型重构)把"防作弊考试"收敛为 test 整体 RMS 简单优先 + quick/full 两档;六条需求 Shipped,阶段四收官。**

## Accomplishments

- **命令闭环(05fc97f)**:`evaluate()` 空壳填充——上游四轮定位(corners/noise/calibration/fisher)+ 血缘登记(含 crlb.json)→ 优化器 16 组合网格 + 多起点一致性 → 全量重训交付 + test 65 张逐张评分 → 选型 → 四类实验 → 判行表与第二意见 → evaluate.json + experiments/*.npz + figures/ 落盘;失败契约 make_failure_evaluate + 退出码 1;单任务异常隔离修复(joblib 生成器断流,5ac256f)
- **选型规则重构(74935b8)**:select_by_test_rms(test 整体 RMS 简单优先,复杂度逐档走查、提升 ≥10% 才升级)为主判据;三判据降为第二意见 + agreement 对账;quick/full 两档落地(交付一律全模型,实验按档缩水);config/calibration.yaml 五个新键 + 中文注释
- **集成对账(d47dcf2)**:tests/test_evaluation_realdata.py 成功标准 1–5 逐条断言、档位感知、血缘复用;8 例新测试覆盖 models 参数校验与选型走查
- **工程结论落账**:findings.md 六条——①同位姿连拍无收益(+m2 全内参对账:k2 晃动最大 0.135/0.214 px 当量)②test RMS 选型定案 m2(+hold-out 稳定性表)③温启动否决与解不唯一 ④匹配口径 CRLB 待办 ⑤画面半径覆盖判据(四角区零观测纯外推,按测量点落位决策)
- **收口**:REQUIREMENTS 六条勾选 + 括注(累计 17/32);ROADMAP 阶段四 completed;STATE 推进 4/6 阶段;pipeline.md ⑥ 行契约与输入列与实际对齐

## Deviations from Plan

计划四任务全执行;执行期用户追加三轮(均已成文):A 修并行 + B 砍冗余(cd5a066)、C 温启动(验证后否决,默认关闭)、选型方法论重构(74935b8)。遗留一项:**匹配口径 CRLB 重跑对账**(findings 结论 4)——quick 档重跑进行中,落地后销账(结论 1 精确倍数 + 判行表逼近 CRLB 判据以重跑后 evaluate 为准);该对账不阻塞收口(成功标准 1–5 已由集成测试断言)。

## Verification

- `uv run pytest -m "not integration" -q` 全绿(484 例);ruff 干净
- 真实数据对账:run_20261020_103552 集成测试绿(旧流程首轮);新流程 quick 档重跑落地后按收口对账清单复核 selection 新架构与 CRLB 口径栏
- 人工核查:判行表与选型结论经用户逐轮核查(含 m2 test 成绩、8 组对账、hold-out 稳定性、内参像素当量),用户确认收口
