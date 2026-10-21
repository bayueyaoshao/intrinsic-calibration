---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-22T01:05:40Z"
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 29
  completed_plans: 25
  percent: 86
---

# STATE: intrinsic-calibration

## Project Reference

- **Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"
- **Current Focus:** Phase 5 — 误差预算 + 诊断 + 角点提取寻优
- **Key Docs:** .planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, docs/pipeline.md

## Current Position

Phase: 5 (误差预算 + 诊断 + 角点提取寻优) — EXECUTING
Plan: 3 of 6

- **Phase:** 5
- **Plan:** 2 of 6 完成(05-02 calib analyze 命令骨架 + ANA-01 端到端)
- **Status:** 执行中(下一计划 05-03:残差结构诊断 + test 三维误差)
- **Progress:** [█████████░] 86%
- **备注:** 05-02 接通 calib analyze——上游六轮定位 + 13 项血缘 +
  budget 直调(违例走失败契约退出码 1)+ 预算条形图;真实数据首跑
  run_20261021_165501 零违例,findings 结论 4 对账通过,全套 523 passed

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 4/6 |
| Requirements shipped | 18/32 |
| Plans this phase | 2/6 |
| Blocked items | 0 |
| Phase 05 P02 | 18min | 3 tasks | 5 files |
| Phase 05 P01 | 20min | 3 tasks | 7 files |
| Phase 04 P01 | 12min | 2 tasks | 9 files |
| Phase 4 P04 | 58min | 3 tasks | 2 files |
| Phase 4 P02 | 3h43min | 4 tasks | 4 files |
| Phase 4 P03 | 18min | 2 tasks | 2 files |
| Phase 4 P05 | 29min | 3 tasks | 5 files |
| Phase 4 P06 | 跨 2 日 | CLI 闭环+集成对账+提速重构+选型重构 | 3 files(+docs) |

## Accumulated Context

### Decisions (from PROJECT.md / research)

- models/ 是投影/雅可比唯一数学源;parity + 数值雅可比测试在任何求解代码之前落地
- 所有切分按 pose_id(GroupKFold 语义),绝不按图像——结构性防泄漏
- 噪声 σ 先于 Fisher;求解先于 Fisher(线性化点)
- 评估 runners 先于角点寻优实验(复用 calibrate_fn 网格)
- 文件是唯一真相源:阶段间仅通过带血缘的落盘产物通信
- opencv-python 锁 <5.0;Python 3.12 + numpy/scipy
- CV 只用于选型;交付内参用全量数据重训

### Todos

- None(阶段二开工前置 quick task 已完成,下一步 /gsd-plan-phase 2)

### Blockers

- None

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261017-trb | 阶段一审查遗留修复:CR-01 空划分崩溃 + WR-01~04(拼图路径/yaml 报错/针孔失败契约/体检配置驱动),106 测试全绿 | 2026-10-17 | 3439c3e | [261017-trb-review-fixes](./quick/261017-trb-review-fixes/) |
| 261017-mmj | 角点检测最终方案落地:SB 为主、classic 兜底 + 失败帧独立产物 + 清理中间实验产物 | 2026-10-17 | dcd5abe | [261017-mmj-corners-sb-classic-fallback](./quick/261017-mmj-corners-sb-classic-fallback/) |
| 261017-pub | 存疑帧/连拍点名帧专属复核文件夹:review_suspects(89 张)+ review_burst_flagged(24 张),一个文件夹看完人工复核 | 2026-10-17 | 0be6887 | [261017-pub-calib-corners](./quick/261017-pub-calib-corners/) |

### Research Flags

- Phase 3: FIM 边际化数学需与 stdDeviationsIntrinsics 交叉验证(MEDIUM 置信设计点)
- Phase 5: 角点寻优实验设计(退化栅格/真值/指标)是最不标准化的一块,阶段开始时需设计专项

## Session Continuity

- **Last action:** 05-02 执行完成(2026-10-22,18min,a011c0c/06b2511/
  9da81d1)— calib analyze 命令端到端:上游六轮定位 + 失败轮拒绝 + 13 项
  血缘一次性登记 + budget 直调(铁律违例冒泡失败契约,退出码 1)+
  plot_budget_bars 预算条形图(参数空间对数轴 + 像素当量 0.1px 参考线)+
  四要素播报;真实数据首跑 run_20261021_165501 verdict=pass 零违例,
  102/1108/1444 倍、最小比值 2.8,findings 结论 4 对账通过(阶段四遗留
  销账)。13 例新测试,全套 523 passed + 42 integration

- **Previous action:** 05-01 执行完成(2026-10-22 凌晨,20min,63f1815/
  a215641/7cf3004)— 配置/契约/库三层:AnalyzeConfig(analyze 段 16 键
  中文校验)+ analyze 产物契约层(11 键 + 失败契约 + 六段 npz 旁车)+
  run_budget 误差预算库(复核 + 预算行 + 铁律硬报错);ANA-01 标记
  Shipped,全套 515 passed

- **Earlier action:** 04-06 执行完成 + 阶段四收口(2026-10-21)——`calib evaluate`
  命令闭环(5ac256f/05fc97f)+ 真实数据全链对账集成测试(d47dcf2)+ 执行期
  三轮提速与重构:并行修复与砍冗余(cd5a066)、温启动真实数据否决(默认关
  闭)、test 整体 RMS 简单优先选型 + quick/full 两档(74935b8)+ findings
  结论 1/2/5 落账(2c1d4b5/8428d83)。选型定案 m2(test 0.1598 px,
  hold-out 0.0904 px 稳定,三判据第二意见不翻案)

- **Next action:** 05-03 残差结构诊断 + test 三维误差(residual_structure /
  test_xyz 两段按 _stage 接入 analyze 六块留位);分析段集成测试直接复用
  test_analysis_realdata 的 _latest_analyze_or_skip 取数助手;真实 analyze
  轮已在盘(run_20261021_165501),上游未变则指纹门直接放行消费

- **Notes for next session:** analyze() 命令体取数已实测跑通——evaluate 上游
  轮从 latest_run_dir 找最新,失败轮经 load_*_report 的 failure_note 检查
  当场拒绝;五段留位接线时把直调改 _stage 体例(单段失败转
  make_failure_analyze 照常出报告,budget 的铁律违例例外必须退出码 1);
  参考位姿 = extrinsics.npz 首视图 rvecs[0]/tvecs[0];每接一段,
  meta.missing_sections 清单同步收缩,集成测试随之把对应段断言从 None
  改为有值

## Decisions

- [Phase 05]: budget 直调不套逐块兜底——CRLB 口径铁律违例冒泡到外层失败契约(退出码 1 + 中文 failure_note + 六段全 None),坏结论不许静默出表;后续五段才用 _stage(单段失败照常出报告)
- [Phase 05]: analyze 上游 13 项文件 + 配置哈希在取数前一次性登记血缘;集成测试血缘指纹门逐位核对 sha256,上游任何一轮重跑过旧轮即不可消费(中文跳过理由 + 要求重跑 calib analyze)
- [Phase 05]: 图与数据分离——plot_budget_bars 只画 budget 段现成键,数字结论以 analyze.json 为准,图失败不拦主产物;k1–k3/p1/p2 不画下界条(与利用率置 None 同口径),近零参数就地标注「画面影响可忽略」
- [Phase 05]: 预算判定与 evaluate 同一条判定线——三统计量比值从 slices.npz 独立重算与 crlb_alert.per_param 落账 isclose(1e-9) 对账,漂移即中文报错,不重写第二套判据;σ 只认调用方传入的 noise.json:sigma_detector_corner,坏值进任何取数前拒绝
- [Phase 05]: k1–k3 与 p1/p2 利用率比值置 None(单独下界被参数间补偿空间撑大),CRLB_CALIBER_NOTE 照抄 experiments.py 现成文案不重写第二份;正式判定在像素空间(逐角点重投影位移)
- [Phase 05]: analysis 产物契约整文件照 evaluation/artifacts.py 体例、不跨模块借私有件;测试用「自洽合成底账」(底账与落账比值同源使复核必过,8 组等差构造标准差可手算),真实数据端到端留给 05-02
- [Phase 04]: 阶段四切分函数取整个 CalibrationConfig(读 cfg.seed + cfg.evaluate.*),照 run_injection_experiment 体例;evaluate 核名白名单独立建表(含 cauchy,D-14)
- [Phase 04]: 零重叠硬门禁双保险:split_by_pose_ids 函数体断言 + 16 例测试对 80 姿态×25 折显式断言姿态级与行级零交集(成功标准 1 执行位);切分分配表纯 str/int 可 json 落盘复查
- [Phase 04]: 优化器对比真实数据定案——四模型推荐均 linear 核(千分之一并列线内无鲁棒核优势);m1 的 soft_l1/cauchy status=0 未过实用收敛,与历史轮 run_20261018_163713 对账一致,写进 SOL-02 结论材料
- [Phase 04]: 多起点一致性实测——m1 判 inconsistent(RMS 极差 2.8e-07 px 而参数位移 0.205 px 超容差,可辨识性发现),m2/m3/m4 consistent;run_multistart 取 CalibrationConfig(要 cfg.seed),worker 层异常兜底转 make_failure 防生成器断流
- [Phase 4]: 选型统计判定核心落地——paired_comparison 三分支(置信区间对等价带,不看 p 值)/bic_full_fit(n=80 锁姿态口径)/crlb_alert_check(卡方双线 df 自适应,硬告警不静默)/build_verdict_table(schema evaluate-verdict-v1 接口先行)/select_model(全 PASS 才当选、等价取简、平局 BIC 裁决、无 BIC 退回取简不静默) — D-09/D-10/D-11/D-12 + EVA-02/EVA-05 判定核心闭环;04-04~04-06 按判行表输入形状供给数据
- [Phase 4]: 四类实验编排落地——experiments.py 四函数统一签名 fn(obs, board, cfg, optimizer_config)(全量配置体例,要 seed),同一条"OpenCV 初始化 → scipy 精化(推荐配置)"路径,验证侧一律只估位姿评分;确定性红线进程级补全:loky 并行子进程是全新进程,cv2 线程锁须随任务元组进 worker 再上一次,否则并行与串行 1e-9 级末位漂移;分组实验 CRLB 每组新算(组自身收敛解为线性化点 + temporal σ),poses80 聚合口径在 experiments.py 源码级禁入 — D-01/D-02/D-04/D-05/D-06/D-07/D-08/D-13 + EVA-01/03/04/05 实验侧闭环
- [Phase 4 收口期]: 温启动被真实数据否决(62/64 对 RMS 差 <0.001 px 但位移超容差 6 对、
  温启动反而慢 404s vs 332s;根因 = 数据在 Fisher 退化方向解不唯一),默认
  warm_start_experiments: false——"秒级精化"只在合成小样成立

- [Phase 4 收口期]: 选型规则重构——test 整体 RMS 简单优先(select_by_test_rms,
  复杂度逐档走查提升 ≥ selection_improve_min=0.10 才升级)升为主判据,旧三判据
  降为第二意见(agreement 打架当场播报);test 集参与选型带轻微乐观偏差,由
  hold-out 第二意见对冲;quick(默认,四实验只考当选模型,cv 3×3、bootstrap 50)
  /full(全模型全量)两档,交付一律全模型

- [Phase 4]: 交付与契约层落地——run_delivery 同签名体例(obs_train, obs_test, board, cfg, optimizer_config),全量重训 + test 逐张并排评分,D-16;BIC 在编排内用 bic_full_fit 算好(RSS 取自全量解,k = 内参数 + 6 × 视图数,n = bic_n_poses),select_model 只消费行内现成值不写第二份数学(D-10);evaluate.json 13 键契约 + 明细 npz 旁车(嵌套明细拍平成 模型__数组名,标量成 0 维数组),json 段只留汇总与 detail_npz 引用(汇总只是索引,D-13);组装顺序防线:明细键带数组但对应段非字典当场中文报错;判行表三图库函数只管画图,图失败不拦主产物;全链冒烟即 04-06 CLI 的调用序列样板 — EVA-02/SOL-02 收口
