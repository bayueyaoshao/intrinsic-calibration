---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-21T09:30:00Z"
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 23
  completed_plans: 23
  percent: 100
---

# STATE: intrinsic-calibration

## Project Reference

- **Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"
- **Current Focus:** Phase 4 收官完成,待启动 Phase 5(误差预算 + 诊断 + 角点寻优)
- **Key Docs:** .planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, docs/pipeline.md

## Current Position

Phase: 04 (evaluation-model-selection) — COMPLETE(2026-10-21)
Plan: 6 of 6 全部完成

- **Phase:** 4
- **Plan:** 6 of 6
- **Status:** Phase complete
- **Progress:** [██████████] 100%(全项目 23 计划;Phase 5/6 待规划)
- **备注:** Phase 3 检查点发现并修复条件数单位伪影(0a9b247),修正后谱
  产物 output/fisher/run_20261019_025859;m1/m2 正常,m3/m4 退化为真信号

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 4/6 |
| Requirements shipped | 17/32 |
| Plans this phase | 6(planned) |
| Blocked items | 0 |
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

- **Last action:** 04-06 执行完成 + 阶段四收口(2026-10-21)——`calib evaluate`
  命令闭环(5ac256f/05fc97f)+ 真实数据全链对账集成测试(d47dcf2)+ 执行期
  三轮提速与重构:并行修复与砍冗余(cd5a066)、温启动真实数据否决(默认关
  闭)、test 整体 RMS 简单优先选型 + quick/full 两档(74935b8)+ findings
  结论 1/2/5 落账(2c1d4b5/8428d83);六条需求 Shipped,累计 17/32。
  选型定案 m2(test 0.1598 px,hold-out 0.0904 px 稳定,三判据第二意见
  不翻案)。遗留一项会话外对账:匹配口径 CRLB 重跑(findings 结论 4 待办,
  quick 重跑落地后销账)

- **Previous action:** 04-05 执行完成(2026-10-19,29min,d3b3844/6a2048f/
  3097e41/914939b/42a5fca)— 交付与契约层:run_delivery 全量重训 × 全部
  参选模型(统一推荐配置 D-13)+ test 独立照片全模型并排逐张评分
  (solve_pose_only 零新数学,逐张逐模型行全存 D-16)+ BIC 在编排内算好
  (rss 取自全量解,k = 内参数 + 6 × 视图数,n = bic_n_poses,D-10 先重训
  后选型);evaluate.json 13 键主契约(EVALUATE_KEYS 写盘前校验 + 失败契约
  make_failure_evaluate)+ 五份明细转 experiments/<段名>.npz 旁车(血缘
  meta 必带,json 只留 detail_npz 引用,读回 allow_pickle=False);判行表
  三图 evaluate_figs.py(CV 误差棒 / LOPO 散点红三角点名 + MAD 判据线 /
  离散对比并排条形,Agg 离线出 PNG);合成数据全链冒烟(网格 → 四实验 →
  交付 → 来源分离组装 → 判行表与选型 → 契约落盘 → 三图,32 秒跑通)+
  同种子重跑逐位一致断言;14 例新测试,全套 460 passed;EVA-02/SOL-02
  标记完成

- **Next action:** ① quick 档 evaluate 重跑落地后对账(核 selection.method/
  walk/agreement 新架构 + slices CRLB 匹配口径栏 → findings 结论 4 销账、
  结论 1 精确倍数补账 + 集成测试跑一轮);② /gsd-plan-phase 5 启动阶段五

- **Notes for next session:** 决策在 04-CONTEXT.md(D-01~D-16);训练/交付
  一律全 640 视图(聚合策略裁决定案);上游取数 = sb 角点 npz + noise.json
  双口径 + m{X}__scipy linear 收敛解 + crlb.json + optimizer.json(推荐配置
  从 recommended_solver 读,四模型均 linear);04-04 四实验返回值统一
  {"schema","models","meta","_X_detail"},04-05 已把它们装配成落盘契约:
  json 段只留汇总(行级大表 per_pose_rows/group_records/rows/test_rows 与
  私有明细键不进 json),组装顺序 = 明细键必须挂在字典段下否则中文报错;
  var_slice/pose_sampling_var 组装侧填充已实证(冒烟第 ④ 步:分组
  empirical_std 平方 → pose_sampling_variance 相减 → 回填,CRLB 对照缺席
  用 {'n_below_hard': 0, ...} 兜底不冒充);真实数据实验量 ≈ CV 100 +
  bootstrap 800 + 分组 32 + LOPO 320 ≈ 1300 次求解(8 并行单次 1.5-3
  分钟)+ 交付轮全量重训 × 4,先跑 CV 校准耗时再放大批量

## Decisions

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
