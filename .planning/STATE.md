---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-19T13:04:01Z"
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 23
  completed_plans: 21
  percent: 50
---

# STATE: intrinsic-calibration

## Project Reference

- **Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"
- **Current Focus:** Phase 04 — evaluation-model-selection
- **Key Docs:** .planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, docs/pipeline.md

## Current Position

Phase: 04 (evaluation-model-selection) — EXECUTING
Plan: 5 of 6

- **Phase:** 4
- **Plan:** 5 of 6
- **Status:** Ready to execute
- **Progress:** [█████████▓] 91%
- **备注:** Phase 3 检查点发现并修复条件数单位伪影(0a9b247),修正后谱
  产物 output/fisher/run_20261019_025859;m1/m2 正常,m3/m4 退化为真信号

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 1/6 |
| Requirements shipped | 5/32 |
| Plans this phase | 6(planned) |
| Blocked items | 0 |
| Phase 04 P01 | 12min | 2 tasks | 9 files |
| Phase 4 P04 | 58min | 3 tasks | 2 files |
| Phase 4 P02 | 3h43min | 4 tasks | 4 files |
| Phase 4 P03 | 18min | 2 tasks | 2 files |

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

- **Last action:** 04-04 执行完成(2026-10-19,58min,92a0163/8a6c734/
  13f657c/666bcaf/5314f39/e90131f)— 四类实验函数 experiments.py:
  cv_pose_grouped(5×5 折姿态分组,折级样本逐折对齐供配对比较,验证侧
  只估位姿评分)+ lopo(逐姿态表 + MAD z 全表复用 selection 数学)+
  slices_by_index(每组新算同口径 CRLB,组自身收敛解为线性化点 + temporal
  σ,poses80 聚合口径源码级禁入)+ bootstrap_pose(4 模型共用同一批分配,
  去重带全连拍,var_bootstrap/param_ci/max_disp_px)+ pose_sampling_variance
  (var_boot − var_slice 方差相减,负值钳 0 带说明);Rule 1 修复:四个
  worker 各自上 cv2 线程锁(并行子进程全新进程,入口设置传不过去),并行
  与串行从 1e-9 漂移修为逐位一致;18 例测试全绿,全套 446 passed;
  EVA-01(实验侧)/EVA-03/EVA-04/EVA-05 标记完成

- **Next action:** 继续阶段四执行(Plan 5:04-05 交付与契约——全量重训 × 4
  + test 65 张并排评分 + evaluate.json 主契约与 npz 旁车 + 判行表三图 +
  全链冒烟;Wave 4,消费 04-04 四实验返回值与 _detail 明细)

- **Notes for next session:** 决策在 04-CONTEXT.md(D-01~D-16);训练/交付
  一律全 640 视图(聚合策略裁决定案);上游取数 = sb 角点 npz + noise.json
  双口径 + m{X}__scipy linear 收敛解 + crlb.json + optimizer.json(推荐配置
  从 recommended_solver 读,四模型均 linear);04-04 四实验返回值统一
  {"schema","models","meta","_X_detail"}:汇总键对齐判行表(cv 用
  fold_heldout 25 个逐折对齐 / stability 用 slice_max_disp_px 与 bootstrap
  max_disp_px / crlb 用 crlb_alert 返回值),_detail 明细数组(失败记 NaN
  行)交产物层转 npz 旁车,JSON 只落汇总键;var_slice/pose_sampling_var
  是占位键——组装侧把 slices empirical_std 平方填 var_slice 后经
  pose_sampling_variance 相减;真实数据实验量 ≈ CV 100 + bootstrap 800 +
  分组 32 + LOPO 320 ≈ 1300 次求解(8 并行单次 1.5-3 分钟),先跑 CV 校准
  耗时再放大批量

## Decisions

- [Phase 04]: 阶段四切分函数取整个 CalibrationConfig(读 cfg.seed + cfg.evaluate.*),照 run_injection_experiment 体例;evaluate 核名白名单独立建表(含 cauchy,D-14)
- [Phase 04]: 零重叠硬门禁双保险:split_by_pose_ids 函数体断言 + 16 例测试对 80 姿态×25 折显式断言姿态级与行级零交集(成功标准 1 执行位);切分分配表纯 str/int 可 json 落盘复查
- [Phase 04]: 优化器对比真实数据定案——四模型推荐均 linear 核(千分之一并列线内无鲁棒核优势);m1 的 soft_l1/cauchy status=0 未过实用收敛,与历史轮 run_20261018_163713 对账一致,写进 SOL-02 结论材料
- [Phase 04]: 多起点一致性实测——m1 判 inconsistent(RMS 极差 2.8e-07 px 而参数位移 0.205 px 超容差,可辨识性发现),m2/m3/m4 consistent;run_multistart 取 CalibrationConfig(要 cfg.seed),worker 层异常兜底转 make_failure 防生成器断流
- [Phase 4]: 选型统计判定核心落地——paired_comparison 三分支(置信区间对等价带,不看 p 值)/bic_full_fit(n=80 锁姿态口径)/crlb_alert_check(卡方双线 df 自适应,硬告警不静默)/build_verdict_table(schema evaluate-verdict-v1 接口先行)/select_model(全 PASS 才当选、等价取简、平局 BIC 裁决、无 BIC 退回取简不静默) — D-09/D-10/D-11/D-12 + EVA-02/EVA-05 判定核心闭环;04-04~04-06 按判行表输入形状供给数据
- [Phase 4]: 四类实验编排落地——experiments.py 四函数统一签名 fn(obs, board, cfg, optimizer_config)(全量配置体例,要 seed),同一条"OpenCV 初始化 → scipy 精化(推荐配置)"路径,验证侧一律只估位姿评分;确定性红线进程级补全:loky 并行子进程是全新进程,cv2 线程锁须随任务元组进 worker 再上一次,否则并行与串行 1e-9 级末位漂移;分组实验 CRLB 每组新算(组自身收敛解为线性化点 + temporal σ),poses80 聚合口径在 experiments.py 源码级禁入 — D-01/D-02/D-04/D-05/D-06/D-07/D-08/D-13 + EVA-01/03/04/05 实验侧闭环
