---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-19T07:49:08.425Z"
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 23
  completed_plans: 18
  percent: 50
---

# STATE: intrinsic-calibration

## Project Reference

- **Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"
- **Current Focus:** Phase 04 — evaluation-model-selection
- **Key Docs:** .planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, docs/pipeline.md

## Current Position

Phase: 04 (evaluation-model-selection) — EXECUTING
Plan: 2 of 6

- **Phase:** 4
- **Plan:** 2 of 6
- **Status:** Executing Phase 04
- **Progress:** [████████░░] 78%
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

- **Last action:** 04-01 执行完成(2026-10-19,12min,9531d43/2c5ade7/9f70707)
  — EvaluateConfig 25 键契约 + evaluate yaml 段 + splitting 四种切分纯函数
  (零重叠硬门禁双保险)+ parallel.py joblib 助手;377 测试绿;EVA-01 标记完成

- **Next action:** 继续阶段四执行(Wave 2:04-02 优化器对比 ∥ 04-03 选型
  统计,两者只依赖 04-01 已就绪)

- **Notes for next session:** 决策在 04-CONTEXT.md(D-01~D-16);训练/交付
  一律全 640 视图(聚合策略裁决定案);上游取数 = sb 角点 npz + noise.json
  双口径 + m{X}__scipy linear 收敛解 + crlb.json;04-01 已交付 EvaluateConfig
  25 键 / splitting 分配表 / run_parallel(消费面见 04-01-SUMMARY.md)

## Decisions

- [Phase 04]: 阶段四切分函数取整个 CalibrationConfig(读 cfg.seed + cfg.evaluate.*),照 run_injection_experiment 体例;evaluate 核名白名单独立建表(含 cauchy,D-14)
- [Phase 04]: 零重叠硬门禁双保险:split_by_pose_ids 函数体断言 + 16 例测试对 80 姿态×25 折显式断言姿态级与行级零交集(成功标准 1 执行位);切分分配表纯 str/int 可 json 落盘复查
