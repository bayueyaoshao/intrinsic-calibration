---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-17T15:13:51.616Z"
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 11
  completed_plans: 5
  percent: 17
---

# STATE: intrinsic-calibration

## Project Reference

- **Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"
- **Current Focus:** Phase 02 — models-solvers-lineage(模型层 M1–M4 + 双求解后端 + 产物血缘)
- **Key Docs:** .planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, docs/pipeline.md

## Current Position

Phase: 02 — PLANNED (ready to execute)

- **Phase:** 2 (模型层 M1–M4 + 双求解后端 + 产物血缘)
- **Plan:** 6 个计划(02-01~02-06,5 个执行波)— next step `/gsd-execute-phase 2`
- **Status:** Ready to execute(计划经校验器两轮:0 阻断,警告已修订闭环)
- **Progress:** `[█░░░░░░░░░] 1/6 phases`

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 1/6 |
| Requirements shipped | 4/32 |
| Plans this phase | 6(planned) |
| Blocked items | 0 |

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

- **Last action:** 阶段二规划完成(2026-10-17)— 研究(02-RESEARCH.md,710 行,公式/flags/稀疏求解全实测)+ 模式映射 + 6 计划(02-01~02-06,5 波)+ 校验两轮通过(0 阻断,3 警告已修订:soft_l1 判定统一 / unproject 改真实现 / 研究问题闭环)
- **Next action:** `/gsd-execute-phase 2`(Wave 1:02-01 模型数学核心 ∥ 02-02 观测集装载)
- **Notes for next session:** 讨论决策在 .planning/phases/02-m1-m4/02-CONTEXT.md;角点输入 = sb 路线 npz(705 全覆盖,run_20261017_104817);阶段一 UAT 遗留(存疑帧 per_view_rms 对比)已收编进阶段二 D-04
