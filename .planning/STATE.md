---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-16T04:48:54.847Z"
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 5
  completed_plans: 0
  percent: 0
---

# STATE: intrinsic-calibration

## Project Reference

- **Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"
- **Current Focus:** Phase 01 — data-foundation-minimal-calibration
- **Key Docs:** .planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, docs/pipeline.md

## Current Position

Phase: 01 (data-foundation-minimal-calibration) — EXECUTING
Plan: 1 of 5

- **Phase:** 1 (数据基座 + 首个端到端标定切片)
- **Plan:** None yet — next step `/gsd-plan-phase 1`
- **Status:** Executing Phase 01
- **Progress:** `[█░░░░░░░░░] 0/6 phases`

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 0/6 |
| Requirements shipped | 0/32 |
| Plans this phase | 0 |
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

- Phase 1 首日:uv lock 并 smoke-test opencv/numpy/scipy 三件套(研究阶段沙箱验证失败)

### Blockers

- None

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261017-mmj | 角点检测最终方案落地:SB 为主、classic 兜底 + 失败帧独立产物 + 清理中间实验产物 | 2026-10-17 | dcd5abe | [261017-mmj-corners-sb-classic-fallback](./quick/261017-mmj-corners-sb-classic-fallback/) |

### Research Flags

- Phase 3: FIM 边际化数学需与 stdDeviationsIntrinsics 交叉验证(MEDIUM 置信设计点)
- Phase 5: 角点寻优实验设计(退化栅格/真值/指标)是最不标准化的一块,阶段开始时需设计专项

## Session Continuity

- **Last action:** 快任务 261017-mmj 完成(2026-10-17)— 角点最终方案落地:SB 为主、classic 兜底,新 run_20261017_083337 705/705,失败帧独立产物齐备,中间实验产物已清理
- **Next action:** 用户人工验收:D-05 四项(01-HUMAN-UAT.md)+ 快任务 41 张 review_sb_failed/marked 抽查;验收通过后关闭阶段一
- **Notes for next session:** 先读 .planning/phases/01-data-foundation-minimal-calibration/01-CONTEXT.md(决定 D-01~D-13);首日待办:uv lock + 冒烟测试 opencv/numpy/scipy 三件套;数据只读已就位(80 姿态×8 连拍 + test 65 张)
