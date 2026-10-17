---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-17T11:35:00.000Z"
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 5
  completed_plans: 5
  percent: 17
---

# STATE: intrinsic-calibration

## Project Reference

- **Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"
- **Current Focus:** Phase 02 — models-solvers-lineage(模型层 M1–M4 + 双求解后端 + 产物血缘)
- **Key Docs:** .planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, docs/pipeline.md

## Current Position

Phase: 02 — NOT STARTED

- **Phase:** 2 (模型层 M1–M4 + 双求解后端 + 产物血缘)
- **Plan:** None yet — next step `/gsd-discuss-phase 2`
- **Status:** Phase 1 关账完毕(2026-10-17,D-05 验收 4/4),准备开启阶段二
- **Progress:** `[█░░░░░░░░░] 1/6 phases`

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 1/6 |
| Requirements shipped | 4/32 |
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
| 261017-pub | 存疑帧/连拍点名帧专属复核文件夹:review_suspects(89 张)+ review_burst_flagged(24 张),一个文件夹看完人工复核 | 2026-10-17 | 0be6887 | [261017-pub-calib-corners](./quick/261017-pub-calib-corners/) |

### Research Flags

- Phase 3: FIM 边际化数学需与 stdDeviationsIntrinsics 交叉验证(MEDIUM 置信设计点)
- Phase 5: 角点寻优实验设计(退化栅格/真值/指标)是最不标准化的一块,阶段开始时需设计专项

## Session Continuity

- **Last action:** 阶段一正式关账(2026-10-17)— D-05 四项人工验收全过(第 2 项暂定,存疑帧最终影响待阶段二逐帧残差复核);收尾:UAT 记录(b51eeb8)、docs/pipeline.md 同步三个 review 复核文件夹与兜底契约
- **Next action:** `/gsd-discuss-phase 2`(模型层 M1–M4 + 双求解后端 + 产物血缘)→ 计划 → 执行
- **Notes for next session:** 先读 .planning/phases/01-data-foundation-minimal-calibration/01-CONTEXT.md(决定 D-01~D-13);首日待办:uv lock + 冒烟测试 opencv/numpy/scipy 三件套;数据只读已就位(80 姿态×8 连拍 + test 65 张)
