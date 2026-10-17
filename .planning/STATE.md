---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-17T12:40:00.000Z"
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

Phase: 02 — DISCUSSED (context ready)

- **Phase:** 2 (模型层 M1–M4 + 双求解后端 + 产物血缘)
- **Plan:** None yet — next step `/gsd-plan-phase 2`
- **Status:** 阶段二讨论完成(2026-10-17,02-CONTEXT.md 就绪);开工前置一个 quick task:修阶段一遗留 CR-01 + WR-01~04
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

- [阶段二开工前] quick task:修阶段一代码审查遗留 CR-01(corners 空划分崩溃)+ WR-01~04,方案见 01-REVIEW.md(CONTEXT D-05)

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

- **Last action:** 阶段二前置讨论完成(2026-10-17)— 四项决定:全矩阵 M1–M4 试车 / 角点用 SB+classic 兜底最优组合 / 存疑帧复核纳入阶段二 / 遗留修复开工前 quick task(02-CONTEXT.md D-01~D-10)
- **Next action:** ① `/gsd-quick` 修阶段一遗留(CR-01 + WR-01~04,清单在 01-REVIEW.md)→ ② `/gsd-plan-phase 2` → ③ `/gsd-execute-phase 2`
- **Notes for next session:** 讨论决策在 .planning/phases/02-m1-m4/02-CONTEXT.md;角点输入 = sb 路线 npz(705 全覆盖,run_20261017_104817);阶段一 UAT 遗留(存疑帧 per_view_rms 对比)已收编进阶段二 D-04
