---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
last_updated: "2026-10-16T02:55:29.214Z"
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# STATE: intrinsic-calibration

## Project Reference

- **Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"
- **Current Focus:** Phase 1 — 数据基座 + 首个端到端标定切片
- **Key Docs:** .planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, docs/pipeline.md

## Current Position

- **Phase:** 1 (数据基座 + 首个端到端标定切片)
- **Plan:** None yet — next step `/gsd-plan-phase 1`
- **Status:** Roadmap created, awaiting planning
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

### Research Flags

- Phase 3: FIM 边际化数学需与 stdDeviationsIntrinsics 交叉验证(MEDIUM 置信设计点)
- Phase 5: 角点寻优实验设计(退化栅格/真值/指标)是最不标准化的一块,阶段开始时需设计专项

## Session Continuity

- **Last action:** Roadmap created (2026-10-16), 32/32 requirements mapped across 6 phases
- **Next action:** `/gsd-plan-phase 1`
- **Notes for next session:** 数据已就位只读(input_data/train 80 姿态×8 连拍 + test 65 张);棋盘 11×8 内角点 25mm 在 config/board.yaml
