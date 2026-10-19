---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-19T07:21:17.122Z"
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 23
  completed_plans: 17
  percent: 50
---

# STATE: intrinsic-calibration

## Project Reference

- **Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"
- **Current Focus:** Phase 04 — 评估实验 + 模型选型 + 优化器对比
- **Key Docs:** .planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, docs/pipeline.md

## Current Position

Phase: 03 (fisher-crlb) — COMPLETE(2026-10-19,验证 passed 5/5)

- **Phase:** 4
- **Plan:** Not started
- **Status:** Ready to execute
- **Progress:** [█████-----] 50%(3/6 相)
- **备注:** Phase 3 检查点发现并修复条件数单位伪影(0a9b247),修正后谱
  产物 output/fisher/run_20261019_025859;m1/m2 正常,m3/m4 退化为真信号

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

- **Last action:** 阶段四规划完成(2026-10-19)— 研究(04-RESEARCH.md,留白
  全裁定:三分支平局/BIC n=80/f_scale auto/扰动幅度表/0.42×σ_CRLB 告警线,
  实测发现 m1__scipy_softl1 未过收敛需 16 组合新血缘重跑)+ 模式映射
  (04-PATTERNS.md,13/13)+ 6 计划(04-01~04-06,5 波)+ 校验两轮通过
  (第 1 轮 5 警告已修订 ce8a36b,第 2 轮 PASSED)
- **Next action:** `/gsd-execute-phase 4`(Wave 1:04-01 切分地基;Wave 2:
  04-02 优化器 ∥ 04-03 选型统计)
- **Notes for next session:** 决策在 04-CONTEXT.md(D-01~D-16);训练/交付
  一律全 640 视图(聚合策略裁决定案);上游取数 = sb 角点 npz + noise.json
  双口径 + m{X}__scipy linear 收敛解 + crlb.json;joblib 随 04-01 引入
