---
status: partial
phase: 01-data-foundation-minimal-calibration
source: [01-VERIFICATION.md]
started: 2026-10-16T09:15:00Z
updated: 2026-10-16T09:15:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. 角点覆盖目检(D-05 步骤 1)
expected: 打开 output/corners/run_20261016_085253/mosaic/overview_mosaic.png(164 格),抽查最糊几张的 marked_copies/classic/ 与 marked_copies/sb/ 同名文件并排:角点落在黑白格交点上,classic 与 sb 两套基本一致;无整排同向偏移
result: [pending]

### 2. 存疑清单审查(D-05 步骤 2)
expected: 打开 mosaic/suspect_mosaic.png(89 格)与 output/ingest/run_20261016_085253/suspect_originals/ 任意 2-3 张原图:存疑判定与人工判断大体一致,无误杀(清晰被标)/漏杀(明显糊没标);有异议则调 config 阈值重跑
result: [pending]

### 3. 连拍断言姿态核查(D-05 步骤 3)
expected: 在 marked_copies/ 翻看连拍断言点名的 3 个姿态(20260717_174622 / 20260720_134806 / 20260720_134914)的 8 连拍:组内波动是连拍间棋盘真实移动(手抖),而非镜像翻转污染
result: [pending]

### 4. 2×2 基线量级认可(D-05 步骤 4)
expected: 查看 2×2 对比表(classic_pinhole 0.140 / classic_fisheye 0.133 / sb_pinhole 1.242 / sb_fisheye 0.083 px):用户认可基线量级,sb_pinhole 1.24px 符合这支广角镜头直觉,SB 41 张 MISS 可接受 → Phase 1 收手;或提出 config 调整走回环
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps
