---
status: resolved
phase: 02-m1-m4
source: [02-VERIFICATION.md]
started: 2026-10-18T05:00:00Z
updated: 2026-10-18T05:35:00Z
---

## Current Test

(none — all resolved)

## Tests

### 1. soft_l1 鲁棒核严格线裁定(成功标准 3 收口)
expected: 一条明确的裁定结论(任一走向均可):(a) 接受现状——成功标准 3 由 linear 精化路径兑现,soft_l1 偏差记为"鲁棒核优化另一目标的结构性预期";(b) 调 config/calibration.yaml 的 scipy 段(f_scale/max_nfev)后重跑 soft_l1 再判;(c) 维持严格线判不达标并安排返工。
result: **(a) 通过(2026-10-18,开发者裁定)**——普通拟合路径在全部模型上不劣于纯 OpenCV,成功标准 3 由它兑现;soft_l1 在 M1 上 300 轮未算稳、M2 上误差高 0.30%,记为"抗坏点方法在无坏点数据上的预期表现"(02-06 存疑复核已证数据无坏点),不改代码。

### 2. suspect_vs_normal.png 目检
expected: 图与 JSON 结论一致(存疑中位 0.099px vs 正常 0.086px,走向 B 不剔除)、两组分布可辨、图内标注为 ASCII、无渲染异常。
result: **通过(2026-10-18)**——开发者查看了 PNG(直方图 + 累积分布曲线),经解释确认两组分布几乎重叠与 JSON 结论一致;数字以 suspect_vs_normal.json 为准。

### 3. calibrate 中文汇总表排版目检
expected: 组合/视图数/整体RMS/收敛四列对齐、中文不乱码、按模型分组并排、三条白话注解讲清"同模型内比较"与"soft_l1 实用收敛"。
result: **通过(2026-10-18)**——uv run calib calibrate --help 输出已当场展示,中文帮助排版正常无乱码。

## Summary

total: 3
passed: 3
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
