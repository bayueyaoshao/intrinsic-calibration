---
status: partial
phase: 02-m1-m4
source: [02-VERIFICATION.md]
started: 2026-10-18T05:00:00Z
updated: 2026-10-18T05:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. soft_l1 鲁棒核严格线裁定(成功标准 3 收口)
expected: 一条明确的裁定结论(任一走向均可):(a) 接受现状——成功标准 3 由 linear 精化路径兑现,soft_l1 偏差记为"鲁棒核优化另一目标的结构性预期";(b) 调 config/calibration.yaml 的 scipy 段(f_scale/max_nfev)后重跑 soft_l1 再判;(c) 维持严格线判不达标并安排返工。
result: [pending]

证据入口:output/calibration/run_20261018_022047(与 034000 逐位一致)中 m1__scipy_softl1/result.json 的 solver_meta(nfev=300、尾部改进 0、回代 RMS 0.2491 超 linear 参照 5.8% 被 5% 守门拒绝);m2__scipy_softl1(result.json:RMS 0.09213935,比纯 OpenCV 高 2.77e-4 即 +0.30%)。02-05 计划 must_haves 明文预案"提交人工裁定",执行与预案吻合,证据全落盘,选 (a) 不需要改代码。

### 2. suspect_vs_normal.png 目检
expected: 图与 JSON 结论一致(存疑中位 0.099px vs 正常 0.086px,走向 B 不剔除)、两组分布可辨、图内标注为 ASCII、无渲染异常。
result: [pending]

证据入口:output/calibration/run_20261018_034000/suspect_review/suspect_vs_normal.png(69KB,matplotlib 双联图:直方图叠加 + 累积分布)。

### 3. calibrate 中文汇总表排版目检
expected: 组合/视图数/整体RMS/收敛四列对齐、中文不乱码、按模型分组并排、三条白话注解讲清"同模型内比较"与"soft_l1 实用收敛"。
result: [pending]

证据入口:终端跑 uv run calib calibrate --help,并目检任意一轮 run 的汇总表。

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
