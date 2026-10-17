---
status: partial
phase: 01-data-foundation-minimal-calibration
source: [01-VERIFICATION.md]
started: 2026-10-16T09:15:00Z
updated: 2026-10-17T11:20:00Z
---

## Current Test

[awaiting final phase-close confirmation]

## Tests

### 1. 角点覆盖目检(D-05 步骤 1)
expected: 打开 output/corners/run_20261016_085253/mosaic/overview_mosaic.png(164 格),抽查最糊几张的 marked_copies/classic/ 与 marked_copies/sb/ 同名文件并排:角点落在黑白格交点上,classic 与 sb 两套基本一致;无整排同向偏移
result: pass(2026-10-17)— 用户检查 marked_copies 与 review_sb_failed(41 张 SB 兜底帧)后确认"角点识别无明显问题";复核在新 run_20261017_104817 进行(marked_copies 双套 + review_sb_failed 41 张,含 classic 兜底角点叠加图)

### 2. 存疑清单审查(D-05 步骤 2)
expected: 打开 mosaic/suspect_mosaic.png(89 格)与 output/ingest/run_20261016_085253/suspect_originals/ 任意 2-3 张原图:存疑判定与人工判断大体一致,无误杀(清晰被标)/漏杀(明显糊没标);有异议则调 config 阈值重跑
result: pass-暂定(2026-10-17)— 用户查看 review_suspects(89 张原图 + 双检测器标记图 + 中文原因汇总)后认为无明显问题,但**最终判定推迟**:等后续阶段模型定下来后,用逐帧重投影残差对比 89 张存疑帧 vs 正常帧,确认存疑帧是否真有害(calibration.json 已有 per_view_rms 逐帧数据,阶段二可做交叉验证)

### 3. 连拍断言姿态核查(D-05 步骤 3)
expected: 在 marked_copies/ 翻看连拍断言点名的 3 个姿态(20260717_174622 / 20260720_134806 / 20260720_134914)的 8 连拍:组内波动是连拍间棋盘真实移动(手抖),而非镜像翻转污染
result: pass(2026-10-17)— 用户经 review_burst_flagged(24 张 = 3 姿态 × 8 连拍,原图 + 双检测器标记 + 每张中文原因含波动数值与容差结论)核查,随角点整体检查一并确认无明显问题;实验链(单应对齐残差 0.07-0.13px vs 原始波动 3.5-6px)已证明波动 97% 以上为真实手抖

### 4. 2×2 基线量级认可(D-05 步骤 4)
expected: 查看 2×2 对比表(classic_pinhole 0.140 / classic_fisheye 0.133 / sb_pinhole 1.242 / sb_fisheye 0.083 px):用户认可基线量级,sb_pinhole 1.24px 符合这支广角镜头直觉,SB 41 张 MISS 可接受 → Phase 1 收手;或提出 config 调整走回环
result: pass(2026-10-17)— 用户认可量级与"鱼眼模型更适合"的基线信号;已向用户澄清:此数字为 train 自拟合 RMS(test 65 张未动,held-out 评估属阶段二),基线定位是端到端链条验证 + 角点全局体检,非最终模型选型

## Summary

total: 4
passed: 4(其中第 2 项为暂定,最终判定推迟到阶段二逐帧残差复核)
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

- 存疑帧(89 张)对下游的实际影响待阶段二量化:用 per_view_rms 对比存疑帧 vs 正常帧的残差分布,验证体检指标与真实误差的相关性(兼作阶段五角点寻优的输入证据)
