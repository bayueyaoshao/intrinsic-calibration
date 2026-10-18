---
status: partial
phase: 01-data-foundation-minimal-calibration
source: [01-VERIFICATION.md]
started: 2026-10-16T09:15:00Z
updated: 2026-10-18T04:10:00Z
---

## Current Test

[awaiting final phase-close confirmation]

## Tests

### 1. 角点覆盖目检(D-05 步骤 1)
expected: 打开 output/corners/run_20261016_085253/mosaic/overview_mosaic.png(164 格),抽查最糊几张的 marked_copies/classic/ 与 marked_copies/sb/ 同名文件并排:角点落在黑白格交点上,classic 与 sb 两套基本一致;无整排同向偏移
result: pass(2026-10-17)— 用户检查 marked_copies 与 review_sb_failed(41 张 SB 兜底帧)后确认"角点识别无明显问题";复核在新 run_20261017_104817 进行(marked_copies 双套 + review_sb_failed 41 张,含 classic 兜底角点叠加图)

### 2. 存疑清单审查(D-05 步骤 2)
expected: 打开 mosaic/suspect_mosaic.png(89 格)与 output/ingest/run_20261016_085253/suspect_originals/ 任意 2-3 张原图:存疑判定与人工判断大体一致,无误杀(清晰被标)/漏杀(明显糊没标);有异议则调 config 阈值重跑
result: pass(2026-10-17 人工目检 + 2026-10-18 逐帧残差复核闭环)— 用户查看 review_suspects(89 张原图 + 双检测器标记图 + 中文原因汇总)后认为无明显问题;当时推迟的"存疑帧是否真有害"已由阶段二 02-06 复核定量回答,见下方 Gaps 闭环记录

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

- ~~存疑帧(89 张)对下游的实际影响待阶段二量化~~ **已闭环(2026-10-18,Phase 2 计划 02-06)**:全矩阵标定 run_20261018_034000 用 m2 精化结果对 705 帧逐帧复核(存疑 89 = train 67 + test 22;正常 616 = train 573 + test 43,test 帧用"只估位姿小求解"同尺打分)。结论**走向 B(两组无差别)**:存疑组误差中位数 0.099px vs 正常组 0.086px,偏差 15.9%,未达 30% 阈值——"看着糊"≠"算得差",存疑帧照常留在数据流中,不剔除。逐帧数字见 output/calibration/run_20261018_034000/suspect_review/suspect_vs_normal.json(对比图 suspect_vs_normal.png 仅作佐证);体检指标与真实误差相关性属"弱相关、非有害",阶段五角点寻优可据此把存疑标记仅作参考权重而非剔除依据
