---
phase: quick-261017-pub
plan: 01
subsystem: corners
tags: [corners, review-folder, visualization]
duration: 约 16 分钟
completed: 2026-10-17
key_commits:
  - 2277f50 (test, 任务 1 RED)
  - da5f2e7 (feat, 任务 1 GREEN)
  - d2f7013 (test, 任务 2 RED)
  - 0be6887 (feat, 任务 2 GREEN)
new_run_dir: output/corners/run_20261017_104817
---

# 快任务 261017-pub:存疑帧与连拍点名帧各一个专属复核文件夹

一句话:给 `calib corners` 加了两个自动生成的复核文件夹 `review_suspects/`(体检判存疑,89 张)与 `review_burst_flagged/`(连拍断言点名,24 张),每个文件夹里原图副本、classic/sb 两套标记图、逐张中文判定原因齐备,人工复核不再跨目录按文件名翻找。

## 改了什么

### 任务 1:通用复核文件夹渲染函数(提交 2277f50 测试 → da5f2e7 实现)

- `src/calib/visualization/overlay.py` 新增 `render_flagged_review(corners_run_dir, folder, frames, board, quality, note, extra)`:与上一任务的 `render_sb_failed_review` 同层同风格,但文件夹名、顶层中文说明、附加计数、逐帧 info 全部由调用方传,函数只负责"归拢成一个文件夹"。
- 产物结构:`<folder>/original/` 逐字节复制原图(`shutil.copy2` 保名字保格式,源文件只读不改写);`<folder>/marked_classic/` 与 `<folder>/marked_sb/` 用 `render_marked` 画(检出画角点叠加、未检出画 MISS 红框、有 note 加 ASCII 短标);`summary.json` 键名英文、值中文,`marked`/`miss` 按检测器计数,`frames` 数组逐条并进调用方 info,`extra` 原样并入顶层。
- 空清单返回 None 且不建目录(与 review_sb_failed 同约定)。
- 测试:`test_render_flagged_review_two_frames`(检出帧用合成棋盘 + 夹具真值角点,未检出帧断言 MISS 红框,原图副本逐字节一致,summary 计数/note/extra/info 全对账)、`test_render_flagged_review_empty_list_creates_nothing`。

### 任务 2:清单组装 + corners 流程接线(提交 d2f7013 测试 → 0be6887 实现)

- `src/calib/cli/main.py` 新增两个帮手(放 `_burst_failures_for` 附近):
  - `_suspect_review_frames`:筛 `precheck.verdict == "suspect"` 的帧;info 同时保留存疑来源原始记号(对账 manifest 用)与中文翻译(四种记号全覆盖:同姿态相对比较:模糊度/曝光、全局线兜底:模糊度/曝光,多来源顿号连接),外加 blur_norm/expo_hi/expo_lo 三个指标原值;note 为 `SUS BLUR+EXPO` 式 ASCII 短标(缩写去重)。
  - `_burst_review_frames`:取两颗检测器点名清单的并集(去重、按数据清单顺序);每帧 `flagged_by` 列全点名者;原因按检测器各一条,组内波动中位数/最大值与容差的数值照抄,超限与否按数值现算(不抄旧结论),措辞覆盖超限/未超两种组合。
- `corners` 流程在 `review_sb_failed` 之后、汇总拼图之前接入两个文件夹:存疑的 `extra` 复用 `manifest_suspect_counts` 的分桶口径,连拍的 `extra` 记 `n_poses` 与点名姿态列表;返回 None 时控制台如实报"未生成"。
- 命令 help 与 docstring 步骤列表补了两个文件夹的一句话说明。

### 任务 3:全量回归 + 主树重跑 + 产物计数断言(无代码变更,无提交)

- `uv run pytest -q` 全仓 **98 个测试全部通过**(review_sb_failed、marked_copies、mosaic 等既有功能无回归)。
- `uv run calib corners` 主树全量重跑(约 8 分钟),新 run 目录:**output/corners/run_20261017_104817**(旧 run_20261017_083337 与 ingest run_20261016_085253 保留未动)。

## 断言结果(新 run 目录实测)

| 断言项 | 结果 |
|---|---|
| review_suspects/original 张数 | 89 ✓ |
| review_suspects/marked_classic、marked_sb 张数 | 89、89 ✓ |
| review_suspects/summary:n_total 与 frames 长度 | 89、89 ✓ |
| 来源分桶与 manifest 交叉验证 | 同姿态相对比较 51 张、全局线兜底 39 张,与 manifest 逐张重数一致 ✓;image_id 集合与 manifest 完全一致 ✓ |
| review_burst_flagged/original、marked_classic、marked_sb 张数 | 24、24、24 ✓ |
| review_burst_flagged/summary:n_total、n_poses | 24、3 ✓;每帧 flagged_by 至少一颗检测器(实测 24 张全部两颗都点名)✓ |
| 回归:review_sb_failed/original 与 marked | 各 41 张 ✓ |
| 流程未变:detection_summary | 两颗检测器 train 640/640、test 65/65,连拍断言失败仍 6 条 ✓ |
| 中文可读性抽查 | 存疑帧:"存疑原因:同姿态相对比较:曝光" + 三个指标原值;连拍帧:"连拍断言失败:同名角点组内波动中位数 4.17px(容差 2.06px,超限);最大 4.53px(容差 12.36px,未超)"(每颗检测器各一条)✓ |
| 源图逐字节不变 | 测试锁定(test_render_flagged_review_two_frames 前后比对)✓ |

## 与计划的偏差

无 —— 三个任务全部按计划执行,断言一次通过,未触发自动修补规则。

## Self-Check: PASSED

代码文件、测试文件、SUMMARY 本体、两个新 summary.json 产物、全部 4 个提交哈希逐一确认存在。

## 人工验收指引(两个新文件夹各看什么)

1. **`output/corners/run_20261017_104817/review_suspects/`**(89 张体检存疑帧):
   - 先开 `summary.json`:每张照片一条,`reason` 是中文判定原因(来自哪条判定线、卡在模糊度还是曝光),`metrics` 里有三个体检指标原值,`suspect_sources` 是原始记号(可与 manifest 对账)。
   - 再看 `original/` 抽几张原图,对照 `marked_classic/` 与 `marked_sb/` 同名标记图,人工判断这帧"存疑"是否影响角点质量。
2. **`output/corners/run_20261017_104817/review_burst_flagged/`**(24 张连拍点名帧,3 个姿态 × 8 连拍):
   - 先开 `summary.json`:`flagged_by` 写明哪颗(或两颗)检测器点名,`reason` 每颗检测器一条,带组内波动中位数/最大值与容差的具体数值和超限结论(当前数据全部是"中位数超限、最大值未超")。
   - 重点人工核查这 3 个姿态是否疑似镜像/翻转污染(连拍内同名角点跳位):对照 `original/` 与两套标记图看角点位置是否在连拍内稳定。
   - 两个文件夹内均不需要跨目录翻找;`review_sb_failed/`(41 张)结构未变,回归不受影响。
