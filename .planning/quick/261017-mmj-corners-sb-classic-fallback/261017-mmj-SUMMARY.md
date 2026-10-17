---
quick: 261017-mmj-corners-sb-classic-fallback
plan: 261017-mmj
type: execute
status: complete
started: "2026-10-17T08:26:22Z"
finished: "2026-10-17T08:44:24Z"
duration: "约 18 分钟(不含人工验收)"
commits:
  - "7d271af: test(corners): failing tests for sb_fallback_classic switch and npz source array"
  - "ab7756b: feat(corners): add sb_fallback_classic switch and npz source array contract"
  - "17fe3e2: test(corners): failing tests for marked-copy note and SB failed-frame review artifacts"
  - "dcd5abe: feat(corners): classic fallback for SB route with per-frame provenance and review_sb_failed artifacts"
files_modified:
  - src/calib/io/config.py
  - config/calibration.yaml
  - src/calib/corners/artifacts.py
  - src/calib/visualization/overlay.py
  - src/calib/cli/main.py
  - tests/test_io_config.py
  - tests/test_corners_artifacts.py
  - tests/test_visualization.py
new_run: output/corners/run_20261017_083337
---

# 快任务 261017-mmj:角点检测定案——SB 为主、classic 兜底 执行总结

一句话:角点阶段从"实验对比"转为"生产定案"——SB 未检出的帧用经典检测器结果兜底,coverage 达 705/705,每帧来源可审计,SB 失败帧证据独立留存,三轮实验中间产物清理完毕。

## 各任务执行结果

### Task 1:兜底配置开关 + SB 路线 npz 的 source 数组契约

- **RED 提交** `7d271af`,**GREEN 提交** `ab7756b`
- `CornersConfig.sb_fallback_classic` 默认 True;yaml 里写整数 1/0 会被拦下并给带字段名 `corners.sb_fallback_classic` 的中文报错(Python 里 True 本身也是整数,所以先判布尔再拒其余一切)
- `config/calibration.yaml` 新增 `sb_fallback_classic: true`,注释写明兜底含义与三轮实验依据
- `save_corners` 新增可选 `source` 数组:落盘读回逐位一致,仍走 `allow_pickle=False` 安全路径;不传时完全不写该键,经典路线产物保持纯净
- **测试**:Task 1 验收命令 `uv run pytest tests/test_io_config.py tests/test_corners_artifacts.py -q` → **20 passed**(新增 5 个测试,RED 阶段确认 5 个先失败)

### Task 2:corners 命令 classic 兜底填槽 + 来源留痕 + review_sb_failed 独立产物

- **RED 提交** `17fe3e2`,**GREEN 提交** `dcd5abe`
- corners 主循环重排:classic 先跑留在局部变量;SB 检出照常填槽(source='sb');SB 未检出且开关开且 classic 有结果 → classic 的角点/方格边长/窗口填进 SB 槽位(source='classic_fallback');两法都未检出 source=''。`src/calib/corners/detect.py` 一字未动(两颗检测器保持纯净,`test_detect_sb_source_guard_single_refinement_call` 通过为旁证)
- 仅 SB 路线落盘带 source 数组;SB 的 meta.json 追记 `sb_fallback_classic` 与 `n_classic_fallback`;classic 的调用与 meta 完全不变
- `detection_summary.json` SB 段三口径:`n_detected`(生产口径,含兜底)、`n_pure_sb`、`n_classic_fallback`;兜底开关关闭时两字段照报(纯 SB = 检出数、兜底 = 0),口径稳定不随开关分支
- `render_marked` 可选 ASCII 短标(画在 MISS 短标下方,字号按画面尺寸比例推导);新增 `render_sb_failed_review`:原图副本(shutil.copy2 保名字保格式)+ 标记图(兜底帧角点叠加标 "SB MISS - CLASSIC FALLBACK",双失败帧 MISS 红框标 "SB+CLASSIC MISS")+ 通俗中文 summary.json;空清单返回 None 且不建目录
- **测试**:Task 2 验收命令 `uv run pytest tests/test_visualization.py tests/test_corners.py tests/test_corners_artifacts.py tests/test_io_config.py -q` → **50 passed**;全量非集成 **91 passed**;真实数据端到端冒烟(含新 corners 全链路)**3 passed**

### Task 3:主树重跑 calib corners 验收 + 清理(无 git 提交,动作全在 output/ 与 /tmp)

- 主树执行 `uv run calib corners`,705 张 × 2 检测器,约 7.5 分钟,退出码 0
- 新 run:**output/corners/run_20261017_083337/**
- 断言脚本(计划原文)输出 **OK**,六项核对数字见下节
- 迁移诊断记录:旧 run 的 `review_failed_sb/DIAGNOSIS.md` 复制到新 run `review_sb_failed/DIAGNOSIS.md`,开头加了一行「本文件夹由 calib corners 自动生成,DIAGNOSIS.md 是实验诊断记录」
- 清理(用户已批准):删除旧 run `experiments/`(4 个实验 json + sb_rescue_marked/ 24 张)与 `review_failed_sb/`(余量 original/ 41 + marked_sb/ 41);删除 /tmp 四个实验脚本(exp_a / exp_b 原已不在,exp_c / exp_d 已删)
- 旧 run 其余证据产物原样保全:4 份 npz + 4 份 meta.json + detection_summary.json + marked_copies/ + mosaic/ 一个不动

## Task 3 六项断言数字(全部通过)

| # | 断言项 | 结果 |
|---|--------|------|
| 1 | classic 检出 | train 640/640、test 65/65,共 **705/705** |
| 2 | SB 生产口径 | train 640 = 纯 SB 600 + 兜底 40;test 65 = 纯 SB 64 + 兜底 1;共 **705/705 = 664 纯 SB + 41 兜底** |
| 3 | npz source 留痕 | corners_train_sb.npz 计数 {sb: 600, classic_fallback: 40, "": 0};corners_test_sb.npz {sb: 64, classic_fallback: 1, "": 0};corners_train_classic.npz **无** source 键;train_sb meta 记 n_classic_fallback=40、sb_fallback_classic=true(test 1) |
| 4 | review_sb_failed 产物 | original/ 41 张 + marked/ 41 张 + summary.json(总数 41、classic 兜底 41、两法都未检出 0)+ 迁移的 DIAGNOSIS.md |
| 5 | burst_check 段 | 键齐全照常输出:失败 6 组(与上轮一致),每颗检测器检查 80 组(SB 从 75 升到 80,兜底帧加入了连拍分组,属预期) |
| 6 | 清理与保全 | 旧 run 的 experiments/ 与 review_failed_sb/ 已删,其余产物一个不动;/tmp 实验脚本已清;断言脚本输出 OK |

## 与计划的偏差(自动处理)

**1. [Rule 3 - 语法所迫] save_corners 的 source 之后参数改为按名传递**
- source 有默认值而 lineage 没有,Python 语法不允许"有默认值的参数排在没默认值的参数前"。用星号把 source/lineage/meta_extra 设为按名传递,保住计划要求的"source 在 detected 与 lineage 之间"排版;全部现有调用本来就按名传 lineage,行为零变化。
- 提交:ab7756b

**2. [计划行文与数据事实不符,以数据为准] 兜底帧是 41 张,不是"40 兜底 + 1 双未检出"**
- 计划行文写"40 张 classic 兜底角点叠加 + 1 张 MISS 红框(双未检出 test 帧 20260722_111500)",但 classic 对全部 705 张(含 20260722_111500)都检出了角点,所以那 1 张 test 失败帧同样被 classic 兜底:实际是 **41 张兜底 + 0 张双未检出**。计划自带的数字断言脚本(n_classic_fallback: train 40 + test 1)与数据事实一致、原样通过;MISS 红框代码路径在单测里用合成双失败帧锁定,本批真实数据零出现。
- 已抽查确认:20260722_111500_158294 的标记图是角点叠加(顶行红色占比 0.0,非 MISS 红框),原图副本在位。

**3. [预期内的连带变化] burst_check 的 SB 分组从 75 升到 80**
- 兜底让 41 帧进入 SB 路线的连拍一致性断言,可分组姿态从 75 变 80;失败组数仍为 6,与上轮口径一致,计划允许("失败数与上轮口径一致即可")。

## 测试结果汇总

- 全量测试:`uv run pytest -q` → **94 passed**(91 常规 + 3 真实数据集成冒烟),0 失败
- `src/calib/corners/detect.py` 未改动(git diff 为空);`test_detect_sb_source_guard_single_refinement_call` 通过

## 人工验收清单(建议按序)

1. **翻标记图**:打开 `output/corners/run_20261017_083337/review_sb_failed/marked/`,41 张应全部是角点叠加 + 左上角红字 "SB MISS - CLASSIC FALLBACK",抽查几张大倾斜视角的,确认 classic 兜底角点贴得准(这是兜底方案的核心证据)
2. **对照原图**:同目录 `original/` 是这 41 张的原图副本(文件名与格式原样),可疑的并排放大核对
3. **查诊断记录**:打开 `output/corners/run_20261017_083337/review_sb_failed/DIAGNOSIS.md`,确认开头有「本文件夹由 calib corners 自动生成,DIAGNOSIS.md 是实验诊断记录」一行,内容与旧版一致(三轮实验结论完整)
4. **核对口径**:控制台/`detection_summary.json` 的 SB 生产口径 705/705(664 纯 SB + 41 兜底);`corners_train_sb.meta.json` 记 n_classic_fallback=40
5. **旧 run 保全抽查**:`output/corners/run_20261016_085253/` 应只剩 4 份 npz、4 份 meta.json、detection_summary.json、marked_copies/、mosaic/(阶段一验证证据,一个不少)

## 遗留事项

- 无阻塞性遗留。已知代价(阶段二跟踪项,DIAGNOSIS 已记录):两法对"角点在哪"的定义有系统差,41 张兜底帧的标定残差可能系统性偏高,评估时需单列统计

## Self-Check: PASSED

- 8 个改动文件 + SUMMARY.md + 迁移的 DIAGNOSIS.md 全部在盘
- 4 个任务提交(7d271af / ab7756b / 17fe3e2 / dcd5abe)全部在 git 历史
- 工作区无本任务遗留改动(仅剩与本任务无关的 .planning/config.json 改动和 .planning/quick/ 未跟踪目录,均未触碰)
