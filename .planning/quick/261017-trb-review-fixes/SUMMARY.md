---
status: complete
task: 阶段一审查遗留修复(CR-01 + WR-01~04)
date: 2026-10-17
commits: [3439c3e, 0f36a5d, dad5180, d40432f, 6f586e9]
plan: .planning/quick/261017-trb-review-fixes/PLAN.md
review_source: .planning/phases/01-data-foundation-minimal-calibration/01-REVIEW.md
---

# 阶段一审查遗留修复 — 完成摘要

阶段二开工前的前置 quick task(02-CONTEXT.md D-05):把阶段一代码审查的 5 条遗留修干净,带清爽底子进模型层。

## 修复清单

| 发现 | 内容 | 提交 |
|------|------|------|
| CR-01 | corners 摘要在"只有 train 没有 test"上崩溃(空列表推断 float64 索引)——行号数组显式 `dtype=int` + 空划分端到端测试 | 3439c3e |
| WR-01 | 存疑拼图在 corners 目录找 suspect_originals(实际在 ingest 目录)——显式传参 > 血缘反推 > 退回原始路径三级定位;夹具改真实布局 + 两条回归测试 | 0f36a5d |
| WR-02 | yaml 语法错误裸英文堆栈——`_read_yaml` 捕获 YAMLError 转中文 ValueError,两道防线同时修复 | dad5180 |
| WR-03 | 针孔求解不捕获 cv2.error——与鱼眼同构:九键齐全、数值 None、中文 failure_note | d40432f |
| WR-04 | 体检回退硬编码检测器选项 + SB 角点失明——`_classic_flag_mask` 配置驱动单一来源;classic 失败时 SB 角点兜底定位棋盘框;消除白跑 | 6f586e9 |

## 验收

- `uv run pytest`:**106 通过 / 0 失败**(含 3 条真实数据集成用例;修复前为 101 条,新增 5 条锁定测试)
- `uv run ruff check src tests`:零告警
- 恰 5 个修复原子提交 + 1 个计划收录提交(50990cb);`.planning/config.json` 历史未提交改动全程未进任何提交
- 真实 `output/` 产物零改动(纯代码修复,新测试全落 tmp_path)
- 01-REVIEW.md Info 节 7 条低优先级问题按计划一概未碰(留后续)

## 偏离说明

- WR-02 测试在计划基础上顺带断言 `load_board` 入口同样中文报错(验收标准要求两个加载入口都覆盖,计划测试清单只列了一个)——加强,无副作用。
