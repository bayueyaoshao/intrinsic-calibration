---
phase: 03-fisher-crlb
plan: 06
subsystem: pipeline
tags: [smoke-test, cli, run-all, docs, requirements, checkpoint]

# Dependency graph
requires:
  - phase: 03-fisher-crlb/03-03
    provides: noise.json 双口径 σ 与传递曲线(接力链第四步的产物与结构契约对象)
  - phase: 03-fisher-crlb/03-05
    provides: `calib fisher` 命令与 crlb.json 契约(接力链第五步的产物与结构契约对象)
  - phase: 02-m1-m4
    provides: 五步链路的标定产物与血缘口径(冒烟链沿用的既有体例)
provides:
  - 五步冒烟接力链(ingest → corners → noise → calibrate → fisher):真实产物上独立进程逐步跑通,noise/fisher 结构键全断言 + 全链血缘指纹闭合断言
  - `calib run-all` 接力到 fisher:一条命令从原始图像直达 crlb.json
  - docs/pipeline.md 第③⑤步契约速查与实际产物一致(含"条件数达退化线→改善摆拍"回环路径)
  - REQUIREMENTS.md NOI-01/02/03 状态收口(NOI-03 于本计划勾选完成)
affects: [04-evaluation 选型(阶段入口是齐备的五步产物链), /gsd-verify-work(Phase 3 验收)]

# Tech tracking
tech-stack:
  added: []   # 零新增依赖
  patterns:
    - "冒烟接力链是各阶段集成测试的超集:临时产物树里独立进程跑真实命令,结构与血缘断言同源复用"
    - "run_all 沿用既有体例:逐步中文播报,一步失败即停,产物落在同一轮目录"
    - "人工核查检查点把数值读进图里核对:看图判断 + 跑命令对数,四项逐条勾认后才收口"

key-files:
  created:
    - .planning/phases/03-fisher-crlb/03-06-SUMMARY.md
  modified:
    - tests/test_smoke_realdata.py
    - src/calib/cli/main.py
    - docs/pipeline.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "冒烟链临时产物树自成一体,不复用历史轮:血缘对不上历史产物,且缩小版数据集注入分钟级可跑,真相源仍是落盘产物"
  - "run-all 接力到 fisher 为止:evaluate/report 属阶段四,一条命令的终点就是本阶段的 crlb.json"
  - "条件数单位伪影的修复不塞进本计划:属 noise 谱诊断范畴,在 main 上由编排者单独提交(0a9b247),本计划合并 main 收编"

patterns-established:
  - "检查点核查留痕体例:SUMMARY 里逐项记核查内容、产物路径与结论,核过的数字可复查"

requirements-completed: [NOI-01, NOI-02, NOI-03]  # 三需求全链闭环 + 人工核查通过,Phase 3 收口

# Metrics
duration: 约 12h(其中约 8h 为 Task 3 人工核查检查点的用户等待与 main 上的修复;纯执行约 3.5h)
completed: 2026-10-19
---

# Phase 3 Plan 6: 全链冒烟接力、run-all 接线与人工核查收口 Summary

**冒烟接力链扩到五步并在真实产物上跑绿(noise/fisher 结构与血缘断言齐备),`calib run-all` 一条命令直达 crlb.json,pipeline.md 契约与 REQUIREMENTS 状态同步收口;人工核查检查点四项全过,过程中揪出条件数单位伪影误报并在 main 上修复——Phase 3 三需求(NOI-01/02/03)全部完成。**

## Performance

- **Duration:** 约 12h(含人工核查等待;纯执行约 3.5h)
- **Started:** 2026-10-18T23:22(+0800,Task 1 提交)
- **Completed:** 2026-10-19T11:19(+0800,SUMMARY 提交)
- **Tasks:** 3/3(Task 3 为人工核查检查点,用户逐项看图/跑命令核实)
- **Files modified:** 4(测试 1 + CLI 1 + 文档 2)

## Accomplishments

- **Task 1(65f26d7)**:冒烟接力链从三步扩展到五步(ingest → corners → noise → calibrate → fisher),新增 noise/fisher 的结构契约断言(键全、数值 null 契约)与全链血缘指纹闭合断言(每个产物的 upstream 指向前序真实文件);链路在真实产物上全绿。
- **Task 2(2f59412)**:`calib run-all` 接力加 noise、fisher 两步(照原体例逐步中文播报),一条命令直达 crlb.json;docs/pipeline.md 第③⑤步契约速查按实际产物逐项更新,新增"条件数达退化线→改善摆拍"回环路径;REQUIREMENTS.md 中 NOI-02/03 勾选完成并标注完成计划号,追溯表与页脚计数同步(11/32)。
- **Task 3(人工核查检查点)**:四项核查全部由用户看图/跑命令核实通过(明细见下节);核查通过即 Phase 3 成功标准 1–5 逐条勾认,NOI-01/02/03 三需求收口。

## Task 3 人工核查记录(2026-10-19,四项全过)

| # | 核查项 | 核查方式与产物路径 | 结论 |
| - | ------ | ------------------ | ---- |
| 1 | 噪声传递曲线平滑单调、无鼓包 | 用户看图:`output/noise/` 当轮 `transfer_curve.png` | 通过 |
| 2 | noise.json 双口径 σ 数值合理 | 逐项读数:`output/noise/` 当轮 `noise.json`(detector / temporal 两口径) | 通过 |
| 3 | 交叉验证数字在门槛内 | 逐项读数:`output/fisher/run_20261018_145407/crlb.json` 各模型 max_rel_dev | 通过 |
| 4 | 16 张特征谱图与条件数分档 | 用户看图 + 跑命令对数:`output/fisher/run_20261019_025859/figures/` 16 张谱图 + `crlb.json` 条件数字段 | 通过(修正后复检,见 Deviations) |

## Deviations from Plan

### 检查点第 4 项暴露条件数单位伪影误报(修复在 main 上,超出本计划范围)

- **Found during:** Task 3 人工核查第 4 项(用户看图质疑)
- **Issue:** 03-05 轮的四模型条件数全部超退化线(m1 达 2.7e9)并归因"摆拍旋转多样性不足"——用户怀疑是"像素内参数值大、畸变系数数值小"的单位差灌水。数值验证属实:m1 原始 2.7e9 → 无量纲化 1.0e3,确为单位伪影误报。
- **Fix:** 修复由编排者在 **main** 上完成,不在本计划分支内:
  - 0a9b247 fix(noise):`spectrum_report` 改在无量纲坐标上算谱(像素内参÷焦距,畸变原样),原始单位条件数另存 `condition_number_raw_units`;补手算对表 + 单位换算不变性 + focal 校验测试
  - d0964a2 style:清理遗留 lint(不改行为)
  - 044eadc docs(03-05):SUMMARY 追加更正段 + deferred-items 更新
- **修正后结论:** 修复后重跑 `calib fisher`,新产物 `output/fisher/run_20261019_025859`(16/16 ok):m1(1.0e3)/m2(8.9e3)正常,m3(5.7e11)/m4(1.2e6)仍退化且极端方向整块落在高阶畸变项(k5/k2/k6、k3/k4/k2)——**真信号**:这批数据定不住复杂模型的高阶畸变项,是阶段四选型防过拟合的输入。CRLB 逐参数数值与修复前逐位一致(修复只影响谱诊断),逐参数 CRLB、Σ 带、交叉验证结论均不受影响。
- **本计划处置:** 合并 main 后本分支立于修复之上,用户已对修正后的谱图与四模型分档数字核查通过,检查点四项全过。

### 合并 main 与回归

- 合并 main(顶端 044eadc)入本分支:无冲突(两边改动文件不重叠——本计划改冒烟测试/CLI 接线/docs/REQUIREMENTS,main 改 noise 谱诊断与其测试)。
- 回归验证:`uv run pytest -m "not integration" -q` **350 passed, 19 deselected**;`uv run ruff check src tests` **全绿**。
- 数字说明:预期"约 353"未出现,原因清楚——本计划新增的冒烟测试在 `tests/test_smoke_realdata.py`,该文件整体标 `pytest.mark.integration`,在非集成跑法下被排除;350 已含 main 上修复新增的 3 个谱诊断测试(test_noise_fisher.py 共 27 用例全过),非零损失。

## Auth Gates

无。

## Known Stubs

无——冒烟链与 run-all 全部跑真实命令、落真实产物;文档契约与实际产物逐一核对过。

## Threat Flags

无新增信任面——本计划只改测试断言、CLI 接力序列与文档;无新输入解析、无网络、无反序列化新面。

## Self-Check: PASSED

- 文件存在:`tests/test_smoke_realdata.py`(五步接力断言)、`src/calib/cli/main.py`(run_all 含 noise/fisher)、`docs/pipeline.md`(crlb.json 契约)、`.planning/REQUIREMENTS.md`(NOI-03 勾选)、本 SUMMARY——全部在。
- 提交存在:65f26d7 / 2f59412 / 合并提交(7cf8aba)/ SUMMARY 提交——`git log` 可查。
- 测试:非集成 350 passed;ruff 全绿;main 上修复的谱诊断测试(test_noise_fisher.py,27 用例)合入后全过。
