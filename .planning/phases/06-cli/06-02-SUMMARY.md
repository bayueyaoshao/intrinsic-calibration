---
status: complete
date: 2026-10-23
phase: 06-cli
plan: 02
subsystem: report
tags: [report, loopback, diff, suggest-exclude, iter]

# Dependency graph
requires:
  - 06-01 的 report/build.py 组装层与 report_data.json 底账快照(metrics_raw 是跨运行 diff 的取数基础)
  - evaluate.json 判定段(selection.verdicts / lopo)、analyze.json 六段判定、corners burst_check、manifest precheck
provides:
  - report/loopback.py:build_loopback_advice 纯函数(内置规则表:十类触发 → 回哪一步①–⑧ + 做什么,全通俗中文)
  - report/diff.py:collect_exclude_candidates(三路剔除候选)+ compare_with_previous(六键跨运行对比,方向词写死)
  - 报告⑧节从占位变真身(建议表/判据缺席/候选表/固定操作指引),⑥之后新增「与上次对比」节
  - 报告轮落盘五件:report.html / report.md / report_data.json / figures/ / suggest_exclude.json
affects: [06-03 (run-all 第⑧步消费 report 命令), 06-04 (换数据重跑路径验证), 下一轮报告 (diff 锚点契约)]

# Tech tracking
tech-stack:
  added: []            # 零新依赖,纯 numpy-free 字典加工
  patterns:
    - "规则表即常量:触发谓词 + 建议行(触发判据/回哪一步/做什么)内置在 loopback.py,判定段缺席记行不冒充,全达标才给单行「无需回环」"
    - "路标逐字对齐:八步步骤名做常量 PIPELINE_STEPS,与 docs/pipeline.md 逐字一致,测试锁定"
    - "diff 锚点 = 上一份 report 轮的 report_data.json(裁定②):锚点查找最新往旧取第一份带底账的轮,只经 json.load,坏底账退而找更旧"
    - "规模差异不冒充回归:bootstrap 次数/实验档位入 metrics_raw,diff 备注如实解释"
key-files:
  created:
    - src/calib/report/loopback.py
    - src/calib/report/diff.py
    - tests/test_report_loopback.py
    - tests/test_report_diff.py
  modified:
    - src/calib/report/build.py (⑧节真身 + s_diff 节 + metrics_raw 扩键 + suggest_exclude.json 落盘)
    - src/calib/report/templates/report.html.j2 (⑧节替换 + 「与上次对比」节)
    - src/calib/report/templates/report.md.j2 (同 HTML)
key-decisions:
  - "回环建议谓词全部机器可读:三判据读 verdicts 行 pass 布尔;预算读 verdict==fail;外推区读 r_max_obs 与 r_corner_frame 差;各向异性读中位>显著线(不是逐角点计数,真实数据中位 0.68<1.5 不误报);敏感度建议逐条转行"
  - "角点寻优「非当前配置最优」复算口径与 analyze 判读原文一致:未显著劣化单元(自参照 RMS<阈值)自参照平均,候选比生产口径低 ≥30% 且单元数 ≥3 才点名——真实数据 classic_w7 0.0143px vs 生产 sb 0.0507px 可逐位复现"
  - "剔除候选「存疑复核计数高」= 被 ≥2 类判定来源同时点名(13 张进表);89 张单来源例行存疑不进表,防止名单淹没(存疑只标记不剔除的既定纪律)"
  - "建议行步骤名 = docs/pipeline.md 八步常量;固定操作指引照裁定①:人工确认后填 config/calibration.yaml 的 exclude_images(唯一剔除名单),再 calib run-all --from calibrate 复用角点噪声产物"
  - "diff 对比键方向语义写死:六键全部「越低越好」(预算比值 = 实测÷理论下界,降 = 更贴近下界 = 变好);当选模型是名字,只报持平/换成谁;变化相对容差 0.1% 内记持平"
requirements-completed: [ITER-02, ITER-03, ITER-04]

# Metrics
duration: ~17 分钟(纯执行;真实数据 calib report 两轮各约 30 秒)
completed: 2026-10-23T02:49+08:00
---

# Phase 6 Plan 02: 回环三件 Summary

**报告从静态证据链升级为能指路的证据链:⑧节回环建议表由规则引擎从判定段现算(真实数据触发 6 条:稳定 FAIL 回①补拍/外推区回①/敏感度三条回①/寻优点名 classic_w7 回②)、suggest_exclude.json 落 37 条候选(lopo 0 + 连拍断言 24 + 入库多来源存疑 13)、⑥后新增「与上次对比」节(锚点上一份报告轮,第二轮六键全真实数字对比),三段全部进 report_data.json 快照。**

## 做了什么(三任务全部完成)

1. **Task 1 — loopback.py 回环建议规则引擎(ITER-04)**(`dd5f823`,TDD:RED 先失败后 GREEN):`build_loopback_advice(evaluate_report, analyze_report)` 纯函数零 IO;内置规则表覆盖 evaluate 三判据判行(泛化→⑥复核/④换简模型、稳定→①补拍先看⑦、逼近 CRLB→③核对口径)+ analyze 六段(预算 fail→③口径对账、外推区→①补拍边缘位姿、敏感度 advice 逐条→①、test 解不出→②、寻优更优候选→②、各向异性超线→①);判定段缺席记「判据缺席」行不冒充,全达标才给单行「无需回环」;八步路标常量与 docs/pipeline.md 逐字一致(测试锁定)。
2. **Task 2 — suggest_exclude.json 候选生成 + 报告⑧节接入(ITER-02)**(`5125a81`):`collect_exclude_candidates` 三路候选(留一姿态点名按 manifest 逐张展开、连拍断言失败组按姿态去重展开、入库体检 ≥2 类来源同时点名的存疑),每项 {image_id, pose_id, source, reason 白话带原文数字};build.py ⑧节占位换真身(建议表 + 判据缺席行 + 候选表 + 固定操作指引),run_report 落盘 suggest_exclude.json(零候选也落空表);双模板⑧节渲染三块。
3. **Task 3 — 跨运行对比「与上次对比」节(ITER-03)**(`7c23090`):`compare_with_previous(current, previous)` 纯函数六键对比(当选模型/test RMS/预算比值三统计量/噪声底 σ),方向词按指标含义写死;单键缺失记「上次无此数据」不崩;bootstrap 次数/档位不同给规模差异备注;首次运行给中文一句;build.py 新增 `_previous_report_doc` 锚点查找(裁定②),metrics_raw 扩预算比值 p95/max 与实验规模键;双模板⑥之后插节(先看本次考试,再看比上次好没好),上次轮名与生成时间随行披露(T-06-03)。

## 验证数字(计划 verify 全部照跑)

| 验证项 | 命令 | 结果 |
|---|---|---|
| Task 1 规则引擎 | `.venv/bin/python -m pytest tests/test_report_loopback.py -x -q` | **8 passed**(0.05s);RED 阶段先确认失败(模块不存在) |
| Task 2 候选+⑧节 | `.venv/bin/python -m pytest tests/test_report_loopback.py tests/test_report_diff.py -x -q` | **13 passed** |
| Task 3 diff+build | 同上 + `tests/test_report_build.py` | **20 passed + 5 passed**(零回退) |
| 真数据首轮 | `uv run calib report` | 退出码 0 → `run_20261022_184718`:diff 状态 ok、锚点 `run_20261022_154326`、test RMS 0.1598 持平、新键「上次无此数据」;⑧节 6 条建议;候选 37 条 |
| 真数据第二轮 | `uv run calib report` | 退出码 0 → `run_20261022_184747`:**六键全真实数字对比**(0.1598/101.7680/1107.6025/1443.8441/0.0087 全持平),锚点 = 上一轮 184718,保留策略正确 |
| 真数据报告测试 | `.venv/bin/python -m pytest tests/test_report_realdata.py -q -m realdata` | **4 passed**(5.1s) |
| 静态检查 | `.venv/bin/ruff check src tests/test_report_*.py` | All checks passed |
| 残留占位扫描 | grep TODO/FIXME/placeholder | 零残留(06-01 登记的 s8 占位已由本计划消除) |

真实数据⑧节建议表(6 条,步骤名全部出自八步路标):稳定 FAIL→①、外推区(r=0.685 对四角 0.797)→①、敏感度建议 3 条→①、寻优 classic_w7 更稳(0.0143px 对 sb 0.0507px)→②。各向异性未触发(中位 0.68 < 显著线 1.5,判定谓词避开了逐角点计数的误报)。suggest_exclude.json 37 条候选 = 连拍断言 3 个姿态 × 8 张(24)+ 入库体检多来源存疑 13(lopo 真实数据零点名)。

## 提交号列表

1. **Task 1** - `dd5f823` (feat): 回环建议规则引擎——判定段到「回哪一步、做什么」的建议行
2. **Task 2** - `5125a81` (feat): 剔除候选清单 + 报告⑧节回环建议接入(ITER-02)
3. **Task 3** - `7c23090` (feat): 跨运行对比「与上次对比」节——上一份报告轮作锚点(ITER-03)

## Files Created/Modified

- `src/calib/report/loopback.py` - 回环建议规则引擎:PIPELINE_STEPS 八步常量 + build_loopback_advice 纯函数(十类触发/缺席记行/全达标单行)
- `src/calib/report/diff.py` - collect_exclude_candidates(三路候选)+ compare_with_previous(六键对比/缺键不崩/规模备注/首次运行)
- `src/calib/report/build.py` - ⑧节真身组装、s_diff 节、_previous_report_doc 锚点查找、metrics_raw 扩 4 键、suggest_exclude.json 落盘
- `src/calib/report/templates/report.html.j2` / `report.md.j2` - ⑧节替换 + 「与上次对比」节(⑥之后)
- `tests/test_report_loopback.py` - 8 测试:逐条单触发/全达标/缺席/步骤名纪律
- `tests/test_report_diff.py` - 12 测试:三路候选单收/零候选/⑧节整链/方向词/缺键/首次运行/规模备注/真锚点整链

## Decisions Made

见 frontmatter key-decisions 五条。核心两条:①回环谓词全部机器可读且与判定段原文口径一致(角点寻优复算口径可逐位复现判读原文数字);②剔除候选「计数高」= 多来源同时点名,例行单来源存疑不进表。

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Plan Gap] collect_exclude_candidates 签名漏了 corners 文档**
- **Found during:** Task 2
- **Issue:** 计划签名 `collect_exclude_candidates(evaluate_report, analyze_report, manifest)` 只有三个文档,但三路候选之一的「连拍断言失败张」实际存在 corners 轮 detection_summary.json 的 burst_check 段(analyze 的 anisotropy excluded 清单是噪声来源过滤,不是连拍断言)
- **Fix:** 加第 4 个参数 `corners_report`;analyze_report 参数保留与计划签名一致(文档注明当前不读)
- **Files modified:** src/calib/report/diff.py
- **Committed in:** 5125a81

**2. [Rule 3 - Plan Gap] metrics_raw 缺 diff 所需的键**
- **Found during:** Task 3
- **Issue:** 计划要求对比「预算比值三统计量」,但 06-01 底账只有 budget_ratio_median;规模备注也需要 bootstrap 次数,快照里没有
- **Fix:** metrics_raw 扩 4 键:budget_ratio_p95 / budget_ratio_max / n_bootstrap_resamples / profile;旧版底账缺键走「上次无此数据」路径(有测试)
- **Files modified:** src/calib/report/build.py
- **Committed in:** 7c23090

---

**Total deviations:** 2 处(Rule 3 × 2,均为签名/底账键的口径缺口,无行为变更)
**Impact on plan:** 都是为兑现计划行为条目(三路候选/三统计量对比/规模备注)的必要补齐,plan 的 success_criteria 全部达成。

## Threat Model Compliance

- **T-06-03(上一份 report_data.json 篡改):** 锚点只经 json.load 读取;「与上次对比」节随行披露上次轮名与生成时间;规模差异给备注不冒充回归;坏底账(读不动)退而找更旧、全坏按首次运行处理,绝不 eval
- **T-06-04(剔除候选抵赖):** 每条候选带 source 与判据原文数字(z 分数/波动/容差/判定来源);人工确认后才进 exclude_images,全自动改配置按裁定拒绝;名单机制唯一(exclude_images),不造第二套

无计划外新增威胁面:全部新落盘都在 output/report/ 轮目录内,输入只有既有的七路 JSON + 上一份底账。

## Notes for Next Plans

- **06-03(run-all 第⑧步):** report 命令体未动,`--from calibrate` 指引文案已在⑧节固定操作指引里背书;并行执行器已收口 run-all(937a220 / cb9df9c),与本案无文件冲突
- **06-04(可复用性):** ⑧节步骤名常量 PIPELINE_STEPS 可复用;diff 锚点契约依赖 report 轮保留策略(只留两轮)
- 同秒连跑两次 calib report 仍会落同名轮并互相覆盖(阶段六 deferred-items 既有观察项,本案未改动该行为;真实两轮间隔 29 秒无碰撞)

## Self-Check: PASSED

- 4 个 created 文件与 3 个 modified 文件全部在盘(git status / ls 核验)
- 3 个任务提交号在 git log 中核验存在:dd5f823 / 5125a81 / 7c23090
- 真实数据报告轮在盘两轮 `output/report/run_20261022_184747` 与 `run_20261022_184758` 五件产物齐(suggest_exclude.json 在列,gitignore 内不进 git);第三轮 184758 由 realdata 测试夹具再跑产生,其 diff 锚点正确指向前一轮 184747——保留策略与锚点行为再次实证
- 上述「验证数字」表内数字均来自当轮实际命令输出

---
*Phase: 06-cli*
*Completed: 2026-10-23*
