---
status: complete
date: 2026-10-23
phase: 06-cli
plan: 01
subsystem: report
tags: [jinja2, report, evidence-chain, html, markdown, cli]

# Dependency graph
requires:
  - phase: 01~05 全阶段
    provides: 七路上游落盘轮(ingest manifest / corners detection_summary / noise / calibration comparison / fisher crlb / evaluate / analyze),latest_run_dir 轮次定位与血缘登记契约
provides:
  - report/build.py 组装纯函数层:build_report_context(out_root) 把七路上游读成全中文 context(全部文案/排名/换算在组装层定稿)
  - run_report(out_root) 顶层一步到位:组装 → jinja2 双模板渲染 → 落盘四件(report.html / report.md / report_data.json / figures/)
  - HTML/Markdown 双模板(零本数据集数字,缺席段条件渲染,术语白话速查,图表编号引用)
  - calib report 真命令:替换占位,新轮落盘成功后只留最新两轮,四要素人工核查播报
  - report_data.json 机器可读底账(schema report-context-v1,含 metrics_raw 原始数块)= 06-02 跨运行对比的取数基础
affects: [06-02 (跨运行对比/回环建议只需往 context 加三段), 06-03 (run-all 接第⑧步), 06-04 (可复用性换数据重跑)]

# Tech tracking
tech-stack:
  added: [jinja2>=3.1 (uv lock 实装 3.1.6 + markupsafe 3.0.4)]
  patterns:
    - "组装与排版分离:数字与文案全部在 build 层定稿成 context 字典,模板只排版、零本数据集数字"
    - "底账原样快照:report_data.json = context 原样落盘,报告里每个数字可逐位回查(反伪造证据链)"
    - "缺席披露不冒充数值:失败轮/缺席段/血缘已清理上游进 disclosures 清单;只有直接取数主文件整个缺失才中文报错退出"
key-files:
  created:
    - src/calib/report/build.py
    - src/calib/report/templates/report.html.j2
    - src/calib/report/templates/report.md.j2
    - tests/test_report_build.py
    - tests/test_report_realdata.py
  modified:
    - pyproject.toml (jinja2 依赖)
    - uv.lock
    - src/calib/cli/main.py (report() 占位 → 真命令体 + import)
key-decisions:
  - "报告数字逐位同源锁:context 里报告用的值全部定稿为显示字符串,report_data.json 原样快照;另设 metrics_raw 私有块保留原始数(浮点),专供 06-02 跨运行 diff,模板不渲染"
  - "s9 图表清单条目携带 _src 私有键(源文件绝对路径),命令层 run_report 按它复制;模板只读 file/caption/section 公开键"
  - "报告轮保留策略放命令层:新快照四件全部落盘成功之后才删更旧轮、只留两轮(上一轮 = 06-02 diff 的读锚);绝不先删后写"
  - "Markdown 模板 autoescape=False(纯文本格式开转义会把正文 & 改写成 &amp;);防篡改由「模板零本数据集数字」保证;HTML 端 autoescape 常开并有测试锁定"
  - "残差热图按 2026-10-22 用户裁定① = 引用 analyze 轮既有 radial_bands.png + coverage_footprint.png,报告层不新画任何图"
  - "crlb.json 键名 views640/poses80 不改名,白话注解(全帧版/姿态聚合版)随行走"
patterns-established:
  - "报告九节骨架:①结论速览 ②数据与体检 ③角点 ④噪声底 ⑤理论极限 ⑥防作弊考试 ⑦误差归因 ⑧回环建议 ⑨产物索引与血缘链(HTML/MD 同 context 同九节)"
  - "判行行体例:有判行素材的节首行渲染 PASS/FAIL 串(HTML 按 FAIL 字样分绿/红样式)"
  - "四要素人工核查播报:看哪个文件 / 文件作用 / 具体看哪里 / 好坏判据"
requirements-completed: [REP-01, REP-02]

# Metrics
duration: ~3h20m(含 1h30m 全量白天档回归 + 50min 注入既有失败取证重跑被中断)
completed: 2026-10-23T02:20+08:00
---

# Phase 6 Plan 01: 报告组装库层与模板 Summary

**`calib report` 从占位变真命令:组装层把七路上游定稿成全中文 context(报告数字与底账 report_data.json 逐位同源),jinja2 双模板只排版出九节证据链报告(HTML autoescape / MD),真实数据首份报告已落盘 run_20261022_154306(当选 m2、17 张图、3 条缺席披露)。**

## 做了什么(三任务全部完成)

1. **Task 1 — jinja2 进依赖 + 组装纯函数层**(`fa1d33d`):pyproject 加 jinja2>=3.1 并 uv lock + uv sync(实装 3.1.6);新建 `src/calib/report/build.py`(约 880 行):`build_report_context(out_root)` 逐段 latest_run_dir 定位七路上游只读不重算;直接取数主文件缺失中文 ValueError 并指路对应命令;失败轮/analyze 六段缺席/血缘已清理上游 → disclosures 如实披露不断链;物理换算尺 = square_size_mm ÷ sq_px 中位数现算(零硬编码);内参表三列不确定度并排(CRLB se / bootstrap σ / 重复张分组 σ);九节骨架与判行/排名/白话文案全部在此定稿;context 可原样 JSON 序列化。
2. **Task 2 — 双模板 + calib report 命令落地**(`ddb3bcf`):build.py 追加 `run_report(out_root)` 顶层一步到位(组装 → 渲染 → 落盘四件 + 图表从各上游轮按 `_src` 原样复制);双模板 `report.html.j2`(autoescape=True + StrictUndefined)与 `report.md.j2` 同 context 同九节、零本数据集数字、术语白话速查内置、判行行 PASS/FAIL、缺席段条件渲染、图表按节内嵌 + ⑨节总索引与血缘链;main.py `report()` 替换占位——命令体只调一个顶层函数(06-02 加段不改命令体),落 `output/report/run_<ts>/{report.html,report.md,report_data.json,figures/}`,新快照落盘成功后删更旧轮只留两轮,收尾四要素播报。
3. **Task 3 — 报告层测试**(`8a5b18a`):`tests/test_report_build.py`(白天快速档,5 个测试,秒级):合成七路树 → 九节骨架/判行如实/换算尺现算/可序列化;主文件缺失中文报错;双模板同源一致(从 context 取定稿串回查渲染产物)+ HTML 转义锁定(含 `<` 和 `&` 的字段值)+ 底账同源 + 只留最新两轮 + 四要素播报。`tests/test_report_realdata.py`(整文件 integration+realdata,4 个测试):真实最新轮跑一次 calib report,退出码 0、四件产物齐、报告数字与底账逐位同源、六类图(角点叠加/残差分带裁定①/CV 曲线/CRLB 对比/敏感度排序/摆位覆盖)逐一进索引且复制在 figures/;前置不全中文理由跳过。

## 验证数字(计划 verify 全部照跑)

| 验证项 | 命令 | 结果 |
|---|---|---|
| 组装层+模板测试 | `.venv/bin/python -m pytest tests/test_report_build.py -q` | **5 passed**(0.9s) |
| 真实数据出报告 | `uv run calib report` | **退出码 0** → `output/report/run_20261022_154306/` 四件齐:当选 m2、图表 17 张、缺席披露 3 条(全部为「上游已清理」如实披露) |
| 真数据报告测试 | `.venv/bin/python -m pytest tests/test_report_realdata.py -q -m realdata` | **4 passed**(1.3s) |
| 全量白天档回归 | `.venv/bin/python -m pytest tests/ -q -m "not realdata"` | **613 passed,1 failed,51 deselected**(1h30m);唯一失败 = `test_noise_injection.py::test_full_injection_real_data`,系阶段五 deferred-items **已登记的既有失败**(同一测试同一根因),与 06-01 三提交零关联 → **零回退** |
| 抽查同源 | report_data.json vs report.html | 判行行「泛化 PASS / 稳定 FAIL / 逼近 CRLB PASS → 综合 FAIL」、换算尺 1px ≈ 0.552mm、test RMS 0.1598px ≈ 0.088mm 逐位同源 |
| 静态检查 | `.venv/bin/ruff check src tests/test_report_build.py tests/test_report_realdata.py` | All checks passed |
| 合成树 CLI 端到端 | /tmp 合成树跑 `.venv/bin/calib report`(用后即删) | 退出码 0、四件产物齐、九节标题齐、判行行与缺席披露文案在、连跑三次只留最新两轮 |

## 提交号列表

1. **Task 1** - `fa1d33d` (feat): jinja2 进依赖 + report/build.py 组装纯函数层
2. **Task 2** - `ddb3bcf` (feat): jinja2 双模板 + calib report 命令落地
3. **Task 3** - `8a5b18a` (test): 报告层测试——合成冒烟 + 真实最新轮渲染
4. **收尾** - (本次 docs 提交): SUMMARY + deferred-items

## Files Created/Modified

- `src/calib/report/build.py` - 组装纯函数层:build_report_context(七路 → 全中文 context + disclosures + metrics_raw)+ run_report(渲染 + 落盘四件)
- `src/calib/report/templates/report.html.j2` - HTML 排版模板(autoescape 常开,零数据集数字)
- `src/calib/report/templates/report.md.j2` - Markdown 排版模板(同 context 同九节)
- `src/calib/cli/main.py` - report() 占位 → 真命令体(调 run_report + 保留两轮 + 四要素播报);import shutil / run_report
- `pyproject.toml`、`uv.lock` - jinja2>=3.1 依赖锁定(3.1.6)
- `tests/test_report_build.py` - 白天快速档 5 测试(合成树,秒级)
- `tests/test_report_realdata.py` - 夜间档 4 测试(真实最新轮,文件级 realdata 标记)

## Decisions Made

见 frontmatter key-decisions 六条。核心一条:报告层「组装与排版分离」+「底账原样快照」双锁——报告里每个数字都是 build 层定稿字符串,report_data.json 原样落盘,逐位同源可回查;机器 diff 用的原始数单独走 metrics_raw 私有块。

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Task 1 的 verify 命令引用了尚不存在的测试文件**
- **Found during:** Task 1(verify 阶段)
- **Issue:** 计划 Task 1 的 verify 写 `pytest tests/test_report_build.py`,但该文件属 Task 3 交付,Task 1 时点不存在
- **Fix:** Task 1 用等价内联合成树冒烟替代(合成最小七路上游 → context 九节齐/判行/缺席披露/JSON 可序列化/科学计数小值);Task 3 已把同等断言固化进正式测试文件
- **Files modified:** 无额外文件(验证方式替代)
- **Verification:** 冒烟全绿;Task 3 后 `tests/test_report_build.py` 5 passed
- **Committed in:** fa1d33d(Task 1 提交内)

**2. [Rule 3 - Plan Gap] run_report() 顶层函数落在 build.py(计划 Task 2 文件清单未列)**
- **Found during:** Task 2
- **Issue:** 计划要求「命令只调一个顶层函数(组装+渲染+落盘一步到位)」,但 Task 2 的 files_modified 只列 templates + main.py——顶层函数只能住在组装层 build.py
- **Fix:** build.py 追加 run_report()(+ import shutil / jinja2 / TEMPLATES_DIR 常量);main.py 命令体保持「只调一个顶层函数」的计划意图
- **Files modified:** src/calib/report/build.py
- **Verification:** 合成树 CLI 端到端退出码 0、四件产物齐
- **Committed in:** ddb3bcf(Task 2 提交内)

---

**Total deviations:** 2 处(Rule 3 × 2,均为计划文件清单/verify 命令的口径缺口,无行为变更、无范围蔓延)
**Impact on plan:** 两处都是让实现与计划意图(顶层单函数命令体 / 正式测试)对齐的必要修正。06-02 只需往 context 加三段,命令体与模板骨架无需返工——已按计划的 success_criteria 达成。

## Issues Encountered

- 全量白天档回归耗时 1h30m(613 passed),期间唯一失败为阶段五已登记既有失败(见下「Known Stubs / 遗留」);执行中曾单独重跑该注入测试取证,约 50 分钟未跑完被中断,既有登记材料已足够,不再重复消耗。
- `calib report` 同秒重跑会落同一个 `run_<ts>` 轮(秒级时间戳,与全仓库轮次命名约定一致,后跑完整覆盖)——已记入阶段六 deferred-items 观察项,是否加亚秒后缀留给 06-02 视 diff 需要定。
- `main.py run_all()` 的收尾播报文案过时(仍称第⑥~⑧步未实现)——属 06-03/06-04 文件范围,只登记不越界。

## Known Stubs

| 位置 | 说明 | 处置 |
|---|---|---|
| `src/calib/report/build.py` 第⑧节 `s8.placeholder` | 「回环建议」节按计划(Tasks Task 2 action)为占位:「本节是『回哪一步、做什么』的自动建议表,由后续版本接入(输入就是本报告第⑥⑦节的判定段)」——**计划内有意占位**,模板正常渲染该文案,不是空数据冒充 | 06-02 接入规则引擎三段时替换 |

除此之外全仓 grep 无 TODO/FIXME/占位文本;所有节的数据都真实接自落盘轮,缺席段走披露路径而非空值冒充。

## Threat Model Compliance

- **T-06-01(模板渲染篡改):** HTML autoescape 常开 + StrictUndefined,`tests/test_report_build.py` 有 `<`/`&` 转义断言锁定;上游产物只经 json.load,不 eval
- **T-06-02(失败/缺席抵赖):** 主文件缺失中文报错退出;缺席段/失败轮/血缘已清理照实写进报告披露清单与 report_data.json(真数据轮 3 条「上游已清理」披露即实证)
- **T-06-SC(jinja2 新依赖):** uv lock 锁定 3.1.6,来源 PyPI 正名包
- 无计划外新增威胁面(报告命令只读七路 JSON + 复制 PNG + 写 output/report/,均在目录契约内)

## Next Phase Readiness

- **06-02(跨运行对比 + 回环建议):** report_data.json 的 metrics_raw 原始数块已就位(9 个关键指标 + 七轮轮次名);上一轮报告按保留策略留存作 diff 读锚;s8 占位节与命令体无需返工,往 context 加三段即可
- **06-03(run-all 接第⑧步):** `calib report` 已是可从命令行直接调用的完整命令(退出码契约与上游检查齐)
- **06-04(可复用性):** 换数据重跑路径已在报告⑨节写明白话指引(reuse_line),模板零数据集数字已由测试锁定
- 遗留事项已登记 `.planning/phases/06-cli/deferred-items.md`(既有注入测试失败承接阶段五 + 两条观察项)

## Self-Check: PASSED

- 5 个 created 文件与 3 个 modified 文件全部在盘(git status / ls 核验)
- 3 个任务提交号在 git log 中核验存在:fa1d33d / ddb3bcf / 8a5b18a
- 真实首份报告轮 `output/report/run_20261022_154306/` 四件产物齐(gitignore 内,不进 git)
- 上述「验证数字」表内数字均来自当轮实际命令输出

---
*Phase: 06-cli*
*Completed: 2026-10-23*
