---
phase: 06-cli
reviewed: 2026-10-22T21:22:59Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - src/calib/report/build.py
  - src/calib/report/loopback.py
  - src/calib/report/diff.py
  - src/calib/report/templates/report.html.j2
  - src/calib/report/templates/report.md.j2
  - src/calib/cli/main.py
  - tests/test_report_build.py
  - tests/test_report_loopback.py
  - tests/test_report_diff.py
  - tests/test_report_realdata.py
  - tests/test_cli_run_all.py
  - tests/test_run_all_variant_realdata.py
findings:
  critical: 0
  warning: 4
  info: 7
  total: 11
status: issues_found
---

# 阶段六:代码评审报告

**Reviewed:** 2026-10-22T21:22:59Z
**Depth:** standard(逐文件 + 跨模块形状核对)
**Files Reviewed:** 12(11 份在册 + 上游形状旁证:canonical.py / experiments.py / corner_opt.py / budget.py / precheck.py / manifest.py)
**Status:** issues_found

## Summary

评审范围 = `f893fa5..HEAD` 十五个提交:报告组装层(build/loopback/diff + 双模板)、
run-all 八步重写与 --from/--until、三命令退出码收口、六个测试文件。逐文件通读,
并把回环规则、剔除候选、跨运行对比读取的每一个键名与上游真实落账形状逐一核对
(lopo.outlier_poses 字典格式、budget.recheck.ratios 中文键、corner_opt.stats._self、
evaluate meta.profile、precheck.suspect_sources——均对上;唯 burst 的 threshold_px
对不上,见 WR-01)。

总体判断:报告「只读不重算、缺席披露不冒充数值、底账逐位同源」三条设计纪律落实
到位且有测试锁定;CRLB 口径铁律在报告层无违例(只展示不重算,双口径并排 + 错配
警告文案,回环「预算 fail → 回③核口径」与铁律正确衔接);HTML 自动转义全覆盖、
无 eval/exec、模板零本数据集数字(「0.75」「头发丝 0.05~0.1mm」为通用参照,不违
REU-02)。缺陷集中在:对真实 burst 落账形状的两处错配(WR-01/02,合成测试把标量
形状锁了进去)、report 命令漏掉本阶段自己定下的退出码收口(WR-03)、真实 STEPS
表顺序只有夜间档能兜住(WR-04)。无 critical。

## Critical Issues

无。

## Warnings

### WR-01: burst `threshold_px` 是二元列表,报告显示退化 + 候选理由丢容差

**File:** `src/calib/report/build.py:476`、`src/calib/report/diff.py:107`
**Issue:** 上游真实落账里连拍断言失败的 `threshold_px` 是二元列表
`[中位容差, 最大容差]`(`src/calib/corners/canonical.py:80`、`src/calib/cli/main.py:324`),
不是标量。两处后果:
1) build.py:476 `(容差 {f.get('threshold_px') or '—'})` 把列表原样渲染进报告③节,
   读者看到「(容差 [0.35, 1.4])」两个不带标注的数;
2) diff.py:107 `_fmt_px` 对列表做 `float()` 抛 TypeError 被 except 吞掉 →
   剔除候选理由里显示「容差 —」,数字整个丢失。
`tests/test_report_diff.py:95-97` 与 `test_report_build.py:85-87` 合成数据喂的是
标量 `1.0`——测试把与生产不符的形状锁死,真数据一跑就露馅。
**Fix:** 取 `threshold_px[1]` 并标注「最大容差」(或中位/最大两个值分行标注);
同步把两处合成测试的 `threshold_px` 改回二元列表,锁真实形状。

### WR-02: 连拍候选无视生产端已点名的 `suspect_image_ids`,整组稀释人工名单

**File:** `src/calib/report/diff.py:99-117`
**Issue:** corners 轮 burst failures 每条已带精确到帧的 `suspect_image_ids`
(`src/calib/cli/main.py:319-321`,按「偏离超最大容差」点名具体跳位帧);
`_burst_candidates` 不读它,把该姿态**全部**连拍从 manifest 整组展开成候选。
留一姿态整组点名是设计使然(整组都值得复核),连拍断言明明有逐帧精度却不用,
人工确认名单混入大张数好照片,稀释剔除名单的指向性。grep 证实 diff.py 与其
测试从不引用 `suspect_image_ids`。
**Fix:** failures 条目带 `suspect_image_ids` 时逐张点名(理由沿用),缺席再退回
整组展开;测试补一条「逐帧点名优先」的用例。

### WR-03: `calib report` 命令漏做退出码收口,坏输入吐全堆栈

**File:** `src/calib/cli/main.py:2402-2412`
**Issue:** 上游主文件缺失是本命令**最常见**的坏输入(build.py:243-245 特意抛中文
ValueError「请先运行 calib noise 再来」),但 report 命令体没有 try/except,
typer 会把完整 Python 堆栈吐给用户(退出码仍是 1,「不吐堆栈」违约)。
同阶段 06-03(T-06-07)已给 ingest/corners/calibrate 三处收口
(main.py:267/766/1114),report 写于 06-01、早于该约定,漏网。
**Fix:** 套用同款 `except (ValueError, OSError)` → 中文一句 + `typer.Exit(code=1)`,
指引语按 report 的坏输入定制(「先按流水线顺序补跑上游命令」)。

### WR-04: 真实 STEPS 表的八步顺序没有快速档测试锁定

**File:** `tests/test_cli_run_all.py:24-43`
**Issue:** `_install_fake_steps` 把 `cli_main.STEPS` **整表替换**成按
PIPELINE_ORDER 排好的假函数,之后所有断言只锁假表的顺序——真实表
(`src/calib/cli/main.py:2445`)若被改乱序、漏一步或指错函数,白天快速档全绿,
只有夜间 realdata 全链(变体测试跑 ①–⑥ 真子进程)能兜住。「顺序对齐
pipeline.md」这一 06-03 的核心收口实际处于无快速档守卫状态。
**Fix:** 补一条直接断言:`[name for name, _ in cli_main.STEPS] == PIPELINE_ORDER`
(顺带可核 `_STEP_CN` / `_STEP_OUTPUT_DIR` 键齐)。

## Info

### IN-01: report 命令 docstring 写「落盘四件」,实际五件

**File:** `src/calib/cli/main.py:2407`
**Issue:** docstring 说「落 output/report/run_<时间戳>/ 四件」,实际 run_report 落
五件(还有 suggest_exclude.json);变体测试 `test_run_all_variant_realdata.py:286-288`
按五件验收,文案漏一件。
**Fix:** docstring 改「五件」并补 suggest_exclude.json。

### IN-02: manifest summary 取法不一致,null 时会整报告崩

**File:** `src/calib/report/build.py:399`
**Issue:** `manifest_doc.get('summary', {}).get('n_test')`——键存在但值为 null 时
返回 None,接 `.get('n_test')` 即 AttributeError,整份报告崩;同函数 415 行用的
`or {}` 写法才是对的。触发面窄(需 test_rms 存在而 summary 为 null),但属同类
键空值守卫不一致。
**Fix:** 改成 `((manifest_doc.get('summary') or {}).get('n_test'))` 同款。

### IN-03: 代码注释行话「语义」漏网

**File:** `src/calib/report/diff.py:177`、`diff.py:206`
**Issue:** 「方向语义按指标含义写死」两处——「语义」是 CLAUDE.md 点名的软件工程
黑话,代码注释在通俗中文硬性约束范围内(report/ 目录其余文件已扫,仅此两处)。
**Fix:** 改白话,如「变化方向按指标含义写死」。

### IN-04: diff 方向词的「升高变好」分支是死代码

**File:** `src/calib/report/diff.py:221-222`
**Issue:** `_change_word` 末尾的 higher 方向分支不可达:`_DIFF_KEYS`(179-186)
第三列只有 `name` / `lower` 两种。留着无害,但读者会误以为有指标走「越高越好」口径。
**Fix:** 若为将来预留,加一行注释说明;否则删。

### IN-05: 三路候选不按 image_id 去重,同一张照片可重复上榜

**File:** `src/calib/report/diff.py:166-170`
**Issue:** 三路并集直接拼接:同一张照片可同时被「留一点名(整组展开)」和
「入库体检多来源点名」各收一条,候选表重复两行(理由不同)。对人工确认是噪声,
suggest_exclude.json 也随之重复。
**Fix:** 同 image_id 合并为一条、理由拼接来源,或第二路起跳过已收张。

### IN-06: 私有键 `_src`(绝对路径)原样序列化进底账 report_data.json

**File:** `src/calib/report/build.py:193`、`build.py:1014-1016`
**Issue:** `_plan_figures` 给每条图表附 `_src` 绝对路径「模板不渲染」,但 context
原样 json 落盘——底账里出现机器本地绝对路径,换机/移目录后失真,也混入排版用
临时键,「context 原样快照」的干净度打折。
**Fix:** 落盘前浅拷贝 context 并剔除 figures 里的 `_src`(渲染照常用内存里的原键)。

### IN-07: 同秒两轮覆盖——已登记观察项,确认 report 层未加重

**File:** `src/calib/report/build.py:977`、`build.py:210-221`、`src/calib/cli/main.py:2415-2420`
**Issue:** 秒级轮名同秒重跑会整轮覆盖,deferred-items.md(06-01 观察项)已登记。
评审专项核实:report 层**未加重**——`_previous_report_doc` 在本轮落盘前读锚
(先读后写,见 06-02 裁定②),清理只删严格更旧的轮且保两轮,对比锚不会被本轮
误删。维持登记不改,符合「与全仓库轮次命名一致」的既有裁定。
**Fix:** 无需改;若将来跨运行对比需要「每跑必成一轮」,再上亚秒后缀。

## 汇总表

| 编号 | 级别 | 文件:行号 | 一句话问题 |
|---|---|---|---|
| WR-01 | WARNING | build.py:476, diff.py:107 | burst 容差是二元列表:报告渲染裸列表、候选理由丢数字,测试锁了假形状 |
| WR-02 | WARNING | diff.py:99-117 | 连拍候选整组展开,无视生产端已点名的 suspect_image_ids |
| WR-03 | WARNING | cli/main.py:2402-2412 | report 命令坏输入吐全堆栈,漏做同阶段 T-06-07 退出码收口 |
| WR-04 | WARNING | tests/test_cli_run_all.py:24-43 | 真实 STEPS 八步顺序只有假表锁定,快速档无守卫 |
| IN-01 | INFO | cli/main.py:2407 | docstring「四件」应为五件(漏 suggest_exclude.json) |
| IN-02 | INFO | build.py:399 | summary 取法与 415 行不一致,null 时 AttributeError |
| IN-03 | INFO | diff.py:177, 206 | 注释行话「语义」两处,违通俗中文硬性约束 |
| IN-04 | INFO | diff.py:221-222 | higher 方向分支死代码 |
| IN-05 | INFO | diff.py:166-170 | 三路候选不按 image_id 去重,同张照片重复上榜 |
| IN-06 | INFO | build.py:193, 1014 | `_src` 绝对路径混进底账快照 |
| IN-07 | INFO | build.py:977 等 | 同秒覆盖为已登记观察项,确认未加重,维持不改 |

## 总判定

**需修复(仅 warning 级;0 critical / 4 warning / 7 info)。**
四条 warning 都是收口级小改(形状适配、一个 try/except、一条断言、一处点名来源
切换),不涉及已跑通链路的数字正确性,也不触碰 CRLB 口径铁律与「只读不重算」
纪律;修完即可收口。安全维度过关(HTML 转义全覆盖有测试锁定、无 eval/exec、
只走 json.load);契约维度八步顺序与 docs/pipeline.md 一致、report_data.json 同源
有真锁,唯 WR-04 的守卫缺口要补;通俗中文维度仅 IN-03 两处注释漏网。

---

_Reviewed: 2026-10-22T21:22:59Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
