---
phase: 05-error-attribution-corner-tuning
plan: 06
subsystem: analysis(误差归因第六块 + 阶段五收口)
tags: [COR-03, anisotropy, analyze, closure, checkpoint-self-checked]
requires:
  - output/noise/run_20261018_165412/temporal_sigma_per_corner.npz(逐姿态逐角点重复性,全 temporal 来源)
  - src/calib/analysis/residual_structure.py(assign_bands / radius_footprint 分带器复用)
  - src/calib/analysis/corner_opt.py(corner_opt 段,互证行引用)
  - 05-05 的 analyze 块 5 骨架(块 6 留位)
provides:
  - src/calib/analysis/anisotropy.py(run_anisotropy:temporal 硬过滤 + clamp 防零除 + 分带/象限切片 + 白话判读 + 互证行)
  - calib analyze 六块全链(missing_sections == [])
  - plot_anisotropy_map / plot_anisotropy_hist(两图,无数据分支)
  - analyze run_20261022_025045(六段全有值真实数据轮,替代并删除 010551 旧轮)
  - docs/findings.md 结论 6–11(全部带血缘数字)
affects:
  - Phase 6 报告(REP-01~03 消费 analyze.json 六段)
  - 角点方法切换决策(classic_w7 证据链,人工拍板)
tech-stack:
  added: []
  patterns:
    - _stage 逐块兜底体例延伸到块 6(纯消费块单块失败记 None 不中断)
    - 分带器一处实现两处用(residual_structure.assign_bands 源码断言锁定)
    - 明细数组走 _anisotropy_detail 私有键 → analysis/anisotropy.npz + 血缘 meta.json(artifacts 层既有机制)
key-files:
  created:
    - src/calib/analysis/anisotropy.py
    - tests/test_analysis_anisotropy.py
  modified:
    - src/calib/cli/main.py
    - src/calib/visualization/analyze_figs.py
    - tests/test_analysis_realdata.py
    - tests/test_analysis_cli.py
    - tests/test_visualization.py
    - docs/pipeline.md
    - docs/findings.md
    - .planning/REQUIREMENTS.md
decisions:
  - run_anisotropy 增加可选参 sigma_detector_floor 与 corner_opt_section(计划 action 文本明示两条入口):floor 由 CLI 层从 noise.json 读入传入,缺席退回 npz 自带 sigma_detector 列中位并落账来源
  - aniso = σu / max(σv, floor),NaN 行输出 NaN 且 valid_mask 占位;显著判读线 1.5 随段落账
  - 段内加键 per_corner_median_aniso(88 点,None=无效)/ aniso_hist(全域 40 桶不截尾)/ image_size,让两张图只消费 analyze.json 段键,不碰明细数组
  - CLI 块 6 返回三元组(section, detail, uv_train),uv_train 供地图的观测足迹底
  - 检查点按用户 2026-10-21 隔夜授权自检(见下文),四要素指引全文留档供晨间复核
metrics:
  duration: 约 90 分钟(含 45 分钟真实数据全链轮)
  completed: 2026-10-22
---

# Phase 5 Plan 6: COR-03 连拍各向异性端到端 + 阶段五收口 Summary

各向异性第六块接入后 `calib analyze` 六块全链在真实数据上一次跑通(analyze run_20261022_025045,六段全有值、零缺席),COR-03 结论落账:纵向重复散布系统性地大于横向(σu/σv 中位 0.68,量级仅检测器噪声的 2–4 倍,不支持换角点方法);流水线契约⑦行、findings 结论 6–11、REQUIREMENTS 六行 Shipped 全部同步,检查点按隔夜授权自检通过。

## 任务完成情况

| 任务 | 内容 | 提交 |
|---|---|---|
| Task 1 | anisotropy.py 库层:temporal 硬过滤(排除计数)/ clamp 防零除 / 分带象限切片 / 白话判读 / 互证行 + 11 测试 | 93b36cd |
| Task 2 | CLI 接块 6(_stage 兜底)+ 两图 + 块 6 兜底注入测试 + realdata 断言翻转 + 真实数据全链轮 | 22f49ac |
| Task 2.5(计划外,Rule 1) | 角点寻优双口径曲线图口径别名错位修复(检查点核查抓出)+ 2 回归测试 + 图就地重渲染 | 8517767 |
| Task 3(检查点,授权自检) | pipeline.md ⑦行 / findings 结论 6–11 / REQUIREMENTS 六行 Shipped + 四要素逐项核查 | 见 docs 提交 |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 角点寻优双口径曲线图永远画"无数据"(05-05 遗留,检查点核查抓出)**
- **Found during:** Task 3 检查点人工核查(好坏判据⑤:逐张看图)
- **Issue:** `plot_corner_opt_curves` 用 `section["stats"].get(reference)` 直接查键,而 main.py 传短名 `reference="self"/"prod_sb"`,段内 stats 键是 `_self`/`_prod_sb`——查不到键永远走"无数据"分支。05-05 两轮真实数据跑出的 4 张曲线图(新旧轮各 2 张,文件大小同为 25618 字节)全是占位图;05-05 的测试只断言 PNG 存在、没断言有曲线,故漏网。
- **Fix:** 段内加别名映射 `{"self": "_self", "prod_sb": "_prod_sb"}`;补 2 条回归测试(短名渲染必须显著大于无数据占位图;缺该口径仍走无数据分支);用已落盘 analyze.json 的 corner_opt 段就地重渲染当前轮两张图(json 是真相源,图只是可视化,血缘不变)。
- **Files modified:** src/calib/visualization/analyze_figs.py, tests/test_visualization.py;重渲染产物 output/analyze/run_20261022_025045/figures/corner_opt_{self,prodsb}.png
- **Commit:** 8517767

### 计划内留档

- tests/test_analysis_anisotropy.py 的出图函数测试原计划"Task 2 一并落地后补",执行时确实在 Task 2 提交内补齐(Task 1 提交只含库层测试,两个提交各自可独立跑绿)。
- 集成测试 3 个既有失败(noise/fisher 血缘断链 ×2 + injection 老测试 ×1)按 deferred-items 登记处置,不夹带修复,详见 .planning/phases/05-error-attribution-corner-tuning/deferred-items.md。

## 验证(全绿)

- 非集成全量:`.venv/bin/python -m pytest tests -q -m "not integration"` → **594 passed, 51 deselected**(修复后复跑)
- 集成:`tests/test_analysis_realdata.py -m integration` → **14 passed**;`test_analysis_realdata + test_analysis_anisotropy` → 26 passed
- 真实数据全链轮:`uv run calib analyze` 退出码 0,`meta.missing_sections == []`,六段全非 None,9 张 PNG + 6 份明细 npz(各带血缘 meta.json)落盘
- ruff:`src tests` 零告警
- 及时清理:旧轮 output/analyze/run_20261022_010551 已删(新轮集成绿后执行)

## 检查点自检(用户 2026-10-21 隔夜授权:"这轮不人工查了,你认真检查没问题就往后走")

本计划的 `checkpoint:human-verify` 由执行方按授权逐项自检,四要素指引留档如下,供晨间复核。

### 人工核查四要素指引

1. **看哪个文件**:`output/analyze/run_20261022_025045/analyze.json`(主契约);同轮 `figures/` 下 9 张 PNG;`analysis/` 下 6 份 npz + .meta.json(血缘旁车,抽 anisotropy.meta.json 看上游 12 项指纹)。
2. **文件作用**:analyze.json 是六块结论主契约(数字以它为准);figures/ 是证据图(可视化);analysis/*.npz 是逐行底账(json 只留汇总索引)。
3. **具体看哪里**:budget.verdict 与 rows(逐参数实测/下界/利用率);residual_structure.extrapolation_note 与 coverage(r_max_obs=0.685、四角 0.797、外推区披露);test_xyz.per_direction 与 notes(Z 判读看 z 分数列);sensitivity.advice 与 uniqueness(3 条建议"不自动执行";解不唯一 1.9943px 如实在账);corner_opt.stats 双口径与 verdict_note(classic_w7 0.0143px vs 生产 sb 0.0507px);anisotropy.summary 与 plain_language(中位 0.68,发丝 1/4 参照)。
4. **好坏判据**:见下表逐条对账。

### 成功判据 1–5 对账(逐条)

| # | 判据 | 结果 | 证据 |
|---|---|---|---|
| 1 | 预算段零违例且三统计量远高于硬线 | **PASS** | n_pointwise_violations=0,最小倍数 2.76;101.8/1107.6/1443.8 倍 vs 硬线 0.4207px;recheck.consistent=true |
| 2 | 每张空间图肉眼可见覆盖足迹与外推阴影,结论行无软件行话 | **PASS** | coverage_footprint(足迹+r 等值线+外推阴影)、radial_bands(外推区粉色阴影+中文标注)、anisotropy_map(足迹浅色底+象限分界)逐一目检;判读文案全白话 |
| 3 | XYZ 结论同时给原始量级与 z 分数 | **PASS** | test_xyz 图双面板(毫米原始 + z 分数),json per_direction 两键并存,notes 固定声明在盘 |
| 4 | 建议针对最大敏感维度且不给"参数精确到 X"承诺 | **PASS** | advice 第 1 条=k2/k3(载荷 78%);uniqueness.note 明写"不存在参数精确到 X 的承诺";bootstrap 最大位移 1.9943px 在账 |
| 5 | 退化曲线校准档零位移、随退化单调变差;生产配置结论由数据出;各向异性有物理量级白话 | **PASS**(修复后) | 修正后曲线图 noise 面板全部方法在 σ=0 处汇于 0,sb 三族阶梯单调;classic_w7 证据由数据点名、只提示方向;anisotropy.plain_language 给"横向散布 ≈ 0.011mm ≈ 发丝直径 1/4" |

**结论:检查点自检通过(5/5),无阻塞性疑义。**

### 已知展示特征(非缺陷,留档)

- anisotropy_map.png 的色标上限被最大值 81.4(近零 σv 被 clamp 接住的有限值)拉高,多数角点集中在色标暗端;数字结论以 json 的 by_band/by_quadrant 为准(挤在 0.64–0.68 窄带)。若晨间复核觉得图面对比度不足,可在 Phase 6 报告计划里把 vmax 改为 P90 截断——属展示层微调,不夹带在本计划。

## Known Stubs

无。所有分析段、图、旁车均为真实数据产物;无占位数据、无 TODO 残留。

## 自查记录(Self-Check)

- [x] src/calib/analysis/anisotropy.py 在盘(93b36cd)
- [x] tests/test_analysis_anisotropy.py 12 例全绿
- [x] 四个任务提交在案:93b36cd / 22f49ac / 8517767 / docs 提交(见 STATE 更新提交)
- [x] analyze run_20261022_025045 六段全有值、missing_sections == []
- [x] grep 自检:pipeline.md 含 "calib analyze"/"analyze.json";REQUIREMENTS "Shipped (Phase 5)" 计 6;findings "血缘" 计 12

## Self-Check: PASSED

全部产物文件在盘、全部提交在案(2026-10-22 执行方核验)。
