---
phase: 05-error-attribution-corner-tuning
plan: 04
subsystem: analysis + visualization + cli + tests
tags: [test-xyz, sensitivity, pixel-equivalent, advice-rules, uniqueness-disclosure, what-if, ana-03, ana-04]

# Dependency graph
requires:
  - phase: 05-error-attribution-corner-tuning (05-02/05-03)
    provides: analyze 命令六块骨架 + _stage 逐块兜底体例 + analysis/*.npz 旁车契约(_ANALYZE_DETAIL_SECTIONS)+ _latest_analyze_or_skip 取数助手 + residual_structure 段(外推区间/最远观测 r 供建议分支引用)
  - phase: 04-evaluation-model-selection
    provides: evaluate.json(optimizer.multistart 解不唯一判定 + delivery.final_params + slices/bootstrap 实验谱)+ selection.extrapolation_interval
  - phase: 03-noise-fisher-crlb
    provides: noise.fisher.marginalized_fisher(what-if 谱合成的唯一数学源)+ fisher npz(eigvals_cov/eigvecs_cov/scale_dimless/F_marg_views640)
  - phase: 01~03
    provides: models 统一投影(project_point_jacobian 反解矩阵唯一来源)+ solve_pose_only 只估位姿 + corners 双 npz + board.yaml
provides:
  - "src/calib/analysis/test_xyz.py:frame_xyz 最小范数反解 dX=(AᵀA)⁻¹Aᵀ 残差(A 由 project_point_jacobian 堆叠,导数唯一来源)+ 可观测性三件套(z 分数 / 深度放大倍数 1/√λmin / 条件数)+ 病态帧(条件数 > analyze.xyz_cond_max)NaN 占行 + run_test_xyz 编排(失败帧占行不中断)"
  - "src/calib/analysis/sensitivity.py:run_sensitivity 谱对账(Fisher 特征谱 × bootstrap/slices 经验谱,逐方向余弦)+ 像素当量上天平(复用 evaluation 的 _px_dispersion,σ 级步长折像素)+ 建议规则表三分支(k2/k3+外推→补边缘姿态;fx/cy 耦合→补转角;p1/p2 近零→不治)+ uniqueness 解不唯一如实披露 + what-if 补拍预演(仅 full 档,合成假设显式声明)"
  - "analyze 块 3/4 接线:test_xyz 段与 sensitivity 段有值(_stage 兜底,budget 仍直调),旁车 test_xyz.npz / sensitivity.npz,三张证据图 test_xyz.png / sensitivity_ranking.png / what_if.png(仅 full 档),missing_sections 收缩到 {corner_opt, anisotropy}"
  - "真实数据对账 run_20261021_182345:test 65 张全反解零失败零病态(条件数中位 35 / 最大 215,深度放大中位 0.18 倍);dir1 载荷 k2 0.422 + k3 0.356(合计 78%,findings 结论 1 的软方向)+ dir2 fx/fy + dir3 cy;余弦 0.967(bootstrap)/ 0.752(slices);advice 三条分支全部命中;uniqueness bootstrap 最大位移 1.9943px 如实披露"
  - "24 例新测试(7 test_xyz + 11 sensitivity + 6 图),全套 563 passed + ruff 零告警;集成扩展 4 例(段形状/数值对账/旁车与图/块 3/4 走 _stage 源码断言)"
affects: [05-05(角点寻优接 corner_opt 段), 05-06(anisotropy 段收口 + 全链对账消费敏感度建议表), Phase 6(report 消费 test_xyz/sensitivity 段与三图)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "单一数学源纪律(D-06/D-10)延续:反解矩阵只从 model.project_point_jacobian 堆叠;what-if 谱合成只调 noise.fisher.marginalized_fisher(每合成位姿一次边际化,不重装 FIM)——两处都有源码级断言锁死(禁 import cv2 / 禁 csr_matrix / 禁第二份投影公式)"
    - "载荷口径:谱存在无量纲坐标(像素参数被 ÷焦距 压平),k2/k3 软方向只在无量纲载荷下可见(findings 结论 1);跨谱对账(bootstrap/slices)在原始坐标做,无量纲↔原始换算往返逐位一致(2 的幂刻度)"
    - "合成证据纪律(裁定②③):what-if 产物与图题都带「合成预演,假设非实测」声明;三条建议全部以「只提示方向,不自动执行任何补拍/重标定」收尾"

key-files:
  created:
    - src/calib/analysis/test_xyz.py
    - src/calib/analysis/sensitivity.py
    - tests/test_analysis_test_xyz.py
    - tests/test_analysis_sensitivity.py
  modified:
    - src/calib/cli/main.py
    - src/calib/visualization/analyze_figs.py
    - tests/test_analysis_realdata.py
    - tests/test_visualization.py

key-decisions:
  - "载荷表取无量纲坐标分量:plan 公式口径落在无量纲谱上,原始坐标载荷会被焦距刻度(1855.77)反抢主导、k2/k3 软方向不可见——实现初版用原始坐标,真实数据首跑 dir1 呈现 fx/fy 与 findings 结论 1 不符,对照计划判据后纠正为无量纲口径并在段内 caliber_note 就地声明"
  - "像素当量的步长实现复用 evaluation.experiments._px_dispersion(σ 级扰动 × 特征向量折像素位移),不写第二份折算;经验谱当量与理论谱当量同一把尺"
  - "建议规则表三分支全部数据驱动现算:k2/k3 分支要求载荷占比 ≥0.5 且选型登记了外推区间(coverage 缺席时显式输出「覆盖判据未知」不冒充);fx/cy 耦合分支要求 ≥2 个成员载荷 ≥0.15;p1/p2 分支按 0.05px 影响线判「不治」,每条建议固定「不自动执行」收尾(裁定③)"
  - "what-if 仅 full 档(裁定②):quick 档 what_if=None + 中文 note;full 档合成位姿网格(距离 400/550/700mm × 转角 ±40°/±20° × 平移)按 r_max 够到外推区筛选,每张合成位姿的观测集 Fisher 边际化后并入,压缩比 = √(diag(F₀⁻¹新)/diag(F₀⁻¹旧)),声明必含「合成,非实拍」"
  - "病态帧 NaN 占行不冒充解释:条件数 > analyze.xyz_cond_max(1e4)记 NaN 行 + n_cond_excluded 计数,不淘汰不插值;乐观偏差(test 集参加过选型)与深度判读(z 分量大 ≠ 误差在深度)两则中文声明随段落账"
  - "test_xyz 的「真值」是位姿只估拟合的回代残差:内参固定为交付解、逐帧只解位姿,故 dX 绝对量级(~1e-10 mm)反映的是「内参不再动时残差可被位姿吸收的程度」,三分量相对结构与 z 分数判读才是本段产品——不冒充「重标定会改动多少」"

requirements-completed: [ANA-03, ANA-04]

# Metrics
duration: 6h38min(含 4h21m 全量集成套件与注入复跑;纯实现约 1h10min)
completed: 2026-10-22
---

# Phase 5 Plan 4: ANA-03 test 三分量反解 + ANA-04 参数敏感度 Summary

**test 集 65 张零失败零病态的最小范数三维反解(条件数中位 35,深度放大 0.18 倍,病态占行机制备而未触发)+ 敏感度双谱对账像素当量上天平:dir1 = k2/k3 载荷 78%(findings 结论 1 软方向,只在无量纲口径可见)、余弦 0.967/0.752、三条应对建议(k2/k3 补边缘姿态 / fx,fy 补转角 / p1,p2 不治)全部分支命中、解不唯一(bootstrap 最大位移 1.9943px)如实披露、what-if 仅 full 档且合成声明到位;24 例新测试全套 563 passed**

## Performance

- **Duration:** 6h38min(墙钟;含 4h21m 全量集成套件 + 注入测试复跑取证)
- **Started:** 2026-10-21T17:47:39Z
- **Completed:** 2026-10-22
- **Tasks:** 3
- **Files:** 4 created + 4 modified

## Accomplishments

- **test_xyz 库层(ANA-03,本阶段唯一新建数学):** 逐帧最小范数反解 dX = (AᵀA)⁻¹Aᵀ·残差,反解矩阵 A 由 `project_point_jacobian` 按 (N,2,3)→(2N,3) 堆叠(导数唯一来源);可观测性并排落盘(z 分数 = dX/σ_理论、深度放大倍数 = 1/√λmin、条件数);病态帧(AᵀA 条件数 > 1e4)记 NaN 占行 + 分类计数,不冒充解释;run_test_xyz 编排失败帧占行不中断
- **sensitivity 库层(ANA-04):** Fisher 特征谱 × bootstrap/slices 经验谱双对账(逐方向余弦);全部换算像素当量上天平(复用 `_px_dispersion`,p1/p2 百分比陷阱不出现);建议规则表三分支(补边缘姿态/补转角/不治)数据驱动现算;uniqueness 段如实披露 bootstrap 最大位移当量与多起点判定,不给「参数精确到 X」式承诺;what-if 补拍预演仅 full 档、合成假设显式声明
- **analyze 接块 3/4:** 两段按 _stage 逐块兜底接入(budget 仍直调,铁律违例照样退出码 1);npz 旁车双份挂载;三张证据图(test_xyz 三分量双条形 + 深度放大标注 / sensitivity_ranking 排序 + 像素当量天平 + 近零「画面影响可忽略」标注 / what_if 压缩比条形,quick 档缺席);好坏判据播报追加「Z 分量判读看 z 分数列;敏感度结论以像素当量为准;建议不自动执行」
- **真实数据首跑 run_20261021_182345:** test 65 张全反解零失败零病态(条件数中位 35.36 / 最大 215.33,深度放大中位 0.18 倍);dX 绝对量级 ~1e-10 mm(位姿只估拟合把残差吸收,见 key-decisions 口径声明);逐帧 RMS 中位 0.119px / 最大 0.537px;敏感度 dir1 载荷 k2 0.422 + k3 0.356 = 78% / dir2 fx,fy / dir3 cy 0.693;余弦 0.967(bootstrap)/ 0.752(slices);σ 级像素当量 dir1 0.095px / dir2 0.606px / p1 0.0076px、p2 0.0086px;advice 三条分支全部命中且全部「不自动执行」收尾;uniqueness 1.9943px / 0.0740px 与 evaluate 逐位一致;missing_sections 收缩到 {corner_opt, anisotropy}
- **测试:** 库层 18 例(test_xyz 7:sqrt 对角手算/反解恒等式/真值场景 z 分数符号/病态占行/单一数学源红线;sensitivity 11:对角手算当量/无量纲↔原始往返逐位一致/uniqueness 透传/三分支建议/quick what_if 空/full 合成声明/F 合并恒等式 √(4/5)/源码红线)+ 图 6 例(出图/无数据降级/合成声明上标题/源码断言)+ 集成扩展 4 例(段形状与中文 note/数值对账含余弦区间/旁车 npz 与三图存在性/块 3/4 走 _stage 而 budget 直调的源码断言)

## Task Commits

TDD 流程(Task 1 先行 RED):

1. **Task 1 RED: test_xyz 失败测试先行** - `05ad259` (test)
2. **Task 1 GREEN: test_xyz 库层实现** - `4592d97` (feat)
3. **Task 2: sensitivity 库层** - `21b8b13` (feat)
4. **Task 3: analyze 接块 3/4 + 三图 + 集成扩展** - `b43f9d2` (feat)

**Plan metadata:** 见本计划收尾 docs 提交(SUMMARY + STATE + ROADMAP + REQUIREMENTS)

## Files Created/Modified

- `src/calib/analysis/test_xyz.py` - ANA-03 全部数学与编排(反解 + 可观测性 + 病态占行;零 OpenCV,导数只出自 project_point_jacobian)
- `src/calib/analysis/sensitivity.py` - ANA-04 全部编排(谱对账 + 像素当量 + 建议规则表 + uniqueness + full 档 what-if;只消费 noise.fisher.marginalized_fisher,不重装 FIM)
- `tests/test_analysis_test_xyz.py` - +7 例库层行为测试(TDD 先行,含源码红线)
- `tests/test_analysis_sensitivity.py` - +11 例(合成 npz 底账 + 三分支建议 + 合成声明 + F 合并恒等式 + 源码红线)
- `src/calib/visualization/analyze_figs.py` - +plot_xyz_compare / plot_sensitivity_ranking / plot_what_if(三图,what-if 图题自带合成声明)
- `src/calib/cli/main.py` - analyze 块 3/4 _stage 接线 + 双旁车挂载 + 三图(what-if 条件挂图)+ 播报扩展
- `tests/test_visualization.py` - +6 例三图测试(合成段夹具 + 无数据降级 + 源码断言)
- `tests/test_analysis_realdata.py` - 段形状断言从 None 改有值 + 3 例新集成(数值对账/旁车与图/源码断言)+ 1 例改名

## Decisions Made

见前置 key-decisions 六条。补充执行层两条:

- 图与数据分离体例延续:三张图只画段内现成键,数字结论以 analyze.json 为准,图失败不拦主产物;what-if 图在 quick 档直接不生成(而非画占位图),与段级 what_if=None + note 同口径
- npz 旁车经 `_ANALYZE_DETAIL_SECTIONS` 契约表挂载(与 05-02/05-03 同体例),明细键拍平规则零新代码

## Deviations from Plan

**1. [Rule 1 - Bug] `_read_spectrum_npz` 占位读法移除**
- **Found during:** Task 2
- **Issue:** 初版保留实现占位的假读法,缺键时报错不含 model_id,定位成本高
- **Fix:** 删占位逻辑,缺键报错并入 model_id 上下文
- **Files:** src/calib/analysis/sensitivity.py;Commit: 21b8b13

**2. [Rule 1 - Bug] loads_table 推导式 NameError(`n` 未定义)**
- **Found during:** Task 2 真实数据冒烟
- **Issue:** `"param": n` 应为 `names[i]`,真实数据一跑即 NameError
- **Fix:** 改为索引取名
- **Files:** src/calib/analysis/sensitivity.py;Commit: 21b8b13

**3. [Rule 1 - Bug] optimizer.multistart 取数双重 `_require`**
- **Found during:** Task 3 集成冒烟
- **Issue:** 同一键取了两次,第二次必抛「缺键 multistart」假错(先核对真实结构 optimizer.multistart.models.m2 后修正)
- **Fix:** 单次 `_require` 取 multistart
- **Files:** src/calib/analysis/sensitivity.py;Commit: b43f9d2

**4. [Rule 1 - Bug] what_if.png 在 quick 档被误画**
- **Found during:** Task 3 真实数据首跑
- **Issue:** 挂图无条件执行,与验收「what-if 图 quick 档缺席」冲突(段级 what_if=None 但图仍出「无数据」版)
- **Fix:** 改为 `sensitivity.get("what_if") is not None` 才挂图;缺陷轮 182236 删除后重跑
- **Files:** src/calib/cli/main.py;Commit: b43f9d2

**5. [Rule 1 - Bug] 深度放大倍数播报格式(.0f 把 0.18 印成「0 倍」)**
- **Found during:** Task 3 真实数据首跑
- **Issue:** 统一 `:.0f` 对 <1 的放大倍数失真
- **Fix:** ≥10 用整数、<10 保留两位的条件格式(播报与测试打印同步)
- **Files:** src/calib/cli/main.py;Commit: b43f9d2

**6. [Rule 3 - 阻塞修复] 载荷口径纠正为无量纲坐标**
- **Found during:** Task 3 真实数据对账
- **Issue:** 实现初版在原始坐标取载荷,真实数据 dir1 呈现 fx/fy,与计划/findings 结论 1 预期的「dir1 = k2/k3 78%」不符;根因是像素参数载荷被焦距刻度(1855.77)反抢主导,软方向只在无量纲坐标可见——plan 公式的口径本就落在无量纲谱上
- **Fix:** 载荷表改取无量纲分量,caliber_note 就地声明口径与缘由;纠正后真实数据 dir1 = k2/k3 0.422+0.356(78%)与 findings 结论 1 逐位吻合
- **Files:** src/calib/analysis/sensitivity.py;Commit: b43f9d2(随 Task 3 一并落库,缺陷轮已删重跑)

**7. [Rule 1 - 风格] ruff 三处微修(PIE804 / RUF015 / F541)**
- **Found during:** Task 2
- **Fix:** np.savez 直接关键字参数 / `next(iter(...))` 替代单元素切片 / 去无效 f 前缀
- **Files:** tests/test_analysis_sensitivity.py, src/calib/analysis/sensitivity.py;Commit: 21b8b13

## Known Stubs

- `corner_opt` 段:计划内显式 None 留位(05-05 COR-02 角点寻优接入),meta.missing_sections 如实登记
- `anisotropy` 段:计划内显式 None 留位(05-06 COR-03 接入)
- what-if 在 quick 档(当前默认)为 None + 中文 note:裁定②的设计行为,非缺陷;full 档路径有测试覆盖(full 合成声明 + F 合并恒等式)

## Issues Encountered

**全量集成套件 46 passed / 3 failed——三个失败全部为既有问题,与 05-04 四个提交零接触**(噪声/角点/注入/CRLB 代码本计划零接触;详见 `deferred-items.md`),已按超范围纪律只登记不修:

1. `test_noise_realdata` 与 `test_fisher_realdata` 各 1 例:复用旧轮血缘登记的上游 `corners/run_20261018_163039` 已被「及时清理」删除(被 102709 替代),血缘链第一环断在产物态,修复需重跑 noise/fisher 两条小时级流水线产生新轮
2. `test_noise_injection::test_full_injection_real_data`:阶段三老测试与演进后的数据契约脱节——`noise_splits: [train, test]`(03-01 起即双划分)× 阶段四引入 hold-out 后 corners_train_sb.npz 收窄为 640 张,测试仍按 640 断言形状(实际 705,多出的 65 张 test 图无基线角点成 NaN 行);另发现 σ=0 基线在 2 个角点上有 0.33/−0.25px 真实非零(疑似检测来源未对齐),属注入子系统既有异常,一并登记

**暴露时机说明:** 05-03 收官 SUMMARY 中「46 integration 零回归」实为子集运行外推(会话记录最大子集为 11 passed/37 deselected);全量 `-m integration` 套件(4h21m)在阶段五由本计划首次完整跑通,上述既有失败因此首次显形。

**当轮产物清理(及时清理):** evaluate 副产品轮 run_20261021_183217(与权威轮 131613 逐位一致)已删;optimizer 副产品轮 run_20261021_224410(仅 optimizer.json,多起点 m2 param_disp_px = 0.07399264201266795 与权威轮 evaluate.json 内嵌值逐位一致——确定性再次验证)已删;权威轮 131613 与 analyze 轮 182345 保留(后者指纹前者)。

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 05-05 接线条件齐备:corner_opt 段留位 + `_stage` 体例 + quick/full 两档 + 检测器唯一入口(corners 模块);集成测试 `_latest_analyze_or_skip` 将自动消费最新 analyze 轮
- 敏感度建议表(三分支 + 不自动执行收尾)已定型,05-06 收口报告与人工核查可直接引用;真实数据敏感度结论:补 6 个够到画面四角的姿态是当前性价比最高的动作(k2/k3 方向 0.095px 当量 × 纯外推区零观测)
- 待办提醒:血缘断链与注入老测试两项超范围修复已在 deferred-items.md 排队,建议在阶段五收口(05-06)或阶段六开篇安排专门一轮,避免带病进入最终验证

## Self-Check: PASSED

- 8 个关键产物文件全部在盘(4 created + 4 modified)
- 4 个任务提交全部在案:05ad259 / 4592d97 / 21b8b13 / b43f9d2
- 快速环 563 passed(49 deselected,含 24 例新测试)+ ruff check src tests 零告警(最终代码态复跑验证);集成子集 test_analysis_realdata 12 passed(4 新)
- 真实数据首跑 run_20261021_182345:test_xyz 与 sensitivity 段有值、五图四旁车在盘、what_if.png 按 quick 档正确缺席、missing_sections = {corner_opt, anisotropy};findings 结论 1(k2/k3 软方向 78%)与结论 5(外推区间)逐位吻合
- 全量集成 46 passed / 3 failed:3 个失败全部定性为既有问题并登记 deferred-items.md(与 05-04 零接触,证据链见该文件)

---
*Phase: 05-error-attribution-corner-tuning*
*Completed: 2026-10-22*
