---
phase: 03-fisher-crlb
plan: 02
subsystem: noise
tags: [fisher-information, crlb, schur-marginalization, sparse-matrix, monte-carlo, numdifftools]

# Dependency graph
requires:
  - phase: 02-m1-m4
    provides: models/ 解析投影与雅可比(唯一数学源)、solve/residuals.py 的 make_jacobian_fn/pack_theta(行序契约)、ObservationSet 观测集
provides:
  - marginalized_fisher:σ=1 基准 FIM 分块组装 + 外参 Schur 边际化(返回 F_marg / 稀疏 JᵀJ / 诊断字典)
  - weighted_fisher:J/σ 加权重算路线(σ 缩放恒等式的独立验证侧)
  - crlb_se:逐内参 CRLB 标准差(σ 严格线性,任意口径一次乘法)
  - SIGMA_BAND_REL_8 + sensitivity_band:Σ ±26.7% 敏感性带(纯乘法)
  - spectrum_report:特征谱/条件数/软方向载荷表/退化旗标(阈值可被 03-05 config 覆盖)
  - tests/test_noise_fisher.py:恒等式/结构/数值基准/蒙特卡洛四类锁定(13 用例)
affects: [03-04 交叉验证, 03-05 calib fisher 命令编排, 阶段四 CRLB 实证校验]

# Tech tracking
tech-stack:
  added: []   # 零新增依赖(numdifftools 为阶段二已有 dev 依赖,只进 tests/)
  patterns:
    - "FIM 雅可比只从 make_jacobian_fn 来 + 列切片分块不重排行列(D-10)"
    - "F_ββ 保持稀疏块对角 + np.linalg.inv 批量 6×6 求逆 + einsum Schur 消元"
    - "σ=1 基准:CRLB 与敏感性带全部退化为一次乘法"
    - "浮点恒等式测试的分母兜底体例(分母按矩阵量级 floor,阈值与 eps·‖F‖ 数学配平)"

key-files:
  created:
    - src/calib/noise/fisher.py
    - tests/test_noise_fisher.py
  modified: []

key-decisions:
  - "恒等式判定拆两层:F 层(Schur 分块 vs 整块消元)按计划线 1e-10(实测 2.4e-12);协方差层(inv vs inv)受 eps·cond 舍入物理极限,取相对矩阵量级 1e-9(实测 1.6e-10)——纯逐元素 1e-10 对横跨 12 个量级的协方差元是浮点里不可达的判定线"
  - "σ 缩放的'机器精度一致'锁在 F 层(重算 FIM vs σ=1 结果÷σ²,实测 ≤7.8e-14 < 1e-12);过 inv 的 SE 层取 1e-8(实测 ~2.9e-10)"
  - "合成场景不走 conftest make_true_views(渲染图像却不返回位姿),按 test_solve_solvers._synthetic_obs 体例投影直生——FIM 只吃几何,位姿与角点逐位一致,蒙特卡洛可从真值出发"
  - "谱报告措辞红线落进 docstring:大特征向量 = 参数空间的'软方向'(该组合方向上精度最差),不写'强耦合'"

patterns-established:
  - "奇异防护:位姿 6×6 块 cond ≥ 1e12 中文报错点名视图序号,绝不静默伪逆(退化是要上报的发现)"
  - "源码静态锁三件套(grep 进测试):无 'pinv'、无 'projectPoints'/'cv2'、必含 'make_jacobian_fn'/'pack_theta';src/calib 全树无 numdifftools"

requirements-completed: [NOI-03]  # NOI-03 的数学部分(合成数据上);真实数据编排由 03-05 接手

# Metrics
duration: 17min
completed: 2026-10-18
---

# Phase 3 Plan 2: Fisher/CRLB 数学核心 Summary

**σ=1 基准的 Fisher 信息矩阵分块组装与外参 Schur 边际化,配 CRLB 严格线性缩放、Σ ±27% 纯乘法敏感性带与特征谱诊断,全部数学被恒等式/结构/数值基准/蒙特卡洛四类测试锁定(Schur 恒等式实测 2.4e-12,蒙特卡洛 200 次 std/SE 比值 0.99–1.10)。**

## Performance

- **Duration:** 17 min
- **Started:** 2026-10-18T08:16:21Z
- **Completed:** 2026-10-18T08:33:03Z
- **Tasks:** 2/2(各按 TDD RED→GREEN 提交,共 4 个提交)
- **Files modified:** 2(均为新建)

## Accomplishments

- `src/calib/noise/fisher.py`(264 行):本模块只做矩阵代数——J 只从 `make_jacobian_fn` 来、θ 只走 `pack_theta`(D-10 红线,docstring 首段写明 + 源码静态锁)
  - `marginalized_fisher`:稀疏 J 列切片分块(绝不重排行序)→ F_θθ/F_θβ 稠密小块、F_ββ 保持稀疏块对角 → 批量 6×6 求逆 → einsum Schur 消元;返回 (F_marg, 稀疏 JᵀJ, 诊断字典);绝不整体 toarray(640 视图规模红线)
  - 奇异防护:任何视图外参 6×6 块条件数 ≥ 1e12 → 中文报错点名视图序号与白话解释,绝不静默伪逆糊过(位姿不可观测本身是要上报的退化发现)
  - `weighted_fisher` / `crlb_se` / `sensitivity_band` / `SIGMA_BAND_REL_8` / `spectrum_report`(谱、条件数、软方向载荷、warning/degenerate 旗标,阈值留函数参数给 03-05 的 config)
- `tests/test_noise_fisher.py`(395 行,13 用例,全部合成小场景):
  - 恒等式:Schur 分块消元 vs 完整 JᵀJ 直接消元(1e-10)与 inv(F_marg) vs inv(JᵀJ) 内参块(1e-9)
  - 结构:F_ββ 跨视图 6×6 块范数 < 1e-12(行序契约的结构锁)
  - 数值基准:numdifftools 数值雅可比 → 数值 FIM,与解析路线逐元素相对差 < 1e-6(实测 9.7e-10)
  - σ 缩放:三档(0.3/1.0/2.5)乘法逐位一致 + J/σ 重算 FIM 机器精度一致(≤7.8e-14)
  - 谱:手算 2×2 已知矩阵逐位对照(特征值升序/条件数/软方向 top-2 载荷名)
  - 蒙特卡洛:200 次加噪(σ=0.3)-scipy 求解,逐参数 std 与 FIM 预测 SE 比值 0.99–1.10,全部落在 [0.85, 1.18] 主判定线(未动用放宽档),用例总时长 ~21 秒
  - 静态锁:fisher.py 无 pinv/projectPoints/cv2、必含 make_jacobian_fn/pack_theta;src/calib 全树无 numdifftools
- 全套测试 273 通过(`uv run pytest -m "not integration" -q`),ruff 干净

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 修正研究骨架的 einsum 下标笔误**
- **Found during:** Task 1 GREEN
- **Issue:** 03-RESEARCH §Code Examples 的 Schur 骨架写 `np.einsum("ivi,vij,ivj->pq", ...)`,下标 "ivi" 把行标也记成 i,输出标签 p/q 未在输入出现——numpy 直接 ValueError
- **Fix:** 改为 `"pvi,vij,qvj->pq"`(p/q 为输出的行列,i/j 为单个 6×6 块内行列),恒等式测试背书
- **Files modified:** src/calib/noise/fisher.py
- **Commit:** 22ee9a7

**2. [Rule 1 - 数值判定线] 恒等式判定按浮点物理配平(计划线 1e-10 保留在 F 层)**
- **Found during:** Task 1 GREEN
- **Issue:** 协方差元量级横跨约 12 个数量级(k3 方差 ~6e4 对 fx 方差 ~1e-8 像素²),纯逐元素相对差 1e-10 要求 ~1e-22 绝对精度,超出双精度的物理极限(实测两条代数全同路线在小元上差 5.7e-8,而相对矩阵量级只差 2e-13——是分母太小,不是数学错);同理 σ 缩放的过-inv 路线受 eps·cond 放大(实测 ~2.9e-10)
- **Fix:** 拆两层——F 层(分块消元 vs 整块消元,分母按矩阵量级 1e-4 兜底)保留计划线 1e-10(实测 2.4e-12);协方差层取相对矩阵量级 1e-9(实测 1.6e-10)。σ 缩放的"机器精度一致"按计划 must_have 的原文锁在 F 层(1e-12,实测 ≤7.8e-14),过-inv 的 SE 层取 1e-8(实测 ~2.9e-10)。判定线与 eps·‖F‖/floor 的数学关系写进测试 docstring
- **Files modified:** tests/test_noise_fisher.py
- **Commit:** 22ee9a7、7a4f6e4

**3. [Rule 3 - 阻塞解除] 合成场景改按 _synthetic_obs 体例投影直生**
- **Found during:** Task 1 RED→GREEN
- **Issue:** 计划指定的 conftest `make_true_views` 渲染整幅棋盘图像却不返回位姿;FIM 恒等式测试需要与角点逐位一致的位姿,蒙特卡洛需要真值起点——复刻它内部随机序列的做法又脆(依赖 numpy 随机流跨版本不变)
- **Fix:** 场景 helper 按 `tests/test_solve_solvers.py::_synthetic_obs` 的现成体例直接按模型投影生成(M2、6 视图、11×8、z=700、±0.25 rad 侧转;条件数实测挑优,让双精度能展现恒等式);渲染夹具留给真正需要图像的角点测试
- **Files modified:** tests/test_noise_fisher.py
- **Commit:** 8828a15

**4. [Rule 2 - 锁定补全] 源码静态锁的文字面约束**
- **Found during:** Task 1/2 GREEN
- **Issue:** grep 锁按字面匹配,docstring 里出现 "pinv"/"numdifftools" 字样就会红(阶段二的 `test_numdifftools_never_in_src` 扫全 src/)
- **Fix:** fisher.py 注释一律用白话改写(如"伪逆""测试专用那套差分工具"),静态锁因此更硬:整个文件连注释都进不了这两个词
- **Files modified:** src/calib/noise/fisher.py
- **Commit:** 22ee9a7

## Known Stubs

None——全部函数为完整实现,无占位/空数据路径。

## TDD Gate Compliance

- Task 1:RED(8828a15,ModuleNotFoundError)→ GREEN(22ee9a7)✓
- Task 2:RED(fc5c261,ImportError)→ GREEN(7a4f6e4)✓
- 4 个提交的 test→feat 顺序经 git log 验证

## Verification Run

- `uv run pytest tests/test_noise_fisher.py -q` → 13 passed(~21 s,含 200 次蒙特卡洛)
- `uv run pytest -m "not integration" -q` → 273 passed, 5 deselected
- `uv run ruff check src/calib/noise/fisher.py tests/test_noise_fisher.py` → 干净
- 计划验收逐条:Schur 恒等式(1e-10,实测 2.4e-12)✓;块对角 <1e-12 ✓;病态视图中文报错点名视图序号 + 源码无 "pinv" ✓;数值 FIM <1e-6 逐元素(实测 9.7e-10)✓;σ 三档缩放(逐位 + F 层机器精度)✓;±27% 带与 1/√14 双锁定 ✓;2×2 手算谱逐位 ✓;蒙特卡洛比值 0.99–1.10 全落主判定线、总时长 <60 s ✓;src/calib 无 numdifftools ✓

## Boundary Notes(给下游计划)

- 本计划交付 NOI-03 的**数学部分**;真实数据的矩阵编排(640 全帧/80 姿态 × 双口径 × 4 模型)由 03-05 编排,OpenCV stdDev 交叉验证由 03-04 接手
- 蒙特卡洛用例是合成数据上的数学单测,不能替代阶段四的真实数据实证校验(测试 docstring 已注明)
- `weighted_fisher` 只为锁死 σ 缩放性质而存在,03-05 的主流程请走 `marginalized_fisher`(σ=1)+ `crlb_se` 乘法路线

## Self-Check: PASSED

- src/calib/noise/fisher.py 存在(264 行 ≥ 120,含 `def marginalized_fisher`)✓
- tests/test_noise_fisher.py 存在(395 行 ≥ 150,含 numdifftools)✓
- 提交 8828a15 / 22ee9a7 / fc5c261 / 7a4f6e4 均在 git log ✓
