---
phase: 02-m1-m4
plan: "04"
subsystem: calibration-pipeline
tags: [scipy, sparse-jacobian, least-squares, robust-loss, tdd, d08-single-source, d09-zero-intrusion]

# Dependency graph
requires:
  - phase: 02-m1-m4(02-01)
    provides: CameraModel 抽象基类 + M1–M4 注册机制 + transform_to_camera 外参链式后半截
  - phase: 02-m1-m4(02-02)
    provides: ObservationSet 冻结数据类(观测集顺序契约)
  - phase: 02-m1-m4(02-03)
    provides: CalibrationResult + compute_result_metrics 同一把尺 + solve_with_opencv(含 _solve_generic_pnp 兜底)+ ScipyConfig 配置段
provides:
  - solve/residuals.py:θ 装箱(pack/unpack)、残差工厂、解析稀疏雅可比工厂(COO 向量化 → csr)、0/1 稀疏结构矩阵;残差与雅可比共用唯一行号函数 row_index(行序契约的结构性防线)
  - solve/scipy_solver.py:solve_with_scipy(trf + lsmr + 解析稀疏雅可比 + jac_sparsity + x_scale='jac' + 可配鲁棒核)、is_practically_converged 实用收敛纯函数、_auto_f_scale(3×1.4826×MAD,近零兜底 1px)、代价尾部改进(运行最小值包络)
  - 模型层新接口 project_point_jacobian((N,2,3) 像素对相机点导数,KB4 含近轴解析极限):解析链式前半截进模型层,D-06 唯一数学源补全
  - transform_to_camera 的 ∂Xc/∂ω 修复:改用 cv2.Rodrigues 同次调用返回的雅可比,任意转角对中心差分吻合 ~1e-10
affects: [02-m1-m4(02-05 试车矩阵的 scipy 侧直接消费), 03-评估(Fisher 信息直接复用解析雅可比), 04-诊断(敏感度分析复用同一雅可比)]

# Tech tracking
tech-stack:
  added: []   # 无新依赖;scipy.optimize.least_squares / scipy.sparse 均为既有栈
  patterns:
    - 行序契约双防线:残差与雅可比共用 row_index(结构),numdifftools 逐列 + scipy 2-point 前向差分 60 随机位置(数值)——行序错位无报错但梯度错,必须双向锁
    - 前向差分基准容差定 1e-4:截断误差 ~(h/2)f'' ≈ 2e-6(h=√机器精度),行序错位偏差 O(1),相差 4 个量级,线画中间两边都不误伤
    - 代价尾部走平判定用运行最小值包络:trf 尝试步被信赖域拒绝时代价回弹,原始曲线锯齿不反映"还能不能改进"
    - 数值路径异常收窄到 (ValueError, ArithmeticError, RuntimeError) 进失败契约;代码契约错误(AssertionError/TypeError)大声抛出不被吞
    - 注册表测试卫生:模块级夹具收尾摘除玩具模型 + 文件末测试显式验证注册表复原(幂等双保险)

key-files:
  created:
    - src/calib/solve/residuals.py
    - src/calib/solve/scipy_solver.py
    - tests/test_solve_scipy_residuals.py
    - tests/test_models_registry.py
  modified:
    - src/calib/models/base.py
    - src/calib/models/pinhole_brown.py
    - src/calib/models/kb4.py
    - tests/test_solve_solvers.py

key-decisions:
  - "模型层补 project_point_jacobian 抽象方法(Rule 2):计划的解析外参链式需要 d像素/d相机点 这前半截,02-01 的接口里没有;不放模型层则求解侧只剩数值差分(研究 Don't Hand-Roll 禁止,且破坏 D-06)。三个实现:针孔族与 unproject 的 2×2 牛顿雅可比同式(正逆解导数同源),KB4 推导 s=θd/r 的全链 + 近轴极限(∂u/∂X=fx/z、∂u/∂Z=−fx·x/z²),玩具模型教科书写照"
  - "transform_to_camera 的 ∂Xc/∂ω 改用 cv2.Rodrigues 返回的雅可比(Rule 1):旧式 −[Xc]× 只是 ω=0 处的极限形式,0.1 rad 小转角中心差分相对误差已 0.7–0.9;cv2 布局 jac[k,3i+j]=∂R_ij/∂ω_k,任意转角吻合 ~1e-10,与 R 本身同一份数学;只服务于错误公式的 cross_product_matrix 一并移除"
  - "oracle 二(2-point 前向差分)容差 1e-4·max(1,|a|,|n|):数值锁精度的活交给 oracle 一(numdifftools 逐列 <1e-6),oracle 二专门抓行序错位——两类误差差 4 个量级,职责分开"
  - "实用收敛判定 basis 全量进 solver_meta(status/nfev/njev/final_cost/tail_improvement/n_cost_evals/rms_linear_ref):Pitfall 3 的'不静默'落在可审计的数据上,排障不用猜"
  - "给 scipy 精化的初值必须已收敛且模型一致:未收敛/异模型初值在 _initial_theta 中文报错,不静默换路——对着别的模型的结果继续算只会产出垃圾"
  - "SOL-01 至此双后端齐(OpenCV 初始化 + scipy 精化 + 鲁棒核 + 实用收敛),需求勾选留给 orchestrator 合并波次后统一处理(并行执行器不动共享规划产物)"

patterns-established:
  - "新增导数一律先写双数值基准测试再写公式(RED 先行),解析式对 numdifftools 逐列 <1e-6 是硬线"
  - "测试里的替身(monkeypatch)要在测试末显式 undo 并复核复原,替身泄漏会污染同会话后续测试"

requirements-completed: []   # SOL-01/MOD-01 的勾选由 orchestrator 在波次合并后统一执行(本执行器不改共享规划产物)

# Metrics
duration: 25min
completed: 2026-10-18
---

# Phase 2 Plan 04: scipy 精化后端 Summary

**残差与解析稀疏雅可比组装(行序契约双防线:共用行号函数 + 双数值基准)+ trf/lsmr 稀疏精化(实用收敛判定全量留痕、鲁棒核 f_scale 自动档)+ 模型层 project_point_jacobian 接口补全与 ∂Xc/∂ω 一般转角修复(Rule 1/2)+ D-08 残差单一来源行为锁与 D-09 玩具模型双后端零侵入——30 个新用例,全量快速回归 246 passed**

## Performance

- **Duration:** 约 25 分钟(2026-10-18T01:40:23Z → 02:05Z,worktree 并行波次)
- **Started:** 2026-10-18T01:40:23Z
- **Completed:** 2026-10-18T02:05:00Z
- **Tasks:** 3/3
- **Files modified:** 8(新建 4,修改 4)

## Accomplishments

- **残差与雅可比(Task 1, TDD):** `solve/residuals.py` —— θ 装箱([内参 P | 各视图 (ω,t)×6],数组或按参数名字典两种入参)、残差工厂(逐视图全 numpy 向量化,8 视图单次雅可比毫秒级)、解析稀疏雅可比工厂(内参列 project_jacobian 直接 reshape、外参列 einsum 链式、COO 三元组拼接 → csr)、0/1 结构矩阵;每残差行恰 P+6 个结构非零,640 视图稠密等价 3.47 GB 的硬前提从第一行代码就走稀疏
- **双数值基准锁定:** 随机真值 θ 上 ① numdifftools(理查森外推)逐列相对误差 <1e-6(M1–M4 全模型);② scipy 缺省 2-point 前向差分在 60 个随机行列位置逐项一致(容差 1e-4,截断 ~2e-6 与错位 O(1) 差 4 个量级)——行序错位(无报错但梯度错、收敛不动,Pitfall 6)在基准二现形
- **scipy 精化后端(Task 2, TDD):** `solve_with_scipy` = trf + lsmr + 解析稀疏雅可比 + jac_sparsity + x_scale='jac' + 可配 loss;OpenCV 初值按参数名逐名取(init_from=模型__来源),无初值走通用初始化兜底(init_from=internal_generic);实用收敛 = status>0 直接收 / status=0 需代价尾部(运行最小值包络后 10 次评估)相对改进 <1e-6 且 RMS 不比 linear 参照差超 5% / status=-1 绝不收;判定依据全量进 solver_meta;数值路径异常收窄进 make_failure 中文契约,失败结构与 OpenCV 后端逐键同构
- **接口补全与旧错修复(见 Deviations):** 模型层新增 project_point_jacobian((N,2,3),KB4 近轴解析极限无 NaN);transform_to_camera 的 ∂Xc/∂ω 在一般转角下的错误公式修复为 cv2.Rodrigues 同源雅可比
- **D-08/D-09 锁定(Task 3):** monkeypatch model.project 加 0.25px → 残差 u 行恰好平移、v 行不动、undo 后逐位复原(行为锁);residuals/scipy_solver 源码无 OpenCV 投影函数名、solve+models 无 numdifftools(源码锁);ToyPinhole 一处注册,OpenCV(通用初始化兜底)与 scipy(同路径精化)双后端不改一行代码求解,RMS < 1px;注册表收尾复原(模块夹具 + 文末显式验证,双保险)
- 回归:全量快速 246 passed / 3 integration 留波次合并后全量跑;本次触碰文件 ruff 全绿

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: 残差与雅可比先行测试** - `7fdec6c` (test)
2. **Task 1 GREEN: 残差与解析稀疏雅可比组装 + 模型层导数接口** - `1689bf2` (feat)
3. **Task 2 RED: scipy 后端行为先行** - `217c100` (test)
4. **Task 2 GREEN: solve_with_scipy 稀疏精化与实用收敛** - `d703b0a` (feat)
5. **Task 3: D-08 残差单一来源锁定 + D-09 玩具模型零侵入** - `8ee3328` (test)

**Plan metadata:** 见本文件提交(docs)

## Files Created/Modified

- `src/calib/solve/residuals.py` - θ 装箱/残差/解析稀疏雅可比/结构矩阵(214 行)
- `src/calib/solve/scipy_solver.py` - scipy 精化后端 + 实用收敛 + f_scale 自动档(243 行)
- `tests/test_solve_scipy_residuals.py` - 装箱往返/双数值基准/行序局部性/结构吻合/性能代理/KB4 近轴/D-08 行为与源码锁(336 行,15 用例)
- `tests/test_models_registry.py` - 玩具模型注册 + 双后端零侵入 + 注册表卫生(219 行,8 用例)
- `src/calib/models/base.py` - +project_point_jacobian 抽象声明;transform_to_camera 的 ∂Xc/∂ω 修复;移除只服务错误公式的 cross_product_matrix
- `src/calib/models/pinhole_brown.py` - +project_point_jacobian(与 unproject 牛顿 2×2 雅可比同式)
- `src/calib/models/kb4.py` - +project_point_jacobian(s=θd/r 全链推导 + 近轴解析极限)
- `tests/test_solve_solvers.py` - +第三批 6 组 scipy 后端用例(真值表/手算对表/精化不劣化/通用初值/异常契约/双后端同构)

## Decisions Made

- 模型层 project_point_jacobian 是 02-01 接口的必要补全而非新功能:计划的解析外参链式(雅可比外参列 = d像素/d相机点 × ∂相机点/∂外参)前半截各模型公式不同必须住模型层;KB4 的 ∂s/∂z 推导中发现 r 全约掉全平面无奇点,∂s/∂x、∂s/∂y 才需要近轴极限(0)——保护面比 project 缩小
- 前向差分基准的容差线 1e-4:基准一是精度锁(<1e-6 硬线),基准二只抓行序错位;两类误差相差 4 个量级,线画中间互不干扰
- 代价尾部改进取运行最小值包络:trf 拒绝尝试步时代价回弹,原始 trace 锯齿会把"走平"误判成"还在降"
- 初值协议显式报错优于静默兜底:未收敛/异模型/缺参数名的初值在入口中文报错(带修复提示),不给"悄悄换一条路算出说不清的结果"留门
- 异常面收窄 (ValueError, ArithmeticError, RuntimeError):数值失败进契约、代码契约错误大声抛——np.linalg.LinAlgError 是 ValueError 子类已被覆盖
- 玩具模型注册表卫生双保险:模块级夹具幂等 pop + 文末测试显式验证注册表回到四个内置模型,无论测试怎么跳过都不留全局污染

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] 模型层缺 project_point_jacobian 接口**
- **Found during:** Task 1 GREEN
- **Issue:** 计划的解析外参链式需要 d像素/d相机点(N,2,3)这前半截,02-01 的 CameraModel 接口没有它;不补则求解侧唯一出路是数值差分——研究 Don't Hand-Roll 明令禁止,且把第二份数学搬进 solve/ 破坏 D-06
- **Fix:** base.py 增抽象声明(含"近轴奇异点必须解析极限顶上,绝不 NaN"硬契约);_PinholeBrown 实现(2×2 核导数与 unproject 牛顿迭代同式,正逆解导数同源);FisheyeKB4 实现(s=θd/r 全链:∂s/∂x=x·B/r²、∂s/∂z=−dθd/dθ/(r²+z²) 全平面无奇点,近轴 ∂s/∂x=∂s/∂y=0、∂s/∂z=−1/z²);测试玩具模型同步实现
- **Files modified:** src/calib/models/base.py, pinhole_brown.py, kb4.py, tests/test_solve_solvers.py
- **Verification:** 双数值基准 M1–M4 全过;KB4 近轴点导数对中心差分一致且无 NaN
- **Committed in:** 1689bf2

**2. [Rule 1 - Bug] transform_to_camera 的 ∂Xc/∂ω 在一般转角下错误**
- **Found during:** Task 1 GREEN(双数值基准首次运行即现形:四模型外参列全错,相对误差 7.6e+1)
- **Issue:** 旧式 ∂Xc/∂ω = −[Xc]× 只是 ω=0 处的极限形式;0.1 rad 小转角下平移列对中心差分吻合 1e-9 而旋转列偏 0.7–0.9——02-01 的 docstring 声称"验证到 6.7e-10"只在 ω=0 附近成立,该导数块此前无测试覆盖
- **Fix:** 改用同一次 cv2.Rodrigues 调用顺手返回的雅可比(布局 rot_jac[k,3i+j]=∂R_ij/∂ω_k),∂Xc_i/∂ω_k = Σ_j ∂R_ij/∂ω_k·Xb_j 逐点 einsum;任意大小转角对中心差分吻合 ~1e-10;移除只服务于错误公式的 cross_product_matrix(无其他引用);docstring 记下历史教训防回退
- **Files modified:** src/calib/models/base.py
- **Verification:** 复测中心差分三档转角(±0.1 / 1.0 / 3.0 rad 量级)全列吻合;双数值基准 M1–M4 全绿
- **Committed in:** 1689bf2

**3. [Rule 3 - Blocking] RED 阶段 ruff 把不存在的模块归为第三方**
- **Found during:** Task 2 RED 提交前
- **Issue:** `from calib.solve.scipy_solver import ...` 在模块文件不存在时被 ruff I001 归到第三方分组,RED 提交过不了 lint(与 02-03 同款)
- **Fix:** RED 提交时 `ruff check --fix` 归位;GREEN 建出模块后重跑 `--fix` 挪回第一方分组
- **Files modified:** tests/test_solve_solvers.py
- **Verification:** 最终 ruff 全绿
- **Committed in:** 217c100 / d703b0a(随任务带入)

---

**Total deviations:** 3 auto-fixed(1 missing critical、1 bug、1 blocking)
**Impact on plan:** 偏差 1 是计划隐含前提的接口补全,偏差 2 是被新测试暴露的上一波遗留错误公式——两者都是"让计划的解析雅可比数学正确"的必要修复,零范围蔓延。

## TDD Gate Compliance

Task 1 与 Task 2 均标 `tdd="true"`,两轮 RED→GREEN 完整走完:

- Task 1 RED `7fdec6c`:测试先行,运行确认 ModuleNotFoundError(No module named 'calib.solve.residuals')后提交;GREEN `1689bf2` 实现至 12 用例全绿(RED 阶段没有测试意外通过,fail-fast 未触发)
- Task 2 RED `217c100`:运行确认 ModuleNotFoundError(No module named 'calib.solve.scipy_solver')后提交;GREEN `d703b0a` 实现至 23 用例全绿
- Task 3 为测试型任务(无 tdd 标记),提交 `8ee3328`,8 用例即时全绿

## Issues Encountered

- 既有 ruff 违例 5 处(02-01 遗留,记录于 deferred-items.md):本次触碰的三个模型文件里的 3 处(base.py RET501、kb4.py UP037、pinhole_brown.py RUF059)随本计划的 ruff --fix 与改名一并清掉——触碰文件不应带着 lint 债交付;models/__init__.py 的 noqa+F401×2 未触碰,仍留待后续
- KB4 project_point_jacobian 推导初稿对 ∂s/∂z 也做了 r 保护,复查发现 r 在链式里全约掉、该分量全平面无奇点——保护范围收窄,代码更简

## User Setup Required

None - 无外部服务配置。

## Requirement Traceability

- 计划 frontmatter 声明 `requirements: [SOL-01, MOD-01]`。SOL-01 的 scipy 精化半边(含鲁棒核与实用收敛)由本计划交付,双后端至此齐——勾选动作留给 orchestrator 波次合并后统一执行(并行执行器不改共享规划产物)。MOD-01(02-01 已勾)的零侵入注册在本计划补上消费方证明:玩具模型一处注册、双后端可用、求解器源码零具体模型依赖(源码级断言锁死)

## Next Phase Readiness

- 02-05(试车矩阵):solve_with_scipy 的初值协议(init=收敛的 CalibrationResult)与 ScipyConfig.losses 矩阵就位;线性参照(RMS 不劣化判据的 rms_linear_ref 入参)留好了鲁棒核对照的钩子
- 03-评估(Fisher 信息):解析稀疏雅可比(residuals.py)与 project_point_jacobian 直接复用——理论界与求解侧同一套导数是 D-06 的落地点
- 阻塞:无

---
*Phase: 02-m1-m4*
*Completed: 2026-10-18*

## Self-Check: PASSED

自检于提交前完成:8 个产出文件(新建 4 + 修改 4)与 5 个任务提交(7fdec6c / 1689bf2 / 217c100 / d703b0a / 8ee3328)均确认存在于本分支。
