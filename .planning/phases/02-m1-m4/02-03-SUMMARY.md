---
phase: 02-m1-m4
plan: "03"
subsystem: calibration-pipeline
tags: [opencv, numpy, scipy-config, tdd, adapter-dispatch, artifact-persistence]

# Dependency graph
requires:
  - phase: 02-m1-m4(02-01)
    provides: CameraModel 协议 + M1–M4 注册机制 + OpenCV 适配器(kind/flags/params_from_cv 解包)
  - phase: 02-m1-m4(02-02)
    provides: ObservationSet 冻结数据类(行序契约:观测集顺序 = 求解结果顺序 = 落盘顺序)
  - phase: 01-data-foundation-minimal-calibration
    provides: solve/baseline.py 阶段一基线(失败契约/形状断言体例的对齐基准)
provides:
  - CalibrationResult 统一结果结构(16 字段;双后端同构;失败契约:键全在、数值一律 None)
  - compute_result_metrics 同一把尺(models.project + float64 回代重算 RMS,Pitfall 9)
  - save_result/load_result/load_extrinsics/load_residuals 三件套落盘读盘(result.json / extrinsics.npz / residuals.npz)
  - solve_with_opencv 后端入口(适配器分派 calibrateCamera/fisheye + 通用初始化兜底,D-09 零侵入)
  - ScipyConfig + CalibrateConfig(corner_route/models/solvers 矩阵)进 io/config.py 与 calibration.yaml
  - conftest.make_true_views 夹具上提(阶段一基线与阶段二求解器共用)
affects: [02-m1-m4(02-04 scipy 精化直接消费 CalibrationResult 与 ScipyConfig), 02-m1-m4(02-05 试车矩阵按 models×solvers 跑), 03-评估(按 result.json/extrinsics.npz/residuals.npz 读结论)]

# Tech tracking
tech-stack:
  added: []   # 无新依赖;cv2/numpy 均为既有
  patterns:
    - 适配器分派:求解器不 import 任何具体模型,只认 opencv_adapter() 的 kind/flags(D-09 零侵入)
    - 同一把尺:OpenCV 内部 RMS 只进 solver_meta 留参考,对外 RMS 一律 compute_result_metrics 回代(Pitfall 9)
    - 绑定层精度契约:calibrateCamera 系只收 float32(降精度喂进去),fisheye 系要 (N,1,x) float64
    - 三件套一体:收敛结果落盘必须带 residuals (V,P,2),缺残差直接中文报错不静默
    - npz 读写强制 allow_pickle=False 防线延伸到结果落盘层

key-files:
  created:
    - src/calib/solve/result.py
    - src/calib/solve/opencv.py
    - tests/test_solve_solvers.py
  modified:
    - src/calib/io/config.py
    - config/calibration.yaml
    - tests/test_io_config.py
    - tests/conftest.py
    - tests/test_solve_baseline.py

key-decisions:
  - "save_result 增加 residuals 可选参数(计划签名没有):CalibrationResult 按计划不带残差字段,残差由 compute_result_metrics 单独产出;收敛结果缺残差直接 ValueError(三件套一体,不静默漏写)"
  - "result.json 的结构键与 CalibrationResult 字段分成两套契约测试元组:rvecs/tvecs 住 extrinsics.npz,不进 JSON(npz 三件套职责划分)"
  - "鱼眼迭代停止条件硬编码 (100, 1e-6):solve_with_opencv 不吃配置段,沿用阶段一 FisheyeConfig 实测默认即可复现行为;M4 的 flags=10 由模型适配器自带,与配置段解耦"
  - "通用初始化兜底给 solvePnP 传空畸变向量(长度 0):任意参数个数的玩具模型不被针孔族 4/5/8 畸变布局绑架,与 default_params 畸变清零严格一致"
  - "SOL-01 仍未完成:本计划只交付 OpenCV 侧与统一结果结构,scipy least_squares 精化侧(含鲁棒核)属 02-04,双后端未齐不提前勾需求"

patterns-established:
  - "新求解后端只做两件事:吃 ObservationSet + CameraModel,吐 CalibrationResult——结果结构与落盘一律复用 result.py,不另造格式"

requirements-completed: []   # SOL-01 双后端整体未齐(scipy 精化侧属 02-04);MOD-01 已由 02-01 勾掉,本计划只是其零侵入注册的消费方证明

# Metrics
duration: 11min
completed: 2026-10-18
---

# Phase 2 Plan 03: 双求解基座 Summary

**CalibrationResult 统一结构 + 三件套落盘(同一把尺 float64 回代,Pitfall 9)+ OpenCVSolver 适配器分派(M1–M4 真值恢复 <1% 误差)与玩具模型通用初始化兜底(D-09 零侵入),外加 calibration.yaml 模型矩阵与 ScipyConfig 白名单校验——16 个求解器用例 + 10 个配置用例,全量快速回归 216 passed**

## Performance

- **Duration:** 约 11 分钟(2026-10-18T01:24:25Z → 01:35Z,worktree 并行波次)
- **Started:** 2026-10-18T01:24:25Z
- **Completed:** 2026-10-18T01:35:19Z
- **Tasks:** 3/3
- **Files modified:** 8(新建 3,修改 5)

## Accomplishments

- **配置层(Task 1):** ScipyConfig 冻结数据类(method/tr_solver/losses/f_scale/max_nfev/三容差)与 CalibrateConfig(corner_route/models/solvers 矩阵 + scipy 嵌套)进 `io/config.py`,六组白名单全中文报错;`calibration.yaml` 的 calibrate 段扩成模型矩阵,fisheye 段注明仅供阶段一复现(M4 flags 由适配器自带);method 拒 lm 的报错带"稠密"内存量级与可选项说明
- **结果结构(Task 2, TDD):** `CalibrationResult` 16 字段 + `make_failure` 失败契约工厂(params 空字典占位、数值一律 None)+ `compute_result_metrics` 同一把尺(经 models.project 与 transform_to_camera 回代,不出现任何 cv2 投影)+ 三件套落盘读盘;失败组合照样写 result.json(数值键全 null),收敛结果缺残差直接中文报错
- **OpenCV 后端(Task 3, TDD):** `solve_with_opencv` 适配器分派——M1/M2/M3 走 calibrateCameraExtended(float32 喂绑定层、flags 128/0/16384)、M4 走 fisheye.calibrate(单位阵 K 起步、flags=10 锁 skew)、无适配器模型走通用初始化兜底(default_params 组 K + 逐视图 solvePnP,空畸变向量);M1–M3 合成真值恢复 RMS < 0.5px 且内参误差 < 1%,M4 鱼眼合成双分支;cv2.error 走 make_failure 中文契约,solver_meta 保留 path/flags_used/opencv_rms
- **夹具上提:** `make_true_views` 从 test_solve_baseline.py 上提到 conftest.py(逻辑不变),阶段一基线与阶段二求解器测试共用
- 回归:solver + baseline 并排 25 passed(阶段一基线未破坏),全量快速 216 passed / 3 integration 按波次合并后全量跑

## Task Commits

Each task was committed atomically:

1. **Task 1: 配置层模型矩阵与 scipy 参数** - `99f180f` (feat)
2. **Task 2: 结果结构 RED** - `d4c5caf` (test)
3. **Task 2: CalibrationResult + 三件套落盘 GREEN** - `8547540` (feat)
4. **Task 3: OpenCVSolver RED** - `76ed130` (test)
5. **Task 3: 适配器分派与通用初始化兜底 GREEN** - `2906680` (feat)

**Plan metadata:** 见本文件提交(docs)

## Files Created/Modified

- `src/calib/solve/result.py` - 统一结果结构 + 同一把尺回代 + 三件套落盘读盘(255 行)
- `src/calib/solve/opencv.py` - OpenCV 后端:适配器分派 + 通用初始化兜底(205 行)
- `tests/test_solve_solvers.py` - 结果结构与后端行为测试,16 用例全合成数据秒级(438 行)
- `src/calib/io/config.py` - ScipyConfig/CalibrateConfig + 六组白名单解析
- `config/calibration.yaml` - calibrate 段扩成模型矩阵(corner_route/models/solvers/scipy)
- `tests/test_io_config.py` - +10 配置校验用例(仓库 yaml 矩阵值、缺省回退、六类坏值)
- `tests/conftest.py` - make_true_views 夹具上提共用
- `tests/test_solve_baseline.py` - 改用共用夹具,9 用例原样通过

## Decisions Made

- `save_result(out_dir, result, residuals=None)`:计划草案签名没有残差参数,但 CalibrationResult 按计划不带残差字段、save_result 也拿不到 obs+model 无法自行回代——残差由调用方从 compute_result_metrics 取来一并传入;收敛结果缺残差抛中文 ValueError(三件套一体),失败结果不写 npz(没有位姿与残差可写时硬写只会撒谎)
- result.json 结构键与数据类字段分两套测试契约:rvecs/tvecs 只进 extrinsics.npz(JSON 里放数组会撑爆可读性,且 npz 才是阶段三/四的消费形态)
- 鱼眼迭代停止条件固定 (100, 1e-6) 硬编码于 opencv.py:后端入口不吃配置段(矩阵在 calibrate 段,是给 02-05 试车矩阵与 02-04 scipy 用的),复现阶段一 FisheyeConfig 实测默认
- 通用初始化兜底用空畸变向量调 solvePnP:玩具模型参数个数任意,不该被 OpenCV 针孔族 4/5/8 畸变布局绑架;与 default_params"畸变清零"严格一致
- 玩具模型兜底测试的合成视角按玩具自己的 default_params 规则生成 K(焦距 0.75×宽):既验证兜底机制(K 组装 + solvePnP 位姿估计)真实工作,又不让"真值焦距 ≠ 初值焦距"的必然残差把断言搞成假失败

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] result.json 结构键契约的拆分**
- **Found during:** Task 2 GREEN
- **Issue:** 计划验收要求"失败结果 JSON 数值键全 null、结构键一个不少",但位姿(rvecs/tvecs)按设计住 extrinsics.npz 不进 JSON——用 16 字段数据类元组循环断言 JSON 键会把 npz 职责误判成缺失
- **Fix:** 测试契约拆成两套元组:`_RESULT_FIELDS`(数据类 16 字段)与 `_RESULT_JSON_KEYS`(JSON 14 键,无 rvecs/tvecs),各自独立断言
- **Files modified:** tests/test_solve_solvers.py
- **Verification:** 25 passed(solver + baseline 并排)
- **Committed in:** 8547540(Task 2 GREEN)

**2. [Rule 3 - Blocking] RED 阶段 ruff 把不存在的模块归为第三方**
- **Found during:** Task 2/3 RED 提交前
- **Issue:** `from calib.solve.result import ...` 在模块文件尚不存在时被 ruff I001 归到第三方分组(它按"文件是否存在"判第一方),RED 提交过不了 lint
- **Fix:** RED 提交时 `ruff check --fix` 归位;GREEN 建出模块文件后重跑 `ruff check --fix` 把 import 挪回第一方分组(两次各修 1 处 I001)
- **Files modified:** tests/test_solve_solvers.py
- **Verification:** GREEN 后 `uv run ruff check src/calib/solve/ tests/` 通过
- **Committed in:** d4c5caf / 8547540 / 76ed130 / 2906680(随任务各自带入)

**3. [Rule 2 - Missing Critical] 玩具模型兜底路径缺独立行为验证**
- **Found during:** Task 3 RED
- **Issue:** 计划的 must_haves 要求"无适配器模型走通用初始化兜底可求解",但未列具体用例;不补则 D-09 的核心卖点(新模型实现协议即可被后端使用)没有回归防线
- **Fix:** 增加 `_ToyNoAdapterModel`(不注册、无适配器的四参数玩具针孔)+ 兜底路径用例:断言 path == "generic_pnp"、收敛、RMS < 0.5px;另加源码静态断言(opencv.py 不得 import 任何具体模型)锁死零侵入
- **Files modified:** tests/test_solve_solvers.py
- **Verification:** 16 collected 全过
- **Committed in:** 76ed130(Task 3 RED)

---

**Total deviations:** 3 auto-fixed(2 missing critical、1 blocking)
**Impact on plan:** 全部是契约精确化与测试防线补强,零范围蔓延;三条主路径(配置矩阵、结果结构、OpenCV 后端)按计划原文执行。

## TDD Gate Compliance

Task 2 与 Task 3 均标 `tdd="true"`,两轮 RED→GREEN 完整走完:

- Task 2 RED `d4c5caf`:测试先行,运行确认 ModuleNotFoundError(No module named 'calib.solve.result')后提交;GREEN `8547540` 实现至 16 用例全绿
- Task 3 RED `76ed130`:运行确认 ModuleNotFoundError(No module named 'calib.solve.opencv')后提交;GREEN `2906680` 实现至全绿(RED 阶段没有测试意外通过,fail-fast 未触发)

## Issues Encountered

- 全量 `ruff check src/ tests/` 有 5 处既有违例,全部位于上一波(02-01)的模型文件(models/__init__.py 非法 noqa + F401×2、base.py RET501、kb4.py UP037、pinhole_brown.py RUF059),与本项目改动无关——按范围边界记录到 `deferred-items.md`,未修

## User Setup Required

None - 无外部服务配置。

## Requirement Traceability

- 计划 frontmatter 声明 `requirements: [SOL-01, MOD-01]`。SOL-01 是"求解器统一接口 + 双后端(OpenCV 初始化 → scipy least_squares 精化,含鲁棒核)"整体:本计划交付统一结果结构、落盘三件套与 OpenCV 侧,scipy 精化侧属 02-04,双后端未齐,不提前勾。MOD-01 已由 02-01 标记完成,本计划是其"注册机制零侵入(D-09)"的消费方证明(源码级断言锁死求解器不 import 具体模型)

## Next Phase Readiness

- 02-04(scipy 精化)直接消费:CalibrationResult 的 init_from 字段留好了血缘位、ScipyConfig 已进配置层、OpenCV 收敛结果的 params/rvecs/tvecs 可作 scipy 初值
- 02-05(试车矩阵)的 models×solvers 矩阵已在 calibration.yaml 与 CalibrateConfig 就位
- 阻塞:无

---
*Phase: 02-m1-m4*
*Completed: 2026-10-18*

## Self-Check: PASSED

自检于提交前完成:8 个产出文件(新建 3 + 修改 5)与 5 个任务提交(99f180f / d4c5caf / 8547540 / 76ed130 / 2906680)均确认存在于本分支。
