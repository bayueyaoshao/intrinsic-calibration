---
phase: 02-m1-m4
plan: 01
subsystem: models
tags: [camera-models, brown-conrady, kb4-fisheye, analytic-jacobian, opencv-parity, numdifftools, registry]

# Dependency graph
requires:
  - phase: 01-pipeline-core
    provides: 项目骨架与测试体例(test_solve_baseline.py 组织范式、conftest 合成夹具、uv 环境)
provides:
  - CameraModel 抽象 + register_model/create_model 注册机制(一处注册、两个求解器都能用,D-09 基座)
  - M1/M2/M3(针孔 Brown-Conrady,8/9/12 参)与 M4(鱼眼 KB4,8 参无 skew)的
    投影、反投影(牛顿迭代真逆解)、解析雅可比(只含内参列)
  - 各模型 OpenCV 适配器(flags 128/0/16384/10、K/D 装包解包、D 长度切片规则)
  - 共享外参链式工具 transform_to_camera(∂Xc/∂ω=−[Xc]×、∂Xc/∂t=I)
  - parity(37 用例)+ 数值雅可比(41 用例)双重测试锁死,float64 对齐 OpenCV 至 ~5e-13px
affects: [02-02 观测集装载, 02-03 双求解后端, 02-05 试车矩阵, 阶段三 Fisher/CRLB, 阶段四评估]

# Tech tracking
tech-stack:
  added: [numdifftools 0.11.1(仅 dev 组,D-10:数值差分基准只进测试)]
  patterns: [CameraModel ABC + 注册表(求解器零具体模型 import), OpenCV 适配器
    封装 flags/K-D 互译为唯一入口, np.divide(out=,where=) 近轴防 0/0 且无告警,
    列相对误差判定雅可比(统一阈值不逐列绝对值)]

key-files:
  created:
    - src/calib/models/base.py
    - src/calib/models/pinhole_brown.py
    - src/calib/models/kb4.py
    - tests/test_models_parity.py
    - tests/test_models_jacobian.py
  modified:
    - src/calib/models/__init__.py
    - pyproject.toml
    - uv.lock

key-decisions:
  - "M4 定死 8 参无 skew(fisheye.projectPoints 实测忽略 K[0][1],带死参数会脱节);适配器 flags=10(FIX_SKEW)"
  - "unproject 全部真逆解(两族各一段牛顿迭代),project→unproject→project 往返 <1e-9px 测试锁定(研究问题 4 定夺落地)"
  - "雅可比判定统一为列相对误差 <1e-6(numdifftools 基准;中心差分地板 ~7e-10,余量充足;研究问题 3 定夺落地)"
  - "KB4 近轴保护用 np.divide(out=1/z, where=r≥1e-12):既防 0/0=NaN,又不产生'先全除再挑'的运行时告警"
  - "适配器解包切片规则:D[:4](M1,返回 5 长)/ D[:5](M2)/ D[:8](M3,返回 14 长)/ D 全 4(M4),往返逐位相等锁定"

patterns-established:
  - "注册机制:类头 @register_model 一处登记,create_model 按名查表;求解器只见 CameraModel 协议,不见具体模型名"
  - "OpenCV dtype/形状规则全部收进适配器唯一入口:模型数学层 float64 进出,绝不散落装包代码"

requirements-completed: [MOD-01, MOD-02]

# Metrics
duration: 17min
completed: 2026-10-18
---

# Phase 2 Plan 1: 模型层数学核心 Summary

**M1–M4 四类相机模型的投影/反投影/解析雅可比 + 注册机制落地为全项目唯一数学源,parity 对齐 OpenCV 至 ~5e-13 像素,78 条新测试双重锁死(D-06:测试先于任何求解代码)**

## Performance

- **Duration:** 17 min(2026-10-18T00:47:49Z → 2026-10-18T01:04:34Z)
- **Tasks:** 3/3
- **Files modified:** 8(新建 5、修改 3)

## Accomplishments
- CameraModel 抽象 + 注册机制:register_model 装饰器、create_model 中文报错带可选清单;
  transform_to_camera/cross_product_matrix 共享外参导数块(∂Xc/∂ω=−[Xc]× 实测语义)
- 四模型实现与 OpenCV 逐点一致:M1/M2/M3 对 cv2.projectPoints、M4 对
  cv2.fisheye.projectPoints,float64 下最大偏差 ~5e-13px(断言线 1e-9,留 3 个数量级)
- 解析雅可比对 numdifftools 数值基准逐模型逐参数(8+9+12+8=37 组合)列相对误差
  <1e-6(实测 ~1e-10);M1–M3 另对 cv2 雅可比 float64 旁证 <1e-4
- 四模型 unproject 真逆解(牛顿迭代),往返误差 <1e-9px;KB4 近轴(含正原点)无 NaN
- 快速回归 181 条全绿(阶段一 103 + 本计划 78),D-10 grep 确认 src/ 无 numdifftools

## Task Commits

Each task was committed atomically:

1. **Task 1: dev 依赖 numdifftools + CameraModel 抽象与注册表** - `5de031b` (build)
2. **Task 2: 针孔族 M1/M2/M3 与鱼眼 M4 实现(投影/雅可比/适配器)** - `98ca90a` (feat)
3. **Task 3: parity 一致性比对测试 + 数值雅可比测试(容差矩阵)** - `fcf41ba` (test)

## Files Created/Modified
- `src/calib/models/base.py` - CameraModel 抽象(6 个方法契约 + opencv_adapter 默认
  None 走通用初始化)+ transform_to_camera 外参链式块 + 形状校验助手(188 行)
- `src/calib/models/pinhole_brown.py` - M1/M2/M3 共享 Brown-Conrady 畸变核
  (切向项在有理分母之外)、雅可比(M3 商法则)、畸变逆解牛顿迭代、OpenCV 适配器(319 行)
- `src/calib/models/kb4.py` - M4 KB4 投影(近轴保护)、雅可比(θ^(2j+3) 幂次)、
  θ 反解牛顿迭代、fisheye 适配器 flags=10(198 行)
- `src/calib/models/__init__.py` - 注册表本体 + 底部 import 触发四模型注册
- `tests/test_models_parity.py` - 37 用例:float64 parity ×22、float32 sanity ×3、
  适配器往返 ×4、反投影往返 ×8、近轴有限值、注册报错
- `tests/test_models_jacobian.py` - 41 用例:numdifftools 逐参数 ×37、cv2 旁证 ×3、
  D-10 静态扫描 ×1
- `pyproject.toml` / `uv.lock` - dev 组加 numdifftools>=0.11(D-10 注释写明用途边界)

## Decisions Made
- 除计划内既定决策外,执行期新定:KB4 雅可比/投影的近轴保护统一改用
  np.divide(out=..., where=...) 写法——np.where 会先算 0/0 再挑选,
  虽然结果正确但触发运行时告警,测试环境不干净
- cv2 4.14 的鱼眼常量名是 CALIB_RECOMPUTE_EXTRINSIC(无 fisheye 前缀的简短名
  不存在),flags 语义值与计划一致(2|8=10)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] KB4 雅可比 k 列幂次差一错误**
- **Found during:** Task 2(提交前脚本自测,行为 3)
- **Issue:** ∂θd/∂k_j 写成 theta**(2*j+1)(j 从 0 数),正确是 theta**(2*j+3)
  (公式按 j 从 1 数:k1 配 θ³、k4 配 θ⁹);自测列相对误差 0.42 暴露
- **Fix:** 幂次改为 2*j+3,注释写明 j 的计数约定
- **Files modified:** src/calib/models/kb4.py
- **Verification:** 自测 + Task 3 全部 37 组合雅可比用例 <1e-6(实测 ~1e-10)
- **Committed in:** 98ca90a(Task 2 提交,自测通过后才提交)

**2. [Rule 3 - Blocking] cv2.fisheye 常量名 AttributeError**
- **Found during:** Task 2(首次导入即炸)
- **Issue:** 计划/研究里写的 RECOMPUTE_EXTRINSIC 在 cv2 4.14 绑定里的实际名是
  CALIB_RECOMPUTE_EXTRINSIC
- **Fix:** 改用实际常量名,数值验证 2|8=10 与计划一致
- **Files modified:** src/calib/models/kb4.py
- **Verification:** flags 自测 = 10;uv run pytest 全绿
- **Committed in:** 98ca90a

**3. [Rule 1 - Bug] KB4 近轴分支的 0/0 运行时告警**
- **Found during:** Task 2 自测(正原点用例)
- **Issue:** np.where(r<eps, 1/z, theta_d/r) 会先对全部点算 theta_d/r(含 0/0),
  结果虽被 where 挑对但产生 RuntimeWarning: invalid value encountered in divide
- **Fix:** 改 np.divide(theta_d, r, out=1/z, where=r>=eps)(投影与雅可比同改),
  不安全分支从不被计算
- **Files modified:** src/calib/models/kb4.py
- **Verification:** 自测无告警输出;近轴点(含 r=0)投影/雅可比全有限
- **Committed in:** 98ca90a

---

**Total deviations:** 3 auto-fixed(2 bug、1 blocking)
**Impact on plan:** 全部为 Task 2 数学实现的就地修正,无范围蔓延;自测先于提交的门禁把两处 bug 拦在了提交前。

## Issues Encountered
- Task 1 的验收"create_model('m0') 报错含 m1"在 Task 1 时点不可满足(注册表
  按计划此刻为空,具体模型 Task 2 才注册)——按计划顺序在 Task 2/3 复核通过
  (tests/test_models_parity.py::test_registry_error_lists_options 锁定)
- uv.lock 随 uv sync 更新(计划文件清单未列,但它是 dev 依赖变更的直接产物,
  随 Task 1 一并提交保证环境可复现)
- Task 2 标注 tdd="true" 且计划明确规定"先以脚本自测通过再提交、正式锁定在
  Task 3":按计划顺序执行,自测脚本(五条行为全量)承担 RED/GATE 职能并实际
  拦下 2 处 bug,正式 pytest 锁定在 Task 3(test 提交)

## User Setup Required
None - 纯本地数值模块,无外部服务。

## Next Phase Readiness
- models/ 数学源就绪:02-03 双求解器(OpenCV 适配器路径 + scipy 残差路径)、
  阶段三 Fisher/CRLB 均可直接 import 这一份投影/雅可比
- D-06 顺序成立:本计划连同比对测试全部落地且通过时,仓库尚无任何新求解代码
- 适配器已提供求解器需要的全部入口:kind/flags/params_to_cv/params_from_cv;
  玩具模型路径(opencv_adapter 默认 None)留给 02-03 的注册零侵入测试
- 无阻塞;Wave 1 并行的 02-02(观测集装载)与本计划无文件交集

## Known Stubs

无占位实现:unproject 是真逆解(计划修订明确不留占位),全部数值路径有测试覆盖。

## Threat Flags

无新增威胁面:纯内存数值计算,无网络/文件/凭据接触;dev 依赖 numdifftools
供应链已在计划威胁表(T-02-01-SC)审计为 [OK],Task 3 的 grep 锁定确认
src/ 无引用。

---
*Phase: 02-m1-m4*
*Completed: 2026-10-18*
