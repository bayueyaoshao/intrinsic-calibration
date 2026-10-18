---
phase: 02-m1-m4
plan: "02"
subsystem: calibration-pipeline
tags: [numpy, npz, dataclass, observation-loading, row-order-contract]

# Dependency graph
requires:
  - phase: 01-data-foundation-minimal-calibration
    provides: sb 路线角点 npz(image_ids/uv/detected/source 四键,行序=清单序)、ingest manifest.json、solve/baseline.py 的 build_object_points
provides:
  - ObservationSet 冻结数据类(双求解后端与存疑帧复核共用的唯一输入形态)
  - load_observation_set 装载函数(行序契约:观测集顺序 = 求解结果顺序 = 落盘顺序)
  - excluded_image_ids 剔除名单查询(供血缘落盘留痕)
affects: [02-m1-m4(02-03 求解器、02-05 试车矩阵、D-04 存疑帧复核), 04-评估(按姿态切分用 pose_ids)]

# Tech tracking
tech-stack:
  added: []   # 无新依赖;numpy/pyyaml 均为既有
  patterns:
    - 行序锁:装载入口对 npz image_ids 与清单逐项断言,错位即抛中文错(研究 Pitfall 10)
    - allow_pickle=False 的 npz 读取防线延续到 solve 层(T-02-02-01)
    - 中文报错统一走 _fail 帮助函数(io/config.py 体例,兼过 ruff TRY004)

key-files:
  created:
    - src/calib/solve/observation.py
    - tests/test_solve_observation.py
  modified: []

key-decisions:
  - "被剔除名单经独立函数 excluded_image_ids 查询(计划行为 2 明示「返回值或独立函数」二选一),ObservationSet 保持草案的六字段不被撑破"
  - "物点网格直接 import baseline.build_object_points(计划验收标准明示复用,不重写网格)"
  - "SOL-01 本计划只落地输入侧,求解器侧(02-03)未动,需求保持未完成"

patterns-established:
  - "行序契约单一入口:后续任何消费角点的代码一律经 load_observation_set,不自行 np.load"

requirements-completed: []   # SOL-01 求解器接口整体未完成(本计划只是其输入侧),留待 02-03 后再勾

# Metrics
duration: 7min
completed: 2026-10-18
---

# Phase 2 Plan 02: 观测集装载 Summary

**ObservationSet 冻结数据类 + load_observation_set 装载函数:把 sb 路线角点 npz 与数据清单逐行对齐(行序锁断言)、逻辑剔除/未检出过滤/NaN 坏数拦截/视图数下限,9 个合成数据用例锁死全部六条装载契约**

## Performance

- **Duration:** 约 7 分钟(2026-10-18T01:08:18Z → 01:15Z,worktree 并行波次 1)
- **Started:** 2026-10-18T01:08:18Z
- **Completed:** 2026-10-18T01:15:00Z
- **Tasks:** 2/2
- **Files modified:** 2(新建 2)

## Accomplishments

- `ObservationSet` 冻结数据类六字段齐全(image_ids/pose_ids/object_points/image_points/image_size/corner_source),行序契约写进模块 docstring 全文(观测集顺序 = 求解结果顺序 = 落盘顺序,Pitfall 10)
- `load_observation_set`:npz 强制 `allow_pickle=False`;清单按切分过滤(顺序即清单序);npz 与清单 image_ids 逐项断言,错位报中文错并指明第一处不一致行号与两侧编号;剔除/未检出/NaN/视图数 < 2 四道防线沿用阶段一 run_baselines 规则;缺 source 键拒绝(非 sb 路线,D-02);split 白名单 train/test
- `excluded_image_ids`:本轮实际被人工剔除名单可查(字典序去重,别切分的名字不算命中,T-03-02 留痕体例)
- 9 个合成用例(≥ 计划要求的 8)锁死六条行为,行序锁与 NaN 断言各有独立用例;全量快速回归 112 passed(阶段一 103 + 新 9),3 条 integration 按波次合并前全量跑

## Task Commits

Each task was committed atomically:

1. **Task 1: ObservationSet 数据类与装载函数** - `77bb7af` (feat)
2. **Task 2: 装载契约测试** - `297fcbc` (test)

**Plan metadata:** 见本文件提交(docs)

## Files Created/Modified

- `src/calib/solve/observation.py` - 观测集数据类 + 装载函数 + 剔除名单查询(235 行)
- `tests/test_solve_observation.py` - 装载契约测试,9 用例全合成数据秒级(268 行)

## Decisions Made

- 被剔除名单走独立函数 `excluded_image_ids` 而非塞进 ObservationSet 第七字段:计划行为 2 明示两种做法皆可,保持数据类与 02-RESEARCH §Pattern 2 草案的六字段一致,求解器输入不带剔除元数据
- 物点网格复用 `baseline.build_object_points`(计划验收标准明示"本文件不重写网格"),测试用 `np.array_equal` 逐值锁死该复用
- manifest 读取按后缀分派 json / yaml.safe_load(计划写"yaml/json 读入"),安全路径与 io/config.py 一致

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] 角点形状与棋盘规格一致性防线**
- **Found during:** Task 1(ObservationSet 数据类与装载函数)
- **Issue:** 计划只要求 NaN/Inf 断言;若角点文件与 board.yaml 的棋盘规格不是同一套(内角点数不同),image_points 组装会静默产出与 object_points 行数不匹配的数组,错误延迟到求解器才爆
- **Fix:** 装载时断言 `uv.shape[1:] == (棋盘内角点数, 2)`,三要素中文报错(期望/实际/下一步:核对 board.yaml)
- **Files modified:** src/calib/solve/observation.py
- **Verification:** `uv run pytest -m "not integration" -q` 112 passed;手动构造 87 点 npz 验证报错路径(形状 (N,88,2) 契约由行为 1 用例正向锁定)
- **Committed in:** 77bb7af(Task 1 提交)

**2. [Rule 3 - Blocking] ruff TRY004 与中文 ValueError 报错的兼容**
- **Found during:** Task 1 提交前检查
- **Issue:** 环境安装的 ruff 默认规则含 TRY004(isinstance 分支里直接 raise ValueError 被判应为 TypeError),与计划要求的"清单结构缺失抛中文 ValueError"冲突
- **Fix:** 中文报错统一走 `_fail(message)` 帮助函数(raise 移进函数体),与 io/config.py 既有 `_fail` 体例完全一致,报错行为不变
- **Files modified:** src/calib/solve/observation.py
- **Verification:** `uv run ruff check` 通过;`uv run pytest -m "not integration" -q` 112 passed
- **Committed in:** 77bb7af(Task 1 提交)

**3. [Rule 2 - Missing Critical] manifest 结构缺失的中文防线**
- **Found during:** Task 1
- **Issue:** 清单缺 images 列表或缺 summary.resolution 时,裸 KeyError/TypeError 对使用者不可读
- **Fix:** `_split_manifest` 里对两处结构做前置校验,缺失报中文 ValueError(文件名 + 缺什么 + 下一步)
- **Files modified:** src/calib/solve/observation.py
- **Verification:** 112 passed;错误路径消息体例与 io/config.py 一致
- **Committed in:** 77bb7af(Task 1 提交)

---

**Total deviations:** 3 auto-fixed(2 missing critical、1 blocking)
**Impact on plan:** 全部是正确性防线与工具链兼容,零范围蔓延;计划主路径(六字段、行序锁、四道过滤防线、六条契约测试)按原文执行。

## TDD 说明

Task 2 标了 `tdd="true"`,但本计划的任务排序把实现(Task 1)放在测试(Task 2)之前,且 `<output>` 的原子提交草案明示 feat → test 顺序——按计划顺序执行,未做独立的 RED 失败提交。测试与实现分属两个原子提交,契约由 9 个用例正向锁定。

## Issues Encountered

None

## User Setup Required

None - 无外部服务配置。

## Requirement Traceability

- 计划 frontmatter 声明 `requirements: [SOL-01]`,但 SOL-01 是"求解器统一接口 + 双后端"整体:本计划只落地其输入侧(ObservationSet),求解器侧属 02-03。故 `requirements-completed` 留空,SOL-01 待 02-03 完成后再勾,避免提前宣告完成。

## Next Phase Readiness

- 02-03(双求解器)可直接以 `ObservationSet` 为输入类型;行序契约、pose_ids 连接键、corner_source 留痕已就位
- 02-05 试车矩阵的真实 sb 角点(640 train)装载验证按计划由其 integration 试车承担
- 无阻塞

---
*Phase: 02-m1-m4*
*Completed: 2026-10-18*

## Self-Check: PASSED

自检于提交前完成:两个新建文件、两个任务提交(77bb7af / 297fcbc)均确认存在。
