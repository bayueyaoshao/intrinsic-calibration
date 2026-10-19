---
phase: 04-evaluation-model-selection
plan: 02
subsystem: evaluation
tags: [optimizer-comparison, robust-loss, multistart, scipy, lineage-artifacts, integration-test]

# Dependency graph
requires:
  - phase: 04-evaluation-model-selection (04-01)
    provides: EvaluateConfig 25 键契约、run_parallel 并行助手、_seed_for 种子派生体例
  - phase: 02-m1-m4
    provides: mX__opencv / mX__scipy 标定产物(共同初值的落盘真相源)、combo_dir_name 命名、_load_init_result 装载体例
  - phase: 03-fisher-crlb
    provides: 集成测试体例(定位最新上游轮 + 契约断言 + 血缘核对,tests/test_fisher_realdata.py)
provides:
  - run_optimizer_grid:16 组合网格(4 核 × 4 模型,共同 OpenCV 初值)+ 每模型推荐优化配置(D-13/D-14)
  - run_multistart:多起点扰动一致性检验(双容差判定,鱼眼 N 翻倍,D-15)
  - artifacts.py:optimizer.json 8 键契约 + 失败契约工厂 + 明细 npz 旁车(allow_pickle=False)
  - output/evaluate/run_20261019_112926:真实数据首轮 optimizer.json(下游 04-03~04-06 的统一配置来源)
affects: [04-03-selection, 04-04-experiments, 04-05-experiments, 04-06-refit-test-scoring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "两阶段并行派发:先 linear 核再鲁棒核(linear 的 RMS 是鲁棒核实用收敛判定参照 rms_linear_ref,必须先算完)"
    - "worker 层失败契约兜底:worker 函数体内 try/except 转 make_failure,异常绝不过 run_parallel 生成器(串行回退时 worker 异常会杀死整个任务流)"
    - "多起点种子派生:_seed_for(f\"mstart|{model}|{i}\", cfg.seed),同配置重跑逐位一致(cv2.setNumThreads 锁线程数)"
    - "参数一致性像素当量:同一参考位姿投影棋盘网格,逐角点位移最大值代替参数数值差"

key-files:
  created:
    - src/calib/evaluation/optimizer.py
    - src/calib/evaluation/artifacts.py
    - tests/test_optimizer_realdata.py
  modified:
    - tests/test_optimizer_comparison.py

key-decisions:
  - "run_multistart 签名取整个 CalibrationConfig(计划文本写 eval_cfg 但动作要 cfg.seed):照 04-01 既定体例(要种子的函数取全量配置),随机性从 cfg.seed 按 (模型, 起点序号) 确定性派生"
  - "四模型推荐均为 linear 核:真实数据上没有任何鲁棒核的 RMS 优势超出千分之一并列线——更简核胜出,不为噪声级差异引入复杂度"
  - "m1 多起点判 inconsistent:RMS 极差 2.8e-07 px 几乎为零但参数位移 0.205 px 超出 0.1 px 容差——RMS 一致参数不一致 = 可辨识性问题的直接证据(D-15 双容差缺一不可的设计意图兑现)"
  - "两阶段并行派发替代单批派发:鲁棒核任务的 rms_linear_ref 依赖同模型 linear 结果,先收齐 linear 再构造鲁棒核任务"

patterns-established:
  - "optimizer.json 契约:8 键(schema_version/created/lineage/recommended_solver/combos/multistart/meta/failure_note)写盘前校验,失败轮结构键一个不少"
  - "推荐规则:只在收敛组合里挑 RMS 最小者,千分之一相对差内视为并列取更简核(linear>soft_l1>huber>cauchy)"
  - "集成试车体例:整文件 integration 标记 + 前置产物缺失整文件跳过 + 模块级夹具只跑一轮共享断言 + 中文摘要 print"

requirements-completed: [SOL-02]

# Metrics
duration: 3h43min
completed: 2026-10-19
---

# Phase 4 Plan 2: 优化器对比(16 组合 + 推荐配置 + 多起点)Summary

**16 组合网格在真实 640 视图上跑通并落盘 optimizer.json:四模型推荐均为 linear 核,m1 的 soft_l1/cauchy 未过实用收敛(与历史轮对账一致),多起点检验抓出 m1 参数不一致(RMS 几乎不动而位移 0.205 px)——D-13「优化器先行」兑现,04-03 起的选型实验有了数据支撑的统一优化配置。**

## Performance

- **Duration:** 3h43min(其中真实数据集成试车三跑:第一次被工具超时中断、第二次暴露落盘契约缺口、第三次 47min 全绿)
- **Started:** 2026-10-19T08:04(+0800,任务 1 首提交)
- **Completed:** 2026-10-19T11:32 UTC
- **Tasks:** 4
- **Files modified:** 4(3 新建 + 1 修改)

## Accomplishments
- **16 组合网格(run_optimizer_grid)**:每模型从落盘产物 mX__opencv 读回同一份 OpenCV 解当初值(Pitfall 8 锁死归因),linear 先行、鲁棒核带 rms_linear_ref 参照,逐行记录 init_rms / 最终 RMS / solver_status / 实际 f_scale(Pitfall 4/8 全落账);单组合意外转失败契约占行不中断网格
- **推荐配置(recommend_optimization_config)**:只在收敛组合里挑 RMS 最小者,千分之一并列取更简核;真实数据实测四模型全部推荐 linear
- **多起点一致性(run_multistart)**:扰动幅度表(内参 ±2%、畸变 ±10%、旋转 ±1°、平移 ±2%)配置驱动,双容差(RMS 极差 <0.001 px 且参数位移当量 <0.1 px)判定,坏解进 bad_solutions 不静默;m4 鱼眼 N 翻倍(20 起点);种子确定性派生,同配置重跑逐位一致
- **optimizer.json 契约(artifacts.py)**:8 键写盘前校验 + 失败契约工厂 + 明细 npz 旁车(allow_pickle=False)+ 血缘(照 calib fisher 的 upstream 登记体例)
- **真实数据首轮落盘**:output/evaluate/run_20261019_112926/optimizer.json,640 视图,m1×soft_l1 与历史轮 run_20261018_163713 对账显式记录(meta.history_reference_run)

## Task Commits

Each task was committed atomically:

1. **Task 1: optimizer.json 契约骨架** - `ef0ad06` (feat)
2. **Task 2: 16 组合优化器网格 + 推荐优化配置(D-13/D-14)** - `078abb5` (feat)
3. **Task 3: 多起点扰动一致性检验(D-15)** - `5f0962e` (feat)
4. **Task 4: 真实数据 16 组合 + 多起点试车与已知事实对账(集成)** - `858de5a` (test,含 Rule 1 契约缺口修复)

## Files Created/Modified
- `src/calib/evaluation/optimizer.py` - run_optimizer_grid / recommend_optimization_config / run_multistart / 扰动与判定私有件 + worker 层失败契约兜底
- `src/calib/evaluation/artifacts.py` - OPTIMIZER_KEYS 8 键契约 / make_failure_optimizer / save_optimizer_report / load_optimizer_report
- `tests/test_optimizer_comparison.py` - 19 例:契约 roundtrip + 小网格行完整 + rms_linear_ref 传递 + 失败契约占行 + 推荐规则 + 种子确定性 + 双容差判定 + 鱼眼翻倍 + 幅度配置
- `tests/test_optimizer_realdata.py` - 6 例集成:契约与 16 行 / 推荐白名单 / m1-softl1 对账 / 血缘文件存在 / 多起点完整 / 不覆盖旧轮

## Decisions Made
- **run_multistart 取 CalibrationConfig**:计划文本签名写 eval_cfg,但动作要 cfg.seed 派生种子;EvaluateConfig 无 seed 键,照 04-01 既定决策(要种子的函数取全量配置)落为 cfg: CalibrationConfig
- **两阶段并行派发**:鲁棒核任务的 rms_linear_ref 依赖同模型 linear 结果,先跑完 linear 再构造鲁棒核任务(保持并行,只是分两批)
- **worker 层异常兜底(Rule 1)**:串行回退时 worker 异常发生在 run_parallel 生成器体内会杀死整个任务流(后续组合/起点静默丢失);_solve_combo / _solve_multistart_start 函数体内 try/except 转 make_failure,消费层兜底保留为双保险
- **cv2.setNumThreads(cfg.cv2_threads) 进 run_multistart**:D-12 配置驱动之外还换来基线求解逐位确定(多线程归约顺序不定会末位漂移,同种子重跑逐位一致守不住)
- **网格返回值直接组齐契约键(Rule 1)**:created / failure_note 由 run_optimizer_grid 给出;lineage 是唯一例外(调用方经 make_lineage 组装,save_optimizer_report 落位)——第一次真实数据轮在落盘校验处暴露此缺口,单测补了契约回归断言

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] worker 异常杀死 run_parallel 生成器,后续任务静默丢失**
- **Found during:** Task 3(测试 test_multistart_start_failure_recorded_not_fatal 揭示)
- **Issue:** 串行回退时 `yield worker(task)` 在生成器体内执行,worker 抛异常即终止生成器,后续起点/组合全部静默丢失
- **Fix:** worker 函数体内 try/except 转 make_failure 失败契约(异常类型与 solve 层同口径:ValueError/ArithmeticError/RuntimeError),异常到不了生成器
- **Files modified:** src/calib/evaluation/optimizer.py
- **Commit:** 5f0962e

**2. [Rule 1 - Bug] run_optimizer_grid 返回值缺 created / failure_note 契约键**
- **Found during:** Task 4(第一次真实数据轮在 save_optimizer_report 落盘校验处被拦,55 分钟计算作废)
- **Issue:** 单测 roundtrip 用手搓完整报告,网格输出直接落盘的路径无人覆盖
- **Fix:** 网格返回值补齐两键(正常返回即 success 轮);单测补「8 键中除 lineage 外全在」回归断言
- **Files modified:** src/calib/evaluation/optimizer.py, tests/test_optimizer_comparison.py
- **Commit:** 858de5a

**3. [Rule 3 - Blocking] cv2.setNumThreads 缺位导致多起点重跑不逐位一致**
- **Found during:** Task 3(种子确定性测试以 ~1e-13 位级漂移失败)
- **Issue:** OpenCV 多线程归约顺序不定,基线求解末位漂移破坏「同种子重跑逐位一致」承诺
- **Fix:** run_multistart 顶部锁线程数(D-12 配置驱动 + 确定性双重理由)
- **Files modified:** src/calib/evaluation/optimizer.py
- **Commit:** 5f0962e

### 签名偏差(计划内文与实现取其一)
- run_multistart(obs, board, **cfg: CalibrationConfig**)——计划文本 eval_cfg,按 04-01 决策取全量配置(见 Decisions Made 首条)

## Issues Encountered
- 真实数据集成试车单轮 47~55 分钟(16 组合 + 50 次多起点求解,640 视图 × 11 万残差),远超计划「分钟级」预估;第一次后台跑被工具超时中断,改用脱离会话方式重启后通过。后续 04-04/04-05 的更大规模实验需按小时级预估工时。
- 无阻塞项。

## Known Stubs
None——所有数据路径都已接线,optimizer.json 的 16 行、recommended_solver、multistart 全部来自真实计算。

## TDD Gate Compliance

本计划 frontmatter 无 type: tdd,任务均以 auto 直实现(实现与测试同提交);测试先行仅体现为单测与实现同步落地,无独立 RED 提交,无门禁违规。

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- **04-03(选型统计)可直接开工**:统一优化配置 = 四模型均 linear 核(D-13 兑现);optimizer.json 读数经 load_optimizer_report(allow_pickle=False)
- **给 04-04/04-05 的输入**:m1 多起点 inconsistent(参数位移 0.205 px)是选型阶段「参数稳定性」判据的直接证据材料;m1 的 soft_l1/cauchy status=0 写进 SOL-02 结论材料(Pitfall 5)
- **工时预估修正**:640 视图 × 一次 scipy 求解在 8 并行下实测 ~1.5-3 分钟;按求解次数 × 3 分钟估Wall time,别按计划草案的「秒级」
- 无阻塞项

## Self-Check: PASSED

- 3 个新建文件存在(optimizer.py / artifacts.py / test_optimizer_realdata.py)
- 4 个任务提交在案(ef0ad06 / 078abb5 / 5f0962e / 858de5a)
- 验证证据:tests/test_optimizer_comparison.py 19/19 绿;tests/test_optimizer_realdata.py 6/6 绿(真实数据,47min);全套 396 passed(-m "not integration");产物 output/evaluate/run_20261019_112926/optimizer.json 契约键全、16 行齐、血缘上游 15 个文件全在

---
*Phase: 04-evaluation-model-selection*
*Completed: 2026-10-19*
