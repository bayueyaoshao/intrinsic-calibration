---
phase: 04-evaluation-model-selection
plan: 01
subsystem: evaluation
tags: [cross-validation, bootstrap, data-splitting, joblib, config-contract, anti-leakage]

# Dependency graph
requires:
  - phase: 02-m1-m4
    provides: 模型注册表(m1–m4 短名白名单)、ObservationSet.pose_ids 切分连接键
  - phase: 03-fisher-crlb
    provides: seed 种子派生体例(_seed_for sha1 三元组)、config 解析校验体例(FisherConfig/_parse_fisher)、injection 并行串行回退体例
provides:
  - EvaluateConfig 25 键冻结契约(04-02~04-06 只消费不改)
  - config/calibration.yaml evaluate: 段(每键中文注释 + 决策号)
  - splitting.py 四种切分纯函数(分折/bootstrap 重采样/取第 i 张分组/留一)+ 零重叠硬门禁
  - parallel.py run_parallel 任务级并行助手(loky 常驻 + 串行回退同序同果)
  - joblib>=1.4 入锁(实测解析 1.6.0)
affects: [04-02-optimizer, 04-03-selection, 04-04-experiments, 04-05-experiments, 04-06-refit-test-scoring]

# Tech tracking
tech-stack:
  added: [joblib>=1.4 (1.6.0)]
  patterns:
    - "种子标签派生:_seed_for(标签 → sha1 mod 2^32 + base_seed),全阶段随机性唯一来源 cfg.seed=42"
    - "零重叠硬门禁:函数体断言 + 单元测试双保险(姿态级与行级 train∩val=∅)"
    - "run_parallel:生成器逐个消费 + progress 回调由 CLI 注入,库层不碰命令行框架"

key-files:
  created:
    - src/calib/evaluation/splitting.py
    - src/calib/evaluation/parallel.py
    - tests/test_evaluation_split.py
  modified:
    - src/calib/io/config.py
    - config/calibration.yaml
    - pyproject.toml
    - uv.lock
    - src/calib/evaluation/__init__.py
    - tests/test_io_config.py

key-decisions:
  - "切分函数签名取整个 CalibrationConfig(读 cfg.seed + cfg.evaluate.*):照 run_injection_experiment 既有 cfg 体例,计划文本 cfg.seed 由此落实"
  - "evaluate 核名白名单独立于 calibrate.scipy 的 _ALLOWED_LOSSES(多 cauchy):D-14 四核矩阵只属阶段四,主线 calibrate 段不跟进"
  - "evaluate_max_workers 校验只要求 >=1 不卡核数上限:并行调度交 joblib,与 injection_max_workers 卡核数体例刻意区分(计划明确规定)"
  - "slice_groups_by_index 在 slice_enabled=false 时中文报错:cfg 参数落到实际职责,防配置关闭后误调用"
  - "任务 2 按 tdd=true 走双提交:RED 失败测试先行(ModuleNotFoundError 实证)+ GREEN 实现"

patterns-established:
  - "切分铁律:一切切分只操作姿态 ID 列表,同姿态连拍同进同出;分配表纯 str/int 可 json 落盘复查"
  - "考试出题与消费分离:splitting 只出姿态级分配表,行落位/标定/评分由 experiments 层消费"
  - "配置段扩充体例:EvaluateConfig + _parse_evaluate(默认从 cfg 实例取、白名单、区间/有序/布尔校验)+ yaml 段每键决策号注释"

requirements-completed: [EVA-01]

# Metrics
duration: 12min
completed: 2026-10-19
---

# Phase 4 Plan 1: 评估地基(配置契约 + 姿态切分 + 并行助手)Summary

**EvaluateConfig 25 键契约一次定死 + 四种姿态切分纯函数(分折/bootstrap/取第 i 张分组/留一)落地,零重叠防泄漏断言以函数体 + 测试双保险生效,joblib 并行助手串行/并行同序同果——04-02~04-06 全部实验计划的公共地基。**

## Performance

- **Duration:** 12 min
- **Started:** 2026-10-19T07:33:15Z
- **Completed:** 2026-10-19T07:45:35Z
- **Tasks:** 2(任务 2 按协议拆 RED/GREEN 两提交)
- **Files modified:** 9(3 新建 + 6 修改)

## Accomplishments
- EvaluateConfig 全量 25 键 + _parse_evaluate 逐键校验(模型/核白名单、计数下限、alpha 有序、布尔拒 1/0),yaml evaluate 段每键带中文注释与决策号;非法配置全部中文报错
- splitting.py 四种切分纯函数:同种子逐位可复现(5 折 × 5 轮每姿态恰 5 次进验证实证),分配表纯 str/int 可 json 落盘复查(D-03)
- 防泄漏硬门禁(ROADMAP 成功标准 1):split_by_pose_ids 函数体断言 + 16 例单元测试对 80 姿态 × 25 折显式断言姿态级与行级零交集
- parallel.py run_parallel:max_workers≤1 串行回退与并行同序同结果(测试断言),loky 生成器逐个消费驱动进度回调
- joblib>=1.4 锁入(1.6.0,纯 Python 轮子),供应链审计沿用 04-RESEARCH 的 [OK] 判定

## Task Commits

Each task was committed atomically:

1. **Task 1: joblib 依赖 + EvaluateConfig 全量契约 + evaluate 配置段 + parallel 并行助手** - `9531d43` (feat)
2. **Task 2: splitting 切分函数失败测试(RED)** - `2c5ade7` (test)
3. **Task 2: splitting 姿态切分纯函数库 + 防泄漏硬门禁测试(GREEN)** - `9f70707` (feat)

_Note: TDD tasks may have multiple commits (test → feat → refactor)_

## Files Created/Modified
- `src/calib/evaluation/splitting.py` - 四种切分纯函数 + unique_poses/pose_row_mask/obs_rows 辅助件 + _seed_for 种子派生
- `src/calib/evaluation/parallel.py` - run_parallel 生成器(joblib 生成器进度 + 串行回退)
- `tests/test_evaluation_split.py` - 行为 1–6 用例 + 防泄漏硬门禁(16 例)
- `src/calib/io/config.py` - EvaluateConfig(25 键)+ _parse_evaluate + CalibrationConfig.evaluate 装配
- `config/calibration.yaml` - evaluate: 段(每键中文注释 + 决策号;注明 seed 42 唯一来源与 1.0e+4 指数写法)
- `pyproject.toml` / `uv.lock` - joblib>=1.4 入锁(带中文锁定理由注释)
- `src/calib/evaluation/__init__.py` - docstring 指路 splitting/parallel
- `tests/test_io_config.py` - evaluate 解析 11 例(默认值锚定 + 7 类非法中文报错)+ run_parallel 串行/并行同序同果

## Decisions Made
- **切分函数取整个 CalibrationConfig**:计划文本写 `assign_cv_folds(pose_ids, cfg)` 且读 `cfg.seed`;EvaluateConfig 无 seed 键,故 cfg 落为 CalibrationConfig(读 cfg.seed + cfg.evaluate.*),与 run_injection_experiment(cfg: CalibrationConfig) 既有体例一致
- **evaluate 核名白名单独立建表**(_ALLOWED_EVAL_LOSSES 含 cauchy):D-14 四核矩阵属阶段四,calibrate.scipy 主线名单保持 {linear, soft_l1, huber} 不动
- **evaluate_max_workers ≥ 1 不卡核数上限**:计划校验规则明确规定;joblib 自行调度,注释写明"不必卡着本机核数写"
- **slice_groups_by_index 检查 slice_enabled 开关**:计划签名含 cfg,给其实际职责(配置关闭即中文报错),其余切分函数不查开关(由 experiments 层消费时判断)
- **组内"每姿态至多 1 张"断言落为行号唯一性断言**:pose_rows 是字典(姿态→行号),键唯一由结构保证;行号唯一才能抓"排序/取位被改坏"的真实事故

## Deviations from Plan

None - plan executed exactly as written.(任务 2 的 test/feat 双提交是 tdd="true" 协议的规定动作,计划 <output> 草案预告 2 条提交、实际 3 条,属协议内。)

## Issues Encountered
None。(RED 阶段按预期以 ModuleNotFoundError 失败,GREEN 一次通过 16/16。)

## TDD Gate Compliance

- RED 门禁:`2c5ade7` 提交失败测试,运行实证失败(ModuleNotFoundError: calib.evaluation.splitting)——失败原因正确(模块未实现),非测试写错
- GREEN 门禁:`9f70707` 实现后 16/16 通过;全套 377 passed(-m "not integration")
- REFACTOR:无需要,实现一次成型

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- 04-02(优化器对比)/ 04-03(选型统计)可直接消费:EvaluateConfig 25 键、run_parallel、_seed_for 体例
- splitting 只出姿态级分配表;把分配落到观测集行、折内标定与评分在 04-04/04-05 的 experiments 层完成(消费 split_by_pose_ids + pose_row_mask + obs_rows)
- 无阻塞项;注意 evaluate_max_workers 默认 8,真实 1300 次求解的资源预估在 04-02/04-04 编排层核对

---
*Phase: 04-evaluation-model-selection*
*Completed: 2026-10-19*

## Self-Check: PASSED

- 3 个新建文件存在(splitting.py / parallel.py / test_evaluation_split.py)
- 3 个任务提交在案(9531d43 feat / 2c5ade7 test RED / 9f70707 feat GREEN)
- 验证证据:tests/test_evaluation_split.py 16/16 绿;tests/test_io_config.py 53/53 绿;全套 377 passed(-m "not integration")
