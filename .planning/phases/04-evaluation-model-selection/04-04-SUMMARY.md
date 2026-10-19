---
phase: 04-evaluation-model-selection
plan: 04
subsystem: evaluation
tags: [experiments, cross-validation, lopo, slice-groups, bootstrap, variance-source-separation, determinism]

# Dependency graph
requires:
  - phase: 04-evaluation-model-selection (04-01)
    provides: splitting 纯函数(assign_cv_folds/leave_one_pose_out/slice_groups_by_index/bootstrap_resample/split_by_pose_ids/obs_rows/pose_row_mask)、run_parallel 并行助手、CalibrationConfig(evaluate 25 键 + seed)
  - phase: 04-evaluation-model-selection (04-02)
    provides: recommended_solver 消费契约(optimizer.json → optimizer_config 参数,四模型均 linear)+ cv2 线程锁教训(04-02)
  - phase: 04-evaluation-model-selection (04-03)
    provides: crlb_alert_check/mad_outlier_poses 判定数学、判行表消费面(cv_summary/stability_summary/crlb_summary 键名)
  - phase: 02-m1-m4 + 03-fisher-crlb
    provides: 模型注册表与投影数学、marginalized_fisher/crlb_se(同口径 CRLB 每组新算)
provides:
  - cv_pose_grouped(schema cv-v1):5×5 折姿态分组交叉验证,折级样本逐折对齐同轮切分,验证侧只估位姿评分(第二层防泄漏),fold_heldout/fold_train + per_pose_rows
  - lopo(schema lopo-v1):留一姿态逐姿态表 + MAD 稳健 z 全表(复用 selection 数学)+ 配置阈值点名问题姿态
  - slices_by_index(schema slices-v1):重复张按姿态分组,每组新算同口径 CRLB(组自身收敛解为线性化点 + temporal σ)+ crlb_alert_check 硬告警接线 + slice_max_disp_px
  - bootstrap_pose(schema bootstrap-v1):按姿态有放回重采样,4 模型共用同一批分配(D-04),去重带全连拍,param_std/param_ci/var_bootstrap + 逐次参数向量明细
  - pose_sampling_variance(D-07 来源分离纯函数):var_boot − var_slice 逐参数相减,负值钳 0 带中文说明
  - 共享胶水:make_calibrate_fn(OpenCV 初始化 → scipy 精化,必带 rms_linear_ref)、_holdout_pose_errors(姿态级平均聚合)、_px_dispersion(像素位移当量)
  - 确定性契约进程级补全:四个 worker 各自上 cv2 线程锁(并行子进程与串行逐位一致)
affects: [04-05-artifacts-cli, 04-06-refit-test-scoring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "worker 级线程锁:并行子进程是全新进程,父进程入口的 cv2.setNumThreads 传不过去——线程数随任务元组带入,worker 入口再上一次锁,否则并行与串行末位漂移 1e-9 级(04-02 教训的进程级补全)"
    - "失败契约双层保险:worker 内 try/except 转失败占行 + 消费循环宽接兜底(串行回退时 worker 异常会杀死 run_parallel 生成器、静默吞掉后续任务,体例照 optimizer.py)"
    - "同口径对照:T-04-04-01 执行位——分组实验的 CRLB 每组新算(σ 只认 noise.json 的 sigma_temporal_corner,缺键中文报错),poses80 聚合口径标识在 experiments.py 源码级禁入(测试扫描锁死)"
    - "确定性随机性只从任务元组走:分配表先产再派任务(bootstrap 一批分配全模型共用,D-04),并行不改变任何随机性"

key-files:
  created:
    - src/calib/evaluation/experiments.py
    - tests/test_evaluation_experiments.py
  modified: []

key-decisions:
  - "实验函数签名统一为 fn(obs, board, cfg: CalibrationConfig, optimizer_config, progress=None):计划草稿 (models, scipy_cfg, eval_cfg, optimizer_config) 落为全量配置体例(要 cfg.seed 派生随机性、cfg.evaluate 读开关阈值、cfg.calibrate.scipy 读精化参数),照 04-01 切分函数/04-02 run_multistart 既定体例"
  - "bootstrap 去重语义:同一姿态抽中多次只算一份(bootstrap 考的是'换一批姿态',重复计权会人为放大被偏爱姿态的话语权);抽中即带全部连拍(切分只操作姿态 ID 的铁律消费位)"
  - "lopo 的 mad_z 用 threshold=-1.0 取全表(z 恒非负)复用 selection.mad_outlier_poses 的同一套 MAD 数学不写第二份实现;真正的点名按 cfg.evaluate.mad_z_threshold 再过一遍"
  - "var_slice/pose_sampling_var 是占位键:计划文本'返回 pose_sampling_var 差值'在分组侧方差缺席时不可算——落为 None + var_slice_note 说明(接口在此定死,数值由 04-05/04-06 组装侧填入后经 pose_sampling_variance 相减)"
  - "行为 3 并行测试不 monkeypatch(子进程拿不到进程内替身,考的就是真实并行路径);max_workers 是运行方式元数据不是数值结果,各自断言记对后抹平再比全量"
  - "任务按 tdd=true 走双提交:三个任务各 RED(ImportError/ModuleNotFoundError 实证)+ GREEN,共 6 提交"

patterns-established:
  - "四实验统一消费面:optimizer_config 形如 {'mX': {'loss': 'linear', 'f_scale_px'?: float}}(来自 optimizer.json recommended_solver,读取由调用方负责);条目缺失/核不在白名单中文报错,绝不静默回退默认配置(D-13)"
  - "observation_set 口径标识随记录落盘:f'views{n_poses}_single_frame'(正式口径 80 姿态即 views80_single_frame)+ sigma_source='temporal',下游不可能拿错口径对账"
  - "失败不是崩溃:单次求解失败转失败契约占行(m3/m4 小训练集不收敛是数据现象),失败率进汇总、逐条带异常类型中文说明"
  - "明细数组挂 _detail 私有键交产物层:折分配表/逐姿态 RMS/逐次参数向量(失败记 NaN 行)全保留,JSON 消费面只见汇总键(产物保真)"

requirements-completed: [EVA-01, EVA-03, EVA-04, EVA-05]

# Metrics
duration: 58min
completed: 2026-10-19
---

# Phase 4 Plan 4: 四类防作弊实验编排(交叉验证 + 留一姿态 + 重复张分组 + bootstrap)Summary

**四个实验函数在 experiments.py 全部落地:同一输入必得同一输出的纯编排——CV/LOPO 量"换一批姿态还准不准",分组/bootstrap 量"参数稳不稳"且两者之差刻画位姿抽样贡献(D-07);全部走 OpenCV 初始化 → scipy 精化(推荐配置)同一条求解路径,验证侧外参绝不训练,失败转契约不中断;并行子进程线程锁补全后与串行逐位一致,EVA-01/03/04/05 实验侧闭环。**

## Performance

- **Duration:** 58 min
- **Started:** 2026-10-19T12:06:28Z
- **Completed:** 2026-10-19T13:04:01Z
- **Tasks:** 3(均按 tdd=true 拆 RED/GREEN 双提交,共 6 提交)
- **Files modified:** 2(2 新建)

## Accomplishments
- **cv_pose_grouped(D-01/D-02,EVA-01)**:按姿态分组 K 折(默认 5 折 × 5 轮),折级样本按分配表顺序排好供 04-03 同折配对比较;验证误差按"先姿态级平均再进折级"聚合(不按角点数加权——同一姿态的连拍近似同一张照片,按角点数加权等于重复计票);验证姿态一律 solve_pose_only 只估位姿评分(防"第二层泄漏"执行位),同种子重跑逐位一致
- **lopo(D-08,EVA-05 实验侧)**:逐姿态留一训练、考题姿态只估位姿打分;MAD 稳健 z 全表(复用 selection 数学)+ 配置阈值点名问题姿态;lopo_enabled 关闭中文报错拒绝跑
- **slices_by_index(D-05/D-06,EVA-03 实验侧)**:重复张按"每姿态第 i 张"分组,组数 = 最少重复张数(数据驱动);每组独立完整标定 + **每组新算同口径 CRLB**(线性化点 = 该组自身收敛解,σ 只认 noise.json 的时域口径,缺键当场中文报错);经验 std 对卡方双告警线走 crlb_alert_check(df = 组数−1 自适应,FAIL 说明带参数名不静默);observation_set/sigma_source 口径标识随记录落盘;poses80 聚合口径源码级禁入(T-04-04-01)
- **bootstrap_pose(D-04,EVA-04 实验侧)**:分配表只产一次、4 模型共用(跨模型配对比较方差,差值才是"模型之间的差"而不是"考题难度的差");抽中姿态去重带全部连拍;param_std/param_ci/var_bootstrap(方差,供相减)/max_disp_px(像素位移当量,键名对齐判行表 stability_summary);逐次参数向量挂明细数组全保留
- **pose_sampling_variance(D-07)**:逐参数 var_boot − var_slice(**方差**相减,非标准差——独立来源的方差才可加);负值钳 0 带白话说明不静默;docstring 讲清"分组固定姿态只换噪声,bootstrap 连姿态都在换,差值就是换姿态本身的不确定度"
- **共享胶水**:make_calibrate_fn(两步求解,精化必带 rms_linear_ref = OpenCV 初值回代 RMS,Anti-Pattern 5;f_scale 只在推荐配置显式给出时覆盖 auto 档)、_px_dispersion(参数差换算像素位移当量——相对百分比对近零畸变参数无意义)、_recommended_kernel(推荐配置缺条目/核越界中文报错,不静默回退)
- **确定性契约进程级补全(Rule 1)**:并行子进程是全新进程,入口处 cv2.setNumThreads 传不过去——四个 worker 各自在入口再上一次线程锁(线程数随任务元组带入),并行与串行从 1e-9 级末位漂移修为逐位一致
- **测试 18 例全绿**(全套 446 passed):每行为固定种子小样合成数据锁死,零重叠/去重行数/像素当量/来源分离差值均手算对账

## Task Commits

Each task was committed atomically:

1. **Task 1: cv_pose_grouped 失败测试(RED)** - `92a0163` (test)
2. **Task 1: 求解胶水 + K 折交叉验证 cv_pose_grouped(D-01/D-02,EVA-01)** - `8a6c734` (feat)
3. **Task 2: lopo 与重复张分组失败测试(RED)** - `13f657c` (test)
4. **Task 2: 留一姿态 lopo + 重复张分组与同口径 CRLB 硬告警(D-05/06/08)** - `666bcaf` (feat)
5. **Task 3: bootstrap 重采样与来源分离方差接口的失败测试(RED)** - `5314f39` (test)
6. **Task 3: bootstrap_pose 重采样 + 噪声/位姿抽样来源分离(D-04/D-07)** - `e90131f` (feat)

_Note: 三任务均 tdd="true",按协议各拆 test(RED)→feat(GREEN)双提交_

## Files Created/Modified
- `src/calib/evaluation/experiments.py` - 四实验函数(cv_pose_grouped/lopo/slices_by_index/bootstrap_pose)+ 来源分离纯函数 pose_sampling_variance + 共享胶水(make_calibrate_fn/_holdout_pose_errors/_px_dispersion/_recommended_kernel/_check_obs_board/_sigma_temporal_from_noise)+ 四 worker + 失败契约记录构造器;模块 docstring 写明铁律/失败契约/并行线程锁/落盘边界
- `tests/test_evaluation_experiments.py` - 18 例:CV 五行为(折样本与零交集/姿态级聚合手算/验证外参不训练计数断言/确定性/失败契约)、胶水两例(rms_linear_ref 携带 + f_scale 覆盖、像素当量手算)、lopo 三例、slices 四例(含 poses80 源码禁令 + 硬告警接线)、bootstrap 四例(含真实并行路径)与来源分离一例

## Decisions Made
- **实验函数签名取全量 CalibrationConfig**:计划 `<action>` 草稿的 (models, scipy_cfg, eval_cfg, optimizer_config) 落为 fn(obs, board, cfg, optimizer_config, progress=None)——实验要 cfg.seed 派生随机性(04-01/04-02 既定体例),散传配置既丢种子也和前两个 plan 的消费面不一致
- **bootstrap 分配表共用 + 去重语义落 worker 层**:D-04 的"共用"由"先产分配再派任务"结构保证(任务元组带姿态表,天然可复查);去重(抽中多次只算一份)放 _bootstrap_worker,与"带全部连拍"同处一站
- **lopo mad_z 全表走 threshold=-1.0 复用**:z 分数恒非负,-1 阈值返回全表——selection 的 MAD 数学是唯一实现,点名只是再过一遍配置阈值
- **来源分离接口先行、数值后接**:bootstrap_pose 返回值带 var_slice/pose_sampling_var 占位键(计划行为 4 原文"由 04-05/04-06 侧组装时填入"),var_slice_note 键说明缺席原因,不静默冒充
- ** poses80 聚合口径禁令做成源码扫描测试**:experiments.py 全文不得出现该标识(含注释)——聚合口径的 σ/√8 折减记录一旦被分组实验引用,对照直接失真,禁令要防的就是"顺手复用"
- **TRY004 定向豁免**:推荐配置条目缺失用中文 ValueError(项目统一体例照 io/config.py),不按类型换 TypeError

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 并行子进程缺 OpenCV 线程锁,并行与串行末位漂移 1e-9 级**
- **Found during:** Task 3(GREEN 阶段 test_bootstrap_parallel_matches_serial 失败,探针定位到数值差异)
- **Issue:** loky 并行子进程是全新进程,实验入口(父进程)的 cv2.setNumThreads 传不过去;子进程多线程归约顺序不定,param_std 等出现 1e-9 级漂移——违反模块"同种子重跑逐位一致"契约,四个实验函数全部受累
- **Fix:** 线程数随任务元组带入,四个 worker(cv 折/lopo/分组/bootstrap)入口各自再上一次锁;模块 docstring 写明"入口和 worker 各上一次"的完整口径
- **Files modified:** src/calib/evaluation/experiments.py
- **Commit:** e90131f

**2. [Rule 1 - 测试自身缺陷] test_px_dispersion 期望 0.125,实为 0.25**
- **Found during:** Task 1(GREEN 阶段)
- **Issue:** 期望值按"网格居中"心算,实际 build_object_points 网格从原点排开(不居中),最远角 x=200mm/深度 800 → 归一化 0.25,Δfx=1px 最大位移 0.25px——实现正确,测试期望错
- **Fix:** 断言改 0.25 并注释核算过程
- **Commit:** 8a6c734

**3. [Rule 1 - 测试自身缺陷] "同一张照片被重复打分"断言与 D-01 语义冲突**
- **Found during:** Task 1(GREEN 阶段)
- **Issue:** 每姿态每轮进验证一次,12 张照片被评 2×4×3=24 次是合法行为(轮间重复是配对比较的前提);原断言"48 次调用全不同"写反了
- **Fix:** 改为 set(pnp_calls) == 全部 12 张视图字节键(多了 = 训练侧混入,少了 = 漏评)
- **Commit:** 8a6c734

**4. [Rule 1 - 测试自身缺陷] _assert_reports_equal 对字符串数组用 equal_nan 崩溃**
- **Found during:** Task 1(GREEN 阶段)
- **Issue:** np.array_equal(..., equal_nan=True) 不支持 <U dtype,直接 TypeError
- **Fix:** 按 dtype.kind 分派——浮点数组才用 equal_nan,字符串/整数数组平比较
- **Commit:** 8a6c734

**5. [Rule 1 - 测试自身缺陷] test_bootstrap_records m2 的参数名集合误用 m1(m2 是 9 参)**
- **Found during:** Task 3(GREEN 阶段)
- **Issue:** names = create_model("m1").param_names 提到双模型循环外,m2 有 9 参对不上
- **Fix:** names 按模型在循环内取
- **Commit:** e90131f

其余:计划 <output> 草案 3 条提交、实际 6 条,属 tdd="true" 协议规定动作(RED/GREEN 双提交 × 3),非偏差。

## Issues Encountered
None。三个 RED 门禁均以正确原因失败(Task 1 ModuleNotFoundError: calib.evaluation.experiments;Task 2/3 ImportError: cannot import name 'lopo'/'bootstrap_pose');GREEN 阶段的失败全部定位为测试自身缺陷(见 Deviations)或上记 Rule 1 真实 bug,修复后一次转绿。

## Known Stubs

| Stub | 位置 | 原因与接管方 |
|------|------|--------------|
| models[mX]["var_slice"] = None | src/calib/evaluation/experiments.py bootstrap_pose 返回值 | 计划行为 4 明确要求占位:分组侧方差需 04-05/04-06 组装侧把 slices_by_index 的 empirical_std 平方后填入,再经 pose_sampling_variance 相减;var_slice_note 键已随行说明缺席原因,不静默 |
| models[mX]["pose_sampling_var"] = None | 同上 | 同上(var_slice 就位后相减即得) |

其余全实现:四实验函数 + 来源分离纯函数 + 六件胶水无占位、无 TODO;joblib 并行 + 串行回退、进度回调挂点均已接。

## TDD Gate Compliance

- RED 门禁 ×3:`92a0163`(ModuleNotFoundError: calib.evaluation.experiments)、`13f657c`(ImportError: cannot import name 'lopo')、`5314f39`(ImportError: cannot import name 'bootstrap_pose')——失败原因均正确
- GREEN 门禁 ×3:`8a6c734`(7/7)、`666bcaf`(14/14)、`e90131f`(18/18);全套 446 passed(-m "not integration")
- REFACTOR:无需要,实现一次成型;ruff format 整批套用不属重构

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- **04-05(产物与命令行)消费面已定死**:四实验返回值统一 {"schema", "models", "meta", "_X_detail"};私有 _detail 键(字符串/整数数组 + NaN 填充浮点数组)交产物层转 npz 旁车,JSON 只落汇总键
- cv_summary/stability_summary/crlb_summary 的判行表键名已对齐 04-03 build_verdict_table 输入形状:cv 用 mean_heldout_px/fold_heldout(25 个,逐折对齐),stability 用 slice_max_disp_px 与 bootstrap max_disp_px,crlb 用 crlb_alert_check 返回值
- 真实数据工时按 04-02 实测(单次 640 视图 scipy 求解 1.5–3 分钟、8 并行):CV 100 次 + bootstrap 800 次 + 分组 8×4 + LOPO 320 次 ≈ 1300 次求解,建议先跑 CV 校准耗时再放大批量(模块 docstring 已注)
- var_slice 组装口径:empirical_std 按参数平方成方差后填 var_slice,pose_sampling_variance(var_bootstrap, var_slice) 相减,负值钳 0 的 note 会逐参数点名
- 无阻塞项

---
*Phase: 04-evaluation-model-selection*
*Completed: 2026-10-19*

## Self-Check: PASSED

- 2 个新建文件存在(src/calib/evaluation/experiments.py / tests/test_evaluation_experiments.py)
- 6 个任务提交在案(92a0163/8a6c734、13f657c/666bcaf、5314f39/e90131f,RED→GREEN 成对)
- 验证证据:tests/test_evaluation_experiments.py 18/18 绿;全套 446 passed(-m "not integration");ruff check + format 干净
