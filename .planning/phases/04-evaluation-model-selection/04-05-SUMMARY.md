---
phase: 04-evaluation-model-selection
plan: 05
subsystem: evaluation + visualization
tags: [delivery, evaluate-contract, npz-sidecar, lineage, figures, smoke-test, bic]

# Dependency graph
requires:
  - phase: 04-evaluation-model-selection (04-02)
    provides: optimizer.json 契约体例(OPTIMIZER_KEYS/_to_jsonable/make_failure_optimizer)与 recommended_solver 消费契约(D-13:四实验与交付统一用推荐配置)
  - phase: 04-evaluation-model-selection (04-03)
    provides: bic_full_fit / build_verdict_table / select_model(判行表与选型规则,BIC 只消费现成值)
  - phase: 04-evaluation-model-selection (04-04)
    provides: 四实验函数(cv_pose_grouped/bootstrap_pose/slices_by_index/lopo,返回值统一 {"schema","models","meta","_X_detail"})、pose_sampling_variance、make_calibrate_fn/_px_dispersion 共享胶水
  - phase: 02 + 03
    provides: solve_pose_only(test 逐张评分零新数学)、load_observation_set 的 split="test" 路径与行序契约、compute_result_metrics(RSS 来源)、noise_figs.py 出图三件套体例
provides:
  - run_delivery(schema delivery-v1):全量重训 × 全部参选模型(统一推荐配置,D-13/CV 只管选型、交付一律全量重训)+ test 独立照片全模型并排逐张评分(D-16,逐张逐模型行全存)+ BIC 在编排内算好(rss 取自全量解残差平方和,k = 内参数 + 6 × 视图数,n = cfg.evaluate.bic_n_poses,D-10 先重训后选型)
  - evaluate.json 主契约:EVALUATE_KEYS 13 键(schema_version/created/lineage/optimizer/cv/bootstrap/slices/lopo/source_separation/selection/delivery/meta/failure_note,SCHEMA_VERSION_EVALUATE = "evaluate-v1"),写盘前契约校验、失败契约 make_failure_evaluate(结构键全、各实验段 None、中文说明强制非空)
  - save/load_evaluate_report:五份私有明细键(_cv_detail/_bootstrap_detail/_slice_detail/_lopo_detail/_delivery_detail)转 experiments/<段名>.npz 旁车 + 同名 .meta.json(血缘必带),json 只留 detail_npz 引用(汇总只是索引,D-13 产物保真);读回强制 allow_pickle=False,引用断裂中文报错
  - 判行表三图(visualization/evaluate_figs.py,Agg 离线出 PNG):plot_cv_errorbars(折级误差棒 95% CI)、plot_lopo_pose_errors(逐姿态散点 + 红三角点名 + MAD 离群判据线)、plot_dispersion_compare(bootstrap vs 分组离散并排条形,差值 = 换姿态贡献,D-07);库函数只管画图,图失败不拦主产物
  - 合成数据全链冒烟(tests/test_evaluation_smoke.py):优化器小网格 → 四实验 → 全量重训交付 → 来源分离组装 → 判行表与选型 → evaluate.json 落盘 → 三图,一条龙跑通(04-06 CLI 只是薄编排的证据)
affects: [04-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "实验/交付函数取全量 CalibrationConfig:run_delivery(obs_train, obs_test, board, cfg, optimizer_config)(models 从 cfg.evaluate 读,线程锁在入口上),照 04-01/02/04 既定体例"
    - "json 主契约 + npz 明细旁车:嵌套明细字典拍平成 模型__数组名(np.savez 只收平面键),标量成 0 维数组读回原样;每个 npz 配 .meta.json 血缘旁车(T-04-05-04)"
    - "组装顺序防线:明细键带了数组但对应段不是字典 → 写盘当场中文报错(明细旁车必须挂在实验段下,悬空明细读回的人找不到目录)"
    - "matplotlib Agg 三件套模块头照 noise_figs.py(Agg 先于 pyplot、Noto CJK 字体优先级、unicode_minus),函数形制 plot_<thing>(数据, out_path) -> Path"
    - "BIC 消费方向定死:run_delivery 内部用 bic_full_fit 算好写进交付记录,select_model 只读行内 bic 键——不写第二份 BIC 数学(D-10)"

key-files:
  created:
    - src/calib/visualization/evaluate_figs.py
    - tests/test_evaluation_smoke.py
  modified:
    - src/calib/evaluation/experiments.py
    - src/calib/evaluation/artifacts.py
    - tests/test_evaluation_experiments.py

key-decisions:
  - "run_delivery 签名收拢:计划文本的 (models, scipy_cfg, eval_cfg, optimizer_config) 散传落为全量 CalibrationConfig 体例(与 04-04 四实验函数同形),models 从 cfg.evaluate.models 读——一套配置对象贯穿评估阶段,不出现两种传参风格"
  - "BIC 在 run_delivery 编排内算好(RSS 取自全量解的全部重投影残差平方和),这正是'先重训、后选型'顺序的理由——04-03 的 select_model 只消费行内现成 bic 值,判行表组装只做落账(D-10 Pattern 4)"
  - "明细旁车私有键名定 _slice_detail(计划文本 _slices_detail 之误):与 04-04 experiments.py 返回键和段名 slices 对齐,组装侧零改名直通"
  - "来源分离组装的口径:分组侧经验标准差先平方成方差再交 pose_sampling_variance 相减(独立来源方差才可加,标准差不能直接减),组装结果回填 bootstrap 的 var_slice/pose_sampling_var 占位键(D-07);CRLB 对照缺席时判行表用 {'n_below_hard': 0, ...} 兜底并注明,不冒充数值"
  - "json 段只留汇总与引用:实验返回值里的行级大表(per_pose_rows/group_records/rows/test_rows)与私有明细键一律不进 json——逐行明细在 npz 数组里(json 是索引不是替代,D-13)"
  - "test 评分行契约补全 image_id 与 failure_note 两键(计划只点名 model_id/image_id/rms_px):逐行可对账 + 失败占行带中文说明,与全项目失败契约体例一致"

patterns-established:
  - "smoke 冒烟样板:04-06 CLI 的调用序列在测试里先跑通一遍(真实 load_observation_set 装载 + 推荐配置直通 + 组装顺序 + 契约落盘 + 出图),命令层只做取数与中文播报"
  - "全链冒烟带 300 秒时间上限:小样配置退化成正式口径(装配有误)当场显形"
  - "同种子重跑逐位一致的断言工具 _assert_reports_equal 按类型分派(数组 equal_nan、NaN 对称性),可在后续轮次复用"

requirements-completed: [EVA-02, SOL-02]

# Metrics
duration: 29min
completed: 2026-10-19
---

# Phase 4 Plan 5: 交付与契约层 Summary

**run_delivery 全量重训 × N + test 65 张全模型并排逐张评分(D-16),evaluate.json 13 键主契约 + experiments/*.npz 明细旁车(血缘必带),判行表三图与合成数据全链冒烟——EVA-02/SOL-02 收口,04-06 CLI 只剩薄编排。**

## Accomplishments

- **run_delivery(D-16/EVA-02 收口)**:每模型用推荐优化配置对全部训练视图重训(OpenCV 初值 → scipy 精化,与 04-02 网格同一接口),交付记录带 final_params/rms_px/rss/n_intrinsic_params/n_views/bic;test 独立照片逐张 solve_pose_only 评分(内参当尺子只挪位姿,零新数学),逐张逐模型行全存(产物保真);单模型重训失败转失败契约占行不中断其余模型,其 test 行记 rms_px=None + 中文说明
- **BIC 就位(D-10)**:RSS 取自全量解 compute_result_metrics 的全部重投影残差平方和,k = 内参数 + 6 × 视图数,n = cfg.evaluate.bic_n_poses——select_model 只消费现成值,不写第二份 BIC 数学
- **evaluate.json 主契约**:EVALUATE_KEYS 13 键写盘前校验(缺键当场中文报错)、失败契约 make_failure_evaluate(各实验段 None、failure_note 非空强制)、save → load 往返逐键还原;五份明细(折分配表/逐次 bootstrap 参数/逐姿态 LOPO 表/分组参数/test 评分行)转 experiments/<段名>.npz + .meta.json 血缘旁车,json 各段只留 detail_npz 引用;读回 allow_pickle=False,主 json 缺失与引用断裂均中文报错
- **判行表三图**:CV 误差棒(折级 95% 置信区间,标题注明每模型折级样本数)、LOPO 逐姿态散点(红三角点名问题姿态 + MAD 离群判据线,阈值优先取报告 meta 实际用值)、离散对比并排条形(bootstrap 总晃动 vs 分组纯噪声晃动,差值 = 换姿态本身的贡献 D-07);Agg 离线出 PNG,图只是可视化,数字结论以 evaluate.json 为准
- **全链冒烟**:合成 15 视角(12 训练 = 6 姿态 × 2 连拍 + 3 考试)小样,优化器小网格 → 四实验 → 全量重训交付 → 来源分离组装(var_slice 填充 + pose_sampling_variance 相减)→ 判行表 + BIC 落账 + select_model → evaluate.json 落盘 → 三图,31.95 秒跑通;另测同种子重跑 cv/bootstrap 逐位一致
- 测试 446 → 460 passed(`pytest -m "not integration"`,新增 14 例:delivery 4 + evaluate 契约 8 + 冒烟 2)

## Task Commits

| Task | Name | Commit(s) |
| ---- | ---- | --------- |
| 1 | run_delivery 全量重训 × 4 + test 65 张并排评分(D-16;tdd=true) | d3b3844 (test RED) → 6a2048f (feat GREEN) |
| 2 | evaluate.json 主契约 + npz 明细旁车(artifacts.py 扩展) | 3097e41 |
| 3 | 判行表三图 + 合成数据全链冒烟 | 914939b |

## Deviations from Plan

**1. [决策,非偏差——体例收拢] run_delivery 签名取全量 CalibrationConfig**
- **Found during:** Task 1
- **Issue:** 计划动作文本草拟的 (models, scipy_cfg, eval_cfg, optimizer_config) 散传签名与 04-01/04-02/04-04 既定的"实验函数取全量配置"体例冲突
- **Fix:** 落为 `run_delivery(obs_train, obs_test, board, cfg: CalibrationConfig, optimizer_config, progress=None)`,models 从 cfg.evaluate.models 读(测试同样如此供给)
- **Files modified:** src/calib/evaluation/experiments.py
- **Commit:** 6a2048f

**2. [Rule 1 - Bug] 计划文本明细键名笔误**
- **Issue:** 计划 Task 2 写 `_slices_detail`,而 04-04 experiments.py 返回键为 `_slice_detail`
- **Fix:** 私有键名定 `_slice_detail`(与段名 slices、上游返回键对齐),组装侧零改名直通
- **Commit:** 3097e41

其余照计划执行。ruff TRY004 定向豁免一处(明细段非字典的中文契约报错,体例照 _check_optimizer_contract,04-04 先例)。

## Known Stubs

- 无新增占位。04-04 遗留的 bootstrap `var_slice`/`pose_sampling_var` 占位键的**组装侧填充已在本计划落地**(全链冒烟第 ④ 步:分组经验标准差平方 → pose_sampling_variance 相减 → 回填);生产装配在 04-06 CLI 按冒烟同序执行。

## Threat Model Mitigations

- T-04-05-01(obs_test 装载):走 load_observation_set(split="test") 真实路径,行序契约入口内置断言,评分行带 image_id 逐行可对账(冒烟测试实证)
- T-04-05-02/03(明细旁车):npz 读写双向 allow_pickle=False(pickle 模式可执行任意代码,项目既定防线),测试锁死
- T-04-05-04(血缘):主 json 与每个明细 npz 的 .meta.json 带同一份 make_lineage 指纹,冒烟断言 upstream 非空
- T-04-05-05(图失败不拦主产物):库函数只管画图,CLI 侧逐图独立 try/except(共享模式,04-06 执行位)

## Next Phase Readiness

- **04-06(命令层)**:全链冒烟即 CLI 调用序列的样板——取数(sb 角点 npz + noise.json 的 sigma_temporal_corner + optimizer.json recommended_solver)→ 四实验 → run_delivery → 来源分离组装 → 判行表/BIC/选型 → save_evaluate_report → 三图;逐图独立 try/except 与中文播报是仅剩的增量
- 真实数据口径提示(04-04 已记):CV 100 + bootstrap 800 + 分组 32 + LOPO 320 ≈ 1300 次求解,先跑 CV 校准耗时再放大批量;交付轮全量 640 视图重训 × 4 模型另计

## Self-Check: PASSED

- 文件在位:src/calib/visualization/evaluate_figs.py、tests/test_evaluation_smoke.py(created);experiments.py/artifacts.py/test_evaluation_experiments.py(modified)——git status 干净,提交均在
- 提交在位:d3b3844 / 6a2048f / 3097e41 / 914939b(git log 核对)
- 验证在位:tests/test_evaluation_smoke.py 10 passed;全套 460 passed, 25 deselected(`pytest -m "not integration"`,148.39s)
