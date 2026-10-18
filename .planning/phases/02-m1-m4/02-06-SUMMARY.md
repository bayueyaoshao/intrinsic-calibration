---
phase: 02-m1-m4
plan: 06
subsystem: solve
tags: [pose-only-solve, suspect-review, matplotlib, lineage, d04, phase-closeout]

# Dependency graph
requires:
  - phase: 02-m1-m4 (02-01~02-04)
    provides: models/ 注册表与 M1–M4 投影/雅可比、ObservationSet 装载、OpenCV/scipy 双后端与 CalibrationResult 落盘
  - phase: 02-m1-m4 (02-05)
    provides: calibrate 全矩阵编排、comparison.json、run 目录血缘契约、02-05 首轮真实数据基准(run_20261018_022047)
  - phase: 01-data-foundation-minimal-calibration
    provides: 角点 npz 四口径、manifest(89 张存疑判定)、01-HUMAN-UAT 的存疑帧影响 Gap
provides:
  - solve_pose_only:只估位姿小求解(6 参数、内参锁定、残差仅 models.project),任何注册模型可用的 held-out 单帧评分
  - score_views:逐帧回代评分(与 compute_result_metrics 同一把尺)
  - review_suspects:D-04 存疑帧复核编排,产出带血缘的 suspect_review/(JSON 结论表 + PNG 对比图)
  - calib calibrate 命令末尾挂接复核(复核失败不影响标定产物)
  - Phase 2 收尾:docs/pipeline.md ④ 契约、01-HUMAN-UAT Gap 闭环、REQUIREMENTS 四条 Shipped
affects: [phase-3-noise-fisher (线性化点取自标定结果), phase-4-evaluation (held-out 评分复用 solve_pose_only/score_views), 角点寻优 (阶段五参考复核相关性结论)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "held-out 评分不重估内参:solve_pose_only 用 cv2.solvePnP 起步、scipy trf 精化 6 位姿参数,残差仅经 models.project(与标定同尺)"
    - "佐证与准绳分离:PNG 只作佐证,数字与结论判定、阈值公式、血缘全在 JSON"
    - "图内文字 ASCII、中文进 JSON 与控制台(沿用 mosaic.py 体例)"
    - "附加产物不反噬主产物:calibrate 里复核失败仅中文提示,标定产物照常落盘"

key-files:
  created:
    - src/calib/solve/scoring.py
    - src/calib/visualization/suspect_compare.py
  modified:
    - src/calib/io/config.py
    - src/calib/cli/main.py
    - config/calibration.yaml
    - tests/test_solve_scoring.py
    - docs/pipeline.md
    - .planning/phases/01-data-foundation-minimal-calibration/01-HUMAN-UAT.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "主对照固定 M2(研究遗留问题 1 定夺):suspect_review_model 默认 m2、白名单校验,不逐模型重复复核;结果优先取 <模型>__scipy(combo_dir_name 须显式传 loss='linear',None 会拼出 _scipy_linear),未跑 scipy 时兜底 __opencv 并在 JSON 记 fallback"
  - "复核逐帧表与 manifest 按 image_id 对齐,manifest 每行必出表中(result 逐帧 RMS 缺失记 None+中文说明),行数=清单行数由构造保证;verdict/来源记号原样保留不清洗"
  - "血缘继承所选 result.json 的 config_sha256(复核与该轮标定用同一套配置),upstream={result.json, manifest.json} 各带 sha256"
  - "真实数据结论走向 B(两组无差别,偏差 15.9%<30%,不剔除):89 张存疑帧保留在数据流中,'看着糊'≠'算得差';阶段五角点寻优可把存疑标记仅作参考权重"
  - "真实 run 产物留存主仓库 output/calibration/run_20261018_034000(12MB,逐字节核对一致):工作树返回后即被移除,UAT 引用的路径必须真实存在(沿 02-05 保真先例)"

patterns-established:
  - "复核类附加产物的容错模式:库函数 raise 带中文原因,CLI 宽接只提示不中断(计划明文:复核失败不影响标定结果)"
  - "测试里注册玩具模型必须在测试体内 register + try/finally 摘除,模块级装饰器会污染全局注册表"
  - "单帧求解失败不中断复核:该帧 RMS 记 NaN 占位带中文说明,分布统计自动跳过"

requirements-completed: [ITER-01]

# Metrics
duration: 50min
completed: 2026-10-18
---

# Phase 2 Plan 06: D-04 存疑帧复核收编 Summary

**只估位姿小求解 + 存疑帧复核编排(真实数据走向 B:89 张存疑帧偏差 15.9%<30%,不剔除)+ Phase 2 收尾三件(pipeline 契约/UAT Gap 闭环/REQUIREMENTS 收官)**

## Performance

- **Duration:** 50 min(含 21 分钟真实全矩阵试车;TDD RED→GREEN 双提交)
- **Started:** 2026-10-18T03:22:00Z
- **Completed:** 2026-10-18T04:06:00Z
- **Tasks:** 3/3
- **Files modified:** 9(计划 files_modified 全覆盖)

## Accomplishments

- **只估位姿小求解(Task 1,TDD):** `solve_pose_only` 对单帧角点在锁定内参下只解 6 位姿参数(solvePnP 起步 + scipy trf 精化,残差仅经 models.project),`score_views` 与 `compute_result_metrics` 同尺;玩具模型经注册表零侵入可用,合成数据位姿恢复误差 <0.5px、无噪时达机器精度
- **存疑帧复核编排(Task 2):** `review_suspects` 覆盖全部 705 帧(存疑 89 = train 67 + test 22;正常 616 = train 573 + test 43),train 用标定逐帧 RMS、test 用只估位姿小求解;30% 相对偏差阈值定走向,结论/阈值公式/分组分位数/逐帧表/血缘全部落盘 JSON,matplotlib 双联对比图(直方图 + 累积分布)作佐证;calibrate 命令挂接,复核失败仅中文提示
- **真实数据结论(Task 2):** 走向 B(两组无差别)——存疑组中位数 0.099px vs 正常组 0.086px,偏差 15.9% < 30%,不剔除;m2__opencv/m2__scipy 与 02-05 首轮逐位一致(diff = 0.0,成功标准 4 的第二次独立实证)
- **阶段收尾(Task 3):** docs/pipeline.md ④ 行换成全矩阵 run 契约(含 suspect_review/),01-HUMAN-UAT 存疑帧 Gap 闭环回写,REQUIREMENTS MOD-01/MOD-02/SOL-01/ITER-01 标 Shipped (Phase 2),进度 8/32

## Task Commits

1. **Task 1(RED):** `9b2be0d` test(02-06): 位姿小求解与逐帧评分的失败测试(TDD RED)
2. **Task 1(GREEN):** `fb51775` feat(02-06): 只估位姿小求解与逐帧评分(任何注册模型可用)
3. **Task 2:** `4db065a` feat(02-06): 存疑帧复核编排 + matplotlib 对比图 + calibrate 挂接(D-04)
4. **Task 3:** `6f1178a` docs(02-06): 存疑复核结论回写 UAT + pipeline 契约同步 + REQUIREMENTS 收尾

## Files Created/Modified

- `src/calib/solve/scoring.py` - score_views / solve_pose_only / review_suspects(30% 阈值、走向措辞模板、血缘、NaN 占位)
- `src/calib/visualization/suspect_compare.py` - 直方图+累积分布双联对比图(Agg 离线,图内 ASCII)
- `src/calib/io/config.py` - CalibrateConfig.suspect_review_model(默认 m2,白名单校验)
- `src/calib/cli/main.py` - calibrate 末尾挂接复核,宽接容错只提示
- `config/calibration.yaml` - suspect_review_model: m2 及注释
- `tests/test_solve_scoring.py` - 16 用例:位姿恢复/机器精度/同尺一致性/玩具模型零侵入/源码契约(禁 import 具体模型)/编排端到端/兜底/空结论/配置校验
- `docs/pipeline.md` - ④ 行:输入补 manifest+配置矩阵,产物改 run 目录布局 + suspect_review/,需求栏 MOD-01~02/SOL-01/ITER-01
- `.planning/phases/01-data-foundation-minimal-calibration/01-HUMAN-UAT.md` - 存疑帧影响 Gap 闭环(走向 B,数字与路径),第 2 项测试改"已闭环"
- `.planning/REQUIREMENTS.md` - MOD-01/MOD-02/SOL-01/ITER-01 → Shipped (Phase 2) 附落点,Last updated 8/32

## 真实数据复核结论(阶段五角点寻优直接引用)

run_20261018_034000(产物已留存主仓库,逐字节核对一致):705 帧逐帧表,对照模型 m2(取 m2__scipy,无兜底),阈值 |中位数差|/正常组中位数 ≥ 0.30 且存疑组更高:

| 组 | 张数(有效) | 中位数 px | 均值 px | 四分位 px |
| --- | --- | --- | --- | --- |
| 存疑 suspect | 89(89) | 0.0995 | 0.1059 | 0.0800 ~ 0.1198 |
| 正常 normal | 616(616) | 0.0859 | 0.1059 | 0.0720 ~ 0.1039 |

**结论:走向 B(两组无差别)**——偏差 15.9% 未达 30% 阈值,存疑帧照常留在数据流中,不剔除(D-01)。数字准绳:`output/calibration/run_20261018_034000/suspect_review/suspect_vs_normal.json`;佐证图:同目录 PNG(69KB)。

## 成功标准对照(ROADMAP Phase 2,供 /gsd-verify-work)

| # | 标准 | 证据 |
| --- | --- | --- |
| 1 | M1/M2/M3 与 projectPoints、M4 与 fisheye.projectPoints 一致 + 数值雅可比逐参数通过 | 02-01 `tests/test_models_parity.py`、`tests/test_models_jacobian.py`;本轮全量 265 passed(含 5 integration)再证 |
| 2 | 双后端对同一 ObservationSet × 任一注册模型返回同一 CalibrationResult;scipy 残差仅由 models.project 构成 | 02-03/02-04 `tests/test_solve_solvers.py`、`tests/test_solve_scipy_residuals.py`;本轮新增源码契约测试锁定 scoring.py 不 import 具体模型 |
| 3 | 真实数据 OpenCV→scipy 精化在至少 M1/M2 收敛且不劣于纯 OpenCV | run_20261018_034000 comparison.json:m1__scipy 0.235517198 = m1__opencv(逐位一致)、m2__scipy 0.091862846 < m2__opencv 0.091862924;soft_l1 偏差已按 02-05 SUMMARY 预案留痕提交人工裁定 |
| 4 | 产物带血缘指纹,任一阶段可独立重跑 | 02-05 编排血缘 + 本轮独立重跑与首轮 m2 两路径 params/rms diff 恰为 0.0(<1e-12);suspect_review 血缘 upstream 双文件 sha256 齐全 |
| 5 | 玩具模型一处注册即可被使用(零侵入) | 02-04 求解侧测试 + 本轮 ToyScoringPinhole 经注册表直接走 solve_pose_only/score_views(测试体内注册、finally 摘除) |

## 计划 must_haves 逐条核对

| 计划真值 | 结果 |
| --- | --- |
| 只估位姿小求解可评任何注册模型的单帧(玩具模型不例外) | 达成(测试锁定 + 源码契约) |
| 逐帧表覆盖 705 张,89/616 分组与判定来源齐全 | 达成(705 行精确对齐 manifest) |
| 主对照固定 M2,结论按 D-04 走向模板落盘 | 达成(走向 B 模板措辞) |
| JSON+PNG 生成,血缘 upstream = result.json + manifest.json | 达成(双 sha256) |
| 重跑后 m2 两路径与首轮一致 <1e-12 | 达成(diff = 0.0) |
| 收尾三件(pipeline/UAT/REQUIREMENTS) | 达成(6f1178a) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 玩具模型模块级注册污染全局注册表**
- **Found during:** Task 1(测试互测)
- **Issue:** ToyScoringPinhole 若用模块级 `@register_model` 装饰,收集期即注册,导致 `test_models_registry.py` 的"摘除玩具模型后注册表应复原"断言失败
- **Fix:** 去掉装饰器,测试体内 `register_model(...)` + `try/finally` 摘除,并断言内置四模型齐全
- **Files modified:** tests/test_solve_scoring.py
- **Verification:** 全量 265 passed(含注册表复原测试)
- **Committed in:** fb51775

**2. [Rule 1 - Bug] combo_dir_name 漏传 loss 导致 scipy 结果查不到**
- **Found during:** Task 2(真实数据首验,逐帧表 0 行)
- **Issue:** `combo_dir_name(model_id, "scipy")` 不传 loss 时拼出 `m2__scipy_linear`,与磁盘目录 `m2__scipy` 不符,复核结果查找失败
- **Fix:** 显式传 `combo_dir_name(model_id, "scipy", "linear")` 并注释该坑
- **Files modified:** src/calib/solve/scoring.py
- **Verification:** 真实数据 705 行全对齐,m2__scipy 命中且无兜底
- **Committed in:** 4db065a

**3. [Rule 1 - Bug] suspect_compare.py 的 noqa E402 指令未被启用(RUF100)**
- **Found during:** Task 2(lint 收口)
- **Issue:** 本仓库 ruff 配置未启用 E402,`# noqa: E402` 成为无用指令报 RUF100
- **Fix:** 换成中文说明注释(导入顺序刻意性不改)
- **Files modified:** src/calib/visualization/suspect_compare.py
- **Verification:** `uv run ruff check` 对本计划文件全绿
- **Committed in:** 4db065a

---

**Total deviations:** 3 auto-fixed(3 bug)
**Impact on plan:** 三处均为正确性/收口必需,无范围蔓延。models/__init__.py 既有 5 处 F401 属 deferred-items.md 登记在案的历史问题,按边界规则未动。

## Issues Encountered

- 真实试车 m1__scipy_softl1 未收敛(status=0,评估上限):与 02-05 首轮逐位一致的既有发现,非本计划引入,按 02-05 SUMMARY 预案留待人工裁定,本计划未改动求解器
- 上下文中途压缩:经提交哈希与工作区状态核对后无损续跑,任务边界与提交原子性未受影响

## User Setup Required

None - 无外部服务配置。

## Known Stubs

None - 无占位实现;复核所有分支(命中/兜底/空结论/单帧失败)均有真实代码路径与测试。

## Next Phase Readiness

- Phase 2 六计划全部完成(02-01~02-06),REQUIREMENTS 8/32;阶段三(噪声底/Fisher)可直接取 run_20261018_034000 的 m2__scipy result.json 作线性化点(血缘相对路径落位)
- 阶段四 held-out 评分可直接复用 score_views/solve_pose_only(同尺已由测试锁定)
- 真实 run 产物已留存主仓库 `output/calibration/run_20261018_034000`(12MB),UAT 引用路径在工作树移除后依然有效

---
*Phase: 02-m1-m4*
*Completed: 2026-10-18*
