---
phase: 05-error-attribution-corner-tuning
plan: 05
subsystem: analysis + corners + visualization + cli + tests
tags: [corner-opt, degradation-synthesis, dual-caliber, sb-flags, taskfailed-envelope, cor-02]

# Dependency graph
requires:
  - phase: 05-error-attribution-corner-tuning (05-01~05-04)
    provides: analyze 六块骨架 + _stage 逐块兜底 + AnalyzeConfig 寻优参数(corner_opt_seed/quick_corner_opt_frames/三族退化档/classic_win_ladder)+ assign_bands/radius_footprint 分带尺 + _latest_analyze_or_skip 指纹门
  - phase: 04-evaluation-model-selection
    provides: evaluation/parallel.run_parallel(任务小元组 + TaskFailed 信封 + worker 内 cv2.setNumThreads)+ evaluate.json delivery.final_params(r 归一化基准)
  - phase: 03-noise-fisher-crlb
    provides: noise.json 顶层键 sigma_detector_corner(未显著劣化线 = 10 × 它)+ injection.py 量化收尾与 sha1 种子范式
  - phase: 01~02
    provides: corners/detect.py detect_classic/detect_sb 唯一检测入口 + window.derive_window_frame 窗口合法域断言 + corners_train_sb.npz(image_ids 行序契约)+ board.yaml + input_data 原图(只读)
provides:
  - "src/calib/analysis/corner_opt.py:apply_degradation 三族退化原语(float 域→clip→rint 量化收尾,恒等档逐位为零)+ _seed_for 四元组 sha1 种子(退化图不落盘的复现正当性)+ build_method_candidates(classic 窗口阶梯按 2w+1 < sq_px 逐帧过滤 + SB flags=0 生产位 + SB_ACCURACY 精化位)+ detect_with 单一入口调度 + build_units 单因素退化单元(quick 13 档/full 加最坏交叉单元)+ run_corner_opt 主编排(按帧最大 r 分层抽帧 → 阶段 A 双参照干净检出 → 阶段 B run_parallel 退化网格 → 双口径统计 + 判读文案)"
  - "analyze 块 5 接线:corner_opt 段有值(schema analyze-corner-opt-v1),旁车 analysis/corner_opt.npz(3744 行逐角点位移底账),双口径曲线图 corner_opt_self.png / corner_opt_prodsb.png,missing_sections 收缩到 {anisotropy}"
  - "真实数据 quick 首跑 run_20261022_010551:36 帧 × 8 方法(classic_w3~w13 + sb + sb_acc)× 13 单元;校准档 noise_0.0 自参照 RMS 恰为 0;噪声档位移单调不减(0→0.30px);verdict 点名 classic_w7 未显著劣化单元平均 0.0143px vs 生产 sb 0.0507px(只提示方向不自动执行)"
  - "16 例新单元测试(退化原语/种子/候选过滤/编排确定性/失败信封)+ 集成 1 例新段断言;全套 579 passed + 集成 test_analysis_realdata 13 passed + ruff 零告警"
affects: [05-06(anisotropy 段收口 + 全链对账可直接引用 corner_opt 判读), Phase 6(report 消费 corner_opt 段与双口径曲线图)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "检测唯一入口红线(静态+行为双防线):corner_opt.py 源码断言不含 cornerSubPix/findChessboardCorners 字样,SB 结果结构上不进二次亚像素精化;方法候选 = detect_classic/detect_sb 的入口参数变体(dataclasses.replace),绝不复制第二份检测实现"
    - "退化合成三纪律(照 injection.py 体例):sha1 四元组种子可复现(退化图不落盘)、float 域→clip→rint 量化收尾(真实相机输出整数灰阶)、恒等档必须逐位为零(校准档自证参照对齐)"
    - "双口径并报(裁定①):stats._self 主口径(各方法对自家干净检出的偏移,量退化敏感度)+ stats._prod_sb 旁证(对生产 SB 干净检出的绝对偏移,跨方法可比),曲线图各出一张、判读文案两列都给"

key-files:
  created:
    - src/calib/analysis/corner_opt.py
    - tests/test_analysis_corner_opt.py
  modified:
    - src/calib/cli/main.py
    - src/calib/visualization/analyze_figs.py
    - tests/test_analysis_realdata.py
    - config/calibration.yaml(analyze 寻优参数,05-01 已落)

key-decisions:
  - "A1 销险实测(2026-10-22,合成棋盘双帧,OpenCV 4.14.0):flags=0 与 CALIB_CB_ACCURACY 检出数相同、逐角点逐位一致——「SB 追加亚像素精化」预期语义未获证实;两档仍都保留候选(结论由数据出),结论随段落账(sb_accuracy_verdict)+ 模块 docstring 固定段。真实数据首跑两档曲线确有差异(合成图太干净暴露不出来),flag 在实数据上有可测效果——销险设计与数据闭环吻合"
  - "指标口径 = 与参照的逐角点欧氏位移,不是重投影误差(Pitfall 4 头号陷阱):窗口吞邻角的系统性偏差逐角点同向,会被求解整个吸收、重投影完全看不见"
  - "参照口径双份:各方法自己的干净检出(自参照)+ 生产 SB(flags=0)现场重检(生产参照,与角点轮次同一 cv2 构建保证同版本可比);方法间干净偏移另存 cross_clean 作旁证"
  - "失败留痕三件套:检测未检出该单元记 NaN + src='not_found'(数据不是事故);worker 异常经 TaskFailed 信封转整帧 NaN 占行 + n_failed_tasks 计数,任务流不断(T-05-16);本次首跑 4/288 任务走信封、235/3744 单元 not_found(集中在 noise_64 检测器失效区,属预期探索区)"
  - "未显著劣化线 = 10 × σ_detector_corner(0.0871px)随段落账,判据数值由当前数据现算不抄死;verdict 若生产配置不占优,如实点名更稳档位并以「只提示方向,不自动执行任何改动」收尾(裁定③同口径)"

requirements-completed: [COR-02]

# Metrics
duration: 约 2h05min(08:06–10:11,含真实数据 quick 首跑约 44 分钟;库层与编排层为先前会话完成)
completed: 2026-10-22
---

# Phase 5 Plan 5: COR-02 角点合成退化寻优端到端 Summary

**把真实棋盘照人为变差(加噪/模糊/低对比)后用 8 种候选配置重提角点,与双参照逐角点比对:36 帧 × 13 单元 × 8 方法全网格跑通(quick 档约 44 分钟),校准档零位移自证对齐、噪声档单调、双口径(stats._self 主口径 / _prod_sb 旁证)并排落账;判读文案以数据点名 classic_w7 档在未显著劣化单元更稳(0.0143px vs 生产 sb 0.0507px),只提示方向不自动执行——角点环节选型自此有数据背书**

## Performance

- **Duration:** 约 2h05min 墙钟(Task 1/2 库层与编排为先前会话提交,本会话完成 Task 3 真实数据首跑 + 集成验证 + 收尾)
- **Completed:** 2026-10-22
- **Tasks:** 3
- **Files:** 2 created + 3 modified(+ config 寻优参数 05-01 已落)

## Accomplishments

- **Task 1 退化原语 + 种子四元组 + 方法候选(A1 销险先行):** apply_degradation 三族统一 float 域 → clip(0,255) → rint 回 uint8(量化收尾不能省——真实相机输出整数灰阶,injection Pitfall 4);noise 复用 injection 语义、blur 核径 6σ+1、contrast 128 为轴缩放,未知族中文 ValueError(T-05-19);_seed_for 四元组 sha1(base_seed + sha1("image|unit|method|rep") mod 2³²),退化图不落盘的复现正当性;build_method_candidates 按 2w+1 < sq_px 逐帧过滤窗口阶梯(超档不进候选不报错——与 window.derive_window_frame 同一条断言)+ SB 两档;A1 销险:一次性脚本双帧实测 CALIB_CB_ACCURACY 语义(与 flags=0 逐位一致)后即删,结论固化进 docstring 与 sb_accuracy_verdict
- **Task 2 run_corner_opt 主编排:** 按帧最大归一化半径 r 分层抽帧(角点寻优关心边缘,外环帧必须有代表);阶段 A 每(帧 × 候选)对干净图检出一次存双参照;阶段 B 任务 = (帧 × 方法 × 重复)小元组经 run_parallel 调度(12 并行,quick 36 帧 × 8 方法 ≈ 288 任务 ≈ 3744 单元检测),worker 顶层可序列化 + 首行 cv2.setNumThreads(任务携带值)(04-02 教训);逐 (方法 × 单元) RMS/中位/最大/95 分位 + r 分带切片,双口径 _self/_prod_sb 同构并排;统计与判读全部由数据现算,不预设结论
- **Task 3 CLI 接块 5 + 双口径曲线族 + 真实数据首跑:** analyze 块 5 经 _stage 兜底接入(取数沿角点血缘:corners_train_sb.npz 观测集 + 清单照片记录,原图只读);plot_corner_opt_curves 三族各一张子图(RMS 实线/中位虚线,生产位加粗点名,未显著劣化线随段落),主口径与旁证口径各出一张(裁定①);四要素指引追加 corner_opt 判读(校准档零位移自证 + 未显著劣化线 + 生产配置相对位置);真实数据 quick 首跑 run_20261022_010551 退出码 0,旧轮 182345 按及时清理删除

## 真实数据首跑结论(run_20261022_010551)

- **规模与健全性:** 36 帧 × 8 方法(classic_w3~w13 + sb + sb_acc)× 13 单元;校准档 noise_0.0 自参照 RMS 恰为 0(参照对齐自证);sb 噪声阶梯单调不减 0 → 0.0179 → 0.0292 → 0.0491 → 0.0870 → 0.1613 → 0.3019 px;旁车 corner_opt.npz 3744 行(= 帧×方法×重复×单元)逐角点位移底账
- **数据背书判读(裁定①双口径):** 未显著劣化线 0.0871px(= 10 × σ_detector 0.00871px);verdict_note 如实点名:**classic_w7 档在未显著劣化单元更稳(自参照平均 0.0143px,生产配置 sb 0.0507px),可考虑切换——只提示方向,不自动执行任何改动**;旁证口径 stats._prod_sb 两列并排可查
- **A1 销险闭环:** 合成图上 sb 与 sb_acc 逐位一致(销险结论「无可测差异」),真实数据上两档曲线确有差异——flag 在实数据上有可测效果,两档保留候选的设计被首跑数据印证
- **失败留痕:** 4/288 任务走 TaskFailed 信封(NaN 占行 + 计数,任务流不断);235/3744 单元检测未检出(not_found,集中在 noise_64.0——检出率 29.2%,即配置要探的检测器失效区,属预期数据不是事故)
- **missing_sections 收缩到 {anisotropy}**(05-06 接入),七图五旁车在盘

## Task Commits

1. **Task 1: 退化原语 + 种子四元组 + 方法候选 + A1 销险** - `82a779a` (feat)
2. **Task 2: run_corner_opt 主编排(双参照 + run_parallel + 双口径统计)** - `46a0a3f` (feat)
3. **Task 3: analyze 接块 5 + 双口径曲线族 + 真实数据 quick 首跑** - `5b7eb8d` (feat)

**Plan metadata:** 见本计划收尾 docs 提交(SUMMARY + STATE + ROADMAP + REQUIREMENTS)

## Files Created/Modified

- `src/calib/analysis/corner_opt.py` - COR-02 全部库层与编排(退化原语/种子/候选枚举/单一入口调度/主编排/双口径统计/判读文案;687 行)
- `tests/test_analysis_corner_opt.py` - +16 例(恒等档逐位为零/种子复现/候选过滤不报错/detect_with 88 角点 <1px/源码红线/小样编排/校准零位移/噪声单调/同种子逐位一致/串行并行一致/TaskFailed 占行)
- `src/calib/cli/main.py` - analyze 块 5 _stage 接线 + 旁车挂载 + 双口径双图 + 四要素播报扩展(重计算块预告与耗时)
- `src/calib/visualization/analyze_figs.py` - +plot_corner_opt_curves(reference 参数切 _self/_prod_sb 两张;段缺席画「无数据」不冒充)
- `tests/test_analysis_realdata.py` - corner_opt 段从 None 改有值断言(36 帧/校准零位移/双口径键/npz 行数/双图在盘/verdict 与销险文案)+ 源码断言块 2/3/4/5 走 _stage 而 budget 直调

## Decisions Made

见前置 key-decisions 五条。补充执行层两条:

- 块 5 取数在块内完成(train 观测集 + 清单照片记录),沿用 evaluate 的排除名单体例——角点寻优要读原图施加退化,清单记录是 image_id → 文件路径的唯一承载
- 曲线图对 full 档交叉单元(噪+糊双因素)不进单因素子图(横轴无解),交叉档结论以 analyze.json 为准(图是可视化,数字结论以 json 为准)

## Deviations from Plan

None - plan executed as written(本会话续跑 Task 3:工作树中已就绪的接线/图/测试代码经全量验证后原样提交,真实数据首跑一次通过,无返工)。

## Known Stubs

- `anisotropy` 段:计划内显式 None 留位(05-06 COR-03 接入),meta.missing_sections 如实登记——非本计划产物
- sb_accuracy_verdict 的合成图实测与真实数据表现存在差异(合成一致/真实有差):已按设计由真实数据曲线闭环,非存根;full 档可进一步加密重复次数压随机性

## Threat Surface Scan

无新增面:零新依赖(cv2.GaussianBlur 栈内);无网络/鉴权/文件写新面(原图只读、退化图不落盘);计划威胁登记 T-05-16~19 全部按缓解计划落地并有测试锁定(TaskFailed 信封/串行并行逐位一致/源码断言/中文 ValueError + 非负断言)。

## Issues Encountered

None(真实数据首跑一次通过;noise_64 检出率下降属配置设计要探的失效区,已按「数据不是事故」口径留痕)。

**当轮产物清理(及时清理):** 被替代的 analyze 轮 run_20261021_182345 已删(新轮 run_20261022_010551 完整、集成测试绿后执行;旧轮为终端消费产物,无下游血缘依赖)。既有 3 个集成失败(noise/fisher 血缘断链 ×2 + 注入老测试 ×1)维持 deferred-items.md 登记,本计划零接触。

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 05-06 收口条件齐备:corner_opt 段有值,五块齐(仅剩 anisotropy);全链对账可直接引用 corner_opt 判读文案与双口径曲线图;classic_w7 切换建议(只提示方向)可进收口报告的「可选动作」栏
- 集成测试 `_latest_analyze_or_skip` 指纹门已指向新轮 run_20261022_010551,05-06 重跑 analyze 后自动轮转
- 待办提醒:deferred-items.md 两项超范围修复(血缘断链重跑 / 注入老测试对齐)仍排队,建议 05-06 或阶段六开篇专门一轮处理

## Self-Check: PASSED

- 5 个关键文件全部在盘(2 created + 3 modified;src/calib/analysis/corner_opt.py 687 行、tests/test_analysis_corner_opt.py 16 例)
- 3 个任务提交全部在案:82a779a / 46a0a3f / 5b7eb8d
- 快速环 579 passed(50 deselected,含 16 例新测试)+ ruff check src tests 零告警;集成 test_analysis_realdata 13 passed(1 例新段断言)
- 真实数据首跑 run_20261022_010551:corner_opt 段有值、校准档零位移、噪声单调、双口径键齐、npz 3744 行、双口径双图在盘、verdict 含「生产」与切换建议、sb_accuracy_verdict 落账
- 源码红线:corner_opt.py 无 cornerSubPix/findChessboardCorners 直调(静态断言测试锁定);A1 一次性脚本无残留(git status 干净)

---
*Phase: 05-error-attribution-corner-tuning*
*Completed: 2026-10-22*
