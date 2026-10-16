---
phase: 01-data-foundation-minimal-calibration
plan: 05
subsystem: visualization
tags: [opencv, drawChessboardCorners, marked-copies, mosaic, overview-sampling, suspect-review, human-loop, phase-closing]

# Dependency graph
requires:
  - phase: 01-01
    provides: calib CLI 骨架、配置解析(VisualizationConfig/MosaicConfig)、uv 环境
  - phase: 01-02
    provides: 4 份角点 npz 产物契约(load_corners)、双检测器检出成功率基线
  - phase: 01-03
    provides: 2×2 基线标定与 comparison.json(D-05 人工核查的量级输入)
  - phase: 01-04
    provides: manifest 全字段体检结论(verdict/suspect_sources/blur_norm)、
              suspect_originals/ 存疑原图副本(存疑拼图的原貌来源)
provides:
  - render_marked:cv2.drawChessboardCorners 标准叠加(检出)/ MISS 红框留痕(未检出),
    JPEG 质量读 config,源文件只读不改(T-05-01 测试锁定)
  - render_all:manifest 全部照片 × classic/sb 两套标记副本,一张不落(D-06/D-07),
    写 marked_copies/summary.json 计数
  - build_mosaic:cv2.resize + numpy 拼贴缩略拼图(ASCII 标注条),零新依赖
  - overview_mosaic 抽样规则:全部存疑 + 每姿态前 N 张 + 模糊度最差 worst_n 张
    (超 200 格按优先级均匀截断);suspect_mosaic 只拼存疑、看未标记原貌
  - write_mosaics:两张拼图 + mosaic_index.json 逐格索引(格号 → image_id/标签/指标值)
  - corners 命令收尾自动产出:npz ×4 + 摘要 + 标记副本两套 + 拼图三件(幂等)
  - 阶段收尾:pipeline.md 措辞对齐 D-01、REQUIREMENTS 四行 Shipped (Phase 1)、
    真实数据全链最终产物与血缘闭合验证、D-05 人工核查清单(待用户答复)
affects: [D-05 人工回环(审阅清单就绪), 02-models-noise(基线 RMS 量级与存疑画像),
          05-corner-optimization(SB 41 帧失明 + 连拍断言 3 组的寻优素材),
          06-report(REP-01 可视化原语可复用 overlay/mosaic)]

# Tech tracking
tech-stack:
  added: []   # 无新增依赖(T-05-SC:cv2.drawChessboardCorners/cv2.putText/cv2.resize 全在栈内)
  patterns: [图内文字一律 ASCII 短标(cv2.putText 不支持中文,中文写在索引/控制台),
             副本只在内存副本上作画、源文件逐字节不变(测试锁定),
             拼图只是索引不是替代(D-06):缩略扫全局,细节回单张副本,
             标签判定缩写优先级 MISS > SUS > OK,
             边框粗细/字高/标注条高全部按画面尺寸比例推导,不写死像素]

key-files:
  created: [src/calib/visualization/overlay.py, src/calib/visualization/mosaic.py,
            tests/test_visualization.py]
  modified: [src/calib/cli/main.py, docs/pipeline.md, .planning/REQUIREMENTS.md]

key-decisions:
  - "标准叠加用 cv2.drawChessboardCorners 一行出图(格子交替着色 + 角点连线),不手画圆圈——研究 Don't Hand Roll 结论,自画既费工又不标准"
  - "MISS 红框留痕:未检出照片也要在两套副本里有位置(红边框 + ASCII 短标),D-06 '一张不落' 在未检出路径同样成立;SB 的 41 张失明帧因此一眼可查"
  - "拼图标签判定缩写三态 OK/SUS/MISS,优先级 MISS > SUS > OK(最要紧的信号占标签);MISS 按经典检测器判定(默认标定路线),SB 差异看 detection_summary"
  - "总览拼图抽样 = 全部存疑 + 每姿态前 N 张 + 模糊度归一值最小的 worst_n 张(值越小越糊),按清单原顺序排列、image_id 去重;超 200 格先保存疑全量、其余均匀截断(当前数据 164 格远够不着)"
  - "存疑拼图看 suspect_originals/ 的未标记原貌而非标记副本——与标记副本互相印证'判定存疑的到底是什么样',副本缺失时退回原图路径不中断"
  - "图内文字全 ASCII:pose_id 截 8 字符 + #连拍序号 + 判定缩写;完整编号与全部指标值在 mosaic_index.json,拼图认格子、索引查编号、副本看细节三级跳"
  - "JPEG 质量从 config 的 visualization.jpeg_quality 透传(默认 90,Pitfall 11:可读性足够磁盘省一半),实测全量副本 1.2GB 在 1–1.5GB 预估内"

patterns-established:
  - "可视化保真三层结构:全量标记副本(证据本体)→ 拼图(索引)→ 逐格索引 JSON(格号到编号的映射),上层永远可下钻到下层"
  - "未检出也是信息:MISS 红框 + summary.json 分计数,检出率不用数文件名"

requirements-completed: [COR-01, IO-02]
# 另:IO-01/IO-03 的 Shipped 状态系本计划收尾时对 Traceability 表的同步
# (实际交付分别在 01-01/01-04,见 REQUIREMENTS.md 各行括注)

# Metrics
duration: 52min
completed: 2026-10-16
---

# Phase 1 Plan 5: 角点可视化保真与阶段收尾 Summary

**全量标记副本(705 张 × classic/sb 两套,SB 41 张 MISS 红框留痕)+ 总览拼图(164 格:常态抽样 + 最差 20 + 全部存疑)与存疑拼图(89 格未标记原貌)+ 逐格索引接入 corners 命令;真实数据一条链(ingest→corners→calibrate)17.5 分钟跑通,2×2 基线全收敛(RMS 0.083~1.24px),血缘链闭合断言全过,阶段收尾文档与需求状态同步完成**

## Performance

- **Duration:** 52 分钟(含真实数据全链 17.5 分钟:ingest 6.2 分钟 + corners 含副本与拼图 7 分钟 + calibrate 4.4 分钟;705 张 × 2 套副本写盘 1.2GB)
- **Started:** 2026-10-16T07:51Z
- **Completed:** 2026-10-16T08:43Z
- **Tasks:** 3/3
- **Files modified:** 6(新建 3 + 修改 3)

## Accomplishments

- **D-06 全量副本落地**:manifest 里 705 张照片(640 train + 65 test)每张 × 每颗检测器各一份标记副本,一张不落;检出画 cv2.drawChessboardCorners 标准叠加,未检出画红框 + "MISS" 短标留痕(classic 705 标记/0 MISS,sb 664 标记/41 MISS——SB 失明帧在副本目录里一眼可数)
- **D-07 双检测器并排可查**:marked_copies/classic/ 与 marked_copies/sb/ 同名文件即同一张照片的两颗检测器结果,人工并排对比哪个角点贴得更准;每套计数写 marked_copies/summary.json
- **拼图只是索引(D-06)**:总览拼图 164 格(89 存疑全收 + 80 姿态各第 1 张 + 模糊度最差 20 张,去重后)扫全局;存疑拼图 89 格看 suspect_originals/ 未标记原貌;mosaic_index.json 逐格记录 image_id/标签/全部指标值,拼图认格子 → 索引查编号 → 副本看细节
- **全链最终验证(自动化断言全过)**:705 条 manifest 全字段、标记副本两套各 705 张、mosaic 三件齐全、comparison.json 4 条;血缘链闭合——4 个标定产物记录的 corners sha256 与实际文件逐一相符,角点产物记录的清单指纹相符
- **2×2 基线全部收敛**(RMS 亚像素量级,详见下表),给 D-05 人工核查提供了量级锚点
- **阶段收尾三件套**:pipeline.md 第①步"自动剔除"过时措辞删除(D-01 只标记),契约速查表 ①②④ 行产物按实际命名同步;REQUIREMENTS.md Traceability 四行 IO-01/IO-02/IO-03/COR-01 → Shipped (Phase 1),IO-03 复选框勾选;全部改动原子提交
- 测试 10 个新用例全绿(全仓 86 过,含 3 个真实数据集成冒烟),ruff 全过

## 全量最终运行记录(2026-10-16,worktree 内完整重跑)

| 阶段 | 产物目录 | 关键数字 |
|------|----------|----------|
| ① ingest | `output/ingest/run_20261016_081008/` | 705 张全字段体检;存疑 89(同姿态比较 51 / 全局线 39);副本 89 张;首轮 diff 表 |
| ② corners | `output/corners/run_20261016_081008/` | classic 705/705、sb 664/705 检出;方格边长中位 45.3/45.8px;连拍断言 80 组查、6 条失败(3 姿态 × 2 检测器);marked_copies 两套各 705 张(1.2GB);mosaic 164 格 + 89 格 + 索引 |
| ④ calibrate | `output/calibration/run_20261016_081704/` | 2×2 全收敛(见下表) |

**2×2 基线对比(comparison.json):**

| 组合 | 视图数 | 整体 RMS(像素) | 收敛 |
|------|--------|------------------|------|
| baseline_classic_pinhole | 640 | 0.1401 | 是 |
| baseline_classic_fisheye | 640 | 0.1332 | 是 |
| baseline_sb_pinhole | 600 | 1.2421 | 是 |
| baseline_sb_fisheye | 600 | **0.0827** | 是 |

白话:RMS 是"把标定结果反投影回去,角点平均偏离多少像素"。四个组合全部落在亚像素到 1.2 像素量级,量级合理;sb_fisheye 最优、sb_pinhole 明显偏差大,符合"广角数据更贴近鱼眼假设"的预期,给 Phase 2 模型层优先级提供了第一份证据。

## Task Commits

Each task was committed atomically:

1. **Task 1: 全量带角点标记副本(classic 与 sb 两套)** - `990f859` (feat)
2. **Task 2: 汇总拼图(索引)与 CLI 集成** - `e292f93` (feat)
3. **Task 3: 全量最终验证、人工核查与阶段收尾** - `8b88467` (docs)

**Plan metadata:** 见下方最终提交(SUMMARY.md)

## Files Created/Modified

- `src/calib/visualization/overlay.py` - render_marked(标准叠加 / MISS 红框,JPEG 质量透传)+ render_all(全量 × 两套,summary.json 计数)
- `src/calib/visualization/mosaic.py` - build_mosaic(缩略拼贴 + ASCII 标注条)+ 总览/存疑两个拼图入口 + write_mosaics(拼图三件 + 逐格索引一次写齐)
- `src/calib/cli/main.py` - corners 命令收尾接 render_all + write_mosaics,控制台中文汇总(格数/MISS 数/目录)
- `tests/test_visualization.py` - 10 用例:副本检出/未检出/源文件逐字节不变/两套目录结构/质量透传;拼图尺寸量级/抽样规则并集/零存疑路径/全量覆盖
- `docs/pipeline.md` - 第①步措辞对齐 D-01;契约速查表 ①②④ 行产物命名同步
- `.planning/REQUIREMENTS.md` - Traceability 四行 Shipped (Phase 1) + IO-03 勾选

## Phase 1 成功标准五条对照(阶段收尾核对)

1. **manifest 全字段 + 剔除留痕可查** ✓(01-01/01-04;本轮 705 条全 precheck,存疑 89 张来源逐条)
2. **角点 npz + 比例化窗口 + 排序规范化断言** ✓(01-02;本轮连拍断言 80 组,3 组异常已点名进 detection_summary)
3. **基线内参 + RMS 落盘 output/calibration/** ✓(0.1401/0.1332/1.2421/0.0827 px,全收敛)
4. **张数不齐/单张姿态不崩溃** ✓(repeats_distribution 数据驱动 {'8': 80};韧性测试在 01-01 锁定)
5. **角点叠加可视化可人工核查** ✓(本计划:全量副本两套 + 拼图 164 格含最差 20 张;人工核查清单见下,待用户答复)

## D-05 人工核查清单(收手标准 = 用户认可;全部路径真实可点)

以下四步请用户按顺序做,认可即 Phase 1 收手;有异议则描述问题,调 `config/calibration.yaml` 后重跑对应环节(每轮 run_<时间戳> 自动留痕可比,D-04 回环):

1. **看总览与最差视图**:打开 `output/corners/run_20261016_081008/mosaic/overview_mosaic.png`(164 格),再抽查最糊的几张单张副本——`marked_copies/classic/20260722_111237_238908.jpg`、`20260722_111232_583444.jpg`、`20260722_111246_814476.jpg`(blur_norm 0.005~0.007,全是 test 侧同批连拍),并与 `marked_copies/sb/` 同名文件并排比:角点贴不贴准?有没有整排系统性偏移?
2. **审存疑清单**:打开 `mosaic/suspect_mosaic.png`(89 格,未标记原貌)与 `suspect_originals/` 里任意 2–3 张原图(重灾姿态:`20260717_173936` 整组 8 张、`20260720_134353` 7 张、`20260720_134213` 4 张)。判定有没有误杀(实际清晰却被标)/漏杀(明显糊却没标)?
3. **查连拍断言异常组**:detection_summary 点名 3 个姿态(`20260717_174622`、`20260720_134806`、`20260720_134914`)——两颗检测器独立给出几乎相同的组内波动(中位 3.5~6.0px),更像是连拍内棋盘真的动过而不是排序污染;请在 marked_copies 里翻看这三组 8 连拍确认。
4. **看 2×2 对比表**:上表四行 RMS 是否在可接受量级(亚像素~2px);sb_pinhole 的 1.24px 偏大是否符合你对这支广角镜头的直觉;SB 41 张 MISS 红框(`marked_copies/sb/` 内)是否可接受。

## Decisions Made

- 见上方 key-decisions(7 条:标准叠加选型、MISS 红框、标签缩写优先级、抽样规则、存疑拼图看原貌、ASCII 标注、JPEG 质量透传)
- **总览格数上限 200**:当前数据 164 格用不上,但换更大的数据集时按"存疑全保、其余均匀截断"的优先级退化,确定性可复现

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] MISS 红框测试的逐字节纯红断言被 JPEG 有损压缩打破**
- **Found during:** Task 1(副本测试首跑)
- **Issue:** 断言 `pixel == (0,0,255)` 失败:JPEG 压缩让纯红边框带毛刺(实测 (0,1,250) 等)
- **Fix:** 改为容差判断"接近纯红"(B<64 且 G<64 且 R>200),断言顶边框整行;检测叠加路径本就不做逐像素相等断言,口径一致
- **Files modified:** tests/test_visualization.py
- **Committed in:** 990f859(Task 1 提交内)

**2. [Rule 3 - Blocking] 拼图入口不接受清单路径**
- **Found during:** Task 2(拼图测试首跑)
- **Issue:** 测试给 write_mosaics 传 manifest.json 路径,函数只收已加载的字典,TypeError
- **Fix:** 三个入口统一过 _as_manifest 归一(字典原样、路径现场加载);CLI 传字典行为不变
- **Files modified:** src/calib/visualization/mosaic.py
- **Committed in:** e292f93(Task 2 提交内)

---

**Total deviations:** 2 auto-fixed(1 测试断言容差化、1 参数归一兼容;均不涉产品行为变更)
**Impact on plan:** 无范围蔓延;真实数据验收数值与计划逐字一致(705×2 套副本、mosaic 三件、comparison 4 条、血缘闭合)。

## Issues Encountered

- worktree 无 input_data(数据只在主检出):沿用前几个计划的做法建软链接指回主仓库,只读契约未破坏;worktree 的 output/ 从空开始,本轮即"首轮"(ingest diff 表写"无上轮可比"——主检出与各 worktree 的 output/ 互不相通属预期)
- 连拍断言本轮点名 3 组(两检测器一致):与 01-02 时期的 3 组为同批姿态,非回归;已列入上方人工核查清单第 3 步
- 全量标记副本 1.2GB(705×2 张 q90):在研究预估 1–1.5GB 内;串行写盘约 4 分钟,Pitfall 11 "不并行化" 判断成立

## Real-Data Verification Record(真实数据验证记录)

- 计划验收脚本逐条通过:`phase-1 full chain artifacts OK`(705 manifest 条目、两套副本各 705、mosaic 两图、comparison 4 条)
- 措辞断言:`docs/pipeline.md` 无"自动剔除"字样;`REQUIREMENTS.md` 含 "Shipped (Phase 1)"
- 血缘断言:4 个标定产物 corners_sha256 与实际文件全等;角点产物→清单指纹相等
- 测试:全仓 86 过(83 单测 + 3 真实数据集成冒烟,冒烟内部重新走三命令接力并断言血缘闭合);ruff 全过

## Known Stubs

无新增占位。noise/fisher/evaluate/analyze/report 五个命令仍是 01-01 建立的计划内占位(D-11 八命令骨架契约),由 Phase 3–6 填充;本计划交付的 overlay/mosaic 均为完整实现。

## User Setup Required

None - 无外部服务配置。

## Next Phase Readiness

- **D-05 回环就绪**:人工核查清单(上)四步全部给出真实路径与数字;用户答复后,调 config 阈值或填 `exclude_images` 清单重跑即可,每轮 run_<时间戳> + suspect_diff 表自动留痕可比
- **给 02-models-noise**:基线量级已锚定(classic_pinhole 0.14px / sb_fisheye 0.083px);存疑 89 张画像与连拍断言 3 组可做"含/不含"对照实验;sb_fisheye 显著优于 sb_pinhole 支持"广角优先鱼眼/有理模型"的 Phase 2 优先级
- **给 05-corner-optimization**:SB 41 帧失明帧清单(副本目录红框)+ 两检测器方格边长实测(中位 45.3/45.8px)是寻优的直接素材
- **给 06-report**:overlay.render_marked / mosaic.build_mosaic 是 REP-01 可视化原语的现成地基(纯函数、无 CLI 依赖)

## Self-Check: PASSED

- 6 个关键文件全部存在(3 新建 + 3 修改);overlay.py 130 行 ≥ 40、mosaic.py 283 行 ≥ 40、tests 315 行 ≥ 30
- 3 个任务提交(990f859 / e292f93 / 8b88467)全部在 git 历史
- 真实数据三轮产物在盘(output/ingest|corners|calibration 最新 run),计划验收脚本与血缘断言全过
- SUMMARY 中全部数字来自实际运行日志与产物 JSON,零硬编码

---
*Phase: 01-data-foundation-minimal-calibration*
*Completed: 2026-10-16*
