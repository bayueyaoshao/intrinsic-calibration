---
phase: 01-data-foundation-minimal-calibration
plan: 02
subsystem: corners
tags: [opencv, chessboard, subpixel, findChessboardCornersSB, cornerSubPix, canonical-order, npz]

# Dependency graph
requires:
  - phase: 01-01
    provides: uv 项目环境(opencv 4.14/numpy 2.5 锁定)、manifest.json 写盘与 latest_run_dir、
              血缘指纹 make_lineage、配置解析(BoardSpec/CornersConfig)、calib 九命令骨架、
              imread_checked 带检查读图
provides:
  - 方格像素边长估计 + 亚像素窗口推导(帧级/逐角点双粒度,2*win+1<sq_px 断言防线)
  - 角点排序规范化层(四种翻转变体映到唯一形态)+ 连拍一致性断言(翻转污染点名)
  - 双检测器统一封装(classic 含二次亚像素精化 / SB 一次到位,同一返回结构)
  - 角点 npz 落盘与安全读取(allow_pickle=False)+ meta.json 旁车留痕
  - `calib corners` 全量命令(705 张真实数据已跑,4 份角点 npz + detection_summary.json)
  - 两检测器成功率第一手数据:classic 100%(train/test 全检出),SB 93.8%/98.5%
affects: [01-03(基线标定直接消费角点 npz), 02-models-noise, 05-corner-optimization(双检测器对比垫底数据),
          迭代回环(检测质量摘要供人工审阅)]

# Tech tracking
tech-stack:
  added: []   # 无新增依赖(T-02-SC:01-01 已锁 uv.lock)
  patterns: [窗口按实测方格像素边长比例推导+逐帧断言, 排序规范化唯一形态(行朝右/列朝下),
            连拍同名角点波动断言(失败点名不剔除), 双检测器同构返回下游不分叉,
            npz 稠密掩码布局 NaN=未检出, 二次精化调用源码级单点约束,
            合成夹具超采样抗锯齿+像素中心映射主点对齐]

key-files:
  created: [src/calib/corners/window.py, src/calib/corners/canonical.py,
            src/calib/corners/detect.py, src/calib/corners/artifacts.py,
            tests/test_corners.py, tests/test_corners_artifacts.py]
  modified: [tests/conftest.py, src/calib/cli/main.py]

key-decisions:
  - "二次亚像素精化在 detect.py 源码中恰好单处调用(帧级/逐角点统一成'同窗口分组精化'),测试锁定——SB 路线绝不二次精化的防线可被机器检查"
  - "SB 的 win_size 字段按同一公式推导入档但注明'仅记录',保证两检测器产物结构对齐、审查口径一致"
  - "连拍断言失败的归因方式:逐帧对'连拍内逐点中位数参照'比偏差,超限帧点名;整组都未超限(失败来自中位数条件)时整组列入存疑"
  - "合成棋盘夹具画法:4 倍超采样画硬边棋盘再 INTER_AREA 缩小,主点按 OpenCV 缩小映射平移 c_big=SS*(c+0.5)-0.5——否则光栅化画格偏差 + 0.375px 系统偏移会让 <0.5px 断言失真"
  - "corners 阶段不做剔除(exclude_images 留给标定命令),未检出帧照常占一行(D-01 只标记精神)"

patterns-established:
  - "头号陷阱断言防线:窗口推导函数内部断言 2*win+1<方格边长,违反当场报错,不靠下游复查"
  - "产物数值断言脚本化:真实数据 npz 的形状/NaN 语义/窗口不等式用一行 python -c 断言,可重复执行"
  - "CLI 冒烟先行:全量真实运行前先用合成小数据集在临时目录走通 ingest→corners 端到端(本次正是它抓住了 latest_run_dir 路径拼接错误)"

requirements-completed: [COR-01]

# Metrics
duration: 36min
completed: 2026-10-16
---

# Phase 1 Plan 2: 角点提取核心数学与 calib corners 命令 Summary

**双检测器全量跑通 705 张真实照片:classic 100% 全检出、SB 93.8%(成功率的常识预期被本数据反转);窗口从实测方格边长(中位 45.3px,远超预估的 15–25px)比例推导且逐帧断言通过;规范化+连拍断言点名 3 个真实手抖姿态、零翻转污染**

## Performance

- **Duration:** 36 分钟(其中真实数据全量角点提取约 13 分钟:705 张 × 2 检测器)
- **Started:** 2026-10-16T05:53:54Z
- **Completed:** 2026-10-16T06:27:57Z
- **Tasks:** 3/3
- **Files modified:** 8(新建 6 + 修改 2)

## Accomplishments

- **三条研究透的数学契约全部落成带测试防线的代码**:窗口推导(2*win+1<sq_px 逐帧断言,帧级/逐角点双粒度)、排序规范化(四种翻转变体映到唯一形态)、连拍一致性断言(翻转污染当场点名不剔除)
- **真实数据第一手答案(开放问题:「经典检测器在本数据的成功率」)**:classic train 640/640(100%)、test 65/65(100%);SB train 600/640(93.8%)、test 64/65(98.5%)——与研究预期相反,本数据上经典检测器反而全检出,SB 漏了 40+1 张,直接进迭代回环与 Phase 5 寻优证据链
- **方格像素边长实测中位数 45.3px(29.7~93.1)**,远超 PROJECT.md 预估的 15–25px——证明 D-08 比例化窗口是硬需求:写死 11×11 窗口在这批数据上会全面失准
- **连拍断言首战告捷**:80 组 × 2 检测器全查,3 个姿态双检测器一致超中位数容差(spread_max 3.7~6.2px,远小于 12~17px 的 max 线,无翻转污染特征)——真实手抖,容差校准的第一批数据(D-04 回环输入)
- 真实数据 npz 数值断言全过:4 份文件、train 640 / test 65 行对齐 manifest、窗口不等式对全部检出帧成立、未检出帧 NaN 占行不消失
- 46 个单元测试全绿(新增 22 个),ruff 全过;SB 严禁二次精化以源码计数断言锁定(`cv2.cornerSubPix(` 全文件恰 1 次)

## Task Commits

Each task was committed atomically:

1. **Task 1: 合成棋盘投影夹具 + 方格像素边长估计与窗口推导** - `d105f87` (feat)
2. **Task 2: 排序规范化 + 连拍一致性断言 + 双检测器封装** - `8133b6d` (feat)
3. **Task 3: 角点产物落盘 + `calib corners` 全量真实运行** - `99fa433` (feat)

**Plan metadata:** 见下方最终提交(SUMMARY.md)

## Files Created/Modified

- `src/calib/corners/window.py` - estimate_sq_px(行/列相邻间距中位数保守取小)+ derive_window_frame / derive_window_per_corner(断言 2*win+1<方格边长,中文说明头号陷阱)
- `src/calib/corners/canonical.py` - canonicalize(行朝右/列朝下唯一形态)+ burst_consistency(返回 ok/spread_median_px/spread_max_px/threshold_px,失败不抛异常)
- `src/calib/corners/detect.py` - detect_classic(粗定位→二次亚像素精化→规范化)/ detect_sb(一次到位,严禁二次精化),同构返回 dict(uv/sq_px/win_size/detector)
- `src/calib/corners/artifacts.py` - save_corners(4 份 npz 命名契约 + meta.json 旁车含血缘/选项/规范化规则版本)+ load_corners(强制 allow_pickle=False)
- `src/calib/cli/main.py` - corners 命令实现(manifest→双检测器全量并跑→按姿态连拍断言→4 份 npz + detection_summary.json→中文对比表);run-all 串联
- `tests/conftest.py` - make_chessboard_image 合成夹具(已知真值 K/D/位姿投影,blur/noise/overexpose 退化变体,4 倍超采样抗锯齿)
- `tests/test_corners.py` - 17 用例:估计误差<5%、四种 sq_px 不等式、双检测器<0.5px、四变体归一、翻转污染检出、源码防线
- `tests/test_corners_artifacts.py` - 5 用例:落盘读回往返、未检出占行 NaN、meta 留痕、四文件名契约、pickle 拒读

## Decisions Made

- **二次精化统一成"同窗口分组精化"单调用点**:cornerSubPix 一次只收一个窗口,帧级(全帧同值)与逐角点(分组循环)统一在同一循环里实现——顺带满足计划的源码计数断言(调用形式全文件恰 1 次),SB 隔离从"约定"升级为"机器可查"
- **SB 的 win_size 仅记录不使用**:两检测器产物结构严格同构(下游不分叉),SB 不精化的事实写进注释与 meta,审查方格尺度时两份口径一致
- **连拍断言归因策略**:失败组内逐帧与"逐点中位数参照"比偏差点名元凶;若失败来自中位数条件而单帧都不超 max 线(本次真实数据正是如此),整组列入存疑——不假装能定位到单帧
- **合成夹具必须抗锯齿**:硬边光栅化给角点带来约 0.7px 画格偏差,INTER_AREA 缩小还有 0.375px 像素中心映射偏移;4 倍超采样 + 主点平移后双检测器对真值误差降至 0.5px 内——<0.5px 断言才真实成立
- **corners 阶段不消费 exclude_images**:只标记不剔除(D-01),剔除是标定命令的逻辑剔除(D-03),职责不混

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] cornerSubPix 拒收 float64 点坐标**
- **Found during:** Task 2(双检测器测试)
- **Issue:** 检测输出统一转成 float64 后传给精化函数,OpenCV 断言 `count >= 0` 失败(只收 32 位浮点)
- **Fix:** 精化前转 float32、精化后转回 float64,产物契约(dtype float64)不变
- **Files modified:** src/calib/corners/detect.py
- **Verification:** test_detector_accuracy_within_half_pixel[classic] 通过
- **Committed in:** 8133b6d(Task 2 提交)

**2. [Rule 1 - Bug] 合成夹具硬边光栅化导致 ~0.7px 画格偏差,0.5px 断言失真**
- **Found during:** Task 2(双检测器精度测试)
- **Issue:** 二值硬边棋盘的检测角点带系统性画格偏差(SB 最大 0.745px);另发现 INTER_AREA 缩小的像素中心映射(u_big=4u+1.5)与投影缩放(4u)不一致,再加 0.375px 系统偏移——合成图"真值"不再可信
- **Fix:** 夹具改为 4 倍超采样画布绘制 + INTER_AREA 缩小(等效光学模糊),大画布投影主点按 c_big = SS*(c+0.5) − 0.5 平移;修复后双检测器误差全部 <0.5px
- **Files modified:** tests/conftest.py
- **Verification:** 17 个 corner 测试全绿;真值恢复可信(经典检测器误差中位数从 ~0.59px 降至 <0.2px 量级)
- **Committed in:** 8133b6d(Task 2 提交)

**3. [Rule 1 - Bug] corners 命令的 latest_run_dir 根路径重复拼接**
- **Found during:** Task 3(合成小数据集 CLI 端到端冒烟)
- **Issue:** `latest_run_dir(OUTPUT_ROOT / "ingest", "ingest")` 拼出 `output/ingest/ingest`,正确产物在 `output/ingest/run_*`——真实全量运行前被冒烟测试当场抓住
- **Fix:** 改为 `latest_run_dir(OUTPUT_ROOT, "ingest")`
- **Files modified:** src/calib/cli/main.py
- **Verification:** 冒烟目录端到端跑通(ingest→corners 4 份产物齐全)
- **Committed in:** 99fa433(Task 3 提交)

---

**Total deviations:** 3 auto-fixed(3 Rule 1 Bug)
**Impact on plan:** 全部为让计划验收断言真实成立所必需,无范围蔓延;真实数据验收数值与计划逐字一致(见下方验证记录)。

## Issues Encountered

- worktree 里没有 input_data(数据只在主检出)与 .venv:沿用 01-01 的做法建软链接;`uv sync` 从 uv.lock 秒级恢复;worktree 的 output/ 为空,先重跑 `calib ingest` 生成 manifest(705 张,0 警告)再跑 corners
- 后台运行管道接了 `tail` 导致中间进度不可见(缓冲),靠进程 CPU 时间确认存活——下次后台长任务直接落盘日志
- 冒烟生成器自身有个投影系数笔误(方格只有 11px),恰好顺带验证了小方格路径(sq=11 → w=4,9<11 成立);非产品代码问题
- SB 在本数据 train 漏检 40 张(6.2%)、test 漏 1 张:按 D-07/D-01 原样保留 NaN 占行,是回环与 Phase 5 的核心对比数据,不是需要当场修的缺陷

## Real-Data Verification Record(真实数据验证记录)

- 产物:`output/corners/run_20261016_061253/`(4 份 npz 文件名精确符合契约 + detection_summary.json + 4 份 meta.json)
- 成功率:classic train 640/640(100.0%)、test 65/65(100.0%);SB train 600/640(93.8%)、test 64/65(98.5%)
- 方格像素边长(检出帧):classic 中位 45.3px(29.7~93.1);SB 中位 45.8px(29.8~92.9)
- 窗口不等式 `2*win_size+1 < sq_px` 对全部检出帧成立(自动化断言,Task 3 verify 脚本通过)
- npz 对齐:image_ids train 640 / test 65,与 manifest 逐行对齐;未检出行 uv/sq_px/win_size 全 NaN、detected=False
- 连拍断言:每检测器 80 组全查,失败 6 组 = 3 个姿态 × 双检测器一致失败(20260717_174622 / 20260720_134806 / 20260720_134914),spread_max 3.7~6.2px 远低于 max 线——真实手抖特征,无翻转污染;存疑帧已点名进 detection_summary.json 供人工审(D-01)
- 全量运行耗时约 13 分钟(705 张 × 2 检测器,单机单线程)

## Known Stubs

无新增占位。`calib corners` 已是完整实现;其余六个命令(noise/calibrate/fisher/evaluate/analyze/report)仍是 01-01 建立的计划内占位(D-11 骨架契约),由后续阶段填充,01-01-SUMMARY 已记录。

## User Setup Required

None - 无外部服务配置。

## Next Phase Readiness

- 4 份真实数据角点 npz + detection_summary.json 就位,01-03 基线标定可直接按 `latest_run_dir(output, "corners")` 消费;schema(稠密掩码 + image_ids 对齐 + 血缘旁车)即研究 Pattern 5 约定
- 连拍断言容差(median_tol=0.05)在 3 个真实姿态上偏紧:回环人工审阅后调 calibration.yaml 的 corners.burst_check 即重跑可校准,无需改代码
- SB 漏检的 41 张帧清单在 detection_summary.json 的 detection 段可推导(detected=False 行),Phase 5 寻优与迭代回环的第一手素材
- 方格像素边长实测 45px 量级:后续所有像素级参数(拼图缩略、叠加标记半径)都有了真实尺度依据

## Self-Check: PASSED

- 8 个关键文件全部存在(4 源码 + 2 测试 + 2 修改)
- 3 个任务提交(d105f87 / 8133b6d / 99fa433)全部在 git 历史
- 真实数据产物 output/corners/run_20261016_061253/ 在盘,数值断言通过
- 检测质量摘要 numbers 全部来自实际运行,零硬编码

---
*Phase: 01-data-foundation-minimal-calibration*
*Completed: 2026-10-16*
