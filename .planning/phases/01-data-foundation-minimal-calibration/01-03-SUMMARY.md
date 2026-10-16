---
phase: 01-data-foundation-minimal-calibration
plan: 03
subsystem: solve
tags: [opencv, calibrateCamera, fisheye-kb4, baseline, reprojection-rms, lineage, smoke-test]

# Dependency graph
requires:
  - phase: 01-01
    provides: uv 环境(opencv 4.14/numpy 2.5 锁定)、配置解析(BoardSpec/CalibrationConfig 含
              calibrate.fisheye 与 exclude_images)、manifest 写盘与 latest_run_dir、
              血缘指纹 make_lineage/sha256_file、calib 九命令骨架
  - phase: 01-02
    provides: 角点 npz 落盘契约(4 份 corners_<划分>_<检测器>.npz,稠密掩码 + image_ids 对齐)
              与 load_corners 安全读取、meta.json 旁车(含上游 manifest 血缘)
provides:
  - build_object_points / calibrate_pinhole / calibrate_fisheye / run_baselines:
    2×2 基线标定核心(剔除清单逻辑剔除 D-03、失败驯服 D-10)
  - save_calibration + compute_residuals:calibration.json 全 schema(失败组合键不少、
    值为 null)+ residuals.npz(残差与标定同一投影实现,天然可比)
  - `calib calibrate` 全量命令(640 视图真实数据 4 组合约 5.5 分钟)+ run-all 串联至第④步
  - 真实数据 2×2 第一手对比:KB4 全面优于针孔(sb_fisheye 0.083px 最优,
    sb_pinhole 1.242px 系统性失配)——Phase 2 模型层优先级证据(D-10 的答案)
  - 真实数据小子集端到端冒烟(独立进程三命令接力 + 血缘两环指纹闭合断言)
  - SKELETON.md 行走骨架架构契约(栈/目录契约/产物血缘/CLI 地图/刻意不做的事)
affects: [01-04(体检消费 manifest 预留位), 01-05(可视化), 02-models-noise(KB4 优先的
          直接证据、基线数字为对照起点), 迭代回环(exclude_images 生效路径已通)]

# Tech tracking
tech-stack:
  added: []   # 无新增依赖(T-03-SC:01-01 已锁 uv.lock)
  patterns: [进求解器前显式 reshape+断言(陷阱5), imageSize 恒 (宽,高)=shape[::-1](陷阱8),
            OpenCV 绑定 dtype 差异就地降精度并注释(calibrateCamera 只收 float32),
            fisheye 必带 RECOMPUTE_EXTRINSIC 防假收敛(合成回归当场抓住),
            失败值语义:结构键不少、数值字段一律 null, 残差回代用与标定同一投影函数,
            分辨率按血缘读角点 meta 记录的上游 manifest(跨轮不串),
            冒烟测试=临时树缩小版新数据集+独立进程三命令接力]

key-files:
  created: [src/calib/solve/baseline.py, tests/test_solve_baseline.py,
            tests/test_smoke_realdata.py,
            .planning/phases/01-data-foundation-minimal-calibration/SKELETON.md]
  modified: [src/calib/cli/main.py, src/calib/io/config.py, config/calibration.yaml]

key-decisions:
  - "fisheye 默认 flags 从 0 改为 2(RECOMPUTE_EXTRINSIC):合成 KB4 数据回归实测
     flags=0 会因外参不随内参重算而假收敛(RMS 上百像素);skew 仍估计(未加 FIX_SKEW)、
     CHECK_COND 仍关(软失败路径保留),实际取值随产物 flags_used 留痕"
  - "calibrate_pinhole 走 calibrateCameraExtended 取逐帧 RMS(A6 已确认可用);
     绑定只收 float32,进函数显式降精度——像素坐标 float32 仍有 ~1e-4px 表达精度,
     远小于 0.5px 断言线,注释说明原因"
  - "run_baselines 签名增加显式 image_size 参数(计划签名未含):分辨率属于上游事实,
     从角点 meta 血缘记录的 manifest 路径读,不重新扫描、不硬编码,跨轮不串"
  - "test 基线只用 train;test 65 张留给后续阶段独立评分,本阶段不碰(计划明示)"
  - "冒烟测试的 input_data 用软链接只读接入(目录/文件级),临时树即缩小版新数据集,
     顺带成为'换数据集仅改目录内容'的第一号证据"

patterns-established:
  - "产物 schema 防御式设计:失败组合 calibration.json 键一个不少、数值字段一律 JSON null,
     下游对四组合的读取逻辑一视同仁(冒烟测试有锁定)"
  - "血缘链闭合断言进测试:标定记角点文件指纹、角点记清单指纹,冒烟测试逐环比对实际文件
     (篡改即失配,T-03-01 的机器证明)"
  - "数值绑定行为差异必须先用合成已知真值数据回归再上真实数据——本次 fisheye 假收敛、
     float32-only 两处绑定陷阱都被合成回归先行抓住,真实数据零返工"

requirements-completed: [COR-01]

# Metrics
duration: 34min
completed: 2026-10-16
---

# Phase 1 Plan 3: 基线标定与端到端贯通 Summary

**真实 train 数据(640 视图)上 2×2 = 4 个基线全部收敛并排落盘:KB4(等距投影鱼眼模型)全面优于针孔+畸变(sb_fisheye 0.083px 最优、classic_fisheye 0.133px 次之,针孔两组合 0.140/1.242px)——"这批广角更像哪类模型"有了第一手答案;三命令独立进程接力在干净临时树上贯通,血缘链两环指纹闭合;行走骨架就此完整**

## Performance

- **Duration:** 34 分钟(其中真实数据全量标定约 5.5 分钟:640 视图 × 4 组合;另含 worktree 内重跑 ingest+corners 约 11 分钟,与执行并行)
- **Started:** 2026-10-16T06:32:24Z
- **Completed:** 2026-10-16T07:06:30Z
- **Tasks:** 3/3
- **Files modified:** 8(新建 4 + 修改 4)

## Accomplishments

- **2×2 基线标定核心落成带测试防线**:`build_object_points`(square_size 唯一落点,IO-03 解耦有源码断言)、`calibrate_pinhole`(calibrateCameraExtended 取逐帧 RMS)、`calibrate_fisheye`(失败驯服:结构键不少、数值字段 null、中文失败说明)、`run_baselines`(剔除清单逻辑剔除 D-03 + 未检出帧跳过 + 可用视图 <2 中文报错);8 个合成回归用例全绿
- **真实数据 2×2 全部收敛并排落盘(D-10 的答案)**:

  | 组合 | 视图数 | 整体重投影 RMS(px) | 收敛 |
  |------|--------|--------------------|------|
  | baseline_classic_pinhole | 640 | 0.1401 | 是 |
  | baseline_classic_fisheye | 640 | 0.1332 | 是 |
  | baseline_sb_pinhole | 600 | 1.2421 | 是 |
  | baseline_sb_fisheye | 600 | **0.0827** | 是 |

  **结论(第一手证据,非猜想):KB4 在两种角点上全都优于针孔——这批广角数据更像等距投影模型,Phase 2 模型层应优先做 KB4 系。** sb_pinhole 的 1.242px 不是 SB 角点质量问题(同一份角点在 KB4 下 0.083px 全场最优),而是 SB 角点与针孔模型的系统性失配(广角边缘畸变针孔吃不下),这本身就是 Phase 2 的重要输入
- **内参交叉验证一致**:针孔 fx/fy = 1855.3/1845.7,鱼眼 fx/fy = 1856.0/1846.4(差 <0.1%);主点 cx,cy ≈ (1269.9, 725.4) vs (1269.5, 726.0)——两条独立求解路径互相印证
- **`calib calibrate` 命令落地**:最新 corners run 的两份 train npz → 剔除过滤 → 四组合 → calibration.json + residuals.npz(残差与标定同一投影实现)→ comparison.json 并排 → 中文对比表(含鱼眼为何可能失败的白话解释);run-all 串联至第④步
- **血缘链机器可证**:每个 calibration.json 的 lineage 记 corners_file + corners_sha256,冒烟测试逐环比对实际文件指纹(标定→角点→清单两环全闭合,T-03-01/02 落地);分辨率按血缘读角点 meta 记录的上游 manifest,跨轮不串
- **真实数据端到端冒烟(integration 标记)**:临时树(config 复制 + train 前 3 姿态/test 前 5 张软链接)= 缩小版新数据集,三命令独立进程接力退出码全 0,产物齐全(24+5 规模、4 份 npz、4 个基线目录、comparison.json 4 条),鱼眼即使不收敛结构也完整——3 用例 17.7 秒跑完
- **SKELETON.md 行走骨架契约**:架构决策表(运行时/CLI/数学库/产物仓库/血缘/模块布局/参数来源)、已触碰的栈、刻意不做的事、后续切片计划
- 全套测试 57 个全绿(54 单元 + 3 集成),ruff 全过

## Task Commits

Each task was committed atomically:

1. **Task 1: 2×2 基线标定函数 + 合成数据回归测试** - `047bc4d` (feat)
2. **Task 2: 标定产物落盘 + `calib calibrate` 真实数据 2×2** - `b65fc1b` (feat)
3. **Task 3: 真实数据端到端冒烟测试 + SKELETON.md 行走骨架契约** - `f380924` (test)

**Plan metadata:** 见下方最终提交(SUMMARY.md)

## Files Created/Modified

- `src/calib/solve/baseline.py` - 四个核心函数 + save_calibration/compute_residuals(中文文档,名词白话;进求解器前显式 reshape+断言)
- `src/calib/cli/main.py` - calibrate 命令全量实现(血缘取上游、四组合循环、comparison.json、中文对比表)+ run-all 串联
- `src/calib/io/config.py` - FisheyeConfig 默认 flags 0→2(偏差 #2 的同步修改)
- `config/calibration.yaml` - calibrate.fisheye.flags: 2,注释记下"flags=0 会假收敛"的实测依据
- `tests/test_solve_baseline.py` - 8 用例:针孔真值恢复(RMS<0.5px、内参误差<1%)、imageSize (宽,高) 语义(2560×1440)、鱼眼不抛异常+结构完整(合成真 KB4 几何)、形状断言中文报错、剔除清单生效(乱序坏角点注入)、四组合名与元信息、<2 视图中文报错、square_size 解耦源码断言
- `tests/test_smoke_realdata.py` - 3 用例(integration 标记):三命令接力+产物规模、血缘两环指纹闭合、四组合结构一视同仁
- `.planning/phases/01-data-foundation-minimal-calibration/SKELETON.md` - 行走骨架契约(后续阶段的架构基准)

## Decisions Made

- **fisheye 默认 flags=2(RECOMPUTE_EXTRINSIC)**:见偏差 #2——这是本计划最重要的技术决策,防止全量真实运行产出"收敛了但全是错的"的鱼眼基线;实际取值随产物 flags_used 留痕,config 注释写明实测依据与 CHECK_COND 的加法
- **分辨率走血缘不重扫**:calibrate 读角点 meta 里记录的上游 manifest 路径拿分辨率——即便期间又跑过新 ingest,标定仍对齐角点当时用的那份清单(冒烟测试有断言)
- **残差回代用与标定同一投影函数**:针孔 cv2.projectPoints、鱼眼 cv2.fisheye.projectPoints——残差与 RMS 天然可比,不存在"两套投影实现各说各话"的风险
- **失败值语义定死并测试锁定**:未收敛组合 calibration.json 键一个不少、数值字段一律 null,下游无需先判成败再挑键(冒烟测试逐键断言)
- **冒烟测试是"缩小版新数据集"**:临时树 + 软链接只读,既是端到端证明也是"换数据集仅改目录内容"的第一号证据

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] OpenCV 4.14 的 calibrateCamera/Extended 绑定只收 float32,64 位输入直接报"Unsupported format"**
- **Found during:** Task 1(API 冒烟确认,先于写代码)
- **Issue:** 研究代码示例用 `.astype(np.float64)` 传角点,本机 OpenCV 4.14.0 实测 float64 输入报 `-210: Unsupported format ...(Point3f)`(fisheye.calibrate 则两种精度都收)
- **Fix:** calibrate_pinhole 内部先按 (n,1,dim) 断言形状、再显式 astype(np.float32) 调用;注释说明降精度依据(像素坐标 float32 仍有 ~1e-4px 表达精度,远小于 0.5px 断言线)
- **Files modified:** src/calib/solve/baseline.py
- **Verification:** 合成回归 8 用例全绿(针孔 RMS <0.5px、内参误差 <1%)
- **Committed in:** 047bc4d

**2. [Rule 1 - Bug] fisheye flags=0 会假收敛:外参不随内参重算,合成 KB4 数据上 RMS 高达上百像素**
- **Found during:** Task 1(鱼眼合成回归测试当场抓住:converged=true 但 RMS 96~207px,真值明明可解到 0.0000px)
- **Issue:** flags=0 时 OpenCV 只按初始内参算一次棋盘位姿、之后不随内参更新,LM 走进 D 发散的假极小(D=[-0.94, 19, -77, -263]);加 CALIB_RECOMPUTE_EXTRINSIC 后同一份数据精确恢复真值 K/D(RMS 0.0000)。计划与研究按"flags=0 默认"写,但这在本构建上产出无意义的鱼眼基线
- **Fix:** 默认 flags 改 2(RECOMPUTE_EXTRINSIC)——config/calibration.yaml 与 FisheyeConfig 同步,注释记录实测依据;skew 仍估计(未加 FIX_SKEW=8)、CHECK_COND 仍关(软失败路径保留,try/except 语义不变);鱼眼合成测试改按 KB4 几何假设造数据(真广角视场、焦距贴内部初值),回归"数据合假设时求解数学正确"
- **Files modified:** config/calibration.yaml, src/calib/io/config.py, src/calib/solve/baseline.py, tests/test_solve_baseline.py
- **Verification:** 合成 KB4 回归 RMS <0.5px;真实数据四组合全部收敛且 KB4 优于针孔(见上表)
- **Committed in:** 047bc4d

**3. [Rule 1 - Bug] excluded_images 按检测器重复收集,同一照片在名单里出现两次**
- **Found during:** Task 1(剔除清单测试断言失败:['pose_03_003', 'pose_03_003'])
- **Issue:** run_baselines 对每颗检测器的数据各过滤一次,被剔照片被记两次
- **Fix:** 返回前按照片编号 set 去重再排序
- **Files modified:** src/calib/solve/baseline.py
- **Verification:** 剔除测试转绿
- **Committed in:** 047bc4d

**4. [Rule 3 - Blocking] run_baselines 需要图像分辨率,计划签名未含该参数**
- **Found during:** Task 2(CLI 接线时发现无分辨率来源)
- **Issue:** 计划的动作描述里 run_baselines(corners_train, cfg, board) 缺 image_size,而 calibrateCamera/fisheye.calibrate 都必填;角点 npz 本身不存分辨率
- **Fix:** 签名增加显式 image_size 参数;CLI 从角点 meta 血缘记录的上游 manifest 读 summary.resolution(优先血缘路径,缺失才退回最新 ingest)——分辨率是上游事实,不重扫、不硬编码
- **Files modified:** src/calib/solve/baseline.py, src/calib/cli/main.py
- **Verification:** 冒烟测试断言标定对齐的是"角点当时用的那份清单"
- **Committed in:** b65fc1b

---

**Total deviations:** 4 auto-fixed(3 Rule 1 Bug + 1 Rule 3 Blocking)
**Impact on plan:** 全部为让计划验收断言真实成立所必需,无范围蔓延;其中偏差 #2 改变了 config 默认值(记录在案,属参数而非结构变化),真实数据验收(4 组合产物 schema/血缘/对比表)与计划逐字一致。

## Issues Encountered

- worktree 无 .venv / output / input_data:软链接主仓 input_data(只读)+ `uv sync` 秒级恢复;corners npz 是 Task 2 的直接输入而 output/ 不共享,后台重跑 ingest+corners(705 张,约 11 分钟)与 Task 1 写码并行,零等待;结果与 01-02 主检出数据一致(classic 100%/100%、SB 93.8%/98.5%、方格中位 45.3px、连拍断言失败 6 组)
- 鱼眼合成测试首版把棋盘放在近针孔几何(f=600、小视场),KB4 初值(max(宽,高)/π≈204)偏离真值过远会假收敛——与偏差 #2 同根;最终测试数据按 KB4 假设构造(深度 250、f=204、入射角至 ~26°),并在测试文档里留了"近针孔数据会假收敛"的提醒
- 全量真实标定约 5.5 分钟(640 视图 × 4 组合),fisheye 组合(RECOMPUTE_EXTRINSIC 每轮重算外参)未见超时压力,无 `CALIB_CHECK_COND` 异常触发

## Real-Data Verification Record(真实数据验证记录)

- 产物:`output/calibration/run_20261016_064911/`(4 个 baseline_* 目录 + comparison.json)
- 四组合全部收敛,无失败组合(failure_note 全 null;flags_used:针孔 0、鱼眼 2,均已留痕)
- 整体 RMS 与逐帧 RMS:residuals.npz 四组合齐全,allow_pickle=False 可读,shape (N,88,2) 全有限(N=640/640/600/600)
- 血缘闭环:lineage.corners_file 指向 `output/corners/run_20261016_063256/` 的对应 npz,corners_sha256 与实际文件逐一相符;excluded_images 键在(本轮清单为空,过滤逻辑由合成测试锁定)
- comparison.json 恰 4 条,中文对比表一眼可读;命令退出码 0
- 上游 corners run 为本 worktree 全量重跑(705 张):classic train/test 640/640、65/65 全检出,SB 600/640、64/65,与 01-02 结论一致

## Known Stubs

无新增占位。`calib calibrate` 已是完整实现;其余五个命令(noise/fisher/evaluate/analyze/report)仍是 01-01 建立的计划内占位(D-11 骨架契约),由 Phase 2–6 填充,01-01-SUMMARY 已记录;run-all 现串联 ingest→corners→calibrate 三步。

## User Setup Required

None - 无外部服务配置。

## Next Phase Readiness

- 4 组基线产物(calibration.json + residuals.npz + comparison.json)在 `output/calibration/run_20261016_064911/`,schema 与血缘契约即 Phase 2 模型层/评估层的输入形态;**KB4 优先**的模型层排序证据已就位(sb_fisheye 0.083px 全场最优,sb_pinhole 1.242px 系统性失配待模型层解释)
- exclude_images 过滤链路已通:人工审阅后往 config/calibration.yaml 的 exclude_images 填 image_id 重跑 `calib calibrate` 即生效,剔除名单自动进产物血缘(D-03/D-04 回环就绪)
- SKELETON.md 契约落盘:后续阶段在本文件的架构决策内填充实现;测试侧新增"血缘链闭合断言"模式,Phase 2+ 产物可复制同一防篡改套路
- 环境注意:worktree 每次需要重建软链接(input_data)与 uv sync;真实数据重跑 calibrate 前需先有本地的 ingest+corners 产物(output/ 不跨检出共享)

## Self-Check: PASSED

- 8 个关键产物文件全部存在(4 新建 + 4 修改)
- 3 个任务提交(047bc4d / b65fc1b / f380924)全部在 git 历史
- 真实数据产物 output/calibration/run_20261016_064911/ 与 corners run_20261016_063256/ 在盘,verify 脚本断言通过
- 对比表数字全部来自实际运行,零硬编码

---
*Phase: 01-data-foundation-minimal-calibration*
*Completed: 2026-10-16*
