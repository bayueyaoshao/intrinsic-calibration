---
phase: 01-data-foundation-minimal-calibration
plan: 04
subsystem: io
tags: [opencv, precheck, laplacian-variance, exposure-clip, visibility-probe, robust-z-score, MAD, quantile, suspect-originals, round-diff]

# Dependency graph
requires:
  - phase: 01-01
    provides: uv 环境、姿态数据模型(scan_dataset_dirs/imread_checked)、manifest 写盘与
              latest_run_dir、血缘指纹、配置解析(PrecheckConfig/CornersConfig)、calib CLI 骨架
  - phase: 01-02
    provides: estimate_sq_px(方格像素边长估计,体检归一复用)、双检测器成功率先例(探针交叉验证的对照)
provides:
  - 体检三指标纯函数:board_roi(FAST_CHECK 预筛→完整查找→回退全图)、exposure_clip(>=250/<=5 裁剪占比)、visibility_probe(双检测器只探成败与框占比)、compute_metrics(扁平指标 dict,blur_norm 按实测方格像素边长平方归一)
  - 双来源存疑判定 judge_suspects:同姿态稳健 z 分数(pose_relative:*)为主、全局 1%/99% 分位线与双检测器可见性(global:*)兜底,MAD=0 跳过,只标记不剔除(D-01/D-02)
  - manifest 全字段 schema:images 逐张带 precheck(全部数字 + visibility + suspect_sources + verdict,NaN 写 null),summary 带 suspect_counts 分桶与分划分双检测器成功率
  - 存疑原图副本 write_suspect_originals(visualization 层,shutil.copy2 只读源)(D-06)
  - 跨轮存疑对比表 write_round_diff:新增/解除/持续三桶按 image_id 对齐点名(D-04)
  - `calib ingest` 全体检命令 + 首轮真实数据存疑画像(89/705 张,判定来源逐条在 manifest)
affects: [01-05(迭代回环审阅存疑清单与副本), 02-models-noise(存疑帧供噪声分析对照),
          05-corner-optimization(SB 漏检 41 帧的寻优素材), 人工迭代回环(每轮 diff 表)]

# Tech tracking
tech-stack:
  added: []   # 无新增依赖(T-04-SC)
  patterns: [指标先原始值后归一值双记录, 体检探针不精化不整理(与角点阶段职责分离),
             稳健 z 分数(中位数绝对差 MAD)+ 全局分位线双来源判定、来源逐条列明可解释,
             判定函数输入几张输出几张(数量锁定即"只标记不剔除"的机器证明),
             产物数字 NaN 一律写 null(合法 JSON), 跨轮对比按 image_id 对齐三桶点名留痕]

key-files:
  created: [src/calib/io/precheck.py, src/calib/visualization/suspects.py, tests/test_io_precheck.py]
  modified: [src/calib/io/manifest.py, src/calib/cli/main.py]

key-decisions:
  - "体检探针复用给模糊度定位:visibility_probe 的经典检出角点直接传给 board_roi(compute_metrics 内),省去同一颗检测器同一条检测白跑第二遍——705 张每轮省约 8 分钟;board_roi 单独调用时仍保持 FAST_CHECK 预筛→完整查找→回退全图的完整路径"
  - "可见性探针只问看没看见:拿到原始检测结果就收手,不做二次亚像素精化、不做排序整理(源码级断言:precheck.py 无 cv2.cornerSubPix( 与 canonicalize()——体检没看见但角点阶段检出的差异才可对比"
  - "NaN 一律写 null 进 manifest:NaN 不是合法 JSON,下游工具读不动;缺失语义由 _finite_or_none 统一兜住"
  - "判定来源排序固定:pose_relative:blur/exposure 在前、global:* 在后,审清单时先看组内比较再看全局兜底(D-02 两来源可辨)"
  - "小样本分位线特性如实接受:n=9 时 1%/99% 线天然夹住极值(测试数据顶部并列构造规避误伤);真实数据 705 张下每条分位线约兜住 7 张,行为符合设计"
  - "write_round_diff 返回 (路径, 三桶计数),首轮也照写文件(写'首轮运行,无上轮可比'),调用方不必判空"

patterns-established:
  - "比例化双记录:每个指标同时留原始值(blur_raw)与归一值(blur_norm = raw / 实测方格像素边长²),判定用归一值,审查可回看原始值"
  - "只标记不剔除的代码级证明:precheck.py 与 suspects.py 全文件无 os.remove/.unlink(/shutil.rmtree/del 语句(grep 断言),judge_suspects 输入几张输出几张(测试锁定)"
  - "跨轮回环留痕:每轮 run_<时间戳> 目录独立 + suspect_diff_vs_<上轮>.txt 三桶点名,空 diff 即回归验证"

requirements-completed: [IO-02, IO-01]

# Metrics
duration: 40min
completed: 2026-10-16
---

# Phase 1 Plan 4: 入库体检深化(三指标 + 双来源判定)Summary

**三项比例化体检指标(棋盘框内拉普拉斯方差/曝光裁剪比例/双检测器可见性探针)+ D-02 双来源存疑判定落地 `calib ingest` 全体检:705 张真实数据两轮运行,存疑 89 张只标记不剔、原图副本精确 89 张、跨轮空 diff(新增 0/解除 0/持续 89)回归验证通过,探针成功率与 01-02 角点阶段逐帧吻合**

## Performance

- **Duration:** 40 分钟(其中真实数据两轮 ingest 各约 7.5 分钟,合计 15 分钟)
- **Started:** 2026-10-16T07:04:58Z
- **Completed:** 2026-10-16T07:45Z
- **Tasks:** 3/3
- **Files modified:** 5(新建 3 + 修改 2)

## Accomplishments

- **体检三指标全部纯函数、天然比例化**:模糊度在棋盘检测框内计算并向外扩一圈,再除以实测方格像素边长平方(归一值);曝光用 >= 250 / <= 5 的裁剪占比;可见性探针经典+SB 各问一次"看没看见",只记成败与棋盘框面积占比,不精化不整理(与角点阶段职责分离,源码级断言锁定)
- **D-02 双来源判定可解释可分辨**:同姿态连拍内稳健 z 分数(中位数绝对差,即"每个数离中位数多远再取中位数"的抗极端值离散度)为主——8 连拍里 1 张异常糊必然带 `pose_relative:blur`;全局 1%/99% 分位线兜底单张姿态;双检测器都失明记 `global:visibility`;来源逐条列明,审清单时"跟同组比出来的"和"踩了全局线的"一眼分开
- **D-01 只标记不剔除是代码级证明而非口头承诺**:precheck.py 与 suspects.py 全文件无任何删除调用(grep 断言),judge_suspects 对 8 张输入返回 8 条(测试锁定);存疑照片照常留在数据流中
- **D-06 manifest 存全部数字 + 存疑原图副本**:每张照片的 precheck 对象带 blur_raw/blur_norm/expo_hi/expo_lo/visibility(缺失写 null)/suspect_sources/verdict;89 张存疑原图 shutil.copy2 到 suspect_originals/(拼图缩略图看不清细节,原图供人工放大)
- **D-04 跨轮可比落地**:write_round_diff 按 image_id 对齐两轮判定,新增/解除/持续三桶逐张点名 + 来源;第二轮空 diff(新增 0/解除 0/持续 89)= 确定性回归验证通过
- **研究开放问题 1/4 的答案被探针独立复证**:可见性探针 classic train/test 100%/100%、SB 93.8%/98.5%,与 01-02 角点阶段的检测结果(SB 漏检 40+1=41 帧)完全一致——两条独立代码路径互证,数据可信
- 19 个体检测试全绿(全仓 73 过),ruff 全过

## 首轮真实数据存疑画像(2026-10-16 run_20261016_073020,人工回环的输入)

| 维度 | 数字 |
|------|------|
| 存疑总数 | 89 / 705 张(12.6%):train 67 / test 22 |
| 按来源分桶(照片张数,可重叠) | 同姿态相对比较 51 张、全局线兜底 39 张 |
| 逐来源计数 | pose_relative:exposure 40、pose_relative:blur 18、global:exposure 28、global:blur 16 |
| 重灾姿态 | 20260717_173936(8/8 全存疑)、20260720_134353(7/8)、20260720_134213(4/8)——回环优先审这三组 |
| blur_norm 分布(692~705 有限值) | 中位 1.02,p1=0.02,p99=5.24(极差 260 倍,归一值跨数据仍悬殊,全局线确有必要) |
| 曝光 | expo_hi 中位 0.57%、最大 19.5%;expo_lo 中位 1.4%、最大 2.4%(高光裁剪是本批数据的主要曝光问题) |
| 可见性 | classic 0 失明;SB 41 帧失明(与 01-02 完全一致);blur_norm 缺失 0 张(经典检测器全检出 ⇒ 每张都框出了棋盘) |
| 双检测器成功率 | classic train 640/640(100.0%)、test 65/65(100.0%);SB train 600/640(93.8%)、test 64/65(98.5%) |
| 重复数分布确认 | {"8": 80} 与 01-01 登记一致,705 张、0 警告、2560x1440 全一致 |

## Task Commits

Each task was committed atomically:

1. **Task 1: 体检三指标(比例化)与测试** - `c32407f` (feat)
2. **Task 2: 双来源存疑判定 + manifest 全字段 + 存疑原图副本** - `6348dc5` (feat)
3. **Task 3: ingest 全体检接线 + 跨轮对比表 + 真实数据两轮运行** - `4c25e00` (feat)

**Plan metadata:** 见下方最终提交(SUMMARY.md)

## Files Created/Modified

- `src/calib/io/precheck.py` - 体检核心(292 行):board_roi / exposure_clip / visibility_probe / compute_metrics / judge_suspects,全纯函数,无删除路径
- `src/calib/visualization/suspects.py` - 存疑原图副本 write_suspect_originals(shutil.copy2,源只读不碰;职责归可视化层,io/ 只出判定与名单)
- `src/calib/io/manifest.py` - write_manifest 接受 precheck_results(images 逐张带 precheck 对象、summary 带分桶与成功率)+ _finite_or_none(NaN→null)+ write_round_diff(三桶对比表)
- `src/calib/cli/main.py` - ingest 升级全体检流程(逐张指标 → 判定 → 全字段 manifest → 原图副本 → 跨轮 diff → 中文汇总:分桶计数/分划分成功率/副本与对比表路径)
- `tests/test_io_precheck.py` - 19 用例(293 行):指标方向 7 + 判定与行为锁定 6 + manifest/副本 3 + 跨轮对比 3

## Decisions Made

- **探针结果复用给模糊度定位**(见 key-decisions 第 1 条):计划的 board_roi 三级查找路径完整保留在函数内,compute_metrics 只是传入已有检出免于重复检测——纯性能决定,行为不变
- **缺失值统一写 null**:NaN 会污染 JSON;缺失语义(如无棋盘帧的归一值)在 schema_note 里写明
- **判定来源顺序固定**(pose_relative 在前、global 在后):同一张照片可能同时踩两条来源,固定顺序让清单可读、可 diff
- **首轮即留"首轮运行,无上轮可比"文件**:D-04 要求每轮清单留痕,首轮的占位说明让"从第几轮开始可比"一目了然
- **测试数据构造注意分位线特性**:小样本下 1%/99% 线天然夹住极值,构造正常值时顶部并列(两个最大值相同)避免全局线误伤——这个特性已在 key-decisions 记录,后续写判定类测试的人直接沿用

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 比例化测试的断言方向与合成夹具的成像物理不符**
- **Found during:** Task 1(指标测试首跑)
- **Issue:** 初版断言假设"方格像素边长越大原始拉普拉斯方差越大"(raw 比值 > 3),实测夹具是反的(12px→8133,40px→3219):4 倍超采样+INTER_AREA 缩小的合成图里,边缘密度(每像素的边缘数)对拉普拉斯方差的影响压过了"锐度能量随方格边长平方缩放"的物理——小样本断言会永远失败
- **Fix:** 改为验证比例化的精确契约:blur_norm == blur_raw / 实测方格像素边长²(误差 < 1e-9),叠加"归一值 ≠ 原始值(除以 ~400 必然拉开)";跨尺度的物理稳定性留待真实数据检验(真实 blur_norm 分布已进 SUMMARY)
- **Files modified:** tests/test_io_precheck.py
- **Verification:** test_blur_norm_normalization_formula 通过
- **Committed in:** c32407f(Task 1 提交)

**2. [Rule 3 - Blocking] manifest 测试的血缘指纹引用了不存在的配置文件**
- **Found during:** Task 2(manifest 全字段测试)
- **Issue:** 测试给 write_manifest 传的 config_files 指向临时目录里不存在的 yaml,make_lineage 的 sha256_file 按设计抛 FileNotFoundError(血缘是完整性证据,缺文件必须响)
- **Fix:** 测试先写一份最小 calibration.yaml 再传入
- **Files modified:** tests/test_io_precheck.py
- **Verification:** 三个 manifest 测试转绿
- **Committed in:** 6348dc5(Task 2 提交)

---

**Total deviations:** 2 auto-fixed(1 测试断言修正、1 测试夹具补齐;均不涉产品代码行为变更)
**Impact on plan:** 全部为让验收断言真实成立所必需,无范围蔓延;真实数据验收数值与计划逐字一致(705 张全字段、副本数精确相等、diff 表三小节)。

## Issues Encountered

- worktree 里没有 input_data(数据只在主检出):沿用前几个计划的做法建软链接指回主仓库,gitignore 原样生效,只读契约未破坏;worktree 的 output/ 为空,恰好让"两轮 ingest"从干净状态开始,首轮写"无上轮可比"、第二轮空 diff 的演示链条完整
- 两轮真实 ingest 各约 7.5 分钟(705 张 × 经典检测 + SB 探针 + 曝光/模糊计算,单线程):按计划"给足超时"放后台跑,轮间无人工干预

## Real-Data Verification Record(真实数据验证记录)

- 产物:`output/ingest/run_20261016_073020/`(首轮)与 `output/ingest/run_20261016_073822/`(第二轮),各含 manifest.json + suspect_originals/ + suspect_diff_vs_*.txt
- 计划验收脚本逐条通过:705 条全部带 precheck.verdict 与 suspect_sources;suspect_originals 文件数 89 == verdict==suspect 精确相等;diff 文本含"新增存疑/解除存疑/持续存疑"三小节且非空;第二轮空 diff(新增 0/解除 0/持续 89)= 体检确定性回归验证
- 源码断言:precheck.py 无 `cv2.cornerSubPix(` / `canonicalize(` / `square_size`;precheck.py 与 suspects.py 无 `os.remove` / `.unlink(` / `shutil.rmtree` / `del `(grep 断言全过)
- 单元测试 19/19,全仓 73 过(`-m "not integration"`);ruff 全过

## Known Stubs

无新增占位。`calib ingest` 已是完整实现;noise/fisher/evaluate/analyze/report 五个命令仍是 01-01 建立的计划内占位(D-11 骨架契约),由后续阶段填充。

## User Setup Required

None - 无外部服务配置。

## Next Phase Readiness

- 首轮存疑清单(89 张,来源逐条)+ 原图副本就位:01-05 与人工回环(D-05)可直接按 suspect_originals/ 与 manifest 的 suspect_sources 审"误杀/漏杀",调 config/calibration.yaml 的 ingest.precheck 阈值后重跑,write_round_diff 直接回答"这轮比上轮多了哪些疑点"
- 跨轮回环闭环要件齐:每轮独立 run 目录 + 三桶 diff 表,角点阶段的连拍断言清单(01-02 的 3 个姿态)与体检存疑可交叉比对
- 可见性探针与角点阶段双路径互证(SB 41 帧失明两处一致):Phase 5 寻优素材可信;classic 全检出 ⇒ 现有标定链(classic 角点)不受存疑标记影响
- 供 02-models-noise 注意:存疑帧(尤其 20260717_173936 整组 8 张)在噪声分析与标定时可按 manifest 标记做"含/不含"对照实验

## Self-Check: PASSED

- 5 个关键文件全部存在(3 新建 + 2 修改),行数达标(precheck.py 292 ≥ 90;suspects.py 35 ≥ 12;tests 293 ≥ 60)
- 3 个任务提交(c32407f / 6348dc5 / 4c25e00)全部在 git 历史
- 真实数据产物两轮 run 目录在盘,计划验收脚本(precheck ingest OK)通过
- SUMMARY 中全部数字来自实际运行(manifest 与控制台输出),零硬编码

---
*Phase: 01-data-foundation-minimal-calibration*
*Completed: 2026-10-16*
