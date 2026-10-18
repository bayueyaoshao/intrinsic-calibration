---
phase: 02-m1-m4
plan: "05"
subsystem: calibration-pipeline
tags: [calibrate, matrix-orchestration, lineage, cli, d01-config-matrix, d12-zero-params, integration, determinism]

# Dependency graph
requires:
  - phase: 02-m1-m4(02-02)
    provides: ObservationSet 冻结数据类 + load_observation_set(source 键路由)
  - phase: 02-m1-m4(02-03)
    provides: CalibrationResult + save_result 三件套契约 + compute_result_metrics 同一把尺 + make_failure 失败契约
  - phase: 02-m1-m4(02-04)
    provides: solve_with_scipy(trf+lsmr+解析稀疏雅可比+实用收敛判定,rms_linear_ref 对照钩子)+ ScipyConfig.losses
  - phase: 01(阶段一)
    provides: calib corners 产物(真实 run_20261017_104817,640 视图 sb 角点)与 ingest 清单——本计划试车的真实上游
provides:
  - solve/matrix.py:run_calibration_matrix 全矩阵编排(逐组合三件套带血缘落盘,返回 comparison 行)、combo_dir_name 命名唯一来源(<模型>__opencv / <模型>__scipy / <模型>__scipy_<去下划线损失名>)、_load_init_result 落盘初值读回(行序契约断言)
  - CLI calibrate 演进:血缘优先定位清单、剔除名单生效、逐组合播报、comparison.json 落盘、中文汇总表 + 三句白话注解(同模型比较 / 模型失配预期 / soft_l1 实用收敛)
  - 真实数据全矩阵试车一轮(run_20261018_022047,12 组合全部落盘)与全部并排数字(本文件下方表格——阶段三 Fisher 线性化点、阶段四评估的直接输入)
  - 确定性实证:同输入两轮独立进程,m4__opencv 与 m4__scipy 的 RMS 与逐参数差异 <1e-12(integration 测试锁死)
affects: [03-评估(Fisher 信息取 m4__scipy 线性化点,复用 comparison/result 产物与命名), 04-诊断(敏感度用同一批产物), report(证据链引用血缘链)]

# Tech tracking
tech-stack:
  added: []   # 无新依赖;编排层只做既有部件的组合
  patterns:
    - 初值从落盘产物读回(load_result + load_extrinsics)而非内存直递:血缘里指向的那份文件与实际消费的那份数值严格是同一份,文件是唯一真相源
    - 编排层兜底宽接(Exception → make_failure 中文契约落盘,单组合病不拖死整批):ruff BLE001 双 noqa 带计划出处注释——窄接反而违背计划明文
    - init_from 两级记号:求解层内部记号(模型__来源 / internal_generic)仅供排障,编排层收敛后升级为产物相对路径(<模型>__opencv/result.json)——血缘对账认路径
    - 鲁棒核对照参照按"同轮同模型 linear 先跑"供给:ref 只给非 linear 损失,linear 不做"对照自己"
    - upstream 记初值产物指纹的前提是该文件本轮真的写过:OpenCV 不在 solvers 时整个键不出现,不伪造指向不存在文件的指纹

key-files:
  created:
    - src/calib/solve/matrix.py
    - tests/test_calibrate_matrix.py
  modified:
    - src/calib/cli/main.py
    - tests/test_smoke_realdata.py

key-decisions:
  - "真实数据 soft_l1 两路径未达严格线(计划成功标准 3 的严格判据):m1__scipy_softl1 实用收敛判定拒绝(评估上限 300 次、尾部改进 0、回代 RMS 超 linear 参照 5.8%>5% 守门),m2__scipy_softl1 实用收敛通过但线性尺 RMS 比纯 OpenCV 高 2.77e-4(0.30%)——按计划明文预案执行:发现已落盘(failure_note/solver_meta 全量留痕),SUMMARY 给 linear 路径证据,提交人工裁定;数学上 soft_l1 优化的是另一目标,线性尺 RMS 略高属预期,STACK 的'SciPy 求解器异常当发现记录不当缺陷抑制'即此情形"
  - "scipy 初值协议:同轮 <模型>__opencv 落盘产物读回 + extrinsics.npz 行序与 result.json 的 image_ids 逐项断言(错位即抛不带病精化);本轮没跑 OpenCV 或未收敛时走通用初始化,solver_meta.init_source 记 same_round_opencv / generic_default 不静默"
  - "test_smoke_realdata.py 的 calibrate 腿随矩阵演进重写(Rule 1):基线单目录时代的三件套断言在 12 组合产物布局下必挂,波次合并前修复;baseline.py 本身不 import 不修改(向后兼容)"
  - "全矩阵真实运行产物从工作树复制留存主仓库 output/calibration/run_20261018_022047(12MB,comparison 逐字节核对一致):工作树返回后即被移除,产物保真规则要求证据链留盘"

patterns-established:
  - "矩阵编排产物的消费入口统一走 combo_dir_name 与 comparison.json,阶段三/四不自己拼目录名"
  - "integration 测试搭真实链路临时树时,上游产物按血缘记录的相对路径原样落位(不是随便放),meta 的'血缘优先'定位才能走通与真实相同的代码路径"

requirements-completed: []   # 勾选由 orchestrator 波次合并后统一执行(并行执行器不改共享规划产物)

# Metrics
duration: 34min
completed: 2026-10-18
---

# Phase 2 Plan 05: calibrate 全矩阵编排 Summary

**配置矩阵驱动的全矩阵标定编排(模型 × OpenCV/scipy × linear/soft_l1 逐组合三件套带血缘落盘 + comparison.json + 中文汇总表)+ 真实 640 视图 12 组合试车 21 分 41 秒全部落盘 + 独立重跑确定性 <1e-12 实证;soft_l1 两路径未达严格线按计划预案记录待人工裁定,linear 路径全部严格通过**

## Performance

- **Duration:** 约 34 分钟(2026-10-18T02:10:42Z → 02:44Z,worktree 并行波次;其中全矩阵真实运行后台占用 21 分 41 秒,预期 15–25 分钟内)
- **Started:** 2026-10-18T02:10:42Z
- **Completed:** 2026-10-18T02:44:30Z
- **Tasks:** 3/3
- **Files modified:** 4(新建 2,修改 2)

## Accomplishments

- **编排库(Task 1):** `solve/matrix.py` —— run_calibration_matrix 读配置矩阵逐组合求解:每模型先 OpenCV(给 scipy 当初值)再 scipy 各损失核,"精化前后"在汇总表永远并排相邻(同模型比较,Pitfall 8);每个组合三件套(result.json / extrinsics.npz / residuals.npz)带血缘落盘——upstream 记角点 npz + (scipy 时)同轮初值 result.json 的路径与 sha256 + 剔除名单留痕(T-03-02);棋盘规格与角点数装配核对(防"板子配错角点文件"接线事故);单组合任何编排层意外转中文失败契约落盘,绝不中断其余组合;cv2 线程数从配置读(D-12)
- **CLI 演进(Task 1):** calibrate 零参数跑全矩阵——血缘优先定位数据清单(meta 记录路径优先于目录约定)、剔除名单生效并进血缘、逐组合一行播报、comparison.json 落盘、中文汇总表(组合/视图数/整体RMS/收敛)+ 三句白话注解(同模型比较、M1/M2/M3 模型失配预期发现、soft_l1 实用收敛)
- **合成端到端测试(Task 2):** 临时树(裁剪 config:m2 × 三路径)+ 合成 6 视图角点(带标准答案)独立进程跑 calibrate——退出码 0、三组合目录三件套、comparison 三行、血缘 sha256 重算一致、init_from 指向同轮产物 + 9 参数键、scipy 两损失 RMS ≤ 纯 OpenCV + 1e-9(4 用例,1.3 秒)
- **integration(Task 3):** 真实上游产物按血缘相对路径落位的临时树上,m4 缩小矩阵两轮独立进程——RMS 与逐参数差异 <1e-12(确定性,成功标准 3)、两轮血缘指纹与实际文件重算一致、init_from 非空(成功标准 4)
- **真实全矩阵试车(Task 3,执行器操作):** 12 组合全部落盘,退出码 0,耗时 21 分 41 秒(预期 15–25 分钟);血缘三要素(created/config_sha256/upstream)12/12 齐全;未收敛组合(m1__scipy_softl1)结构 14 键全 + 中文 failure_note;m4__opencv 回代 RMS 0.0857px 与阶段一 sb fisheye 基线(~0.08px)同量级;产物已留存主仓库 output/calibration/run_20261018_022047

## 真实数据全矩阵并排数字(阶段三/四直接引用)

视图数全部 640,角点路线 sb,图像 2560x1440,轮次 run_20261018_022047(2026-10-18T02:20:46Z → 02:42:27Z):

| 组合 | 收敛 | 回代 RMS(像素) | scipy 状态 | 精化 vs 同模型 OpenCV |
| --- | --- | --- | --- | --- |
| m1__opencv | 是 | 0.23551719828213488 | — | 基准 |
| m1__scipy | 是 | 0.23551719828213488 | status=3,nfev=2 | 逐位一致(已到最优,初值即解) |
| m1__scipy_softl1 | **否** | —(失败契约) | status=0,nfev=300,尾部改进 0 | 回代 RMS 0.2491 超 linear 参照 5.8%(>5% 守门,实用收敛拒绝) |
| m2__opencv | 是 | 0.09186292401766323 | — | 基准 |
| m2__scipy | 是 | 0.09186284584467135 | status=0,nfev=300,尾部改进 1.65e-8 | 优 7.8e-8(严格线内) |
| m2__scipy_softl1 | 是 | 0.09213934828836506 | status=0,nfev=300,尾部改进 0 | 高 2.77e-4(+0.30%,超 +1e-6 严格线) |
| m3__opencv | 是 | 0.08539740220888348 | — | 基准 |
| m3__scipy | 是 | 0.08539740217815185 | status=2 | 优 3.1e-11 |
| m3__scipy_softl1 | 是 | 0.08543189860327219 | status=0 | 高 3.4e-5(实用收敛内) |
| m4__opencv | 是 | 0.08569590636042337 | — | 基准(阶段一基线量级核对:~0.08px 吻合) |
| m4__scipy | 是 | 0.08569590636042337 | status=3,nfev=3 | 逐位一致(已到最优) |
| m4__scipy_softl1 | 是 | 0.08573087965363672 | status=0 | 高 3.5e-5(实用收敛内) |

**读法(白话):** 每行 scipy 的对比只跟同模型的 OpenCV 基准比(精化前后),不跨模型比;M1/M2/M3 是针孔模型,在强鱼眼镜头上整体 RMS 明显高于 M4 属"模型不合身"的预期发现,正式判定在阶段四防作弊考试。**linear 路径全部严格通过**(m1 逐位一致、m2 略优、m3 略优、m4 逐位一致)——精化链路本身工作正常。

**soft_l1 人工裁定材料(计划明文预案):** 严格判据要求 m1/m2 的 soft_l1 收敛且回代 RMS ≤ 同模型 OpenCV + 1e-6。实测:
1. m1__scipy_softl1 未收敛——到达 300 次评估上限,尾部改进 0,回代 RMS 0.2491 比 linear 参照 0.23552 差 5.8%,被"鲁棒核没把解压歪"的 5% 守门拒绝(判定依据全量在 result.json 的 solver_meta:nfev/tail_improvement/rms_linear_ref/init_source=same_round_opencv)。m1 是失配最重的针孔模型(残差分布重尾),soft_l1 把大量角点降权后步进方向与残差尺度耦合,评估预算内没走到平点;
2. m2__scipy_softl1 实用收敛(尾部走平 + RMS 只比 linear 参照高 0.30%,在 5% 守门内),但线性尺 RMS 比纯 OpenCV 高 2.77e-4,超 +1e-6 严格线。数学上 soft_l1 优化的是另一条目标(把大残差权重压低),它 的解在线性尺上略差是结构性预期,不是实现缺陷。
**linear 路径证据(成功标准 3 的实证):** m1__scipy = m1__opencv(逐位)、m2__scipy 优 7.8e-8、m3__scipy 优 3.1e-11、m4__scipy = m4__opencv(逐位)——OpenCV 初值 + scipy 精化链路在全矩阵上"不劣化"成立;soft_l1 是否放宽严格线或调整 f_scale/评估上限,留人工裁定(裁定入口:output/calibration/run_20261018_022047 各 result.json)。

## Task Commits

Each task was committed atomically:

1. **Task 1: 全矩阵编排库 + CLI 演进 + 冒烟测试跟进** - `7ce96dd` (feat)
2. **Task 2: 合成数据端到端编排与血缘契约测试** - `c288bcb` (test)
3. **Task 3: 真实数据全矩阵试车 + 独立重跑一致性(integration)** - `192787a` (test)

**Plan metadata:** 见本文件提交(docs)

## Files Created/Modified

- `src/calib/solve/matrix.py` - 全矩阵编排:命名唯一来源/落盘初值读回/血缘组装/逐组合兜底(310 行)
- `src/calib/cli/main.py` - calibrate 命令演进:血缘优先清单定位/矩阵调用/逐组合播报/comparison/中文汇总表
- `tests/test_calibrate_matrix.py` - 合成端到端 4 用例(快速)+ 真实 m4 两轮确定性/血缘 2 用例(integration,共 339 行)
- `tests/test_smoke_realdata.py` - calibrate 腿随 12 组合布局重写(组合目录/血缘第二环/初值指纹/失败组合结构)

## Decisions Made

- 初值读回协议选"落盘产物"而非内存直递:血缘 upstream 指向的文件与实际消费的数值严格同一份,对账语义才成立;代价是多一次磁盘读,可忽略
- upstream 的 init_result.json 键在"本轮跑过 OpenCV 组合"时记录(即使该组合未收敛,失败契约也写了 result.json,诚实留痕);OpenCV 不在 solvers 时键不出现——不伪造不存在文件的指纹
- 失败组合的 init_from 保持 None(没收敛就没有可指认的初值产物),但 solver_meta.init_source 仍记实际初值来源——来源与结果分开留痕
- 鲁棒核参照只给非 linear 损失:linear 自己不做"对照自己";同轮同模型 linear 先跑的 RMS 存 dict 供后续损失取
- 真实全矩阵运行产物复制留存主仓库(工作树将被移除,证据链必须留盘;comparison 逐字节核对一致)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_smoke_realdata.py 的 calibrate 腿随矩阵演进重写**
- **Found during:** Task 1
- **Issue:** 冒烟测试的标定断言停留在基线单目录时代(calibration_run/<model>/ 三件套),矩阵演进后产物布局变为 calibration_run/<model>__<solver>/ × 12 + comparison.json,旧断言必挂(波次合并后 integration 全量跑会显形)
- **Fix:** 重写为 12 组合目录遍历 + 血缘第二环(角点指纹 + scipy 初值产物指纹指向同模型 __opencv)+ 失败组合结构完整性;MATRIX_COMBOS / RESULT_JSON_KEYS 常量同步布局
- **Files modified:** tests/test_smoke_realdata.py
- **Commit:** 7ce96dd

**2. [Rule 3 - Blocking] ruff BLE001 对编排层兜底宽接的误伤**
- **Found during:** Task 1
- **Issue:** 环境实际执行的 ruff 0.17.1 规则面超出项目文档记载的 E4/E7/E9/F 默认,两条 `except Exception` 兜底(matrix.py 两处)被 BLE001 拦截;窄接异常面会让"单组合任何意外都不中断其余组合"的计划明文失效
- **Fix:** 双 `# noqa: BLE001` 带计划出处注释(故意宽接:单组合任何异常都转失败契约落盘,绝不中断其余组合)
- **Files modified:** src/calib/solve/matrix.py
- **Commit:** 7ce96dd

**3. [Rule 3 - Blocking] 工作树缺 input_data / output 运行时数据(gitignored 不随工作树分发)**
- **Found during:** Task 3 准备
- **Issue:** integration 测试与真实全矩阵运行依赖仓库的 input_data(只读)与 output/corners、output/ingest 上游产物,工作树里没有
- **Fix:** 建运行时软链接(input_data、output/corners、output/ingest → 主仓库对应目录,只读语义不变);output/calibration 保持工作树独立真实目录(新产物不写主仓库);真实全矩阵产物事后复制留存主仓库(见 Decisions)
- **Files modified:** 无仓库内文件(软链接在 gitignore 内)

### 计划内但值得记录的经验发现(非缺陷)

- soft_l1 两路径未达严格线:见上文"soft_l1 人工裁定材料"——计划已预案该情形(实测不收敛则落盘 + linear 证据 + 人工裁定),不属偏差

## TDD Gate Compliance

- Task 2 带 tdd="true",但计划的任务排序是"实现(Task 1)先行、集成测试(Task 2)后验"——矩阵编排库在测试文件创建前已存在并提交(7ce96dd),RED 阶段(先写失败测试)在本计划的执行顺序下不适用:测试文件落地后 4 用例首次运行即绿,无"测试意外通过"需要 fail-fast 排查(被测对象 Task 1 已交付是计划明文安排,不是测试失效)
- 门序列核对:本计划单个 test 提交(c288bcb)晚于 feat 提交(7ce96dd)——与 TDD 惯例的 RED→GREEN 顺序相反,按计划任务顺序执行;单元级 RED/GREEN 已在 02-04(求解器层)完成,本计划的测试是编排级集成验证

## Issues Encountered

- 真实全矩阵运行中 m1__scipy_softl1 未收敛:不是执行器问题,是数据×模型×鲁棒核的实测发现(判定依据全量落盘),按计划预案提交人工裁定
- 运行时长 21 分 41 秒,在研究预期 15–25 分钟内(与 integration 测试并发跑,互不干扰:cv2 线程已关,单组合单线程)

## User Setup Required

None - 无外部服务配置。

## Requirement Traceability

- 计划 frontmatter 未声明 requirements 字段(02-05 是管线演进计划,交付物为编排库与试车证据)——无勾选动作;SOL-01 相关的"全矩阵跑通"证据(12 组合落盘 + 确定性)已在本 SUMMARY 留档,阶段三/四直接消费

## Next Phase Readiness

- 03-评估(Fisher 信息):线性化点取 m4__scipy 的产物(逐位等于 OpenCV 解,双后端一致的最优点);解析雅可比(02-04 residuals.py)直接复用;comparison.json 是模型选型的入口表
- 04-诊断:逐组合 residuals.npz(V×88×2)已落盘,误差预算与敏感度分析的原料齐
- soft_l1 裁定项:见"soft_l1 人工裁定材料"——若人工裁定放宽严格线或调 f_scale/评估上限,只动 config/calibration.yaml(scipy 段),编排零改动
- 阻塞:无

---
*Phase: 02-m1-m4*
*Completed: 2026-10-18*

## Self-Check: PASSED

自检于提交前完成:4 个产出文件(新建 2 + 修改 2)、SUMMARY 本体与 3 个任务提交(7ce96dd / c288bcb / 192787a)均确认存在于本分支;真实全矩阵产物核对(12 组合目录 / comparison 12 行 / 血缘三要素 12/12 / 失败组合 14 结构键全)与留存副本(comparison 逐字节一致)均通过脚本验证。
