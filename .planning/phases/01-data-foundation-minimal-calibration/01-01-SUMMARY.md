---
phase: 01-data-foundation-minimal-calibration
plan: 01
subsystem: io
tags: [opencv, typer, pyyaml, uv, data-ingest, lineage]

# Dependency graph
requires: []
provides:
  - 可安装可测试的项目骨架(pyproject + uv.lock,opencv 锁 4.14/numpy 2.5/scipy 1.18 成对)
  - calib 九命令 CLI 骨架(ingest 已实现,其余占位,后续阶段只填实现不动结构)
  - 姿态数据模型(ImageRecord/scan_dataset*)与韧性扫描(张数数据驱动)
  - config/calibration.yaml 全流程参数默认值 + 人工剔除清单落点(D-03/D-12)
  - 配置解析数据类(BoardSpec/CalibrationConfig,中文校验错误)
  - 血缘指纹机制(make_lineage/sha256_file,ITER-01 地基)
  - 真实数据 manifest(output/ingest/run_*/manifest.json,705 张全登记)
affects: [01-02, 01-03, 02-models-noise, 所有后续阶段(全部消费 manifest/配置/血缘)]

# Tech tracking
tech-stack:
  added: ["opencv-python 4.14.0(锁 <5.0)", "numpy 2.5.4", "scipy 1.18.1", "typer 0.27.3",
          "pyyaml 6.0.3", "matplotlib 3.10.x", "pytest 9.1.1", "ruff 0.17.1", "uv 0.11.32 + Python 3.12.13(钉在 .python-version)"]
  patterns: [目录名即姿态 ID/文件名字典序即拍摄时序, 产物按 run_<UTC时间戳> 分目录留痕,
             配置→不可变数据类→下游不见原始 dict, 每个落盘产物嵌血缘指纹,
             警告留痕不崩溃(空目录/坏文件名/缺划分), image_id 安全模式校验(T-01-02)]

key-files:
  created: [pyproject.toml, uv.lock, .python-version, README.md, config/calibration.yaml,
            src/calib/io/config.py, src/calib/io/dataset.py, src/calib/io/manifest.py,
            src/calib/util/lineage.py, src/calib/cli/main.py,
            tests/test_env_smoke.py, tests/test_io_config.py, tests/test_io_dataset.py, tests/conftest.py,
            src/calib/ 及 11 个子包 __init__.py]
  modified: [.gitignore]

key-decisions:
  - "uv 解释器钉 3.12(.python-version),防止 uv 自动选 3.14 偏离技术栈锁定"
  - "扫描函数双入口:scan_dataset_dirs(配置显式两目录,CLI 用)与 scan_dataset(根目录约定,测试用),同一条代码路径"
  - "缺 train/ 或 test/ 划分目录:按 0 张 + 警告处理(只有训练集的数据是合法输入),两个划分全空才硬报错"
  - "PyPI 直连(经代理)被限速 ~12KB/s,改用清华镜像 UV_DEFAULT_INDEX 环境变量装包;镜像配置未写进仓库(环境级设置)"

patterns-established:
  - "血缘指纹模式:所有产物写盘前调 make_lineage,记录配置哈希/上游哈希/时间戳/轮次号"
  - "中文错误模式:_fail(字段名, 实际值, 要求) 统一格式,配置错误一看就知改哪"
  - "韧性扫描模式:结构性小问题(空目录/坏文件名/缺划分)警告留痕继续跑,数据契约破坏(分辨率不一致)硬报错"

requirements-completed: [IO-01, IO-03]

# Metrics
duration: 55min
completed: 2026-10-16
---

# Phase 1 Plan 1: 行走骨架上半段(环境 + 数据清单)Summary

**uv 项目地基(OpenCV 4.14/numpy 2.5/scipy 1.18 成对锁定)+ `calib ingest` 在全部 705 张真实照片上产出带血缘指纹的 manifest.json(640 训练 + 65 测试、80 姿态 × 8 连拍、0 警告),`calib --help` 即八步流水线中文地图**

## Performance

- **Duration:** 55 分钟(其中约 40 分钟耗在与被限速的 PyPI 直连搏斗,换清华镜像后秒级完成)
- **Started:** 2026-10-16T04:51:06Z
- **Completed:** 2026-10-16T05:46:15Z
- **Tasks:** 3/3
- **Files modified:** 27(新建 26 + 修改 .gitignore)

## Accomplishments

- 环境三件套按锁定落地:Python 3.12.13 + OpenCV 4.14.0(<5.0)+ numpy 2.5.4 + scipy 1.18.1,uv.lock 进 git;STATE.md 的首日待办(uv lock + 冒烟)完成
- **研究假设 A1/A2/A6 全部确认为"可用"**(无 skip):`calibrateCameraExtended`、`findChessboardCornersSB`、`estimateChessboardSharpness` 在本地 OpenCV 4.14.0 均存在——下游计划(01-02 角点、01-03 标定)可直接依赖,无需回退路径
- `calib ingest` 真实数据全量验证通过:705 条记录(640+65)、80 姿态、重复数分布精确等于 {"8": 80}、分辨率 2560x1440 全一致、警告 0 条;全部数值来自实际扫描,代码零硬编码张数
- 24 个单元测试全绿(环境冒烟 5 + 配置/血缘 10 + 数据模型 9),ruff 全过
- 九命令 CLI 骨架立全:帮助文字即八步通俗中文地图(D-11/D-13),未实现命令打印"本命令对应流水线第X步,由后续阶段实现"

## Task Commits

1. **Task 1: 环境落地——pyproject、uv lock、依赖冒烟** - `c93fd13` (feat)
2. **Task 2: 配置解析校验与血缘指纹** - `c589802` (feat)
3. **Task 3: 姿态数据模型、manifest 写盘与 CLI 九命令骨架** - `459051d` (feat)
4. 追加:`740ecc2` (style,ruff 自动修复 UTC 别名)、`942352a` (refactor,scan_dirs 更名贴合计划 key_link 结构契约)

**Plan metadata:** 见下方最终提交(SUMMARY.md)

## Files Created/Modified

- `pyproject.toml` / `uv.lock` / `.python-version` - 依赖锁定(opencv>=4.14,<5.0、numpy==2.5.*、scipy==1.18.* 成对)、calib 入口、pytest marker、解释器钉 3.12
- `config/calibration.yaml` - 全流程算法参数默认值 + exclude_images 人工剔除清单(D-03/D-12),阈值注释标明待迭代回环校准(D-04)
- `src/calib/io/config.py` - board/calibration 解析为不可变数据类,逐键校验,中文错误带字段名与实际值;只用 safe_load(T-01-01)
- `src/calib/io/dataset.py` - ImageRecord/scan_dataset_dirs/scan_dataset/imread_checked/verify_consistent_resolution;韧性规则 + image_id 安全模式(T-01-02)
- `src/calib/io/manifest.py` - manifest.json 写盘(血缘 + 汇总 + images 数组)与 latest_run_dir(下游找上游产物)
- `src/calib/util/lineage.py` - sha256_file + make_lineage(ITER-01 血缘地基)
- `src/calib/cli/main.py` - typer 九子命令,ingest 实现,cv2 线程按配置限制
- `src/calib/**/__init__.py` ×12 - 模块骨架,中文 docstring(布局契约)
- `tests/` - conftest(make_tiny_jpg 夹具)+ 三个测试文件共 24 用例
- `output/ingest/run_*/manifest.json` - 真实数据首轮产物(不进 git)

## Decisions Made

- 解释器钉 3.12:uv 默认解析到 3.14.6,用 `.python-version` 钉回技术栈锁定值
- 扫描函数双入口(scan_dataset_dirs 配置显式两目录 / scan_dataset 根目录约定):CLI 走配置路径(D-12 参数源唯一),测试走根目录约定,共享同一条解析路径
- 缺划分目录降级为警告:允许"只有训练集"的合法数据形态;两个划分全空才报错(防配置路径写错静默产出空清单)
- 镜像走环境变量不进仓库:`UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` 是本机环境事实,不是项目属性;后续会话装包需要它(见 Next Phase Readiness)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] hatchling 构建缺 README.md**
- **Found during:** Task 1(uv sync)
- **Issue:** pyproject 声明 readme = "README.md" 但文件不存在,hatchling 构建直接失败
- **Fix:** 补写 README.md(项目速览 + 快速上手 + 八步流程,通俗中文)
- **Files modified:** README.md
- **Verification:** uv sync 成功
- **Committed in:** c93fd13(Task 1 提交)

**2. [Rule 3 - Blocking] uv 自动选了 Python 3.14.6,偏离技术栈锁定**
- **Found during:** Task 1(uv lock)
- **Issue:** requires-python >=3.12 让 uv 解析到系统里更新的 3.14.6;CLAUDE.md 锁 Python 3.12
- **Fix:** `uv python pin 3.12` 写入 .python-version,重锁
- **Files modified:** .python-version、uv.lock
- **Verification:** uv run pytest 显示 Python 3.12.13
- **Committed in:** c93fd13(Task 1 提交)

**3. [Rule 3 - Blocking] PyPI 直连经代理限速 ~12KB/s,uv sync 卡 40+ 分钟**
- **Found during:** Task 1(uv sync)
- **Issue:** opencv 轮子(~92MB)以约 0.7MB/分钟的速度流入,远超研究预估;磁盘写速实测 1.9GB/s,瓶颈确认在网络
- **Fix:** 终止原进程,清掉 5 个残留半成品临时目录,改用清华镜像环境变量重跑,数秒完成
- **Files modified:** 无仓库文件(环境级操作);uv.lock 内容不变(镜像不改变版本解析)
- **Verification:** 27 包全部安装,版本与锁定一致
- **Committed in:** 无需提交

**4. [Rule 1 - Bug] classic_flags 元组默认值未通过"必须是列表"校验**
- **Found during:** Task 2(测试)
- **Issue:** yaml 缺省时数据类默认值是元组,而校验只认 list,默认值自己把自己拦下了
- **Fix:** isinstance 放宽为 (list, tuple),统一归一成元组
- **Files modified:** src/calib/io/config.py
- **Verification:** test_missing_exclude_images_defaults_to_empty 转绿
- **Committed in:** c589802(Task 2 提交)

**5. [Rule 2 - 韧性补强] 缺 train/ 或 test/ 目录原为硬报错**
- **Found during:** Task 3(测试)
- **Issue:** 只有训练集(暂无独立测试集)是合法数据形态,原实现直接抛错过严;与"空目录警告跳过"的韧性精神不一致
- **Fix:** 单个划分缺失→警告 + 按 0 张;两个划分全空→保留硬报错(防配置路径写错静默产出空清单)
- **Files modified:** src/calib/io/dataset.py
- **Verification:** 24 测试全绿;真实 ingest 行为不变
- **Committed in:** 459051d(Task 3 提交)

---

**Total deviations:** 5 auto-fixed(2 blocking-环境、1 blocking-构建、1 bug、1 韧性补强)
**Impact on plan:** 全部为让计划落地所必需,无范围蔓延;真实数据上的验收数值与计划逐字一致。

## Issues Encountered

- worktree 里没有 input_data(gitignored,数据只在主检出):用指向主仓库的软链接解决,gitignore 原样生效,数据只读契约未破坏
- 测试初稿里多处目录树路径漏了 train/ 层级(测试自身笔误,非产品代码问题),逐处修正;顺带发现并修复了第 5 条偏差
- .gitignore 首版写成 `/output/`,计划的验收 grep 要求 `^output/`,按计划字面规格改为 `output/`(忽略范围等价)

## Known Stubs

全部为**计划内显式占位**(D-11:八命令骨架一次立全,后续阶段只填实现不动结构):

- `src/calib/cli/main.py` 的 corners/noise/calibrate/fisher/evaluate/analyze/report 七个命令:打印"本命令对应流水线第X步,由后续阶段实现"后正常退出——计划明确规定,01-02/01-03 及 Phase 2-6 填充
- manifest 的 images 数组暂无 precheck 对象:schema_note 字段已注明预留(体检指标属下一计划)
- 血缘的 upstream 字段在本阶段为空字典:ingest 的上游是原始数据(不哈希 451MB 图像),从 corners 阶段起上游产物开始入指纹

## User Setup Required

None - 无外部服务配置。(环境提示:后续会话若需重新装包,镜像变量 `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` 可避限速;uv.lock 已提交,常规 `uv sync` 不联网也能装。)

## Next Phase Readiness

- manifest.json 结构(血缘/汇总/images 五键)与 latest_run_dir 已就位,01-02(角点)与 01-03(标定)可直接消费
- A1(estimateChessboardSharpness 可用)/ A2(SB 检测器可用)/ A6(calibrateCameraExtended 可用)三条研究假设已本地确证,下游无回退路径之忧
- CalibrationConfig 已含 corners/calibrate 段默认值,01-02/01-03 直接读
- 唯一环境注意事项:PyPI 直连限速(见 Deviations #3),已用镜像解决且依赖已全部就位

## Self-Check: PASSED

- 15 个关键产物文件全部存在(pyproject/uv.lock/源码/测试/SUMMARY)
- 6 个提交(c93fd13 / c589802 / 459051d / 740ecc2 / 942352a / 2932579)全部在 git 历史
- 真实数据首轮产物目录 output/ingest/run_20261016_054402 等 3 轮留痕在盘

---
*Phase: 01-data-foundation-minimal-calibration*
*Completed: 2026-10-16*
