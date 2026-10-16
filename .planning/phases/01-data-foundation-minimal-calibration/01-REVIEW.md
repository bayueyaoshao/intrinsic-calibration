---
phase: 01-data-foundation-minimal-calibration
reviewed: 2026-10-16T08:40:50Z
depth: standard
files_reviewed: 39
files_reviewed_list:
  - .gitignore
  - .python-version
  - README.md
  - config/board.yaml
  - config/calibration.yaml
  - docs/pipeline.md
  - pyproject.toml
  - src/calib/__init__.py
  - src/calib/cli/__init__.py
  - src/calib/cli/main.py
  - src/calib/corners/__init__.py
  - src/calib/corners/artifacts.py
  - src/calib/corners/canonical.py
  - src/calib/corners/detect.py
  - src/calib/corners/window.py
  - src/calib/io/__init__.py
  - src/calib/io/config.py
  - src/calib/io/dataset.py
  - src/calib/io/manifest.py
  - src/calib/io/precheck.py
  - src/calib/models/__init__.py
  - src/calib/solve/__init__.py
  - src/calib/solve/baseline.py
  - src/calib/util/__init__.py
  - src/calib/util/lineage.py
  - src/calib/visualization/__init__.py
  - src/calib/visualization/mosaic.py
  - src/calib/visualization/overlay.py
  - src/calib/visualization/suspects.py
  - tests/conftest.py
  - tests/test_corners.py
  - tests/test_corners_artifacts.py
  - tests/test_env_smoke.py
  - tests/test_io_config.py
  - tests/test_io_dataset.py
  - tests/test_io_precheck.py
  - tests/test_smoke_realdata.py
  - tests/test_solve_baseline.py
  - tests/test_visualization.py
findings:
  critical: 1
  warning: 4
  info: 7
  total: 12
status: issues_found
---

# Phase 01:代码审查报告

**审查时间:** 2026-10-16T08:40:50Z
**深度:** standard
**审查文件数:** 39
**状态:** issues_found

## Summary

对第 01 阶段(数据地基 + 走通骨架)提交的全部源码、测试、配置与文档做了逐文件审查(standard 深度,含跨模块调用链追踪)。整体判断:代码质量高于平均水平——安全防线(npz 禁 pickle、yaml 只用 safe_load、image_id 白名单)真实有效且被测试锁定;源码防线测试(精化调用唯一性、safe_load、square_size 解耦)是有价值的做法;合成数据回归测试带真值断言,不是摆设;83 个非集成测试全部通过,ruff lint 零告警,无调试残留、无裸 print、无硬编码密钥。

但对抗性审查确认了 1 个必修缺陷:corners 命令在"只有训练集、没有测试集"的合法输入上直接崩溃(空划分的 numpy 索引类型错误)——这正是 dataset.py 明文承诺支持的输入。另有 4 个应当修复的问题:存疑拼图与存疑原图副本的跨阶段路径断链(被测试夹具按错误布局掩盖)、yaml 语法错误击穿两道中文报错防线、针孔求解与鱼眼的"失败记录不中断"契约不对称、体检回退路径硬编码检测器选项违反 D-12。7 个低优先级问题见 Info 节。

验证方式说明:关键结论均经本机实测确认(numpy 空数组索引抛 IndexError;yaml.ScannerError 不是 ValueError 子类;`grep suspect_originals` 全链路交叉比对)。

## Critical Issues

### CR-01:corners 命令在空划分(如只有 train 没有 test)上崩溃,违背数据层承诺的合法输入

**文件:** `src/calib/cli/main.py:332-336`(同模式还有 342-343)
**问题:** 数据扫描层(`src/calib/io/dataset.py:73-78`)明确承诺"划分目录整个不存在……比如只有训练集、暂无独立测试集的数据是合法输入",ingest 也能正常产出 n_test=0 的数据清单。但 corners 命令的检测质量摘要里,空划分会构造出 `np.array([])`——空列表推断成 **float64** 类型数组,用它做花式索引当场抛 `IndexError: arrays used as indices must be of integer (or boolean) type`,整个 corners 命令崩溃(连带 run-all 中断)。本机已实测复现该行为。代码里 `success_rate` 的 `if rows.size else None`(341 行)说明作者预期过空划分,但索引在到达这个保护之前就炸了。落盘路径(318 行)用的是 Python 列表,实测 `uv[[]]` 不报错——所以只有摘要这一段带病。

**复现路径:** `input_data/test` 缺失或为空 → `calib ingest` 正常(警告留痕)→ `calib corners` 崩溃。

**修复:**
```python
# src/calib/cli/main.py:332(显式声明整数类型,空划分时是空 int 数组,索引合法)
rows = np.array([i for i, rec in enumerate(images) if rec["split"] == split], dtype=int)
```
建议补一条"test 为空时 corners 照常产出 4 份 npz(该划分为 0 行)+ 摘要 success_rate=null"的测试锁定。

## Warnings

### WR-01:存疑拼图在错误的运行目录里找 suspect_originals/,跨阶段产物断链被测试夹具掩盖

**文件:** `src/calib/visualization/mosaic.py:237`(注解文案 264 行;写入方 `src/calib/cli/main.py:129`;掩盖性夹具 `tests/test_visualization.py:251`)
**问题:** 存疑原图副本由 ingest 阶段写入 **ingest** 运行目录(`output/ingest/run_X/suspect_originals/`,docs/pipeline.md 阶段契约也把它归在第①步产物)。但存疑拼图 `suspect_mosaic` 在 **corners** 运行目录下找它(`corners_run_dir / "suspect_originals" / ...`)——真实流水线里这个路径永远不存在,于是每一张都静默走"退回原始路径"分支。今天内容恰好等价(copy2 的就是原图,画面无差),但:① mosaic_index.json 的说明(264 行)指引读者"存疑照片的未标记原貌在 suspect_originals/"——在 corners 运行目录里找不到这个文件夹,指引落空;② "与角点标记副本互相印证看未标记原貌"的设计意图实际从未接通;③ 测试夹具(test_visualization.py:251)把 suspect_originals 写在 corners 运行目录下——按**错误布局**构造,恰好把断链掩护住了。

**修复:** 让 `suspect_mosaic` 拿到 ingest 运行目录(优先方案:从 corners 产物 meta 的 lineage.upstream["manifest.json"].path 解析出 manifest 所在目录——`_manifest_path_for`(cli/main.py:409)已有同款先例;或由 CLI 显式传入),并把测试夹具改为按真实布局(ingest 目录)构造、另加一条"corners 目录下没有 suspect_originals 时仍取到副本"的回归测试。

### WR-02:yaml 语法错误击穿两道中文报错防线(--help 也会裸栈崩溃)

**文件:** `src/calib/io/config.py:169-175`(`_read_yaml`)、`src/calib/cli/main.py:71-76`(`_setup_runtime`)
**问题:** config.py 头部承诺"解析失败必须给出带字段名与实际值的通俗中文错误",但只覆盖了**结构校验**失败(ValueError);yaml **语法**错误(如缩进用了 tab)由 `yaml.safe_load` 抛 `yaml.ScannerError`——实测确认它不是 ValueError 子类。后果有二:① `_read_yaml` 不捕获,用户拿到的是 PyYAML 的原始英文堆栈,违反本项目自己的报错契约;② 命令回调 `_setup_runtime` 的 `except (ValueError, OSError)` 接不住它,而该回调的注释明确承诺"--help / 配置还没就位时不拦着用户"——`calib ingest --help` 这类本不该碰配置语义的调用,会因一个坏 yaml 直接崩溃。

**修复:**
```python
# config.py::_read_yaml
try:
    return yaml.safe_load(f)
except yaml.YAMLError as e:
    raise ValueError(
        f"配置文件 {path} 的 yaml 语法有误,请检查缩进与冒号:{e}"
    ) from e
```
转换成 ValueError 后,`_setup_runtime` 现有的 `except (ValueError, OSError)` 即自然兜住,两道防线同时修复。

### WR-03:针孔求解不捕获 cv2.error,与鱼眼"失败记录不中断"的既定契约不对称

**文件:** `src/calib/solve/baseline.py:94-95`(对照鱼眼分支 137-147 的 try/except)
**问题:** D-10 的契约是"失败会被完整记录(结构键一个不少、数值字段一律 None),绝不中断其余组合",鱼眼分支照此实现了 try/except cv2.error + 中文 failure_note。但 `calibrate_pinhole` 对 `cv2.calibrateCameraExtended` 没有任何保护。而 `run_baselines` 自己的准入门槛只有"可用视图 ≥ 2"(229 行)——两三个近平面视角加 5 畸变系数正是 calibrateCamera 的病态区,一旦它抛 cv2.error:报错是原始 C++ 堆栈而非中文,四个组合全部中断,run-all 跟着断,与模块头部"失败会被完整记录……绝不中断其余组合"的承诺直接矛盾。

**修复:** 与鱼眼同构包裹:
```python
try:
    rms, K, D, rvecs, tvecs, _s_int, _s_ext, per_view = cv2.calibrateCameraExtended(
        objs32, imgs32, tuple(image_size), None, None, flags=flags)
except cv2.error as e:
    return {"rms_overall": None, "K": None, "D": None, "rvecs": None, "tvecs": None,
            "per_view_rms": None, "flags_used": flags, "converged": False,
            "failure_note": f"针孔模型求解未收敛:OpenCV 报错——{e}"}
```

### WR-04:体检回退路径硬编码检测器选项,违反 D-12;classic 未检出而 SB 检出的照片模糊度通道永久失明

**文件:** `src/calib/io/precheck.py:80-86`(`board_roi` 回退)、`src/calib/io/precheck.py:163-165`(`compute_metrics` 只喂 classic 角点)
**问题:** 三处叠加缺陷:
1. `board_roi` 的完整查找硬编码 `CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE`,无视 `cfg.classic_flags`。D-12 硬性规则是"换数据集只改配置不改代码",用户改了配置后,可见性探针按配置跑、棋盘框定位按写死选项跑——同一帧内两个口径的 classic 结论可能不一致,阈值迭代回环(D-04)会被这种不一致干扰。
2. 探针失败后的冗余重跑:classic 探针已用配置选项失败,`board_roi` 还要用 FAST_CHECK + 硬编码选项把**同一颗检测器**再跑最多两遍(几百张图时是纯浪费),且从不尝试用已到手的 `sb_corners` 定位棋盘框。
3. 结果是 classic 未检出而 SB 检出的照片,`roi_found=False` → `blur_norm=NaN` → 同姿态模糊度 z 分数与全局分位线两条模糊度判定来源对它双双失明,哪怕 SB 已经把棋盘框了出来。box_area_ratio 的文档写"谁检出用谁,经典优先",模糊度却只用 classic——同一模块内口径不一致。

**修复:** 回退查找改用现有 `_CLASSIC_FLAG_VALUES` 按配置组装(与 `visibility_probe` 同一来源);`compute_metrics` 在 classic 失败而 `sb_ok` 时把 `probe["sb_corners"]` 喂给 `board_roi`。顺带消除每张失败图最多两次的白跑。

## Info

### IN-01:21/36 个源码与测试文件不符合 ruff format

**文件:** 全仓库(例:`src/calib/cli/main.py`、`tests/test_visualization.py` 等 21 个)
**问题:** CLAUDE.md 技术栈表声明 ruff 用于 "Lint + format",`ruff check` 全绿,但 `ruff format --check` 报 21 files would be reformatted(续行缩进、空行等风格漂移)。格式化不统一会让后续每一轮 diff 夹带噪音。
**修复:** 跑一次 `uv run ruff format src tests` 收敛,并考虑在 CI/收尾任务里加 `ruff format --check`。

### IN-02:blur_z_threshold 名不副实——同一个阈值实际同时管曝光

**文件:** `src/calib/io/config.py:53-54`、`config/calibration.yaml:26-29`(对照实现 `src/calib/io/precheck.py:237、252`)
**问题:** 配置字段与两侧文档都写"模糊度稳健 z 分数的报警线",但 `judge_suspects` 把它同时用于 expo_hi/expo_lo 的 z 分数判定。想单独收紧曝光灵敏度的人改不动它,文档误导调参。
**修复:** 要么文档改为"模糊度与曝光共用的稳健 z 分数报警线",要么拆成两个阈值键。

### IN-03:源码防线测试用 CWD 相对路径,换目录跑 pytest 会误报

**文件:** `tests/test_corners.py:157`、`tests/test_corners_artifacts.py:148`
**问题:** 读源码断言用 `pathlib.Path("src/calib/corners/detect.py")`(相对当前工作目录),而同仓库其他测试(test_io_config.py:16、test_solve_baseline.py:29)统一用 `REPO_ROOT = Path(__file__).resolve().parent.parent`。从仓库根以外目录启动 pytest(IDE 测试面板、`--project` 方式)这两条防线会假失败。
**修复:** 三处统一改用 REPO_ROOT 拼路径。

### IN-04:全流水线的"断言防线"用裸 assert,`python -O` 会整体剥掉

**文件:** `src/calib/corners/window.py:53、107`、`src/calib/solve/baseline.py:60`
**问题:** 头号陷阱防线(窗口越界当场报错)与形状闸门依赖 `assert` 语句;Python 以 `-O` 运行时所有 assert 被剥离,防线静默消失,带病窗口会无声流进标定——恰是本项目最怕的失败模式。当前 `uv run` 不带 -O,风险未触发。
**修复:** 改为 `raise ValueError(...)`(消息已是现成中文,直接搬),语义不变且不可被优化掉。

### IN-05:SB 路线的"仅记录"窗口字段会提前引爆逐帧断言;corner 粒度下记录值破坏产物契约

**文件:** `src/calib/corners/detect.py:101`(对照 artifacts.py:8 的契约文案)
**问题:** 两点:① `detect_sb` 不用窗口,却仍调 `derive_window_frame`——方格极小的帧上,经典路线在 corner 粒度下本可幸存(逐点断言按局部边长),SB 却会因帧级断言先炸掉整条 corners 命令;② corner 粒度下记录的标量 `win_size = max(wins)` 来自**局部**边长最大者,可能不再满足 artifacts.py:8 承诺的审查关系"2*win+1 < sq_px(帧级保守值)",下游照契约复核会误报(test_detector_window_rule_on_synthetic 只覆盖了 frame 粒度)。
**修复:** SB 的 win_size 记录改为"推导但不断言"的旁路;artifacts.py 文档注明 corner 粒度下该字段含义(逐点窗口的最大值,需按局部边长复核)。

### IN-06:存疑计数逻辑两处重复实现,漂移风险

**文件:** `src/calib/cli/main.py:38-49`(`manifest_suspect_counts`)、`src/calib/io/manifest.py:90-122`(`_precheck_summary`)
**问题:** "按来源前缀分桶、同族只计一张"的规则在 CLI 与 manifest 模块各写了一遍。将来桶规则调整(如新增来源族)漏改一处,控制台汇总与 manifest 汇总就会对不上,直接动摇"数字可对账"的项目原则。
**修复:** 保留 manifest.py 一份并导出,CLI 复用(直接读 `write_manifest` 后的 summary,或调用同一函数)。

### IN-07:manifest 下游消费不复验 image_id,产物路径可被脏清单带出运行目录

**文件:** `src/calib/visualization/overlay.py:116`、`src/calib/visualization/mosaic.py:237`
**问题:** image_id 的路径安全白名单(`_SAFE_IMAGE_ID`,dataset.py:30)只在**扫描时**校验;render_all 与 mosaic 直接拿 manifest 里的字符串拼产物文件名。"文件是唯一真相源"的架构下,manifest 本身是被消费的文件——若被手工编辑或意外损坏,含 `../` 的 image_id 可把产物写进运行目录之外。当前威胁模型内(自产自销)风险低,属纵深防御缺口。
**修复:** 消费入口(`render_all`/`write_mosaics`)对每个 image_id 复用 `_SAFE_IMAGE_ID` 校验,不合法即中文报错。

---

_审查时间:2026-10-16T08:40:50Z_
_审查者:Claude (gsd-code-reviewer)_
_深度:standard_
