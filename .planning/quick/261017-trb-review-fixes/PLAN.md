# 阶段一审查遗留修复(1 必修 + 4 应修)

## 任务描述

修复 `.planning/phases/01-data-foundation-minimal-calibration/01-REVIEW.md` 报告的 5 条遗留发现(CR-01 必修 + WR-01~04 应修)。审查报告含逐条定位与现成修复方案,以它为准;本计划把行号对齐到当前代码,并补齐每条的配套测试与验收标准。这套代码是阶段二(模型层 M1–M4 + 双求解后端)的地基,开工前先修干净。

**五条互相独立,按序执行即可(顺序无依赖);建议 CR-01 最先。**

### 硬性约束(全程有效)

- 所有新注释、报错信息、测试文档字符串用通俗中文;专业术语首现加一句白话解释(项目硬性规则)。
- 每条修复一个原子提交,共 5 个;提交信息中文、正文说清"为什么"(各任务块附草案)。只 `git add` 该条涉及的明确文件,**禁止 `git add -A` / `git add .`**——仓里有 `.planning/config.json` 的一处与本次无关的历史未提交改动,绝不能带进提交。
- 修完全量测试通过:`uv run pytest`(含真实数据冒烟);`uv run ruff check src tests` 无新告警。
- 不重跑 `output/` 产物:纯代码修复,新测试全部落在 `tmp_path` 下,真实 `output/` 一律不碰。
- 主仓直接执行,不用 worktree。
- **不做修复方案之外的顺手重构**(审查报告 Info 节的 7 条低优先级问题一概不碰,留给后续阶段)。

### 行号对照说明

审查报告的行号相对当前代码有偏移(报告后代码又有提交),以**内容定位**为准。当前实际位置:

| 发现 | 报告行号 | 当前实际位置(2026-10-17) |
|------|---------|--------------------------|
| CR-01 | cli/main.py:332-343 | `src/calib/cli/main.py:488`(检测质量摘要段 `rows = np.array(...)`) |
| WR-01 | mosaic.py:237 | `src/calib/visualization/mosaic.py:237`(suspect_mosaic 内副本路径拼接);写入方先例 `_manifest_path_for` 在 `src/calib/cli/main.py:647-662`;掩盖性夹具 `tests/test_visualization.py:534-575`(`_mosaic_fixtures`,副本写在 573 行) |
| WR-02 | config.py:169-175 | `src/calib/io/config.py:175-181`(`_read_yaml`) |
| WR-03 | baseline.py:94-95 | `src/calib/solve/baseline.py:94-95`(`calibrate_pinhole` 的裸调用);对照鱼眼分支 137-147 |
| WR-04 | precheck.py:80-86 / 163-165 | `src/calib/io/precheck.py:79-87`(board_roi 回退)/ 162-165(compute_metrics 喂角点) |

---

## 任务 1:CR-01 — corners 摘要在"只有 train 没有 test"的合法输入上崩溃

**为什么必修:** 数据层(`dataset.py`)明文承诺"划分目录整个不存在是合法输入",ingest 也正常产出 n_test=0 的清单;但 corners 的检测质量摘要把空行号列表推断成 float64 空数组,拿它做花式索引当场抛 `IndexError`,整个 corners 命令(连带 run-all)崩溃。这正是阶段二之前必须堵住的坑。

### 改哪些文件

- `src/calib/cli/main.py`(1 行改动)
- `tests/test_cli_main.py`(新建)

### 具体改法

`src/calib/cli/main.py:488`,检测质量摘要循环里:

当前:`rows = np.array([i for i, rec in enumerate(images) if rec["split"] == split])`
改为:`rows = np.array([i for i, rec in enumerate(images) if rec["split"] == split], dtype=int)`

显式声明整数类型:空划分时得到的是空 int 数组(索引合法),不再被 numpy 推断成 float64。行尾加一句中文注释说明为什么要显式 dtype(空列表推断成 float64、用它做索引会抛 IndexError,497 行的 `success_rate = ... if rows.size else None` 保护才能真正到达)。落盘段(462 行起)用的是 Python 列表,本来就安全,不动。

### 配套测试(新建 `tests/test_cli_main.py`)

文件头注释用中文说明本文件测"CLI 命令入口的行为"(区别于各模块的单元测试)。新增一条测试 `test_corners_empty_test_split_produces_all_npz_and_null_rate`:

1. 用 `make_chessboard_image` 夹具生成 2 张 640×480 合成棋盘图,写到 `tmp_path/input_data/train/poseA/poseA_00.jpg` 与 `poseA_01.jpg`(两个姿态或同姿态 2 连拍均可,取同姿态 `poseA`、`n_repeats=2`)。
2. 写最小合法配置:`tmp_path/config/board.yaml`(board_type chessboard / cols 11 / rows 8 / square_size_mm 25.0 / unit mm)与 `tmp_path/config/calibration.yaml`(只写 data 段即可,其余键全有默认值)。
3. 手工构造 ingest 产物(不跑 ingest 命令,快):`tmp_path/output/ingest/run_20260101_000000/manifest.json`,顶层 `{"images": [...]}`,每条含 `image_id` / `path`(合成 jpg 的绝对路径字符串)/ `split: "train"` / `pose_id: "poseA"` / `repeat_idx` / `n_repeats: 2` / `precheck`(verdict "ok"、blur_norm 1.0、expo_hi 0.0、expo_lo 0.0、suspect_sources [])。**只有 train,没有任何 test 记录。**
4. `monkeypatch.chdir(tmp_path)` 后直接调用 `from calib.cli.main import corners; corners()`(typer 命令函数是普通函数,可进程内调用;路径全部相对当前工作目录解析,产物落在 tmp 树,真实 `output/` 不受影响)。
5. 断言:
   - 调用不抛异常;
   - `output/corners/run_*/` 下 4 份 npz 齐全,其中 `corners_test_classic.npz` 与 `corners_test_sb.npz` 的 `image_ids` 形状为 `(0,)`(`np.load(..., allow_pickle=False)` 读);
   - `detection_summary.json` 里 `detection["test"]["classic"]["success_rate"] is None` 且 `n_images == 0`(sb 同理)。

### 验收标准

- 构造"只有 train"的目录树后进程内调用 `corners()` 正常完成,产出 4 份 npz + 摘要;
- `uv run pytest tests/test_cli_main.py -x` 通过;
- 真实 `output/` 目录无任何变化。

### 原子提交草案

```
fix(cli): corners 摘要在空划分上崩溃——行号数组显式整数类型

数据层承诺"只有 train 没有 test"是合法输入,ingest 也照常产出
n_test=0 的清单;但检测质量摘要把空行号列表推断成 float64 数组,
拿它做花式索引当场 IndexError,整个 corners 命令(连带 run-all)
中断——332 行式的 success_rate=None 保护根本没机会生效。

显式 dtype=int:空划分得到空整数数组,索引合法,摘要如实记
success_rate=null。补一条空 test 划分的端到端测试锁定该行为。
```

---

## 任务 2:WR-01 — 存疑拼图在 corners 目录找 suspect_originals,跨阶段路径断链

**为什么修:** 存疑原图副本由 ingest 阶段写在 **ingest** 运行目录(`output/ingest/run_X/suspect_originals/`),但存疑拼图在 **corners** 运行目录下找——真实流水线里这个路径永远不存在,每张都静默退回原始路径;mosaic_index.json 里"存疑照片的未标记原貌在 suspect_originals/"的指引落空,"与角点标记副本互相印证"的设计意图从未接通。测试夹具又按错误布局(corners 目录)构造,恰好把断链掩护住了。

### 改哪些文件

- `src/calib/visualization/mosaic.py`(主要改动)
- `src/calib/cli/main.py`(1 处调用加参数)
- `tests/test_visualization.py`(夹具改真实布局 + 回归测试)

### 具体改法

1. **`suspect_mosaic` 与 `write_mosaics` 增加可选参数** `ingest_run_dir: Path | None = None`。ingest 运行目录按以下优先级解析(在 `write_mosaics` 里解析一次,传给 `suspect_mosaic`):
   - 显式参数传入(优先,CLI 走这条路);
   - 否则从角点产物血缘解析:读 `corners_run_dir / "corners_train_classic.meta.json"` 的 `lineage.upstream["manifest.json"].path`,存在且文件在,则取其父目录——与 `cli/main.py` 的 `_manifest_path_for`(647-662 行)同款先例,血缘链闭合、不串轮;
   - 都没有:退回 `corners_run_dir` 本身(旧行为兜底;副本缺失时本来就有"退回原始路径"分支,不会崩)。
2. **`suspect_mosaic` 内副本路径**(237 行)由 `corners_run_dir / "suspect_originals" / ...` 改为 `ingest_run_dir / "suspect_originals" / f"{image_id}.jpg"`。
3. **存疑格的索引记录加一个键** `"source": str(image_path)`(实际采用的图片路径:ingest 副本或退回的原始路径)。这是逐格索引的对账信息,也让"到底用了哪张"可被回归测试直接断言,不用肉眼比对拼图像素。
4. **`cli/main.py` corners() 里的调用**(约 576 行)改为 `write_mosaics(out_dir, manifest, cfg, ingest_run_dir=ingest_run)`——corners 命令本来就握着 `ingest_run`,显式传入最直白。
5. **`write_mosaics` 写的 mosaic_index.json 说明文案**(约 264 行)把"存疑照片的未标记原貌在 suspect_originals/"改为指明该文件夹在**入库(ingest)轮次运行目录**下(顺带说明 source 键的含义),指引不再落空。

### 配套测试(改 `tests/test_visualization.py`)

1. **`_mosaic_fixtures` 改按真实布局构造**(534-575 行):
   - ingest 目录 `tmp_path / "ingest" / "run_t"`:内放 `manifest.json`(清单本来就该在这里)与 `suspect_originals/img2.jpg`(573 行那句挪过来);
   - corners 目录 `tmp_path / "corners" / "run_t"`:**不再**放 suspect_originals;
   - train classic 的 `save_corners` 传入 `lineage={"round_id": "t", "upstream": {"manifest.json": {"path": str(manifest_path)}}}`,让血缘解析路径可测;
   - 夹具返回值多带 ingest 目录(或返回 manifest 路径即可,manifest 已在 ingest 目录内)。
2. **新增回归测试** `test_suspect_mosaic_uses_ingest_copy_when_corners_dir_has_none`:按真实布局调用 `write_mosaics`,断言 corners 目录下不存在 `suspect_originals/`,而 `mosaic_index.json` 的 `suspect.cells` 中 img2 一格的 `source` 以 ingest 目录的 `suspect_originals/img2.jpg` 结尾——证明副本真正被取到,而不是退回原始路径。
3. **新增(或并入上条)副本缺失分支测试**:ingest 目录也不放副本时,`source` 等于清单里的原始路径(退回分支行为锁定,拼图不中断)。
4. 既有三个 `write_mosaics` 测试(578 / 608 / 627 行)随夹具改真实布局后照常通过,断言基本不动(它们只看格数与索引,不看图片来源)。

### 验收标准

- 回归测试通过:corners 目录没有 suspect_originals 时,存疑拼图仍取到 ingest 目录的副本,索引 `source` 指向该副本;
- 血缘解析路径可用:不显式传 `ingest_run_dir`、但 meta 记录了 upstream 清单路径时,同样能找到副本;
- `uv run pytest tests/test_visualization.py -x` 通过。

### 原子提交草案

```
fix(visualization): 存疑拼图改到入库目录找 suspect_originals 副本

存疑原图副本由 ingest 阶段写在 ingest 运行目录,存疑拼图却在
corners 运行目录下找——真实流水线里这个路径永远不存在,每张都
静默退回原始路径:拼图索引指引读者去一个不存在的文件夹,"与
角点标记副本互相印证未标记原貌"的设计意图从未接通。测试夹具
又按错误布局构造,恰好把断链掩护住了。

现在按"CLI 显式传入 > 角点产物血缘解析 > 退回 corners 目录"
的优先级定位 ingest 目录(_manifest_path_for 同款先例),逐格
索引记录每格实际采用的图片来源(source 键),夹具改回真实布局
并加回归测试:corners 目录没有副本时仍取到 ingest 副本。
```

---

## 任务 3:WR-02 — yaml 语法错误裸抛英文堆栈,击穿两道中文报错防线

**为什么修:** config.py 头部承诺"解析失败必须给出带字段名与实际值的通俗中文错误",但只覆盖结构校验;yaml **语法**错误(如 tab 缩进)抛的 `yaml.ScannerError` 不是 ValueError 子类(审查已实测确认),`_read_yaml` 不捕获,用户拿到 PyYAML 原始英文堆栈;更糟的是 `_setup_runtime` 的 `except (ValueError, OSError)` 接不住它——`calib ingest --help` 这种本不该碰配置语义的调用也会因一个坏 yaml 直接崩溃,违背该回调"--help / 配置还没就位时不拦着用户"的承诺。

### 改哪些文件

- `src/calib/io/config.py`(`_read_yaml`,175-181 行)
- `tests/test_io_config.py`(新增 2 条测试)

### 具体改法

`_read_yaml` 的 `yaml.safe_load(f)` 包一层异常转换(审查给出的现成方案):

```python
try:
    return yaml.safe_load(f)
except yaml.YAMLError as e:
    raise ValueError(
        f"配置文件 {path} 的 yaml 语法有误,请检查缩进与冒号:{e}"
    ) from e
```

要点:转成带文件路径的中文 ValueError,消息末尾附 PyYAML 的原始信息(行号/列号对定位有用,不丢);`raise ... from e` 保留因果链。`_setup_runtime` 现有的 `except (ValueError, OSError)` 自然兜住,**该文件不需要再改**。docstring 补一句中文说明:语法错误也归入中文 ValueError 防线。

### 配套测试(加到 `tests/test_io_config.py` 的"非法配置"小节)

1. `test_yaml_syntax_error_chinese_error_with_path`:写一份带 tab 缩进的 `calibration.yaml`(如 `"data:\n\ttrain: a\n  test: b\n"`),`load_calibration` 抛 `ValueError`,断言消息含"语法"与配置文件路径文字(`pytest.raises(ValueError, match=...)` 加路径断言)。
2. `test_setup_runtime_survives_broken_yaml`:monkeypatch.chdir 到只含坏 `config/calibration.yaml` 的临时目录,直接调用 `calib.cli.main._setup_runtime()` 不抛任何异常——对应"`calib ingest --help` 不因坏配置崩溃"的承诺。

### 验收标准

- 坏语法 yaml 上 `load_calibration` / `load_board` 都抛中文 ValueError(带路径、附原始信息),不再露出 PyYAML 英文堆栈;
- 坏配置目录下 `_setup_runtime()` 安静返回;
- `uv run pytest tests/test_io_config.py -x` 通过。

### 原子提交草案

```
fix(io): yaml 语法错误转中文 ValueError,补齐两道报错防线

yaml.ScannerError 不是 ValueError 子类(实测确认),语法错误
(如 tab 缩进)直接裸抛 PyYAML 英文堆栈:既违反本项目"报错
中文且带定位信息"的自我承诺,又击穿 _setup_runtime 的
except (ValueError, OSError)——calib ingest --help 这类不该
碰配置语义的调用也会被一个坏 yaml 搞崩。

_read_yaml 捕获 yaml.YAMLError 转成带文件路径的中文
ValueError(附原始信息、保留因果链),两道防线同时修复。
```

---

## 任务 4:WR-03 — 针孔求解不捕获 cv2.error,与鱼眼"失败记录不中断"契约不对称

**为什么修:** D-10 的契约是"失败会被完整记录(结构键一个不少、数值字段一律 None),绝不中断其余组合",鱼眼分支照此实现了 try/except + 中文 failure_note;但 `calibrate_pinhole` 对 `cv2.calibrateCameraExtended` 没有任何保护。`run_baselines` 的准入门槛只有"可用视图 ≥ 2"——两三个近平面视角加 5 个畸变系数正是 calibrateCamera 的病态区,一旦抛 cv2.error:报错是原始 C++ 堆栈而非中文,四个组合全部中断,run-all 跟着断。

### 改哪些文件

- `src/calib/solve/baseline.py`(`calibrate_pinhole`,94-95 行)
- `tests/test_solve_baseline.py`(新增 1 条测试)

### 具体改法

与鱼眼分支(137-147 行)同构包裹(审查给出的现成方案):

```python
try:
    rms, K, D, rvecs, tvecs, _s_int, _s_ext, per_view = cv2.calibrateCameraExtended(
        objs32, imgs32, tuple(image_size), None, None, flags=flags)
except cv2.error as e:
    return {"rms_overall": None, "K": None, "D": None, "rvecs": None,
            "tvecs": None, "per_view_rms": None, "flags_used": flags,
            "converged": False,
            "failure_note": f"针孔模型求解未收敛:OpenCV 报错——{e}"}
```

失败值语义与鱼眼完全一致:结构键一个不少、数值字段一律 None、中文 failure_note 带 OpenCV 原始信息。模块头与 `calibrate_pinhole` docstring 里"失败记录不中断"的承诺本来就是对两个模型写的,补齐实现即可,措辞不用大改。

### 配套测试(加到 `tests/test_solve_baseline.py`)

新增 `test_pinhole_cv2_error_recorded_not_raised`:

1. 用 `monkeypatch` 把 `cv2.calibrateCameraExtended` 换成必然触发真实 `cv2.error` 的替身。构造 `cv2.error` 实例优先用 `cv2.error(cv2.Error.StsError, "测试注入的求解失败", "calibrateCameraExtended", "test", 0)` 这种五参形态;若当前 OpenCV 4.14 的构造签名不兼容,退而求其次:先保存原函数,替身内部用空输入调用原函数(`original([], [], (640, 480), None, None, flags=0)`,空点集会让 OpenCV 断言失败抛出货真价实的 cv2.error)——两种方式任选其一,以"替身抛出的确是 cv2.error"为准。
2. 用 `_make_true_views(make_chessboard_image, n_views=2)` 造 2 个真值视角,调 `calibrate_pinhole(views, (640, 480), board)`。
3. 断言:不抛异常;九个结构键(`rms_overall / K / D / rvecs / tvecs / per_view_rms / flags_used / converged / failure_note`)一个不少;`converged is False`;`failure_note` 是非空中文(含"针孔");全部数值字段为 None;`flags_used == 0`。

### 验收标准

- 替身注入 cv2.error 时 `calibrate_pinhole` 返回完整失败结构而非抛异常;
- 失败字典与鱼眼分支的失败值语义逐键一致(结构键齐全、数值 None);
- `uv run pytest tests/test_solve_baseline.py -x` 通过。

### 原子提交草案

```
fix(solve): 针孔求解与鱼眼同构——cv2.error 记录为发现,不中断

D-10 契约:"失败会被完整记录(结构键一个不少、数值字段一律
None),绝不中断其余组合"。鱼眼分支照此实现,针孔分支却对
calibrateCameraExtended 没有任何保护——而 run_baselines 的
准入门槛只有"可用视图 >= 2",两三个近平面视角加 5 个畸变系数
正是它的病态区:一旦抛 cv2.error,四个组合全断,run-all 跟着
断,报错还是原始 C++ 堆栈。

与鱼眼同构包裹:失败时返回一致的失败结构(中文 failure_note
带 OpenCV 原始信息),配套测试锁定"注入 cv2.error 不抛异常、
结构键齐全"。
```

---

## 任务 5:WR-04 — 体检回退硬编码检测器选项(违反 D-12);classic 未检出而 SB 检出的照片模糊度通道失明

**为什么修:** 三处叠加缺陷:① `board_roi` 的完整查找硬编码 `CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE`,无视 `cfg.classic_flags`——D-12 硬性规则是"换数据集只改配置不改代码",用户改配置后同一帧内探针与棋盘框定位两个口径不一致,会干扰阈值迭代回环(D-04);② classic 探针失败后 `board_roi` 还要用 FAST_CHECK + 硬编码选项把同一颗检测器再跑最多两遍,纯浪费;③ 结果是"classic 未检出而 SB 检出"的照片 `roi_found=False` → `blur_norm=NaN`,同姿态模糊度 z 分数与全局分位线两条判定来源对它双双失明——而 box_area_ratio 的文档口径是"谁检出用谁,经典优先",同一模块内自相矛盾。

### 改哪些文件

- `src/calib/io/precheck.py`(`board_roi` 79-87 行、`compute_metrics` 162-165 行、`visibility_probe` 116-118 行)
- `tests/test_io_precheck.py`(新增 2 条测试)

### 具体改法

1. **新增模块级小函数** `_classic_flag_mask(cfg: CornersConfig) -> int`:遍历 `cfg.classic_flags` 按位或 `_CLASSIC_FLAG_VALUES` 的对应常量。`visibility_probe` 内现行的三行拼装(116-118 行)改调它——回退查找与探针从此同一来源。
2. **`board_roi` 增加可选参数** `cfg: CornersConfig | None = None`,回退查找(79-87 行)改为**单次**调用:
   - `mask = _classic_flag_mask(cfg if cfg is not None else CornersConfig())`(不传 cfg 的独立调用按配置数据类的默认选项兜底——默认值单一来源,不再写死裸常量组合);
   - 删除"FAST_CHECK 预筛 + 硬编码完整查找"两段:fast_check 若在配置里,自然已并入 mask;同一颗检测器、同一份选项在同一张图上的第二次调用必然同样结果,两遍就是白跑。
   - docstring 的查找顺序说明同步改写:①调用方带进的成功检测角点;②按配置选项完整查找一次;③都失败回退整张图。
3. **`compute_metrics` 的角点喂入**(162-165 行)改为三段:`probe["classic_corners"]` if classic_ok,否则 `probe["sb_corners"]` if sb_ok,否则 None。classic 失败而 SB 检出时,SB 角点同样能定位棋盘框(模糊度只在框内算,不关心角点来自哪颗检测器)——模糊度通道不再失明,口径与 box_area_ratio 的"谁检出用谁,经典优先"对齐。
4. `compute_metrics` docstring 里 `blur_norm` 的说明补一句:classic 未检出而 SB 检出时,棋盘框由 SB 角点定位。

### 配套测试(加到 `tests/test_io_precheck.py`)

1. `test_board_roi_fallback_uses_configured_flags_and_no_double_run`(配置驱动 + 消除白跑,一条锁定两件事):
   - 用 `monkeypatch` 给 `cv2.findChessboardCorners` 套一层间谍包装(记录每次调用的 flags 参数,内部透传真实函数);
   - 纯噪声图(240×320)上调 `compute_metrics(noise, COLS, ROWS, CornersConfig(classic_flags=("fast_check",)))`;
   - 断言:记录到的**每一次**调用的 flags 都恰等于 `cv2.CALIB_CB_FAST_CHECK`(回退用的是配置选项,不是写死的 ADAPTIVE+NORMALIZE——旧代码在此必现第三种取值);classic 总调用次数恰为 2(探针 1 次 + 回退 1 次;旧代码是探针 1 + FAST_CHECK 1 + 硬编码 1 = 3 次,白跑被消除)。噪声图上必然全部失败,`roi_found` 仍为 False、回退整图,顺带锁定回退行为不变。
2. `test_compute_metrics_feeds_sb_corners_when_classic_fails`(SB 兜底模糊度通道):
   - 用 `monkeypatch.setattr("calib.io.precheck.visibility_probe", ...)` 换成假探针,返回 `classic_ok=False / sb_ok=True / classic_corners=None / sb_corners=真值角点网格 / box_area_ratio=有效数 / corners_count_ok=True`(`make_chessboard_image` 拿真值网格,形状 (88,2));
   - 在真实合成棋盘图上调 `compute_metrics`;
   - 断言:`roi_found is True`、`sq_px` 为有限数、`blur_norm` 为有限数且非 NaN(模糊度通道打通);同时给 `cv2.findChessboardCorners` 再套间谍,断言它**一次都没被调**——角点确实来自 SB 喂入,没有回退重跑 classic。
3. 既有 `test_board_roi_falls_back_to_full_image_on_noise`(不传 cfg 裸调)在新签名下照旧通过,不需要改(默认配置兜底路径的行为锁定)。

### 验收标准

- 回退查找的检测器选项与 `calibration.yaml` 的 `corners.classic_flags` 联动(改配置即改行为,不再有写死常量组合);
- classic 未检出而 SB 检出的照片:`roi_found=True`、`blur_norm` 有限,classic 的完整查找不再重复白跑;
- 裸调 `board_roi`(不传 cfg)在无棋盘图上仍回退整图、不抛异常;
- `uv run pytest tests/test_io_precheck.py -x` 通过。

### 原子提交草案

```
fix(io): 体检回退按配置组装检测选项,SB 角点兜底模糊度通道

board_roi 的完整查找硬编码 ADAPTIVE_THRESH + NORMALIZE_IMAGE,
无视 cfg.classic_flags——违反 D-12"换数据集只改配置不改代码":
用户改配置后,可见性探针与棋盘框定位对同一帧给出两个口径的
classic 结论,干扰阈值迭代回环。classic 探针失败后它还用
FAST_CHECK + 硬编码选项把同一颗检测器再跑最多两遍(几百张图
时纯浪费)。后果是 classic 未检出而 SB 检出的照片 roi_found=
False、blur_norm=NaN,同姿态 z 分数与全局分位线两条模糊度
判定来源对它双双失明——与 box_area_ratio"谁检出用谁"的口径
自相矛盾。

回退查找改用 _CLASSIC_FLAG_VALUES 按配置组装(与探针同一
来源,单次调用消除白跑);classic 失败而 sb_ok 时把
probe["sb_corners"] 喂给 board_roi,模糊度通道不再失明。
```

---

## 整体验收

1. `uv run pytest` 全量通过(含真实数据冒烟 integration 用例;日常可先 `uv run pytest -m "not integration"` 快验);
2. `uv run ruff check src tests` 零告警(与修复前一致,无新增);
3. `git log --oneline` 恰好 5 个修复提交(允许最后补一个 `docs(quick): ...` 提交收录本计划文件,**绝不包含 `.planning/config.json`**);每次提交前用 `git status` 确认 `.planning/config.json` 仍在未暂存区;
4. 真实 `output/` 目录无任何改动(新测试全在 pytest 临时目录里自建目录树);
5. 五条审查发现的修复行为各自被测试锁定:空划分 corners 不崩(CR-01)、拼图取到 ingest 副本(WR-01)、坏 yaml 中文报错且 --help 防线不破(WR-02)、针孔失败不中断(WR-03)、回退随配置 + SB 兜底模糊度(WR-04)。
