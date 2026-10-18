# Phase 3: 噪声底 + Fisher/CRLB 可达精度预测 - 模式对照表(给规划与执行用)

**整理日期:** 2026-10-18
**分析的新建/修改文件数:** 14(7 个 src 新文件 + 1 个 src 扩展 + 1 个配置扩展 + 1 个 CLI 扩展 + 5 个测试文件)
**找到现成可仿写的文件:** 14 / 14(其中 2 个为部分匹配,详见"无可仿文件"节)

> 读法说明:每小节给出"新文件该仿哪个现成文件",并摘录现成代码的具体行号与原文。
> 术语白话:**雅可比** = 残差对参数的偏导数矩阵(参数动一点,像素动多少);
> **Fisher 信息矩阵** = 数据里"关于参数的信息量"的矩阵,JᵀJ/σ²;
> **Schur 消元** = 把不关心的外参从误差计算里合并掉,只留内参的精度界。

## 文件分类

| 新建/修改文件 | 角色 | 数据流 | 最接近的现成文件 | 匹配程度 |
|-------------------|------|-----------|----------------|---------------|
| `src/calib/noise/immerkaer.py` | utility(单图数值计算) | transform(一图进、两数出) | `src/calib/corners/detect.py` | 角色匹配 |
| `src/calib/noise/temporal.py` | service(逐姿态统计) | batch(npz+清单 → 逐姿态表) | `src/calib/solve/observation.py` + `cli/main.py` 的 `_burst_failures_for` | 角色匹配 |
| `src/calib/noise/injection.py` | service(并行批量实验) | batch + file-I/O | `cli/main.py` corners 主循环 + `src/calib/corners/detect.py` | 部分匹配(并行是新东西) |
| `src/calib/noise/transfer.py` | service(曲线汇总插值) | transform | `cli/main.py` 的 `_stats_block` + `solve/scoring.py` | 部分匹配 |
| `src/calib/noise/fisher.py` | service(矩阵代数) | transform(数组进、矩阵出) | `src/calib/solve/residuals.py` + `solve/matrix.py` | 精确(J 必须同源) |
| `src/calib/noise/crosscheck.py` | service(OpenCV 桥接) | request-response | `src/calib/solve/opencv.py` | 精确 |
| `src/calib/noise/artifacts.py` | utility(写盘读盘) | file-I/O | `src/calib/corners/artifacts.py` + `solve/result.py` | 精确 |
| `src/calib/io/config.py`(扩展) | config(解析校验) | parse/validate | 自身现有 `CornersConfig`/`_parse_corners` 体例 | 精确(自仿) |
| `config/calibration.yaml`(扩展) | config(参数来源) | — | 自身现有 `corners:` 段注释体例 | 精确(自仿) |
| `src/calib/cli/main.py`(扩展) | controller(CLI) | request-response | 自身现有 `corners()`/`calibrate()` 两个函数 | 精确(自仿) |
| `tests/test_noise_immerkaer.py` | test | unit | `tests/test_models_jacobian.py` + `tests/conftest.py` | 角色匹配 |
| `tests/test_noise_temporal.py` | test | unit | `tests/test_corners_artifacts.py` 体例 + conftest 合成夹具 | 角色匹配 |
| `tests/test_noise_injection.py` | test | unit + integration 标记 | `tests/test_smoke_realdata.py` 的 marker 用法 + conftest | 角色匹配 |
| `tests/test_noise_fisher.py` | test | unit(数值基准) | `tests/test_models_jacobian.py` | 精确 |
| `tests/test_noise_crosscheck.py` | test | unit | `tests/test_models_parity.py` + `test_solve_solvers.py` | 角色匹配 |
| `tests/test_smoke_realdata.py`(扩展) | test | integration | 自身现有三命令接力结构 | 精确(自仿) |

## 模式对照(逐文件)

### `src/calib/noise/fisher.py`(service,矩阵代数)— 本阶段数学核心

**拟仿文件:** `src/calib/solve/residuals.py`(雅可比唯一来源)+ `src/calib/solve/matrix.py`(多组合编排与逐组合落盘)

**铁律(D-10,阶段二锁定):J 只准从 `make_jacobian_fn` 来,绝不另写第二份投影/导数公式。**

**要照抄的模式 1:J 的取得与按列分段(不重排行)。**
`src/calib/solve/residuals.py` 第 26-34 行(导入体例)与第 164-200 行(工厂本体):

```python
from calib.models.base import CameraModel, as_params, transform_to_camera
from calib.solve.observation import ObservationSet
# ...
def make_jacobian_fn(model: CameraModel, obs: ObservationSet):
    """解析稀疏雅可比工厂:返回 callable(θ) → scipy.sparse.csr_matrix。
    形状 (2·V·P, P+6V)。……"""
    # theta 由 pack_theta 装出:内参在前 (P,),逐视图 (ω,t) 6 个在后
    # fisher.py 里只做列分段:J_int = J[:, :n_int], J_pose = J[:, n_int:]
    # 绝不重排行/列——行序契约由 row_index(view, point, uv_axis, n_points) 唯一承载
```

θ 装箱必须走现成的 `pack_theta(model, params, rvecs, tvecs)`(第 52-84 行),它接参数**字典**(按 `model.param_names` 取名,缺名中文报错)——线性化点从 `result.json` 的 `params` 字典读回来后直接可喂。

**要照抄的模式 2:中文报错在入口点名、绝不静默绕过。**
`src/calib/solve/residuals.py` 第 60-66 行的体例(fisher.py 的"奇异 6×6 块"报错照此写):

```python
raise AssertionError(
    f"参数字典缺 {model.model_id} 的参数名:{missing}"
    f"(全部应为 {list(model.param_names)})。请按参数名给全,"
    f"装箱不猜位置。"
)
```

**要照抄的模式 3:多组合编排——"配置清单 → 逐组合 → 每组合一个子目录 → 逐组合落盘不中断"。**
`src/calib/solve/matrix.py` 第 240-309 行 `run_calibration_matrix` 的结构(fisher.py 的 2 观测集 × 2 口径 × 4 模型 = 16 份 CRLB 照此组织):

```python
# matrix.py 的编排骨架(第 282-307 行):
for model_id in cal.models:
    model = create_model(model_id)
    # ...逐组合算,try/except 兜底:单组合任何异常转 make_failure 落盘,绝不中断循环
    if progress:
        progress(row)   # 库层不依赖 typer,播报回调由 CLI 注入(第 54 行 ProgressFn)
```

失败兜底照抄 `matrix.py` 第 181-188 行:`except Exception as e:  # noqa: BLE001  # 故意宽接:单组合任何异常都转失败契约落盘`。

**要照抄的模式 4:模型按名查表。** `src/calib/models/__init__.py` 第 42-54 行 `create_model(model_id)`,未知名抛中文错误并列出全部可选名。

---

### `src/calib/noise/crosscheck.py`(service,OpenCV 桥接)

**拟仿文件:** `src/calib/solve/opencv.py`(适配器使用 + float32 装包 + Extended 解包的唯一现成样板)

**要照抄的模式 1:经适配器调 OpenCV,flags 与 K/D 解包全走适配器,绝不手拼。**
`src/calib/solve/opencv.py` 第 122-151 行 `_solve_via_calibrate_camera`——这就是 4.14 `calibrateCameraExtended` **8 元组**返回的正确解包写法(RESEARCH §Pattern 4 实测:8 元组,不是旧资料的 11 元组):

```python
adapter = model.opencv_adapter()
# ...
# OpenCV 4.14 的 calibrateCamera 绑定只收 32 位浮点(Point2f/Point3f),
# 传 64 位会报 "Unsupported format"——这里显式降精度再调用
objs32 = [p.astype(np.float32) for p in objs]
imgs32 = [p.astype(np.float32) for p in imgs]
flags = int(adapter.flags)
rms, K, D, rvecs, tvecs, _std_int, _std_ext, _per_view = cv2.calibrateCameraExtended(
    objs32, imgs32, tuple(obs.image_size), None, None, flags=flags)
params = adapter.params_from_cv(np.asarray(K, dtype=np.float64),
                                np.asarray(D, dtype=np.float64).ravel())
```

crosscheck.py 的差别只有两处:①第 6 个返回值 `stdDeviationsIntrinsics` 这次**要**(opencv.py 里丢弃了);②物点/角点装包体例照第 39-54 行 `_prepare_view`(入口形状断言 + `(n,1,dim)` 变形)。

**要照抄的模式 2:OpenCV 抛错不炸主流程。** `opencv.py` 第 81-93 行:`except cv2.error as e:` → 转成带中文说明的失败记录(crosscheck 对应 crlb.json 的 `status: "failed"` + 中文 failure_note,不阻塞 CRLB 主产物)。

**要照抄的模式 3:M4 分支。** `opencv.py` 第 154-169 行展示 fisheye 走的是另一套函数(`cv2.fisheye.calibrate`,无 Extended 版)——crosscheck 里 M4 直接记"无可比输出"结论,不硬调。

---

### `src/calib/noise/artifacts.py`(utility,写盘读盘)

**拟仿文件:** `src/calib/corners/artifacts.py`(npz + meta.json 旁车)+ `src/calib/solve/result.py`(失败契约:结构键全在、数值 null、中文说明)

**要照抄的模式 1:npz + meta.json 旁车、lineage 必带。**
`src/calib/corners/artifacts.py` 第 32-88 行 `save_corners` 的结构:

```python
out_dir.mkdir(parents=True, exist_ok=True)
np.savez(npz_path, **arrays)          # 只放纯数值数组与定长 unicode 编号列,不需要 pickle
meta = {
    "file": npz_path.name, ...,
    "lineage": lineage,               # make_lineage 组装,按参数名传
}
with open(out_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)   # ensure_ascii=False 中文原样
```

**要照抄的模式 2:读 npz 一律 `allow_pickle=False`。**
`src/calib/corners/artifacts.py` 第 91-98 行:

```python
with np.load(Path(path), allow_pickle=False) as data:
    return {key: data[key] for key in data.files}
```

**要照抄的模式 3:失败契约——失败结果照样落盘,结构键一个不少。**
`src/calib/solve/result.py` 第 158-222 行 `save_result` 的结构(json 必写、npz 只在成功时写;第 81-111 行 `make_failure` 是"数值字段一律 None + failure_note 非空中文"的工厂)。noise.json / crlb.json 沿用:每个记录必有 `lineage`、`converged`(或 `status`)、中文 `failure_note` 键。

---

### `src/calib/noise/temporal.py`(service,逐姿态连拍统计)

**拟仿文件:** `src/calib/cli/main.py` 第 194-238 行 `_burst_failures_for`(按 pose_id 分组的唯一现成样板)+ `src/calib/solve/observation.py`(npz 读回 + 行序锁)

**要照抄的模式 1:按姿态分组。** `cli/main.py` 第 206-210 行:

```python
groups: dict[str, list[int]] = {}
for i, rec in enumerate(images):
    if rec["split"] == "train":
        groups.setdefault(rec["pose_id"], []).append(i)
```

temporal.py 同样从 manifest 的 `images` 列表取 `pose_id` / `repeat_idx` / `n_repeats`(manifest 每张照片条目都带,见 `io/manifest.py` 第 114-116 行 entry 结构与 `io/dataset.py` 的 ImageRecord)。

**要照抄的模式 2:角点 npz 读回的形状断言与行序锁。**
`src/calib/solve/observation.py` 第 154-175 行:

```python
with np.load(Path(corners_npz), allow_pickle=False) as data:
    npz_ids = [str(x) for x in np.asarray(data["image_ids"]).ravel()]
    uv = np.asarray(data["uv"], dtype=np.float64)
    detected = np.asarray(data["detected"], dtype=bool)
    source = [str(x) for x in np.asarray(data["source"]).ravel()]
# ...
if uv.ndim != 3 or uv.shape[1:] != (n_points, 2):
    raise AssertionError(f"角点数组形状不合:期望 (行数,{n_points},2)……")
```

temporal.py 读角点时照抄这套(含"未检出整行 NaN"的语义,`detected` 为 False 的帧不进波动统计)。

**要照抄的模式 3:"不足 2 张就换路"的退化分支。** `cli/main.py` 第 213-214 行:`if len(det_idxs) < 2: continue`(temporal 对应:该姿态退化到 Immerkaer 单帧估计,`source` 字段标 `"immerkaer_transfer"`)。

---

### `src/calib/noise/immerkaer.py`(utility,单图数值计算)

**拟仿文件:** `src/calib/corners/detect.py`(小而纯的函数体例:cv2 进、dict 出、失败返回 None 不返回半成品)

**要照抄的模式 1:函数体例——入参带 cfg、返回结构化 dict、docstring 声明数组形状。**
`src/calib/corners/detect.py` 第 44-49 行:

```python
def detect_classic(gray: np.ndarray, cols: int, rows: int, cfg: CornersConfig) -> dict | None:
    """经典检测器:粗定位 → 二次亚像素精化(窗口按实测方格边长)→ 规范化。
    返回 dict(uv (rows*cols, 2) float64, sq_px, win_size, detector="classic"),
    未检出返回 None。"""
```

immerkaer 的 `immerkaer_sigma(gray, cfg) -> dict`(`sigma_std` 与 `sigma_robust` 两键并报)照此体例;公式本体按 03-RESEARCH §Code Examples"Immerkaer 单帧估计"一节(核 L = [[1,−2,1],[−2,4,−2],[1,−2,1]],`cv2.filter2D` 卷积)。

**要照抄的模式 2:配置项翻译成行为。** `detect.py` 第 29-41 行把 yaml 里的选项名翻译成行为/常量的写法——immerkaer 的"主线用标准式还是稳健式"配置项(`noise.immerkaer.variant` 之类)照此在 config 层给白名单、这里只认白名单值。

---

### `src/calib/noise/injection.py`(service,并行批量实验)

**拟仿文件:** `cli/main.py` 第 369-436 行 corners 主循环(逐张读图 + 进度播报 + 兜底来源留痕)+ `src/calib/corners/detect.py`(检测器复用,不复制逻辑)

**要照抄的模式 1:逐张读图与进度播报。** `cli/main.py` 第 405-436 行:

```python
for i, rec in enumerate(images):
    gray = imread_checked(Path(rec["path"]))
    # ...逐张处理
    if (i + 1) % 100 == 0:
        typer.echo(f"  已处理 {i + 1}/{n} 张……")
```

(`imread_checked` 来自 `io/dataset.py`,读图失败即中文报错——注入实验里这是子进程内的失败留痕,不是中断。)

**要照抄的模式 2:检测失败/兜底的来源留痕语义。** `cli/main.py` 第 426-434 行与 `corners/artifacts.py` 第 10-12 行:`"sb"` / `"classic_fallback"` / `""` 三态 source 数组——注入实验逐档照搬这套语义(RESEARCH §Pattern 2"失效留痕"直接复用,曲线高 σ 段的失败率抬头是数据不是错误)。

**要照抄的模式 3:SB 检测只跑一次到位、严禁二次精化。** `detect.py` 第 85-108 行 `detect_sb`——注入实验重跑检测器时**直接调用** `detect_sb` / `detect_classic`,配置对象用 `cfg.corners` 原样传入,绝不复制检测逻辑(RESEARCH 职责表:注入层只负责加噪、调度、落盘)。

**无现成仿写的部分(按 03-RESEARCH §Pattern 2 新写):** `concurrent.futures.ProcessPoolExecutor` 按图像切任务;种子公式 `seed + sha1(f"{image_id}|{level}|{rep}") % 2**32`;float 域加噪后 `clip(0,255)` 再四舍五入回 uint8。合成图像不落盘、只落逐角点位移(可复现性来自种子三元组)。

---

### `src/calib/noise/transfer.py`(service,曲线汇总与插值)

**拟仿文件:** `cli/main.py` 第 183-191 行 `_stats_block`(min/中位数/max 统计块的现成体例)+ `src/calib/solve/scoring.py`(汇总统计的结构化输出体例)

**要照抄的模式 1:统计块体例。** `cli/main.py` 第 183-191 行:

```python
def _stats_block(values: np.ndarray) -> dict | None:
    """检出帧的统计(min/中位数/max);一个都没检出时返回 None。"""
    if values.size == 0:
        return None
    return {"min": float(np.min(values)), "median": float(np.median(values)),
            "max": float(np.max(values))}
```

**无现成仿写的部分(按 03-RESEARCH 指引新写):** 插值用 `scipy.interpolate.PchipInterpolator`(保形插值,不写过冲的曲线);"喂 FIM 的每轴 σ = 曲线值/√2"的换算写死在代码注释与字段名里(`sigma_corner_per_axis`)。

---

### `src/calib/io/config.py`(扩展:NoiseConfig / FisherConfig)

**拟仿文件:** 自身现有 `CornersConfig` + `_parse_corners`(第 106-119、348-369 行)——新配置段一体照搬这条"白名单 → 逐字段校验 → frozen dataclass"流水线

**要照抄的模式 1:dataclass 体例(带中文白话 docstring)。** 第 106-119 行:

```python
@dataclass(frozen=True)
class CornersConfig:
    """第②步找角点参数:经典与新款(SB)双检测器并跑,各存一份。"""
    classic_flags: tuple[str, ...] = ("adaptive_thresh", "normalize_image")
    sb_flags: int = 0
    sb_fallback_classic: bool = True
    subpix: SubpixConfig = field(default_factory=SubpixConfig)
    burst_check: BurstCheckConfig = field(default_factory=BurstCheckConfig)
```

**要照抄的模式 2:解析函数体例(白名单 + 逐字段中文报错)。** 第 348-369 行:

```python
def _parse_corners(raw: dict) -> CornersConfig:
    section = _check_is_dict(raw.get("corners", {}), "corners")
    flags = section.get("classic_flags", CornersConfig.classic_flags)
    if not isinstance(flags, (list, tuple)) or not all(isinstance(f, str) for f in flags):
        _fail("corners.classic_flags", flags, "必须是一列选项名(如 [adaptive_thresh])")
    bad = [f for f in flags if f not in _ALLOWED_CLASSIC_FLAGS]
    if bad:
        _fail("corners.classic_flags", bad,
              f"含有不认识的选项;可选:{sorted(_ALLOWED_CLASSIC_FLAGS)}")
    return CornersConfig(...)
```

**要照抄的模式 3:顶层装配。** 第 210-221 行 `CalibrationConfig` 加 `noise:`/`fisher:` 两个字段(默认 `field(default_factory=...)`),第 551-560 行 `load_calibration` 返回处各加一行 `_parse_noise(top)` / `_parse_fisher(top.get("fisher", {}))`。bool 字段校验注意第 360-365 行的坑:先判 `isinstance(x, bool)` 再拒其余(yaml 手写 `1/0` 当 bool 是常见笔误)。

---

### `config/calibration.yaml`(扩展:noise: / fisher: 两段)

**拟仿文件:** 自身现有 `corners:` 段(第 35-62 行)——注释体例:每项带白话解释、依据、单位;初值待校准的写"待首轮分布校准"

体例样板(第 37-47 行):`classic_flags` 每行注释解释选项含义、`sb_fallback_classic` 注释含三轮实验依据。noise/fisher 新段的每个键照此写:键名 + 白话含义 + 默认值依据 + 单位。σ 网格上下限、档数、重复数、并行 worker 上限全部进这里(D-12:参数只从 config 读)。

---

### `src/calib/cli/main.py`(扩展:填 noise / fisher 两个空壳)

**拟仿文件:** 自身现有 `corners()`(第 349-646 行)与 `calibrate()`(第 674-772 行)

**待填空壳的位置:** 第 649-651 行 `def noise(): _not_implemented(3, "测噪声底")`;第 775-777 行 `def fisher(): _not_implemented(5, "算理论精度极限")`——函数名与 help 文案已在阶段一定版,**保留不改**,只填函数体。

**要照抄的模式 1:命令体的固定顺序。** `corners()` 第 369-386 行:

```python
started = datetime.now(UTC)
board = load_board(BOARD_PATH)
cfg = load_calibration(CALIBRATION_PATH)
# 上游产物:最新一轮(文件是唯一真相源,阶段间只靠文件通信)
ingest_run = latest_run_dir(OUTPUT_ROOT, "ingest")
timestamp = started.strftime("%Y%m%d_%H%M%S")
round_id = f"run_{timestamp}"
out_dir = OUTPUT_ROOT / "corners" / round_id
lineage = make_lineage(
    config_files={"board.yaml": BOARD_PATH, "calibration.yaml": CALIBRATION_PATH},
    upstream={"manifest.json": ingest_run / "manifest.json"},
    round_id=round_id,
)
```

noise 命令:`OUTPUT_ROOT / "noise" / round_id`,upstream 记角点 npz + manifest;fisher 命令:`OUTPUT_ROOT / "fisher" / round_id`,upstream 记角点 npz + noise.json + 标定产物(result.json + extrinsics.npz)。

**要照抄的模式 2:上游定位走血缘优先。** `calibrate()` 第 654-671 行 `_manifest_path_for`:优先用上游 meta 里记录的路径(防串轮),缺失才退回 `latest_run_dir`——fisher 定位阶段二标定产物照此办。

**要照抄的模式 3:附加产物失败不阻塞主产物。** `calibrate()` 第 753-769 行:

```python
try:
    review = review_suspects(...)
    ...
except Exception as e:  # noqa: BLE001  # 故意宽接:复核是标定的附加产物,失败不影响标定结果(计划明文)
    typer.echo(f"  存疑帧复核未跑成(不影响标定结果本身):{type(e).__name__}——{e}")
```

fisher 的 OpenCV 交叉验证正属"附加产物"(RESEARCH §Pattern 4 失败契约)——照抄这个 try/except + 中文提示体例。

**要照抄的模式 4:收尾中文汇总逐条 echo。** `calibrate()` 第 734-749 行的对比表体例(对齐的中文表头 + 每行一个组合 + 白话注解);进度回调注入见第 716-726 行(`_report` 闭包传给库层,库层不 import typer)。

**要照抄的模式 5:run-all 收尾更新。** 第 795-807 行 `run_all` 现在跑到 calibrate 为止——noise/fisher 实现后把两步加进接力(照第 800-805 行体例)。

---

## 测试文件模式

### `tests/test_noise_fisher.py` — **精确仿 `tests/test_models_jacobian.py`**(数值基准测试的现成全套)

**要照抄的模式 1:numdifftools 数值基准 + 相对误差门槛。** `test_models_jacobian.py` 第 61-81 行:

```python
jac_ana = model.project_jacobian(pts, params).reshape(-1, len(params))
jac_num = nd.Jacobian(lambda p: model.project(pts, p).ravel())(params)
denom = max(float(np.linalg.norm(jac_num[:, col])), 1e-12)
err = float(np.linalg.norm(jac_ana[:, col] - jac_num[:, col])) / denom
assert err < 1e-6, f"...相对误差 {err:.2e} 超线 1e-6..."
```

FIM 的数值基准:小合成场景上 `nd.Jacobian` 造数值雅可比 → 数值 FᵀJᵀJ 比对解析路线(经 `make_jacobian_fn`),门槛照 03-RESEARCH 测试表(<1e-6)。**D-10 红线:numdifftools 只进 tests/**。

**要照抄的模式 2:静态锁定测试(grep 锁)。** `test_models_jacobian.py` 第 119-132 行 `test_numdifftools_never_in_src`——fisher 可仿写一条"src/calib/noise/ 里不出现投影公式/不 import cv2 投影函数"之类的静态断言(可选)。

**要照抄的模式 3:恒等式测试体例。** Schur 恒等式(`F_marg` 路线 vs 全矩阵求逆取内参块,逐位一致)与 σ 线性缩放恒等式,断言写法照第 76-81 行"分母兜底 + 带量级的中文断言消息"。

**合成场景地基:`tests/conftest.py` 已有的夹具直接复用:**
- `make_true_views(n_views)`(第 175-204 行):n 个已知真值视角的角点 + 真值 K——FIM 测试的合成观测就吃它;
- `make_chessboard_image(degrade="noise")`(第 157-168 行):**加已知 σ 噪声的合成图现成有**(σ=12 那段)——immerkaer 回收测试的"已知噪声图"直接可仿这一段写参数化版本。

### `tests/test_noise_immerkaer.py` / `test_noise_temporal.py` / `test_noise_injection.py`

体例仿 `tests/test_corners_artifacts.py`(产物读写往返 + 形状断言 + 中文失败消息)与 `tests/test_solve_observation.py`(输入构造 → 断言 → 异常路径断言)。要点:
- immerkaer:合成纯色图 + 已知 σ → 估计值相对误差 <5%(常数锁定测试,03-RESEARCH A2);
- temporal:合成连拍(同一角点 + 已知扰动)→ 逐角点 σ 回收 + 单帧退化分支 + 失败契约;
- injection:小样(合成 2-3 图 × 2 档 × 1 重复)断言位移数组形状、同种子逐位复现、σ=0 档恒零;真实全量标注 `pytest.mark.integration`(体例照 `test_smoke_realdata.py` 第 34 行 `pytestmark = pytest.mark.integration`)。

### `tests/test_noise_crosscheck.py` — 仿 `tests/test_models_parity.py` + `test_solve_solvers.py`

合成小场景(20 视图级)上跑 `calibrateCameraExtended`,断言"结构比对(尺度归一后逐参数 <5%)"协议本身成立——免真实数据锁协议(03-RESEARCH §Pattern 4)。合成视角用 conftest 的 `make_true_views`。

### `tests/test_smoke_realdata.py`(扩展)

**要照抄的模式:独立进程接力 + 产物键全检查。** 第 95-109 行的接力循环:

```python
for step in ("ingest", "corners", "calibrate"):
    result = subprocess.run(["uv", "run", "--project", str(REPO_ROOT), "calib", step], ...)
    if result.returncode != 0:
        raise AssertionError(f"冒烟链在 calib {step} 处断裂(退出码 {result.returncode})...")
```

扩展点:接力序列加 `"noise"`, `"fisher"` 两步(fisher 依赖 calibrate 产物,排在后面);仿第 200-226 行 `test_matrix_structure_complete_even_if_not_converged` 给 noise.json / crlb.json 加"结构键全在 + 数值 null 契约"断言;仿第 153-197 行给 noise→fisher 的血缘链加指纹闭合断言。

## 共享模式(横切,全部新文件都吃)

### 1. 中文报错体例(全项目硬性)
**来源:** `io/config.py` 第 224-229 行 `_fail`(带字段名 + 实际值 + 下一步动作);`models/base.py` 第 139-149 行 `as_params`(入口形状断言)。
**应用于:** 全部 7 个 src 新文件。格式固定:`f"配置/输入 <名字> 不合法:<白话要求>;实际是 {值!r}。<下一步动作>"`。

### 2. npz 只进纯数组、读写一律 `allow_pickle=False`
**来源:** `corners/artifacts.py` 第 17-19、91-98 行;`solve/result.py` 第 231-242 行。
**应用于:** artifacts.py 全部读写、temporal/injection 读角点、fisher 读 extrinsics。文字列用定长 unicode(`np.asarray(字符串列表)`)即可不带 pickle。

### 3. 血缘指纹
**来源:** `util/lineage.py` 第 48-73 行 `make_lineage(config_files, upstream, round_id)`;组装样板在 `cli/main.py` 第 382-386 行。
**应用于:** noise 与 fisher 两个命令、每个落盘产物。upstream 键名照惯例用文件名(如 `"corners_train_sb.npz"`、`"noise.json"`、`"result.json"`)。

### 4. 失败契约(结构键全、数值 null、中文说明)
**来源:** `solve/result.py` 第 81-111 行 `make_failure`;编排级兜底在 `solve/matrix.py` 第 181-188 行;CLI 附加产物兜底在 `cli/main.py` 第 767-769 行。
**应用于:** artifacts.py(写盘侧)、fisher(逐模型 try/except)、crosscheck(整段可失败不阻塞)、注入实验逐档失败率(是数据不是错误)。

### 5. 参数只从 config 读(CLI 零参数,D-12)
**来源:** `io/config.py` 白名单校验全套;`cli/main.py` 第 64-66 行路径常量、`corners()` 全函数零参数。
**应用于:** noise/fisher 命令签名保持无参;σ 网格、重复数、并行数、退化阈值、敏感性档位全部进 `calibration.yaml` 的 `noise:`/`fisher:` 段。

### 6. 产物目录与重跑不覆盖
**来源:** `cli/main.py` 第 379-381 行 `run_<时间戳>`;`io/manifest.py` 的 `latest_run_dir`。
**应用于:** `output/noise/run_<时间戳>/` 与 `output/fisher/run_<时间戳>/`。

### 7. 库层不依赖 CLI 层
**来源:** `solve/matrix.py` 第 54 行 `ProgressFn = Callable[[dict], None]`(进度回调注入,typer 只在 CLI 层)。
**应用于:** noise/fisher 的库函数收 `progress: Callable | None = None` 参数,播报由 CLI 注入——测试跑库函数时不需要 typer。

### 8. 雅可比唯一数学源(D-10,红线)
**来源:** `solve/residuals.py` 模块 docstring 第 1-24 行(行序契约与"绝不自创下标"的防线说明)+ `make_jacobian_fn` 第 164-200 行。
**应用于:** fisher.py 的 J 只从该工厂来;哪怕只要内参列,也走同一工厂再按列分段(`J[:, :n_int]`),不另写组装。

## 无足够现成仿写、按 03-RESEARCH 新写的部分

| 文件/部分 | 角色 | 数据流 | 原因与出处 |
|------|------|-----------|----------------|
| `injection.py` 的进程并行 | service | batch | 仓库尚无任何多进程代码;按 03-RESEARCH §Pattern 2 用标准库 `concurrent.futures.ProcessPoolExecutor(max_workers=min(8, os.cpu_count()))`,worker 数上限进 config |
| `transfer.py` 的 PCHIP 插值 | service | transform | 仓库无插值代码;`scipy.interpolate.PchipInterpolator` 单调保形,按 03-RESEARCH"Don't Hand-Roll"表 |
| `fisher.py` 的 Schur 消元与 CRLB 代数 | service | transform | 仓库无矩阵代数模块;公式与骨架代码 03-RESEARCH §Code Examples 已给全(F_ββ 块对角批量求逆 + einsum 组装),照骨架写、J 从工厂取 |
| `crosscheck.py` 的尺度归一比对协议 | service | transform | 协议本身是本阶段新产出(RESEARCH §Pattern 4 三步),OpenCV 调用部分仿 `solve/opencv.py` |

## 元信息

**检索范围:** `src/calib/`(9 个子包全部)、`tests/`、`config/`
**读过的现成文件数:** 15(residuals / observation / artifacts / detect / config / lineage / opencv / cli.main / matrix / result / models.base / models.__init__ / conftest / test_models_jacobian / test_smoke_realdata;另核 manifest.py、noise/__init__.py、visualization/overlay.py 局部)
**提取日期:** 2026-10-18
