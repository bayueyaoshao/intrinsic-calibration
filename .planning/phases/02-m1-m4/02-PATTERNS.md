# Phase 2: 模型层 M1–M4 + 双求解后端 + 产物血缘 - 模式映射

**映射日期:** 2026-10-17
**分析文件数:** 25(新建 17、修改 5、文档回写 3)
**找到参照:** 22 / 25(matplotlib 出图本体与两个"全新接口"无现成代码参照,已标注)

> 术语白话:**参照(analog)**= 代码库里已有的、新文件应该照抄结构的那个文件;**失败契约** = 求解失败时返回的固定结构(键一个不少、数值一律 null、中文失败说明);**血缘** = 落盘产物里记录"从哪些文件、哪份配置算出来的"的指纹字典。

## 文件分类

角色取值:model(数学/数据模型)、service(计算服务)、utility(纯函数工具)、config(配置)、route(CLI 命令)、test(测试)、doc(文档)。数据流取值:transform(数组进数组出)、file-I/O(读写落盘产物)、batch(批量计算编排)、doc-edit(纯文档)。

| 新建/修改文件 | 状态 | 角色 | 数据流 | 最近参照 | 匹配度 |
|----------------|------|------|--------|----------|--------|
| `src/calib/models/base.py` | 新建 | model | transform | `src/calib/corners/detect.py`(统一接口模式) | role-match |
| `src/calib/models/pinhole_brown.py` | 新建 | model | transform | `src/calib/corners/canonical.py` | role-match |
| `src/calib/models/kb4.py` | 新建 | model | transform | `src/calib/corners/canonical.py` | role-match |
| `src/calib/models/__init__.py` | 修改(现仅 docstring) | config(注册表) | — | `src/calib/corners/detect.py` 模块级字典 + `io/config.py` 白名单 | role-match |
| `src/calib/solve/observation.py` | 新建 | model+装载 | file-I/O | `src/calib/solve/baseline.py` 206-243 行 | exact |
| `src/calib/solve/result.py` | 新建 | model+落盘 | file-I/O | `src/calib/solve/baseline.py` `save_calibration` | exact |
| `src/calib/solve/opencv.py` | 新建 | service | batch | `src/calib/solve/baseline.py` `calibrate_pinhole/fisheye` | exact |
| `src/calib/solve/scipy_solver.py` | 新建 | service | batch | `src/calib/solve/baseline.py` `calibrate_fisheye`(cfg 传参/失败契约) | role-match |
| `src/calib/solve/residuals.py` | 新建 | utility | transform | `src/calib/solve/baseline.py` `compute_residuals` | exact |
| `src/calib/solve/scoring.py` | 新建 | service | batch | `baseline.py` 163-168 行逐帧回代 | role-match |
| `src/calib/io/config.py` | 修改 | config | — | 自身 `_parse_fisheye` / `_parse_corners` | exact |
| `config/calibration.yaml` | 修改 | config | — | 自身 `calibrate.fisheye` 段 | exact |
| `src/calib/cli/main.py` | 修改(calibrate 命令) | route | batch | 自身 `calibrate` 命令 670-743 行 | exact |
| `src/calib/solve/baseline.py` | **不动**(保留可复现) | — | — | — | — |
| `tests/test_models_parity.py` | 新建 | test | transform | `tests/test_solve_baseline.py` + RESEARCH 代码模板 | partial |
| `tests/test_models_jacobian.py` | 新建 | test | transform | 同上 | partial |
| `tests/test_models_registry.py` | 新建 | test | — | `test_solve_baseline.py` 源码静态断言模式 | role-match |
| `tests/test_solve_observation.py` | 新建 | test | file-I/O | `test_solve_baseline.py` 剔除/视图数用例 | exact |
| `tests/test_solve_solvers.py` | 新建 | test | batch | `test_solve_baseline.py` 真值恢复 + 结构完整用例 | exact |
| `tests/test_solve_scipy_residuals.py` | 新建 | test | — | `test_solve_baseline.py` monkeypatch + 源码断言 | exact |
| `tests/test_calibrate_matrix.py` | 新建(integration) | test | batch | `tests/test_smoke_realdata.py` | exact |
| D-04 存疑对比出图(visualization 层新函数/文件) | 新建 | service(画图) | file-I/O | `visualization/mosaic.py` 模块结构;matplotlib 本体无先例 | partial |
| `pyproject.toml` | 修改(dev 组加 numdifftools) | config | — | 自身 19-23 行 | exact |
| `tests/conftest.py` | 可能修改(合成真值视角夹具上提) | test | — | 自身 `make_chessboard_image` | exact |
| `docs/pipeline.md` + `01-HUMAN-UAT.md` Gap 回写 | 修改 | doc | doc-edit | 既有文档体例 | exact |

## 模式指派

### `src/calib/models/base.py`(model,transform)— CameraModel 抽象 + 共享导数工具

**参照:** `src/calib/corners/detect.py`(全项目唯一的"统一接口 + 同构返回、下游不分叉"先例);抽象基类本身代码库无先例,接口签名照 02-RESEARCH.md §Pattern 1。

**统一接口的既有示范**(`detect.py` 1-16 行模块 docstring + 44-48 行函数签名):
```python
# detect.py 模块开头:先讲两条路线的差异与共同约定,再讲"绝不返回半成品"
def detect_classic(gray: np.ndarray, cols: int, rows: int, cfg: CornersConfig) -> dict | None:
    """经典检测器:粗定位 → 二次亚像素精化(窗口按实测方格边长)→ 规范化。

    返回 dict(uv (rows*cols, 2) float64, sq_px, win_size, detector="classic"),
    未检出返回 None。
    """
```
新文件照抄三点:模块 docstring 开篇用白话讲清"四类模型共享同一套投影与雅可比,理论界与实际求解才可比"(现有 `models/__init__.py` 1-7 行已有底稿,保留并入);每个抽象方法的 docstring 写明输入输出形状与 dtype(float64 进出);失败语义写死(非法参数抛中文 AssertionError,不返回半成品)。

**纯 numpy 数学函数的形状约定**(`src/calib/corners/canonical.py` 28-39 行):
```python
def canonicalize(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """把行主序角点序列规范化为唯一形态……

    corners: (rows*cols, 2) 任意变体的行主序角点;
    返回:  (rows*cols, 2) 规范化后的新数组(不修改输入)。
    """
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
```
`base.py` 的共享导数工具(旋转扰动块 ∂Xc/∂ω = −[Xc]×、归一化坐标对相机坐标导数)照此:入参先 `np.asarray(..., dtype=np.float64)`、docstring 写形状、返回新数组。

**共享模式的错误写法**(`baseline.py` `_prepare_view` 50-65 行)——形状断言 + 三要素中文报错(期望什么/实际什么/怎么查),模型层的参数形状断言直接套用:
```python
    if arr.shape != (n_points, 1, dim):
        raise AssertionError(
            f"{what} 形状不合:期望每视图 ({n_points},{dim}) 或 ({n_points},1,{dim}),"
            f"实际是 {arr.shape}。角点数组必须与棋盘的 {n_points} 个内角点一一对齐,"
            f"请检查角点文件的规范化顺序是否正确。"
        )
```

---

### `src/calib/models/pinhole_brown.py` 与 `src/calib/models/kb4.py`(model,transform)

**参照:** `src/calib/corners/canonical.py` —— 全项目最接近"纯 numpy 数学公式模块"的文件:模块 docstring 用大段白话解释为什么(为什么需要规范化、判定机制),函数逐个带形状说明与返回值语义。

**结构照抄点(canonical.py 1-21 行):** 模块开头讲清模型族的白话定义(M1/M2/M3 = 同一个 Brown-Conrady 畸变核、系数个数不同;KB4 = "等距投影"),再讲嵌套方式与近轴保护这些"为什么"。`baseline.py` 1-18 行的名词白话段("重投影误差 = 重画位置与实际检测位置的差距")是术语首现解释的标准写法。

**数值实现本体:** 公式不照抄任何现有文件(现有投影走 cv2 黑盒),照 02-RESEARCH.md §Code Examples 四段实测验证代码:`project_brown`(切向项加在有理分母之外)、`project_kb4`(`np.where(r < 1e-12, 1.0/z, theta_d/r)` 近轴保护)、雅可比三组成块。每个公式旁的注释照 `baseline.py` 87-89 行的写法解释数值决策:
```python
    # OpenCV 4.14 的 calibrateCamera 绑定只收 32 位浮点(Point2f/Point3f),
    # 传 64 位会报"Unsupported format"——这里显式降精度再调用:
    # 像素坐标最大几千,float32 约有 1e-4 像素的表达精度,远小于 0.5px 的断言线。
```

**dtype 硬规则(来自 RESEARCH Pitfall 1):** 模型层一律 float64 进出;OpenCV 适配器(各模型的类方法)把 float32 装包/解包隔离成唯一入口,适配器内注释照上面这段的"为什么 + 量级"格式。

---

### `src/calib/models/__init__.py`(注册表)

**参照:** 模块级字典 + 名字白名单两个既有先例拼合;注册装饰器本体无先例,照 02-RESEARCH.md §Pattern 1 的 12 行草案。

**模块级映射字典**(`detect.py` 28-33 行):
```python
# calibration.yaml 里经典检测器的选项名 → cv2 常量(名字在配置层已校验过)
_CLASSIC_FLAG_VALUES = {
    "adaptive_thresh": cv2.CALIB_CB_ADAPTIVE_THRESH,
    ...
}
```

**非法取值的中文报错**(`io/config.py` 289-292 行):
```python
    bad = [f for f in flags if f not in _ALLOWED_CLASSIC_FLAGS]
    if bad:
        _fail("corners.classic_flags", bad,
              f"含有不认识的选项;可选:{sorted(_ALLOWED_CLASSIC_FLAGS)}")
```
`create_model(model_id)` 查表未命中照此格式:报"不认识的模型名 + 可选清单",并提示写进 calibration.yaml 的 models 段。现有 1-7 行 docstring 保留,追加注册机制白话(一处注册、两个求解器都能用)。

---

### `src/calib/solve/observation.py`(装载ObservationSet)

**参照:** `src/calib/solve/baseline.py` `run_baselines` 的装载/剔除/断言循环(206-243 行)——这段逻辑直接搬过来,从"按检测器字典"改成"按 corner_route 选 sb 路线"。

**装载循环(照抄结构,`baseline.py` 217-243 行):**
```python
    ids = [str(x) for x in np.asarray(data["image_ids"]).ravel()]
    uv = np.asarray(data["uv"], dtype=np.float64)
    detected = np.asarray(data["detected"], dtype=bool)
    ...
    for i, image_id in enumerate(ids):
        if image_id in exclude:
            excluded_seen.append(image_id)
            continue  # 人工剔除清单:逻辑剔除,原数据不动(D-03)
        if not detected[i]:
            continue  # 未检出的帧是 NaN 角点,进不了标定
        if not np.isfinite(uv[i]).all():
            raise AssertionError(
                f"检测器「{detector}」的照片 {image_id} 标记为已检出,"
                f"但角点里混入了非数值(NaN/无穷),角点文件可疑,请重新运行 calib corners。"
            )
        views.append(uv[i])
        kept_ids.append(image_id)
    if len(views) < 2:
        raise ValueError(
            f"检测器「{detector}」剔除后只剩 {len(views)} 个可用视图"
            f"(检出且未被人工剔除),至少需要 2 个才能标定内参。"
            ...
```
新增:`corner_source` 从 npz 的 `source` 数组随行携带;`image_ids` 断言与 npz 逐项相等(RESEARCH Pitfall 10 行序契约)。

**npz 读取防线**(`corners/artifacts.py` `load_corners` 91-98 行,新装载函数照抄):
```python
def load_corners(path: Path | str) -> dict[str, np.ndarray]:
    """读取角点 npz,返回 {键: 数组} 字典。

    强制 allow_pickle=False:遇到含 pickle 对象的文件直接抛错,
    绝不执行其中的任意代码(反序列化安全防线,测试有锁定)。
    """
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
```

**冻结数据类字段风格**(`io/config.py` 40-44 行的 `DataDirs`):ObservationSet 用 `@dataclass(frozen=True)`,每个字段一行中文注释说明用途与形状(RESEARCH §Pattern 2 已给字段草案,照抄并补注释)。

---

### `src/calib/solve/result.py`(CalibrationResult + 落盘/读盘)

**参照:** `src/calib/solve/baseline.py` `save_calibration`(300-349 行)——失败契约、JSON 结构、npz 三件套的直接前身。

**失败契约落盘(照抄结构,313-324 行):**
```python
    converged = bool(result["converged"])
    if converged:
        per_view_dict = {image_id: float(rms) for image_id, rms
                         in zip(result["image_ids"], result["per_view_rms"])}
        payload = {
            "rms_overall": float(result["rms_overall"]),
            "K": np.asarray(result["K"], dtype=np.float64).tolist(),
            ...
        }
    else:
        payload = {"rms_overall": None, "K": None, "D": None, "per_view_rms": None}

    document = {
        "lineage": lineage,
        ...
        "converged": converged,
        "failure_note": result["failure_note"],
        **payload,
    }
    json_path = out_dir / "calibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
```
docstring 里照抄 306-308 行的值语义声明("键一个不少,上游读取逻辑一视同仁")。新增:`params` 按 param_names 逐名给出、`solver_meta`、`init_from`;`extrinsics.npz` 与 `residuals.npz` 落盘(np.savez,数组全部纯数值/定长 unicode,禁 pickle 路径不破)。

**JSON 写盘统一格式:** `json.dump(..., ensure_ascii=False, indent=2)` + `open(..., encoding="utf-8")` —— 全代码库所有写 JSON 处一致(baseline.py 339-340、artifacts.py 86-87、cli/main.py 527-528),新文件照抄。

---

### `src/calib/solve/opencv.py`(OpenCVSolver)

**参照:** `src/calib/solve/baseline.py` `calibrate_pinhole`(68-116 行)与 `calibrate_fisheye`(119-177 行)——函数签名、float32 装包、try/except 失败契约、float64 解包全部照抄,只是从"写死 pinhole/fisheye"改成"从模型的 OpenCV 适配器取 flags/切片"。

**求解 + 失败契约核心(照抄 93-116 行):**
```python
    flags = 0  # 默认:估计 5 个畸变系数 k1,k2,p1,p2,k3(项目模型 M2)
    # 与鱼眼分支同构:求解抛 cv2.error(如病态输入让方程组解不动)时,
    # 失败完整记录、不中断其余组合——run_baselines 的准入门槛只有
    # "可用视图 >= 2",两三个近平面视角正是 calibrateCamera 的病态区
    try:
        rms, K, D, rvecs, tvecs, _std_int, _std_ext, per_view = cv2.calibrateCameraExtended(
            objs32, imgs32, tuple(image_size), None, None, flags=flags)
    except cv2.error as e:
        return {"rms_overall": None, "K": None, "D": None, "rvecs": None,
                "tvecs": None, "per_view_rms": None, "flags_used": flags,
                "converged": False,
                "failure_note": f"针孔模型求解未收敛:OpenCV 报错——{e}"}
```
注意:阶段二里 rms_overall/per_view_rms 改为 models.project 回代统一算(RESEARCH Pitfall 9),OpenCV 返回的 RMS 只进 solver_meta——这是对参照的唯一偏离,docstring 里写明白话原因("两把尺子刻度差 1e-4px,跨后端对比必须同一把尺")。

**fisheye 分支差异点**(139-149 行):K 传单位阵、D 传零向量交 OpenCV 内部初始化,criteria 从 cfg 组装——M4 适配器路径照抄此形态,flags 默认演进为 10(RECOMPUTE|FIX_SKEW)。

**D-09 兜底路径:** 模型无适配器时 `default_params` 组 K + 逐视图 `cv2.solvePnP` 估位姿,拼"初值版 CalibrationResult"(失败契约同样适用)。

---

### `src/calib/solve/scipy_solver.py`(ScipySolver)

**参照:** `baseline.py` `calibrate_fisheye` 的函数形态(views + image_size + board + cfg → 结果字典,失败契约同构);`least_squares` 调用本体无先例,照 RESEARCH §Pattern 2 实测配置(trf + lsmr + jac_sparsity + x_scale='jac')。

**cfg 传参风格**(143-145 行):求解参数从冻结 dataclass 读,不写死——
```python
    flags = int(cfg.flags)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                int(cfg.max_iter), float(cfg.epsilon))
```
scipy 版对应 `method/tr_solver/loss/f_scale/max_nfev/ftol/xtol/gtol` 全从新的 ScipyConfig 读。

**"实用收敛"判定**(RESEARCH Pitfall 3):写成一个纯函数 + 单测;失败契约复用——status 停 0 且代价未平台时 `converged=False`、中文 failure_note 如实写"迭代预算用尽但代价已平台";判定依据全进 solver_meta,不静默。

**初值装箱:** 从 OpenCVSolver 的 CalibrationResult 读 params + 逐视图 rvecs/tvecs 直接拼 θ 向量;残差与雅可比 import `solve/residuals.py`(D-08:残差仅由 models.project 构成)。本文件不 import 任何具体模型,只见 `CameraModel` 协议(反模式清单第 4 条)。

---

### `src/calib/solve/residuals.py`(残差与稀疏雅可比组装)

**参照:** `baseline.py` `compute_residuals`(275-297 行)——逐帧回代的循环骨架;"逐视图循环、视图内全 numpy"的向量化要求按 RESEARCH Pitfall 5(禁逐点 Python 循环)。

**回代骨架(照抄 292-297 行,投影函数换成 models.project):**
```python
    residuals = np.full((len(result["image_ids"]), n, 2), np.nan, dtype=np.float64)
    for i, (rvec, tvec) in enumerate(zip(result["rvecs"], result["tvecs"])):
        proj, _jacobian = project(objp, rvec, tvec, result["K"], result["D"])
        observed = np.asarray(result["views"][i], dtype=np.float64).reshape(n, 2)
        residuals[i] = proj.reshape(n, 2) - observed
    return residuals
```
**行序约定写进模块 docstring**(RESEARCH Pattern 2):残差按 视图→点→(u,v) 两行交错排列,与雅可比行严格对齐——docstring 首段照 canonical.py 的"为什么"写法讲清错位后果(Pitfall 6:无报错但收敛不动)。稀疏结构本体照 RESEARCH §Code Examples 的 COO 三元组向量化拼接,不自建稀疏索引(Don't Hand-Roll 表)。

---

### `src/calib/solve/scoring.py`(逐帧回代评分 + 只估位姿小求解)

**参照:** `baseline.py` 163-168 行的逐帧 RMS 计算:
```python
    # 逐帧 RMS:把物点按求得的位姿重新投影(KB4 模型的投影函数),与检测角点比差
    per_view = np.empty(len(imgs), dtype=np.float64)
    for i, (obj, img) in enumerate(zip(objs, imgs)):
        proj, _jacobian = cv2.fisheye.projectPoints(obj, rvecs[i], tvecs[i], K, D)
        diff = proj.reshape(-1, 2) - img.reshape(-1, 2)
        per_view[i] = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
```
差异:投影统一改 models.project;新增 test 帧"只估位姿"6 参数小求解(同样 `least_squares`,内参固定),对任何注册模型成立——玩具模型也不例外(D-09)。

---

### `src/calib/io/config.py` 扩展(config)

**参照:** 自身。三段既有模式直接套:

1. **冻结 dataclass + 字段中文注释**(`FisheyeConfig` 107-118 行,含"为什么默认 2"的白话注释)——新增 `ScipyConfig`(method/losses/f_scale/max_nfev/ftol/xtol)与 CalibrateConfig 的 models/solvers/corner_route 字段照此,f_scale 的 "auto" 语义在注释里写明白话(3×1.4826×MAD,用中位数衡量的波动大小)。
2. **取值白名单校验**(`_ALLOWED_CLASSIC_FLAGS` + 289-292 行 bad 清单报错)——models 取值白名单 {m1,m2,m3,m4}、solvers 取值白名单 {opencv, scipy}、losses 白名单 {linear, soft_l1, huber} 照抄。
3. **`_parse_fisheye` 的逐字段校验**(308-320 行)——每个字段 `isinstance` + 范围检查 + `_fail(带点路径名, 实际值, 白话要求)`,新 `_parse_scipy` 逐行照此展开。

`_fail` 的三要素格式(160-165 行)不动:字段名 + 白话要求 + 实际读到的值。

---

### `src/calib/cli/main.py` calibrate 命令演进(route)

**参照:** 自身现有 `calibrate` 命令(670-743 行)。骨架保留:读配置 → `latest_run_dir(OUTPUT_ROOT, "corners")` 定位上游 → 装载 → 跑矩阵 → `run_<时间戳>` 目录 → 逐组合 `make_lineage` + 落盘 → comparison.json → 中文对比表。

**逐组合血缘组装(照抄 707-716 行,upstream 换成新矩阵的输入):**
```python
            lineage = make_lineage(
                config_files={"board.yaml": BOARD_PATH,
                              "calibration.yaml": CALIBRATION_PATH},
                upstream={npz_path.name: npz_path},
                round_id=round_id,
            )
            lineage["corners_file"] = str(npz_path)
            lineage["corners_sha256"] = sha256_file(npz_path)
            lineage["excluded_images"] = outcome["excluded_images"]
```
scipy 精化路径的 upstream 加 OpenCV 初值 result.json(RESEARCH Pattern 3)。

**中文汇总表(照抄 731-738 行的格式化表格 + 白话注解段):**
```python
    typer.echo(f"  {'组合':<28}{'视图数':<8}{'整体RMS(像素)':<16}{'收敛':<6}")
    for row in comparison:
        rms_text = f"{row['rms_overall']:.4f}" if row["rms_overall"] is not None else "—"
        flag = "是" if row["converged"] else f"否({row['failure_note'][:24]}……)"
```
演进:组合名换成 `<model>__<solver>`;表后白话注解照 736-738 行体例,写明 Pitfall 8("精化不劣于纯 OpenCV"是同模型内比较,M1/M2 在强鱼眼镜头上 RMS 仍高属预期发现)。

**上游定位防串轮**(652-667 行 `_manifest_path_for` 的"血缘链闭合"模式):D-04 join manifest 时照此——优先用角点 meta 里记录的上游清单路径。

**进度播报**(ingest 命令 116-117 行):全矩阵试车 15-25 分钟,照抄 `if (k + 1) % 100 == 0: typer.echo(...)` 的节奏播报。

**D-04 的 manifest join 先例**(246-294 行 `_suspect_review_frames`):按 `precheck.verdict == "suspect"` 筛帧、保留 `suspect_sources` 原始记号 + 中文翻译、陌生取值原样保留不吞掉——存疑复核的数据 join 照此组装,结论落盘走 make_lineage。

---

### 测试文件(6 个新建)

**参照:** `tests/test_solve_baseline.py`(组织范式)+ `tests/test_smoke_realdata.py`(integration 范式)+ `tests/conftest.py`(夹具)。

**测试模块组织**(`test_solve_baseline.py` 1-11 行):模块 docstring 开篇列全"锁定行为清单"——
```python
"""2×2 基线标定的合成数据回归测试:已知真值的相机参数,断言求解数学正确。

全部用例跑在合成数据上(相机内参/位姿全已知,角点是带"标准答案"的):
- 针孔模型:8 个不同位姿的已知真值视角,断言 RMS < 0.5px 且 fx/fy/cx/cy
  对真值误差 < 1%;
...
```
每个新测试文件照此:docstring 列锁定的决策编号(D-06/D-08/D-09……)+ 断言量级 + 为什么选这个量级。

**合成真值夹具**(conftest.py `make_chessboard_image` + test_solve_baseline.py `_make_true_views` 32-53 行):
```python
def _make_true_views(make_chessboard_image, n_views: int = 8):
    """造 n_views 个不同位姿的已知真值视角(默认内参,深度统一 800)。

    返回 (角点列表, 真值 K)。深度固定是必须的:合成夹具按"焦距×物理尺寸/距离"
    反推方格物理边长,各视角深度不一会让物点网格尺度自相矛盾。
    """
```
`test_solve_solvers.py` 直接复用;parity/jacobian 测试的点集构造(近轴 r≈1e-6、大半径、随机种子≥3)按 RESEARCH §Validation Architecture 写,fixture 参数化模式照 `make_chessboard_image` 的"全部有默认值,测试只写关心的那一两项"。

**失败契约断言循环**(test_solve_baseline.py 142-156 行):
```python
    # 结构键一个不少(失败的值语义:数值字段为 None,键必须在)
    for key in ("rms_overall", "K", "D", "rvecs", "tvecs", "per_view_rms",
                "flags_used", "converged", "failure_note"):
        assert key in result, f"鱼眼结果缺结构键:{key}"
    ...
    else:
        note = result["failure_note"]
        assert isinstance(note, str) and note.strip(), "未收敛时必须有中文失败说明"
```
CalibrationResult 的新键表(model_id/solver/solver_meta/params/init_from/lineage)逐个加进这个循环——两个后端各跑一遍。

**monkeypatch 替身模式**(test_solve_baseline.py 159-171 行,`test_solve_scipy_residuals.py` 的 D-08 锁定直接套用):
```python
    original = cv2.calibrateCameraExtended

    def _explode(*args, **kwargs):
        return original([], [], (640, 480), None, None, flags=0)

    monkeypatch.setattr(cv2, "calibrateCameraExtended", _explode)
```
D-08 版:monkeypatch `model.project` 加已知偏移,断言残差逐项同移。

**源码静态断言模式**(test_solve_baseline.py 276-293 行 `test_square_size_decoupled_from_pixel_logic`)——D-09/D-08 的 import 图锁定与"求解器不 import 具体模型"断言照此:
```python
    baseline_src = (REPO_ROOT / "src/calib/solve/baseline.py").read_text(encoding="utf-8")
    assert "square_size" in baseline_src, "build_object_points 应引用 square_size"

    window_src = (REPO_ROOT / "src/calib/corners/window.py").read_text(encoding="utf-8")
    assert "square_size" not in window_src, \
        "窗口推导是纯像素级逻辑,不得引用方格物理边长(square_size)"
```

**integration 范式**(`test_smoke_realdata.py`):文件头 `pytestmark = pytest.mark.integration`;module 级 fixture 搭缩小数据树(软链接 input_data,只读契约);`subprocess.run(["uv", "run", ..., "calib", step], cwd=临时树)` 独立进程接力;失败时带全量 stdout/stderr 报错(96-100 行);血缘链逐环 sha256 断言(143-174 行)。`test_calibrate_matrix.py` 照此,断言全矩阵产物齐全 + 新进程重跑可复现 + 存疑复核产物生成。

**中文断言消息**:所有 assert 带 f-string 说明量纲与超线值(如 66 行 `f"整体 RMS {result['rms_overall']:.3f}px 超线"`)——新测试逐条照抄。

---

### D-04 存疑对比出图(visualization 层)

**参照:** `src/calib/visualization/mosaic.py` 的模块结构(1-19 行:先讲定位与边界、再讲产物布局、图内标注只用 ASCII、中文说明在 JSON 与控制台)。**matplotlib 本体代码库无先例**(`visualization/__init__.py` 1-4 行只声明"仅用 matplotlib,架构决定",现有出图全是 cv2/numpy 拼贴)——叠加分布图按 RESEARCH A3 直方图/累积分布叠加实现,属本阶段第一个 matplotlib 图。产物落盘 `suspect_review/suspect_vs_normal.json + .png`,血缘 upstream = 标定 result.json + manifest.json。

---

### `pyproject.toml` / `config/calibration.yaml` / `docs` 回写

- **pyproject.toml**:dev 组(19-23 行)加 `"numdifftools>=0.11"` 一行,注释注明 D-10(只作测试基准,绝不进求解路径)。
- **calibration.yaml**:`calibrate` 段(64-74 行)按 RESEARCH §Pattern 4 草案扩展(corner_route/models/solvers/scipy 子段),每键中文注释讲默认值依据,fisheye 段 flags 默认值演进为 10 时照 66-71 行体例写实测依据注释。
- **docs/pipeline.md** 阶段契约速查表 + **01-HUMAN-UAT.md** Gap 闭环回写:文档编辑,沿用既有白话体例。

## 共享模式(横切全部新文件)

### 1. 失败契约(最高优先级)

**来源:** `src/calib/solve/baseline.py` 100-104、150-156、306-308 行
**适用:** OpenCVSolver、ScipySolver、scoring 小求解、result.py 落盘、全部求解类测试
```python
        return {"rms_overall": None, "K": None, "D": None, "rvecs": None,
                "tvecs": None, "per_view_rms": None, "flags_used": flags,
                "converged": False,
                "failure_note": f"针孔模型求解未收敛:OpenCV 报错——{e}"}
```
规则:converged=False 时结构键一个不少、数值字段一律 None(落盘成 JSON null)、failure_note 非空中文;求解抛 cv2.error/异常时捕获记录、不中断其余组合。"实用收敛"判定(scipy status=0 但代价已平台)的依据全部进 solver_meta,不静默。

### 2. 血缘指纹

**来源:** `src/calib/util/lineage.py` `make_lineage`(48-73 行)+ `cli/main.py` 707-715 行组装先例
**适用:** result.json、extrinsics/residuals npz、comparison.json、suspect_review 产物——全部新落盘产物
```python
    lineage = make_lineage(
        config_files={"board.yaml": BOARD_PATH, "calibration.yaml": CALIBRATION_PATH},
        upstream={npz_path.name: npz_path},
        round_id=round_id,
    )
```
零新机制;scipy 路径 upstream 加 OpenCV 初值产物。

### 3. npz 禁 pickle

**来源:** `src/calib/corners/artifacts.py` 97-98 行
**适用:** observation.py 读角点、result.py 的 npz 读盘函数、全部读 npz 的测试
```python
    with np.load(Path(path), allow_pickle=False) as data:
```
新写 npz 只放纯数值数组与定长 unicode(照 63-73 行 arrays 组装),字符串数组禁 object dtype。

### 4. 冻结 dataclass + 中文校验配置

**来源:** `src/calib/io/config.py` 107-118 行(dataclass)、160-165 行(`_fail` 三要素)、289-292 行(白名单)
**适用:** ScipyConfig、CalibrateConfig 扩展、ObservationSet 的 dataclass 部分
```python
def _fail(field_name: str, actual, requirement: str) -> None:
    """抛出带字段名与实际值的中文配置错误(统一格式,便于定位)。"""
    raise ValueError(
        f"配置字段 {field_name} 不合法:{requirement};实际读到的值是 {actual!r}。"
        f"请检查对应的 yaml 文件。"
    )
```
所有配置只经 load_calibration 进入,CLI 零参数(D-12)。

### 5. 通俗中文注释/报错体例(硬性 D-13)

**来源:** `baseline.py` 1-18 行(模块 docstring 名词白话段)、87-89 行(数值决策"为什么 + 量级")、50-64 行(形状断言三要素)
**适用:** 全部新文件。规则:术语首现就地一句白话;注释讲"为什么"不讲"是什么";报错含 期望值/实际值/下一步动作 三要素。

### 6. CLI 零参数 + 上游定位 + 进度播报

**来源:** `cli/main.py` 62-65 行路径常量、`io/manifest.py` `latest_run_dir`(193-212 行)、ingest 命令 116-117 行进度
**适用:** calibrate 命令演进。命令不接受算法参数,矩阵进 calibration.yaml;上游用 latest_run_dir 定位、manifest 路径优先血缘记录防串轮。

### 7. 合成真值测试地基

**来源:** `tests/conftest.py` `make_chessboard_image`(73-172 行)+ `test_solve_baseline.py` `_make_true_views`
**适用:** test_solve_solvers、test_models_registry;parity/jacobian 的新点集构造沿用其"默认参数 + 调用方只写关心的项"风格。

## 无参照文件(规划者用 02-RESEARCH.md 的模板)

| 文件/部分 | 角色 | 数据流 | 原因 |
|-----------|------|--------|------|
| `models/base.py` 的 ABC 接口签名 | model | transform | 代码库无抽象基类先例;照 RESEARCH §Pattern 1 已实测支撑的签名 |
| matplotlib 叠加分布图本体 | service | file-I/O | visualization 层尚无 matplotlib 代码(现有出图全 cv2/numpy);照 RESEARCH A3 |
| scipy `least_squares` 调用与稀疏雅可比组装 | service/utility | transform/batch | 项目首次引入;照 RESEARCH §Code Examples 实测配置,禁自建(Don't Hand-Roll 表) |

## 元数据

**参照检索范围:** src/calib 全部 26 个 py 文件、tests 全部 12 个文件、config/、pyproject.toml
**重点通读:** solve/baseline.py(349 行)、io/config.py(395 行)、cli/main.py(788 行)、corners/{detect,artifacts,canonical}.py、util/lineage.py、tests/{conftest,test_solve_baseline,test_smoke_realdata}.py
**模式提取日期:** 2026-10-17
