# Phase 5: 误差预算 + 诊断 + 角点提取寻优 - Pattern Map

**Mapped:** 2026-10-21
**Files analyzed:** 14(9 个新建 + 2 个修改 + 8 个新建测试,部分文件承担多需求块)
**Analogs found:** 14 / 14(全部在本仓库找到同角色同数据流的现成类体;零个无类体文件)

## File Classification

| 新建/修改文件 | 角色 | 数据流 | 最接近的现有类体 | 匹配质量 |
|---|---|---|---|---|
| `src/calib/analysis/artifacts.py` | 产物契约层 | file-I/O | `src/calib/evaluation/artifacts.py` | exact |
| `src/calib/analysis/budget.py` | 统计编排(ANA-01) | transform(消费落盘产物) | `src/calib/evaluation/experiments.py`(slices 像素空间段)+ `evaluation/selection.py` | role-match |
| `src/calib/analysis/residual_structure.py` | 统计编排(ANA-02) | transform | `src/calib/solve/scoring.py`(score_views/solve_pose_only 同一把尺)+ `experiments.py` 分组统计体例 | role-match |
| `src/calib/analysis/test_xyz.py` | 统计编排(ANA-03,唯一新建数学) | transform(逐帧小求解+线性代数) | `src/calib/solve/scoring.py` `solve_pose_only` + `models/base.py` `project_point_jacobian` | role-match |
| `src/calib/analysis/sensitivity.py` | 统计编排(ANA-04) | transform(特征分解+像素当量) | `src/calib/evaluation/experiments.py` `_px_dispersion` + `noise/fisher.py` 特征谱段 | role-match |
| `src/calib/analysis/corner_opt.py` | 实验编排(COR-02,唯一重计算块) | batch(并行网格实验+file-I/O) | `src/calib/noise/injection.py` | exact(范式骨架) |
| `src/calib/analysis/anisotropy.py` | 统计编排(COR-03) | transform(消费 npz) | `src/calib/noise/temporal.py`(sigma_per_corner 生产方,读法同源) | role-match |
| `src/calib/visualization/analyze_figs.py` | 可视化 | file-I/O(matplotlib→PNG) | `src/calib/visualization/evaluate_figs.py` | exact |
| `src/calib/cli/main.py`(analyze 命令体替换 :1795-1797 占位) | CLI 编排 | batch(六块独立兜底) | 同文件 `evaluate` 命令(:1243-1792) | exact |
| `src/calib/io/config.py`(AnalyzeConfig + `_parse_analyze` + CalibrationConfig 加字段) | 配置 | config | 同文件 `EvaluateConfig`(:273-369)+ `_parse_evaluate`(:816-1001) | exact |
| `config/calibration.yaml`(新增 analyze: 段) | 配置 | config | 同文件 evaluate: 段(:185-290) | exact |
| `tests/test_analysis_budget.py` 等 8 个测试 | test | - | `tests/test_evaluation_experiments.py`(cv_setup/warm_setup/slice_setup 夹具、`_assert_reports_equal`、中文报错断言)+ `tests/conftest.py` `make_chessboard_image` | exact |

## Pattern Assignments

### `src/calib/analysis/artifacts.py`(产物契约层,file-I/O)

**类体:** `src/calib/evaluation/artifacts.py`(整文件 382 行照抄体例,analyze 换自己的键集合)

**模块头声明体例**(行 1-29 的写法:两层结构说明 + 失败契约来源 + 血缘来源 + "不跨模块借私有件"声明):
```python
"""analyze.json 产物契约与落盘:误差归因六块的输出侧。
...
本文件体例整文件照 src/calib/evaluation/artifacts.py,不跨模块 import 其
私有函数(各阶段产物契约各自成文,改一边不至于震塌另一边)。
"""
```

**契约键先验 + 失败契约工厂**(类体 evaluation/artifacts.py 行 195-265):`ANALYZE_KEYS` 元组一次定死;`make_failure_analyze(lineage, note)` 结构键全在、六块数值段一律 `None`、failure_note 非空否则中文 ValueError;`_check_analyze_contract` 缺键当场中文 AssertionError。照抄 `EVALUATE_KEYS`(:195-209)与 `make_failure_evaluate`(:227-254)的形状,把八个段换成 ANA-01/02/03/04 + COR-02/03。

**私有明细键 → npz 旁车**(行 214-224 + 285-341):每个需求块的明细大数组挂 `_xxx_detail` 私有键,写盘时经 `_flatten_detail`(嵌套字典双下划线拍平,`m2__group_uv` 这种键名就是它产的)转 `analysis/<块名>.npz`,配同名 `.meta.json` 带血缘;json 各段只留 `detail_npz` 引用(D-13 汇总只是索引)。
```python
_ANALYZE_DETAIL_SECTIONS = (
    ("_budget_detail", "budget"),
    ("_residual_detail", "residual_structure"),
    # ... 六块各一行
)
```

**npz 读回防线**(行 377-380,全仓唯一允许的 npz 读法):
```python
# npz 读取强制 allow_pickle=False:npz 的 pickle 模式可以执行
# 任意代码,关死它(项目全部读回路径的既定防线)
with np.load(npz_path, allow_pickle=False) as data:
    details[section] = {key: data[key] for key in data.files}
```

**numpy 类型转 json**(行 92-110 `_to_jsonable`):递归 ndarray→列表、np.generic→原生标量、Path→str;inf 原样通过。

---

### `src/calib/analysis/budget.py`(ANA-01 误差预算,transform)

**类体:** `src/calib/evaluation/experiments.py` 的 slices 像素空间对账段 + `evaluation/selection.py` 的告警线。

**口径铁律的读取入口**(`experiments.py:842-876` `_sigma_detector_from_noise`):analyze 只从 noise.json 读 `sigma_detector_corner`,失败轮/缺键/非正有限值全部中文 AssertionError;docstring 里"时域口径虚高 35 倍"的教训文案照抄口径说明风格。

**逐行对账 + 三统计量 + 零违例核查**(ANa-01 复核的现成写法,`experiments.py:1169-1223`):
```python
meas_rows = np.std(np.stack(uv_stack), axis=0, ddof=1).ravel()
bound_rows = np.mean(np.stack(bd_stack), axis=0).ravel()
ratios = meas_rows / np.maximum(bound_rows, 1e-12)
n_violations = int((meas_rows < bound_rows).sum())
crlb_alert["n_pointwise_violations"] = n_violations
crlb_alert["min_ratio_pointwise"] = float(ratios.min())
```
analyze 块 1 从 slices.npz 的 `m2__group_uv` / `m2__group_bound_px` 重算同一组量,断言与 evaluate.json `slices.models.m2.crlb_alert` 三统计量逐位一致(不一致 = 口径漂移,当场中文报错)。

**硬报错执行位**(`selection.py:280-393` `crlb_alert_check`):不要重写卡方线——直接调用它(df 自适应、双档线、判定式随结论落盘)。analyze 的硬报错 = 复用 evaluate 已落账的 `crlb_alert.lines.hard` + `n_pointwise_violations`,任何违例抛中文 AssertionError 后走失败契约退出码 1(错误文案体例照 :361-369 的 hard_hits 拼装)。

**k1–k3 参考表声明文案**(`experiments.py:1235-1241`,原样复用的现成文案体例):
```python
"crlb_caliber_note": (
    "crlb_se_mean 是参数空间参考表(σ=检测器级,内参块下界,"
    "位姿理想已知):fx/fy/cx/cy 等数据定得住的参数对照有效;"
    "k1..k6 等强相关参数的单独下界被参数间补偿空间撑大,不是"
    "噪声大,单独比较无意义。正式的「逼近 CRLB」判定在像素空间,"
    "见 crlb_alert(逐角点重投影位移,2026-10-20 定案)。"
),
```

**消费面键名契约**(2026-10-21 实测核实):`output/evaluate/run_*/experiments/slices.npz` 键 = `m2__group_params (8,9)` / `m2__group_uv (8,80,88,2)` / `m2__group_bound_px (8,2*80*88)` / `m2__group_se (8,9)` / `m2__param_names (9,)`;前缀按 `evaluate.json selection.selected` 拼(Pitfall 6:quick 档只有当选模型,别假设四模型全在)。

---

### `src/calib/analysis/residual_structure.py`(ANA-02 残差方向分解,transform)

**类体:** `src/calib/solve/scoring.py` 的同一把尺 + `experiments.py` 的逐帧统计体例。

**train 侧逐帧残差的唯一算法**(scoring.py:56-76 `score_views` → `compute_result_metrics`):残差 = `model.project(transform_to_camera(物点)) − 实测`,analyze 不重写,train 帧直接读 `output/calibration/run_*/m2__scipy/residuals.npz`(640,88,2),行序对齐 `extrinsics.npz` 的 `image_ids`。

**test 侧逐帧重评分**(scoring.py:79-107 `solve_pose_only` 签名与两段式路线——反投影初值 + scipy 精化;真实案例注释:直接零畸变 PnP 会进 4.82px 错误盆地):
```python
def solve_pose_only(uv_view: np.ndarray,
                    object_points: np.ndarray,
                    params: np.ndarray,
                    model: CameraModel) -> tuple[np.ndarray, np.ndarray, float]:
    """内参当尺子用、只挪棋盘的位姿,给没参与标定的照片打分。"""
```
调用纪律:输入形状校验抛中文 AssertionError(:109-125 体例)、单帧解不出按 ValueError 让调用方记失败不中断(模块头"失败语义"段)。

**r 归一化半径公式**(RESEARCH Pattern 2 定案,findings 结论 5;本阶段新写但只是一行 numpy):
```python
r = np.sqrt(((uv[:, 0] - cx) / fx) ** 2 + ((uv[:, 1] - cy) / fy) ** 2)
```
归一化基准(fx/fy/cx/cy 用哪一轮哪个模型)必须写进产物 meta(Pitfall 11);分带器一处实现,ANA-02 与 COR-03 共用。

---

### `src/calib/analysis/test_xyz.py`(ANA-03 三分量反解,transform)

**类体:** 数学上无现成整文件(本阶段唯一新建数学),但每个零件都有已验证接口,RESEARCH 已给全核心段。

**全部只用这三个既有接口**(签名核实于源码):

1. `models/base.py:65-81` `project_point_jacobian`——返回 `(N,2,3)` 解析导数,docstring 明确"前半截若不进模型层,求解侧就只剩数值差分一条路"(D-06 唯一数学源,analysis/ 绝不自己写导数公式):
```python
@abstractmethod
def project_point_jacobian(self, pts_cam: np.ndarray,
                           params: np.ndarray) -> np.ndarray:
    """投影对**相机坐标系点**的解析雅可比(外参链式的前半截)。
    返回:    (N,2,3) float64"""
```
2. `solve/scoring.py:79` `solve_pose_only`(见上一节);
3. `models/base.py:152-192` `transform_to_camera`——`pts_cam, _ = transform_to_camera(objp, rvec, tvec)`,第二返回值是位姿导数(ANA-03 不用,接收为 `_`)。

**组装与反解**(RESEARCH Code Examples 已定案,直接照抄):
```python
resid = (uv_pred - uv_view).ravel()                    # (176,) 逐角点像素残差
J_pt = model.project_point_jacobian(pts_cam, params)   # (88,2,3) 解析导数
A = J_pt.reshape(2 * len(uv_view), 3)                  # 堆叠 (176,3)
ata = A.T @ A
dX = np.linalg.solve(ata, A.T @ resid)                 # 最小范数反解 (3,)
sigma_theory = np.sqrt(np.diag(np.linalg.inv(ata)))    # 理论σ(dX 各分量)
eigvals = np.linalg.eigvalsh(ata)                      # 可观测性
```

**失败帧占行体例**(照 `experiments.py:1252-1258`:失败组整行 NaN + 计数落盘,不中断):病态帧(条件数 >1e4,dZ 不具备解释力)记 NaN 行 + 条件数列,聚合时显式计数(Pitfall 7)。

---

### `src/calib/analysis/sensitivity.py`(ANA-04 敏感度排序,transform)

**类体:** `experiments.py:257-289` `_px_dispersion` + `noise/fisher.py:870-885` 特征谱段。

**像素当量公共货币**(`_px_dispersion`,现成函数直接调或照抄方法;docstring 本身就是"为什么不用百分比"的现成白话文案,findings 结论 1 反面教材的出处):
```python
def _px_dispersion(params_list, model, object_points, ref_rvec, ref_tvec) -> float:
    """参数向量组换算成"像素位移当量":同一参考位姿下,每两组参数分别
    投影棋盘物点网格,逐角点位移的最大值(两两取最坏)。"""
```
ANA-04 的用法:沿特征方向 v 取 ±k·σ 步长扰动参数,投影 88 角点取最大位移;参考位姿取"第 0 视图"并随产物声明(findings 结论 1 惯例)。扰动单方向版 = 把两两循环退化成 {基准, 扰动} 一对,方法同源。

**特征谱的既有口径**(`fisher.py:870-885`,fisher_m2.npz 里已存好的就是这一段算的;ANA-04 只消费 npz,不重算):
```python
# 特征谱用与 crlb.json 同一套无量纲坐标(像素内参 ÷ 焦距),
# scale_dimless 一并存档,审计时可还原回原始单位。
cov640_dimless = np.linalg.inv(F640) / np.outer(scale640, scale640)
model_entry["_fisher_detail"] = {
    "eigvals_cov": np.linalg.eigvalsh(cov640_dimless),
    "eigvecs_cov": np.linalg.eigh(cov640_dimless)[1],
    "scale_dimless": scale640,
    "param_names": list(model.param_names),
}
```
读回纪律:`np.load(..., allow_pickle=False)`,键按 `fisher_m2.npz` 实际键名(`eigvals_cov (9,)` / `eigvecs_cov (9,9)` / `scale_dimless` / `param_names`)。

**应对建议规则表**:无现成类体,是 findings 结论 5 判据表 + pipeline.md 回头路的产品化;实现体例照 `selection.py` 的 verdict/verdict_note(判定式与阈值随结论落盘,不凭空下结论)。

---

### `src/calib/analysis/corner_opt.py`(COR-02 退化网格实验,batch,唯一重计算块)

**类体:** `src/calib/noise/injection.py`(整文件 344 行是完整范式骨架:改图 → 重检 → 逐角点比对 → NaN 留痕 → npz+json 双落盘)。

**种子确定性推导**(`injection.py:37-47` `_seed_for`,COR-02 扩成 (图像, 退化单元, 方法, 重复) 四元组,公式与注释照抄):
```python
def _seed_for(image_id: str, level_value: float, rep: int, base_seed: int) -> int:
    """从 (图像, σ 档, 重复) 三元组推导该副本的随机种子。
    公式:base_seed + sha1(f"{image_id}|{level}|{rep}") mod 2**32——
    确定性散列,任何一元不同种子就不同;同三元组永远同种子。"""
    digest = hashlib.sha1(
        f"{image_id}|{level_value!r}|{rep}".encode()).hexdigest()
    return (base_seed + int(digest, 16)) % 2**32
```

**退化图应用与量化纪律**(`injection.py:50-70` `add_gaussian_noise`;COR-02 的 `apply_degradation` 三族照抄"float 域 → clip → rint 回 uint8"收尾,量化步骤不能省,docstring 理由原文在 :56-58):
```python
noisy = arr.astype(np.float64) + rng.normal(0.0, sigma, size=arr.shape)
np.clip(noisy, 0.0, 255.0, out=noisy)
return np.rint(noisy).astype(np.uint8)
```

**部分成功约定**(`injection.py:106-142` `_inject_one_image` 的骨架):单副本检测失败记 NaN 是数据不是事故;`disp = np.full((..., n_points, 2), np.nan)` 预填 + `result["uv"] - uv_clean` 逐角点位移 + `src` 数组留检测来源。COR-02 的 worker 换成"单帧 × 全部退化单元 × 全部方法候选",双参照(方法自参照 + 生产 SB 干净参照)各自存一份位移数组。

**检测唯一入口红线**(`injection.py` 模块头 :4-8;`corners/detect.py:143/202` 两个入口签名):
```python
def detect_classic(gray, cols, rows, cfg: CornersConfig) -> dict | None: ...
def detect_sb(gray, cols, rows, cfg: CornersConfig) -> dict | None: ...
```
方法候选 = 检测入口参数,不新增检测实现;classic 换 winSize 阶梯、SB 换 flags。**SB 绝不二次精化**(`detect.py:53-64` `_subpix_refine` docstring:"全文件只有这里调用 cv2.cornerSubPix…SB 分支绝不调用本函数",有源码防线测试锁定)。

**窗口阶梯合法域**(`corners/window.py:42-59` `derive_window_frame`,COR-02 过滤档位的现成断言):
```python
w = max(int(min_win), int(sq_px // 2) - 1)
assert 2 * w + 1 < sq_px, (
    f"亚像素窗口断言失败:搜索区 {2 * w + 1} 像素未严格小于方格像素边长 {sq_px:.2f}。..."
)
```
COR-02 用法:每帧先 `estimate_sq_px` 定 `sq_px`,阶梯里 `2*w+1 >= sq_px` 的档**直接不进候选**(不报错,是设计内过滤);逐帧值来自各方法自己的干净检出 sq_px。

**并行调度换阶段四统一体例**:`evaluation/parallel.py:61-109` `run_parallel(tasks, worker, max_workers, progress)`——任务小元组、`TaskFailed` 信封(:31-45,joblib 任一任务抛异常会杀整条结果流的教训写在模块头)、`inner_max_num_threads=核数÷进程数` 写死。**worker 内再上 `cv2.setNumThreads(cfg.cv2_threads)`**(并行子进程是全新进程,04-02 教训;cfg 值经任务元组传入)。注入实验的 ProcessPoolExecutor 路线(:253-262)是旧体例,COR-02 不要照抄它,用 run_parallel。

**落盘双件套**(`injection.py:308-344`):`np.savez` 纯数组(位移 + 网格 + 编号 + 检出掩码 + 来源) + 逐单元 json 索引 + meta 里写"合成图不落盘、种子可复现"的正当性声明(:335-338 文案体例)。

---

### `src/calib/analysis/anisotropy.py`(COR-03 各向异性,transform)

**类体:** `src/calib/noise/temporal.py`(sigma_per_corner 的生产方,消费同一键结构)。

**原料键契约**(temporal.py:170-183 生产端写法,COR-03 按此读回):
```python
sigma_per_corner = np.std(rows["uv"][det_idxs], axis=0, ddof=1)
...
"sigma_per_corner": sigma_per_corner,   # (点数, 2) 数组,退化姿态为 None
```
实际 npz:`output/noise/run_*/temporal_sigma_per_corner.npz` 键 = `pose_ids (80,)` / `sigma_per_corner_all (80,88,2)` / `sources (80,) <U8`。

**来源过滤(Pitfall 9)**:`sources == "temporal"` 才是真连拍重复;`immerkaer_transfer` 折算姿态行可能是 NaN,必须过滤 + NaN 行显式排除并计数落盘。读回照例 `allow_pickle=False`。

**分带与切片**:与 ANA-02 共用同一 r 分带器(一处实现两处用——RESEARCH Pattern 6 明示);防零除 = 分母 clamp 到 detector σ 量级下限;判读给物理量级白话(memory 规则,对照 σ=0.0087px)。

---

### `src/calib/visualization/analyze_figs.py`(可视化,file-I/O)

**类体:** `src/calib/visualization/evaluate_figs.py`(整文件 266 行,函数级一一对应照抄)。

**模块头三件套**(:14-30,原样照抄——Agg 必须先于 pyplot、字体清单、unicode_minus):
```python
import matplotlib

matplotlib.use("Agg")  # 离线后端先于 pyplot 就位,导入顺序是刻意的
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK",
    "AR PL UMing CN", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
```

**函数形状**(:37-90 `plot_cv_errorbars`):每图一个顶层函数,输入纯字典/数组、`out_path.parent.mkdir(parents=True, exist_ok=True)`、`fig.savefig(out_path, dpi=150)` + `plt.close(fig)`、返回 `Path`;轴标签与标题全通俗中文,docstring 末尾注明"图是可视化,数字结论以 analyze.json 为准"。缺失数据不冒充 0(:217-253 的"无数据"文字标注体例)。

---

### `src/calib/cli/main.py` analyze 命令体(CLI 编排,batch)

**类体:** 同文件 `evaluate` 命令(:1243-1792),逐段照抄骨架;替换 :1795-1797 的 `_not_implemented(7, "误差归因")` 占位(@app.command 的中文 help 已写好,保留)。

**上游定位 + 血缘登记**(:1276-1332,ANa 各块的"各找最新一轮 + 全部登记防串轮"现成写法):
```python
corners_run = latest_run_dir(OUTPUT_ROOT, "corners")   # from calib.io.manifest import latest_run_dir
...
for path, how in (
    (corners_train_npz, "calib corners(训练侧 sb 路线)"),
    ...
):
    if not path.is_file():
        raise ValueError(
            f"上游产物缺失:{path}。…请先运行 {how} 再来。")
```
analyze 的上游多一轮 evaluate:`latest_run_dir(OUTPUT_ROOT, "evaluate")` + `load_evaluate_report`(evaluation/artifacts.py:344-382,引用的 npz 缺文件自动中文报错);失败轮检查照 :1296-1306(`failure_note` 非空当场拒)。血缘 `make_lineage(config_files={...}, upstream={evaluate.json, slices.npz, crlb.json, fisher_m2.npz, noise.json, corners npz, result.json, extrinsics.npz, residuals.npz, temporal_sigma_per_corner.npz, manifest.json}, round_id=round_id)`。

**六块独立兜底的 `_stage` 模式**(:1417-1423,analyze 六块逐块套用):
```python
def _stage(name: str, run):
    try:
        return run()
    except Exception as e:  # noqa: BLE001  故意宽接:单实验失败转缺席段
        typer.echo(f"  {name}未跑成(该段记 None,不中断其余实验):"
                   f"{type(e).__name__}——{e}")
        return None
```
**外层失败契约**(:1618-1623):任何中途失败 `save_analyze_report(out_dir, make_failure_analyze(lineage, note), lineage)` 后 `raise typer.Exit(code=1) from e`。

**组装主契约**(:1625-1656):`partial_sections` 列缺席块;meta 记各上游轮次名 + profile + duration_s;六块明细键按 section 有无挂 `report["_xxx_detail"]`。

**图失败不拦主产物**(:1669-1691,逐图 try/except 的现成文案):
```python
try:
    plot_xxx(...)
    n_figs += 1
except Exception as e:  # noqa: BLE001  故意宽接:图是附加产物
    typer.echo(f"  XX图未生成(不影响数据产物):{type(e).__name__}——{e}")
```

**人工核查四要素收尾播报**(:1762-1789,memory 硬性规则的现成格式——"看哪个文件 / 文件作用 / 具体看哪里 / 好坏判据"四行 typer.echo,好坏判据里把 ANA-01 的口径与告警线数值讲清;analyze 照此写自己的四要素)。

**quick/full 档位播报**(:1345-1361):`profile_note` 文案 + `typer.echo` 上游轮次名;COR-02 的 quick 档(抽样帧、单因素栅格)沿同一开关 `cfg.analyze.profile`。

---

### `src/calib/io/config.py` + `config/calibration.yaml`(配置)

**类体:** `EvaluateConfig`(:273-369)+ `_parse_evaluate`(:816-1001)+ yaml evaluate: 段(:185-290)。

**dataclass 体例**(:273-284):`@dataclass(frozen=True)`,docstring 开头"白话背景"段讲清这步回答什么问题,再加"契约承诺"段(键集合一次定死,改契约必须回本文件与 yaml 同步改);每个字段带注释写默认值的依据出处。

**解析器体例**(:816-828 + :407 `__fail`):
```python
def _parse_analyze(raw: dict) -> AnalyzeConfig:
    cfg = AnalyzeConfig()
    section = _check_is_dict(raw.get("analyze", {}), "analyze")
    ...
    value = section.get("xxx", cfg.xxx)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        _fail("analyze.xxx", value, "必须是 >= 1 的整数(次数/数量配置)")
```
要点:布尔键先 `isinstance bool` 再拒其余(整数 1/0 是 yaml 手写最常见笔误,:825);`_fail` 抛中文 ValueError;枚举走白名单(`_ALLOWED_MODEL_IDS` 同款)。`CalibrationConfig`(:390-404)加一行 `analyze: AnalyzeConfig = field(default_factory=AnalyzeConfig)`。

**yaml 段体例**(yaml :185-290):每键带通俗中文注释 + 默认值依据(引用 docs/findings.md 结论编号或压测数据);档位 quick/full 两档一套流程的说明文案照 :270-285 体例。

**命令行零算法参数(D-12)**:analyze 命令签名 `def analyze() -> None:` 无参数,全部从 config 读——照 evaluate(:1271-1273)`board = load_board(BOARD_PATH); cfg = load_calibration(CALIBRATION_PATH)`。

---

### tests/test_analysis_*.py(8 个新测试,test)

**类体:** `tests/test_evaluation_experiments.py`(1048 行,项目最大的实验测试)。

**合成夹具三件套**(conftest.py 已有,零新基建):
- `make_chessboard_image`(conftest.py:74-96)——合成棋盘投影图工厂,**自带 `degrade` 参数(None/blur/noise/overexpose)**,COR-02 单测的直接类体,返回 (图像 uint8, 真值角点, 方格像素边长);
- `make_true_views`(conftest.py:176-204)——真值视角工厂(固定 K、固定深度),ANA-02/03 的反解恒等式测试用。

**夹具组装体例**(test_evaluation_experiments.py:87-112 `cv_setup`):
```python
@pytest.fixture
def cv_setup(make_true_views):
    views, _k = make_true_views(n_views=12)
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    obs = ObservationSet(image_ids=..., pose_ids=..., object_points=build_object_points(board),
                         image_points=np.asarray(views, dtype=np.float64), image_size=(640, 480),
                         corner_source=("sb",) * 12)
    base = CalibrationConfig()
    cfg = replace(base, evaluate=replace(base.evaluate, ..., evaluate_max_workers=1))
    return {"obs": obs, "board": board, "cfg": cfg, "optimizer_config": ...}
```
关键纪律:`evaluate_max_workers=1`(串行回退)——monkeypatch 替换模块级函数才生效,注释原文在 :90-92;派生夹具套娃(`warm_setup(cv_setup)` :353-365,先解全量再当温启动)。

**逐位一致断言工具**(:56-84 `_assert_reports_equal`):dict/list/ndarray/float 分派,浮点 `equal_nan=True`;analyze 失败契约与同种子重跑测试直接复用(从 test_evaluation_experiments import 或复制)。

**中文报错断言体例**(:383-385):
```python
with pytest.raises(ValueError) as exc_info:
    ...
assert "温启动" in str(exc_info.value)
```

**契约键全断言体例**(:574 起的 slices 测试 + :125-138):跑通后逐键断言 `report["schema"]`、键集合、明细数组形状;失败契约测试照 :959(test_delivery_model_failure_contract)——monkeypatch 令某块抛错,断言该段 None + 其余段照常 + failure_note 非空。

**monkeypatch 体例**(:393-400):`monkeypatch.setattr(experiments_mod, "solve_with_opencv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应调用")))`——负向断言"绝不该走的路"。

**测试文件规划**(RESEARCH Validation Architecture):`test_analysis_budget.py` / `test_analysis_residual.py` / `test_analysis_test_xyz.py`(反解恒等式:合成 A、已知 dX 生成残差 → 反解恢复 dX)/ `test_analysis_sensitivity.py` / `test_analysis_corner_opt.py`(合成棋盘图、串行小样冒烟)/ `test_analysis_anisotropy.py` / `test_analysis_cli.py`(tmp_path 造假上游产物树)/ `test_analysis_realdata.py`(integration 标记)。

## Shared Patterns

### 1. 中文失败契约(失败也落盘)
**来源:** `src/calib/evaluation/artifacts.py:56-89` + `src/calib/cli/main.py:1618-1623`
**适用于:** analysis/artifacts.py、cli analyze 命令体
结构键一个不少、数值字段 null、failure_note 非空中文;外层 except 落失败契约后退出码 1。"失败只是数值还没算出来,不是换一种结构"。

### 2. 血缘指纹与上游登记
**来源:** `src/calib/util/lineage.py:48-73` `make_lineage` + `cli/main.py:1276-1332`
**适用于:** cli analyze 命令体(全部上游轮次:evaluate/corners/noise/calibration/fisher 各找最新 + 逐个登记)
`make_lineage(config_files={"board.yaml":…, "calibration.yaml":…}, upstream={名: 路径}, round_id=round_id)`;主 json 与每份 npz 的 .meta.json 带同一份血缘。

### 3. npz 读写防线
**来源:** `evaluation/artifacts.py:184, 377-380`
**适用于:** 全部读回 slices/fisher/bootstrap/temporal npz 的块(ANA-01~04、COR-03)
`np.load(path, allow_pickle=False)` 写死,永不省略;写侧 `np.savez` 只收纯数值数组 + 定长 unicode 编号列。

### 4. 逐块独立兜底 + 图失败不拦主产物
**来源:** `cli/main.py:1417-1423`(`_stage`)+ `:1669-1691`(逐图 try/except)
**适用于:** cli analyze 命令体六块 + analyze_figs 的每张图
单块/单图失败只播报中文转 None,其余照常;`# noqa: BLE001  故意宽接` 注释体例随行。

### 5. CRLB 口径铁律
**来源:** `experiments.py:842-876`(σ 读取)+ `:1169-1230`(像素空间对账)+ `selection.py:280-393`(告警线)
**适用于:** budget.py(ANA-01 硬报错必须与 evaluate 同一条判定线)
唯一权威口径 = 检测器级 σ × 内参块下界 × 雅可比推到逐角点重投影位移;σ 只从 noise.json 的 `sigma_detector_corner` 读;复核与 evaluate 已落账统计量不一致当场中文报错;k1–k3 比值列 null + `crlb_caliber_note` 声明。

### 6. 检测唯一入口 + 窗口断言
**来源:** `corners/detect.py:143, 202` + `corners/window.py:42-59`
**适用于:** corner_opt.py(COR-02)
检测只调 `detect_classic`/`detect_sb`;SB 结果绝不进 `_subpix_refine`;classic 窗口档按每帧 `estimate_sq_px` 过滤 `2w+1 < sq_px`,超档不进候选不报错。

### 7. 种子确定性 + 并行异常信封
**来源:** `noise/injection.py:37-47`(`_seed_for`)+ `evaluation/parallel.py:31-109`(`TaskFailed`/`run_parallel`)
**适用于:** corner_opt.py(COR-02 网格)
种子四元组 sha1 推导;并行走 run_parallel(loky + inner_max_num_threads),worker 内补 `cv2.setNumThreads`;max_workers<=1 串行回退与并行同序同果。

### 8. 同一把尺(残差与评分)
**来源:** `solve/scoring.py:56-107` + `models/base.py:65`
**适用于:** residual_structure.py、test_xyz.py
残差 = project − 实测,只经 `compute_result_metrics`/`solve_pose_only`/`model.project`;导数只从 `project_point_jacobian` 来,analysis/ 不出现任何导数公式。

### 9. 可视化三件套
**来源:** `visualization/evaluate_figs.py:14-30`
**适用于:** analyze_figs.py
Agg 先于 pyplot、中文字体清单、unicode_minus,模块头原样照抄;库函数只管画图,不依赖 typer。

### 10. 配置校验与 D-12
**来源:** `io/config.py:407`(`_fail`)+ `:816-828` + yaml evaluate 段体例
**适用于:** AnalyzeConfig、_parse_analyze、yaml analyze 段、cli analyze 命令签名(零算法参数)

## No Analog Found

无。全部新文件在本仓库找到同角色类体;ANA-03(test_xyz.py)是唯一"新建数学"块,但其全部零件(投影/点雅可比/只估位姿求解/transform_to_camera)都是已测试锁定的既有接口,RESEARCH 已给足核心反解代码,planner 按 role-match 类体(solve/scoring.py)组织文件结构即可。

## Metadata

**Analog search scope:** src/calib/{analysis,corners,evaluation,io,models,noise,solve,util,visualization}、tests/、config/
**Files scanned:** 约 30 个源文件 + 3 个测试/配置文件;深读 16 个
**Pattern extraction date:** 2026-10-21
**消费面键名契约核实:** RESEARCH.md 已逐键核实(2026-10-21,evaluate run_20261021_070436 / fisher run_20261019_025859 / calibration run_20261018_163713 / noise run_20261018_165412),本文件不重复展开,直接引用 RESEARCH §Pattern 1-6 的键名与形状
