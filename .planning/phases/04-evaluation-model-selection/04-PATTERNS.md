# Phase 4: 评估实验 + 模型选型 + 优化器对比 - Pattern Map

**Mapped:** 2026-10-19
**Files analyzed:** 13(新建 9 + 修改 4)
**Analogs found:** 13 / 13(全部有同仓库直接参照,无"无参照"文件)

本阶段没有一处需要发明新体例:编排抄 Phase 3 的 `run_fisher_matrix`,落盘抄 `noise/artifacts.py`,并行抄 `noise/injection.py`,网格抄 `solve/matrix.py`,配置抄 `FisherConfig`,CLI 命令体例就在同一文件里(`fisher()` 命令 999–1164 行)。

## File Classification

| 新建/修改文件 | 角色 | 数据流 | 最近参照 | 匹配质量 |
|---------------|------|--------|----------|----------|
| `src/calib/evaluation/splitting.py` | utility(纯函数) | transform(姿态清单 → 折分配/掩码) | `src/calib/solve/observation.py` + `noise/fisher.py` aggregate_observations | 精确(分组逻辑同款) |
| `src/calib/evaluation/experiments.py` | service(实验编排) | batch(约 1200 次求解任务) | `src/calib/noise/fisher.py` run_fisher_matrix + `noise/injection.py` run_injection_experiment | 精确(编排+并行双参照) |
| `src/calib/evaluation/optimizer.py` | service(网格编排) | batch(16 组合 + 多起点) | `src/calib/solve/matrix.py` run_calibration_matrix | 精确 |
| `src/calib/evaluation/selection.py` | utility(纯统计) | transform(误差表 → 判行表) | `solve/scoring.py` 的阈值三分支判定 + `noise/fisher.py` spectrum_report 分档;scipy.stats 用法无先例,按 04-RESEARCH Pattern 3/4/7 | 角色匹配 |
| `src/calib/evaluation/artifacts.py` | artifact 契约 | file-I/O(json+npz 落盘) | `src/calib/noise/artifacts.py` | 精确 |
| `src/calib/visualization/evaluate_figs.py` | component(出图) | file-I/O(PNG) | `src/calib/visualization/noise_figs.py` | 精确 |
| `tests/test_evaluation_split.py` / `test_selection_stats.py` / `test_evaluation_experiments.py` / `test_evaluation_smoke.py` | test | — | `tests/test_noise_injection.py` + `tests/conftest.py` 合成夹具 | 精确 |
| `src/calib/cli/main.py`(evaluate 空壳填充) | controller | request-response(CLI 命令) | 同文件 `fisher()` 命令 999–1164 行 | 精确(同文件) |
| `src/calib/io/config.py`(EvaluateConfig + 解析) | config | — | 同文件 `FisherConfig` 240–266 行 + `_parse_fisher` 652–709 行 | 精确(同文件) |
| `config/calibration.yaml`(新增 evaluate 段) | config | — | 同文件 `fisher:` 段 151–177 行 | 精确(同文件) |
| `pyproject.toml`(加 joblib) | config | — | 同文件 dependencies 段 7–17 行 | 精确(同文件) |
| `src/calib/evaluation/__init__.py`(docstring 扩充) | package | — | `src/calib/noise/__init__.py` | 精确 |

---

## Pattern Assignments

### `src/calib/evaluation/splitting.py`(utility,transform)

**参照:** `src/calib/solve/observation.py`(pose_ids 连接键与中文断言体例)+ `src/calib/noise/fisher.py`(按姿态分组 groups.setdefault 同款写法)

**切分连接键已就位**(`solve/observation.py` 38–50 行):
```python
@dataclass(frozen=True)
class ObservationSet:
    """一次标定任务的观测集(装载产物,不可变)。

    五个长度为 V 的字段逐行对应同一张照片,行序契约由它们共同承载。
    """
    image_ids: tuple[str, ...]      # (V,) 照片编号,对齐键:D-04 对数据清单、落盘产物都用它逐行对齐
    pose_ids: tuple[str, ...]       # (V,) 姿态编号,阶段四按姿态切分(防数据泄漏)的连接键,现在就留
```
切分函数只操作 `obs.pose_ids`,按布尔掩码选行——**禁止**直接对 640 行图像随机划分(04-RESEARCH Anti-Patterns 第 1 条)。

**按姿态分组的既有写法**(`noise/fisher.py` 379–382 行):
```python
    # 按姿态分组(同姿态连拍;保持清单里首次出现的顺序,读产物的人好对照)
    groups: dict[str, list[int]] = {}
    for k, rec in enumerate(records):
        groups.setdefault(str(rec["pose_id"]), []).append(k)
```

**零重叠断言必须写进函数体**(新写,体例照 `observation.py` 117–122 行的中文报错格式):
```python
    raise AssertionError(
        f"角点文件的照片编号与数据清单(split={split})对不上:{mismatch}。"
        f"行序契约要求两者逐行一致(观测集顺序 = 求解结果顺序 = 落盘顺序),"
        f"这两份产物多半不是同一轮的:请重跑 calib corners,"
        f"或改用与角点文件同一轮 ingest 生成的 manifest.json。"
    )
```

**要点:** 分折/bootstrap 重采样/切片/留一全部是"姿态 ID 列表 → 掩码/分配表"的纯函数;种子只从 `cfg.seed`(全局 42)派生,折分配表落盘可复查(D-03);体例上 docstring 用通俗中文、术语首现加白话解释(照 `observation.py` 模块头 1–20 行)。

---

### `src/calib/evaluation/experiments.py`(service,batch 编排)

**参照:** `src/calib/noise/fisher.py`(编排骨架)+ `src/calib/noise/injection.py`(并行与任务打包)

**编排骨架**(`noise/fisher.py` 663–686 行函数签名 + 709–813 行循环体):
```python
def run_fisher_matrix(corners_npz_path: Path | str, manifest_path: Path | str,
                      records: list[dict], noise_report: dict,
                      calibration_run_dir: Path | str, board: BoardSpec,
                      cfg, exclude_images: tuple[str, ...] = (),
                      progress=None) -> dict:
    """16 份 CRLB 的完整编排:双观测集 × 双 σ 口径 × 逐模型,逐组合落盘不中断。
    ...
    """
    # 库层不得依赖命令行框架(typer):进度回调由 CLI 注入(共享模式 7)
    from calib.models import create_model
```
四个实验函数照此形态:`run_<experiment>(obs, calibrate_fn, cfg, out_dir, progress, lineage) -> dict`,返回纯字典报告(不自己落盘 json,交 artifacts.py / CLI)。

**单组合失败转契约、绝不中断**(`noise/fisher.py` 804–808 行):
```python
                    except Exception as e:  # noqa: BLE001  单组合任何异常转失败契约
                        record = _failure_record(obs_name, sigma_name,
                                                 model.param_names, e)
                        n_failed += 1
```
失败记录结构键全、数值 None、中文说明(`noise/fisher.py` 493–516 行 `_failure_record`;`solve/result.py` 81–111 行 `make_failure` 是同一体例的源头)。m3/m4 在小训练集不收敛是数据不是事故(04-RESEARCH Pitfall 3)。

**并行与任务打包**(`noise/injection.py` 241–262 行,joblib 版按 04-RESEARCH Pattern 5 替换池子,串行回退分支照抄):
```python
    if noise_cfg.injection_max_workers <= 1:
        # 串行:不开子进程(小样与调试用),结果与并行完全一致(种子只看三元组)
        results = {}
        for done, task in enumerate(dispatch, start=1):
            image_id = task[0]
            try:
                results[image_id] = _inject_one_image(task)
            except Exception as e:  # noqa: BLE001  # 故意宽接:单张失败不拖垮整批
                worker_errors[image_id] = f"{type(e).__name__}:{e}"
            if progress:
                progress({"done": done, "total": len(dispatch), "image_id": image_id})
    else:
        with ProcessPoolExecutor(max_workers=noise_cfg.injection_max_workers) as pool:
            futures = {pool.submit(_inject_one_image, t): t[0] for t in dispatch}
```
worker 的入参必须是可整读的打包元组(`injection.py` 106–114 行 `_inject_one_image(task: tuple)` 的解包写法);max_workers 从 config 读(`injection_max_workers: 8` 体例,新键建议 `evaluate_max_workers`)。

**确定性种子派生**(D-03/D-15 的复现正当性来源,`noise/injection.py` 37–47 行):
```python
def _seed_for(image_id: str, level_value: float, rep: int, base_seed: int) -> int:
    """从 (图像, σ 档, 重复) 三元组推导该副本的随机种子。

    公式:base_seed + sha1(f"{image_id}|{level}|{rep}") mod 2**32——
    确定性散列,任何一元不同种子就不同;同三元组永远同种子。
    """
    digest = hashlib.sha1(
        f"{image_id}|{level_value!r}|{rep}".encode()).hexdigest()
    return (base_seed + int(digest, 16)) % 2**32
```
折分配/bootstrap 重采样/多起点扰动全部照此派生(三元组换成 折轮次/姿态/起点序号)。

**统一求解路径的胶水**(既有积木,`solve/matrix.py` 211–213 行是唯一正确接法):
```python
        init = _load_init_result(init_dir, model.model_id) if init_dir else None
        result = solve_with_scipy(obs, model, scipy_cfg, init=init, loss=loss,
                                  rms_linear_ref=rms_linear_ref)
```
关键契约:折内每次精化必须带 `rms_linear_ref`(linear 核基准 RMS),否则 `is_practically_converged`(`solve/scipy_solver.py` 57–78 行)判定失真——04-RESEARCH Anti-Patterns 第 5 条。签名参考 `solve/scipy_solver.py` 155–160 行(`loss: str = "linear"`、`f_scale` 按 cfg 的 auto 档)与 `solve/opencv.py` 57 行 `solve_with_opencv(obs, model)`。

**验证侧评分**(既有函数,零新数学,`solve/scoring.py` 78–103 行 `solve_pose_only`,返回 `(rvec, tvec, rms)`;训练侧拟合误差用 55–75 行 `score_views`)。**验证姿态的外参绝不参与训练**,一律 solve_pose_only 评(04-RESEARCH Pitfall 6)。

---

### `src/calib/evaluation/optimizer.py`(service,batch 网格)

**参照:** `src/calib/solve/matrix.py`(16 组合网格的编排原型)

**网格循环 + 组合目录名**(`solve/matrix.py` 57–68、282–307 行):
```python
def combo_dir_name(model_id: str, solver: str, loss: str | None = None) -> str:
    """组合目录名:m2__opencv;m2__scipy(linear 主线)与 m2__scipy_softl1(其余损失)。
    ...
    命名规则必须一处定死、全链只用本函数。
    """
    if solver == "opencv":
        return f"{model_id}__opencv"
    if loss == "linear":
        return f"{model_id}__scipy"
    return f"{model_id}__scipy_{(loss or 'linear').replace('_', '')}"
```
优化器对比直接复用 `combo_dir_name`(不要自创命名);逐模型先跑 linear 再跑鲁棒核、linear 的 RMS 作后续核的 `rms_linear_ref` 参照(`solve/matrix.py` 279–307 行 `linear_rms` 字典写法)。

**统一初始化(D-14 锁死)**:全部 16 组合从同一 OpenCV 解出发——初值从落盘产物读回,不做内存传递(`solve/matrix.py` 104–141 行 `_load_init_result`:result.json 收敛检查 + extrinsics 行序核对)。每组合记录初始 RMS 与最终 RMS,"提升"才可归因到核(Pitfall 8)。

**汇总行键一个不少**(`solve/matrix.py` 71–87 行 `_combo_row`:name/model/solver/loss/n_views/rms_overall/converged/failure_note/solver_status;失败组合照样占一行)。

---

### `src/calib/evaluation/selection.py`(utility,纯统计)

**参照:** 阈值判定的仓库体例 + 04-RESEARCH Pattern 3/4/7(统计公式无仓库先例,按研究报告实现)

**三分支判定的仓库同款先例**(`solve/scoring.py` 332–355 行,存疑组 vs 正常组的走向判定——"不显著 ≠ 等价"必须写成三分支,不许只用 p 值):
```python
        if deviation >= threshold and med_s > med_n:
            verdict = "worse"
            conclusion = (
                f"结论(走向 A):存疑组明显更差——存疑 {stats_suspect['n_valid']} 张"
                f"有效帧的误差中位数 {med_s:.3f}px 是正常组({stats_normal['n_valid']} 张,"
                f"中位数 {med_n:.3f}px)的 {ratio:.2f} 倍(偏差 "
                f"{deviation * 100:.1f}%,达到 {threshold * 100:.0f}% 阈值)。")
        elif deviation >= threshold:
            verdict = "better"
            ...
        else:
            verdict = "similar"
```
配对检验按 04-RESEARCH Pattern 3:`scipy.stats.ttest_rel(diff).confidence_interval(0.95)` 与等价带 ±δ 比较出三分支(实际等价→取简 / 区间太宽→BIC 裁决 / 真实差异→held-out 小者胜);正态性用 `shapiro`,不过改 `wilcoxon` 兜底;所用方法记进产物 meta。

**分组统计的既有体例**(`solve/scoring.py` 162–173 行 `_group_stats`:无效值不计入、None 占位)。

**分档判读的既有体例**(`noise/fisher.py` 305–310 行 spectrum_report 的 flag 三档 + 648–659 行 verdict/verdict_note 写法:判定式与阈值随结论落盘,"不凭空下结论"):
```python
    if cc["status"] == "ok" and threshold is not None:
        section["verdict"] = ("pass" if cc["compare"]["max_rel_dev"] < threshold
                              else "fail")
        section["verdict_note"] = (
            f"硬门槛:max_rel_dev < {threshold:.0%}(个位数百分比留裕量的执行位)"
            f"——实际 {cc['compare']['max_rel_dev']:.3%},判 {section['verdict']}。")
```
判行表照此:每判据记数值 + PASS/FAIL + 判定式说明,不搞加权打分(D-12)。

**MAD 离群判据**:用 `scipy.stats.median_abs_deviation(x, scale='normal')`(自带 1.4826 因子,不手写);阈值 3.5 与既有模糊度判据同源(`config/calibration.yaml` 27–29 行 `blur_z_threshold: 3.5` 注释体例),config 化。

**卡方告警线**(D-06):`scipy.stats.chi2.ppf(alpha, df=7)`,0.42/0.56 两线经 venv 实测(04-RESEARCH Pattern 7);触发时走"硬告警 + 判行表 FAIL",不静默通过。

---

### `src/calib/evaluation/artifacts.py`(artifact 契约)

**参照:** `src/calib/noise/artifacts.py`(整文件照抄体例,精确匹配)

**契约键清单 + 写前校验**(`noise/artifacts.py` 28–37、70–76、252–259 行):
```python
# noise.json 顶层契约键(下游与集成测试按这套键取数;失败时也一个不少)
REPORT_KEYS = ("schema_version", "created", "lineage", ...)

def _check_contract(report: dict) -> None:
    """写盘前先验契约:顶层结构键一个不少,缺键当场中文报错。"""
    missing = [key for key in REPORT_KEYS if key not in report]
    if missing:
        raise AssertionError(
            f"noise.json 契约键缺失:{missing}。顶层必须有这些键({len(REPORT_KEYS)}"
            f" 个,失败时也一样),请用 make_failure_report / 命令体组装完整的报告。")
```

**失败契约工厂**(`noise/artifacts.py` 40–67 行 `make_failure_report`:非空中文说明强制、结构键全、数值 None——evaluate.json 照此造 `make_failure_evaluate`)。

**json 主契约 + npz 明细旁车**(核心体例,`noise/artifacts.py` 236–288 行 `save_crlb_report`):
```python
    for model_id, model_block in models.items():
        detail = model_block.pop("_fisher_detail", None)   # 私有明细键:写盘时取出转 npz,绝不进 json
        ...
        np.savez(npz_path, **{
            key: np.asarray(arr) for key, arr in detail.items()})
        # 旁车 meta:血缘必带,中文 meta 原样可读
        meta = {
            "file": npz_name,
            "model_id": model_id,
            "n_arrays": len(detail),
            "lineage": lineage,
        }
        ...
        # json 里只留引用(汇总只是索引不是替代,D-13)
        model_block["detail_npz"] = npz_name
```

**numpy 类型转 json**(`noise/artifacts.py` 216–233 行 `_to_jsonable`:ndarray→嵌套列表、标量→原生、inf 原样通过)——evaluate.json 里 bootstrap 区间/逐折误差全走它。

**读回**(`noise/artifacts.py` 291–314 行 `load_crlb_report`):`np.load(..., allow_pickle=False)` 一律强制(全部读回路径的既定防线);主 json 缺失直接抛错不静默。

---

### `src/calib/visualization/evaluate_figs.py`(component,file-I/O)

**参照:** `src/calib/visualization/noise_figs.py`(整文件体例,精确匹配)

**模块头三件套**(`noise_figs.py` 12–23 行):
```python
matplotlib.use("Agg")  # 离线后端先于 pyplot 就位,导入顺序是刻意的
import matplotlib.pyplot as plt
...
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", ...]
plt.rcParams["axes.unicode_minus"] = False
```
函数形态照 `plot_spectrum`(73–128 行):`plot_<thing>(data: dict, out_path: Path | str) -> Path`,开头 `out_path.parent.mkdir(parents=True, exist_ok=True)`,结尾 `fig.savefig(out_path, dpi=150); plt.close(fig); return out_path`;轴标签用通俗中文;图是数据的可视化,数字结论以 json 为准。CV 误差棒图参考 `plot_transfer_curve` 的 errorbar 写法(50–55 行)。

---

### `src/calib/cli/main.py` — `evaluate` 命令(controller)

**参照:** 同文件 `fisher()` 命令 999–1164 行(上游取数 → 血缘 → 编排调用 → 失败也落盘 → 中文汇总表),待填空壳在 1167–1169 行:
```python
@app.command(help="⑥ 防作弊考试:用没见过的新题考试、换着分组复测、逐一抽查,用数据挑出最合适的相机模型")
def evaluate() -> None:
    _not_implemented(6, "防作弊考试")
```

**上游取数与防串轮**(`cli/main.py` 1016–1026 行):
```python
    corners_run = latest_run_dir(OUTPUT_ROOT, "corners")
    manifest_path = _manifest_path_for(corners_run, "sb")
    ...
    noise_run = latest_run_dir(OUTPUT_ROOT, "noise")
    noise_report = load_noise_report(noise_run)["report"]
    calib_run = latest_run_dir(OUTPUT_ROOT, "calibration")
```

**血缘组装**(`cli/main.py` 1031–1050 行:upstream 逐个登记 corners npz / manifest / noise.json / 各模型 scipy result+extrinsics;evaluate 再加 crlb.json):`make_lineage(config_files=..., upstream=..., round_id=...)`(`util/lineage.py` 48–73 行)。

**中途失败也落盘失败契约再退出**(`cli/main.py` 1077–1082 行):
```python
    except Exception as e:  # 故意宽接:失败契约,失败也落盘(T-03-05-04 同体例)
        note = f"CRLB 阶段中途失败:{type(e).__name__}——{e}"
        save_crlb_report(out_dir, make_failure_crlb(lineage, note), lineage)
        typer.echo(f"算 CRLB 失败(失败报告已照常落盘,数值字段为 null):{note}")
        typer.echo(f"  失败报告:{out_dir / 'crlb.json'}")
        raise typer.Exit(code=1) from e
```

**进度回调由 CLI 注入**(库层不依赖 typer,`cli/main.py` 1058–1061 行):
```python
    def _report(row: dict) -> None:
        flag_text = row["flag"] or "正常"
        typer.echo(f"  完成 {row['model']} × {row['obs_set']} × {row['sigma']}"
                   f" 口径:{row['status']}(分档:{flag_text})")
```

**中文汇总表体例**(`cli/main.py` 1104–1123 行:对齐列 + 失败行照占 + 白话注解段 1125–1129 行)。产物目录:`OUTPUT_ROOT / "evaluate" / round_id`(docs/pipeline.md 考试阶段契约)。附加产物(图)失败不拦主产物(`cli/main.py` 1092–1100 行 try/except 体例)。CLI 零算法参数,一切从 config 读(D-12,`cli/main.py` 8–9 行硬性规则)。

---

### `src/calib/io/config.py` — `EvaluateConfig` + `_parse_evaluate`(config)

**参照:** 同文件 `FisherConfig`(240–266 行)+ `_parse_fisher`(652–709 行),精确匹配

**配置类体例**(`config.py` 239–266 行):`@dataclass(frozen=True)`、类 docstring 通俗中文白话背景、每个字段带中文行注(阈值写明来历与合法范围)。

**解析校验体例**(`config.py` 652–709 行):
```python
    cfg = FisherConfig()
    section = _check_is_dict(raw.get("fisher", {}), "fisher")

    models = section.get("models", cfg.models)
    if (not isinstance(models, (list, tuple)) or not models
            or not all(isinstance(x, str) for x in models)):
        _fail("fisher.models", models, "必须是非空的模型短名名单(如 [m1, m2, m3, m4])")
    bad_models = [x for x in models if x not in _ALLOWED_MODEL_IDS]
    ...
    # 注意顺序:先判 bool 再拒其余(整数 1/0 是 yaml 手写时最常见的笔误)
    lin_check = section.get("linearization_check", cfg.linearization_check)
    if not isinstance(lin_check, bool):
        _fail(...)
```
要点:默认值从 `cfg = EvaluateConfig()` 取;模型短名走 `_ALLOWED_MODEL_IDS` 白名单;数值区间/有序约束(如等价带 > 0、MAD 阈值 > 0、worker 数 ≥ 1、多起点 N ≥ 1)逐项校验;`load_calibration` 782–783 行处追加 `evaluate=_parse_evaluate(top)`;`CalibrationConfig`(288–300 行)加 `evaluate` 字段。

---

### `config/calibration.yaml` — 新增 `evaluate:` 段(config)

**参照:** 同文件 `fisher:` 段 151–177 行(白话标题注释 + 每键中文注 + 写法注意)。

**种子来源既有约定**(`calibration.yaml` 15–17 行):
```yaml
# ---- 全局 ----
seed: 42                      # 随机种子(交叉验证切分、bootstrap 等),保证同配置可复现
```
evaluate 段需覆盖:D-01 折数/重复轮、D-04 bootstrap 次数、D-08 LOPO 开关、`evaluate_max_workers`(照 89–93 行 `injection_max_workers: 8` 的注释体例:1 = 串行调试、上限受核数约束)、D-09 等价带 `tie_band_px`、D-10 BIC 开关与 n 口径、MAD 离群阈值 3.5、D-15 多起点 N 与扰动幅度表、双容差(RMS/参数位移当量)、D-06 卡方告警显著性。yaml 指数必须带符号(`1.0e+4`,165 行注:1.0e4 会被当文字)。

---

### `pyproject.toml` — 加 joblib(config)

**参照:** 同文件 dependencies 段 7–17 行体例(每条依赖带中文注释说明锁定理由):
```toml
dependencies = [
    # OpenCV 锁定 4.x 大版本:5.0(2026-08)刚发布,calib3d 行为易变,
    # 与"可审计、可复现的数值结果"冲突;等 5.x 攒够补丁版再评估迁移
    "opencv-python>=4.14,<5.0",
```
新增 `"joblib>=1.4"`(带注释:约 1300 次求解的进程并行,loky 常驻 worker;锁 >=1.4 因 return_as="generator" 自 1.3 引入)。装法 `uv add joblib`(04-RESEARCH Standard Stack)。

---

### `tests/test_evaluation_split.py` / `test_selection_stats.py` / `test_evaluation_experiments.py` / `test_evaluation_smoke.py`(test)

**参照:** `tests/test_noise_injection.py`(小样合成 + 配置直构造体例)+ `tests/conftest.py`(合成棋盘夹具)

**小样测试的地基**(`test_noise_injection.py` 97–117 行):
```python
def _small_noise_cfg() -> NoiseConfig:
    """小样配置:1 档 σ=4 × 1 次重复、串行(不开子进程)、网格写死保确定性。"""
    return NoiseConfig(injection_n_levels=1, injection_sigma_min=4.0,
                       injection_sigma_max=4.0, injection_n_repeats=1,
                       injection_max_workers=1)


@pytest.fixture
def small_experiment(make_chessboard_image, tmp_path):
    """小样注入地基:2 张合成棋盘图 + 基线角点 npz + 小样 NoiseConfig。"""
```
要点:配置对象直接构造(绕过 yaml 解析,秒级);确定性断言写法"同种子重跑逐位一致"(`test_noise_injection.py` 139–140 行);私有函数也直接测(`_seed_for`、`_anchor_grid` 进 import,30–35 行)——切分函数的零重叠断言、卡方线数学、BIC 已知值检验照此直测。

**合成观测集夹具**:`tests/conftest.py` 173–204 行 `make_true_views(n_views)` 给出带真值 K 的角点列表;冒烟测试(6 姿态 × 2 张)可从它拼 ObservationSet(frozen dataclass 直接构造,字段契约见 `observation.py` 38–50 行),不必读真实 npz。真实数据全量跑标注 `integration` marker(`pyproject.toml` 42–46 行,默认跳过)。

**模块 docstring 体例**(`test_noise_injection.py` 1–9 行:小样证明什么 / 集成怎么跑 / 参数只从配置读)。

---

## Shared Patterns

### 1. 失败契约(结构键全、数值 None、中文说明)
**来源:** `src/calib/solve/result.py` 81–111 行 `make_failure` → `noise/fisher.py` 493–516 行 `_failure_record` → `noise/artifacts.py` 40–67 行 `make_failure_report`
**适用:** experiments / optimizer / selection / artifacts 全部新文件。
```python
def _failure_record(obs_name: str, sigma_name: str, param_names, exc: Exception) -> dict:
    """失败契约的工厂:结构键全在、数值字段一律 None、中文说明非空。"""
    note = f"CRLB 组合({obs_name} × {sigma_name} 口径)未算成:{type(exc).__name__}——{exc}"
    return { ..., "status": "failed", "failure_note": note }
```
单次求解失败(m3/m4 小训练集不收敛)转记录不中断;失败率进汇总并计入判行表备注。

### 2. json 主契约 + npz 明细旁车 + 血缘指纹
**来源:** `src/calib/noise/artifacts.py` 全文件
**适用:** artifacts.py 逐条照抄;明细数组(折分配表、逐次 bootstrap 参数向量、LOPO 逐姿态误差表)落 npz,json 只留引用;`make_lineage`(`util/lineage.py` 48–73 行)由 CLI 组装传入,进主 json 与每个 npz 旁车。npz 读回一律 `allow_pickle=False`。

### 3. 库层不依赖命令行框架
**来源:** `noise/fisher.py` 679 行 `progress=None` 参数 + `solve/matrix.py` 54 行 `ProgressFn = Callable[[dict], None]` + CLI 侧 `cli/main.py` 1058 行注入
**适用:** experiments / optimizer / selection 所有编排入口的最后一个参数都是 `progress`;typer 只出现在 `cli/main.py`(既有静态测试锁死此约束)。

### 4. 确定性随机(种子派生)
**来源:** `noise/injection.py` 37–47 行 `_seed_for`(sha1 三元组 mod 2^32)
**适用:** 分折 / bootstrap 重采样 / 多起点扰动;base_seed = `cfg.seed`(42);同种子重跑逐位一致是集成测试复用历史轮的前提。

### 5. 行序契约与中文断言
**来源:** `solve/observation.py` 9–14 行(模块头定死行序)+ 102–122 行 `_assert_row_order`
**适用:** 切分、验证评分、CRLB 对照——凡按行选子集/对齐产物处,先断言 image_ids/pose_ids 逐行一致再算;报错用 `_fail`/`AssertionError` + 中文 + 下一步动作。

### 6. 统一求解路径:OpenCV 初始化 → scipy 精化
**来源:** `solve/matrix.py` 211–213 行 + `solve/scipy_solver.py` 155–160 行
**适用:** 全部约 1300 次求解;鲁棒核组合必传 `rms_linear_ref`(linear 基准);`f_scale` 实际生效值(solver_meta["f_scale"])记进产物 meta(Pitfall 4)。初值从落盘产物读回,不做内存私递。

### 7. matplotlib Agg + 中文字体 + 附加产物不拦主流程
**来源:** `visualization/noise_figs.py` 12–23 行 + `cli/main.py` 1092–1100 行
**适用:** evaluate_figs.py;CLI 里每张图独立 try/except,失败只 echo 不影响数据产物。

---

## No Analog Found

| 文件 | 角色 | 数据流 | 原因与替代 |
|------|------|--------|-----------|
| `selection.py` 的 scipy.stats 部分(ttest_rel 置信区间 / shapiro / wilcoxon / chi2 / median_abs_deviation) | utility | transform | 仓库尚无统计检验先例——按 04-RESEARCH Pattern 3/4/7 的公式与 venv 实测阈值实现;**判定分支的呈现体例**仍照 `solve/scoring.py` 332–355 行 |
| `experiments.py` 的 joblib 调用 | service | batch | joblib 未装、无仓库先例——Parallel 生成器骨架按 04-RESEARCH Pattern 5;串行回退分支照抄 `injection.py` 241–251 行 |
| BIC(3 行公式) | utility | transform | 无先例——按 04-RESEARCH Pattern 4 手写(n=80、k=内参+6V),公式即全部,不引库 |

## Metadata

**Analog search scope:** `src/calib/{noise,solve,cli,io,visualization,util,evaluation}`、`tests/`、`config/`、`pyproject.toml`
**Files scanned:** 17 个参照文件全读(约 6500 行),另有 2 处 grep 定位(`io/config.py` 符号表、`solve/opencv.py` 签名)
**Pattern extraction date:** 2026-10-19
