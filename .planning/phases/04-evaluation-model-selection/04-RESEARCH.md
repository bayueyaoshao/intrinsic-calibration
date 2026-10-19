# Phase 4: 评估实验 + 模型选型 + 优化器对比 - Research

**Researched:** 2026-10-19
**Domain:** 姿态分组交叉验证 / 重采样统计(bootstrap、重复张分组、留一姿态)/ 模型选型统计检验 / scipy 优化器与鲁棒核对比 / joblib 进程并行
**Confidence:** HIGH(代码复用面与数据现状为本地实测;统计方法与 scipy 语义经官方文档与 venv 实测双重确认;少数扰动幅度类参数为经验起点,已进假设清单)

## Summary

本阶段是"防作弊考试"的全面落地:约 1300 次标定求解(交叉验证 100 + bootstrap 800 + 留一姿态 320 + 优化器网格 16 + 多起点约 80 + 全量重训 4),围绕四个可独立复用的确定性实验函数展开,最终产出三判据判行表与选定模型。项目既有积木覆盖度极高:`ObservationSet` 已带 `pose_ids` 字段(切分只需按姿态选行)、`solve_with_scipy` 已支持全部四种精化核、`score_views`/`solve_pose_only` 已能做训练侧拟合误差与独立照片评分、`marginalized_fisher`/`crlb_se` 可原样复用算 D-05 要求的"80 视图单张"新口径 CRLB。唯一的新依赖是 joblib(pyproject 未装,PyPI 当前版 1.6.0,slopcheck 判 OK)。

研究回答了 CONTEXT.md 留白的全部八个裁定点:(1) 配对检验用 `scipy.stats.ttest_rel` + 其置信区间方法(scipy 1.10 起提供,venv 1.18.1 实测可用),"统计平局"不看 p 值而是看差值置信区间是否落在等价带内;(2) BIC 的 n 锁定 80 有效姿态、k 写全部被估参数(内参 + 6×V 外参,各模型同加常数不影响比较);(3) f_scale 不进主对比矩阵,沿用现有 `auto`(3×1.4826×初残差 MAD,实测 0.17–0.40 px,与残差量级匹配);(4) 多起点扰动给出一套具体幅度(内参相对 ±2%、畸变相对 ±10%、旋转约 1° 旋转向量扰动)、N=10 起步;(5) 离群判据用 MAD z 分数 3.5(与既有模糊度判据同体例);(6) "实测 < CRLB" 硬告警给出精确数学:8 组经验标准差低于 0.42×σ_CRLB 即告警(每参数 1% 显著性,卡方阈值经 venv 实测);(7) joblib 用 loky 常驻进程 + `return_as="generator"` 做进度播报,worker 数 config 化;(8) 复用面逐函数列清,`calib evaluate` 空壳填充路径明确。

**Primary recommendation:** 以"姿态切分纯函数 + 实验编排纯函数 + 统计选型纯函数"三层组织 `src/calib/evaluation/`,所有实验统一走"OpenCV 初始化 → scipy 精化"既有求解路径,新增 `uv add joblib`,产物遵循 json 主契约 + npz 明细旁车 + 血缘指纹的既定体例。

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

以下 16 条决策已定案,研究不再探讨替代方案:

- **D-01:5 折 × 重复 5 轮**——80 姿态分 5 折(每折验证 16 姿态 ≈ 128 视图,训练 64 姿态 ≈ 512 视图);5 种不同随机切分重复 5 轮,每姿态恰好 5 次进验证。held-out 指标每模型 25 个折级样本,可画带误差棒的对比;总训练求解 = 4 模型 × 25 折 = 100 次。5 折的训练侧比全量少 20%,held-out 估计略偏悲观(保守方向,可接受)
- **D-02:训练用全视图 + 验证按姿态聚合**——折内训练把选中姿态的 8 连拍全部当独立视图喂入(与交付重训口径一致,选型结论直接可迁移);验证误差先把同姿态 8 张的误差平均成姿态级,再汇总折级——防止同姿态 8 张把验证计数虚增 8 倍
- **D-03:纯随机分折 + 固定种子**——用 config 既有全局 seed(42),折分配表落盘可复查;80 样本随机分足够均衡。**附带裁决**:训练/交付一律用全 640 视图,不做 poses80 聚合口径的 CV(Phase 3 遗留的"聚合策略正式裁决"就此定案——聚合观测集只用于噪声测量与 Fisher,不用于标定交付;连拍相关性损耗由重复张分组实验实测,见 D-08)
- **D-04:bootstrap 200 次 × 全 4 模型**——按姿态有放回重采样 80 个姿态;抽到的姿态带其全部 8 连拍(640 视图口径,模拟"换一批姿态");4×200 = 800 次求解,joblib 并行约 1–2 小时。参数稳定是三判据之一,每个候选模型都需要稳定性证据
- **D-05:重复张分组实验对照"新算同口径 CRLB"**——分组实验每姿态取第 i 张(i=1..8,按文件名时间戳排序)组成 8 个数据组(每组 80 姿态 × 1 张),各组独立完整标定得 8 份参数 → 经验参数不确定度(纯噪声影响)。对照组**新算**一种 CRLB:80 姿态 × 每姿态 1 张、σ 用 temporal 口径(0.30 px/轴),复用 Phase 3 Fisher 数学核心(边际化/Schur 消元:把外参从精度计算里合并掉,只留内参的精度界)。**禁止**用现有 poses80 聚合记录近似(它是 8 连拍均值口径,σ/√8 折减,与单张不可比,会造成"实测 < CRLB"假告警)
- **D-06:"实测 < CRLB"硬告警保持**——分组实验经验不确定度若低于同口径 CRLB 下界(好于理论极限,物理上不可能)触发硬告警不静默通过(ROADMAP 成功标准 3);前提是 D-05 的同口径对照
- **D-07:bootstrap 与分组实验分离两种不确定度来源**——分组实验 = 纯噪声(位姿集合固定,仅噪声不同);bootstrap = 位姿抽样 + 噪声;两者之差刻画"位姿抽样"贡献(ROADMAP 成功标准 4)
- **D-08:LOPO 用预测误差语义 × 全 4 模型**——留一姿态(Leave-One-Pose-Out,每次留 1 个姿态当验证、其余 79 个训练)80 次 × 4 模型 = 320 次求解;每姿态的 held-out 误差离群者即"问题姿态",顺带得到 LOO 泛化估计。影响型(留它训练看参数变化)不做——与 Phase 5 敏感度分析重叠;剔除收益验证留给 Phase 6 一键重标
- **D-09:统计平局取简**——两模型 held-out 比较:同折配对相减(同一折切分下两模型误差相减,消除折间变异)后,若差异不大于折间波动(配对检验不显著)判"统计平局",平局取更简模型(参数少的)
- **D-10:BIC 只作平局裁决器**——held-out 主判据;统计平局时由 BIC(贝叶斯信息准则,惩罚参数数量的模型比较量,越小越好)更小者裁决;n = 有效姿态数 = 80(成功标准写死)。BIC 不做硬门槛
- **D-11:Fisher 退化发现不预否决,靠三判据制衡**——m3/m4 照常参选全部实验;若真过拟合,会在 CV held-out 上自己显形(训练误差降、验证误差升);高阶项 CRLB = inf(数据里没有信息定住它)→ bootstrap 区间爆炸 → 参数稳定判据拦住。Fisher 预警标注进选型报告;若复杂模型胜出,必须解释其与 Fisher 发现的关系
- **D-12:三判据全 PASS 才能当选**——每个候选模型一行判行表:泛化(CV held-out)+ 稳定(bootstrap 区间 + 分组离散度)+ 逼近 CRLB(分组实测 vs D-05 同口径理论,实测低于下界即 FAIL 并触发 D-06 告警);多模型全 PASS 时 held-out 最小者胜,平局按 D-09/D-10 取简。不搞加权打分
- **D-13:优化器对比先行**——先在全量 640 角点视图上跑优化器/鲁棒核网格,每模型产出"推荐优化配置"(收敛且重投影误差不劣者的代表配置);CV/bootstrap/LOPO 及全量重训统一用该配置。避免"选型用默认配置、对比发现更优配置后 CV 结论返工"
- **D-14:对比矩阵 = 4 核 × 4 模型**——scipy 精化核 {linear, soft_l1, huber, cauchy}(鲁棒核:降低离群残差影响力的损失函数)× M1–M4;OpenCV 结果为共同初始化与基线(共 16 组合,现有产物已覆盖其中 linear 与 soft_l1)。f_scale(鲁棒核的切换阈值)用默认值起步,是否加扫描交研究定
- **D-15:多起点一致性检验 = 扰动初值 × N≈10 次**——从 OpenCV 初始化出发,对内参/外参各加随机扰动重复求解,判据 = 参数与重投影误差双容差内收敛到同一解;全 4 模型都做,m4(鱼眼)是重点观察对象(初值不当时易卡坏解);N 终值与扰动幅度交研究定
- **D-16:test 评分全模型并排**——选定模型全量重训(640 视图,已锁)后,4 个模型的最终内参都在 test 65 张独立照片上评分并排(评分只是投影计算,几乎不费时);逐张误差表全存。报告能看到选定模型 vs 其他模型的真实泛化差距,顺带验证选型结论

### Claude's Discretion(研究/规划阶段用数据和技术判断定,无需再问用户)

- D-09 配对检验的具体方法(如配对 t 检验 / Wilcoxon, n=25 折级样本的正态性判断)
- D-10 BIC 的自由度口径(残差计数:按有效观测数还是姿态数;嵌套模型比较的写法)
- D-14 f_scale 是否扫描及取值
- D-15 多起点 N 终值与扰动幅度(内参/外参各扰多少)
- 实验产物目录结构与 JSON 契约细节(遵循既有血缘体例:json 主契约 + npz 明细旁车)
- CV 曲线/判行表的可视化选型
- joblib 并行的任务粒度与 worker 数(config 化,参考 noise 注入实验的 max_workers 体例)
- "问题姿态"的离群判据(如 MAD 倍数阈值,config 化)

### Deferred Ideas (OUT OF SCOPE)

- 影响型留一姿态(留它训练看参数变化的影响力诊断)— Phase 5 敏感度分析(ANA-04)重叠,本阶段不做
- 问题姿态剔除后的一键重标与跨运行 diff — Phase 6(ITER-03)
- CV 的 views640 vs poses80 双口径对照 — 不做;连拍相关性损耗的实测对照由重复张分组实验承担(Phase 3 D-01 设计意图,D-03 已裁决训练/交付一律全视图)
- 剔除重标的回环触发条件表 — Phase 6(REP-02)

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVA-01 | 按姿态切分训练/验证(同一姿态的所有照片只进同一边,结构性防数据泄漏)+ K 折交叉验证 | `ObservationSet.pose_ids` 已就位;切分纯函数按姿态选行掩码(D-01/02/03);零重叠单元测试为成功标准 1 的硬断言 |
| EVA-02 | 模型选择只看 held-out 泛化误差(辅以 BIC、参数可辨识性);选定后全量数据重训出最终内参 | ttest_rel 置信区间平局判定(D-09)+ BIC n=80 口径(D-10)+ 三判据判行表(D-12);全量重训复用 `solve_with_scipy` |
| EVA-03 | 重复张分组实验——各组独立完整标定后对比内参差异,得到纯噪声传播的经验参数不确定度,并用于 CRLB 理论预测的实证校验 | 切片分组构造("每姿态取第 i 张")+ D-05 新算同口径 CRLB(复用 marginalized_fisher)+ D-06 卡方告警判定(阈值 0.42 经 venv 实测) |
| EVA-04 | bootstrap ×N 参数稳定性(与重复张分组实验对比,分离"噪声"与"位姿抽样"两种因素) | D-04 按姿态重采样 + D-07 两来源分离;joblib 并行 800 次求解 |
| EVA-05 | 留一姿态 ×N——问题姿态定位与解的脆弱性检查 | D-08 预测误差语义 LOPO ×4 模型;MAD z=3.5 离群判据(config 化) |
| SOL-02 | 多优化器/鲁棒核/初始化策略对比实验,验证是否按预期提升精度 | D-13 优化器先行 + D-14 四核×四模型矩阵(`solve_with_scipy` 已支持全部四核)+ D-15 多起点扰动方案 + D-16 test 65 张全模型并排(`solve_pose_only` 现成) |

</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| 姿态分组切分与折分配 | `evaluation/splitting.py`(纯函数) | `solve/observation.py` 提供行掩码 | 切分是实验逻辑;行序契约归 ObservationSet |
| 四实验函数(CV/bootstrap/切片/LOPO) | `evaluation/experiments.py`(纯函数) | CLI 注入进度回调 | ROADMAP 成功标准 1 要求"独立可复用的确定性函数";库层不依赖命令行框架(既有静态测试锁死) |
| 同口径 CRLB 新算与硬告警 | `noise/fisher.py` 数学原样复用 | `evaluation/` 编排调用 | Fisher 数学唯一来源约束;evaluation 只组装输入输出 |
| 选型统计(配对检验/BIC/判行表) | `evaluation/selection.py` | — | 纯计算,只依赖 scipy.stats 与落盘产物 |
| 优化器/鲁棒核对比与多起点 | `evaluation/optimizer.py`(编排) | `solve/scipy_solver.py` 的 loss 参数 | 求解能力已存在,evaluation 只铺网格与汇总 |
| test 65 张评分 | `solve/scoring.py` 的 `solve_pose_only` | `evaluation/` 编排 | 既有函数,零新数学 |
| 进程并行 | joblib(loky 后端) | `evaluation/` 决定任务粒度 | 新依赖;`injection.py` 的 ProcessPoolExecutor 是既有替代体例,本阶段按 CLAUDE.md 推荐栈用 joblib |
| 判行表/CV 曲线可视化 | `visualization/`(matplotlib) | — | matplotlib PNG 是既定产物形态(plotly/bokeh 排除) |
| CLI 入口与配置 | `cli/main.py` 的 `calib evaluate` 空壳 | `io/config.py` 新增 EvaluateConfig | Phase 1 D-11 预留的第六命令;参数只从 config 读(D-12 既有裁决) |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | 1.18.1(已装) | `stats.ttest_rel`(配对 t 检验,结果对象自带置信区间方法)、`stats.wilcoxon`(非正态兜底)、`stats.shapiro`(正态性判断)、`stats.median_abs_deviation`(MAD,离群判据)、`stats.chi2`(CRLB 告警阈值)、`optimize.least_squares`(四核精化,已接入) | 全部所需统计与优化能力已在栈内,**零新增统计依赖** |
| numpy | 2.5.x(已装) | 折分配表、参数向量堆叠、npz 明细落盘 | 既有 |
| joblib | 1.6.0(PyPI 当前,装 ≥1.4 即满足推荐) | 约 1320 次求解的进程并行(loky 常驻 worker + `return_as="generator"` 进度播报) | CLAUDE.md 推荐栈点名;worker 常驻避免每任务重复导入 OpenCV;`pyproject.toml` 尚无此依赖,**需 `uv add joblib`** [VERIFIED: pip index + PyPI JSON] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | 3.10.x(已装) | CV 误差棒图、判行表条形图、LOPO 逐姿态误差散点 | 判行表可视化(discretion 裁定:不引入新可视化库) |
| pandas | 2.3.x(已装) | 判行表/逐折表/逐姿态表的 CSV 导出 | 汇总表输出,不进热路径 |
| opencv-python | 4.14.x(已装,锁 <5.0) | 共同初始化基线(`solve_with_opencv`) | 既有 |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| joblib.Parallel | 既有 `concurrent.futures.ProcessPoolExecutor`(injection.py 体例) | 两者在本机都可跑;joblib 的 loky worker 常驻 + 生成器进度更省事,且 CLAUDE.md 已点名;CONTEXT 也明示"joblib 需新增依赖"。不并行回退(串行)必须保留(≤1 时) |
| ttest_rel 置信区间带判平局 | statsmodels 的 TOST(双单侧等价检验) | 需新增重量级依赖;置信区间带已够用(见 Pattern 3),不引 statsmodels |
| scipy.stats 中位数绝对偏差 | 手写 5 行 MAD | `median_abs_deviation(x, scale='normal')` 现成且自带 1.4826 因子,不手写 |

**Installation:**

```bash
uv add joblib   # 锁入 pyproject;当前解析为 1.6.0
```

**Version verification:** `pip index versions joblib` → 最新 1.6.0,requires_python ≥3.10,纯 Python wheel(`py3-none-any`,无编译扩展、无安装钩子);项目 venv 实测未装。[VERIFIED: pip index + PyPI JSON API + venv 实测]

## Package Legitimacy Audit

> 已按 Package Legitimacy Gate 协议执行:slopcheck 0.6.1 本地可用,对唯一新增包 joblib 判 **[OK]**;PyPI 元数据经 JSON API 直接核验。

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| joblib | PyPI | 10+ 年(2010 年起,scikit-learn 生态核心件) | 极高(sklearn 硬依赖) | github.com/joblib/joblib | [OK] | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                     config/calibration.yaml (seed=42 + evaluate 段)
                                      │
   上游产物(只读取数)                ▼
┌────────────────────────────────────────────────────────────────┐
│ sb 角点 npz ──► ObservationSet(pose_ids 就位)                  │
│ noise.json(σ 双口径:detector 0.0087 / temporal 0.2995 px)    │
│ m{X}__scipy linear 收敛解(优化器对比的初始化与对照)             │
└───────────────┬────────────────────────────────────────────────┘
                ▼
   ① 优化器对比(D-13 先行,全量 640 视图)
   4 scipy 核 × 4 模型 = 16 组合 + OpenCV 基线 + 多起点 N≈10
                │ 每模型产出"推荐优化配置"
                ▼
   ② 四实验函数(统一用推荐配置,joblib 并行)
   ┌──────────────┬──────────────┬──────────────┬──────────────┐
   │ CV 5折×5轮   │ bootstrap    │ 重复张分组   │ LOPO 80×4    │
   │ 100 次求解   │ 200×4=800    │ 8 组 + 新算  │ 320 次求解   │
   │              │ 次求解       │ 同口径 CRLB  │              │
   └──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┘
          ▼              ▼              ▼              ▼
   折级误差表      参数向量数组    8 份参数 +      逐姿态误差表
   (25 样本/模型)  (区间/离散度)  slice CRLB     (问题姿态点名)
          └──────────────┴──────┬───────┴──────────────┘
                                ▼
   ③ 选型统计(D-09/10/11/12)
   配对检验(平局判定)→ BIC(平局裁决)→ 三判据判行表
                                │ 判行表全 PASS 且 held-out 最小者
                                ▼
   ④ 交付(D-16)
   全量重训(640 视图)→ test 65 张 × 4 模型并排评分
                                │
                                ▼
   output/evaluate/run_<时间戳>/
   evaluate.json 主契约 + experiments/* 明细 + 血缘指纹
```

主用例可沿箭头走通:上游取数 → 优化器定配置 → 四实验并行跑 → 统计选型 → 全量重训与 test 评分 → 判行表落盘。

### Recommended Project Structure

```
src/calib/evaluation/
├── __init__.py
├── splitting.py       # 姿态分组切分:分折/重采样/切片/留一,纯函数 + 分配表落盘
├── experiments.py     # 四实验函数:cv_pose_grouped / bootstrap_pose / slices_by_index / lopo
├── optimizer.py       # 优化器×核 16 组合网格 + 多起点一致性(编排,复用 scipy_solver)
├── selection.py       # 配对检验、BIC、三判据判行表(纯计算)
└── artifacts.py       # evaluate 契约落盘(参照 noise/artifacts.py 体例)
config/calibration.yaml  # 新增 evaluate: 段(worker 数/等价带/MAD 阈值/多起点 N 与幅度)
output/evaluate/run_<ts>/ # 产物目录(docs/pipeline.md 考试阶段契约)
tests/test_evaluation_split.py / test_evaluation_experiments.py / test_selection_stats.py
```

### Pattern 1: 姿态分组切分(防泄漏的结构性保障)

**What:** 一切切分只操作姿态 ID 列表;姿态定了,该姿态全部 8 连拍同进同出;由姿态 ID 掩码生成 ObservationSet 行掩码。
**When to use:** CV 分折、bootstrap 重采样、切片分组、LOPO——本阶段全部四种实验。

```python
# 模式示意(与 ObservationSet 行序契约配合)
def split_by_pose_ids(obs: ObservationSet, train_poses: set[str]) -> tuple[ObservationSet, ObservationSet]:
    train_mask = np.array([p in train_poses for p in obs.pose_ids])
    assert train_mask.any() and (~train_mask).any(), "切分两侧都不得为空"
    assert not (set(np.asarray(obs.pose_ids)[train_mask]) & (set(obs.pose_ids) - train_poses)), \
        "姿态泄漏:同一姿态出现在训练与验证两侧"
    return obs_rows(obs, train_mask), obs_rows(obs, ~train_mask)
```

零姿态重叠的单元测试是 ROADMAP 成功标准 1 的硬断言,必须先于实验函数落地。[CITED: .planning/research/PITFALLS.md Pitfall 1]

### Pattern 2: 实验函数的统一形态(确定性 + 失败转记录)

**What:** 每个实验函数形如 `run_<experiment>(obs, calibrate_fn, cfg, out_dir, progress, lineage) -> dict`,输入输出全落盘;单次求解失败用 `make_failure` 契约记录不中断(m3/m4 在小训练集下可能不收敛,这是数据不是事故)。
**When to use:** 四实验 + 优化器网格全部照此体例;这也是集成测试"血缘指纹对得上就复用历史轮"的前提。

```python
# calibrate_fn 统一走既有路径:OpenCV 初始化 → scipy 精化(折内每次带 linear 基准 RMS,
# 供 is_practically_converged 判定——漏掉它会让实用收敛判定失真)
def calibrate_fn(train_obs, model, cfg, loss):
    init = solve_with_opencv(train_obs, model, cfg)
    return solve_with_scipy(train_obs, model, cfg, init=init.params, loss=loss,
                            rms_linear_ref=init.rms)
```

[VERIFIED: src/calib/solve/{matrix,result}.py 既有体例]

### Pattern 3: 配对检验与"统计平局"判定(D-09 裁定)

**What:** 同折配对相减得 25 个差值样本;平局判定不用 p 值,用差值的 95% 置信区间与等价带 ±δ 的关系。
**When to use:** 任意两模型 held-out 比较(25 折级样本;LOPO 的 80 姿态级样本同法)。

```python
from scipy.stats import shapiro, ttest_rel, wilcoxon

diff = err_a - err_b                      # 同折相减,消除折间变异
# 正态性:Shapiro-Wilk;n=25 时小样本,若 p_norm < 0.05 改用 Wilcoxon 符号秩检验兜底
normal = shapiro(diff).pvalue >= 0.05
ci = (ttest_rel if normal else wilcoxon_alternative_ci)(diff).confidence_interval(0.95)
# 等价带 δ = max(0.005 px, 较简模型平均误差的 2%),config 化(evaluate.tie_band_px 等)
# 三分支判定:
#   CI 整体在 [-δ, +δ] 内            → 实际等价 → 取简(D-09)
#   CI 含 0 且超出 ±δ(区间太宽)     → 统计平局 → BIC 裁决(D-10)
#   CI 不含 0 且 |均值差| > δ        → 差异真实 → held-out 小者胜
```

依据:ttest_rel 结果对象的 `confidence_interval()` 方法 scipy 1.10.0 引入,项目 venv 1.18.1 实测可用 [VERIFIED: venv 实测 + docs.scipy.org];Wilcoxon 的 `method='auto'` 在 n≤25 且无零/并列时走精确法,有零/并列时回退正态近似 [CITED: docs.scipy.org/wilcoxon + github.com/scipy/scipy issue #17530]——折级误差差值极少出现精确 0,风险低,但在产物 meta 里记下所用方法。scipy 无内置 TOST(等价性检验),置信区间带是其标准替代,不为此引 statsmodels。[ASSUMED: A1]

### Pattern 4: BIC 平局裁决(D-10 裁定)

**What:** 高斯残差假设下的 BIC = n·ln(RSS/n) + k·ln(n);n=80 有效姿态(锁定),k = 全部被估参数。
**When to use:** 仅在配对检验判"统计平局"时计算比较,写入判行表"平局裁决"列。

```python
def bic_full_fit(result, n_poses: int = 80) -> float:
    rss = float(np.sum(np.asarray(result.residuals) ** 2))   # 全量重训解的全部重投影残差
    n_params = result.n_intrinsic_params + 6 * result.n_views  # 内参 + 每视图 6 个外参
    return n_poses * np.log(rss / n_poses) + n_params * np.log(n_poses)
```

k 为什么写全外参:各模型都加同一常数 6V·ln(80),比较结果不变;但"按全部被估参数计"是教科书口径,写全可辩护(判行表注明口径即可)。n 为什么是 80:残差有 640×88×2 = 112640 个,但它们被 80 个姿态强相关;若 n 写 640,ln(n) 对复杂模型的惩罚弱化约 8 倍,系统偏向 m3/m4——Phase 3 研究(PITFALLS.md)已锁定 n=80。[CITED: .planning/research/PITFALLS.md; D-10 写死]

### Pattern 5: joblib 并行 + 进度播报(第七个 discretion 裁定)

**What:** 任务粒度 = 一次完整标定求解(秒级);loky 常驻 worker;`return_as="generator"` 逐个消费结果驱动 CLI 注入的进度回调。
**When to use:** CV 100 / bootstrap 800 / LOPO 320 / 多起点约 80 次求解的编排层。

```python
from joblib import Parallel, delayed

def run_parallel(tasks, worker, max_workers: int, progress):
    if max_workers <= 1:                       # 串行回退(调试与小样),与并行同种子同结果
        for i, t in enumerate(tasks, 1):
            yield worker(t); progress({"done": i, "total": len(tasks)})
        return
    results = Parallel(n_jobs=max_workers, return_as="generator")(
        delayed(worker)(t) for t in tasks)
    for i, r in enumerate(results, 1):
        yield r; progress({"done": i, "total": len(tasks)})
```

要点:(1) worker 的入参必须是可整读的打包元组(ObservationSet 是 frozen dataclass、纯 numpy 数组,可 pickle,单折约几百 KB,loky 序列化无压力);(2) `return_as="generator"` 自 joblib 1.3 引入,1.6.0 现役 [CITED: joblib.readthedocs.io/parallel_generator];(3) worker 数走 config(键名建议 `evaluate_max_workers`,参照 `injection_max_workers: 8` 体例,本机 20 核默认 8);(4) 单任务异常捕获转 `make_failure` 记录,与 injection.py 的"单张失败不拖垮整批"一致。

### Pattern 6: 多起点扰动(D-15 裁定)

**What:** 从 OpenCV 初始化出发加受控扰动,N=10 起步(config 化),判据是参数与重投影误差双容差内收敛到同一解。
**When to use:** 全 4 模型;m4(鱼眼)重点观察,可单独加大 N 到 20。

扰动幅度建议(经验起点,进假设清单 A3):

| 参数组 | 扰动方式 | 幅度 |
|--------|----------|------|
| fx, fy | 相对乘性 | ±2% |
| cx, cy | 相对乘性 | ±2% |
| 畸变系数(k/p 族) | 相对乘性 | ±10%(畸变项数值本身小,相对幅度要大才起作用) |
| 旋转 rvec | 随机轴 + 固定转角 | 约 1°(在旋转向量空间:单位随机方向 × 1°) |
| 平移 tvec | 相对范数 | ±2% |

种子从全局 seed=42 按 (模型, 起点序号) 确定性派生(参照 injection.py 的 sha1 三元组模式),重跑逐位一致。收敛判据建议:各起点 RMS 极差 < 0.001 px(绝对容差)+ 参数一致性用"像素位移当量"容差(把内参差换算成典型视场处的投影位移,阈值 0.1 px,config 化)。m4 若有起点卡坏解,记录进契约不静默——这正是 D-15 要观察的现象。

### Pattern 7: 同口径 CRLB 新算与硬告警(D-05/D-06 裁定)

**What:** 每个切片组 i(80 姿态 × 每姿态第 i 张)各算一份 Fisher(输入 = 该切片观测集 + temporal σ 0.2995 px/轴 + 该切片自己收敛解作线性化点),复用 `marginalized_fisher`/`crlb_se` 原样数学;8 份 SE 取均值作对照基准;分组经验标准差低于告警线即 FAIL。
**When to use:** 判行表第三判据"逼近 CRLB"的唯一合规对照(现有 crlb.json 没有 80 视图单张口径,禁止拿 poses80 聚合记录近似)。

```python
# 告警阈值:每组 80 姿态 → 8 份参数向量 → 每参数经验 std s(自由度 7)
# (n-1)s²/σ² ~ 卡方(7);s < sqrt(chi2.ppf(alpha,7)/7) * σ_CRLB 时"实测显著低于理论下界"
# alpha=0.01 → chi2.ppf(0.01,7)=1.2390 → 线 0.42(硬告警,判行表 FAIL + D-06 告警)
# alpha=0.05 → chi2.ppf(0.05,7)=2.1673 → 线 0.56(边缘提示,不拦)
# 两阈值经项目 venv scipy 1.18.1 实测确认 [VERIFIED: venv 实测]
```

背景量级:8 样本标准差的相对误差 1/√(2×7) ≈ 26.7%,与 Phase 3 的 Σ ±26.7% 带同一哲学;0.42 线在 1 个相对误差开外,假告警概率每参数约 1%,4 模型 × 8~12 参数量级下期望假告警次数远小于 1,可接受。

### Anti-Patterns to Avoid

- **按图像切分:** 任何直接对图像行随机划分的写法都会让同姿态 8 张跨边——只允许操作姿态 ID 列表。
- **验证误差按图像计数:** 同姿态 8 张的验证误差必须先平均成姿态级再汇总(D-02),否则样本量虚增 8 倍、置信区间假性收窄。
- **用 p>0.05 判平局:** 不显著 ≠ 等价(样本小则区间宽,什么都"不显著");必须用置信区间带(Pattern 3 三分支)。
- **BIC 用 n=640:** 复杂模型惩罚弱化约 8 倍,选型系统性偏向 m3/m4,直接违反 D-10。
- **折内求解不带 linear 基准:** `is_practically_converged` 需要 `rms_linear_ref`;漏传会让收敛判定退化为 status>0 与尾部平台拼凑,跨折不可比。
- **多起点只看 RMS 一致:** 参数不同但 RMS 相同 = 可辨识性问题本身,两个容差都要查(D-15"双容差")。

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 配对检验与置信区间 | 手写 t 统计量与分位数 | `scipy.stats.ttest_rel(...).confidence_interval()` | 现成且经 venv 验证;手写易错自由度 |
| 非参数兜底检验 | 手写秩统计 | `scipy.stats.wilcoxon` | 零/并列处理有讲究,库已处理 |
| MAD 与离群判据 | 手写 MAD 公式 | `scipy.stats.median_abs_deviation(x, scale='normal')` | 自带 1.4826 正态一致性因子 |
| 卡方告警阈值 | 手算/硬编码临界值 | `scipy.stats.chi2.ppf(alpha, df)` | 阈值随自由度变(重复数不齐时自由度=组数−1),代码化才可迁移(REU-01 精神) |
| BIC | 找现成库 | 3 行公式手写(Pattern 4) | 公式即全部;库(statsmodels)反而把 n 的口径藏起来 |
| 进程并行与进度 | 自己管进程池生命周期 | joblib loky + `return_as="generator"` | worker 常驻、异常传播、进度迭代都是现成的 |
| 旋转向量扰动 | 手写旋转矩阵乘法 | `cv2.Rodrigues`/既有 solve 层的 rvec 处理(随机轴×角度构造 rotvec 即可,3 行) | 项目里 rotvec 已是外参存储形态 |

**Key insight:** 本阶段唯一的"新数学"只有统计判定的组装逻辑(配对相减、区间带、BIC、卡方线),其余全部是既有积木的编排——研究结论是"不要写求解器、不要写检验器、不要写并行器,只写实验函数与判行表"。

## Common Pitfalls

### Pitfall 1: 姿态泄漏(跨边切分)
**What goes wrong:** 同姿态 8 张分进训练与验证两侧,验证误差被训练过,选型结论作废。
**Why it happens:** 直接对 640 行随机划分;`ObservationSet` 行序按图像排列,pose_id 不显眼。
**How to avoid:** 切分函数只收姿态 ID 集合;零重叠断言写进函数体 + 独立单元测试(成功标准 1);折分配表落盘可复查(D-03)。
**Warning signs:** CV 折级误差异常整齐地低于 test 评分误差。

### Pitfall 2: poses80 聚合 CRLB 冒充单张口径
**What goes wrong:** 用现有 crlb.json 的 poses80 记录对照分组实验实测,因 σ/√8 折减,理论界被人为压低 8 倍,触发大面积"实测 < CRLB"假告警。
**Why it happens:** 现成记录拿来就用;两种口径名字都带"80"。
**How to avoid:** D-05 明令禁止;新算记录带 `observation_set: "views80_single_frame"` 之类口径标识落盘,与 poses80 记录并存在 crlb.json 体例下。
**Warning signs:** 告警率接近 100% 而不是零星。

### Pitfall 3: m3/m4 在小训练集上不收敛被当成事故
**What goes wrong:** LOPO(79 姿态训练)或 CV 折内,m3(12 参数有理模型)/m4 触发求解失败,编排层抛异常中断整批。
**Why it happens:** 复杂模型在数据略少或切分略偏时线性化不稳;Fisher 侧已预警其高阶项退化。
**How to avoid:** 沿用 `make_failure` 契约:失败转结构化记录(结构键齐全、数值置 None),实验照常跑完;汇总时报告失败率并计入判行表备注(D-11:Fisher 预警标注进报告)。
**Warning signs:** 某模型失败集中在特定折/特定留出姿态。

### Pitfall 4: f_scale 的 auto 值随折浮动
**What goes wrong:** `auto` 按初残差 MAD 计算,不同折/不同重采样初值略不同,跨折比较时核的"软阈值"不一致。
**How to avoid:** 这不是 bug 而是口径:每次求解把实际生效的 f_scale 记进产物 meta(现有 `_auto_f_scale` 返回值落盘);优化器对比阶段(D-13)全量口径的 auto 值即推荐配置的一部分。
**Warning signs:** 同模型不同折 f_scale 相差数倍(说明某折初值离谱,查初始化)。

### Pitfall 5: 现有 m1__scipy_softl1 未收敛的正确处置
**What goes wrong:** Phase 2 产物中 m1 softl1 组合 status=0 未过实用收敛;若优化器对比直接复用历史轮,把失败组合静默混进对比表。
**How to avoid:** 全 16 组合在新血缘目录重跑(秒级×16,成本可忽略,与用户"算力便宜优先并排多存"偏好一致);历史轮(run_20261018_163713)在 meta 里作对照引用。softl1-m1 失败本身写进 SOL-02 结论,是"鲁棒核对最简模型有害"的证据不是缺陷。
**Warning signs:** 对比表里出现历史轮与本次结果不一致却无说明。

### Pitfall 6: 验证侧外参参与训练(第二层泄漏)
**What goes wrong:** 即使数据切分正确,若验证姿态的外参在训练阶段被一起优化,held-out 误差仍失真。
**How to avoid:** 验证侧一律用 `solve_pose_only`(固定内参、单独解该姿态外参、返回 rms)——既有函数正为此而生;训练侧残差统计用 `score_views`。
**Warning signs:** 验证误差低得不合理(接近训练误差)。

### Pitfall 7: joblib 任务闭包携带大对象
**What goes wrong:** `delayed(worker)(...)` 引用闭包里的大数组,loky 每任务重复序列化,800 次求解被传输开销淹没。
**How to avoid:** 任务打包成小元组(观测集子集按任务传,公共配置对象传引用即 pickle 一次);先用 CV 100 次的墙钟时间校准 worker 数再放 bootstrap 800 次。
**Warning signs:** 单任务平均耗时远大于串行实测单次求解秒级水平。

### Pitfall 8: 优化器对比的初始化不一致
**What goes wrong:** 16 组合若各用各的初始化,收敛差异分不清是核的功劳还是起点的运气。
**How to avoid:** D-14 已锁:全部从同一 OpenCV 解出发;每组合记录初始 RMS 与最终 RMS,"提升"才能归因到核。

## Code Examples

### 折级 held-out 误差的产出(数据流核心)

```python
# 每折:训练解得 params 后,验证姿态逐个 solve_pose_only,再按姿态聚合(D-02)
fold_rows = []
for pose_id in val_poses:
    obs_pose = obs_rows(val_obs, pose_mask(val_obs, {pose_id}))
    rvec, tvec, rms = solve_pose_only(obs_pose.image_points[0], obs_pose.object_points[0],
                                      params, model)
    fold_rows.append({"pose_id": pose_id, "rms_px": rms})   # 逐姿态明细全存(产物保真)
fold_heldout = float(np.mean([r["rms_px"] for r in fold_rows]))  # 姿态级平均 → 折级标量
```

[VERIFIED: src/calib/solve/scoring.py 的 solve_pose_only 签名]

### 判行表主契约(体例示意)

```json
{
  "schema": "evaluate-verdict-v1",
  "lineage": {"config_sha256": "...", "upstream": {"corners": "...", "noise": "...", "calibration": "..."}},
  "recommended_solver": {"m1": {"loss": "linear"}, "m4": {"loss": "soft_l1", "f_scale_px": 0.166}},
  "verdicts": [
    {"model_id": "m1", "n_params": 8,
     "generalization": {"cv_heldout_rms_px": 0.101, "pass": true},
     "stability": {"bootstrap_ci_rel_width": 0.12, "slice_cv_rel": 0.08, "pass": true},
     "crlb": {"n_params_below_042": 0, "n_params_below_056": 2, "pass": true},
     "overall": "PASS", "bic": 1234.5, "notes": "Fisher 预警: 无"}
  ],
  "selected": "m2",
  "selection_rule": "held-out 最小;平局按 D-09 等价带 + D-10 BIC"
}
```

阈值口径:稳定性判据的具体线(如 bootstrap 相对区间宽 < 某值)进 config 由规划任务定值;判行表只记录数值与 PASS/FAIL,不搞加权打分(D-12)。

### 逐张 test 评分(D-16)

```python
# 4 个模型的最终内参逐一在 test 65 张上评分:逐张逐模型误差表全存
rows = []
for model_id, final_params in final_params_by_model.items():
    for rec in test_records:
        rvec, tvec, rms = solve_pose_only(rec.object_points, rec.image_points,
                                          final_params, get_model(model_id))
        rows.append({"model_id": model_id, "image_id": rec.image_id, "rms_px": rms})
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ttest_rel 只返回 p 值,置信区间手算 | 结果对象自带 `confidence_interval()` | scipy 1.10.0(2022) | 平局判定可以区间带化,无需手写分位数 |
| joblib 并行只能整体收集结果 | `Parallel(return_as="generator")` 逐个消费 | joblib 1.3(2023-04) | 进度播报与流式落盘都好写;当前 1.6.0 |
| Wilcoxon 零/并列强制回退近似 | `method='exact'` 小样本可精确处理并列 | scipy 近年版本(具体边界随版本) | n=25 兜底检验可靠;meta 记录所用方法即可 |
| OpenCV 5.0 发布(2026-08) | 项目锁 <5.0 不动 | 既定 | 本阶段全部数值目标在 4.14 基线上可审计 |

**Deprecated/outdated:**
- `Parallel(n_jobs=...)` 的 `verbose` 数字进度:仍可用,但生成器迭代进 CLI 回调更贴合"库层不依赖命令行框架"约束。
- 手写 `(x - med) / (1.4826 * MAD)`:`median_abs_deviation(scale='normal')` 一行替代。

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | scipy 无内置 TOST 等价检验;置信区间带是其标准替代,不引 statsmodels | Pattern 3 | 低——区间带法本身就是可辩护的平局判定,只是措辞上少一个"等价检验"名头 |
| A2 | 多起点扰动幅度表(±2%/±10%/1°/±2%)为经验起点,无单篇文献锁定 | Pattern 6 | 低——幅度只在"足以离开初值局部、又不至于落入荒谬解"的宽窗内敏感;config 化后可调 |
| A3 | ObservationSet 子集 pickle 尺寸约几百 KB/折(640×88×2 float64 约 1.1 MB 为上限参考) | Pattern 5 | 低——即便大 10 倍,loky 每 worker 只传一次 |
| A4 | 全部约 1320 次求解墙钟约 1–2 小时(D-04 用户预估口径;单次秒级 × 并行 8) | Standard Stack | 低——超时长只影响等待,不影响正确性;worker 数 config 可调 |
| A5 | joblib 1.6.0 与 numpy 2.5/scipy 1.18 兼容(纯 Python 无编译扩展) | Package Audit | 极低——纯 Python 包无 ABI 问题 |
| A6 | 等价带 δ = max(0.005 px, 较简模型均值 2%) 的具体数值 | Pattern 3 | 中——影响平局判定松紧;已 config 化,判行表会呈现实际值,可复核后调整 |
| A7 | 多起点 RMS 容差 0.001 px / 参数位移当量 0.1 px | Pattern 6 | 中——同 A6,config 化呈现,可调 |

## Open Questions (RESOLVED)

1. **bootstrap 参数稳定判据的"区间宽"阈值取多少**
   - What we know: 判行表第二判据需要一条 PASS 线(如 bootstrap 相对区间宽、切片离散度相对值);D-12 只说三判据各自 PASS,未定线。
   - What's unclear: m1–m4 的实际区间宽度要跑完才知道;先验定线有拍脑袋风险。
   - Recommendation: 规划时按"config 化默认值 + 判行表同时呈现绝对数值"处理——默认线可参照 Phase 3 谱分析的量级(条件数正常模型 vs 退化模型之间差数个量级,先给宽松默认如相对宽 < 50% PASS),真正定线留到数据在手后由判行表数值支撑;这属于规划任务级裁量,不阻塞执行。
   - **RESOLVED:** 已按建议采纳——默认线落为 EvaluateConfig.stability_disp_px_max = 0.05 px(04-01 Task 1 配置契约);判行表稳定判据消费该阈值并同表呈现绝对数值(04-03 Task 2 build_verdict_table),数据到手后可复核调线。

2. **D-05 对照 CRLB 取 8 份切片 Fisher 的均值还是单切片**
   - What we know: 每切片一份 Fisher(8 次,秒级)成本可忽略;切片间位姿结构相同,SE 差异极小。
   - What's unclear: 严格对齐应该"每切片对照每切片",但经验 std 是 8 份合并算的,没有逐切片对应物。
   - Recommendation: 8 份 SE 取均值(逐参数)作对照基准,8 份原始 SE 全部落盘供复核;规划里作为一个明确的小任务写死口径。
   - **RESOLVED:** 已采纳——8 份 SE 逐参数均值作对照、8 份原始 SE 全落盘,口径写死在 04-04 Task 2 slices_by_index(同口径 CRLB 对照组)。

3. **m4 在 CV/LOPO 小训练集上的失败率**
   - What we know: Fisher 已判 m4 高阶方向退化;Phase 2 全量 640 视图上 m4 收敛正常(RMS 0.0857)。
   - What's unclear: 512 视图(CV 折)/ 79 姿态(LOPO)下是否仍收敛;失败率直接影响其判行表"稳定性"维度的样本量。
   - Recommendation: 不预判(D-11);实验函数失败契约已兜底,汇总时失败率作为独立列呈现。
   - **RESOLVED:** 已采纳——四实验(04-04)失败契约兜底、失败率进汇总;04-03 build_verdict_table 泛化判据行记录 n_failed_folds,失败率随判行表落盘。

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| joblib | 全部实验并行 | ✗(项目 venv 未装) | PyPI 最新 1.6.0 | 无需回退:`uv add joblib` 即可;极端情况可用既有 ProcessPoolExecutor 体例 |
| scipy(stats + optimize) | 配对检验/BIC/卡方/精化核 | ✓ | 1.18.1(venv 实测) | — |
| numpy | 全部数组与落盘 | ✓ | 2.5.x | — |
| opencv-python | OpenCV 初始化基线 | ✓ | 4.14.x(锁 <5.0) | — |
| matplotlib | CV 曲线/判行表图 | ✓ | 3.10.x | — |
| pytest | 验证架构 | ✓ | 既有套件全绿 | — |
| CPU 核数 | 并行 worker | ✓ | 20 核(默认 worker 8,config 化) | 串行回退(worker ≤1) |

**Missing dependencies with no fallback:** 无(joblib 一条 `uv add` 解决)。
**Missing dependencies with fallback:** 无。

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest(既有,Phase 1–3 套件全绿) |
| Config file | 既有 pytest 组织(无独立新配置需求) |
| Quick run command | `uv run pytest tests/test_evaluation_split.py tests/test_selection_stats.py -x -q` |
| Full suite command | `uv run pytest` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVA-01 | 姿态切分零重叠;折分配表在 seed=42 下逐位可复现;每姿态恰好 5 次进验证 | unit | `uv run pytest tests/test_evaluation_split.py -x` | ❌ Wave 0 |
| EVA-02 | 平局判定三分支(等价/统计平局/真实差异)在合成差值样本上判定正确;BIC 公式对已知 RSS/k/n 给出已知值;平局时取简规则正确 | unit | `uv run pytest tests/test_selection_stats.py -x` | ❌ Wave 0 |
| EVA-03 | 切片分组构造正确(每姿态第 i 张,按时间戳排序);卡方告警线数学(0.42/0.56)对合成 std 生效 | unit | `uv run pytest tests/test_evaluation_experiments.py -k slice -x` | ❌ Wave 0 |
| EVA-04 | bootstrap 按姿态重采样(抽中姿态带全 8 张);种子确定性(同种子两次重跑分配表逐位一致) | unit | `uv run pytest tests/test_evaluation_experiments.py -k bootstrap -x` | ❌ Wave 0 |
| EVA-05 | LOPO 迭代完整 80 次;MAD z=3.5 判据对注入的已知离群姿态点名正确;MAD=0 兜底路径 | unit | `uv run pytest tests/test_evaluation_experiments.py -k lopo -x` | ❌ Wave 0 |
| SOL-02 | 16 组合表构造与失败契约(m1-softl1 类失败不中断);多起点扰动种子确定性;双容差判定 | unit | `uv run pytest tests/test_evaluation_experiments.py -k optimizer -x` | ❌ Wave 0 |
| 全链 | 小合成观测集(如 6 姿态×2 张、合成角点)跑通 CV→选型最小闭环,产物契约键齐全 | integration(冒烟) | `uv run pytest tests/test_evaluation_smoke.py -x` | ❌ Wave 0 |

冒烟/集成小样用合成角点数据(几秒内完成),不依赖 640 张真实求解;集成测试复用规则(血缘指纹对得上就复用历史轮)沿用既定体例。

### Sampling Rate

- **Per task commit:** 对应新测试文件的快速子集(秒级)
- **Per wave merge:** `uv run pytest` 全量
- **Phase gate:** 全量绿 + 真实数据首轮 `calib evaluate` 产物经人工核查(判行表给四要素指引:看哪个文件/文件作用/看哪里/好坏判据——用户全局要求)

### Wave 0 Gaps

- [ ] `tests/test_evaluation_split.py` — EVA-01(先于实验函数落地,防泄漏断言是地基)
- [ ] `tests/test_selection_stats.py` — EVA-02(统计判定逻辑独立可测)
- [ ] `tests/test_evaluation_experiments.py` — EVA-03/04/05 + SOL-02
- [ ] `tests/test_evaluation_smoke.py` — 合成数据全链冒烟
- [ ] `uv add joblib` — 唯一新依赖(Wave 0 安装)

## Security Domain

> 项目 config.json `security_enforcement: true`,ASVS Level 1。本阶段为离线数值工具,无网络面、无认证/会话/密码学需求。

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | 离线单机工具,无用户体系 |
| V3 Session Management | no | 无会话 |
| V4 Access Control | no | 本地文件系统权限即边界 |
| V5 Input Validation | yes | config 经既有 `io/config.py` 校验体例(EvaluateConfig 照 FisherConfig 的校验写法:区间/排序/非空断言,中文报错);npz 读回一律 `allow_pickle=False`(既有防线,防 pickle 反序列化执行任意代码);yaml 用 safe_load(既有) |
| V6 Cryptography | no | 血缘指纹用 sha256 做完整性标识(非保密用途),无加密需求 |
| V14 Data Protection | partial | 产物只写 `output/`(不进 git 的既定目录契约);不写 `input_data/`(只读) |

### Known Threat Patterns for 本 stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| 恶意 npz(pickle 模式执行代码) | Tampering | `np.load(..., allow_pickle=False)` 全部读回路径坚持既有防线 |
| 配置注入(超范围 seed/worker 数导致资源耗尽) | DoS(本地) | EvaluateConfig 数值区间校验(worker 数上限、N 上限) |
| 上游产物被篡改后实验静默用错数据 | Tampering | 血缘指纹链(config hash + 上游产物 hash)写进每个产物 meta(既有体例,本阶段沿用) |

## Sources

### Primary (HIGH confidence)

- 项目 venv 实测(scipy 1.18.1):`chi2.ppf(0.01,7)=1.2390`、`chi2.ppf(0.05,7)=2.1673`、`ttest_rel(...).confidence_interval()` 可用、`wilcoxon` n=25 可用、joblib 未装 —— 本会话直接执行验证
- 本地代码契约:`src/calib/solve/`(observation/scipy_solver/scoring/result/matrix)、`src/calib/noise/`(fisher/artifacts/injection)、`src/calib/cli/main.py`(evaluate 空壳 1167–1169 行)、`config/calibration.yaml`、`pyproject.toml`、`output/calibration/run_20261018_163713`(12 组合实测,含 m1-softl1 未收敛)、`output/noise/run_20261018_165412/noise.json`(σ 双口径)、`output/fisher/run_20261019_025859/crlb.json`(16 份记录,无 80 视图单张口径)
- `.planning/research/PITFALLS.md`、`ARCHITECTURE.md`(项目级既有研究:BIC n=80、姿态切分强制、实验函数形态)
- PyPI JSON API(curl 实测):joblib 1.6.0,requires_python ≥3.10,纯 Python wheel

### Secondary (MEDIUM confidence)

- [scipy.optimize.least_squares 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) — loss/f_scale 语义:损失按 ρ(f²)=C²·ρ(f²/C²) 评估,f_scale 是内外残差软边界(WebSearch 结果与官方页面文字一致)
- [scipy.stats.wilcoxon 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) + [scipy issue #17530](https://github.com/scipy/scipy/issues/17530) — method='auto' 的精确/近似选择与零/并列行为
- [scipy.stats.ttest_rel 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html) — confidence_interval 方法自 1.10.0 引入
- [joblib 官方文档:parallel_generator](https://joblib.readthedocs.io/en/latest/auto_examples/parallel_generator.html) 与 [Parallel API](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html) — return_as="generator" 进度消费模式(1.3 引入)

### Tertiary (LOW confidence)

- 多起点扰动幅度惯例(±2%/±10%/1°):标定文献中常见量级,未逐篇核对 → 已进假设清单 A2,config 化后不构成执行风险

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 唯一新依赖 joblib 经 PyPI + slopcheck 双核验;其余全在栈内且版本经 venv 实测
- Architecture: HIGH — 复用面逐函数对照过源码签名;实验函数形态有 Phase 2/3 两个成熟先例(matrix.py / fisher.py 编排)
- Pitfalls: HIGH — 全部来自本仓库实测产物(m1-softl1 失败、crlb.json 口径缺口)与既有研究文档,非泛泛清单
- 统计方法(平局带/BIC/卡方线): HIGH(公式与阈值经 venv 数值验证)/ 方法选型的"社会惯例"部分(δ 取值、扰动幅度)为 MEDIUM,已 config 化并进假设清单

**Research date:** 2026-10-19
**Valid until:** 2026-11-19(joblib 版本号可能前进;锁 ≥1.4 后不构成阻塞)

---
*Phase: 04-evaluation-model-selection*
*Researched: 2026-10-19*
