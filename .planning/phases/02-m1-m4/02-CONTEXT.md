# Phase 2: 模型层 M1–M4 + 双求解后端 + 产物血缘 - Context

**Gathered:** 2026-10-17
**Status:** Ready for planning

<domain>
## Phase Boundary

models/ 成为投影/雅可比的唯一数学源,双求解后端可互换,阶段间以带血缘的落盘产物通信:

- CameraModel 抽象(投影/反投影/雅可比/参数布局/初值策略)+ 注册机制 + M1–M4 四个模型实现
- 与 OpenCV 的一致性比对测试 + 数值雅可比测试,**先于任何求解代码落地**(阶段三 Fisher/CRLB 要与求解共用同一份数学和导数,理论与实际才严格可比)
- 双求解后端:OpenCVSolver(OpenCV 现成函数出初值)与 ScipySolver(scipy 最小二乘精化,含鲁棒核),可互换、返回同一 CalibrationResult 结构
- 真实数据试车:全矩阵(见 D-01/D-02)
- 存疑帧复核:89 张存疑帧 vs 正常帧的逐帧残差对比(阶段一人工验收的遗留收编,见 D-03)
- 血缘指纹贯穿全部新产物(ITER-01,util/lineage.py 地基已在)

**不在本阶段:** 噪声底与 CRLB(阶段三)、评估实验/模型选型/优化器系统性对比(阶段四,含 SOL-02)、误差归因(阶段五)。

对应需求:MOD-01、MOD-02、SOL-01、ITER-01。

</domain>

<decisions>
## Implementation Decisions

### 真实数据试车范围(用户拍板)
- **D-01:全矩阵一次跑齐**——4 个模型 M1–M4 × 求解路径(纯 OpenCV / OpenCV 初始化→scipy 精化)全部在真实数据上试车,不只跑路线图底线的 M1+M2。单次求解秒级、总耗时分钟级;并排多存信息,阶段三四直接引用,不回头补跑
- **D-02:角点输入用阶段一定版的最优组合**——SB 为主 + classic 兜底那份角点(corners npz 的 sb 路线,705 张全覆盖 = 664 SB + 41 兜底,逐帧带 source 来源字段;标定用其中 train 640 张)。classic 全套那份不进阶段二试车
- **D-03(由 D-02 推论):混合角点定义的代价不在本阶段量化**——SB 与 classic 角点间约 0.135px 系统差,如需评估留阶段四五用 classic 角点对照

### 存疑帧复核(阶段一 UAT 遗留收编)
- **D-04:纳入阶段二**——scipy 精化结果出来后,对比 89 张存疑帧 vs 616 张正常帧的逐帧重投影残差分布(阶段一 baseline 已有 per_view_rms 逐帧数据结构先例)。结论两种走向:存疑组明显更差 → 体检指标与真实误差相关,后续剔除清单有依据;两组无差别 → "看着糊"≠"算得差",不剔除。结论落盘成带血缘产物,并回写阶段一 UAT 的 Gap 闭环记录

### 阶段一遗留修复(开工前置)
- **D-05:CR-01 + WR-01~04 在阶段二开工前用 quick task 修掉**(详见 01-REVIEW.md;约半小时:corners 空划分崩溃 bug + 存疑拼图路径断链 + yaml 语法错误裸栈 + 针孔求解失败契约不对称 + 体检硬编码检测器选项)。修复属于阶段一收尾,不计入阶段二计划

### 上游已锁定决策(重申,下游直接遵守)
- **D-06:models/ 是投影/雅可比唯一数学源;一致性比对 + 数值雅可比测试先于任何求解代码**(项目固有决策,roadmap 成功标准 1/2 的前提)
- **D-07:模型清单固定**——M1 针孔+k1k2p1p2、M2 +k3(即阶段一 pinhole 基线)、M3 有理 k1–k6+p1p2、M4 鱼眼 KB4(即阶段一 fisheye 基线);M1/M2/M3 与 cv2.projectPoints 比对、M4 与 cv2.fisheye.projectPoints 比对
- **D-08:ScipySolver 残差仅由 models.project 构成**(代码审查 + 测试锁定);两后端返回同一 CalibrationResult
- **D-09:注册机制零侵入**——测试中新增一个玩具模型,仅一处注册即可被两个求解器使用
- **D-10:numdifftools 只作测试基准**,绝不进求解/理论计算路径(CLAUDE.md 技术栈约定;需加入 dev 依赖组)

### Claude's Discretion
- ScipySolver 的变量组织(内外参联合优化 vs 分块交替)——研究阶段定;注意 640 视图 × 6 外参 ≈ 3840 个外参变量,least_squares 的稀疏结构(jac_sparsity / 分块组装)怎么用值得研究
- ObservationSet / CalibrationResult 具体字段——沿用阶段一失败契约:converged=false 时结构键一个不少、数值字段一律 null、中文 failure_note
- 一致性比对的浮点容差具体值
- `calib calibrate` 命令如何演进到模型注册机制(阶段一 2×2 基线产物保持向后可读)
- 测试文件组织、中文措辞细节

</decisions>

<specifics>
## Specific Ideas

- 用户原话:"角点输入用上个步骤中 classic+sb 的最优组合跑"
- 用户风格(再次验证):算力便宜时全矩阵并排多存,不怕机器时间——与阶段一 D-06/D-10 一脉相承
- M4 KB4 在非鱼眼广角上收敛不稳属预期内发现(阶段一 2×2 里 sb_fisheye 0.083px 收敛良好);若 scipy 精化异常,按发现记录不中断(沿用阶段一失败契约)
- 求解先于 Fisher(线性化点来自初次标定)——阶段二产物要给阶段三留好接口(标定结果可作线性化点读入)

</specifics>

<canonical_refs>
## Canonical References

**下游 agent(研究/规划/执行)动工前必读。**

### 阶段契约
- `.planning/ROADMAP.md` §Phase 2 — 目标与 5 条成功标准(parity/雅可比测试、双后端同构、真实数据收敛、血缘指纹、注册机制零侵入)
- `.planning/REQUIREMENTS.md` — MOD-01/02、SOL-01、ITER-01 细节
- `docs/pipeline.md` 阶段契约速查表 — 第④步输入输出契约(本阶段会扩展它,收尾时同步)

### 研究结论
- `.planning/research/ARCHITECTURE.md` — models/(base.py / pinhole_brown.py / kb4.py 文件布局、registry、parity 测试定义)与 solve/(OpenCVSolver/ScipySolver 接口、实验名解析)的模块设计
- `.planning/research/PITFALLS.md` — 实现陷阱(fisheye 形状挑剔、float32 精度等)

### 阶段一衔接
- `.planning/phases/01-data-foundation-minimal-calibration/01-CONTEXT.md` — D-10 基线契约、D-12 参数只从 config 读、D-13 全通俗中文
- `.planning/phases/01-data-foundation-minimal-calibration/01-HUMAN-UAT.md` — Gaps 节是 D-04 存疑帧复核的原始出处
- `.planning/phases/01-data-foundation-minimal-calibration/01-REVIEW.md` — D-05 前置 quick task 的修复清单(CR-01 + WR-01~04,含现成修复方案)
- `.planning/phases/01-data-foundation-minimal-calibration/01-VERIFICATION.md` — 阶段一产物现状

### 配置与代码基座
- `config/calibration.yaml` + `config/board.yaml` — 现有配置,新模型/求解器选项照 D-12 扩展
- `src/calib/solve/baseline.py` — 失败契约/形状预处理/per_view_rms 回代的直接前身
- `src/calib/util/lineage.py` — make_lineage 血缘指纹,新产物直接复用

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/calib/util/lineage.py` `make_lineage` — 血缘指纹已就位(created/calib_version/config_sha256/upstream/round_id),ITER-01 的"新进程独立重跑"验证以此为准
- `src/calib/solve/baseline.py` — `build_object_points`(物点网格唯一来源,square_size 只进这里)、`_prepare_view`(fisheye 形状断言)、float32 降精度注释、失败契约、`compute_residuals` 逐帧回代——CalibrationResult 的直接前身
- `src/calib/io/config.py` — frozen dataclass + 中文校验报错模式;FisheyeConfig/CalibrateConfig 是新求解配置的参照
- corners npz 格式:`image_ids (N,) / uv (N,88,2) / detected (N,) / source (N,)`(sb 路线含逐帧检测来源,即 D-02 的输入)
- `tests/test_solve_baseline.py`、`conftest.py` — 求解类测试的组织参照

### Established Patterns
- 参数只从 config 读(D-12);产物落 `output/<阶段>/run_<时间戳>/<名>/`;文档/报错/注释全通俗中文(D-13)
- npz 禁 pickle、yaml safe_load、image_id 白名单(01-REVIEW 确认防线有效)
- 检测质量摘要的成功率计算有空划分保护(quick task 修复后 CR-01 消除)

### Integration Points
- `src/calib/models/` 目前只有空 `__init__.py`——全新模块,按 ARCHITECTURE 布局落文件
- `src/calib/solve/baseline.py` 保留(阶段一基线仍可复现),新求解器与其并存
- `src/calib/cli/main.py` `calibrate` 命令接入新后端;`calib noise`/`fisher` 空壳待阶段三

</code_context>

<deferred>
## Deferred Ideas

- 混合角点定义(SB + classic 兜底,~0.135px 系统差)的代价量化——如需要,阶段四五用 classic 角点对照(D-03)
- 优化器/鲁棒核/初始化策略的系统性对比实验(SOL-02)——阶段四
- M3 有理模型是否过拟合的判定——阶段四 held-out 评估
- stdDeviationsIntrinsics 与自有 FIM 的交叉验证——阶段三(STATE 研究标记位)

</deferred>

---

*Phase: 02-m1-m4*
*Context gathered: 2026-10-17*
