# Phase 4: 评估实验 + 模型选型 + 优化器对比 - Context

**Gathered:** 2026-10-19
**Status:** Ready for planning

<domain>
## Phase Boundary

"防作弊考试"全套落地:按姿态分组切分(同一姿态的所有照片只进同一边,结构性杜绝泄漏)的交叉验证、重复张分组实验、bootstrap、留一姿态,模型选型只看 held-out 泛化误差(没参与标定的照片上的预测偏差),优化器/鲁棒核对比有结论;选定后全量重训出候选最终内参,并在 test/65 张独立照片上评分。

对应需求:SOL-02 + EVA-01、EVA-02、EVA-03、EVA-04、EVA-05。

**不在本阶段:** 误差归因与敏感度排序(Phase 5)、问题姿态剔除后的一键重标(Phase 6)、角点方法寻优(Phase 5)、报告组装(Phase 6)。本阶段的判行表与产物是 Phase 5/6 的输入。

### 上游已锁定的决策(继承,不再讨论)

- 切分一律按 pose_id(姿态),绝不按图像——结构性防泄漏(02/03 相既有决策)
- CV 只用于选型;交付内参用全量数据重训一次(PROJECT.md 算法惯例)
- 参选模型固定 M1–M4 四个,经注册机制提供(02-CONTEXT D-07);不预否决任何一个
- 三判据"泛化误差最小 + 参数稳定 + 逼近 CRLB"互不可替代(PROJECT.md Key Decision)
- Fisher/CRLB 侧现状(Phase 3 实测,选型的先验输入):m1(条件数 1.0e3)、m2(8.9e3)无量纲谱正常;m3(5.7e11)、m4(1.2e6)判退化,极端方向整块落在高阶畸变项(k5/k2/k6、k3/k4/k2)——"这批数据定不住复杂模型的高阶畸变项"
- crlb.json 已有 16 份记录(2 观测集 × 2 σ 口径 × 4 模型),但**没有**"80 视图单张"口径——本阶段需新算(见 D-05)
- 双口径 σ 实测:detector 0.0087 px / temporal 0.2995 px/轴(时域含位姿抖动)

</domain>

<decisions>
## Implementation Decisions

### A. 交叉验证切分(EVA-01)

- **D-01:5 折 × 重复 5 轮**——80 姿态分 5 折(每折验证 16 姿态 ≈ 128 视图,训练 64 姿态 ≈ 512 视图);5 种不同随机切分重复 5 轮,每姿态恰好 5 次进验证。held-out 指标每模型 25 个折级样本,可画带误差棒的对比;总训练求解 = 4 模型 × 25 折 = 100 次。5 折的训练侧比全量少 20%,held-out 估计略偏悲观(保守方向,可接受)
- **D-02:训练用全视图 + 验证按姿态聚合**——折内训练把选中姿态的 8 连拍全部当独立视图喂入(与交付重训口径一致,选型结论直接可迁移);验证误差先把同姿态 8 张的误差平均成姿态级,再汇总折级——防止同姿态 8 张把验证计数虚增 8 倍
- **D-03:纯随机分折 + 固定种子**——用 config 既有全局 seed(42),折分配表落盘可复查;80 样本随机分足够均衡。**附带裁决**:训练/交付一律用全 640 视图,不做 poses80 聚合口径的 CV(Phase 3 遗留的"聚合策略正式裁决"就此定案——聚合观测集只用于噪声测量与 Fisher,不用于标定交付;连拍相关性损耗由重复张分组实验实测,见 D-08)

### B. 重采样实验(EVA-03/04/05)

- **D-04:bootstrap 200 次 × 全 4 模型**——按姿态有放回重采样 80 个姿态;抽到的姿态带其全部 8 连拍(640 视图口径,模拟"换一批姿态");4×200 = 800 次求解,joblib 并行约 1–2 小时。参数稳定是三判据之一,每个候选模型都需要稳定性证据
- **D-05:重复张分组实验对照"新算同口径 CRLB"**——分组实验每姿态取第 i 张(i=1..8,按文件名时间戳排序)组成 8 个数据组(每组 80 姿态 × 1 张),各组独立完整标定得 8 份参数 → 经验参数不确定度(纯噪声影响)。对照组**新算**一种 CRLB:80 姿态 × 每姿态 1 张、σ 用 temporal 口径(0.30 px/轴),复用 Phase 3 Fisher 数学核心(边际化/Schur 消元:把外参从精度计算里合并掉,只留内参的精度界)。**禁止**用现有 poses80 聚合记录近似(它是 8 连拍均值口径,σ/√8 折减,与单张不可比,会造成"实测 < CRLB"假告警)
- **D-06:"实测 < CRLB"硬告警保持**——分组实验经验不确定度若低于同口径 CRLB 下界(好于理论极限,物理上不可能)触发硬告警不静默通过(ROADMAP 成功标准 3);前提是 D-05 的同口径对照
- **D-07:bootstrap 与分组实验分离两种不确定度来源**——分组实验 = 纯噪声(位姿集合固定,仅噪声不同);bootstrap = 位姿抽样 + 噪声;两者之差刻画"位姿抽样"贡献(ROADMAP 成功标准 4)
- **D-08:LOPO 用预测误差语义 × 全 4 模型**——留一姿态(Leave-One-Pose-Out,每次留 1 个姿态当验证、其余 79 个训练)80 次 × 4 模型 = 320 次求解;每姿态的 held-out 误差离群者即"问题姿态",顺带得到 LOO 泛化估计。影响型(留它训练看参数变化)不做——与 Phase 5 敏感度分析重叠;剔除收益验证留给 Phase 6 一键重标

### C. 模型选型判据(EVA-02)

- **D-09:统计平局取简**——两模型 held-out 比较:同折配对相减(同一折切分下两模型误差相减,消除折间变异)后,若差异不大于折间波动(配对检验不显著)判"统计平局",平局取更简模型(参数少的)
- **D-10:BIC 只作平局裁决器**——held-out 主判据;统计平局时由 BIC(贝叶斯信息准则,惩罚参数数量的模型比较量,越小越好)更小者裁决;n = 有效姿态数 = 80(成功标准写死)。BIC 不做硬门槛
- **D-11:Fisher 退化发现不预否决,靠三判据制衡**——m3/m4 照常参选全部实验;若真过拟合,会在 CV held-out 上自己显形(训练误差降、验证误差升);高阶项 CRLB = inf(数据里没有信息定住它)→ bootstrap 区间爆炸 → 参数稳定判据拦住。Fisher 预警标注进选型报告;若复杂模型胜出,必须解释其与 Fisher 发现的关系
- **D-12:三判据全 PASS 才能当选**——每个候选模型一行判行表:泛化(CV held-out)+ 稳定(bootstrap 区间 + 分组离散度)+ 逼近 CRLB(分组实测 vs D-05 同口径理论,实测低于下界即 FAIL 并触发 D-06 告警);多模型全 PASS 时 held-out 最小者胜,平局按 D-09/D-10 取简。不搞加权打分

### D. 优化器对比与交付(SOL-02)

- **D-13:优化器对比先行**——先在全量 640 视图上跑优化器/鲁棒核网格,每模型产出"推荐优化配置"(收敛且重投影误差不劣者的代表配置);CV/bootstrap/LOPO 及全量重训统一用该配置。避免"选型用默认配置、对比发现更优配置后 CV 结论返工"
- **D-14:对比矩阵 = 4 核 × 4 模型**——scipy 精化核 {linear, soft_l1, huber, cauchy}(鲁棒核:降低离群残差影响力的损失函数)× M1–M4;OpenCV 结果为共同初始化与基线(共 16 组合,现有产物已覆盖其中 linear 与 soft_l1)。f_scale(鲁棒核的切换阈值)用默认值起步,是否加扫描交研究定
- **D-15:多起点一致性检验 = 扰动初值 × N≈10 次**——从 OpenCV 初始化出发,对内参/外参各加随机扰动重复求解,判据 = 参数与重投影误差双容差内收敛到同一解;全 4 模型都做,m4(鱼眼)是重点观察对象(初值不当时易卡坏解);N 终值与扰动幅度交研究定
- **D-16:test 评分全模型并排**——选定模型全量重训(640 视图,已锁)后,4 个模型的最终内参都在 test 65 张独立照片上评分并排(评分只是投影计算,几乎不费时);逐张误差表全存。报告能看到选定模型 vs 其他模型的真实泛化差距,顺带验证选型结论

### Claude's Discretion

研究/规划阶段用数据和技术判断定,无需再问用户:

- D-09 配对检验的具体方法(如配对 t 检验 / Wilcoxon, n=25 折级样本的正态性判断)
- D-10 BIC 的自由度口径(残差计数:按有效观测数还是姿态数;嵌套模型比较的写法)
- D-14 f_scale 是否扫描及取值
- D-15 多起点 N 终值与扰动幅度(内参/外参各扰多少)
- 实验产物目录结构与 JSON 契约细节(遵循既有血缘体例:json 主契约 + npz 明细旁车)
- CV 曲线/判行表的可视化选型
- joblib 并行的任务粒度与 worker 数(config 化,参考 noise 注入实验的 max_workers 体例)
- "问题姿态"的离群判据(如 MAD 倍数阈值,config 化)

</decisions>

<specifics>
## Specific Ideas

- 用户方法论:每一步结论都用数据证明——所以 m3/m4 不预否决,让 held-out 与三判据实证说话(D-11)
- 用户偏好:算力便宜(秒级求解 × 并行)时优先多存信息、并排对比,不怕多花机器时间(D-04/D-08/D-16 全模型跑齐均源于此)
- 产物保真:逐折/逐次实验的完整明细落盘(折分配表、每次 bootstrap 的参数向量、LOPO 逐姿态误差表),汇总表只是索引
- 人工核查:判行表与选型结论呈现时,必须给"看哪个文件、看哪里、好坏判据"的核查指引(用户全局要求)
- 成功标准对账映射:ROADMAP §Phase 4 五条成功标准 ↔ D-01/02(标准 1 的切分断言与四实验函数)、D-09/10/12(标准 2 并排表与选型规则)、D-05/06(标准 3 CRLB 对照与硬告警)、D-04/07/08(标准 4 来源分离与问题姿态)、D-13/14/15/16(标准 5 对比表、多起点、全量重训与 test 评分)

</specifics>

<canonical_refs>
## Canonical References

**下游 agent(研究/规划/执行)动工前必读。**

### 需求与路线
- `.planning/REQUIREMENTS.md` §SOL-02、§EVA-01~05(约 36–44 行)— 六条需求的原文定义
- `.planning/ROADMAP.md` §Phase 4 — 目标、五条成功标准、依赖(Phase 2 标定函数、Phase 3 CRLB 对照)

### 前相决策(直接约束本阶段)
- `.planning/phases/03-fisher-crlb/03-CONTEXT.md` — D-01 双观测集/views640-vs-poses80 对照点、D-13 全量 640 注入、crlb.json 接口约定
- `.planning/phases/03-fisher-crlb/deferred-items.md` — 修正后的 Fisher 发现(m3/m4 高阶项退化)与"聚合策略正式裁决"移交说明
- `.planning/phases/02-m1-m4/02-CONTEXT.md` — D-07 模型清单固定 M1–M4、SOL-02 移交阶段四、SB+classic 角点定版
- `.planning/phases/01-data-foundation-minimal-calibration/01-CONTEXT.md` — D-11 八命令骨架(`calib evaluate` 空壳待填)、D-12 参数只从 config 读

### 代码契约
- `docs/pipeline.md` — 考试阶段(⑤)产物契约、CLI 子命令名、回环路径
- `config/calibration.yaml` — `seed: 42`(全局种子)、`calibrate:` 段(models/solvers/scipy 子段)、`fisher:` 段(本阶段新增 evaluate 段的体例参照)
- `src/calib/solve/` — OpenCVSolver/ScipySolver/ObservationSet/CalibrationResult/scoring(实验函数的直接复用面)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/calib/models/` — M1–M4 投影/雅可比/注册表(实验里任何模型按 model_id 取用,零侵入)
- `src/calib/solve/` — ObservationSet 装载与行序契约、OpenCVSolver/ScipySolver、CalibrationResult 同构基座、scoring(重投影误差统计)——四类实验函数(切分 → 标定 → 评分)的现成积木
- `src/calib/noise/fisher.py` — marginalized_fisher/crlb_se 数学核心(D-05 新算同口径 CRLB 直接复用,加一种观测集配置)
- `src/calib/noise/artifacts.py` — crlb.json 契约与"json 主契约 + npz 明细旁车"体例(D-05 新记录照此落盘)
- `src/calib/cli/main.py` — `calib evaluate` 空壳命令(Phase 1 D-11 预留,本阶段填实现)
- `config/calibration.yaml` `seed: 42` — 全部随机性(分折/bootstrap 重采样/扰动初值)的种子来源
- Phase 3 注入实验的 `injection_max_workers` 并行体例 — joblib/进程并行的 config 化参照

### Established Patterns
- 阶段间仅通过带血缘的落盘产物通信(配置哈希 + 上游产物哈希 + 时间戳),任一实验可独立重跑
- 库层不依赖命令行框架(进度回调由 CLI 注入),静态测试锁死
- 中文报错、yaml safe_load、npz allow_pickle=False、单组合失败转契约记录不中断
- 集成测试复用规则:确定性产物血缘指纹对得上就直接复用历史轮,不重跑

### Integration Points
- 上游取数:corners npz(sb 路线)→ ObservationSet;noise.json 双口径 σ;`m{X}__scipy` linear 收敛解(CV/优化器对比的初始化与对照);crlb.json(判行表"逼近 CRLB"维度的既有记录)
- 下游交付:`output/evaluate/`(目录名遵循 pipeline.md 考试阶段契约)产物 → Phase 5 误差归因(复用实验函数与 LOPO 表)、Phase 6 报告(判行表、CV 曲线、选型结论)
- **joblib 尚未安装**(pyproject 无此依赖)——CLAUDE.md 推荐栈含 joblib ≥1.4,本阶段引入时需加依赖并 `uv lock`

</code_context>

<deferred>
## Deferred Ideas

- 影响型留一姿态(留它训练看参数变化的影响力诊断)— Phase 5 敏感度分析(ANA-04)重叠,本阶段不做
- 问题姿态剔除后的一键重标与跨运行 diff — Phase 6(ITER-03)
- CV 的 views640 vs poses80 双口径对照 — 不做;连拍相关性损耗的实测对照由重复张分组实验承担(Phase 3 D-01 设计意图,D-03 已裁决训练/交付一律全视图)
- 剔除重标的回环触发条件表 — Phase 6(REP-02)

</deferred>

---

*Phase: 04-evaluation-model-selection*
*Context gathered: 2026-10-19*
