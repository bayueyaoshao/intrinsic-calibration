# Phase 3: 噪声底 + Fisher/CRLB 可达精度预测 - Context

**Gathered:** 2026-10-18
**Status:** Ready for planning

<domain>
## Phase Boundary

标定前回答"这批数据天生最准能到多少",三个交付:

- `calib noise` → noise.json:同姿态 8 连拍实测时域噪声底(重复数 <2 的姿态自动退化 Immerkaer 单帧估计,来源逐条标注);报告区分检测器噪声与位姿抖动两部分
- 受控注入实验:在真实图像上加像素噪声重跑检测器,产出"像素噪声 → 角点定位误差"经验传递曲线(数据全量落盘)
- `calib fisher` → crlb.json:每个内参的 CRLB 方差下界 + 信息矩阵特征谱/条件数 + 位姿退化检测(fx–cy、k3–k6 类耦合方向);外参经 Schur 消元从估计中边际化;FIM 内参标准差与 OpenCV stdDeviationsIntrinsics 交叉验证;Σ 不确定性(8 样本约 ±27% 相对误差)以敏感性带方式随 CRLB 落盘,不输出裸下界

**不在本阶段:** 评估实验/模型选型/优化器对比(阶段四)、误差预算与角点合成退化寻优——加噪之外的模糊/低对比梯子与 winSize 扫描(阶段五)、报告组装(阶段六)。

对应需求:NOI-01、NOI-02、NOI-03。

</domain>

<decisions>
## Implementation Decisions

### Fisher 观测集口径
- **D-01:640 全帧为正式口径 + 80 姿态聚合版一并算出对照**——正式 CRLB 与实际标定(阶段二/四把 640 帧当独立位姿求解)同口径,与 OpenCV stdDeviationsIntrinsics 交叉验证天然对齐;80 姿态对照版同时算出,两版差值 =「连拍相关性吃掉多少信息」,本身是有价值发现
- **D-02:对照版每姿态逐点均值聚合**——同姿态 8 连拍角点逐点取平均(噪声理论降 √8),即 PROJECT.md 预留的"角点均值聚合候选";正式的聚合策略对比裁决仍留阶段四,本阶段只固定对照版口径

### σ 口径:喂给 CRLB 的噪声
- **D-03:双口径并排**——检测器口径(Immerkaer 实测像素噪声经传递曲线折算的角点误差)= 理论主线(纯 iid 高斯,与经典 CRLB 推导及 stdDev 交叉验证最对齐);时域总波动口径(同姿态连拍角点波动总 std,含位姿抖动)= 实际预期主线(与阶段四重复张分组实测直接可比)。两套 CRLB 并排落盘,各配 Σ ±27% 敏感性带
- **D-04:标量主线 + 逐角点对照**——正式 CRLB 与交叉验证用标量 σ(每口径一个全局数);连拍逐点波动直接给出的逐角点 σ(88 个)一并落盘作诊断对照,不进正式 FIM

### 线性化点与模型范围
- **D-05:全 4 模型各出一份 CRLB 并排**——M1–M4 各自在自己的线性化点、同一套 Σ 下算;沿用阶段二"全矩阵多存"风格,阶段四选型直接引用不回头补算
- **D-06:线性化点取 scipy linear 精化结果(mX__scipy)**——高斯假设下的经典 CRLB 推导本就对应无加权极小点,口径最纯正;softl1/opencv 结果不作线性化点
- **D-07:交叉验证范围分级**——M1/M2 硬性对齐到个位数百分比(成功标准);M3(有理模型)尽力对齐,对不上按发现记录不阻塞;M4 记录"OpenCV fisheye 无可比输出"结论即算完成

### 传递曲线注入实验
- **D-08:全量 640 张真实图像进注入实验**——在真实图像上加高斯像素噪声后重跑定版检测配置(SB 为主 + classic 兜底),经验最充分,顺带产出「哪些姿态/图像区域对噪声更敏感」的全量数据;运行时间并行后十几分钟到小时级,符合"算力便宜多存"风格
- **D-09:σ 网格由研究阶段定,每档重复 3–5 次**——网格要求:覆盖实测 Immerkaer σ 邻域 + 延伸到检测器失效点(对数网格约 7–8 档);重复次数压掉注入随机性,曲线做均值±散带

### 上游已锁定决策(重申,下游直接遵守)
- **D-10:FIM 的雅可比必须复用 models/**(阶段二 D-06 唯一数学源)——残差对内参雅可比 = project_jacobian 经 transform_to_camera 链式组装,绝不另写第二份数学
- **D-11:噪声 σ 先于 Fisher;求解先于 Fisher**——noise.json 与标定结果(线性化点)都是 calib fisher 的上游
- **D-12:参数只从 config 读 + 血缘指纹落盘 + 全通俗中文**(阶段一 D-12/D-13,阶段二 ITER-01 惯例)
- **D-13:产物保真**——注入实验逐张误差数字全落盘、时域波动逐姿态逐角点全落盘;汇总表只是索引不是替代(阶段一 D-06 通用规则)

### Claude's Discretion
- 位姿抖动与检测器噪声的分离方法(如 时域²−检测器² 减法或等价做法),以及位姿抖动部分的呈现单位
- Immerkaer 单帧估计的计算区域(整图/平坦区采样策略)
- 每姿态 σ 汇成全局标量 σ 的统计量(中位数/均值等)与离群姿态处理
- 特征谱/条件数的退化判定阈值、耦合方向对的清单与呈现形式
- Σ ±27% 敏感性带的具体档位与呈现
- 80 姿态对照版的外参来源(从该姿态 8 帧外参聚合或另估)
- noise.json / crlb.json 的字段结构、注入实验产物目录细节
- 换线性化点重算一次的敏感性检查是否纳入(建议纳入,一次性验证 FIM 对线性化点不敏感)
- 测试文件组织、中文措辞细节

</decisions>

<canonical_refs>
## Canonical References

**下游 agent(研究/规划/执行)动工前必读。**

### 阶段契约
- `.planning/ROADMAP.md` §Phase 3 — 目标与 5 条成功标准(σ 来源标注、传递曲线、Schur 消元 CRLB、stdDev 交叉验证、Σ 敏感性落盘)
- `.planning/REQUIREMENTS.md` — NOI-01/02/03 细节
- `docs/pipeline.md` 阶段契约速查表 — 第③步 `calib noise`(corners.npz+图像 → noise.json)、第⑤步 `calib fisher`(corners.npz+noise.json+标定结果 → crlb.json);本阶段收尾时同步
- `.planning/PROJECT.md` §Context/Key Decisions — 「8 连拍提供噪声信息,80 姿态才提供位姿信息」「共享投影/雅可比实现」等根决策

### 前序阶段衔接
- `.planning/phases/02-m1-m4/02-CONTEXT.md` — D-06 唯一数学源、全矩阵 13 组合真实数据产物(本阶段线性化点来源)、CalibrationResult/extrinsics.npz 结构
- `.planning/phases/01-data-foundation-minimal-calibration/01-CONTEXT.md` — D-12 参数只从 config 读、D-13 通俗中文、D-06 产物保真规则
- `.planning/STATE.md` — 研究标记位「FIM 边际化数学需与 stdDeviationsIntrinsics 交叉验证(MEDIUM 置信设计点)」,本阶段 D-07 落地

### 研究结论
- `.planning/research/ARCHITECTURE.md` — noise/ 模块布局(时域 σ、Immerkaer 兜底、fisher.py)、FIM = 边际化 JᵀJ/σ² 公式形态、models/ 为叶模块的依赖方向
- `.planning/research/PITFALLS.md` — 亚像素窗口与连拍泄漏等已知陷阱

### 代码基座
- `src/calib/models/base.py` — project_jacobian(内参列)/project_point_jacobian(外参链式前半)/transform_to_camera(后半,dXc_d_pose (N,3,6)),FIM 雅可比组装的全部原料
- `src/calib/solve/residuals.py` — 残差组装(与 FIM 的 J 同构,读它确认行序)
- `src/calib/solve/result.py` + `output/calibration/run_20261018_034000/mX__scipy/` — 线性化点产物:含 params/extrinsics.npz/image_ids/lineage
- `src/calib/io/config.py` + `config/calibration.yaml` — frozen dataclass 配置模式,noise/fisher 配置照此扩展
- `src/calib/util/lineage.py` — make_lineage 血缘指纹直接复用
- corners npz 格式:`image_ids (N,) / uv (N,88,2) / detected (N,) / source (N,)`(sb 路线,train 640 张)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/calib/models/`(base.py / pinhole_brown.py / kb4.py)— M1–M4 解析雅可比已落地并经数值验证(~1e-6 相对误差测试锁定),FIM 直接 import,零新数学
- `src/calib/solve/observation.py` ObservationSet — 观测集装载与行序契约,FIM 的 J 矩阵行序必须与之一致才能与残差/交叉验证对齐
- `src/calib/solve/matrix.py` — 配置矩阵化与产物目录模式(每组合一个子目录),noise/fisher 的多口径多模型产物组织可照搬
- `src/calib/corners/detect.py` — 定版检测配置(SB 为主 + classic 兜底),注入实验重跑检测器直接复用
- `src/calib/visualization/` — 出图原语分层惯例,传递曲线/特征谱图照此落 output/figures/

### Established Patterns
- 产物落 `output/<阶段>/run_<时间戳>/`,带 lineage(created/config_sha256/upstream/round_id);npz 禁 pickle;yaml safe_load
- 失败契约:converged=false 时结构键一个不少、数值字段 null、中文 failure_note(noise/fisher 沿用)
- 参数只从 config 读;中文 AssertionError 在入口点名(见 models/base.py 校验工具)

### Integration Points
- `src/calib/noise/` 目前只有空 `__init__.py` — 全新模块,按 ARCHITECTURE 布局落文件(时域 σ / Immerkaer / 传递曲线 / fisher.py)
- `src/calib/cli/main.py` 的 `noise`/`fisher` 空壳待填,命令名与 help 文案已在阶段一定版
- 线性化点输入 = 阶段二产物 mX__scipy 的 result.json + extrinsics.npz(经 lineage.upstream 记录血缘);角点输入 = corners_train_sb.npz

</code_context>

<specifics>
## Specific Ideas

- 用户风格(第三次验证):算力便宜时全矩阵/全量并排多存——D-01 双观测集、D-03 双口径、D-05 全模型、D-08 全量 640 张一脉相承
- 双口径 CRLB 的解读方式:检测器口径 =「数据物理极限」,时域口径 =「当前采集流程下的实际预期」;两者差距是位姿抖动的代价,报告里要能各说各话
- 注入实验顺带回答「哪些姿态/区域对噪声更敏感」——这是全量 640 张(而非抽样)的额外理由
- 连拍相关性损耗(640 版 vs 80 版 CRLB 之差)预期是阶段四重复张分组实验的理论预告,落盘时留好对照接口

</specifics>

<deferred>
## Deferred Ideas

- 模糊/低对比退化梯子与 winSize 扫描(角点提取寻优 COR-02/COR-03)——阶段五;本阶段注入实验只做加噪维度
- 每姿态聚合策略的正式对比裁决(均值 vs 中位数 vs 选帧)——阶段四重复张分组实验;本阶段对照版固定均值口径(D-02)
- 实测参数不确定度 vs CRLB 的实证校验(EVA)——阶段四;本阶段产出的 crlb.json 是其理论锚点
- 误差预算/数据利用率(实测 vs CRLB 比值)与敏感度应对建议——阶段五 ANA;本阶段只出 CRLB 与退化方向,不做应对建议

</deferred>

---

*Phase: 03-fisher-crlb*
*Context gathered: 2026-10-18*
