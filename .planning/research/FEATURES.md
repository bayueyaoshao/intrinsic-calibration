# Feature Research

**Domain:** 相机内参标定与精度评估框架(研究型、评估驱动)
**Researched:** 2026-10-15
**Confidence:** HIGH(对标定工具生态的主流功能判定 HIGH;差异化功能的空白判定基于多源交叉验证)

## Feature Landscape

### Table Stakes (Users Expect These)

任何标定工具缺失这些功能即不完整。竞品参照:OpenCV `calibrateCamera`/`cv::fisheye`、Kalibr、MATLAB Camera Calibrator、libCalib、mrcal。

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| 棋盘角点检测 + 亚像素精化 | 所有工具的基础能力;OpenCV `findChessboardCornersSB` + `cornerSubPix` 是事实标准。本数据方格仅 15–25px,检测鲁棒性是硬前提 | LOW | 至少接入 2 个候选实现(SB 与 subPix),为后续寻优留接口;SB 对小方格/模糊更稳(Yang et al. 2018) |
| 标准针孔 + Brown 模型标定(k1k2p1p2、k1k2k3p1p2) | 所有竞品的默认模型;没有它连"基线"都不存在 | LOW | OpenCV 原生支持;归入统一 CameraModel 抽象的 M1/M2 |
| 每视图重投影误差统计表(均值/RMS/最大值,按图像列出) | MATLAB `showReprojectionErrors`、Kalibr 报告、mrcal 均提供;是标定输出的最低限证据 | LOW | 注意:重投影误差只作诊断展示,不作质量判据(mrcal 明确反对) |
| 参数标准误/协方差输出 | MATLAB `estimationErrors`(±1.96σ 置信区间)、Kalibr `--recover-covariance`、Calib.io Calibrator 都提供;研究者默认可得 | LOW | 由求解器 Hessian 逆直接得到;是参数稳定判据的原始来源 |
| 迭代外点视图剔除(标定→剔除高误差视图→重标定) | MATLAB 内置 outlier 流程、libCalib 自动剔除;长流程必备 | LOW-MEDIUM | 与稳健核(Tukey/Huber)互补;剔除动作需记录进证据链 |
| 外参可视化(相机位姿 vs 标定板) | MATLAB `showExtrinsics`、Kalibr、mrcal 均有;位姿分布退化(如平面内旋转不足导致 fx–cy 耦合)靠它人眼初筛 | LOW | matplotlib 3D 即可;报告必含 |
| 畸变可视化(去畸变预览/畸变场箭头图) | 所有工具都有某种形式的畸变直观展示;广角桶形畸变下用户默认要看 | LOW | 还应叠加"角点覆盖范围 vs 全画幅"标示,边缘畸变区是否被支撑一眼可见 |
| 多初始化 + 迭代精化策略 | OpenCV 内置 DLT/单应初始化;鱼眼等距模型需独立初始化路径;无初始化则 LM 不收敛 | LOW-MEDIUM | 每个模型类自带 `initialize()`,是抽象接口的一部分 |
| 模型序列化导出/加载(YAML/JSON:K、畸变系数、模型类型) | Kalibr 输出 yaml、OpenCV FileStorage、MATLAB 导出 cameraParams;无导出则结果不可复用 | LOW | 内参 JSON + 模型注册名,附带元数据(数据指纹、选型依据指针) |
| 入库预检(模糊度/曝光/棋盘可见性筛查) | libCalib 自动质量筛查、Calib.io 最佳实践反复强调;坏帧混入会污染全部下游证据 | LOW-MEDIUM | Laplacian 方差(模糊)、直方图跨度(曝光)、SB 检出完整性(棋盘);产出 per-image 质量表而非直接丢弃 |

### Differentiators (Competitive Advantage)

这些是"评估驱动标定"的立身之本。竞品对照结论:**没有任何一款主流工具将"可达精度预测 + 重采样实验 + 误差预算"作为一等公民交付**;mrcal 在不确定性方向最接近但走的是 splined 模型路线,不做 CRLB 预测、角点寻优证据和参数级误差预算。

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| 时域噪声底实测(同姿态 8 连拍)+ Immerkaer 单帧回退 | 全流程的"米尺":CRLB 需要 σ 输入,角点寻优需要噪声基线对照。竞品中仅 mrgingham 有粗糙的噪声测量尝试,无工具产品化此功能 | MEDIUM | 8 连拍给角点位置方差(时域)+ 像素域 Immerkaer 1996 核(空间);缺失时退化并**在报告中标注来源** |
| Fisher 信息 / CRLB 可达精度预测(标定前)+ 位姿分布退化检测 | 核心差异化:在花时间标定之前回答"这批数据理论上能到什么精度、哪个参数欠观测"。文献有据(Nature 2026 fiducial 标定、Fisher Information Explorer)但无工具内置 | HIGH | 依赖共享雅可比(架构硬约束)+ 噪声底 σ;需做 fx–cy 等参数相关阵分析;警惕 Fisher 矩阵滥用(Vallisneri 2008)——只用于方差下界,不用于绝对误差声称 |
| 角点提取寻优(候选方法×参数在合成退化实验下对比定位误差) | PROJECT.md 已判定角点是全流程精度瓶颈(方格 15–25px);用已知噪声/模糊/压缩的合成退化扫描,把"当前条件最优检测器"变成数据结论。无任何竞品做系统化寻优 | MEDIUM-HIGH | 合成台:对真实角点邻域裁片施加可控退化;评价指标 = 定位误差 RMS + 可重复性;文献基准:Yang 2018(+80% 检出)、arXiv 2410.13371(散焦下 RPE −33.7%)证明该维度确有巨大空间 |
| 姿态级 K 折 CV 模型选择(M1–M4,held-out 泛化误差为唯一选型判据) | 直接回应畸变高阶项过拟合问题(Calib.io:k2–k4 在数据支撑区外爆炸);mrcal 有 cross-validation diff 但无自动选型闭环;Kalibr/OpenCV/MATLAB 均无 | MEDIUM | 关键工程点:**按姿态切分**(同姿态 8 张几乎相同,按图像切分=泄漏);BIC + 参数可辨识性作辅助证据;选定后全量重训一次交付 |
| 重复索引切片实验 ×8(同姿态集、仅噪声不同) | 纯噪声→经验参数不确定度,并作为 CRLB 的实证锚点("理论下界 vs 实测散布"同图对照)。控制变量比 bootstrap 更干净;竞品无此实验形态 | LOW-MEDIUM | 引擎就绪后只是 8 次重解 + 散布统计;是三判据中"逼近 CRLB"的实证桥 |
| Bootstrap ×N 参数不确定度 | 文献已验证可行(Hagemann et al. 2021, arXiv:2107.13484:bootstrap 估计优于经典协方差,尤其中小样本);作为与解析协方差、重复切片三角互证的第三来源 | LOW-MEDIUM | 按姿态重采样(非按图像);与重复切片实验共享跑批基础设施 |
| 留一姿态 ×80(问题姿态定位) | 把 MATLAB/MATLAB 用户手工"逐张删图找坏图"的流程自动化为系统性诊断:哪个姿态在拖累全局 | MEDIUM | 80 次重解需并行/缓存;输出按姿态的 leave-one-out 误差增量排名 |
| 误差预算:实际误差 vs CRLB + 方向分解(按视图/图像区域/径向模式) | 回答 Core Value 的后半句"距离数据极限还有多远";径向分解直接指认欠覆盖畸变阶。无竞品提供参数级误差预算 | HIGH | 需共享投影/雅可比保证理论与实测可比;径向模式分解把残差投到 r 的多项式基上 |
| 参数敏感度排序 + 应对建议(如"补拍边缘位姿") | 把诊断翻译成行动:哪个参数对总误差贡献最大、加什么数据最划算。mrcal 有 uncertainty-vs-coverage 定性结论,无自动排序与建议 | MEDIUM-HIGH | 由 Fisher 对角占优 + 误差预算分解联合推导;建议规则表可先人工化 |
| 证据链报告(角点叠加、残差热图、CV 曲线、CRLB 对照、敏感度排序、噪声来源标注) | Kalibr 有 PDF 报告、MATLAB 有图窗,但都是"结果快照"而非"论证链":本报告要能回答审稿式问题"凭什么是这个最优" | MEDIUM | visualization 出原语图、report 组装(架构已分层);每节先给结论再给证据图 |
| 模型注册机制(新模型零侵入扩充) | 面向演化的差异化:Out of Scope 的 MEI/OCam、thin-prism 后期可插拔;竞品模型清单都是硬编码 | LOW-MEDIUM | 抽象 = project/unproject/jacobian/initialize/name;注册表 + 装饰器即可 |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| 在线/实时标定 | "边采边标很酷" | 与离线批处理定位冲突;实时约束破坏 CV/重采样实验(计算量 ×N);竞品中 Kalibr 的在线 validator 也只做检查不做标定 | 离线全量流水线;必要时提供单阶段 CLI 子命令 |
| 多相机/阵列支持 | "以后可能有多相机" | 单相机是本项目的硬性数据契约;多相机引入外参耦合,误差预算全部失效 | 明确拒绝;输入侧预检可校验图像一致性指纹 |
| 相机品牌/接口适配层 | "接入我们的相机" | 内参只由图像决定;适配层零精度收益、纯维护负担(PROJECT.md 已裁决) | 只读 input_data 目录契约 |
| 以训练 RMSE 最小化为选型目标 | "误差数字越小越好" | 训练误差随模型复杂度单调下降→必然选中最复杂模型→过拟合;mrcal 明确反对用重投影误差作质量度量;Calib.io 指出 RMSE 低于噪声 σ 即过拟合信号 | held-out 泛化误差唯一选型 + CRLB/参数稳定双判据 |
| 默认开启高阶全畸变模型(k4–k6+thin-prism+tilted) | "模型越全越准" | 参数可辨识性崩溃;数据支撑区外行为爆炸;fx–cy 与畸变强耦合产生"漂亮误差、错误参数" | CV 从 M1–M4 中选;高阶模型仅经注册机制按需加入并过同一评估关 |
| 自动"去畸变全图"导出管线 | "标定完总要去畸变吧" | balance/尺寸选择是下游任务相关决策,默认值必然对某类用户错误;偏离核心价值 | 提供带显式参数的 undistort 工具函数(v2+),不进主报告链 |
| 黑盒一键"auto-everything"无证据输出 | "别让我看中间过程" | 与核心价值(每步结论用数据证明)正面冲突;不可审计的结果对研究用户是负资产 | 一条命令跑全链 + 全程证据落盘,而非隐藏过程 |
| 多标定板类型现在实现(ChArUco/圆点) | "支持更多板更通用" | 当前数据只有棋盘格;提前实现=无数据验证的死代码 | 检测器抽象接口预留(v2+ 接 ChArUco) |
| 采集工具/拍摄 GUI | "顺便把采集做了" | input_data 只读是硬契约;采集质量问题是外部流程的职责 | 预检报告指出"这批数据缺什么",反向指导外部采集 |

## Feature Dependencies

```
[io 数据接入+入库预检]
    └──requires──> [目录契约 config/board.yaml]

[corners 角点提取] ──requires──> [io]
    └──feeds──> [solve 标定引擎]

[合成退化测试台] ──requires──> [corners]
    └──enhances──> [角点寻优结论]

[noise 噪声底] ──requires──> [corners](角点时域抖动) + [io](像素域 Immerkaer)
    └──feeds──> [Fisher/CRLB](σ 输入)

[models M1–M4 + 共享投影/雅可比]  (地基,无上游依赖)
    └──feeds──> [solve] + [Fisher/CRLB] + [evaluation]

[solve 标定引擎] ──requires──> [corners] + [models]

[Fisher/CRLB 可达精度预测] ──requires──> [models 雅可比] + [noise σ]
    └──feeds──> [误差预算] + [敏感度排序]

[CV 模型选择(姿态级 K 折)] ──requires──> [solve 全模型可用] + [姿态级切分工具]

[重复切片×8 / bootstrap×N / LOPO×80] ──requires──> [solve](批跑基础设施)
    └──enhances──> [CRLB 实证锚点](重复切片)

[误差预算 actual vs CRLB + 方向分解] ──requires──> [Fisher/CRLB] + [重采样实验] + [CV 选型结果]
    └──feeds──> [敏感度排序]

[敏感度排序+应对建议] ──requires──> [误差预算]

[report 证据链报告] ──requires──> [visualization 原语] + 以上全部阶段产物

[最终交付内参] ──requires──> [CV 选型结论] →(全量数据重训一次)
```

### Dependency Notes

- **Fisher/CRLB requires models+noise:** σ 来自噪声底模块、雅可比来自共享投影实现——这正是 PROJECT.md 的硬性架构约束(理论界与实际求解严格可比);两者就绪前 CRLB 数字无意义。
- **CV 选型 requires 全模型 solve:** 四个模型任一不可稳定收敛,选型结论即偏;多初始化是前置条件。
- **重复切片 enhances CRLB:** 8 个纯噪声实现给出经验参数散布,与 Fisher 预测同图对照,使"逼近 CRLB"判据从纯理论变成实证。
- **误差预算 requires 三方汇合:** 只有 CV(选对了模型)+ 重采样(真实不确定度)+ CRLB(理论下界)齐备,actual-vs-limit 的差距归因才成立。
- **按图像切分 conflicts with 无泄漏评估:** 同姿态 8 张几乎相同,图像级切分使 val 集成为 train 的近似复制品,泛化误差被灾难性低估——这不是功能缺失而是必须写进代码约束的反模式。
- **敏感度排序与误差预算同源:** 排序是预算的投影视图,应在同一分析模块内实现,避免两套口径。

## MVP Definition

### Launch With (v1)

最小可论证闭环:能对 input_data 产出"带完整论证的最优内参"。

- [ ] 目录契约数据接入 + 入库预检(模糊/曝光/棋盘可见性,per-image 质量表)— 下游一切证据的数据卫生前提
- [ ] 棋盘角点检测(SB 与 subPix 双候选)+ 角点叠加可视化 — 瓶颈环节的最基础证据
- [ ] 噪声底:8 连拍时域 σ + Immerkaer 回退,来源标注 — CRLB 的输入米尺
- [ ] M1–M4 统一 CameraModel + 注册机制 + 共享投影/雅可比 + 标定引擎(多初始化、稳健核)— 架构硬约束的载体
- [ ] 姿态级 K 折 CV 模型选择(held-out 唯一判据,BIC/可辨识性辅助;选定后全量重训)— 三判据之一
- [ ] 解析参数协方差 + 参数稳定判据(跨 CV 折)— 三判据之二
- [ ] Fisher/CRLB 可达精度预测 + 位姿退化检测(fx–cy 耦合告警)— 三判据之三的理论侧
- [ ] 基础诊断可视化:per-view 误差表、外参图、畸变场图、残差热图
- [ ] 证据链报告 + CLI 一条命令 + Python 库 API — 交付形态

### Add After Validation (v1.x)

- [ ] 重复索引切片 ×8 实验(CRLB 实证锚点)— 触发:引擎稳定后立即做,把判据三从理论变实证
- [ ] 角点提取全参数寻优(合成退化台)— 触发:v1 角点证据显示定位误差占主导时
- [ ] Bootstrap ×N — 触发:需与解析协方差三角互证时
- [ ] 留一姿态 ×80 问题姿态定位 — 触发:CV 误差分布出现长尾时
- [ ] 误差预算方向分解(按视图/区域/径向)+ 敏感度排序与建议 — 触发:CRLB 对照显示实测显著高于下界时

### Future Consideration (v2+)

- [ ] 新模型注册落地(thin-prism、有理+tilted)— 按需:数据出现 KB4/有理模型解释不了的残差模式时
- [ ] ChArUco/圆点检测器抽象实现 — 按需:出现非棋盘数据时
- [ ] MEI/OCam 全向模型 — 按需:折反射相机数据出现时
- [ ] undistort/rectify 导出工具(显式参数)— 按需:下游消费方明确提出时

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| 数据接入 + 入库预检 | HIGH | LOW | P1 |
| 角点检测 + 亚像素精化(双候选) | HIGH | LOW | P1 |
| M1–M4 模型 + 共享投影/雅可比 + 标定引擎 | HIGH | MEDIUM | P1 |
| 姿态级 K 折 CV 模型选择 | HIGH | MEDIUM | P1 |
| 噪声底(时域 + Immerkaer 回退) | HIGH | LOW-MEDIUM | P1 |
| Fisher/CRLB 预测 + 退化检测 | HIGH | HIGH | P1 |
| 参数协方差 + 稳定判据 | HIGH | LOW | P1 |
| 基础诊断可视化(per-view/外参/畸变/残差) | HIGH | LOW | P1 |
| 证据链报告 + CLI + 库 API | HIGH | MEDIUM | P1 |
| 重复索引切片 ×8(CRLB 实证) | HIGH | LOW-MEDIUM | P2 |
| 角点合成退化寻优 | HIGH | MEDIUM-HIGH | P2 |
| Bootstrap ×N | MEDIUM | LOW-MEDIUM | P2 |
| 留一姿态 ×80 定位 | MEDIUM | MEDIUM | P2 |
| 误差预算分解 + 敏感度排序 | HIGH | HIGH | P2 |
| 模型注册机制扩展示例(thin-prism 等) | LOW | LOW-MEDIUM | P3 |
| ChArUco/圆点检测器 | LOW(当前数据无关) | MEDIUM | P3 |
| MEI/OCam 全向模型 | LOW(明确 out of scope) | HIGH | P3 |
| undistort 导出工具 | LOW-MEDIUM | LOW | P3 |

**Priority key:**
- P1: Must have for launch(缺一则"三判据可论证"不成立)
- P2: Should have, add when possible(实验套件,补全证据深度)
- P3: Nice to have, future consideration(扩展性兑现,当前数据不需要)

## Competitor Feature Analysis

| Feature | OpenCV calibrateCamera | Kalibr | MATLAB Camera Calibrator | mrcal | 本项目 |
|---------|------------------------|--------|--------------------------|-------|--------|
| 角点检测 | SB + subPix,最强内置 | 棋盘/AprilGrid 检测器 | 自动 + 手动挑点 | mrgingham(自研) | 双候选 + 合成退化寻优证据 |
| 噪声底测量 | 无 | 无 | 无 | mrgingham 有粗糙尝试 | 8 连拍时域 + Immerkaer 回退,标注来源 |
| 可达精度预测(CRLB) | 无 | 无 | 无 | 无(有投影不确定性,路线不同) | 标定前 per-parameter Fisher 预测 |
| 模型选型 | 手动换 flag 试 | 手动换 target/model | 手动换参数化 | splined vs lean 对比靠人 | M1–M4 姿态级 CV 自动选型 |
| 重采样实验 | 无 | --recover-covariance(单次解析) | bootstrap 需自己写脚本 | cross-validation diff(两半集) | 切片×8 + bootstrap×N + LOPO×80 套件 |
| 不确定性输出 | 稀疏(文档少) | 协方差矩阵 | estimationErrors ±95% CI | 投影不确定性图(其核心强项) | 解析协方差 + 经验散布 + CRLB 三角互证 |
| 外点/坏姿态定位 | flag 手动剔除 | 无系统化 | 手动删图迭代 | cross-validation diff 定性 | LOPO ×80 自动定位 + 敏感度建议 |
| 过拟合防护 | 无(用训练误差) | 无 | 警告"patterns too similar" | 态度上反对但无自动机制 | held-out 唯一选型 + 姿态级切分硬约束 |
| 误差预算/极限距离 | 无 | 无 | 无 | 部分定性 | actual vs CRLB + 方向分解 |
| 报告 | 无(纯 API) | PDF(误差快照) | 图窗 + 可导出 | 工具链文本输出 | 证据链报告(论证链,非快照) |
| 交付形态 | 库 | ROS 包 + CLI | GUI/App | CLI 工具族 + 库 | CLI 一条命令 + Python 库 + 报告 |

**竞争定位结论:** mrcal 是精神上最接近的竞品(公开宣称"重投影误差不是质量度量"),其强项是 splined 模型下的投影不确定性可视化;但它面向下游 stereo/tracking 场景,不做 CRLB 理论下界、角点寻优证据、参数级误差预算,且 C 系生态对研究迭代不友好。本项目差异化空间真实存在且文献支撑充分。

## Sources

- mrcal 官方文档(organization/tour-uncertainty/tour):重投影误差非质量度量、投影不确定性、cross-validation diff — HIGH
- Hagemann, Knorr, Stilla, "Inferring bias and uncertainty in camera calibration", arXiv:2107.13484 — bootstrap 不确定度与偏差检测 — HIGH
- Immerkaer, "Fast noise variance estimation", CVIU 1996 — 空间噪声估计核 — HIGH
- Yang et al., "An Efficient Subpixel Corner Detector for Chessboards", CVPRW 2018 — SB 检出能力 +80% — HIGH
- arXiv:2410.13371(2024)散焦鲁棒角点检测,较 OpenCV RPE −33.7% — MEDIUM(单源)
- OpenCV 官方文档 4.x(calibrateCamera/fisheye/flags/畸变向量布局)— HIGH
- MATLAB Computer Vision Toolbox 文档:Evaluating Accuracy of Single Camera Calibration、standard errors/置信区间 — HIGH
- Calib.io 博客系列(Understanding Parameter Uncertainty、Camera Calibration Quality):RMSE 过拟合信号、高阶项支撑区外爆炸、置信椭圆 — HIGH
- Kanatani 几何 AIC 模型选择框架 — MEDIUM(训练数据 + 二次转引)
- Kalibr GitHub(报告内容、covariance recovery、targets)— HIGH
- Vallisneri, "Use and abuse of the Fisher information matrix", 2008 — Fisher 使用边界警示 — HIGH
- Nature 2026 fiducial 标定 Fisher/CRLB 应用、Fisher Information Explorer(MATLAB FEX)— MEDIUM(单源)

---
*Feature research for: 相机内参标定与评估框架(评估驱动型)*
*Researched: 2026-10-15*
