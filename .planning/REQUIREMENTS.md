# Requirements: 相机内参标定与评估框架 (intrinsic-calibration)

**Defined:** 2026-10-16
**Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### 数据接入 (IO)

- [x] **IO-01**: 按姿态 ID 数据模型加载 train(每姿态目录,重复张数 ≥1、允许不齐)与 test(独立采集),目录契约只读;manifest 记录姿态/重复索引/重复数分布
- [x] **IO-02**: 入库预检——模糊度/曝光/棋盘可见性筛查,废帧剔除留痕;预检指标比例化(按棋盘占比等),不依赖绝对像素阈值
- [x] **IO-03**: 配置解析与校验(board.yaml + calibration.yaml);棋盘参数零硬编码,square_size 与内参解耦(改错只影响距离类指标并断言提醒)

### 角点 (COR)

- [x] **COR-01**: 角点检测与亚像素精化,候选方法/参数集可配置;窗口扫描范围从实测方格像素边长推导(比例化),不写死像素值
- [ ] **COR-02**: 角点提取寻优——合成退化实验(加噪/模糊/低对比)下对比候选方法定位误差,以数据证明当前条件最优
- [ ] **COR-03**: 角点提取误差 x/y 方向分解——同姿态连拍同名角点重复性的各向异性分析

### 噪声与可达精度 (NOI)

- [ ] **NOI-01**: 时域噪声底(同姿态连拍相减)+ Immerkaer 单帧退化路径,来源标注;重复数 <2 的姿态自动退化为单帧估计
- [ ] **NOI-02**: 像素噪声→角点定位误差映射(受控注入实验,建立经验传递曲线)
- [ ] **NOI-03**: Fisher 信息/CRLB——每个内参方差理论下界 + 位姿分布退化检测;外参经 Schur 消元边际化;实测误差低于 CRLB 判定为系统偏差信号

### 模型 (MOD)

- [x] **MOD-01**: CameraModel 抽象(投影/反投影/雅可比/参数布局/初值策略)+ 注册机制;models/noise/evaluation 共享同一投影与雅可比实现(单一数学源)
- [x] **MOD-02**: 四类模型实现:M1 针孔+k1k2p1p2、M2 +k3、M3 有理 k1–k6p1p2、M4 鱼眼 KB4;附"与 OpenCV 投影结果一致"的比对测试和数值雅可比校验测试

### 求解 (SOL)

- [x] **SOL-01**: 求解器统一接口 + 双后端(OpenCV 初始化 → scipy least_squares 精化,含鲁棒核),后端可互换、结果同构(CalibrationResult)
- [ ] **SOL-02**: 多优化器/鲁棒核/初始化策略对比实验,验证是否按预期提升精度

### 评估 (EVAL)

- [ ] **EVA-01**: 按姿态切分训练/验证(同一姿态的所有照片只进同一边,结构性防数据泄漏)+ K 折交叉验证
- [ ] **EVA-02**: 模型选择只看 held-out 泛化误差(辅以 BIC、参数可辨识性);选定后全量数据重训出最终内参
- [ ] **EVA-03**: 重复张分组实验——每姿态各取第 i 张连拍组成一组(组数取全体姿态的最少重复张数),各组独立完整标定后对比内参差异,得到纯噪声传播的经验参数不确定度,并用于 CRLB 理论预测的实证校验
- [ ] **EVA-04**: bootstrap ×N 参数稳定性(与重复张分组实验对比,分离"噪声"与"位姿抽样"两种因素)
- [ ] **EVA-05**: 留一姿态 ×N——问题姿态定位与解的脆弱性检查

### 分析 (ANA)

- [ ] **ANA-01**: 误差预算——实际误差 vs CRLB(数据利用率);实测低于理论下界硬报错
- [ ] **ANA-02**: 误差方向分解——按视图/图像区域/径向模式;畸变外推(角点覆盖范围之外)风险显式披露
- [ ] **ANA-03**: 验证集(test)误差在相机坐标系 X/Y/Z 三方向分量对比,哪个分量大,由投影雅可比/FIM 可观测性给出理论解释
- [ ] **ANA-04**: 内参数敏感度排序(协方差/特征分解)+ 针对最大敏感维度的应对建议(补拍位姿/加约束等)

### 报告与交付 (REP)

- [ ] **REP-01**: 可视化原语——角点叠加/残差热图/CV 曲线/CRLB 对比/敏感度排序/摆位覆盖图,matplotlib 统一输出 PNG
- [ ] **REP-02**: 证据链报告组装(HTML/Markdown)——各阶段结论、图表、血缘指纹汇总,含最终内参 + 协方差 + 模型选型结论 + 误差预算表
- [ ] **REP-03**: CLI 全流水线一条命令(run-all)+ 各阶段独立子命令 + Python 库接口

### 迭代与回环 (ITER)

- [x] **ITER-01**: 阶段间仅通过带血缘(配置指纹+上游哈希+时间戳)的落盘产物通信,输入输出契约显式;每阶段可独立运行/重跑/检查
- [ ] **ITER-02**: 问题视图/姿态标记与剔除清单机制,剔除后重标定一键执行
- [ ] **ITER-03**: 运行清单(run manifest)+ 跨运行对比——报告含本次 vs 上次关键指标 diff
- [ ] **ITER-04**: 回环触发条件表内置,报告自动给出"回哪一步、做什么"的建议

### 可复用性 (REU)

- [ ] **REU-01**: 数据契约弹性——姿态数/每姿态重复张数/测试集张数全部数据驱动,不硬编码;重复数不齐时:分组数按最少重复数截断、聚合按张数加权、单张姿态退化单帧噪声估计,均不影响标定正确性
- [ ] **REU-02**: 换数据集零代码——新数据符合目录契约放入后,仅调整 config 即可 run-all 一键完成标定+分析+报告
- [ ] **REU-03**: 换棋盘规格零代码——board.yaml 是棋盘参数唯一来源;角点参数扫描范围随实测方格边长自适应

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### 标定板扩展

- **BOARD-01**: ChArUco 板支持(经检测器抽象加入,部分遮挡鲁棒)
- **BOARD-02**: 圆点阵列板支持(椭圆拟合亚像素)

### 模型扩展

- **MODEL-01**: 全向相机模型(MEI/Scaramuzza OCam,经模型注册机制加入)

### 维护性

- **MAINT-01**: OpenCV 5.x 迁移审计(待 5.0.x 积累 2–3 个补丁版本后)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| 多相机/相机阵列标定 | 项目硬性限定单相机(input_data 全部图像同一相机相同参数环境) |
| 相机品牌/接口适配 | 内参只由图像决定,与设备无关 |
| 在线/实时标定 | 离线批处理定位 |
| 标定数据采集工具 | 图像由外部流程提供,input_data 只读 |
| C++ 实现或嵌入部署 | Python + OpenCV + numpy/scipy 已定栈,离线场景性能足够 |
| autodiff 框架(jax/torch)求雅可比 | 要求手写解析雅可比单一数学源;autodiff 引入第二套实现和重量级依赖 |
| 交互式报告(plotly/bokeh) | matplotlib PNG 是证据链报告的正确产物形态 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.
(注:BOARD-01/02、MODEL-01、MAINT-01 属 v2 预留需求,不在当前路线图内,故有意不进本表。)

| Requirement | Phase | Status |
|-------------|-------|--------|
| IO-01 | Phase 1 | Shipped (Phase 1)(01-01 姿态数据模型与 manifest 写盘;01-04 体检复用其扫描记录) |
| IO-02 | Phase 1 | Shipped (Phase 1)(01-04 三指标比例化体检+双来源存疑判定;只标记不剔除,剔除清单机制归 ITER-02) |
| IO-03 | Phase 1 | Shipped (Phase 1)(01-01 配置解析与校验,棋盘参数零硬编码、square_size 与内参解耦) |
| COR-01 | Phase 1 | Shipped (Phase 1)(01-02 双检测器与比例化窗口推导;01-05 全量标记副本与拼图供人工核查) |
| MOD-01 | Phase 2 | Shipped (Phase 2)(02-01 CameraModel 抽象+注册机制;models/ 单一投影与雅可比,标定评分与存疑复核共用同一实现) |
| MOD-02 | Phase 2 | Shipped (Phase 2)(02-01 四类模型:M1/M2 针孔布朗、M3 有理、M4 鱼眼 KB4;OpenCV 投影一致性测试与数值雅可比校验把关) |
| SOL-01 | Phase 2 | Shipped (Phase 2)(02-03/02-04 求解器统一接口 + OpenCV 初始化 → scipy least_squares 精化双后端,CalibrationResult 同构落盘) |
| ITER-01 | Phase 2 | Shipped (Phase 2)(02-05 全矩阵编排:run 目录内 result.json/comparison.json 全带血缘(配置指纹+上游哈希+时间戳);02-06 存疑复核表延续同一契约) |
| NOI-01 | Phase 3 | Pending |
| NOI-02 | Phase 3 | Pending |
| NOI-03 | Phase 3 | Pending |
| EVA-01 | Phase 4 | Pending |
| EVA-02 | Phase 4 | Pending |
| EVA-03 | Phase 4 | Pending |
| EVA-04 | Phase 4 | Pending |
| EVA-05 | Phase 4 | Pending |
| SOL-02 | Phase 4 | Pending |
| ANA-01 | Phase 5 | Pending |
| ANA-02 | Phase 5 | Pending |
| ANA-03 | Phase 5 | Pending |
| ANA-04 | Phase 5 | Pending |
| COR-02 | Phase 5 | Pending |
| COR-03 | Phase 5 | Pending |
| REP-01 | Phase 6 | Pending |
| REP-02 | Phase 6 | Pending |
| REP-03 | Phase 6 | Pending |
| ITER-02 | Phase 6 | Pending |
| ITER-03 | Phase 6 | Pending |
| ITER-04 | Phase 6 | Pending |
| REU-01 | Phase 6 | Pending |
| REU-02 | Phase 6 | Pending |
| REU-03 | Phase 6 | Pending |

**Coverage:**

- v1 requirements: 32 total
- Mapped to phases: 32 ✓ (100%)
- Unmapped: 0

---
*Requirements defined: 2026-10-16*
*Last updated: 2026-10-18 after Phase 2 execution complete (MOD-01~02 / SOL-01 / ITER-01 shipped,累计 8/32)*
