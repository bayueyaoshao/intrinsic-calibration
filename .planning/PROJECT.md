# 相机内参标定与评估框架 (intrinsic-calibration)

## What This Is

一个评估驱动的相机内参标定与精度评估框架:输入预先采集好的标定图像(`input_data/`),输出当前数据条件下可论证的最优内参,以及完整的证据链报告——包括噪声底与可达精度预测(Fisher 信息/CRLB)、角点提取寻优证据、抗过拟合的模型选型、误差预算与参数敏感度诊断。用户是做相机标定研究的工程师,要求每一步结论都用数据证明。

## Core Value

对给定的一批标定图像,产出**可论证为当前数据下最优**的内参:held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB 理论下界,三条判据同时达标,并量化"距离数据极限还有多远"。

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

<!-- Current scope. Building toward these. -->

- [ ] 数据接入:按固定目录约定加载 `input_data/train/`(当前数据 80 姿态 × 8 连拍)与 `input_data/test/`(当前 65 张独立采集),含入库预检(模糊度/曝光/棋盘可见性);数据契约为弹性(姿态数/重复张数/测试张数数据驱动,详见 REQUIREMENTS REU-01)
- [ ] 噪声底分析:利用同姿态 8 连拍实测时域噪声;缺失时退化为单帧估计(Immerkaer/平坦区域),报告中标注来源
- [ ] Fisher 信息 / CRLB 可达精度预测:标定前预测每个内参的方差理论下界,并检测位姿分布退化(如 fx–cy 耦合)
- [ ] 角点提取寻优:候选方法/参数在合成退化实验下对比定位误差,用数据证明当前条件最优
- [ ] 四类候选相机模型:针孔+Brown(k1k2p1p2 / k1k2k3p1p2 / 有理模型 k1–k6p1p2)与鱼眼等距 KB4,统一 CameraModel 抽象 + 注册机制,新模型零侵入扩充
- [ ] 抗过拟合标定:按姿态切分 train/val,K 折交叉验证,模型选择只看 held-out 泛化误差(辅以 BIC、参数可辨识性)
- [ ] 多优化器对比:不同非线性优化方法/鲁棒核策略,验证是否按预期提升精度
- [ ] 四类评估实验:K 折 CV(按姿态)、重复索引切片 ×8(纯噪声→经验参数不确定度+CRLB 实证校验)、bootstrap ×N、留一姿态 ×80(问题姿态定位)
- [ ] 误差预算与诊断:实际误差 vs CRLB、误差方向分解(按视图/图像区域/径向模式)、参数敏感度排序与应对建议
- [ ] 报告输出:汇总各阶段证据,生成含可视化的报告(角点叠加、残差热图、CV 曲线、CRLB 对比、敏感度排序)
- [ ] 迭代回环:阶段产物带血缘落盘、问题姿态剔除重标、跨运行对比、回环触发条件表内置
- [ ] 可复用性:换数据集/换棋盘规格零代码(仅改 config),张数不齐不影响标定正确性
- [ ] 交付形态:CLI 一条命令跑完整流水线 + 各阶段可独立调用的 Python 库接口 + 报告

详细 32 条需求(REQ-ID)见 REQUIREMENTS.md。

### Out of Scope

- 多相机/相机阵列标定 — 本项目硬性限定单相机(所有 input_data 图像必须来自同一相机相同参数环境)
- 相机品牌/接口适配 — 内参只由图像决定,与设备无关
- 在线/实时标定 — 离线批处理定位
- 标定数据采集工具 — 图像由外部流程提供,input_data 只读
- 全向相机模型(MEI/Scaramuzza OCam)— 当前数据非折反射相机,需要时经模型注册机制后期加入
- 其他标定板类型(ChArUco/圆点阵列)— 当前数据为棋盘格;检测器抽象预留扩展

## Context

- **数据现状**(已就位,只读):`input_data/train/` 80 个姿态子目录 × 每姿态 8 张连拍(共 640 张),同姿态重复拍摄用于实测角点检测噪声与时域噪声底;`input_data/test/` 65 张独立采集图像用于最终验证。全部 2560×1440 JPEG,文件名含时间戳。
- **图像特征**:广角镜头,桶形畸变明显;棋盘在画面中占比偏小、主要集中在中部,边缘强畸变区可能缺角点覆盖——畸变参数可观测性是误差预算重点,由 Fisher 分析量化。
- **标定板**:棋盘格,11×8 内角点(88 个),方格 25mm,规格已在 `config/board.yaml`。内角点密集(方格约 15–25px),角点提取大概率是全流程精度瓶颈。
- **数据结构推论**:训练/验证切分必须按姿态为单位(同姿态 8 张几乎相同,按图像切分会泄漏);同姿态 8 连拍提供噪声信息,80 个姿态才提供位姿信息。
- **方法论**:借鉴 ML 评估方法论——交叉验证防过拟合、重采样实验测参数稳定性、信息论下界(CRLB)量化数据利用率;"最优"必须可论证。
- **关键架构约束**:models / noise(FIM)/ evaluation(诊断)必须共享同一套投影与雅可比实现,否则理论界与实际求解不可比。
- **算法惯例**:模型/超参经 CV 选定后,最终交付内参用全量数据重训一次;CV 只用于选型。

## Constraints

- **Tech stack**: Python 3.12 + OpenCV 4.x(锁 <5.0) + numpy/scipy — 统计分析/可视化/快速迭代需要,离线场景性能足够;OpenCV 5.0 刚发布,churn 风险与可审计数值目标冲突
- **全局架构原则**: 阶段间仅通过带血缘的落盘产物通信(文件是唯一真相源),每阶段可独立运行/重跑/检查,输入输出契约显式 — 防止"分析无从下手"和迭代全量重跑
- **数据契约(硬性)**: input_data 全部图像必须来自同一相机、相同拍摄参数环境 — 单相机标定项目的前提
- **目录契约**: `input_data/train|test` 只读输入;`config/` 存放 board.yaml 与算法配置;运行产物写 `output/`(不进 git)
- **模块布局**: src/calib/ 下 io / corners / noise / models / solve / evaluation / analysis / visualization / report,与流水线阶段一一对应
- **单相机**: 所有标定、评估、报告均针对一台相机的一次标定任务

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 单相机硬性约束 | 所有 input_data 图像来自同一相机相同参数环境 | — Pending |
| 评估按姿态切分,不按图像 | 同姿态 8 张几乎相同,按图像切分会数据泄漏 | — Pending |
| 同姿态 8 连拍用于噪声实测 + 角点均值聚合候选 | 提供噪声信息而非位姿信息;聚合策略由切片实验裁决 | — Pending |
| 四类候选模型(M1–M4)+ 可扩展模型框架 | 覆盖当前数据疑似广角特性;注册机制保证后期扩充零侵入 | — Pending |
| 模型选择只看 held-out 泛化误差 | 训练误差随复杂度单调下降,不能作为选型依据 | — Pending |
| 重复索引切片实验替代 bootstrap 作为纯噪声度量 | 位姿集合相同仅噪声变化,控制变量更干净;CRLB 实证锚点 | — Pending |
| "最优"三判据共同定义 | 泛化误差最小 + 参数稳定 + 逼近 CRLB,三者互不可替代 | — Pending |
| 共享投影/雅可比实现(models/noise/evaluation) | 理论界与实际求解必须严格可比 | — Pending |
| 交付 CLI + Python 库 + 报告 | 兼顾易用性与实验/二次开发灵活性 | — Pending |
| 可视化与报告分层:visualization 出图,report 组装 | 绘图原语被各阶段复用,职责不混 | — Pending |
| 数据契约弹性:重复张数允许不齐(切片按最少重复数截断/聚合加权/单张退化) | 换数据即用,张数差异不阻塞标定 | — Pending |
| 换数据集/换棋盘零代码,config 是唯一调整点 | 框架是可复用工具不是一次性脚本;角点窗口等参数比例化自适应是前提 | — Pending |
| 流程图用 Mermaid 维护于 docs/pipeline.md,通俗文字表述 | agent+git 可维护、GitHub 原生渲染;阶段结构变化时收尾任务必须同步更新 | — Pending |
| 锁 opencv-python <5.0 | OpenCV 5.0(2026-08)太新,calib3d churn 风险;本项目精度瓶颈不在求解器版本 | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-10-16 after requirements finalization*
