# Roadmap: 相机内参标定与评估框架 (intrinsic-calibration)

**Created:** 2026-10-16
**Granularity:** standard
**Mode:** mvp(每个阶段交付一条在真实数据上能从头跑到尾、结果可用的完整能力,而不是只搭一层技术地基)
**Coverage:** 32/32 v1 requirements mapped ✓

阶段顺序依据 docs/pipeline.md 的八步流程和研究结论:先建模型层再写求解代码(投影和雅可比的数学只实现一份,处处复用)、先测噪声再算理论精度极限、先有一次标定结果再算理论极限(要在标定结果附近做近似)、先建好评估实验再回头优化角点方法。每个阶段收尾必做:若流程结构变化则同步 docs/pipeline.md、每个计划完成后单独提交、更新 REQUIREMENTS.md 里的需求状态。

## Phases

- [x] **Phase 1: 数据基座 + 最小完整标定流程** - 照片体检 + 角点提取第一版 + OpenCV 最简标定,在真实数据上从头跑到尾,得到第一个可用的内参与角点叠加可视化,废帧剔除留痕 (completed 2026-10-16)
- [x] **Phase 2: 模型层 M1–M4 + 双求解后端 + 产物血缘** - 共享投影/雅可比单一数学源(与 OpenCV 一致性比对 + 数值雅可比测试)与可互换求解器 (completed 2026-10-18)
- [x] **Phase 3: 噪声底 + Fisher/CRLB 可达精度预测** - 标定前量化"这批数据最好能做到什么程度"并检测位姿退化 (completed 2026-10-19)
- [x] **Phase 4: 评估实验 + 模型选型 + 优化器对比** - 按姿态防泄漏的交叉验证、重复张分组/bootstrap/留一姿态实验,产出有数据支撑的模型选型 (completed 2026-10-21,选型定案 m2:test RMS 简单优先)
- [x] **Phase 5: 误差预算 + 诊断 + 角点提取寻优** - 实际误差 vs CRLB、误差方向分解、敏感度排序、角点合成退化实验 (completed 2026-10-22)
- [ ] **Phase 6: 报告组装 + CLI 全流水线 + 可复用性硬化** - 一条命令产出可复现的证据链报告,换数据/换棋盘零代码

## Phase Details

### Phase 1: 数据基座 + 最小完整标定流程

**Goal:** 用户在真实数据(input_data/ 640 train + 65 test)上一条命令跑通"入库体检 → 找角点 → 出内参",得到一个基线标定结果与角点叠加可视化,废帧剔除留痕
**Mode:** mvp
**Depends on:** Nothing (first phase)
**Requirements:** IO-01, IO-02, IO-03, COR-01
**Success Criteria** (what must be TRUE):

  1. `calib ingest` 产出 manifest.json:每张图带 pose_id/repeat_idx/split 与预检结论(模糊度/曝光/棋盘可见性,指标比例化),废帧剔除留痕可查
  2. `calib corners` 产出 corners_train.npz/corners_test.npz:亚像素角点,窗口参数从实测方格像素边长推导(不写死像素值);同姿态连拍角点排序经规范化断言(无镜像翻转污染)
  3. 最小标定命令在真实 train 数据上产出基线内参与重投影 RMS(OpenCV 单模型即可),结果落盘 output/calibration/
  4. 重复张数不齐或单张姿态不导致崩溃(数据驱动,无硬编码张数)
  5. 角点叠加可视化图(sampling + 最差若干视图)可人工核查角点是否偏移

**Plans:** 5/5 plans complete
Plans:
**Wave 1**

- [x] 01-01-PLAN.md — 环境/包骨架/配置/姿态数据模型/CLI 九命令地图与 `calib ingest` 数据清单(行走骨架上)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — 角点核心:双检测器 + 比例化窗口推导 + 排序规范化 + 连拍断言 + `calib corners`(COR-01, D-07/08/09)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-03-PLAN.md — 2×2 基线标定 + `calib calibrate` + 真实数据端到端贯通 + SKELETON.md(行走骨架完整, D-10)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 01-04-PLAN.md — 入库体检深化:双来源存疑判定 + 存疑原图副本 + 跨轮可比(IO-02, D-01/02/04)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 01-05-PLAN.md — 全量角点标记副本与拼图 + 全链最终验证与人工核查 + 阶段收尾(D-05/06)

### Phase 2: 模型层 M1–M4 + 双求解后端 + 产物血缘

**Goal:** models/ 成为投影/雅可比的唯一数学源(M1–M4 + 注册机制),双求解后端(OpenCV 初始化 → scipy 精化)可互换且结果同构,阶段间以带血缘的落盘产物通信——理论界与实际求解从此严格可比
**Mode:** mvp
**Depends on:** Phase 1 (角点数据、_types/配置基座)
**Requirements:** MOD-01, MOD-02, SOL-01, ITER-01
**Success Criteria** (what must be TRUE):

  1. M1/M2/M3 与 OpenCV projectPoints、M4 与 fisheye.projectPoints 的一致性比对通过(同参数同输出,浮点容差内),数值雅可比测试对每个模型每个参数通过
  2. OpenCVSolver 与 ScipySolver 均可对同一 ObservationSet + 任一注册模型求解,返回同一 CalibrationResult 结构;scipy 残差仅由 models.project 构成(代码审查 + 测试锁定)
  3. 对真实数据,OpenCV 初始化 → scipy 精化(含鲁棒核)在至少 M1 与 M2 上收敛,重投影误差不劣于纯 OpenCV 结果
  4. 每个阶段产物带血缘指纹(配置哈希 + 上游产物哈希 + 时间戳),任一阶段可在新进程中独立重跑
  5. 注册机制验证:测试中新增一个玩具模型仅通过一处注册即可被求解器使用(零侵入)

**Plans:** 6/6 plans complete

Plans:
**Wave 1**

- [x] 02-01-PLAN.md — 模型层数学核心:M1–M4 投影/雅可比/注册表 + parity/雅可比双重测试锁死(MOD-01/02,D-06/D-07/D-10)
- [x] 02-02-PLAN.md — ObservationSet 观测集装载与行序契约(SOL-01 输入侧,D-02)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02-03-PLAN.md — 配置矩阵化 + CalibrationResult 同构基座 + OpenCVSolver(适配器分派 + 通用初始化兜底,D-09/D-12)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 02-04-PLAN.md — ScipySolver 稀疏精化 + 实用收敛判定 + 残差单一来源锁定(D-08)+ 玩具模型零侵入(D-09)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 02-05-PLAN.md — calibrate 命令矩阵化 + 真实数据全矩阵试车 + 血缘落盘与独立重跑(D-01/D-02,SOL-01/ITER-01)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 02-06-PLAN.md — 存疑帧复核 89 vs 616(M2 固定对照,D-04)+ 阶段收尾(文档契约/需求状态)

### Phase 3: 噪声底 + Fisher/CRLB 可达精度预测

**Goal:** 标定前即可回答"这批数据天生最准能到多少":时域噪声底(8 连拍实测,缺失退化 Immerkaer)、像素噪声→角点误差传递曲线、每内参 CRLB 方差下界与位姿退化检测(外参 Schur 边际化)
**Mode:** mvp
**Depends on:** Phase 2 (共享雅可比、线性化点来自初次标定)
**Requirements:** NOI-01, NOI-02, NOI-03
**Success Criteria** (what must be TRUE):

  1. `calib noise` 对真实数据产出 σ(来源标注:时域/单帧退化;重复数 <2 的姿态自动单帧估计),报告区分 detector 噪声与位姿抖动
  2. 受控注入实验产出"像素噪声 → 角点定位误差"经验传递曲线(数据落盘)
  3. `calib fisher` 产出每个内参的 CRLB σ 与信息矩阵特征谱/条件数,能标出耦合退化方向(如 fx–cy、k3–k6 类);外参经 Schur 消元从估计中消去(Schur 消元:把不关心的外参变量从误差计算里合并掉,只留内参的精度界)
  4. FIM 推导的内参 SE 与 OpenCV stdDeviationsIntrinsics 在同模型同 Σ 下一致到个位数百分比(最强交叉验证)
  5. Σ 不确定性(8 样本 ~27% 相对误差)以敏感性方式随 CRLB 一并落盘,不输出裸下界

**Plans:** 6/6 plans complete

Plans:
**Wave 1**

- [x] 03-01-PLAN.md — 噪声测量线:NoiseConfig 配置段 + Immerkaer 单帧估计 + 受控注入实验 + 传递曲线(NOI-02 主,NOI-01 单帧半,D-08/09/12/13)
- [x] 03-02-PLAN.md — FIM/Schur/CRLB 数学核心(σ=1 基准):分块组装 + 边际化恒等式 + σ 线性缩放 + Σ 带 + 谱/条件数 + 蒙特卡洛(NOI-03 数学,D-10)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03-03-PLAN.md — 时域噪声底 + noise.json 契约 + `calib noise` 命令闭环 + 真实数据首跑(NOI-01 主,D-03/04/11/13)
- [x] 03-04-PLAN.md — OpenCV stdDev 交叉验证协议(8 元组桥接 + 尺度归一结构比对 + M1 固定位/M4 不可比分支,合成数据锁定)(NOI-03,D-07)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 03-05-PLAN.md — FisherConfig + 80 姿态聚合 + 16 份 CRLB 矩阵 + crlb.json + `calib fisher` 闭环 + 真实数据试车对账(NOI-03 主,D-01/02/05/06/07)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03-06-PLAN.md — 五步接力冒烟 + run-all 接线 + pipeline 契约同步 + 需求状态 + 人工核查(全需求收口,含 checkpoint)

### Phase 4: 评估实验 + 模型选型 + 优化器对比

**Goal:** "防作弊考试"全套落地:按姿态分组切分(同一姿态的所有照片只进同一边,结构性杜绝泄漏)的交叉验证、重复张分组实验、bootstrap、留一姿态,模型选型只看 held-out 泛化误差(没参与标定的照片上的预测偏差),优化器/鲁棒核对比有结论;选定后全量重训出候选最终内参
**Mode:** mvp
**Depends on:** Phase 2 (复用其标定函数), Phase 3 (重复张分组实验要对照其 CRLB 理论预测)
**Requirements:** EVA-01, EVA-02, EVA-03, EVA-04, EVA-05, SOL-02
**Success Criteria** (what must be TRUE):

  1. 单元测试断言任何切分下训练/验证的姿态零重叠;四类实验(K 折交叉验证 / 重复张分组(组数取最少重复张数)/ bootstrap 按姿态重采样 / 留一姿态)均实现为独立可复用的实验函数(同一输入必得同一输出),结果 JSON 落盘
  2. K 折交叉验证产出 M1–M4 的训练误差 vs held-out 误差并排表,选型结论只由 held-out 误差得出(辅以 BIC 信息准则,n=有效姿态数)
  3. 重复张分组实验产出经验参数不确定度,并与 Phase 3 的 CRLB 理论预测对照(实证校验);"实测 < CRLB"(好于理论极限,不可能)触发硬告警,不静默通过
  4. bootstrap 与重复张分组实验对照,可分离"纯噪声"与"位姿抽样"两种不确定度来源;留一姿态表能点名问题姿态
  5. 优化器/鲁棒核/初始化策略对比表落盘,含收敛状态与多起点一致性检验(鱼眼模型初值不当时求解会卡在坏解上,须验证没有这种情况);选定模型用全量 train 重训,并在 test/65 张独立照片上评分一次

**Plans:** 6/6 plans complete
Plans:
**Wave 1**

- [x] 04-01-PLAN.md — 考试地基:joblib 依赖 + EvaluateConfig 全量契约 + evaluate 配置段 + 姿态切分纯函数库(零泄漏断言硬门禁)+ 并行助手(EVA-01,D-01/02/03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 04-02-PLAN.md — 优化器对比先行:16 组合网格 + 推荐优化配置 + 多起点双容差一致性 + optimizer.json 落盘与真实数据试车(SOL-02 前半,D-13/14/15)
- [x] 04-03-PLAN.md — 选型统计纯函数:配对检验三分支平局判定 + BIC + 卡方硬告警线 + MAD 离群判据 + 三判据判行表(EVA-02 判定核心/EVA-05 判据,D-06/08/09/10/11/12)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 04-04-PLAN.md — 四类实验函数:K 折交叉验证 + 留一姿态 + 重复张分组与同口径 CRLB 硬告警 + bootstrap 与来源分离(EVA-01 实验侧/EVA-03/04/05,D-04/05/07)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 04-05-PLAN.md — 交付与契约:全量重训 × 4 + test 65 张并排评分 + evaluate.json 主契约与 npz 旁车 + 判行表三图 + 全链冒烟(EVA-02/SOL-02 收口,D-16)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 04-06-PLAN.md — `calib evaluate` 命令闭环 + 真实数据全链对账(成功标准 1–5)+ 人工核查判行表 + 需求状态收口(EVA-02/SOL-02,含 checkpoint;执行期追加:并行修复/砍冗余/温启动否决/test RMS 选型重构,见 04-06-SUMMARY)

### Phase 5: 误差预算 + 诊断 + 角点提取寻优

**Goal:** "误差归因"完整交付:实际误差 vs CRLB 的数据利用率、误差方向分解(视图/图像区域/径向模式 + test 集 X/Y/Z 三分量)、参数敏感度排序与应对建议、角点提取合成退化寻优实验(用数据证明当前角点方法最优)
**Mode:** mvp
**Depends on:** Phase 3 (CRLB 理论界), Phase 4 (复用其评估实验函数)
**Requirements:** ANA-01, ANA-02, ANA-03, ANA-04, COR-02, COR-03
**Success Criteria** (what must be TRUE):

  1. 误差预算表落盘:每内参实际误差 vs CRLB(数据利用率比值),实测低于理论下界硬报错
  2. 误差方向分解可查:按视图/图像区域/径向模式的残差结构,畸变外推(角点覆盖范围之外)风险显式披露且所有空间图表带覆盖足迹
  3. test 集误差按相机坐标系 X/Y/Z 分量分解,哪个分量大由投影雅可比/FIM 可观测性给出理论解释
  4. 敏感度排序(协方差/特征分解)落盘,并针对最大敏感维度给出可操作建议(补拍位姿/加约束)
  5. 角点合成退化实验(加噪/模糊/低对比栅格,含 winSize 扫描与 SB 候选)产出定位误差曲线,以数据锁定当前条件最优角点配置;同姿态连拍角点 x/y 方向重复性各向异性分析落盘

**Plans:** 6/6 plans complete
Plans:
**Wave 1**

- [x] 05-01-PLAN.md — 阶段⑦地基:AnalyzeConfig 配置段 + analyze 产物契约层 + ANA-01 误差预算库层(与 evaluate 同一条判定线的复核 + 硬报错 + k1–k3 口径声明)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 05-02-PLAN.md — `calib analyze` 命令骨架(上游六轮定位/血缘/失败契约/四要素播报)+ ANA-01 端到端:真实数据预算表与证据图 + findings 结论 4 对账

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 05-03-PLAN.md — ANA-02 残差方向分解端到端:视图/区域/r 分带 + r⁴ 覆盖足迹(findings 结论 5 产品化)+ 外推区显式披露 + 空间图带足迹

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 05-04-PLAN.md — ANA-03 test 三分量反解(本阶段唯一新建数学,全既有接口)+ ANA-04 敏感度排序、像素当量、补拍建议与解不唯一披露(Fisher what-if 仅 full 档,裁定②)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 05-05-PLAN.md — COR-02 角点合成退化寻优端到端(唯一重计算块:加噪/模糊/低对比 × winSize 阶梯 × SB 候选;双口径并报,裁定①;quick/full 两档)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 05-06-PLAN.md — COR-03 连拍各向异性 + 六块全链真实数据收口 + pipeline.md/findings/需求状态同步 + 人工核查(含 checkpoint)

### Phase 6: 报告组装 + CLI 全流水线 + 可复用性硬化

**Goal:** 一条命令 `calib run-all` 在新数据上完成标定+分析+报告:证据链 HTML/MD 报告(结论+图表+血缘指纹)、运行清单与跨运行对比、回环建议表、问题姿态剔除重标一键执行;换数据集/换棋盘仅改 config
**Mode:** mvp
**Depends on:** Phase 5 (全链路产物)
**Requirements:** REP-01, REP-02, REP-03, ITER-02, ITER-03, ITER-04, REU-01, REU-02, REU-03
**Success Criteria** (what must be TRUE):

  1. `calib run-all` 从 input_data/ 直达 output/report/(HTML+MD),报告含最终内参+协方差、模型选型结论、误差预算表、全部图表(角点叠加/残差热图/CV 曲线/CRLB 对比/敏感度排序/摆位覆盖图),每个数字可溯源到产物
  2. 报告每节以判行开头(三判据 PASS/FAIL),并按内置回环触发条件表自动给出"回哪一步、做什么"建议
  3. 问题姿态剔除清单机制 + 剔除后重标定一键执行,重标结果与原结果在报告中 diff(跨运行对比)
  4. 用一份缩小/变体的测试数据集(姿态数/重复张数不同、张数不齐)验证:仅改 config 即可 run-all 跑通,标定正确性不受张数不齐影响
  5. 各阶段 CLI 子命令均可独立运行于既有上游产物(`--from/--until` 语义),Python 库接口可用(非 CLI 路径有测试)

**Plans:** 4 plans

Plans:
**Wave 1**

- [ ] 06-01-PLAN.md — 报告组装库层与模板:report/build.py 组装纯函数 + jinja2 双模板 + `calib report` 命令落地(report_data.json 契约 + 缺席段/断链披露)(REP-01/02,用户裁定①)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 06-02-PLAN.md — 回环三件:回环建议规则引擎 + suggest_exclude.json + 跨运行 diff(锚点=上一份 report 轮)接进报告(ITER-02/03/04,用户裁定①②)
- [ ] 06-03-PLAN.md — CLI 全流水线收口:run-all 八步接 report + --from/--until + 失败断点中文指引 + 退出码收口 + 合成测试(REP-03)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 06-04-PLAN.md — 可复用性硬化与全链验收:张数不齐变体测试 + 新数据迁移文档 + REU 逐条对账 + docs/需求状态同步 + 真数据人工 checkpoint(REU-01/02/03,含 checkpoint)

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. 数据基座 + 最小完整标定流程 | 5/5 | Complete   | 2026-10-16 |
| 2. 模型层 M1–M4 + 双求解后端 + 产物血缘 | 6/6 | Complete    | 2026-10-18 |
| 3. 噪声底 + Fisher/CRLB 可达精度预测 | 6/6 | Complete    | 2026-10-19 |
| 4. 评估实验 + 模型选型 + 优化器对比 | 6/6 | Complete    | 2026-10-21 |
| 5. 误差预算 + 诊断 + 角点提取寻优 | 6/6 | Complete   | 2026-10-22 |
| 6. 报告组装 + CLI 全流水线 + 可复用性硬化 | 0/4 | Planned | - |

## Phase Closing Checklist (applies to every phase)

1. docs/pipeline.md 同步:阶段结构/产物契约/回环路径变化时更新(ITER-maintenance)
2. 原子提交:每个 plan 完成后原子 commit
3. REQUIREMENTS.md traceability 状态更新
