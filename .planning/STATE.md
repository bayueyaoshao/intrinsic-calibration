---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-22T04:00:11.834Z"
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 29
  completed_plans: 29
  percent: 83
---

# STATE: intrinsic-calibration

## Project Reference

- **Core Value:** 对给定的一批标定图像,产出可论证为当前数据下最优的内参(held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB),并量化"距离数据极限还有多远"
- **Current Focus:** Phase 5 — 误差预算 + 诊断 + 角点提取寻优
- **Key Docs:** .planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, docs/pipeline.md

## Current Position

Phase: 5 (误差预算 + 诊断 + 角点提取寻优) — COMPLETE(6/6 计划收口)
Plan: 6 of 6 完成(05-06 COR-03 连拍各向异性端到端 + 阶段五收口)

- **Phase:** 5
- **Plan:** 6 of 6 完成(05-06 COR-03 连拍各向异性端到端 + 阶段五收口)
- **Status:** Phase complete(下一步:Phase 6 报告)
- **Progress:** [██████████] 100%
- **备注:** 验收收口 2026-10-22:gsd-verifier 目标倒推 6/6 达标、
  0 缺口,VERIFICATION.md 在盘(ANA-01..04/COR-02/03 逐条证据对账,
  关键数字与 findings 结论 6–11 逐位一致);findings 结论 6–11 轮名
  勘误 025045→044356 + evaluate 降载取代注记(结论 3/5 的 50 次
  bootstrap 数字保留历史出处);旧 evaluate 轮 131613 已按及时清理
  删除(降载新轮 071708 选型逐位一致);当天另收两条 quick:evaluate
  降载(ikq,耗时 118→53.5 分钟)+ 测试分两档(t9o,47 个真数据用例
  打 realdata 标记,白天 -m "not realdata" 609 个分钟级)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 4/6 |
| Requirements shipped | 20/32 |
| Plans this phase | 5/6 |
| Blocked items | 0 |
| Phase 05 P03 | 30min | 3 tasks | 6 files |
| Phase 05 P02 | 18min | 3 tasks | 5 files |
| Phase 05 P01 | 20min | 3 tasks | 7 files |
| Phase 04 P01 | 12min | 2 tasks | 9 files |
| Phase 4 P04 | 58min | 3 tasks | 2 files |
| Phase 4 P02 | 3h43min | 4 tasks | 4 files |
| Phase 4 P03 | 18min | 2 tasks | 2 files |
| Phase 4 P05 | 29min | 3 tasks | 5 files |
| Phase 4 P06 | 跨 2 日 | CLI 闭环+集成对账+提速重构+选型重构 | 3 files(+docs) |
| Phase 05 P04 | 6h38min | 3 tasks | 8 files |
| Phase 05 P05 | 2h05min | 3 tasks | 5 files |
| Phase 05 P06 | 1h30min | 4 tasks | 9 files |

## Accumulated Context

### Decisions (from PROJECT.md / research)

- models/ 是投影/雅可比唯一数学源;parity + 数值雅可比测试在任何求解代码之前落地
- 所有切分按 pose_id(GroupKFold 语义),绝不按图像——结构性防泄漏
- 噪声 σ 先于 Fisher;求解先于 Fisher(线性化点)
- 评估 runners 先于角点寻优实验(复用 calibrate_fn 网格)
- 文件是唯一真相源:阶段间仅通过带血缘的落盘产物通信
- opencv-python 锁 <5.0;Python 3.12 + numpy/scipy
- CV 只用于选型;交付内参用全量数据重训

### Todos

- None(阶段二开工前置 quick task 已完成,下一步 /gsd-plan-phase 2)

### Blockers

- None

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261017-trb | 阶段一审查遗留修复:CR-01 空划分崩溃 + WR-01~04(拼图路径/yaml 报错/针孔失败契约/体检配置驱动),106 测试全绿 | 2026-10-17 | 3439c3e | [261017-trb-review-fixes](./quick/261017-trb-review-fixes/) |
| 261017-mmj | 角点检测最终方案落地:SB 为主、classic 兜底 + 失败帧独立产物 + 清理中间实验产物 | 2026-10-17 | dcd5abe | [261017-mmj-corners-sb-classic-fallback](./quick/261017-mmj-corners-sb-classic-fallback/) |
| 261017-pub | 存疑帧/连拍点名帧专属复核文件夹:review_suspects(89 张)+ review_burst_flagged(24 张),一个文件夹看完人工复核 | 2026-10-17 | 0be6887 | [261017-pub-calib-corners](./quick/261017-pub-calib-corners/) |
| 261022-ikq | evaluate 实验降载:训练侧每姿态 1 张(exposures_per_pose 旋钮)+ bootstrap 100→30/50→10 + lopo 30→10,选型结论与旧轮逐一吻合,耗时 118→53.5 分钟 | 2026-10-22 | 4b6234b | [261022-ikq-evaluate-1-bootstrap-lopo](./quick/261022-ikq-evaluate-1-bootstrap-lopo/) |
| 261022-t9o | 测试套件分两档:6 个真数据文件叠加 realdata 标记(47 用例),白天 `-m "not realdata"` 跳过、夜间全量照跑,not integration 口径逐位不变 | 2026-10-22 | cc57451 | [261022-t9o-realdata-marker](./quick/261022-t9o-realdata-marker/) |

### Research Flags

- Phase 3: FIM 边际化数学需与 stdDeviationsIntrinsics 交叉验证(MEDIUM 置信设计点)
- Phase 5: 角点寻优实验设计(退化栅格/真值/指标)是最不标准化的一块,阶段开始时需设计专项

## Session Continuity

- **Last action:** 05-05 执行完成(2026-10-22,约 2h05min 含真实数据首跑,
  82a779a/46a0a3f/5b7eb8d)— COR-02 角点合成退化寻优端到端:库层
  (三族退化原语 float→clip→rint 量化收尾 + sha1 四元组种子 + 候选
  逐帧过滤 + A1 销险实测)+ run_corner_opt 主编排(分层抽帧/双参照/
  run_parallel 12 并行/TaskFailed 信封/双口径统计)+ analyze 块 5 接线

  + 双口径曲线族图;真实数据首跑 run_20261022_010551(36 帧 × 8 方法 ×
  13 单元,quick 约 44 分钟):校准档零位移、噪声档单调
  (0→0.30px),verdict 点名 classic_w7 更稳(0.0143px vs 生产 sb
  0.0507px,只提示不执行);16 例新测试,全套 579 passed + 集成 13
  passed,旧 analyze 轮 182345 已按及时清理删除

- **Previous action:** 05-04 执行完成(2026-10-22,6h38min 含集成套件,
  05ad259/4592d97/21b8b13/b43f9d2)— ANA-03 test 三分量反解 + ANA-04
  参数敏感度端到端:test_xyz 库层(最小范数反解,z 分数/深度放大/条件数,
  病态 NaN 占行)+ sensitivity 库层(双谱对账 + 像素当量上天平 + 建议
  三分支 + uniqueness 披露 + full 档 what-if)+ analyze 块 3/4 接线 +
  三张证据图;真实数据 k2/k3 载荷 78%(结论 1)、余弦 0.967/0.752、
  advice 三条全命中、uniqueness 1.9943px;24 例新测试。
  TDD(Task 1 RED 05ad259 → GREEN 4592d97)

- **Earlier action:** 05-03 执行完成(2026-10-22,30min,7f8bf88/2604094/
  1196bed/a08cf34)— ANA-02 残差方向分解端到端:库层(分带器左闭右开
  超界不丢/散布与系统偏移分离/覆盖足迹/r⁴ 律折算)+ 双空间图 + analyze
  块 2 _stage 接线;真实数据与 findings 结论 5 逐位吻合,旧轮已按及时
  清理删除。16 例新测试

- **Next action:** 05-06 COR-03 连拍各向异性 + 六块全链真实数据收口 +
  pipeline.md/findings/需求状态同步 + 人工核查(含 checkpoint)——
  五块齐,仅剩 anisotropy 段留位

- **Notes for next session:** 最新 analyze 轮 run_20261022_010551 在盘
  (corner_opt 段有值:双口径 stats._self/_prod_sb 并排、校准档零位移、
  threshold 0.0871px、sb_accuracy_verdict 落账;七图五旁车),指纹门
  自动消费;角点寻优结论备查——classic_w7 档在未显著劣化单元自参照
  平均 0.0143px vs 生产 sb 0.0507px,verdict 只提示方向不自动执行,
  05-06 收口报告可进「可选动作」栏;noise_64 检出率 29.2%(检测器
  失效区,预期数据);4/288 任务走 TaskFailed 信封留痕;全量集成
  3 个既有失败维持 deferred-items.md 登记(血缘断链 ×2 + 注入老测试
  ×1),05-06 或阶段六开篇专门一轮修复,避免带病进最终验证;
  敏感度结论备查——补 6 个够到画面四角的姿态是当前性价比最高动作
  (k2/k3 方向 0.095px 当量 × 外推区零观测)

## Decisions

- [Phase 05]: 敏感度载荷口径取无量纲坐标——像素参数被 ÷焦距刻度压平,k2/k3 软方向只在无量纲载荷下可见(findings 结论 1 机制落成实现判据);跨谱对账在原始坐标做,2 的幂刻度往返逐位一致 — 真实数据首跑暴露原始坐标载荷被焦距刻度反抢主导,dir1 呈现 fx/fy 与 findings 结论 1 不符;对照 plan 判据纠正为无量纲口径后 78% 逐位吻合
- [Phase 05]: 建议规则表三分支数据驱动现算(补边缘姿态/补转角/不治),每条固定「只提示方向不自动执行」收尾(裁定③);what-if 仅 full 档、合成假设必带「非实拍」声明(裁定②),quick 档 what-if 图直接不生成 — 合成证据纪律:建议是提示不是动作,what-if 是合成假设不是实测;quick 档缺席图与段级 None + note 同口径
- [Phase 05]: test_xyz 三分量与 what-if 谱合成的单一数学源:反解矩阵只从 project_point_jacobian 堆叠,谱合成只调 marginalized_fisher,两处源码级断言锁死(禁 import cv2/禁第二份 FIM 装配);病态帧 NaN 占行 + 乐观偏差与深度判读声明随段落账 — D-06/D-10 延续:分析层绝不重写投影/FIM 数学;病态与口径现象显式声明不冒充解释

- [Phase 05]: 散布与偏移数字分离——band_stats 先减本带均值再取长度算 rms/median/max(整带同向挪动只动 mean_du/mean_dv),「整体偏」与「乱散」不可混淆;分带器一处实现,ANA-02 与 COR-03 共用同一把尺,严禁第二份
- [Phase 05]: 外推披露 σ 口径与 budget 同源(slices.empirical_std ±σ 扰动 0.0966px),findings 结论 5 的 0.135px 是最大组间散布口径只做量级对照;判据表数值全部由公式从当前数据现算不抄死
- [Phase 05]: r 归一化基准声明式披露——basis = evaluate delivery final_params @ 轮名随段与 meta 双落账,并与 findings 所用标定基准相对差核对(漂移中文披露不静默、非阻塞)
- [Phase 05]: budget 直调不套逐块兜底——CRLB 口径铁律违例冒泡到外层失败契约(退出码 1 + 中文 failure_note + 六段全 None),坏结论不许静默出表;后续五段才用 _stage(单段失败照常出报告)
- [Phase 05]: analyze 上游 13 项文件 + 配置哈希在取数前一次性登记血缘;集成测试血缘指纹门逐位核对 sha256,上游任何一轮重跑过旧轮即不可消费(中文跳过理由 + 要求重跑 calib analyze)
- [Phase 05]: 图与数据分离——plot_budget_bars 只画 budget 段现成键,数字结论以 analyze.json 为准,图失败不拦主产物;k1–k3/p1/p2 不画下界条(与利用率置 None 同口径),近零参数就地标注「画面影响可忽略」
- [Phase 05]: 预算判定与 evaluate 同一条判定线——三统计量比值从 slices.npz 独立重算与 crlb_alert.per_param 落账 isclose(1e-9) 对账,漂移即中文报错,不重写第二套判据;σ 只认调用方传入的 noise.json:sigma_detector_corner,坏值进任何取数前拒绝
- [Phase 05]: k1–k3 与 p1/p2 利用率比值置 None(单独下界被参数间补偿空间撑大),CRLB_CALIBER_NOTE 照抄 experiments.py 现成文案不重写第二份;正式判定在像素空间(逐角点重投影位移)
- [Phase 05]: analysis 产物契约整文件照 evaluation/artifacts.py 体例、不跨模块借私有件;测试用「自洽合成底账」(底账与落账比值同源使复核必过,8 组等差构造标准差可手算),真实数据端到端留给 05-02
- [Phase 04]: 阶段四切分函数取整个 CalibrationConfig(读 cfg.seed + cfg.evaluate.*),照 run_injection_experiment 体例;evaluate 核名白名单独立建表(含 cauchy,D-14)
- [Phase 04]: 零重叠硬门禁双保险:split_by_pose_ids 函数体断言 + 16 例测试对 80 姿态×25 折显式断言姿态级与行级零交集(成功标准 1 执行位);切分分配表纯 str/int 可 json 落盘复查
- [Phase 04]: 优化器对比真实数据定案——四模型推荐均 linear 核(千分之一并列线内无鲁棒核优势);m1 的 soft_l1/cauchy status=0 未过实用收敛,与历史轮 run_20261018_163713 对账一致,写进 SOL-02 结论材料
- [Phase 04]: 多起点一致性实测——m1 判 inconsistent(RMS 极差 2.8e-07 px 而参数位移 0.205 px 超容差,可辨识性发现),m2/m3/m4 consistent;run_multistart 取 CalibrationConfig(要 cfg.seed),worker 层异常兜底转 make_failure 防生成器断流
- [Phase 4]: 选型统计判定核心落地——paired_comparison 三分支(置信区间对等价带,不看 p 值)/bic_full_fit(n=80 锁姿态口径)/crlb_alert_check(卡方双线 df 自适应,硬告警不静默)/build_verdict_table(schema evaluate-verdict-v1 接口先行)/select_model(全 PASS 才当选、等价取简、平局 BIC 裁决、无 BIC 退回取简不静默) — D-09/D-10/D-11/D-12 + EVA-02/EVA-05 判定核心闭环;04-04~04-06 按判行表输入形状供给数据
- [Phase 4]: 四类实验编排落地——experiments.py 四函数统一签名 fn(obs, board, cfg, optimizer_config)(全量配置体例,要 seed),同一条"OpenCV 初始化 → scipy 精化(推荐配置)"路径,验证侧一律只估位姿评分;确定性红线进程级补全:loky 并行子进程是全新进程,cv2 线程锁须随任务元组进 worker 再上一次,否则并行与串行 1e-9 级末位漂移;分组实验 CRLB 每组新算(组自身收敛解为线性化点 + temporal σ),poses80 聚合口径在 experiments.py 源码级禁入 — D-01/D-02/D-04/D-05/D-06/D-07/D-08/D-13 + EVA-01/03/04/05 实验侧闭环
- [Phase 4 收口期]: 温启动被真实数据否决(62/64 对 RMS 差 <0.001 px 但位移超容差 6 对、
  温启动反而慢 404s vs 332s;根因 = 数据在 Fisher 退化方向解不唯一),默认
  warm_start_experiments: false——"秒级精化"只在合成小样成立

- [Phase 4 收口期]: 选型规则重构——test 整体 RMS 简单优先(select_by_test_rms,
  复杂度逐档走查提升 ≥ selection_improve_min=0.10 才升级)升为主判据,旧三判据
  降为第二意见(agreement 打架当场播报);test 集参与选型带轻微乐观偏差,由
  hold-out 第二意见对冲;quick(默认,四实验只考当选模型,cv 3×3、bootstrap 50)
  /full(全模型全量)两档,交付一律全模型

- [Phase 4]: 交付与契约层落地——run_delivery 同签名体例(obs_train, obs_test, board, cfg, optimizer_config),全量重训 + test 逐张并排评分,D-16;BIC 在编排内用 bic_full_fit 算好(RSS 取自全量解,k = 内参数 + 6 × 视图数,n = bic_n_poses),select_model 只消费行内现成值不写第二份数学(D-10);evaluate.json 13 键契约 + 明细 npz 旁车(嵌套明细拍平成 模型__数组名,标量成 0 维数组),json 段只留汇总与 detail_npz 引用(汇总只是索引,D-13);组装顺序防线:明细键带数组但对应段非字典当场中文报错;判行表三图库函数只管画图,图失败不拦主产物;全链冒烟即 04-06 CLI 的调用序列样板 — EVA-02/SOL-02 收口
- [Phase 05]: COR-02 双口径并报(裁定①)落地:stats._self 主口径(各方法对自家干净检出的偏移,量退化敏感度)+ _prod_sb 旁证(对生产 SB 干净检出的绝对偏移,跨方法可比),曲线图各一张、判读文案两列都给;未显著劣化线 = 10 × σ_detector_corner(0.0871px)随段落账——真实数据首跑 verdict 如实点名 classic_w7 更稳(未显著劣化单元自参照平均 0.0143px vs 生产 sb 0.0507px),只提示方向不自动执行(裁定③同口径) — 裁定①判读口径硬性要求;阈值由当前数据现算不抄死
- [Phase 05]: A1 销险实测:CALIB_CB_ACCURACY 与 flags=0 在合成棋盘上检出数相同、逐角点逐位一致(「SB 追加亚像素精化」预期语义未获证实),两档仍都保留候选——真实数据首跑两档曲线确有差异,flag 在实数据上有可测效果;销险一次性脚本用完即删,结论固化进模块 docstring 与 sb_accuracy_verdict。指标口径 = 与参照的逐角点位移而非重投影(窗口吞邻角偏差重投影不可见,Pitfall 4 头号陷阱) — 结论由数据出不预设;一次性脚本按 memory 清理纪律不留痕
- [Phase ?]: COR-03 判读定案:纵向重复散布系统性大于横向(σu/σv 中位 0.68),量级仅检测器噪声 2–4 倍,不支持换角点方法;角点方法选择杠杆在退化鲁棒性
- [Phase ?]: analyze 六块全链收口(run_20261022_025045 六段全有值);检查点按 2026-10-21 隔夜授权自检 5/5 通过,曲线上抓出 05-05 口径别名缺陷并修复(8517767)
