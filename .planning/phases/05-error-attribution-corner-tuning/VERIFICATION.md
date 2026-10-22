---
phase: 05-error-attribution-corner-tuning
verified: 2026-10-22T06:40:00Z
status: human_needed
score: 6/6 must-haves verified
overrides_applied: 0
gaps: []
human_verification:
  - test: "确认评审修复 WR-01(敏感度建议分支三文案分档)的行为符合预期"
    expected: "看 src/calib/analysis/sensitivity.py 建议规则表分支三(作用:近零参数 p1/p2 判'不治'的逻辑);具体看:当 p1/p2 任一像素当量 ≥0.05px 时,证据文案必须逐参数如实写'低于/不低于影响线',绝不允许整体断言'均低于';好坏判据:tests/test_analysis_sensitivity.py 的 test_advice_branch3_one_above_line_reports_truthfully 存在且当前 analyze.json 中 p1 0.0076/p2 0.0086 均低于线、文案为'不治'(已核对无误,剩余确认点是分档措辞是否合意)"
    why_human: "修复报告自带'requires human verification'标记,文案措辞是否合意属人的判断"
  - test: "确认评审修复 WR-02(误差预算铁律门禁挪到重计算块之前)的执行顺序"
    expected: "看 src/calib/cli/main.py 的 analyze 命令体(作用:六块分析的编排顺序);具体看:块 1 run_budget 直调位置须在角点寻优等分钟级重计算块之前,铁律违例时不再白算后段;好坏判据:代码顺序 budget → residual → test_xyz → sensitivity → corner_opt → anisotropy,且 budget 违例走失败契约退出码 1(已静态核对顺序无误,剩余确认点是运行时播报体验)"
    why_human: "控制流挪动的运行时效果需实际跑一次 CLI 才能亲见,本次验收按只读约束不启动流水线"
---

# 阶段五验收报告:误差预算 + 诊断 + 角点提取寻优

**阶段目标:** "误差归因"完整交付——实际误差 vs CRLB 数据利用率、误差方向分解(视图/区域/径向 + test 三分量)、参数敏感度排序与建议、角点合成退化寻优、连拍各向异性。
**验收时间:** 2026-10-22(只读验收:未改代码、未重跑流水线、未启动 realdata 测试,数字全部直读落盘 JSON)
**总判定:阶段目标达成。六项需求全部达标,缺口 0;余 2 项评审修复自带的人工确认标记(见文末)。**

## 逐需求判定

| 需求 | 判定 | 代码证据 | 产物证据(run_20261022_044356/analyze.json)与关键数字 |
|------|------|----------|--------------------------------------------------------|
| ANA-01 误差预算 | 达标 | src/calib/analysis/budget.py(372 行;独立重算对账 + 中文 AssertionError 硬报错 + crlb_caliber_note) | budget 段:verdict=pass、违例 0、逐行最小比值 2.756;三统计量 101.77/1107.60/1443.84 倍;判定线 hard 0.4207/soft 0.5564;k1–k3/p1/p2 比值=None 带口径声明;recheck consistent=True——与 findings 结论 6(2.76/101.8/1107.6/1443.8、cx/cy 约 362/378)逐位对账;图 budget_bars.png 双面板在盘 |
| ANA-02 残差分带诊断 | 达标 | src/calib/analysis/residual_structure.py(641 行;radius_footprint/assign_bands/band_stats/coverage_summary + 禁 import cv2 源码断言) | residual_structure 段:分带 RMS 0.0815→0.0862→0.0962→0.1350 单调走高;coverage:r_max_obs 0.6853/四角 0.7974/外推区间 [0.685,0.797]/r>0.5 观测 21.1% 承载 k2 信息 61.4%;extrapolation_note 显式披露 r⁴ 律与"外推区必须先补覆盖"——与结论 7(0.082→0.135/0.685/0.797/21.1%/61.4%)逐位对账;归一化基准 max_rel_diff 0.0(consistent);图 radial_bands.png + coverage_footprint.png 均带足迹与外推区阴影(已目检) |
| ANA-03 test 三分量反解 | 达标 | src/calib/analysis/test_xyz.py(248 行;project_point_jacobian 堆叠反解 + 可观测性三件套) | test_xyz 段:65/65 全成功(n_cond_excluded=0、n_solve_failed=0);条件数中位 35.36/最大 215.33;深度放大中位 0.182;per_direction 给 x/y/z 的原始量级与 z 分数双口径;两条披露(乐观偏差 + Z 判读看 z 分数)随段落账——与结论 8(65/65、35.4/215.3、0.18)逐位对账;图 test_xyz.png 在盘 |
| ANA-04 参数敏感度 | 达标 | src/calib/analysis/sensitivity.py(665 行;谱对账 + 像素当量 + 建议规则表 + what-if 仅 full 档) | sensitivity 段:建议三分支全命中(方向1 k2/k3 载荷占比 78%、当量 0.095px→补 6 个够四角姿态;方向2 fx/fy 80%、0.606px→补转角;p1/p2 0.0076/0.0086px 低于 0.05px 线→不治);uniqueness:bootstrap 最大 1.9943px、位姿抽样占 99.6%、多起点 0.0740px consistent;what_if=None + note"quick 档不做 what-if(裁定②)",figures 无 what_if.png——与结论 9 逐位对账;图 sensitivity_ranking.png(0.095/0.606/0.992)在盘 |
| COR-02 角点提取寻优 | 达标 | src/calib/analysis/corner_opt.py(687 行;退化原语/sha1 种子/方法候选/双口径统计/TaskFailed 信封;检测唯一入口红线) | corner_opt 段:quick 36 帧 × 8 方法(classic_w3–w13 + sb + sb_acc)× 13 单元;stats._self 与 stats._prod_sb 双口径键齐全(裁定①);threshold_px=0.08708(=10×σ_detector 0.008708,随段落盘);校准档 noise_0.0 自参照 RMS 恰为 0;verdict_note 点名 classic_w7 0.0143px vs 生产 sb 0.0507px、"只提示方向不自动执行";sb_accuracy_verdict 记 A1 销险(flags=0 与 ACCURACY 逐位一致)——与结论 10 逐位对账;图 corner_opt_self.png + corner_opt_prodsb.png 三退化族曲线在盘(已目检) |
| COR-03 连拍各向异性 | 达标 | src/calib/analysis/anisotropy.py(442 行;temporal 硬过滤 + clamp 防零除 + 复用 assign_bands) | anisotropy 段:80/80 真连拍姿态(折算姿态 0 混入、NaN 0 行)、7040 有效值;σu 中位 0.0210px/σv 0.0347px、比值中位 0.676(≈0.68);P90 2.67/最大 81.4(有限值无 inf);>1.5 有 1237 个但按角点中位超线 0 个(per_corner_median_aniso 88/88 有效);分带 0.649–0.677 窄带;floor 取自 noise.json(CLI 读入);cross_ref 与 corner_opt 互证——与结论 11 逐位对账;图 anisotropy_map.png + anisotropy_hist.png 在盘(已目检) |

REQUIREMENTS.md 六行(ANA-01~04、COR-02/03)均已标 Shipped,与本报告证据一致;ROADMAP 五条成功标准逐条有产物落点(SC1→budget 段,SC2→residual_structure 段+双图,SC3→test_xyz 段,SC4→sensitivity 段,SC5→corner_opt 段+anisotropy 段)。

## 产物抽查(直读 JSON 数数字)

| 项目 | 结果 |
|------|------|
| analyze.json 六段 | budget/residual_structure/test_xyz/sensitivity/corner_opt/anisotropy 全有值,failure_note=None、无缺席段;lineage 登记 13 项上游 + config_sha256;analysis/ 目录六份明细 npz + 六份血缘 meta.json 在盘 |
| figures 目录 | 九张 PNG 全在盘且逐一目检为真图(预算双面板/分带/足迹/三分量/敏感度/寻优双口径×2/各向异性×2),含披露标注;pipeline.md 阶段⑦契约行同账"九张" |
| evaluate 基准轮 run_20261022_071708 | selection.method=test_rms_simplicity、selected=m2;m2 把 test RMS 0.3020→0.1598px(47.1%),m3/m4 低于门槛不升级;multistart m2 param_disp_px=0.07399264201266795 与 analyze.json uniqueness 内嵌值逐位一致——新旧基准轮口径自洽 |
| 一致性说明 | findings 结论 6–11 血缘行写的 analyze 轮名是 run_20261022_025045(已被评审修复后的复验轮 044356 按及时清理替代删除);044356 全部关键数字与 findings 引用值逐位可复现,仅轮名引用属文档级滞后,建议下次动 findings 时顺手改为 044356(信息级,不算缺口) |

## 反模式扫描与行为核查

- 债务标记:八份阶段五核心文件(budget/residual_structure/test_xyz/sensitivity/corner_opt/anisotropy/artifacts/analyze_figs)grep TBD/FIXME/XXX/HACK/PLACEHOLDER/TODO 零命中。
- 桩检测:analyze.json 六段无空壳——每段都有逐行/逐角点数值底账(如 budget 9 行、corner_opt 8 方法×13 单元、anisotropy 88 角点明细),npz 旁车齐全;无 `return None`/空数组充数。
- 行为抽查:按只读约束未启动 pytest/CLI,以"落盘 JSON 与 findings 结论 6–11 逐位对账"代替端到端复跑——六个需求全部对上,等同验证了上一轮真实运行的输出正确性。

## 已知边界(不计缺口,按任务书豁免)

1. 3 个既有集成测试失败(血缘断链 ×2 + 注入测试口径过时)已登记 deferred-items.md,根因早于阶段五。
2. output/evaluate/run_20261022_130025(仅 optimizer.json 的残缺轮)与阶段五无关,忽略。
3. 旧 evaluate 轮(131613 等)按及时清理删除,属正常;新基准轮 071708 已复核口径一致。

## 人工核查项(2 项,四要素见 frontmatter)

1. WR-01 敏感度建议分支三分档文案合意性(评审修复报告自带人工确认标记;逻辑已被新增反例测试锁定)。
2. WR-02 铁律门禁挪前的运行时播报体验(静态顺序已核对无误;按只读约束未实际跑 CLI)。

## 结论

阶段五目标达成:6/6 需求全部达标,0 缺口。误差归因六块在真实数据上一次跑通且数字互相咬合(预算对账逐位一致、敏感度建议与覆盖短板同源、各向异性与角点寻优互证),九张证据图真实可读。余 2 项人工确认属评审修复流程自带标记,不阻塞阶段收口;可进入阶段六(报告组装 + CLI 全流水线,将直接消费本阶段六段产物与九图)。

---

_验收人:gsd-verifier(只读模式)_ | _证据基线:output/analyze/run_20261022_044356 + output/evaluate/run_20261022_071708 + docs/findings.md 结论 6–11_
