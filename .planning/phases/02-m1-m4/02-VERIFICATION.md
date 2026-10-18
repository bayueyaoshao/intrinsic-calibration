---
phase: 02-m1-m4
verified: 2026-10-18T04:44:46Z
status: passed
score: 6/6 must-haves verified(SC3 拆成"精化主线"与"鲁棒核严格线"两条,前者已证,后者经人工裁定接受:2026-10-18 开发者选 (a),成功标准 3 由 linear 路径兑现,soft_l1 偏差记为无坏点数据上的预期表现,见 02-HUMAN-UAT.md)
overrides_applied: 0
human_verification_resolved: 2026-10-18(3/3 通过,裁定与目检记录见 02-HUMAN-UAT.md)
human_verification:
  - test: "soft_l1 鲁棒核严格线裁定:核对 run_20261018_022047(与 034000 逐位一致)中 m1__scipy_softl1/result.json 的 solver_meta(nfev=300、尾部改进 0、回代 RMS 0.2491 超 linear 参照 5.8% 被 5% 守门拒绝)与 m2__scipy_softl1(result.json:RMS 0.09213935 比纯 OpenCV 高 2.77e-4 即 +0.30%),在三选一中给出裁定:(a) 接受现状——成功标准 3 由 linear 精化路径兑现,soft_l1 偏差记为'鲁棒核优化另一目标的结构性预期';(b) 调 config/calibration.yaml 的 scipy 段(f_scale/max_nfev)后重跑 soft_l1 再判;(c) 维持严格线判不达标并安排返工。"
    expected: "一条明确的裁定结论(任一走向均可),02-05/02-06 SUMMARY 已按预案把证据全部落盘,裁定本身不需要改代码即可走 (a)。"
    why_human: "这是对'成功标准 3 的严格线要不要放宽'的价值判断,计划 02-05 明文预案'提交人工裁定',验证器无权代替开发者接受或拒绝偏差。"
  - test: "人工目检 output/calibration/run_20261018_034000/suspect_review/suspect_vs_normal.png(69KB,matplotlib 双联图:直方图叠加 + 累积分布),确认两组分布可辨、图内标注为 ASCII、与 JSON 数字方向一致(存疑中位 0.099px vs 正常 0.086px)。"
    expected: "图与 JSON 结论一致、可读、无渲染异常。"
    why_human: "图像外观与可读性无法用命令行程序化判定。"
  - test: "在终端跑一次 uv run calib calibrate --help 并目检任意一轮 run 的中文汇总表排版(组合/视图数/整体RMS/收敛四列、按模型分组并排、三条白话注解),确认可读且注解语义正确。"
    expected: "表格对齐、中文不乱码、注解讲清'同模型内比较'与'soft_l1 实用收敛'。"
    why_human: "终端排版效果属人工目检范畴。"
---

# Phase 2: 模型层 M1–M4 + 双求解后端 + 产物血缘 — 验证报告

**阶段目标(from ROADMAP):** models/ 成为投影/雅可比的唯一数学源(M1–M4 + 注册机制),双求解后端(OpenCV 初始化 → scipy 精化)可互换且结果同构,阶段间以带血缘的落盘产物通信——理论界与实际求解从此严格可比
**验证时间:** 2026-10-18T04:44:46Z
**状态:** human_needed(5 条成功标准中 4 条全部实证通过;第 3 条的精化主线严格通过、鲁棒核严格线按计划明文预案提交人工裁定)
**是否复验:** 否(首次验证)

## 目标达成情况

### 可观测真值(ROADMAP 5 条成功标准,SC3 拆两条)

| # | 真值 | 状态 | 证据 |
| --- | --- | --- | --- |
| 1 | SC1:M1/M2/M3 与 cv2.projectPoints、M4 与 cv2.fisheye.projectPoints 一致性比对通过;数值雅可比对每模型每参数通过 | ✓ VERIFIED | `tests/test_models_parity.py`(37 用例,float64 断言线 1e-9px、实测约 5e-13,含 float32 sanity 1e-3、适配器往返、反投影往返 <1e-9px);`tests/test_models_jacobian.py`(按 8+9+12+8=37 个"模型×参数"组合逐一参数化,numdifftools 基准列相对误差 <1e-6)。两文件在本次验证实跑中通过;`grep -rn numdifftools src/` 零命中(D-10 静态锁) |
| 2 | SC2:双后端对同一 ObservationSet × 任一注册模型求解,返回同一 CalibrationResult;scipy 残差仅由 models.project 构成 | ✓ VERIFIED | 同构断言 `test_result_structure_isomorphic_across_backends`(tests/test_solve_solvers.py:579,dataclasses.fields + 同一把回代尺);D-08 行为锁(tests/test_solve_scipy_residuals.py:286-312,monkeypatch project 加 0.25px → 残差 u 行逐项平移、v 行不动、还原后逐位复原);D-08/D-10 源码静态锁(同文件 :323/:334);实查 `grep projectPoints src/calib/solve/{residuals,scipy_solver,scoring}.py` 零命中;result.py:148-150 回代只经 model.project + transform_to_camera |
| 3 | SC3 主线:真实数据上 OpenCV 初始化 → scipy 精化在至少 M1 与 M2 上收敛,重投影误差不劣于纯 OpenCV | ✓ VERIFIED | run_20261018_022047 与 run_20261018_034000 两轮独立全矩阵(每轮 12 组合、comparison.json 12 行,两轮数字逐位一致):m1__scipy = 0.23551719828213488 与 m1__opencv **逐位相等**;m2__scipy = 0.09186284584467135 **优于** m2__opencv 0.09186292401766323(−7.8e-8);m3 −3.1e-11;m4 逐位相等 |
| 4 | SC3 鲁棒核严格线:M1/M2 的 soft_l1 路径按严格判据(收敛且 RMS ≤ 同模型纯 OpenCV + 1e-6)达标 | ? UNCERTAIN | **按 02-05 计划明文预案提交人工裁定。** 实测:m1__scipy_softl1 未收敛(converged=False,status=0、nfev=300、尾部改进 0.0、回代 RMS 0.2491 超 linear 参照 5.8% 被 5% 守门拒绝;失败契约完整:数值键全 null、中文 failure_note、判定依据全量在 solver_meta);m2__scipy_softl1 实用收敛但线性尺 RMS 0.09213934828836506 比纯 OpenCV 高 2.77e-4(+0.30%),超 +1e-6 严格线。证据与"linear 路径证据"均已落盘并写进 02-05 SUMMARY,与计划预案逐字吻合。软核优化的是另一条目标,线性尺略差属结构性预期还是需返工,属人的裁定 |
| 5 | SC4:每阶段产物带血缘指纹(配置哈希 + 上游产物哈希 + 时间戳),任一阶段可在新进程独立重跑 | ✓ VERIFIED | 12/12 个 result.json 齐备 created/config_sha256/upstream 三要素;scipy 组合血缘带第二环 init_result.json;实算 sha256 与血缘记录一致(角点 npz 与初值 result.json 各一项);integration 测试(tests/test_calibrate_matrix.py)在临时树两轮独立进程跑 m4 全路径,RMS 与逐参数差 <1e-12(本次验证实跑通过);本人独立复核:两轮真实 run 的 m2 params/rms 逐位相等、extrinsics.npz **逐字节相同** |
| 6 | SC5:测试中新增玩具模型仅一处注册即被求解器使用(零侵入) | ✓ VERIFIED | tests/test_models_registry.py:类头一行 @register_model(:50)后 create_model("toy_pinhole") 可查(:143-149),solve_with_opencv 走 generic_pnp 路径与 solve_with_scipy 双后端均收敛(:172/:187);源码静态断言求解器不 import 具体模型;注册表收尾复原断言。02-06 另证评分层零侵入(ToyScoringPinhole 经 solve_pose_only/score_views) |

**得分:** 5/6 条真值实证通过,1 条(SC3 鲁棒核严格线)按计划预案待人工裁定

### 计划级 must_haves 抽查(6 份 PLAN 前置声明,全部核对)

| 计划 | 关键真值抽查 | 结果 |
| --- | --- | --- |
| 02-01 | create_model 查表 m1–m4(8/9/12/8 参)、未知名中文报错含可选清单 | ✓ 实跑验证(报错含 'm1' 与 '可选') |
| 02-02 | 装载契约六防线(行序锁/剔除/未检出/NaN/视图下限/来源留痕) | ✓ 9 用例通过;source 键缺失防线在代码可查 |
| 02-03 | 配置矩阵白名单校验(method 拒 lm/losses/f_scale)、失败契约、通用初始化兜底 | ✓ test_io_config 与 test_solve_solvers 通过;opencv.py:64-78 适配器分派 + generic_pnp 兜底 |
| 02-04 | 实用收敛纯函数真值表、f_scale=auto(3×1.4826×MAD)、双数值雅可比基准 | ✓ 全部通过;run 产物 solver_meta 印证判定依据落盘 |
| 02-05 | calibrate 零参数全矩阵、失败组合不中断循环、阶段一 baseline.py 不改不 import | ✓ 12 组合落盘;matrix.py:181 兜底 except;git 实查 baseline.py 最后提交属阶段一(d40432f) |
| 02-06 | 复核覆盖 705 张(89+616)、走向模板、重跑与首轮一致 <1e-12、收尾三件 | ✓ suspect_vs_normal.json 实查 705 行、存疑 89、走向 B(偏差 15.9%<30%);m2 两路径与首轮逐位一致;pipeline.md ④/UAT 回写/REQUIREMENTS 表均已更新 |

### 必需产物(三级核对:存在 / 有实质 / 已接线)

| 产物 | 预期 | 状态 | 明细 |
| --- | --- | --- | --- |
| src/calib/models/base.py | CameraModel 抽象 + 外参导数工具 | ✓ VERIFIED | 192 行(≥80),抽象方法齐全,Rodrigues 雅可比版 transform_to_camera |
| src/calib/models/pinhole_brown.py | M1/M2/M3 + 牛顿逆解 + 适配器 | ✓ VERIFIED | 359 行(≥150),model_id="m1" 在案 |
| src/calib/models/kb4.py | M4 KB4 + θ 反解 + fisheye 适配器 | ✓ VERIFIED | 248 行(≥100),flags=10,无 skew |
| src/calib/models/__init__.py | 注册表 + create_model | ✓ VERIFIED | 66 行,底部 import 触发注册,四模型实查可查 |
| src/calib/solve/observation.py | ObservationSet + 装载契约 | ✓ VERIFIED | 235 行(≥80),allow_pickle=False、行序逐项断言在案 |
| src/calib/solve/result.py | 统一结果结构 + 同一把尺 + 三件套 | ✓ VERIFIED | 255 行(≥100),compute_result_metrics 只经 model.project |
| src/calib/solve/opencv.py | OpenCV 后端(适配器分派 + 兜底) | ✓ VERIFIED | 205 行(≥100),不 import 具体模型(测试锁定) |
| src/calib/solve/residuals.py | 残差/稀疏雅可比/结构矩阵 | ✓ VERIFIED | 214 行(≥100),无 projectPoints(静态锁) |
| src/calib/solve/scipy_solver.py | scipy 精化 + 实用收敛 | ✓ VERIFIED | 243 行(≥120),import residuals(:36) |
| src/calib/solve/matrix.py | 全矩阵编排 + 血缘 | ✓ VERIFIED | 309 行(≥80),combo_dir_name/make_lineage/兜底在案 |
| src/calib/solve/scoring.py | 位姿小求解 + 复核编排 | ✓ VERIFIED | 394 行(≥100),残差与初值只经模型接口 |
| src/calib/visualization/suspect_compare.py | matplotlib 对比图 | ✓ VERIFIED | 96 行,真实 PNG 69KB 已产出 |
| tests/(9 个关键文件) | 双重测试锁 | ✓ VERIFIED | 本次实跑 174 用例全过 |

### 关键连线核查

| 来自 | 去往 | 经由 | 状态 | 明细 |
| --- | --- | --- | --- | --- |
| models/__init__.py | pinhole_brown.py / kb4.py | 底部 import 触发注册 | ✓ WIRED | :65;create_model 四模型实查可用 |
| tests/test_models_parity.py | cv2.projectPoints / fisheye.projectPoints | 同参数同点集比对 | ✓ WIRED | :106/:113 断言 1e-9 |
| solve/opencv.py | models 适配器 | model.opencv_adapter() | ✓ WIRED | :64-78 按 kind 分派 |
| solve/opencv.py | observation.py | ObservationSet 入参 | ✓ WIRED | solve_with_opencv 签名 |
| solve/result.py | models.project | 回代同一把尺 | ✓ WIRED | :148-150(计划 pattern 写 create_model,实际模型实例化在 matrix.py,实质要求"残差只走 models.project"满足) |
| solve/scipy_solver.py | solve/residuals.py | 残差与雅可比唯一来源 | ✓ WIRED | :36 |
| solve/residuals.py | model.project | 残差构成 | ✓ WIRED | monkeypatch 行为锁 + 静态锁 |
| solve/matrix.py | util/lineage.py | make_lineage | ✓ WIRED | :44/:156 |
| solve/matrix.py | opencv 初值产物 | init_from 落盘读回 | ✓ WIRED | :211-216;真实产物 init_from="m2__opencv/result.json" |
| cli/main.py | load_observation_set | sb 角点装载 | ✓ WIRED | :699 |
| cli/main.py | review_suspects | calibrate 末尾挂接 | ✓ WIRED | :754,宽接不吞标定成功(:767) |
| tests/test_models_registry.py | 双求解器 | 玩具模型传入 | ✓ WIRED | :36-37/:172/:187 |

### 数据流核查(Level 4)

| 产物 | 数据变量 | 来源 | 真实数据 | 状态 |
| --- | --- | --- | --- | --- |
| run_20261018_022047 / 034000(12 组合 × 三件套) | params/rms/残差 | 真实 640 视图 sb 角点 | 是(RMS 0.0857–0.2355px,合理量级) | ✓ FLOWING |
| suspect_vs_normal.json | frame_table(705 行) | result.json 逐帧 RMS + test 帧位姿小求解 + manifest join | 是(存疑 89 = train 67 + test 22,与 manifest 对齐) | ✓ FLOWING |
| suspect_vs_normal.png | frame_table | 同一份 JSON | 是 | ✓ FLOWING |

### 行为抽查

| 行为 | 命令 | 结果 | 状态 |
| --- | --- | --- | --- |
| 快速全量测试 | uv run pytest -m "not integration" -q | 260 passed(16.65s) | ✓ PASS |
| 集成测试(真实数据 m4 两轮确定性 + 血缘) | uv run pytest -m integration -q | 5 passed(637s,<1e-12 一致性) | ✓ PASS |
| 9 个关键测试文件 | pytest tests/test_models_*.py tests/test_solve_*.py tests/test_calibrate_matrix.py tests/test_io_config.py -q | 174 passed(564s,含其内置 integration) | ✓ PASS |
| 注册表 + 中文报错 | python -c "create_model('m0')" | ValueError 含 'm1' 与 '可选' | ✓ PASS |
| calibrate 命令入口 | uv run calib calibrate --help | 矩阵化中文帮助正常输出 | ✓ PASS |
| 血缘指纹真伪 | 对 corners npz 与 init result.json 重算 sha256 | 与血缘记录一致 | ✓ PASS |
| 跨轮确定性(本人独立复核) | 对比两轮 run 的 m2 params/rms/extrinsics.npz | 逐位相等 / npz 逐字节相同 | ✓ PASS |

### 探针执行

无 `scripts/*/tests/probe-*.sh`,计划亦未声明探针。SKIPPED(无探针对象;等价的独立进程重跑已由 integration 测试与两轮真实 run 承担)。

### 需求覆盖(REQUIREMENTS.md 对照)

| 需求 | 来源计划 | 描述要点 | 状态 | 证据 |
| --- | --- | --- | --- | --- |
| MOD-01 | 02-01/02-03/02-04 | CameraModel 抽象 + 注册机制 + 单一数学源共享 | ✓ SATISFIED | models/base.py 192 行抽象;create_model 注册表;solve/scoring 全部只经模型接口(源码静态锁);评分与复核共用同一投影 |
| MOD-02 | 02-01 | 四类模型 M1–M4 + OpenCV 一致性比对 + 数值雅可比测试 | ✓ SATISFIED | pinhole_brown.py/kb4.py;parity 37 用例(1e-9px,实测 ~5e-13)+ 雅可比 37 组合(<1e-6) |
| SOL-01 | 02-02/02-03/02-04/02-05 | 统一接口 + 双后端(OpenCV 初始化 → scipy 精化含鲁棒核)、可互换、结果同构 | ✓ SATISFIED | 双后端 + CalibrationResult 同构断言 + 失败契约同构 + ScipyConfig.losses(linear/soft_l1)+ 实用收敛判定;真实数据双轮全矩阵 12 组合落盘 |
| ITER-01 | 02-05/02-06 | 阶段间以带血缘(配置指纹+上游哈希+时间戳)产物通信,可独立重跑 | ✓ SATISFIED | 12/12 血缘三要素;sha256 重算一致;新进程临时树两轮重跑 <1e-12;旧产物可对账(baseline.py 未动) |

**孤立需求核查:** REQUIREMENTS.md 把 MOD-01/MOD-02/SOL-01/ITER-01 四条映射到 Phase 2,四条在计划 frontmatter 的 requirements 字段中全部出现,无遗漏、无孤立。

**一处文档不一致(非门槛项):** REQUIREMENTS.md 第 30/31/35/61 行的 v1 清单复选框仍是 `- [ ]`,而底部 Traceability 表与 Coverage 统计(8/32)已正确标 Shipped (Phase 2)。属 02-06 收尾的 cosmetic 遗漏,建议随手补勾。

### 反模式扫描

| 文件 | 位置 | 模式 | 严重度 | 影响 |
| --- | --- | --- | --- | --- |
| src/calib/solve/observation.py | :155-160 | CR-01(02-REVIEW):配置放行 corner_route: classic,但装载强制要求 source 键而 classic 产物天生没有 → 该配置路径必然响亮报错 | ⚠ Warning | 默认 sb 路径与全部成功标准不走 classic,不阻挡任何真值;属配置层契约自相矛盾,修法明确(收紧白名单为 {sb} 或缺 source 时按路线补常量),建议阶段三开工前顺手修 |
| src/calib/solve/scipy_solver.py | :186-193 | WR-01:jac_sparsity 传解析雅可比时被 scipy 静默忽略(已实测),结构防线实际由共用行列排布承担 | ⚠ Warning | 白组装秒级浪费 + 防线名不副实;删传参或改 docstring 即可 |
| src/calib/models/pinhole_brown.py / kb4.py | :242-251 / :177-186 | WR-02:unproject 牛顿迭代无收敛/奇异性终检,病态参数可静默产出未收敛射线 | ⚠ Warning | 当前测试参数良性;外推诊断类下游消费前建议加终检(评审已给修法) |
| src/calib/solve/matrix.py | :208-211 | WR-03:血缘组装先于初值消费判定(记了没吃的);extrinsics.npz 被消费却未记指纹(吃了没记的) | ⚠ Warning | 本次实查确认时序属实;血缘三要素与重跑确定性不受影响,但阶段三拿产物当线性化点前应补齐 |
| src/calib/solve/scoring.py | :357-366 | WR-04:复核血缘漏记 test 角点 npz | ⚠ Warning | 同上,ITER-01 精神的完整性缺口 |
| src/calib/models/__init__.py | :65 | ruff 2 处遗留(noqa 指令非法 + F401×2),deferred-items.md 登记在案 | ℹ Info | 纯 cosmetic(原 5 处已修 3 处文件级),不阻挡阶段 |
| src/calib/cli/main.py | fisher/evaluate 命令 | _not_implemented 占位 | ℹ Info | 阶段一行走骨架的有意设计,属阶段三/四范围,非本阶段欠账 |

**债务标记扫描:** 本阶段全部产出文件 `grep TBD|FIXME|XXX` 零命中;`placeholder/not yet implemented` 零命中。

### 人工验证项

见 frontmatter `human_verification`(3 项:soft_l1 裁定 / 复核 PNG 目检 / 汇总表目检)。其中第 1 项是计划 02-05 明文预案的裁定入口,裁定走向 (a) 无需改任何代码。

### 差距综述

**没有发现阻挡目标达成的硬缺口。** 5 条成功标准中,4 条(SC1/SC2/SC4/SC5)以代码 + 测试 + 真实产物三层证据实证通过,且本次验证在 HEAD 上独立复跑了全部 265 条用例(260 快速 + 5 集成)与两轮真实 run 的关键断言(12 组合、血缘 12/12、sha256 重算、跨轮逐位一致),与 SUMMARY 宣称无一不符。

第 3 条标准拆开看:**精化主线已严格达标**(m1 逐位不劣、m2 严格更优,m3/m4 同样不劣);**鲁棒核(soft_l1)未达严格线**——m1 未在 300 次评估内收敛并被 5% 守门如实拒绝,m2 收敛但线性尺 RMS 高 0.30%。02-05 计划在 must_haves 里**事先写明了这一情形的处置预案**(发现落盘 + linear 证据 + 提交人工裁定),执行与预案逐字吻合,证据链完整可审计。因此该项不是"实现缺失"而是"待裁定的偏差":判定 soft_l1 的严格线是否放宽(或调 f_scale/评估上限后重跑),是开发者的价值判断,验证器按升级门规则移交人工。

代码评审的 1 个 Critical(CR-01)经核对**不摧毁任何成功标准**:它位于非默认的 classic 角点路线配置承诺与装载防线的矛盾处,全部成功标准的证据路径(sb 路线)不受影响,属可明确修复的范围外缺口,已列为 Warning 并给出修法。WR-03/WR-04 两处血缘完整性缺口建议在阶段三消费这些产物之前补齐。

---

_验证: 2026-10-18T04:44:46Z_
_验证人: Claude (gsd-verifier)_
