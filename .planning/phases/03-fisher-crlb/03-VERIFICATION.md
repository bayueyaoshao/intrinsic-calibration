---
phase: 03-fisher-crlb
verified: 2026-10-19T05:30:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification: null
gaps: null
deferred:
  - truth: "实测误差 vs CRLB 的实证校验(CRLB 下界是否被真实标定逼近)"
    addressed_in: "Phase 4"
    evidence: "deferred-items.md 与 03-02-SUMMARY 边界注记:蒙特卡洛是合成数据上的数学单测,真实数据实证校验属阶段四 EVA 需求;本阶段 crlb.json 是其理论锚点"
  - truth: "m3/m4 高阶畸变项定不住(条件数退化真信号)的处置"
    addressed_in: "Phase 4"
    evidence: "deferred-items.md:该发现是阶段四模型选型防过拟合的输入;本阶段职责止于上报退化方向"
human_verification: []
human_verification_resolved: 2026-10-19(03-06 检查点 4 项由用户看图/跑命令核实通过,记录见 03-06-SUMMARY.md「Task 3 人工核查记录」)
---

# Phase 3: 噪声底 + Fisher/CRLB 可达精度预测 — 验证报告

**Phase Goal(from ROADMAP):** 标定前即可回答"这批数据天生最准能到多少":时域噪声底(8 连拍实测,缺失退化 Immerkaer)、像素噪声→角点误差传递曲线、每内参 CRLB 方差下界与位姿退化检测(外参 Schur 边际化)
**Verified:** 2026-10-19T05:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## 核心验证方式说明

本验证不采信任何 SUMMARY 陈述,全部证据来自:①直接读真实产物文件并用 python 逐项对账数字;②实算血缘哈希(18/18 闭合);③实跑测试套件(350 通过)与 ruff(全绿);④源码接线逐条 grep。验证过程只读,未修改任何源码与产物。

## Goal Achievement

### Observable Truths(ROADMAP 5 条成功标准逐条对账)

| # | Truth | Status | Evidence(本次实读产物与实测数字) |
| --- | --- | --- | --- |
| 1 | `calib noise` 对真实数据产出 σ,来源逐条标注(时域/单帧折算),重复数 <2 自动单帧估计;报告区分 detector 噪声与位姿抖动 | ✓ VERIFIED | `output/noise/run_20261018_165412/noise.json`:顶层契约 14 键齐(schema_version/lineage/双口径 σ/来源计数/outlier_poses/splits/transfer_curve/injection/meta/failure_note);sigma_detector_corner=0.008708、sigma_temporal_corner=0.299471(像素/轴,时域 ≥ 检测器,合理性成立);train 80/80 姿态 source=temporal、n_poses_fallback=0(数据本 全部 8 连拍,数据驱动非写死);test 65/65 逐张 source=immerkaer_transfer 且 sigma_corner_per_axis 全为有限正值;抖动分离在 `temporal_sigma_per_corner.npz`:sigma_per_corner_all (80,88,2)、sigma_jitter、clamped 数组逐姿态在案(本轮 80 姿态均未截断,jitter 中位 0.0335);单帧退化分支由 `tests/test_noise_temporal.py` 单测锁定(检出帧 <2 → source=immerkaer_transfer + 假曲线命中);重复数 <2 路径在真实数据上以 test 65 张独立照整体体现 |
| 2 | 受控注入实验产出"像素噪声 → 角点定位误差"经验传递曲线(数据落盘) | ✓ VERIFIED | `injection_displacements.npz` 形状 (640, 9, 4, 88, 2) float64,σ=0 参照档最大绝对位移**恰为 0.0**(基线对齐校准),NaN 槽位 12320/4055040(0.3%,失败留痕非静默);`transfer_curve.json` 9 档(σ=0 校准行 role=reference + 8 档对数网格 0.25→23.72),每档五键齐(rms_euclidean/scatter_low/scatter_high/failure_rate/fallback_rate),rms 单调上升 0→0.1518,最高档失败率抬头至 2.73%(检测器失效点证据);`transfer_curve.png` 在盘;种子三元组复现性由单测锁定(同种子逐位一致) |
| 3 | `calib fisher` 产出每内参 CRLB σ 与特征谱/条件数,能标出耦合退化方向;外参经 Schur 消元 | ✓ VERIFIED | `output/fisher/run_20261019_025859/crlb.json`:16/16 份记录逐份可寻址(4 模型 × views640/poses80 × detector/temporal),每份含 se/eigvals/condition_number/top_directions/flag/max_pose_block_cond;条件数(0a9b247 无量纲化修复后):m1 1.0e3、m2 8.85e3 正常,m3 5.72e11、m4 1.17e6 判 degenerate——极端方向整块落在高阶畸变项(m3: k5/k2/k6;m4: k3/k4/k2),`normalization` 字段中文写明无量纲化口径,`condition_number_raw_units` 留原始单位对照;Schur 消元:`fisher.py:108 marginalized_fisher` 分块消元,恒等式测试实测 2.4e-12,`fisher_mX.npz` 的 F_marg 仅 (P,P) 内参块(8/9/12/8),外参已消去 |
| 4 | FIM 推导的内参 SE 与 OpenCV stdDeviationsIntrinsics 一致到个位数百分比(最强交叉验证) | ✓ VERIFIED | crlb.json 的 opencv_crosscheck 段:m1 max_rel_dev=5.36e-05(0.0054%)、m2=5.52e-05(0.0056%)——远优于"个位数百分比"硬门槛(协议按尺度归一结构比对,规避 OpenCV σ̂ 尺度因子坑,scale_k=0.169/0.068);m3=0.239% 尽力档记录;M4=unavailable_fisheye 带中文结论;线性化点换点检查 rel_diff 全部 ≤0.369%(D-06 预期 <1%) |
| 5 | Σ 不确定性(~27% 相对误差)以敏感性方式随 CRLB 落盘,不输出裸下界 | ✓ VERIFIED | 16 份记录逐份实算验证 band_low == se×(1−rel) 且 band_high == se×(1+rel)(rel 从记录 band_rel 读),**16/16 全部成立**;`fisher.py:57 SIGMA_BAND_REL_8 = 1/√14` 常量注释写明 8 样本相对标准误来历;`config/calibration.yaml` sigma_band_rel: 0.267 带白话注释;80 姿态版 σ_used = 640 版/√8(0.008708→0.003079、0.2995→0.1059,实算吻合,iid 折减假设落盘) |

**Score:** 5/5 truths verified

### 附加对账:计划级 must_haves 抽查(6 份 PLAN,全部核对)

| 计划 | 关键 must_have 抽查 | 结果 |
| --- | --- | --- |
| 03-01 | Immerkaer 双估计、注入逐张逐档逐重复落盘、五键传递曲线、参数只从 config 读 | ✓ immerkaer_sigma 在 `src/calib/noise/immerkaer.py:38`;displacements 全量在盘;PchipInterpolator 在 `transfer.py:21`;`grep injection_n_levels` 仅 config.py 与 yaml;A2 常数回收测试在 test_noise_immerkaer.py |
| 03-02 | Schur 恒等式、σ 线性缩放、块对角、无 pinv、蒙特卡洛、numdifftools 只进 tests | ✓ test_noise_fisher.py(27 用例,含 `test_schur_marginalization_matches_direct_inversion` 与无量纲化修复测试)在 350 通过内;`grep pinv/projectPoints src/calib/noise/fisher.py` = 0;`grep -r numdifftools src/` = 0;tests 里 `import numdifftools as nd` 在案 |
| 03-03 | noise.json 契约、失败契约工厂、CLI 填充、血缘 | ✓ `temporal.py:98 temporal_noise`、`artifacts.py:91 save_noise_report`、`main.py:667 noise()`;库层零 typer import(grep 仅注释提及);make_failure_report 在 artifacts.py:40 |
| 03-04 | 8 元组解包、compare_structural、M4 分支、无 CALIB 字面量 | ✓ `crosscheck.py:78/157/234` 三函数在;`grep CALIB_ crosscheck.py` = 0;M4 unavailable_fisheye 在真实 crlb.json 体现 |
| 03-05 | FisherConfig、旋转均值红线、16 份编排、失败不中断、crlb.json 契约 | ✓ `config.py:240 FisherConfig`、`fisher.py:404 Rotation.from_rotvec(rs).mean()`(逐分量平均禁令有注释与测试)、`fisher.py:663 run_fisher_matrix`、`artifacts.py:236 save_crlb_report`、`main.py:1000 fisher()` |
| 03-06 | 五步接力、run-all 接线、pipeline.md 同步、REQUIREMENTS 更新、人工核查 | ✓ `test_smoke_realdata.py:123` 五步序列;`main.py:1183 run_all()` 依次调 ingest→corners→calibrate→noise→fisher;docs/pipeline.md 第③⑤步契约与实际产物字段逐一对应(实读比对);REQUIREMENTS.md NOI-01/02/03 已勾选并标注完成计划号 |

### Deferred Items(后续阶段承接,不构成本阶段缺口)

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | 实测误差 vs CRLB 的实证校验 | Phase 4 | deferred-items.md;03-02-SUMMARY 边界注记(蒙特卡洛为合成数学单测,不替代真实数据实证) |
| 2 | m3/m4 高阶畸变项退化的处置(选型防过拟合) | Phase 4 | deferred-items.md:该发现明确标注为"阶段四选型防过拟合的输入" |

### Required Artifacts(三级核对:存在 / 有实质 / 已接线)

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `src/calib/noise/immerkaer.py`(72 行) | Immerkaer 单帧双估计 | ✓ VERIFIED | `immerkaer_sigma` 在:38;被 injection/temporal/cli 三处消费 |
| `src/calib/noise/injection.py`(344 行) | 受控注入实验 | ✓ VERIFIED | `run_injection_experiment` 在:153;import detect_sb/detect_classic(零检测逻辑复制,grep findChessboardCorners=0);真实产物 (640,9,4,88,2) 在盘 |
| `src/calib/noise/transfer.py`(138 行) | 传递曲线 + PCHIP + √2 换算 | ✓ VERIFIED | PchipInterpolator 在:21,114;interp_corner_sigma 被 temporal.py:30 import 并 4 处调用 |
| `src/calib/noise/temporal.py`(263 行) | 时域噪声底 + 来源标注 + 抖动分离 | ✓ VERIFIED | `temporal_noise` 在:98;CLI noise() 调用;真实明细 npz 在盘 |
| `src/calib/noise/artifacts.py`(314 行) | noise.json/crlb.json 契约 + 明细 npz + 血缘旁车 | ✓ VERIFIED | save/load 双向、allow_pickle=False 全程;fisher_m1.meta.json 实读含 lineage 15 上游 |
| `src/calib/noise/fisher.py`(903 行) | FIM/Schur/CRLB/谱/16 份编排 | ✓ VERIFIED | 全部函数在案(见 must_haves 抽查);J 只从 make_jacobian_fn(:46) |
| `src/calib/noise/crosscheck.py`(330 行) | stdDev 交叉验证协议 | ✓ VERIFIED | 三函数在案;flags 全走 opencv_adapter(:100);真实 m1/m2 比对 0.005% 级 |
| `src/calib/visualization/noise_figs.py`(128 行) | 曲线图 + 谱图 | ✓ VERIFIED | plot_transfer_curve(:26)/plot_spectrum(:73);16 张谱图 + 曲线图真实落盘 |
| `src/calib/io/config.py` NoiseConfig/FisherConfig | 参数只从 config 读 | ✓ VERIFIED | :203/:240 两配置类 + _parse_noise(:568)/_parse_fisher(:652);yaml 两段每键白话注释(实读) |
| `src/calib/cli/main.py` noise()/fisher()/run_all() | 命令闭环 | ✓ VERIFIED | :667/:1000/:1183;接线全部核实(见关键链路表) |
| 真实产物 3 组 | noise/fisher/calibration | ✓ VERIFIED | run_20261018_165412(8 文件)、run_20261019_025859(crlb.json+4 npz+16 图)、run_20261018_163713(4 模型 scipy 线性化点全在) |

### Key Link Verification(接线)

| From | To | Via | Status | Details |
| ---- | --- | --- | ------ | ------- |
| injection.py | corners/detect.py | `from calib.corners.detect import detect_classic, detect_sb`(:28) | WIRED | 零第二份检测实现 |
| injection.py / temporal.py / artifacts.py | corners npz 读回 | allow_pickle=False | WIRED | 4 文件逐处 grep 命中 |
| transfer.py | scipy.interpolate | PchipInterpolator | WIRED | :21 import、:114 实例化 |
| fisher.py | solve/residuals.py | make_jacobian_fn + pack_theta(:46) | WIRED | D-10 唯一数学源,静态锁测试在 |
| fisher.py | scipy Rotation | Rotation.from_rotvec().mean()(:404) | WIRED | 逐分量平均禁令有红线注释 + 测试 |
| fisher.py | crosscheck.py | crosscheck_model(:592-594) | WIRED | 逐模型调用进编排 |
| temporal.py | transfer.py | interp_corner_sigma(:30,:172,:192,:219) | WIRED | 单帧折算与抖动分离都过传递曲线 |
| cli/main.py | noise.json 上游 | "noise.json" 键(:1036) | WIRED | 血缘指纹实算闭合 |
| cli/main.py | mX__scipy 线性化点 | m{X}__scipy/extrinsics.npz(:1041) | WIRED | 4 模型产物全在,run_20261018_163713 |
| cli/main.py | run_injection_experiment / save_noise_report / run_fisher_matrix / save_crlb_report | import + 调用 | WIRED | :40/:36/:38/:35 |
| crosscheck.py | opencv_adapter | 唯一入口(:100) | WIRED | grep CALIB_ 字面量 = 0 |
| run_all | 五步接力 | ingest→corners→calibrate→noise→fisher | WIRED | :1195-1205 逐步调用,终点 crlb.json |
| test_noise_fisher.py | numdifftools | import nd(:24) | WIRED | src/ 全树零命中(静态锁) |
| crlb.json → noise.json → corners npz | 血缘链 | make_lineage sha256 | WIRED | **18/18 上游实算哈希逐项闭合**(corners run_20261018_163039 → noise run_20261018_165412 → fisher run_20261019_025859,线性化点 run_20261018_163713) |

### Data-Flow Trace(Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| noise.json | 双口径 σ | 640 张真实图注入实验 + 80 姿态连拍波动 | 是(位移数组实测值,NaN 仅 0.3% 失败槽) | ✓ FLOWING |
| transfer_curve.json | 每档 rms | injection_displacements.npz 聚合 | 是(σ=0 档恰 0、单调上升、失效点上翘) | ✓ FLOWING |
| crlb.json | 16 份 se/带/谱 | corners npz + noise.json σ + mX__scipy 线性化点 | 是(血缘 15 上游哈希闭合) | ✓ FLOWING |
| 谱图/曲线图 | matplotlib 输入 | crlb.json / transfer_curve.json | 是(16 张 + 1 张在盘) | ✓ FLOWING |

无空props、无静态兜底数据、无硬编码空数组流入渲染或产物。

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| 全量非集成测试 | `uv run pytest -m "not integration" -q` | 350 passed, 19 deselected(39.5s) | ✓ PASS |
| 阶段三专属测试 | `uv run pytest tests/test_noise_*.py tests/test_fisher_realdata.py tests/test_io_config.py -m "not integration" -q`(fisher_realdata 除外) | 115 passed | ✓ PASS |
| Lint | `uv run ruff check src tests` | All checks passed | ✓ PASS |
| CLI 注册 | `uv run calib --help` | 九命令齐,③noise/⑤fisher 中文 help 在位 | ✓ PASS |
| Σ 带恒等式(16 份) | python 逐份实算 band_low/high == se×(1∓rel) | 16/16 成立 | ✓ PASS |
| 血缘闭合(18 上游) | sha256 逐项实算比对 | 18/18 OK,0 mismatch | ✓ PASS |
| σ/√8 折减 | 0.008708/√8 = 0.003079 实算 | 与 poses80 sigma_used 吻合 | ✓ PASS |
| JSON NaN 泄漏 | grep NaN/Infinity 两份 json | 0 命中(评审 WR-01 担忧在真实产物未发生) | ✓ PASS |
| 集成测试重跑 | `pytest -m integration` | ? SKIP(噪声链 77 分钟级;产物 + 血缘哈希 + 03-06 五步接力冒烟已提供等效证据) | ? SKIP |

### Probe Execution

无声明探测脚本、无 scripts/*/tests/probe-*.sh 约定路径 — SKIPPED(不适用)。

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| NOI-01 | 03-01, 03-03, 03-06 | 时域噪声底 + Immerkaer 单帧退化,来源标注,<2 重复自动退化 | ✓ SATISFIED | noise.json 双口径 + 80/80 姿态标注 + test 65 张全折算 + 抖动分离明细 npz |
| NOI-02 | 03-01, 03-03, 03-06 | 像素噪声→角点误差映射(受控注入 + 经验传递曲线) | ✓ SATISFIED | injection_displacements.npz (640,9,4,88,2) + transfer_curve.json 九档五键 + σ=0 校准档恰零 |
| NOI-03 | 03-02, 03-04, 03-05, 03-06 | Fisher/CRLB 下界 + 位姿退化检测 + Schur 边际化 + 交叉验证 | ✓ SATISFIED | crlb.json 16/16 记录 + 谱/条件数/软方向 + F_marg 内参块 + m1/m2 交叉验证 0.005% 级 |

**ORPHANED 需求:无。** REQUIREMENTS.md 中 Phase 3 恰好映射 NOI-01/02/03 三条,全部被计划认领、全部标记 Shipped 并带完成计划号;追溯表(118-120 行)与页脚计数(11/32)同步。

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| (阶段全部改动文件) | — | TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER/占位待实现 | — | **零命中**(逐文件 grep 实扫) |
| 03-REVIEW.md 登记项 | — | 10 warning(鲁棒性边界与契约一致性,如 WR-01 NaN 掩码缺失、WR-03 noise_splits 配置被无视、WR-05 yaml 写死 8 worker)+ 7 info | ℹ️ Info | 0 critical;全部为建议性未修,不影响任何成功标准在真实产物上的成立(真实 crlb.json/noise.json 实读零 NaN、默认配置下产物自洽);属后续收尾清单,不构成本阶段缺口 |

### Human Verification Required

**无新增待人工核查项。** 03-06 检查点 4 项已于 2026-10-19 由用户看图/跑命令核实通过(记录在 03-06-SUMMARY.md「Task 3 人工核查记录」),本验证沿用其结论,不重复请求:

1. 传递曲线图平滑单调、σ=0 档为零、高 σ 段失败率抬头 — **已通过**(用户目检 `output/noise/run_20261018_165412/transfer_curve.png`;本次实读 json 数字与图一致:rms 单调 0→0.1518、最高档失败率 2.73%)
2. noise.json 双口径 σ 数值合理 — **已通过**(用户逐项读数;本次实读 detector 0.008708 / temporal 0.299471,时域 ≥ 检测器)
3. 交叉验证数字在门槛内 — **已通过**(用户读数 run_20261018_145407;本次实读修复后重跑轮 run_20261019_025859 的 max_rel_dev 与其一致:m1 5.36e-05 / m2 5.52e-05)
4. 16 张特征谱图与条件数分档 — **已通过**(用户在 0a9b247 修复后对 run_20261019_025859 图与数字复检;本次实读 16 张图在盘、m1/m2 正常、m3/m4 退化为真信号)

上述之外,五条成功标准的所有判据均已由产物数字 + 测试 + 血缘哈希自动化覆盖,无剩余仅能人眼判断的项。

### Gaps Summary

无缺口。三条成功标准(实际 5 条)全部在真实产物上成立:噪声侧双口径 σ、逐姿态来源标注、抖动分离与全保真明细齐备;注入实验位移数据逐张逐档逐重复落盘且 σ=0 校准档恰零;fisher 侧 16 份 CRLB 记录并排、Σ 敏感性带逐份恒等、无量纲化修复后的条件数分档给出真信号(m3/m4 高阶畸变定不住)、M1/M2 交叉验证对齐到 0.005% 量级。血缘链 18/18 哈希实算闭合,测试 350 通过、ruff 全绿,源码静态红线(无 pinv/无第二份数学/无 numdifftools 进 src/库层无 typer)全部成立。执行期发现并修复的条件数单位伪影(0a9b247)已通过产物(normalization 字段 + condition_number_raw_units 对照 + 修复测试)与用户复检双证据闭环。

---

_Verified: 2026-10-19T05:30:00Z_
_Verifier: Claude (gsd-verifier)_
