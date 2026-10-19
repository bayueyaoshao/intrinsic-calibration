---
phase: 03-fisher-crlb
plan: 05
subsystem: noise
tags: [crlb, fisher-information, orchestration, crlb-json, cli, integration-test]

# Dependency graph
requires:
  - phase: 03-fisher-crlb/03-02
    provides: marginalized_fisher / crlb_se / sensitivity_band / spectrum_report(σ=1 基准数学核心)
  - phase: 03-fisher-crlb/03-04
    provides: crosscheck_model(OpenCV stdDev 交叉验证协议,D-07 分级判读)
  - phase: 03-fisher-crlb/03-03
    provides: noise.json 双口径 σ(σ_detector_corner / σ_temporal_corner,T-03-05-01 的取数入口)
  - phase: 02-m1-m4
    provides: m1–m__scipy 三件套(线性化点,D-06)与 ObservationSet 行序契约
provides:
  - aggregate_observations + build_aggregated_observation_set:80 姿态聚合观测集(逐点均值 + 弦长旋转均值 + σ/√n 折减,D-01/D-02)
  - run_fisher_matrix:16 份 CRLB 矩阵编排(2 观测集 × 2 σ 口径 × 4 模型),单组合失败转契约记录不中断
  - crlb.json 产物契约(CRLB_KEYS 六键 + 逐模型 fisher_mX.npz 明细旁车,inf 直通 json)
  - `calib fisher` 命令闭环(上游最新轮定位 + 全血缘 + 特征谱图 + 中文汇总表)
  - tests/test_fisher_realdata.py:成功标准 3/4/5 在真实数据上的对账(集成,7 用例)
affects: [04-evaluation 选型(CRLB 带是"逼近数据极限"的判据基线), 06-analyze 误差归因, phase-3 收口]

# Tech tracking
tech-stack:
  added: []   # 零新增依赖(scipy Rotation / numpy / matplotlib 既有锁定版本)
  patterns:
    - "库层不依赖命令行框架:进度回调由 CLI 注入(run_fisher_matrix 的 progress 参数),静态测试锁死"
    - "16 份记录逐份可寻址:observation_sets[views640|poses80][detector|temporal],失败也结构键全(CRLB_RECORD_KEYS 17 键)"
    - "明细走 npz 旁车、json 只留路径引用(D-13):_fisher_detail 私有键由 save_crlb_report 取出转 fisher_mX.npz"
    - "零信息参数的 CRLB 记 inf(有内容的结论,不是错误):Python json Infinity 约定,本项目产物由本项目读回"
    - "确定性计算可复用:crlb.json 血缘指纹(corners npz + noise.json 的 sha256)对得上就直接复用历史轮,不重跑确定性计算"

key-files:
  created:
    - tests/test_fisher_realdata.py
  modified:
    - src/calib/noise/fisher.py
    - src/calib/noise/artifacts.py
    - src/calib/cli/main.py
    - src/calib/visualization/noise_figs.py
    - src/calib/io/config.py
    - config/calibration.yaml
    - tests/test_noise_fisher.py
    - tests/test_io_config.py

key-decisions:
  - "线性化点只认 m{X}__scipy linear 收敛解(D-06):softl1/opencv/未收敛解一律中文报错拒绝,三道对齐断言(收敛、extrinsics 行序、观测集逐行)堵串轮"
  - "80 姿态聚合版的 σ 折减按 σ/√n(iid 假设)并显式落盘 iid_note:640/80 两版差值 = 连拍相关性损耗,阶段四的重复张分组实验是它的实测对照(D-01 设计意图)"
  - "交叉验证参数对账(T-03-05-02)放在编排内:现场重解的 OpenCV 参数与已存 m{X}__opencv 产物超 float32 容差即记 status=failed(中文说明),CRLB 记录本身不受牵连——对不上账的模型不许有'验证通过'的假结论"
  - "inf 经 Python json Infinity 约定直通(读回仍是 inf),不转字符串不转 null:零信息是可读结论,上一波语义(b6225dc)的延续"
  - "特征谱图/血缘谱图归附加产物层:单张图失败只提示不拦主产物(照存疑帧复核的兜底体例)"

patterns-established:
  - "crlb.json 寻址体例:models.mX.observation_sets.<obs>.<sigma> 下 17 键,下游先验键再取数,失败轮同构"
  - "集成测试复用规则:crlb.json 生成是确定性的,血缘指纹对得上的完整轮直接复用(5.5 分钟 × 每次重跑不划算);血缘缺账的旧轮判不完整自动触发全新一轮"

requirements-completed: [NOI-03]  # CRLB 预测全链在真实数据上闭环(成功标准 3/4/5 对账全绿)

# Metrics
duration: 57min
completed: 2026-10-18
---

# Phase 3 Plan 5: 16 份 CRLB 矩阵编排与 `calib fisher` 命令闭环 Summary

**真实数据 640 视图上 16/16 份 CRLB 记录全算成:M1/M2 与 OpenCV stdDev 的结构比对 max_rel_dev ≈ 0.005%(硬门槛 10% 判 pass),Σ ±26.7% 敏感性带逐份恒等式成立,四模型条件数全部超退化线(软方向 fx/fy,摆拍旋转多样性不足的实测发现)——NOI-03 闭环,Phase 3 三需求产物链齐备。**

## Performance

- **Duration:** 57 min
- **Started:** 2026-10-18T14:07:38Z
- **Completed:** 2026-10-18T15:04:39Z
- **Tasks:** 3/3(RED 先行的 TDD 体例:聚合/FisherConfig 测试先行)
- **Files modified:** 8(4 源码 + 2 测试 + 配置 + yaml)

## Accomplishments

- **Task 1(9b458ed RED → 51733f8 GREEN)**:`FisherConfig`(阈值有序性/带半宽范围/布尔类型校验,含 yaml 指数必须带符号的写法注释)+ `aggregate_observations`(逐点均值只吃检出帧、来源留痕、弦长旋转均值红线锁定)+ `build_aggregated_observation_set`(未检出姿态中文报错)。
- **Task 2(22ddf99)**:`run_fisher_matrix` 16 份编排(单组合失败转契约不中断)、crlb.json 契约与逐模型 Fisher 明细 npz、`calib fisher` CLI(全血缘 + 失败也落盘失败契约 + 中文汇总表)、`plot_spectrum` 特征谱图;迷你矩阵/故障隔离/参数对账/静态红线测试锁死。
- **Task 3(35e5f8f)**:真实数据试车(独立进程 `calib fisher`,325 秒)+ 7 用例对账全部通过。

## 真实数据实测(成功标准 3/4/5 对账,run_20261018_145407)

| 模型 | 条件数 | 分档 | CRLB 中位数(detector 口径,px) | 交叉验证 max_rel_dev | 判定 |
| ---- | ------ | ---- | ------------------------------ | -------------------- | ---- |
| m1 | 2.70e+09 | degenerate | 3.3e-03 | 5.36e-05 | pass(<10%) |
| m2 | 2.81e+09 | degenerate | <1e-04 | 5.52e-05 | pass(<10%) |
| m3 | 5.68e+11 | degenerate | 1.37e-02 | 2.39e-03 | 尽力档记录 |
| m4 | 1.07e+08 | degenerate | 2.3e-03 | unavailable_fisheye | 记结论 |

- 双口径 σ:detector 0.0087 / temporal 0.2995(时域含位姿抖动)。
- 线性化点换点检查 rel_diff 全部 ≤ 0.37%(D-06 预期 <1%,无一超限)。
- **上报发现**:四模型条件数全部超过退化线 1e6,最软方向集中在 fx/fy
  (m3 集中在 k5/k2/k6)——这批摆拍的部分参数组合方向上精度先天受限,
  典型原因是姿态旋转多样性不足;进阶段四选型与摆拍改进建议,不阻塞。
- D-01 对照成立:views640 与 poses80 同口径 CRLB 逐份存在且不相等(两版
  差值 = 连拍相关性损耗,可计算,阶段四对照组)。

## 更正(2026-10-19,用户核查发现)

上表"四模型条件数全部超退化线、软方向 fx/fy、归因摆拍旋转多样性不足"
是**单位伪影误报**,已在 0a9b247 修复:像素内参与无量纲畸变系数混在同一个
协方差里,条件数被单位差抬高数个量级(m1 原始 2.7e9 → 无量纲化 1.0e3)。
`spectrum_report` 已改在无量纲坐标上算谱并重跑,修正后的产物在
`output/fisher/run_20261019_025859`(CRLB 逐参数数值与本轮逐位一致,该修复
只影响谱诊断):m1/m2 正常,m3(5.7e11)/m4(1.2e6)仍退化且极端方向整块
落在高阶畸变项(k5/k2/k6、k3/k4/k2)——真实发现改为"这批数据定不住复杂
模型的高阶畸变项",是阶段四选型防过拟合的输入。详见 deferred-items.md。
逐参数 CRLB、Σ 带、交叉验证结论不受影响。

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] _fisher_detail 逐模型覆盖丢失**
- **Found during:** Task 2(测试先红暴露)
- **Issue:** 明细字典在模型循环体内重置,`report["_fisher_detail"]` 只剩最后一个模型的 FIM 数组
- **Fix:** 明细挂到各模型的 `_fisher_detail` 键,save 侧按模型取出转 npz
- **Files modified:** src/calib/noise/fisher.py;**Commit:** 22ddf99

**2. [Rule 1 - Bug] 模型级失败只补 2/4 份记录**
- **Found during:** Task 2(故障隔离测试红)
- **Issue:** 补失败记录时判"观测集字典是否为空"而非"该 σ 键是否缺失",填完 detector 后字典非空,temporal 被跳过
- **Fix:** 改判 `sigma_name not in ...`,失败模型 4 份记录结构键全
- **Files modified:** src/calib/noise/fisher.py;**Commit:** 22ddf99

**3. [Rule 1 - Bug] 交叉验证参数对账对 ndarray 调 .items()**
- **Found during:** Task 2(参数对账测试红)
- **Issue:** `cc["params_opencv"]` 是按 param_names 排布的数组(不是字典),`.items()` 当场 AttributeError
- **Fix:** `zip(model.param_names, values)` 逐名对账(float32 容差语义不变)
- **Files modified:** src/calib/noise/fisher.py;**Commit:** 22ddf99

**4. [Rule 1 - Bug] plot_spectrum 对内存态 ndarray 的真值歧义**
- **Found during:** Task 3(集成试车发现 figures/ 目录空)
- **Issue:** `spectrum.get("eigvals") or []` 对 numpy 数组抛 ValueError——json 读回态(列表)不炸、编排内存态全炸,16 张谱图静默落空(附加产物兜底吞了报错)
- **Fix:** 改判 None;集成测试补"谱图至少一张"断言防复发
- **Files modified:** src/calib/visualization/noise_figs.py, tests/test_fisher_realdata.py;**Commit:** 35e5f8f

**5. [Rule 1 - Bug] CLI 血缘上游误用扁平产物名**
- **Found during:** Task 3(血缘对账测试红)
- **Issue:** 上游按 `m{X}__scipy_result.json`(run 根下)探测,实际标定产物在 `m{X}__scipy/` 子目录——`is_file()` 全不命中,scipy/opencv 产物静默缺席血缘
- **Fix:** 改用子目录相对路径;复用规则同步要求血缘含逐模型 scipy 条目(缺账旧轮自动重跑)
- **Files modified:** src/calib/cli/main.py, tests/test_fisher_realdata.py;**Commit:** 35e5f8f

### 计划内签名微调(非偏差,已在执行中说明)

- `run_fisher_matrix` 形参比计划草案多 `manifest_path` 与 `exclude_images`(装载观测集的行序契约入口与 D-03 剔除清单,口径与标定一致)。
- `aggregate_observations` 返回 `{"poses": ..., "image_size": ...}` 嵌套结构(构造聚合观测集需要图像尺寸)。

## Auth Gates

无。

## Known Stubs

无——所有数据源均已接通:CLI 取真实上游产物(最新 corners/noise/calibration 轮),
16 份记录、谱图、明细 npz 全部真实落盘;`calib fisher` 无占位输出。

## Threat Flags

无新增信任面——crlb.json 只写 `output/`(gitignored),npz 读写全程
allow_pickle=False,yaml safe_load,无网络/无反序列化新面。

## Self-Check: PASSED

- 文件存在:`tests/test_fisher_realdata.py`、`src/calib/noise/fisher.py`(run_fisher_matrix)、`src/calib/noise/artifacts.py`(save_crlb_report)、cli/main.py(OUTPUT_ROOT / "fisher")、io/config.py(FisherConfig)——全部在。
- 提交存在:9b458ed / 51733f8 / 22ddf99 / 35e5f8f —— `git log` 可查。
- 测试:非集成 347 passed;集成 7 passed(真实数据,run_20261018_145407,16/16 记录 ok)。
- TDD 门:RED(9b458ed)→ GREEN(51733f8)→ feat(22ddf99)→ test(35e5f8f)顺序成立。
