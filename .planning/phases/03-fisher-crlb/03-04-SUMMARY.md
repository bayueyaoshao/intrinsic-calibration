---
phase: 03-fisher-crlb
plan: 04
subsystem: noise
tags: [opencv-stddev, crosscheck, scale-normalization, failure-contract, fisheye-boundary]

# Dependency graph
requires:
  - phase: 02-m1-m4
    provides: opencv_adapter()(flags 与 K/D 换算唯一入口,M1/M3 选项差异已封装)、ObservationSet
  - phase: 03-fisher-crlb/03-02
    provides: marginalized_fisher(σ=1 基准 FIM)、crlb_se(σ 一次乘法折算)
provides:
  - opencv_stddev_intrinsics:Extended 标定 8 元组桥接,返回 OpenCV 解 + stdDev 向量(前 P 位对齐 + 18 元原值)+ M4/异常分支
  - compare_structural:单尺度 k 拟合 + 尺度归一逐参数相对偏差(Pitfall 1 判读规则在 docstring)
  - crosscheck_model:协议三步编排(双线性化点 SE、换点不敏感对照、σ_detector 副报告),03-05 即插即用
  - tests/test_noise_crosscheck.py:协议本体在合成数据上的 10 用例锁定
affects: [03-05 calib fisher 命令编排(crlb.json 交叉验证段), 成功标准 4 的真实数据执行]

# Tech tracking
tech-stack:
  added: []   # 零新增依赖(cv2/numpy 既有锁定版本,T-03-04-05)
  patterns:
    - "OpenCV flags/K/D 装包解包只走 model.opencv_adapter(),源码静态锁无选项字面量(T-03-04-02)"
    - "calibrateCameraExtended 4.14 按 8 元组解包(元组数漂移当场 ValueError,真实调用路径测试覆盖)"
    - "主判据 = 尺度归一结构比对,绝对比对只作副报告不设门槛(免噪声标定口径歧义)"
    - "失败/不可用状态返回完整结构键(数值全 None),读产物代码不用先猜键在不在"

key-files:
  created:
    - src/calib/noise/crosscheck.py
    - tests/test_noise_crosscheck.py
  modified: []

key-decisions:
  - "M4 分支按适配器 kind='fisheye' 识别而非硬编码 model_id 字符串——协议层面语义等价(M4 即鱼眼族),未来新鱼眼模型自动走同一'记录结论'路径"
  - "副报告与换点不敏感对照做成可选参(sigma_detector / linearization=None 占位):03-05 按需传,测试验证占位路径"
  - "OpenCV 标定存在 ~1e-10 的跑间并行归约噪声(实测),同点重算的换点敏感度判定线取 1e-6 而非机器精度"
  - "失败透传补全顶层结构键:crosscheck_model 对 failed/unavailable 也返回 se/compare/secondary_report 等键(全 None),crlb.json 结构恒定"

patterns-established:
  - "交叉验证判读规则(Pitfall 1)写死在 compare_structural docstring:比值逐参数几乎同一个数 = 尺度因子问题(数学对);散乱 = 数学真有问题"
  - "监视桩测试体例:dir(cv2.fisheye) 里 calibrate* 全换成'被调即炸'的桩,证明分支绝不硬调"

requirements-completed: [NOI-03]  # 交叉验证协议部分(合成数据锁定);真实数据门槛的执行在 03-05

# Metrics
duration: 16min
completed: 2026-10-18
---

# Phase 3 Plan 4: OpenCV stdDev 交叉验证协议 Summary

**OpenCV stdDeviationsIntrinsics 与我们 σ=1 边际化 SE 的交叉验证协议落地:尺度归一后合成 20 视图场景逐参数一致到 2.0e-05(门槛 5%),公共尺度 k=0.1009 与注入 σ=0.1 对账,M1 固定位剔除 / M3 尽力 / M4 记结论 / 异常转失败契约四条分支全部被测试锁死。**

## Performance

- **Duration:** 16 min
- **Started:** 2026-10-18T13:32:52Z
- **Completed:** 2026-10-18T13:48:42Z
- **Tasks:** 1/1(TDD RED→GREEN,共 2 个提交)
- **Files modified:** 2(均为新建)

## Accomplishments

- `src/calib/noise/crosscheck.py`(317 行),全部经 models/ 适配器与 03-02 的 Fisher 零件拼装,无第二份数学:
  - `opencv_stddev_intrinsics`:协议第①步——适配器拿 flags、float32 装包体例照 solve/opencv.py、`calibrateCameraExtended` 按 **8 元组**解包(4.14 实测,注释标明,版本漂移当场 ValueError);返回 OpenCV 解(params/rvecs/tvecs/rms/per_view_errors)+ stdDev 前 P 位对齐向量 + 18 元原值(固定参数位恒 0,天然对账位);M4(kind="fisheye")直接返回 `unavailable_fisheye` 中文结论,绝不调用任何鱼眼函数;`cv2.error` 转 `failed` + 中文 failure_note
  - `compare_structural`:协议第③步主判据——`k = Σ(opencv·ours)/Σ(ours²)` 最小二乘单尺度拟合,opencv/k 归一后逐参数相对偏差;参与位必须为正有限值(防除零 NaN,中文报错点名),未参与位 NaN 占位;Pitfall 1 判读规则写死 docstring
  - `crosscheck_model`:三步编排——②在 OpenCV 解的点上经 `marginalized_fisher` + `crlb_se` 算我们的 σ=1 SE,另在调用方给的 linearization 点算一版作"换点不敏感"对照(D-06 引用);主判据 + σ_detector 副报告(绝对比对不设门槛);失败/不可用原样透传且结构键补全
- `tests/test_noise_crosscheck.py`(311 行,10 用例全合成数据,总时长 ~1.6 s):
  - 恒等参数映射:M1/M2/M3 参数名 == OpenCV 18 元布局前 P 位(逐模型断言,静态锁)
  - M2 硬门槛:stdDev 形状/正值(行为 1)+ k=0.1009 ∈ (0.05, 0.2)、尺度归一后最大相对偏差 2.0e-05 < 5%(行为 2)
  - M1:k3 固定位 stdDev 恒 0 对账、按前 8 位切齐无 NaN 无除零、< 5% 硬门槛同样成立(行为 3)
  - M3 尽力档:协议全程跑通、结果齐全即过,不对上不失败(D-07 分级,行为 3)
  - M4:监视桩证明 cv2.fisheye 的 calibrate* 一个都没被碰 + 中文结论(行为 4)
  - cv2.error → failed 契约、compare_structural 手算对表(k 方向锁定)、源码静态锁无 "CALIB" 字面量(行为 5 + T-03-04-02 缓解)
- 全套 `uv run pytest -m "not integration" -q` → **315 passed**,ruff check 干净

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 换点不敏感对照的判定线按实测数值噪声配平(1e-12 → 1e-6)**
- **Found during:** Task 1 GREEN
- **Issue:** 同一个点当线性化点重算,敏感度实测 1.5e-10 而非严格 0——OpenCV 标定内部并行归约存在跑间微小差异(两次调用结果在双精度尾数级不同),机器精度判定线不可达
- **Fix:** 判定线放宽到 1e-6(真实换点敏感度是百分量级,不会贴线),来历写进测试注释
- **Files modified:** tests/test_noise_crosscheck.py
- **Commit:** 19680de

**2. [Rule 2 - 契约补全] 失败/不可用透传补全顶层结构键**
- **Found during:** Task 1 GREEN
- **Issue:** 计划未写明 crosscheck_model 在 failed/unavailable 状态下的返回形状;若原样透传桥接层 dict,顶层键(se/compare/secondary_report 等)缺失,读 crlb.json 的代码得先猜键在不在,与"结构键一个不少"的失败契约(研究 Pattern 4)不符
- **Fix:** 透传时补全全部顶层键(数值一律 None),crlb.json 交叉验证段结构恒定
- **Files modified:** src/calib/noise/crosscheck.py
- **Commit:** 19680de

**3. [Rule 1 - Bug] 手算对表用例的构造方向修正(测试侧笔误)**
- **Found during:** Task 1 GREEN
- **Issue:** 初版手算用例构造 opencv = 10×ours 却期望 k=0.1;按公式 k=Σ(opencv·ours)/Σ(ours²) 该构造 k 恒等于 10。真实场景实测 k=0.1009 ≈ σ̂,即 OpenCV 标准差 = 我们 σ=1 界 × 自估噪声尺度(比我们小一个噪声尺度)——计划公式与 k∈(0.05,0.2) 门槛在真实方向下自洽,是测试夹具方向反了
- **Fix:** 夹具改为 opencv = 0.1×ours(真实方向),k 精确等于 0.1、归一后偏差 0 的手算对表成立
- **Files modified:** tests/test_noise_crosscheck.py
- **Commit:** 19680de

## Known Stubs

None——全部函数为完整实现,无占位/空数据路径(σ_detector=None 与 linearization=None 是计划明文的可选参占位,不是未实现)。

## TDD Gate Compliance

- Task 1:RED(589d91d,ModuleNotFoundError)→ GREEN(19680de)✓
- test→feat 顺序经 git log 验证

## Verification Run

- `uv run pytest tests/test_noise_crosscheck.py -q` → 10 passed(~1.6 s)
- `uv run pytest -m "not integration" -q` → 315 passed, 6 deselected
- `uv run ruff check src/calib/noise/crosscheck.py tests/test_noise_crosscheck.py` → 干净
- 计划验收逐条:M2 结构比对逐参数 < 5%(实测 2.0e-05)且 k ∈ (0.05, 0.2)(实测 0.1009)✓;M1 活跃集对齐无 NaN、恒等映射三断言绿 ✓;M4 分支 unavailable_fisheye 中文结论、cv2.error 分支 failed + 中文 note ✓;crosscheck.py 无 "CALIB" 字面量(grep + 测试双重锁)✓
- 威胁登记册五条处置全部落实:T-03-04-01 协议三步进 docstring 与测试;T-03-04-02 flags 全走适配器 + 源码静态锁 + M1 固定位对账;T-03-04-03 cv2.error 转中文失败契约;T-03-04-04 8 元组解包注释标明实测版本、测试覆盖真实调用路径;T-03-04-05 零新增依赖

## Boundary Notes(给下游计划)

- 03-05 直接 `from calib.noise.crosscheck import crosscheck_model` 即用:传 `linearization=(params, rvecs, tvecs)`(mX__scipy 点)与 `sigma_detector`(noise.json 读数),返回 dict 的 compare 段即成功标准 4 的判据,按 D-07 分级落 crlb.json(M1/M2 看硬门槛、M3 按发现记录、M4/failed 段数值全 None)
- 真实数据门槛(M1/M2 个位数百分比)由 03-05 的真实试车执行;本计划锁的是协议本身(合成数据可复跑)
- 副报告的 ours_se_scaled 与 opencv_se 并排落盘即可,不设门槛——差异 = 双方噪声标定口径差,报告解读时引用 secondary_report.note 的措辞
- OpenCV 标定有 ~1e-10 跑间噪声(实测发现),任何"与已存 mX__opencv 参数逐位一致"类断言都应留双精度尾数级余量

## Self-Check: PASSED

- src/calib/noise/crosscheck.py 存在(317 行 ≥ 100,含 `def compare_structural`)✓
- tests/test_noise_crosscheck.py 存在(311 行 ≥ 100,含 calibrateCameraExtended)✓
- 提交 589d91d / 19680de 均在 git log ✓
