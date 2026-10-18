---
phase: 02
status: findings
depth: standard
files_reviewed: 15
files_reviewed_list:
  - src/calib/models/base.py
  - src/calib/models/pinhole_brown.py
  - src/calib/models/kb4.py
  - src/calib/models/__init__.py
  - src/calib/solve/observation.py
  - src/calib/solve/result.py
  - src/calib/solve/opencv.py
  - src/calib/io/config.py
  - src/calib/solve/residuals.py
  - src/calib/solve/scipy_solver.py
  - src/calib/solve/matrix.py
  - src/calib/cli/main.py
  - src/calib/solve/scoring.py
  - src/calib/visualization/suspect_compare.py
  - config/calibration.yaml
findings:
  critical: 1
  warning: 4
  info: 5
  total: 10
generated: 2026-10-18T04:27:41Z
---

# Phase 02: 代码评审报告

**评审时间:** 2026-10-18T04:27:41Z
**深度:** standard(逐文件分析 + 跨文件契约核对)
**评审文件数:** 15
**状态:** findings(发现问题)

## Summary

本次评审覆盖阶段二的模型层(M1–M4)、双求解后端、观测集装载、全矩阵编排、存疑帧复核与配置。整体结论:核心数学是可信的——手推核对了针孔族径向/切向/有理模型的全套解析雅可比、KB4 投影与点雅可比(含近轴极限,泰勒展开验证 s→1/z、∂s/∂x→0 等极限全部正确)、`transform_to_camera` 的 Rodrigues 雅可比链式(02-04 修复后的版本与 cv2 布局 rot_jac[k, 3i+j] 逐项一致)、残差/雅可比/稀疏结构三者的行序装配(共用 `_structure_positions`,逐项核对无误)。OpenCV 适配器的 flags、D 向量切片(M1 取 5 长前 4 / M3 取 14 长前 8)经本机 OpenCV 4.14 合成数据实测全部成立。

发现的问题集中在**契约与防线的"宣称 vs 实际"**层面:1 个阻断级(配置放行的 classic 路线必然崩溃)、4 个警告(被 scipy 静默忽略的 jac_sparsity、反投影牛顿迭代无收敛防护、两处血缘记录与实际消费不符),以及 5 个信息级。血缘链上"记了没吃过的文件、吃了没记的文件"两处缺口值得在阶段三接入 Fisher 前修掉——阶段三的线性化点恰恰要从这些产物读数。

deferred-items.md 里记录的 5 处 ruff 违例未重复上报(其中 4 处在当前代码里已修,仅 models/__init__.py:65 的 noqa 指令仍在遗留清单);soft_l1 未收敛属已记录行为,本次只审了收敛判定逻辑本身,未翻案。

## Critical Issues

### CR-01: `corner_route: classic` 是配置放行却必然崩溃的死路(契约自相矛盾)

**文件:** `src/calib/solve/observation.py:155-160`(主责);`src/calib/io/config.py:27,449-453`;`src/calib/cli/main.py:661`

**问题:** 配置层明确放行并宣传 classic 路线——`_ALLOWED_CORNER_ROUTES = {"sb", "classic"}`(config.py:27),`_parse_calibrate` 的中文报错还引导用户"corner_route 配回 classic 时行为与旧版一致"(cli/main.py:661 同义承诺)。但 `load_observation_set` 强制要求角点 npz 带 `source` 键,缺了就抛"它不是 sb 路线的角点文件……请改用 corners_<切分>_sb.npz"。而 `save_corners`(corners/artifacts.py:70-72)只在 sb 路线写 `source` 键,classic 产物天生没有。结果:用户把 `corner_route` 配成 `classic` 后,`calib calibrate` 100% 抛错,且报错文案把用户往 sb 文件上引——与配置层的承诺直接矛盾。`review_suspects` 的 test 评分路径(scoring.py:190)也踩同一颗雷。故障是响的、不是静默算错,所以定 Critical 而非更高,但它是一个文档化配置路径的必然失败。

**修复:** 二选一(建议前者):
```python
# observation.py:载入处,缺 source 键时按路线补常量,而不是拒绝
with np.load(Path(corners_npz), allow_pickle=False) as data:
    if "source" in data.files:
        source = [str(x) for x in np.asarray(data["source"]).ravel()]
    else:
        # classic 路线没有兜底概念,来源恒为 classic 检测器本体
        source = ["classic"] * uv.shape[0]
```
或在 config 层把 `_ALLOWED_CORNER_ROUTES` 收紧为 `{"sb"}`,并在 `_parse_calibrate` 报错里写明"classic 路线的求解装载尚未支持",等 loader 支持后再放开——两处契约必须先对齐一处。

## Warnings

### WR-01: `jac_sparsity` 传给 scipy 后被完全忽略——一个并不存在的"结构性防线"

**文件:** `src/calib/solve/scipy_solver.py:186-193`;`src/calib/solve/residuals.py:203-214`

**问题:** `least_squares(..., jac=make_jacobian_fn(...), jac_sparsity=make_sparsity(...))` 同时传了可调用解析雅可比与稀疏结构。scipy(项目锁定的 1.18.1,已实测)对可调用 `jac` 完全不使用 `jac_sparsity`:无告警、传与不传结果逐位相同(实测确认)。`make_sparsity` 的 docstring 声称"least_squares 的 jac_sparsity 用""scipy 按结构分组解子问题,这是它对结构清单的信赖"——这个论断与实际行为不符。影响有两层:640 视图规模下每次求解白白组装约 169 万个非零元的 COO 结构矩阵(秒级浪费);更糟的是制造了一个假防线印象,维护者可能以为结构错误会被 scipy 拦下(不会)。真正的结构防线其实存在且有效——`jacobian_fn` 与 `_structure_positions` 共用同一套行列排布——只是名字挂错了。

**修复:** 删掉 `jac_sparsity=make_sparsity(...)` 传参与 `make_sparsity` 的导入;若保留该函数供测试对照结构,把 residuals.py 模块 docstring 与 make_sparsity docstring 改为如实陈述:"scipy 对解析雅可比忽略该参数,行序一致性由 `_structure_positions` 与 `jacobian_fn` 共用保证,测试逐项锁定"。

### WR-02: 两个 unproject 的牛顿迭代无收敛性与奇异性防护,违反模型层"绝不返回半成品"硬契约

**文件:** `src/calib/models/pinhole_brown.py:242-251`;`src/calib/models/kb4.py:177-186`

**问题:** base.py 把"非法输入抛中文 AssertionError、绝不返回 NaN/Inf、绝不返回半成品"定成模型层硬契约,但两个反投影实现都不设防:
- 针孔族:`det = a11*a22 - a12*a12`(pinhole_brown.py:245)可为零或近零——M3 有理模型分母过零附近必然出现——除零直接产出 inf/NaN 原样返回;50 轮跑满仍不收敛也不报错,静默返回未收敛射线。
- KB4:`f_prime = 1+3k1θ²+…+9k4θ⁸`(kb4.py:181-182)在强畸变参数组合下可为 0 或负(θ 多项式非单调),牛顿步发散同样无检查。
往返契约测试用的是标定收敛点附近的良性参数,挡不住外推诊断、坏初值组合这些路径;未收敛的射线一旦被下游(阶段三线性化点、阶段五反投影诊断)消费,NaN 会毒化整条链——这正是 base.py:80 点名要防的事故形态。

**修复:** 迭代结束后加终检,违反即按契约抛错:
```python
# 循环后(pinhole_brown):
res_final = np.stack([x_pred - xd, y_pred - yd], axis=1)
if not np.isfinite(res_final).all() or np.abs(res_final).max() > 1e-8:
    raise AssertionError(
        f"模型 {self.model_id} 的 unproject 牛顿迭代未收敛或遇奇异点"
        f"(最大残差 {np.abs(res_final).max():.3e}),当前参数组合下畸变逆解不存在,"
        f"请检查畸变系数是否病态。")
```
KB4 同理(对 `|f|` 与 `f_prime` 的绝对值设下界)。

### WR-03: scipy 组合的血缘记录与实际消费不符——记了没吃的,吃了没记的

**文件:** `src/calib/solve/matrix.py:208-211`(时序);`src/calib/solve/matrix.py:153-155`(upstream 组装)

**问题:** 两处血缘失真,都发生在 scipy 精化组合:
1. **记了没吃的**:`_lineage_for(..., init_dir, ...)` 在 `_load_init_result` 判定**之前**执行。当同轮 OpenCV 组合存在但未收敛时,`_load_init_result` 返回 None、scipy 实际走通用初始化(`init_source: generic_default`),但血缘 upstream 仍记着 `init_result.json`——那份未收敛文件一个字节都没被读。审计者对照血缘会得出"这个精化结果吃了 OpenCV 产物"的假结论,与 `_lineage_for` 自己写的"记一个指向不存在文件的指纹等于伪造证据"精神相悖(这里是指向存在但未被消费的文件)。
2. **吃了没记**:scipy 精化的初值位姿实际来自 `<模型>__opencv/extrinsics.npz`(`_load_init_result` → `load_extrinsics`),但 upstream 只指纹了 `result.json`。`extrinsics.npz` 被单独重生成或损坏时,初值已换、血缘不变——违背"配置哈希 + 上游产物哈希"的契约原文。阶段三拿这些产物当线性化点,这个缺口会直接传导。

**修复:**
```python
# matrix.py:_run_scipy_combo 内,先判消费、再组血缘
init = _load_init_result(init_dir, model.model_id) if init_dir else None
lineage = _lineage_for(cfg_files, corners_npz,
                       init_dir if init is not None else None,
                       excluded_images, round_id)
# _lineage_for 内把 extrinsics.npz 一并记入:
if init_dir is not None:
    upstream["init_result.json"] = init_dir / "result.json"
    upstream["init_extrinsics.npz"] = init_dir / "extrinsics.npz"
```

### WR-04: 存疑帧复核的血缘漏记 test 角点产物

**文件:** `src/calib/solve/scoring.py:357-366`

**问题:** `review_suspects` 的结论(存疑 vs 正常的逐帧误差对比)直接消费 `corners_test_<route>.npz`(`_score_test_frames` → `load_observation_set`,scoring.py:190),test 帧的评分数字就出自这份文件;但血缘 upstream 只记了所选组合的 `result.json` 与 manifest,没记这份角点 npz。换一份不同的 test 角点文件(哪怕只是重跑一轮 corners)重做复核,结论可以不同而血缘完全一致——违背"任一结论可溯源到产物指纹"的 ITER-01 精神。train 侧逐帧 RMS 由 result.json 间接承保,链上是闭合的;test 侧是实打实的缺口。

**修复:**
```python
test_npz = corners_run / f"corners_test_{cfg.calibrate.corner_route}.npz"
lineage = make_lineage(
    config_files={},
    upstream={
        "result.json": calibration_run / chosen_combo / "result.json",
        "manifest.json": manifest_path,
        **({"corners_test.npz": test_npz} if test_npz.is_file() else {}),
    },
    round_id=...)
```

## Info

### IN-01: `_n_dist` 类属性全仓无人读取——死代码

**文件:** `src/calib/models/pinhole_brown.py:113,324,338,353`

**问题:** `_PinholeBrown._n_dist` 声明为"子类定:4/5/8",三个子类也赋了值,但畸变向量长度实际处处由 `len(param_names) - 4` 隐式推出,`_n_dist` 没有任何读取方(grep 全仓确认)。留着会误导读者以为它参与布局推导。

**修复:** 删掉四处声明与赋值;或在 `_PinholeBrown.__init_subclass__` 里校验 `len(cls.param_names) - 4 == cls._n_dist`,让它至少干点实事。

### IN-02: residuals.py 模块 docstring 对 `row_index` 的描述与实现不符

**文件:** `src/calib/solve/residuals.py:8`(docstring);`src/calib/solve/residuals.py:37-49`

**问题:** 模块 docstring 称"残差与雅可比共用同一个行号函数 row_index(结构性防线)",实际上 `residual_fn` 与 `jacobian_fn` 都不调用 `row_index`——真正的共用防线是 `(n,2)` ravel 约定加 `_structure_positions`(这个是真有效,已核对三处排布逐项一致);`row_index` 只有测试在用。

**修复:** 改 docstring 为如实陈述("行序契约由 `row_index` 形式化定义、由 ravel 约定与 `_structure_positions` 落实,测试用 `row_index` 逐行断言"),防线的名字要指对地方,否则后人重构时可能把真防线当冗余删掉。

### IN-03: `np.divide` 的 `out=` 实参急切求值,合法 z=0 输入会刷 divide-by-zero 警告

**文件:** `src/calib/models/kb4.py:59,83-84,137`

**问题:** `out=1.0/z`、`out=x/z` 写在 `np.divide(..., where=...)` 的实参位置,Python 会先对全量点算出 `1.0/z` 再进 divide——鱼眼投影对 z=0(90° 入射)是合法输入,于是每个这类点都触发一次 RuntimeWarning。被 where 掩蔽的位置不会把 inf/NaN 带进结果,功能正确,但批量求解时告警刷屏,且"掩蔽位永不生效"只靠输入合法性隐式保证。

**修复:** 用 `with np.errstate(divide="ignore", invalid="ignore"):` 包住,或改写为不含除零的表达式(如 `out=np.reciprocal(np.where(z == 0, 1.0, z))` 也不行——同样急切;正确姿势是 errstate 或先把 z 钳位)。

### IN-04: 鱼眼停止条件两处定义,漂移风险

**文件:** `src/calib/solve/opencv.py:36`;`src/calib/io/config.py:130-137`

**问题:** `_FISHEYE_CRITERIA = (…, 100, 1e-6)` 在 opencv.py 硬编码,`FisheyeConfig.max_iter/epsilon` 在 config 里存了同值字段但阶段二路径不再读。两边注释各自声明了分工,但同一组常量两处维护,改 config 不会影响阶段二行为——将来一定有人踩。

**修复:** 在 `FisheyeConfig` 的 docstring 里加一句显式指认("阶段二的鱼眼停止条件在 solve/opencv.py:_FISHEYE_CRITERIA,两处值必须一致"),或加一条一致性断言测试把两处值锁在一起。

### IN-05: `_initial_theta` 的调用方误用类 ValueError 被吞进失败契约

**文件:** `src/calib/scipy_solver.py:132-148`(三处 `raise ValueError`);`src/calib/solve/scipy_solver.py:233-243`(兜底 except)

**问题:** `_initial_theta` 对"初值未收敛""初值与目标模型不一致""参数字典缺名"三类**编程错误**精心写了中文报错,设计意图是大声拦下(报错文案全是"不能混用""请按参数名给全再精化"的契约口吻);但 `solve_with_scipy` 的 `except (ValueError, …)` 会把它们转成一行"未跑成"的失败结果落盘。排障时,误用与数值失败在汇总表里长一个样,容易被误读。当前矩阵编排里 `_load_init_result` 已提前挡住这些情况,所以实际打不着——风险在库调用方(阶段四的评估实验函数会直接调 `solve_with_scipy`)。

**修复:** 给这三类误用定义专门异常(如 `class InitContractError(ValueError)`),except 子句显式排除它让其穿透;或把这三处改抛 `AssertionError`(现契约下 assert 系错误本来就故意穿透,scipy_solver.py:234-237 注释言明)。

---

_评审时间: 2026-10-18T04:27:41Z_
_评审人: Claude (gsd-code-reviewer)_
_深度: standard_
