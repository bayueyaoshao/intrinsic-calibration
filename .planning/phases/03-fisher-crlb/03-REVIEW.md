---
phase: 03-fisher-crlb
reviewed: 2026-10-19T03:39:16Z
depth: standard
files_reviewed: 21
files_reviewed_list:
  - config/calibration.yaml
  - docs/pipeline.md
  - src/calib/cli/main.py
  - src/calib/io/config.py
  - src/calib/noise/artifacts.py
  - src/calib/noise/crosscheck.py
  - src/calib/noise/fisher.py
  - src/calib/noise/immerkaer.py
  - src/calib/noise/injection.py
  - src/calib/noise/temporal.py
  - src/calib/noise/transfer.py
  - src/calib/visualization/noise_figs.py
  - tests/test_fisher_realdata.py
  - tests/test_io_config.py
  - tests/test_noise_crosscheck.py
  - tests/test_noise_fisher.py
  - tests/test_noise_immerkaer.py
  - tests/test_noise_injection.py
  - tests/test_noise_realdata.py
  - tests/test_noise_temporal.py
  - tests/test_smoke_realdata.py
findings:
  critical: 0
  warning: 10
  info: 7
  total: 17
status: issues_found
---

# Phase 3: Code Review Report(Fisher 信息 / CRLB 理论精度极限)

**Reviewed:** 2026-10-19T03:39:16Z
**Depth:** standard
**Files Reviewed:** 21
**Status:** issues_found

## Summary

本阶段评审覆盖 fisher/noise 两条链路的 12 个源文件与 9 个测试文件。总体结论:**数学核心是可靠的**——Schur 消元(把位姿参数从 Fisher 矩阵里消掉、只留内参的矩阵代数手法)、σ=1 基准下的线性缩放、谱报告、Σ 敏感性带,都有恒等式对照、numdifftools 数值基准、蒙特卡洛、手算 5×5 对照四层测试锁定;两处近期修复(谱报告改用无量纲坐标算条件数、交叉验证用尺度归一结构比对)确认已正确落地并有对应测试(test_noise_fisher.py:345-416 的手算谱与单位不变性断言;compare_structural 的手算测试)。未发现会静默算错主结果的问题,未发现安全问题(yaml.safe_load、npz allow_pickle=False 全部落实)。

10 个 WARNING 集中在两类:**鲁棒性边界**(单张坏图炸全阶段、网格外插值为负、负特征值静默放行)与**契约一致性**(lineage 缺失、失败契约覆盖不全、配置放行但运行层回退、产物自相矛盾)。这些不会让 happy path 算错,但会让异常场景产生自相矛盾的产物或违反项目自己写下的承诺,建议在收尾前修掉 W-01/W-02/W-03。

## Critical Issues

无。未发现安全漏洞、数据丢失风险或会静默算错主结果的问题。

## Warnings

### WR-01: 交叉验证 point_sensitivity 未排除无界参数,NaN 可直写 crlb.json

**文件:** `src/calib/noise/crosscheck.py:296-299`
**问题:** `point_sensitivity = float(np.max(np.abs(se_at_linearization - se_at_opencv) / se_at_opencv))` 没有做有限值掩码。CRLB 对某些参数可以是无界(`inf`,如病态模型下某内参完全定不住):此时 `inf - inf = nan` 或 `x / inf = 0` 混入,任一参数为 inf 时 `np.max` 结果变成 NaN,原样写进 crlb.json。项目失败契约的底线是"失败写 null、绝不写 NaN",这里被击穿。
**依据:** 同文件 `compare_structural` 对 inf 做了完整掩码与正值校验;`src/calib/noise/fisher.py:834-838` 的线性化自检也明确 `np.isfinite` 掩码后才取 max——同一概念两处实现,一处有防线一处没有,说明这是遗漏而非设计。
**修复:**
```python
diff_arr = np.abs(se_at_linearization - se_at_opencv) / se_at_opencv
finite_diff = diff_arr[np.isfinite(diff_arr)]
point_sensitivity = float(np.max(finite_diff)) if finite_diff.size else None
```

### WR-02: 注入实验 Immerkaer 预扫描无逐图容错,一张坏图炸全阶段

**文件:** `src/calib/noise/injection.py:209-214`
**问题:** 主进程预扫描循环对每张图直接 `imread_checked` + `immerkaer_sigma`,没有 try/except。任何一张图损坏/读取失败,整个注入实验抛异常,整轮 noise 命令走失败报告。
**依据:** 模块自己的 docstring(14-16 行)写着研究 Pitfall 7 的承诺:"个别照片读图失败、检测失败都只是'数据'不是'事故'……整体照常跑完"。并行注入路径(`_inject_one_image` 调用处,248-259 行)也确实逐张兜了,唯独这条前置扫描没兜——同一承诺在同一模块内自相矛盾。
**修复:** 预扫描逐图 try/except,失败图记 NaN;中位数改 `np.nanmedian`。`_anchor_grid`(81-85 行)已有"全 NaN 报中文错"的防线,不会把全灭场景静默放过。

### WR-03: noise_splits 配置被无视,test 图无条件读入且写进明细产物

**文件:** `src/calib/cli/main.py:754-757, 833-845`
**问题:** `noise()` 命令无条件读取 test 划分的图并计算 Immerkaer 逐张明细(`immerkaer_per_image.npz`),不看 `noise_splits` 配置;但 noise.json 里的 `splits.test.n_images` 按 `noise_splits` 过滤后为 0。用户把 `noise_splits` 改成 `[train]` 后,产物自相矛盾:明细 npz 里全是 test 图,汇总 json 却说 test 一张没算。
**依据:** `config/calibration.yaml` 的 noise 段注释明说"只想算训练集可改成 [train]"——配置承诺了一个实际没生效的开关。注入实验(`run_injection_experiment`,injection.py:199-200)是尊重 `noise_splits` 的,两口径不一致。
**修复:** test 图的读取与 `_immerkaer_detail` 都按 `"test" in noise_cfg.noise_splits` 门控;关闭时明细 npz 不含 test 或整段跳过,并在 meta 里写明。

### WR-04: injection_sigma_min/max 单边设置时配置放行、运行层静默双回退

**文件:** `src/calib/noise/injection.py:86-90`、`src/calib/io/config.py:584-595`
**问题:** 配置层对 `injection_sigma_min` / `injection_sigma_max` 各自独立校验(单个 > 0 即通过);运行层 `_anchor_grid` 却要求**两个同时非 None** 才用写死值,只写一边时静默走自动锚定。用户写死了一端、另一端按实测锚定的预期完全落空,且没有任何提示。
**修复:** 配置层拒绝单边设置("两个都写,或都留空"),或 `_anchor_grid` 尊重已设的一端、只对缺的一端自动锚定。

### WR-05: yaml 硬编码 injection_max_workers: 8,少于 8 核的机器上仓库自带配置直接非法

**文件:** `config/calibration.yaml:93`、`src/calib/io/config.py:616-622`
**问题:** 仓库自带的 calibration.yaml 写死 `injection_max_workers: 8`,而配置校验要求 `1 <= workers <= cpu_count`。在任何少于 8 逻辑核的机器上,**每一条 calib 命令**都会因为加载这份配置报错——不是 noise 阶段报错,是全阶段不可用。
**依据:** dataclass 里的默认值是自适应的 `min(8, cpu)`(config.py:230 附近),本意就是处理小核机器;yaml 字面量把它压过了。
**修复:** yaml 删掉该键改用自适应默认;或加载时钳制到 cpu_count 并给出中文提示。

### WR-06: transfer_curve.json 不带血缘指纹,证据链断头

**文件:** `src/calib/noise/transfer.py:131-138`
**问题:** `save_transfer_curve` 写出的 json 没有 lineage 字段。同一批产物里 noise.json 带 lineage、逐姿态明细 npz 带旁车 meta.json、注入实验 meta 带 lineage,唯独传递曲线(σ→角点误差的换算依据)无法追溯它是从哪份注入产物算出来的。
**依据:** 项目硬性设计原则"来源可追溯:每个结果文件自带'用什么配置、从哪些文件算出来的'指纹"。全链路只有这一个产物断头。
**修复:** `save_transfer_curve` 增加 lineage 参数(由 CLI 组装传入),写进 json 的 meta;下游 fisher 读曲线时顺带校验。

### WR-07: 基线角点与照片清单串轮时,注入实验静默跑出全 NaN 而不报错

**文件:** `src/calib/noise/injection.py:228-236`
**问题:** 构建任务时 `baseline.get(image_id)` 查不到(典型场景:角点 npz 是上一轮、照片清单是本轮),只走"占位整行 NaN"分支,实验照常"成功"完成。产物里 88 张图全部失败,看起来像检测全崩,真相却是输入没对齐。
**依据:** 同类场景下 `temporal.py`(147-153 行)与 `fisher.py`(`_load_linearization_point` 三道对齐断言)都会响亮报错,唯独注入实验缺这道防线。注意 `detected=False`(该图基线本来就没检出)是合法路径,要区分的是"图 id 根本不在 npz 里"这一串轮信号。
**修复:**
```python
if image_id not in baseline:
    raise ValueError(
        f"照片 {image_id!r} 在基线角点 npz 里找不到:角点产物与照片清单"
        f"不是同一轮,请确认 corners 路径指向本轮产物。")
```

### WR-08: interp_corner_sigma 网格外不钳制,PCHIP 外推可返回负 σ

**文件:** `src/calib/noise/transfer.py:117-128`
**问题:** 注入网格下限是 `max(0.25, σ_med/4)`,而逐姿态 Immerkaer 均值完全可能低于这个下限。`interp_corner_sigma` 对网格外的 σ 做 PCHIP 外推(保形插值的数值方法,只保证单调不保证物理),可能返回负值或非物理小值,随后直接流入 `sigma_detector_used` 参与 CRLB 换算——负 σ 会让 Fisher 矩阵标定出方向反的精度界。
**修复:** 越界时钳制到网格区间端点(并在产物里记一条"已钳制"的说明),或直接中文报错拒绝外推。

### WR-09: 特征值出现负数时条件数为负/inf,退化分档判空字符串静默放行

**文件:** `src/calib/noise/fisher.py:290-293, 305-310`
**问题:** `cond = f_eigvals[-1] / f_eigvals[0]`,若最小特征值(λmin,Fisher 矩阵谱分解后最小的那个特征根,数值上可以是负数——表示线性化点处某个参数组合根本定不住)为负,cond 为负或 inf;而退化判定 `cond >= cond_degenerate` 对负数判 False,最终 flag 记空字符串。**恰好在线性化点最退化、最需要报警的场景,报警哑掉了**。
**修复:** 分档前先判 `f_eigvals[0] <= 0` 直接记 degenerate(附中文说明),再走数值条件数分档。

### WR-10: fisher/noise 命令的上游装载在受保护块之外,失败契约只护了后半段

**文件:** `src/calib/cli/main.py:1017-1026`(fisher)、`685-689`(noise)
**问题:** 上游 manifest / noise.json / 角点 npz 的读取发生在 try 块之外。上游产物缺失时用户看到的是裸英文 `FileNotFoundError` 堆栈,而不是"失败也照常落盘失败报告"的中文契约——失败契约实际只覆盖了计算段。
**依据:** noise 命令内部(848-853 行)对计算期失败已经做到了"落盘失败报告",同一条命令的前置装载却可以绕过它,契约覆盖不一致。
**修复:** 把上游装载移进 try 块,失败时同样走 `make_failure_report` / `make_failure_crlb` 落盘,并写清"缺哪个文件、该先跑哪个阶段"。

## Info

### IN-01: 汇总表退化 flag 直接显示代码值 "warning",未按项目用语规则翻译

**文件:** `src/calib/cli/main.py:1115-1117`
**问题:** 终端汇总直接打印内部分档值 `"warning"`。配置语境下这个词的含义是"警戒"档(介于正常与退化之间),按项目硬性用语规则应显示中文。
**修复:** 显示层做一次映射:degenerate→退化、warning→警戒、空→正常。

### IN-02: 重复代码块:injection.py 与 temporal.py 各持一份 _rec_get/_VARIANT_KEYS;results 重复初始化

**文件:** `src/calib/noise/injection.py:99-104, 243`、`src/calib/noise/temporal.py`(同名单/字典)
**问题:** `_rec_get` 与 `_VARIANT_KEYS` 在两个模块各有一份逐字相同的实现,漂移风险(改一处忘另一处)。另 injection.py:243 串行分支里 `results = {}` 是死代码——238 行已初始化过。
**修复:** 提取到 `calib/noise/common.py` 之类的小共享模块;删除 243 行的重初始化。

### IN-03: cli/main.py 约 1212 行,五个阶段命令体全部内联,与模块布局原则渐行

**文件:** `src/calib/cli/main.py`(全文件)
**问题:** 项目架构原则是"模块布局与流水线阶段一一对应",但 noise/fisher/calibrate/evaluate/run-all 五个命令的业务编排(装配、回环提示、汇总打印)都长在 CLI 文件里,CLI 层只剩参数解析的边界在溶解。
**修复:** 不急于本阶段动;后续阶段把命令体逐步下沉到对应模块,CLI 只留参数解析与调用。

### IN-04: 成功文案硬编码 "16 份记录",models 配置可配

**文件:** `src/calib/cli/main.py:1104`
**问题:** 文案写死"16 份记录"(4 模型 × 2 观测集 × 2 σ 口径),但 `fisher.models` 是可配置列表,配 3 个模型时应为 12 份,文案失实。
**修复:** 文案用 `len(记录)` 或 `4 * len(models) * 4` 之类的实际计数拼出来。

### IN-05: test_noise_realdata 硬编码 88 个角点,与 board.yaml 耦合

**文件:** `tests/test_noise_realdata.py:199-201`
**问题:** 位移数组最后一维硬编码 `88`。同一测试的其余断言都从配置或产物元数据读,唯独这里写死;换棋盘规格时这条断言会先红,读代码的人要查一遍才知道是配置变了还是真坏了。
**修复:** 从 board.yaml 读 `cols * rows` 计算,保持测试与配置单一来源一致。

### IN-06: aggregate_observations 的角点均值与位姿均值口径不一致(剔除清单非空时)

**文件:** `src/calib/noise/fisher.py:386-394`
**问题:** uv 均值吃全部检出帧(含 `exclude_images` 逻辑剔除的帧),聚合位姿只吃解里有外参的帧。剔除清单非空时,二者统计口径不同:均值角点带着被剔除帧的影响。当前默认剔除清单为空,所以现在不出错——是埋着的口径地雷。
**修复:** uv 均值与位姿均值统一用同一帧集合,并在 docstring 里写明口径。

### IN-07: crosscheck 对非 cv2.error 的解包异常不设防;跨模块导入私有函数

**文件:** `src/calib/noise/crosscheck.py:46, 125-136`
**问题:** `calibrateCameraExtended` 的多返回值解包在返回个数变化时抛 `ValueError`,不是 `cv2.error`,逃出模块"不抛异常"的失败契约(仅靠 run_fisher_matrix 上层宽接兜底,callers 升级为整轮失败而非该模型失败)。另 `from calib.solve.opencv import _prepare_view` 导入私有函数,私有边界被穿透。
**修复:** 解包处就地 try/except 转 None(与 cv2.error 同路径);`_prepare_view` 提为公开函数或复制必要逻辑并注明来源。

---

_Reviewed: 2026-10-19T03:39:16Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
