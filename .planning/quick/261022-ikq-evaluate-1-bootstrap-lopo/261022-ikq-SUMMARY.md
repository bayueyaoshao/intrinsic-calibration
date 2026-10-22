---
quick: 261022-ikq-evaluate-1-bootstrap-lopo
plan: 261022-ikq
type: execute
status: complete
started: "2026-10-22T05:32:40Z"
finished: "2026-10-22T07:04:41Z"
duration: "约 1 小时 32 分(含测试收口与集成轮等待;三个任务提交约 40 分钟)"
commits:
  - "b35fc1b: feat(quick): evaluate 降载配置旋钮 exposures_per_pose + 重复次数下调"
  - "d3c0d71: feat(quick): 训练侧每姿态取前 N 张——切分辅助 + 三个 worker 接线"
  - "81a5bc3: test(quick): 实验级测试按降载新口径收口 + 旋钮/确定性/豁免三路测试"
  - "4b6234b: docs(quick): 模块 docstring 抽查/批量次数去写死——随次数降载同步口径"
files_modified:
  - config/calibration.yaml
  - src/calib/io/config.py
  - src/calib/evaluation/splitting.py
  - src/calib/evaluation/experiments.py
  - src/calib/analysis/sensitivity.py
  - tests/test_io_config.py
  - tests/test_evaluation_split.py
  - tests/test_evaluation_experiments.py
  - tests/test_evaluation_smoke.py
---

# 快任务 261022-ikq:evaluate 实验降载——训练侧每姿态 1 张 + bootstrap/lopo 次数下调 执行总结

一句话:evaluate 的交叉验证 / bootstrap / 留一姿态三类实验训练侧从"带全部连拍"(约 640 行)降到"每姿态取文件序前 1 张"(约 80 行,约 8 倍提速),验证侧(考题姿态)与重复张分组实验一字不动;quick 档 bootstrap 50→10 次、留一抽查 30→10 个,full 档 bootstrap 100→30 次——依据 docs/findings.md 结论 1(每姿态 1 张与连拍 8 张无实质差别,参数晃动的大头是位姿抽样不是检测噪声)。

## 各任务执行结果

### Task 1:配置旋钮 exposures_per_pose 与重复次数下调(提交 b35fc1b)

- `config/calibration.yaml` evaluate 段在 `cv_n_rounds` 与 `bootstrap_n` 之间新增 `exposures_per_pose: 1`,注释三层齐全:白话定义(训练侧每姿态最多取文件序前 N 张、按文件名排序即拍摄时间序、确定性不引入新随机源、验证侧不受影响)/ 依据(findings 结论 1,约 8 倍算力白费)/ 豁免清单(slices、噪声估计、Fisher、calib calibrate 主标定、calib analyze、CV 折数轮数全不读本键)
- 次数下调三处:`bootstrap_n: 100 → 30`(注释写明 2026-10-22 二次降载、先例 200→100、30 次相对误差约 13% 量级判定足够、给极值统计留尾巴、quick 档 10 次最大位移按下界读)、`quick_bootstrap_n: 50 → 10`、`quick_lopo_n: 30 → 10`(注释同步)
- `EvaluateConfig.exposures_per_pose: int = 1` 字段落位(cv_n_rounds 之后、bootstrap_n 之前,dataclass 默认值体系不动);`_parse_evaluate` 加校验:整数、先拒 bool 伪装、>= 1,违规中文报错点名 `evaluate.exposures_per_pose`(0 / -1 / true / 1.5 实测全被拦下,1 与 8 正常解析)
- `sensitivity.py` 模块 docstring 判读文案"50 次重标定的实际散布"改为"多次重标定的实际散布";grep 全 src/ 复核:cli/main.py 播报读的是配置值本身(不改),experiments.py 两处 docstring 写死数字由追加提交 4b6234b 清掉(见偏差 5)
- 验收:`tests/test_io_config.py` 68 passed;真实 yaml 四值断言 exposures_per_pose=1 / quick_bootstrap_n=10 / bootstrap_n=30 / quick_lopo_n=10 全过

### Task 2:切分辅助 exposures_per_pose_mask + 三个 worker 训练侧接线(提交 d3c0d71)

- `splitting.py` 新增纯函数 `exposures_per_pose_mask(obs, n_exposures)`:归组体例照 `_pose_groups`,组内按 `obs.image_ids` 排序取前 N 张(与 `slice_groups_by_index` 同一把尺,注释点名"严禁第二份排序逻辑");函数体再校验一道(整数、先拒 bool、>= 1,中文报错点名键名)——与配置层双保险(T-QK-01);docstring 写清与铁律的关系:**只做训练侧组内抽样**,姿态到训练/考题的归属一个不变("切分只操作姿态 ID、连拍同进同出"表述原样保留),调用方先 `pose_row_mask` 圈训练姿态再按位与,验证侧永远不带它,`split_by_pose_ids` 零重叠断言照旧生效
- 三个 worker 接线(张数经**任务元组**传入,worker 保持顶层函数可序列化):
  - `_cv_fold_worker`:训练侧 `pose_row_mask(...) & exposures_per_pose_mask(...)`,val_obs 一行不动;except 兜底 n_train_views 同口径
  - `_bootstrap_worker`:train_obs 同式按位与;docstring 更新为"去重后每姿态取文件序前 n_exposures 张(2026-10-22 降载裁定,依据 findings 结论 1)"
  - `_lopo_worker`:保留 `split_by_pose_ids`(零重叠门禁 + 验证侧评分都靠它),对返回的 train_obs 再套掩码;except 分支同口径
- 三个编排函数任务元组各追加 `eval_cfg.exposures_per_pose`;三处 meta 各记 `"exposures_per_pose"`(血缘留痕,T-QK-03);cv_pose_grouped / lopo / bootstrap docstring 训练侧表述统一改口径
- slices 侧一行未动:`_slice_group_worker` / `slices_by_index` / `slice_groups_by_index` 严禁接线的豁免位原样(git diff 复核为零改动)
- `tests/test_evaluation_split.py` 补 3 例:文件序前 N 张截取(n=1 只留第 1 张 / n=2 留前 2 / n 大于重复数全留,乱序进场证明确按文件名排序)/ 非法 n 中文报错(0、true、1.5)/ 按位与后训练验证姿态零交集回归(铁律消费位,T-QK-02)
- 验收:`tests/test_evaluation_split.py` 22 passed;experiments.py 内掩码出现 9 处(>= 3 门)

### Task 3:实验级测试更新与收口(提交 81a5bc3 + 4b6234b)

- bootstrap 既有断言改新口径:测试改名 `test_bootstrap_records_every_rep_and_takes_first_exposure_per_pose`,断言 `n_rows == len(set(pose_ids))`(默认每姿态 1 张,不再 × 2),docstring 同步
- 新增三路实验级测试:旋钮生效(exposures_per_pose=2 → 每次求解训练行数 = 去重姿态数 × 2,meta 记录口径)/ 确定性(captured 的 image_ids 恒为每姿态文件序第 1 张的集合 + 同种子两次 bootstrap 逐位一致)/ CV 折级行数口径(训练侧 = 训练姿态数 × exposures_per_pose,验证侧 = 考题姿态数 × 2 全连拍)
- slices 豁免回归:默认 / exposures_per_pose=1 / =8 三种配置下 `slice_groups_by_index` 输出逐位一致(新旋钮对分组实验完全免疫)
- `tests/test_io_config.py`:默认契约 25→26 锚定(新增 exposures_per_pose 默认 1 断言);新校验测试(0 / -1 / true / 1.5 中文报错点名,1 与 8 正常解析)
- 收口:`uv run pytest tests -q -m "not integration"` → **605 passed, 0 失败**;`ruff check src tests` 零告警(uv 与 .venv 双入口各验一次)

## 与计划的偏差(自动处理)

**1. [Rule 3 - 阻塞修复] test_io_config.py 的真实 yaml 锚点随 Task 1 同步**
- 计划把 test_io_config.py 排在 Task 3,但 Task 1 的验收命令要求 test_io_config.py 全绿,而 `test_evaluate_repo_yaml_values` 锚定真实 yaml 的 bootstrap_n=100、quick 50/30——yaml 一下调它就红。锚点同步(100→30、50/30→10/10、补 exposures_per_pose=1)进 Task 1 提交;dataclass 默认值断言按计划原样不动(默认 200/50/30 保持绿,验证"yaml 是操作真源"的既有先例)。

**2. [小样数据现象,不为绿改语义] 5 处既有测试钉 exposures_per_pose=2 保持改前训练规模**
- 全套跑红后逐个对账:小样合成数据(6 姿态 × 2 张)上每姿态 1 张 = 3 行训练,m2 会大面积触发求解器"实用收敛"误报(status=0、回代 RMS 已为 0 但代价尾部相对改进超阈)——这是既有失败契约里"小训练集不收敛是数据现象"的同类表现,不是接线 bug。考折级记账 / 聚合算术 / 离群点名 / 防泄漏计数 / 全链冒烟的 5 处测试(`test_cv_fold_samples_and_zero_overlap`、`test_cv_pose_level_aggregation`、`test_cv_validation_extrinsics_never_trained`、`test_lopo_outlier_pose_flagged`、smoke_setup)把 `exposures_per_pose=2` 钉进配置(理由入注释),测试语义与改前完全一致;默认 1 张下的训练侧行数口径由新增的 `test_cv_train_side_rows_scale_with_exposures_per_pose` 专项另考。计划原文"按新口径更新硬编码 n_train_views"落在该专项测试与 bootstrap 改名测试上。
- `tests/test_evaluation_smoke.py` 不在计划文件清单里,冒烟地基与 CV 小样同前提,同理由补钉(偏差 2 的连带)。

**3. [Rule 1 - 口径一致性] 编排层 TaskFailed 兜底的 n_train_views 同步改按位与口径**
- 计划点名了 worker 内 except 分支(416 / 673 行),但 `cv_pose_grouped` 与 `lopo` 编排层的 TaskFailed 占行也全量计数——不改会出现"worker 失败记降载行数、编排层失败记全连拍行数"的同表两种口径。两处一并改为按位与后的计数(task[11] / task[9] 取元组里的 n_exposures)。

**4. [Rule 1 - 文档同步] splitting.py `bootstrap_resample` docstring"抽中即带其全部连拍"更新**
- 该句被本次改动废弃,同步为"训练侧每姿态取文件序前 exposures_per_pose 张(2026-10-22 降载裁定)"。

**5. [Rule 1 - 文档同步] experiments.py 两处写死次数的 docstring 数字中性化(提交 4b6234b)**
- 模块头"先跑 CV(100 次)校准耗时,再放 bootstrap 大批量(800 次)"与 lopo docstring"这 30 个姿态合群":非判读文案、计划未点名,但次数下调后已成陈旧数字,按"引用数字先核对"纪律一并中性化(次数以 config 为准)。

## 铁律与豁免核对(计划核验项)

| # | 核验项 | 结果 |
|---|--------|------|
| 1 | 非集成全套 + ruff | **605 passed** + 51 deselected;ruff 零告警 |
| 2 | 真实 yaml 四值 | exposures_per_pose=1、quick_bootstrap_n=10、bootstrap_n=30、quick_lopo_n=10 |
| 3 | slices 路线零改动 | `git diff` 无 slice 函数体;默认/1/8 三配置分组输出逐位一致(回归测试钉死) |
| 4 | 三个实验 meta 记录新键 | cv / lopo / bootstrap meta 各含 `exposures_per_pose` |
| 5 | 噪声 / Fisher / calibrate / analyze 不受影响 | 源码零改动(本任务只动 io/config、evaluation/splitting、evaluation/experiments 的三 worker 与 analysis/sensitivity 的一句文案) |
| 6 | 确定性 | 取前 N 张按文件名排序,无新随机源;同种子 bootstrap 重跑逐位一致测试在位 |

## 威胁缓解核对

- T-QK-01(篡改 / 非法 exposures_per_pose):配置层 + 掩码函数体双道校验,0 / -1 / true / 1.5 四路中文报错测试钉死
- T-QK-02(篡改 / 训练验证边界):按位与只发生在训练侧;`split_by_pose_ids` 零重叠断言未动;按位与后两侧姿态零交集有显式回归测试
- T-QK-03(抵赖 / 血缘):三个实验 meta 记录 exposures_per_pose,读产物的人能知道训练规模口径

## 测试结果汇总

- `uv run pytest tests -q -m "not integration"` → **605 passed**(本轮新增 9 例:split 3 + experiments 4 + io_config 2;基线 596 = 05-06 收口 594 + 评审修复轮补的 2 例),51 个集成标记用例按惯例跳过
- `uv run ruff check src tests` / `.venv/bin/ruff check src tests` → 零告警
- 集成套件对账:见下节

## 集成套件对账

执行收口时点:非集成门(本任务验收口径)已全绿。另起的集成全量轮(`-m "integration"`,含真实数据 `calib evaluate` 端到端,正好压过本次改动的代码路径)跑了约 50 分钟进入真实 evaluate 阶段后被运行环境中止,**未产出最终对账**——本 SUMMARY 不对集成结果下结论,建议编排层/下一轮按 deferred-items 既有 3 个失败基线(血缘断链 ×2 + 注入老测试 ×1)补一次"有无新增失败"对账。

低风险依据(仅供对账参考,非结论):(a) evaluate 产物 meta 只增键不锁键集合,消费面全是 setdefault/取值;(b) analysis / report 层无任何 n_train_views 行数假设(grep 为空);(c) findings 结论 1 的 8 组 80 行独立标定在本批真实数据上全部收敛,80 行训练规模可解性有数据背书;(d) m3/m4 小训练集不收敛走既有失败契约占行,不会崩轮;(e) 换新数据后的第一次 `calib evaluate` 真实轮天然就是对本次改动的端到端检验,run meta 里可核对 exposures_per_pose=1 落账。

## 遗留事项

- 无阻塞性遗留。换新数据时 evaluate 的耗时收益预期约 8 倍(训练侧 640→80 行),首跑真实数据时可在 run meta 里核对 exposures_per_pose=1 落账
- 集成全量轮被运行环境中止未出对账(见"集成套件对账"节):建议下一轮补跑 `-m "integration"` 对账既有 3 失败基线,确认无新增

## Self-Check: PASSED

- 9 个改动文件全部在盘(config/calibration.yaml、src/calib/io/config.py、src/calib/evaluation/splitting.py、src/calib/evaluation/experiments.py、src/calib/analysis/sensitivity.py、tests/test_io_config.py、tests/test_evaluation_split.py、tests/test_evaluation_experiments.py、tests/test_evaluation_smoke.py)
- 4 个提交(b35fc1b / d3c0d71 / 81a5bc3 / 4b6234b)全部在 git 历史
- 工作区无本任务遗留改动(仅剩 .planning/quick/261022-ikq-evaluate-1-bootstrap-lopo/ 未跟踪目录,归编排层 docs 提交)
