---
quick: 261022-ikq-evaluate-1-bootstrap-lopo
plan: 261022-ikq
type: execute
wave: 1
depends_on: []
files_modified:
  - config/calibration.yaml
  - src/calib/io/config.py
  - src/calib/evaluation/splitting.py
  - src/calib/evaluation/experiments.py
  - src/calib/analysis/sensitivity.py
  - tests/test_io_config.py
  - tests/test_evaluation_split.py
  - tests/test_evaluation_experiments.py
autonomous: true
must_haves:
  truths:
    - "evaluate 的交叉验证 / bootstrap / 留一姿态实验,训练侧每姿态只取文件序前 exposures_per_pose 张(默认 1),单次求解训练规模从约 640 行降到约 80 行量级(约 8 倍提速)"
    - "验证侧(考题姿态)仍带全部连拍;训练/考题零重叠铁律不破坏(组内抽样不跨边界)"
    - "取'文件序前 N 张'是确定性操作,不引入新随机源——同配置重跑逐位一致"
    - "重复张分组(slices)实验行为与现在逐位一致;噪声估计 / Fisher / calib calibrate 主标定 / calib analyze 全部不受影响"
    - "次数下调:quick 档 bootstrap 10 次、留一抽查 10 个;full 档 bootstrap 30 次"
    - "三个实验的 meta 记录 exposures_per_pose(产物可追溯)"
  artifacts:
    - path: "config/calibration.yaml"
      provides: "evaluate.exposures_per_pose: 1 新键(含依据注释与豁免清单)+ 次数下调三处"
      contains: "exposures_per_pose"
    - path: "src/calib/io/config.py"
      provides: "EvaluateConfig.exposures_per_pose 字段 + >= 1 中文校验"
      contains: "exposures_per_pose"
    - path: "src/calib/evaluation/splitting.py"
      provides: "exposures_per_pose_mask 纯函数辅助(每姿态文件序前 N 张掩码)"
      contains: "exposures_per_pose_mask"
    - path: "src/calib/evaluation/experiments.py"
      provides: "CV 折 / bootstrap / 留一姿态三个 worker 训练侧走新辅助 + meta 记录"
      contains: "exposures_per_pose_mask"
  key_links:
    - from: "src/calib/evaluation/splitting.py"
      to: "src/calib/evaluation/experiments.py"
      via: "三个 worker 训练侧组装 pose_row_mask & exposures_per_pose_mask 按位与"
      pattern: "exposures_per_pose_mask"
    - from: "src/calib/io/config.py"
      to: "src/calib/evaluation/experiments.py"
      via: "cfg.evaluate.exposures_per_pose 经任务元组传入 worker(worker 是顶层函数,不能闭包 cfg)"
      pattern: "exposures_per_pose"
---

<objective>
evaluate 实验降载:训练侧从"带全部连拍"降到"每姿态 1 张"(640→80 行量级,约 8 倍提速),无特殊要求的重复次数同步下调。

背景:换新数据时 evaluate 太慢。bootstrap 已按姿态有放回抽,但 worker 取抽中姿态的**全部连拍**(每次约 405 行);CV 折与留一姿态训练侧同样带全部连拍(约 632 行)。统计依据是 docs/findings.md 结论 1:每姿态 1 张与连拍 8 张无实质差别(参数晃动主要是位姿抽样,不是检测噪声)。

豁免清单(一行不动):重复张分组 slices(它本身就是量连拍效应)、噪声估计、Fisher、calib calibrate 主标定、calib analyze、交叉验证折数/轮数(选型证据的误差棒)。

用途:输出给 execute 提示词,三个任务按序执行。
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/STATE.md
@docs/findings.md
@config/calibration.yaml
@src/calib/io/config.py
@src/calib/evaluation/splitting.py
@src/calib/evaluation/experiments.py
@tests/test_evaluation_experiments.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: 配置旋钮 exposures_per_pose 与重复次数下调</name>
  <files>config/calibration.yaml, src/calib/io/config.py, src/calib/analysis/sensitivity.py</files>
  <action>
1. config/calibration.yaml 的 evaluate 段(第 195-230 行一带),在 `cv_n_rounds` 与 `bootstrap_n` 之间插入新键 `exposures_per_pose: 1`,通俗中文注释写三层意思:
   a) 白话定义:交叉验证 / bootstrap / 留一姿态三类实验的**训练侧**,每个姿态最多取"文件序前 N 张"连拍(文件名含时间戳,按文件名排序就是拍摄时间序),确定性、不引入新随机源;验证侧(考题姿态)不受影响,仍带全部连拍。
   b) 依据:docs/findings.md 结论 1——每姿态 1 张与连拍 8 张无实质差别,参数晃动的大头是位姿抽样不是检测噪声;训练侧带全部连拍只白费约 8 倍算力。
   c) 豁免清单:重复张分组(slices)、噪声估计、Fisher、calib calibrate 主标定、calib analyze 全部不受影响;交叉验证折数/轮数是选型证据也不动。
2. 同段次数下调(yaml 是操作真源;config.py 的 dataclass 默认值不动——既有先例 yaml bootstrap_n=100 vs 默认 200 已允许分档,test_io_config 对默认值的断言保持绿):
   - `bootstrap_n: 100` → `30`,注释补一句:30 次给极值统计留尾巴;quick 档只有 10 次,最大位移(max_disp_px,即参数扰动换算成的最大投影位移)按下界读,只看量级别抠小数。
   - `quick_bootstrap_n: 50` → `10`,注释同步。
   - `quick_lopo_n: 30` → `10`,注释同步。
3. src/calib/io/config.py:
   - EvaluateConfig 在 `cv_n_rounds` 字段之后、`bootstrap_n` 之前加字段 `exposures_per_pose: int = 1`,通俗中文注释(定义 + 依据 findings 结论 1 + 1 = 每姿态 1 张)。
   - `_parse_evaluate` 里加校验,体例照第 1028-1034 行 quick_* 键循环:整数、先拒 bool 伪装、>= 1,违规经 `_fail` 中文报错(报错文案点名 evaluate.exposures_per_pose)。
4. 判读文案核对:src/calib/analysis/sensitivity.py 第 7 行模块 docstring 写着"50 次重标定的实际散布"——次数已硬编码,改成不带数字的表述(如"多次重标定的实际散布");再 grep 全 src/ 确认没有其他把 bootstrap/lopo 次数写死的判读文案(cli/main.py 第 1375-1376、1493 行播报读的是配置值本身,核对即可不必改)。
  </action>
  <verify>
    <automated>uv run pytest tests/test_io_config.py -q && uv run python -c "from calib.io.config import load_calibration; c = load_calibration('config/calibration.yaml'); assert c.evaluate.exposures_per_pose == 1 and c.evaluate.quick_bootstrap_n == 10 and c.evaluate.bootstrap_n == 30 and c.evaluate.quick_lopo_n == 10, (c.evaluate.exposures_per_pose, c.evaluate.quick_bootstrap_n, c.evaluate.bootstrap_n, c.evaluate.quick_lopo_n)"</automated>
  </verify>
  <done>yaml 四个值落位且注释齐全;config 字段与校验生效(0 / true / 1.5 会被中文报错拦下);sensitivity.py 不再有硬编码次数文案;test_io_config 全绿。</done>
</task>

<task type="auto">
  <name>Task 2: 切分辅助 exposures_per_pose_mask + 三个 worker 训练侧接线</name>
  <files>src/calib/evaluation/splitting.py, src/calib/evaluation/experiments.py, tests/test_evaluation_split.py</files>
  <action>
1. src/calib/evaluation/splitting.py 新增纯函数 `exposures_per_pose_mask(obs: ObservationSet, n_exposures: int) -> np.ndarray`:
   - 归组体例照 `_pose_groups`;每组行按 `obs.image_ids` 排序(文件名含时间戳,排序即时间序——与 slice_groups_by_index 用同一把尺,严禁第二份排序逻辑),保留前 n_exposures 张,返回长度 = 观测集行数的布尔掩码。
   - 校验:n_exposures 必须是 >= 1 的整数(先拒 bool),违规中文报错并点名 evaluate.exposures_per_pose。
   - docstring 必须写清与铁律的关系:本辅助只做**训练侧组内抽样**——同一姿态内部取前 N 张,绝不跨训练/考题边界;调用方先用 pose_row_mask 圈定训练姿态,再与本研究掩码按位与;验证侧永远不带它。split_by_pose_ids 的零重叠断言照旧生效,铁律表述("切分只操作姿态 ID、连拍同进同出")在验证侧语义上原样保留。
2. src/calib/evaluation/experiments.py 三处接线。张数从 `cfg.evaluate.exposures_per_pose` 经**任务元组**传入 worker(worker 是顶层函数才能被并行序列化,不能闭包 cfg),导入加 exposures_per_pose_mask:
   a) `_cv_fold_worker`(第 326 行):task 元组末尾追加 n_exposures;训练侧组装第 353 行改为 `obs_rows(obs, pose_row_mask(obs, train_poses) & exposures_per_pose_mask(obs, n_exposures))`;val_obs 一行不动;except 兜底分支第 416 行的 n_train_views 同口径改用按位与后的掩码,不许仍旧全量计数。
   b) `_bootstrap_worker`(第 1334 行):task 元组追加 n_exposures;第 1348 行 train_obs 同式按位与;docstring"抽中姿态去重后带全部连拍"更新为"去重后每姿态取文件序前 n_exposures 张(2026-10-22 降载裁定,依据 findings 结论 1)"。
   c) `_lopo_worker`(第 627 行):task 元组追加 n_exposures;保留 `split_by_pose_ids`(零重叠门禁 + 验证侧评分都靠它),对返回的 train_obs 再套 `obs_rows(train_obs, exposures_per_pose_mask(train_obs, n_exposures))`;except 分支第 673 行 n_train_views 同口径。
3. 三个编排函数 `cv_pose_grouped` / `bootstrap_pose` / `lopo` 的任务元组生成处各追加 `eval_cfg.exposures_per_pose`;三处 meta 各加 `"exposures_per_pose": int(eval_cfg.exposures_per_pose)`(bootstrap meta 在第 1537 行一带,血缘留痕体例)。
4. docstring 同步:cv_pose_grouped 第 448 行"全部连拍"、lopo/bootstrap docstring 的训练侧表述,统一改为"训练侧每姿态取文件序前 N 张(exposures_per_pose,默认 1),验证侧仍带全部连拍"。
5. slices 侧一行不动:`_slice_group_worker` / `slices_by_index` / `slice_groups_by_index` 是豁免位,严禁接线。
6. tests/test_evaluation_split.py 补辅助单测(纯函数,合成 obs 每姿态 2-3 张):
   - 掩码只留每姿态文件序第 1 张(n=1);n=2 留前 2 张;n 大于重复数时全留;
   - n_exposures=0 或 true 触发中文报错;
   - 与 pose_row_mask 按位与后,训练/验证两侧姿态交集为空(铁律消费位回归)。
  </action>
  <verify>
    <automated>uv run pytest tests/test_evaluation_split.py -q && grep -c "exposures_per_pose_mask" src/calib/evaluation/experiments.py | xargs test 3 -le</automated>
  </verify>
  <done>辅助函数落位且单测全绿;三个 worker 训练侧行数 = 训练姿态数 × exposures_per_pose(默认 1),验证侧与 slices 逐字未动;meta 记录新键。</done>
</task>

<task type="auto">
  <name>Task 3: 实验级测试更新与收口</name>
  <files>tests/test_evaluation_experiments.py, tests/test_io_config.py</files>
  <action>
1. tests/test_evaluation_experiments.py 更新既有断言:
   - 第 739-743 行 `test_bootstrap_records_every_rep_and_picks_all_burst_views`:默认 exposures_per_pose=1 下,断言改为 `n_rows == len(set(pose_ids))`(每姿态 1 张);测试名与 docstring 同步改口径(不再叫"带全部连拍"),spy_solve 已捕获行数可直接复用。
   - 通读 CV 与 LOPO 既有测试,凡硬编码 n_train_views 数值或"训练带全部连拍"注释的,按新口径更新(先跑一遍测试看哪些红,逐个对账改断言,不许为绿改语义)。
2. 新增实验级测试(体例照同文件既有测试):
   - 旋钮生效:构造配置把 exposures_per_pose 设为 2(参考 `_boot_cfg` 的 dataclasses.replace 体例),断言 bootstrap 每次求解训练行数 = 去重姿态数 × 2。
   - 确定性:同一姿态多张只取文件序第 1 张——对 captured 的 image_ids 集合断言恒为每姿态按文件名排序的第一张;同种子两次 bootstrap_pose 跑 `_assert_reports_equal` 逐位一致(体例照 test_bootstrap_same_seed_reruns_bit_identical)。
   - CV 侧:一例断言折级训练侧行数 = 折内训练姿态数 × exposures_per_pose,且 n_val_views 与改前口径一致(验证侧全连拍)。
   - slices 豁免回归:exposures_per_pose=1 与默认两种配置下,`slice_groups_by_index` 输出的组数与组内容完全一致(新旋钮不进分组实验)。
3. tests/test_io_config.py 新增校验测试(体例照第 774 行 quick_lopo_n: 0 的写法):exposures_per_pose 写 0 / -1 / true / 1.5 各触发中文报错且报错文案含 "exposures_per_pose";合法值 1 与 8 正常解析。
4. 全套收口:uv run pytest -q 单元层全绿(基线 594 passed + 本轮新增),ruff check src tests 干净。
  </action>
  <verify>
    <automated>uv run pytest -q 2>&1 | tail -3 && uv run ruff check src tests</automated>
  </verify>
  <done>全套单元测试通过(旧断言更新到位、新旋钮/确定性/豁免回归三路测试齐全),ruff 无告警;slices 行为与改前逐位一致。</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| 边界 | 说明 |
|------|------|
| yaml 手写配置 → 实验编排 | 手填的 exposures_per_pose / 次数若为 0 或负数,会让训练侧为空、求解当场崩或静默出垃圾结果 |

## STRIDE 威胁登记

| 威胁 ID | 类别 | 组件 | 处置 | 缓解措施 |
|---------|------|------|------|----------|
| T-QK-01 | 篡改 | exposures_per_pose 校验 | mitigate | config 解析层 >= 1 整数校验(先拒 bool),splitting 辅助函数体内再校验一道,违规中文报错 |
| T-QK-02 | 篡改 | 训练/验证边界 | mitigate | 掩码按位与只发生在训练侧;split_by_pose_ids 零重叠断言原样保留,测试对按位与后两侧姿态交集显式断言 |
| T-QK-03 | 抵赖 | 产物血缘 | mitigate | 三个实验 meta 记录 exposures_per_pose,读产物的人能知道训练规模口径 |
</threat_model>

<verification>
1. `uv run pytest -q` 全绿(基线 594 + 新增),ruff 干净
2. `load_calibration('config/calibration.yaml')` 四值断言:exposures_per_pose=1、quick_bootstrap_n=10、bootstrap_n=30、quick_lopo_n=10
3. grep 确认 slices 路线源码与测试零改动:`git diff --stat` 不出现 slices 相关函数体
4. meta 含 exposures_per_pose 键(三个实验)
</verification>

<success_criteria>
- 交叉验证 / bootstrap / 留一姿态训练侧每姿态只取文件序前 1 张(默认),训练规模 640→80 行量级;验证侧与改前完全一致
- slices / 噪声 / Fisher / 主标定 / analyze 零影响(源码未动 + 回归测试钉死)
- quick 档 bootstrap 10 次、留一抽查 10 个;full 档 bootstrap 30 次;无硬编码次数判读文案
- 同配置重跑逐位一致;产物 meta 可追溯训练口径
</success_criteria>

<output>
Create `.planning/quick/261022-ikq-evaluate-1-bootstrap-lopo/261022-ikq-SUMMARY.md` when done(体例照既有 quick SUMMARY,提交信息中文、体例照 git log:feat(quick)/test(quick))
</output>
