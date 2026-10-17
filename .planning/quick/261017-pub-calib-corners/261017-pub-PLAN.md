---
phase: quick-261017-pub
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/calib/visualization/overlay.py
  - src/calib/cli/main.py
  - tests/test_visualization.py
autonomous: true
requirements: [quick-261017-pub]
tags: [corners, review-folder, visualization]

must_haves:
  truths:
    - "体检判存疑的帧(当前 89 张)在 review_suspects/ 一个文件夹内看全:原图副本 + classic/sb 两套标记图 + 逐张中文判定原因,不需要跨目录翻找"
    - "连拍一致性断言点名的帧(两颗检测器并集,当前 24 张)在 review_burst_flagged/ 同样看全,原因含每颗点名检测器的组内波动中位数/最大值与哪个指标超限"
    - "review_sb_failed 功能回归不受影响(当前 41 张,计数与结构不变)"
    - "清单为空不建目录;summary.json 键名用英文、值和说明用中文;源图自始至终不被改写"
  artifacts:
    - path: "src/calib/visualization/overlay.py"
      provides: "通用双检测器复核文件夹渲染函数(与 render_sb_failed_review 同层同风格)"
      contains: "def render_flagged_review"
    - path: "src/calib/cli/main.py"
      provides: "存疑帧/连拍点名帧的清单组装 + 中文原因拼装 + corners 流程接线"
      contains: "review_suspects"
    - path: "tests/test_visualization.py"
      provides: "新函数与两个组装帮手的行为锁定测试(与上一任务 review_sb_failed 测试同构)"
      contains: "test_render_flagged_review"
  key_links:
    - from: "src/calib/cli/main.py (corners 命令)"
      to: "src/calib/visualization/overlay.py"
      via: "render_flagged_review 调用 ×2(folder 分别为 review_suspects / review_burst_flagged)"
      pattern: "render_flagged_review\\("
    - from: "src/calib/cli/main.py (_suspect_review_frames / _burst_review_frames)"
      to: "manifest 的 precheck 字段 / detection burst_failures"
      via: "verdict == 'suspect' 筛选与 suspect_image_ids 并集去重"
      pattern: "suspect_image_ids"
    - from: "标记图 sb 路线"
      to: "arrays['sb'] 最终角点(含 classic 兜底帧)"
      via: "detected 数组判定,与 marked_copies/sb 同源同判据"
      pattern: "detected"
---

<objective>
为 `calib corners` 增加两个自动生成的"专属复核文件夹",解决用户批评的"拼图给编号 → 跨目录按文件名翻图"问题:

- `review_suspects/`:体检判存疑的帧(manifest 里 precheck.verdict == "suspect",当前数据 89 张)
- `review_burst_flagged/`:连拍一致性断言点名的姿态组的帧(classic 与 sb 两颗检测器各自点名清单取并集,当前数据 3 姿态 × 8 张 = 24 张)

每个文件夹内齐备:original/ 原图副本、marked_classic/ 与 marked_sb/ 两颗检测器各自的角点标记图(未检出画 MISS 红框)、summary.json 通俗中文逐张判定原因。与上一快任务(261017-mmj)的 review_sb_failed 完全同构,直接复用它的函数结构与测试风格。

Purpose: 让用户一个文件夹看完人工复核,不再跨目录按文件名翻找(用户明确批评过的流程)。
Output: overlay.py 新渲染函数 + main.py 清单组装与接线 + 同构测试 + 全量重跑后的新 run 产物断言。
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@./CLAUDE.md(通俗中文硬性要求:注释/CLI 文案/测试注释全部通俗中文,术语首现白话解释)
@.planning/STATE.md

@src/calib/visualization/overlay.py(render_marked / render_all / render_sb_failed_review——直接模板,新函数放它后面同层)
@src/calib/cli/main.py(corners 命令;上一任务在 render_all 之后、write_mosaics 之前接入了 review_sb_failed,本任务同点位再接两个文件夹)
@tests/test_visualization.py(上一任务的 render_sb_failed_review 测试在 214–272 行,新测试与它同构)

数据事实(写断言用,来自当前产物):
- output/ingest/run_20261016_085253/manifest.json:705 张,其中 89 张 precheck.verdict == "suspect";每张含 path/split/pose_id/repeat_idx/n_repeats/precheck{suspect_sources, blur_norm, expo_hi, expo_lo, verdict};suspect_sources 实际取值只有 4 种:pose_relative:blur(18)、pose_relative:exposure(40)、global:blur(16)、global:exposure(28),一张可同时占多个来源
- output/corners/run_20261017_083337/detection_summary.json:burst_check.failures 共 6 条(3 姿态 × classic/sb 各一条),每条含 detector/pose_id/suspect_image_ids(每姿态 8 张,两颗检测器点名的是同样 24 张,并集去重后 24)/spread_median_px/spread_max_px/threshold_px[中位容差, 最大容差];当前数据全部是"中位数超限、最大值未超"
- review_sb_failed 回归基线:41 张(original/ 41 + marked/ 41)
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: overlay.py 新增通用复核文件夹渲染函数 render_flagged_review + 测试</name>
  <files>src/calib/visualization/overlay.py, tests/test_visualization.py</files>
  <behavior>
    - 测试 1(两帧内容):2 帧清单(A 帧 classic/sb 都检出带角点、B 帧两颗都未检出)→ original/ 存 2 张原图副本(保持原文件名、逐字节一致);marked_classic/ 与 marked_sb/ 各 2 张标记图(B 帧两套都是 MISS 红框:顶边整行接近纯红;A 帧非红框);summary.json:n_total == 2、marked/miss 按检测器计数正确、frames 数组每条含 image_id 加上调用方传入的 info 字段(中文值原样落盘)、顶层含调用方传入的 note 中文说明与 extra 计数字段;源图逐字节不变
    - 测试 2(空清单):frames 为空 → 返回 None,且不创建以 folder 参数命名的目录
  </behavior>
  <action>
    在 overlay.py 的 render_sb_failed_review 之后新增函数(与它同层、同风格、同中文 docstring 惯例):

    render_flagged_review(corners_run_dir, folder, frames, board, quality=90, note=None, extra=None) -> dict | None

    - folder:文件夹名("review_suspects" / "review_burst_flagged" 由调用方传,函数本身不写死)
    - frames 每项:image_id(照片编号)、path(原图路径,只读)、uv_classic(classic 路线最终角点或 None)、uv_sb(sb 路线最终角点或 None——含 classic 兜底帧,判据与 marked_copies/sb 一致即"该路线 detected 数组")、note(可选 ASCII 短标,沿用 render_marked 的 note 参数,如 "SUS BLUR+EXPO" / "BURST FLAG";cv2.putText 画不了中文)、info(dict,原样并进 summary 的该帧条目:pose_id、repeat_idx、split、判定原因等)
    - 产物:folder/original/ 逐字节复制原图(shutil.copy2 保持原文件名与格式,同 review_sb_failed);folder/marked_classic/<image_id>.jpg 与 folder/marked_sb/<image_id>.jpg 用 render_marked 画(检出画角点叠加、未检出画 MISS 红框、有 note 加 ASCII 短标)
    - summary.json:键名英文、值中文;顶层 = {"note": 中文说明(参数传入), "n_total": N, "marked": {"classic": x, "sb": y}, "miss": {"classic": m, "sb": m}, "jpeg_quality": Q, **(extra 或 {})};"frames" 数组每条 = {"image_id": ..., **info};返回同一份字典供控制台汇总
    - 空清单直接返回 None 且不建目录(与 review_sb_failed 同约定)
    - 真实性承诺同款:原图只读,副本与标记全画在副本上

    测试加在 tests/test_visualization.py 的 render_sb_failed_review 测试之后(214–272 行区域后面),复用现有夹具 _write_fake_image / _grid_uv / _is_reddish / _write_fake_manifest 与 make_chessboard_image,A 帧可用合成棋盘 + 夹具真值角点;测试注释通俗中文。
  </action>
  <verify>
    <automated>uv run pytest tests/test_visualization.py -q && grep -c "def render_flagged_review" src/calib/visualization/overlay.py</automated>
  </verify>
  <done>render_flagged_review 落地:两套标记目录 + 原图副本 + 中文 summary 全部按参数 folder 命名生成;空清单返回 None 不建目录;全部既有测试不回归;原子提交(feat,遵循仓库 conventional commits 风格,不提交 .planning 文档)</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: main.py 组装两份清单(中文原因)+ corners 流程接线 + 控制台汇总 + 测试</name>
  <files>src/calib/cli/main.py, tests/test_visualization.py</files>
  <behavior>
    - 测试 3(存疑清单组装):假 images(2 张 suspect 带不同来源、其中 1 张多来源,1 张 ok)+ 假 arrays(每颗检测器 uv/detected 数组,sb 的 detected 含"兜底帧"为 True 的情形)→ 只返回 suspect 帧;每帧 info 的 suspect_sources 保留原始 token 可对账,reason 为中文翻译(4 种 token 全覆盖:pose_relative:blur → 同姿态相对比较:模糊度、pose_relative:exposure → 同姿态相对比较:曝光、global:blur → 全局线兜底:模糊度、global:exposure → 全局线兜底:曝光),metrics 里 blur_norm/expo_hi/expo_lo 数值原样;uv_classic/uv_sb 按 detected 取角点或 None;note 是 ASCII 短标(如 "SUS BLUR+EXPO")
    - 测试 4(连拍点名清单组装):假 burst_failures(classic 点名 poseA 的 [a1,a2]、sb 点名 poseA 的 [a2,a3],中位超限/最大未超)→ 返回并集 [a1,a2,a3] 去重且按 manifest 顺序;每帧 flagged_by 列出点名它的检测器(a2 两颗都在);每检测器 reason 中文串含波动中位数与容差数值及"超限"、最大波动与容差数值及"未超"字样;帧带 pose_id/repeat_idx/split;未点名的帧不进清单
  </behavior>
  <action>
    main.py 新增两个模块级帮手(放 _burst_failures_for 附近,docstring 通俗中文):

    1) _suspect_review_frames(images, arrays) -> list[dict]:筛 precheck.verdict == "suspect" 的帧;info = {"pose_id", "repeat_idx", "split", "suspect_sources"(原始 token 数组), "reason"(中文,如 "存疑原因:同姿态相对比较:曝光、全局线兜底:模糊度",多来源用顿号连接), "metrics": {"blur_norm", "expo_hi", "expo_lo"} 数值原样};note 按 token 映射 ASCII(含 blur → "BLUR"、exposure → "EXPO",多来源 "+" 连接,前缀 "SUS ");uv_classic/uv_sb 分别取 arrays["classic"]/arrays["sb"] 的角点(detected 为 False 时给 None;sb 数组的 detected 天然含 classic 兜底帧,与 marked_copies/sb 同判据,不需要额外处理)

    2) _burst_review_frames(images, arrays, burst_failures) -> list[dict]:按 image_id 汇集全部 failures 的 suspect_image_ids(两颗检测器并集去重);每帧 info = {"pose_id", "repeat_idx", "split", "flagged_by"(点名它的检测器列表), "reason"(dict,键为检测器名,值中文串:"连拍断言失败:同名角点组内波动中位数 X.XXpx(容差 Y.YYpx,超限);最大 Z.ZZpx(容差 W.WWpx,未超)"——超限与否由 spread_median_px >= threshold_px[0]、spread_max_px >= threshold_px[1] 现算,三种组合都覆盖措辞:超限/未超/两项都超)};note 固定 "BURST FLAG"

    3) corners() 接线:在 render_sb_failed_review 调用之后、write_mosaics 之前,依次组装两份清单并调用 render_flagged_review(folder="review_suspects" / "review_burst_flagged",quality 取 cfg.visualization.jpeg_quality,note 传中文说明"本文件夹由 calib corners 自动生成……",extra 传顶层计数:存疑的按来源分桶张数(复用 manifest_suspect_counts 的口径)+ 连拍的 n_poses 与点名姿态列表);返回 None 时控制台打"无……帧,未生成 xxx/"(同 review_sb_failed 的措辞惯例),非 None 打计数与目录路径

    4) 同步更新 corners 命令的 help 文案与 docstring 步骤列表:在"SB 失败帧独立证据"之后补上两个新复核文件夹的一句话说明

    测试 3/4 加在 tests/test_visualization.py 新 section"复核文件夹:清单组装(体检存疑 / 连拍点名)"下,直接 from calib.cli.main import 两个帮手(main.py import 无副作用,不需要跑 CLI);假 arrays 用 {"uv": np.full((n, cols*rows, 2), nan) 部分行填角点, "detected": bool 数组} 形状即可;测试注释通俗中文。
  </action>
  <verify>
    <automated>uv run pytest tests/test_visualization.py tests/test_corners_artifacts.py -q && grep -c "review_burst_flagged" src/calib/cli/main.py</automated>
  </verify>
  <done>两个复核文件夹在 calib corners 流程内 marked_copies 之后自动生成;summary.json 逐张中文原因(存疑 = 来源翻译 + 三个指标数值原样;连拍 = 每颗点名检测器的中位/最大波动与超限结论)+ 顶部计数;控制台中文汇总各一行;空清单不建目录;原子提交(feat,不提交 .planning 文档)</done>
</task>

<task type="auto">
  <name>Task 3: 全量测试回归 + 主树重跑 calib corners + 新 run 产物计数断言</name>
  <files>(不改代码;产物写 output/ 下,git 忽略,不产生提交——除非前两任务有需修补的遗漏)</files>
  <action>
    主树执行(不用 worktree——产物在 git 忽略的 output/ 下,隔离副本会丢产物):

    1) uv run pytest -q 全绿(全仓回归,确认 review_sb_failed 既有功能与 mosaic 等不受影响)
    2) uv run calib corners 全量重跑(分钟级);注意重跑会生成新 run_* 目录,旧 run_20261017_083337 保留不动
    3) 对新 run 目录逐项断言(取 ls -td output/corners/run_* | head -1):
       - review_suspects/:original 89 张、marked_classic 89 张、marked_sb 89 张;summary.json 的 n_total == 89、frames 长度 == 89、按来源分桶计数与 manifest 实际一致(用 python 读 manifest 数 suspect 数交叉验证)
       - review_burst_flagged/:original 24 张、marked_classic 24 张、marked_sb 24 张;summary.json n_total == 24、n_poses == 3、每帧 flagged_by 至少一颗检测器
       - 回归:review_sb_failed/ 的 original 与 marked 各 41 张
       - 流程未变:detection_summary.json 两颗检测器 train 640/640、test 65/65,burst failures 仍 6 条
       - 抽查 summary.json 中文可读:随便翻一条存疑帧与一条连拍点名帧,确认原因含具体数值且用语通俗
    4) 若断言有出入,回到对应任务修代码后重跑,不许手改产物糊弄过去。
  </action>
  <verify>
    <automated>uv run pytest -q && uv run calib corners && RUN=$(ls -td output/corners/run_* | head -1) && test $(ls $RUN/review_suspects/original | wc -l) -eq 89 && test $(ls $RUN/review_suspects/marked_sb | wc -l) -eq 89 && test $(ls $RUN/review_burst_flagged/original | wc -l) -eq 24 && test $(ls $RUN/review_burst_flagged/marked_classic | wc -l) -eq 24 && test $(ls $RUN/review_sb_failed/original | wc -l) -eq 41 && uv run python -c "import json,sys; from pathlib import Path; run=sorted(Path('output/corners').glob('run_*'))[-1]; s1=json.loads((run/'review_suspects/summary.json').read_text()); s2=json.loads((run/'review_burst_flagged/summary.json').read_text()); assert s1['n_total']==89 and len(s1['frames'])==89, s1['n_total']; assert s2['n_total']==24 and s2['extra']['n_poses']==3 if 'extra' in s2 else s2.get('n_poses')==3; print('summary 计数断言通过')"</automated>
  </verify>
  <done>新 run:review_suspects 89 原图 + 89×2 标记 + summary 计数对;review_burst_flagged 24 原图 + 24×2 标记 + summary 对;review_sb_failed 41 回归不受影响;detection_summary 成功率与断言条数与上一 run 一致;全仓测试绿(产物在 output/ 下不进 git,无代码变更则无提交)</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| input_data → output/review_* | 只读源图经 shutil.copy2 进产物目录;无网络、无外部输入 |
| manifest/detection_summary → summary.json | 上游 JSON 字段原样转写,不引入新的不可信输入面 |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-quick-pub-01 | Tampering | 原图副本链路(original/ 与标记图) | mitigate | 沿用既有真实性承诺:源文件只读,测试锁定逐字节不变(test_render_flagged_review 断言 src.read_bytes() 前后一致) |
| T-quick-pub-02 | Repudiation | 复核判定原因的可追溯性 | mitigate | summary.json 同时保留 suspect_sources 原始 token 与中文翻译、连拍原因保留具体数值与容差,可对账 manifest 与 detection_summary |
| T-quick-pub-03 | DoS / 资源 | 89+24 张图的复制与重绘 | accept | 与既有 marked_copies(705×2 张)同量级,分钟级,无新增风险 |
| T-quick-pub-SC | Tampering | 包安装 | accept | 本任务零新增依赖,只复用 cv2/numpy 既有栈 |
</threat_model>

<verification>
1. uv run pytest -q 全仓绿(含上一任务 review_sb_failed 的既有测试,回归)
2. 新 run 目录计数:review_suspects 89/89/89、review_burst_flagged 24/24/24、review_sb_failed 41(回归)
3. summary.json:键名英文、值中文;存疑帧含来源翻译 + blur_norm/expo_hi/expo_lo 原值;连拍帧含每颗检测器的中位/最大波动与超限结论
4. 源图逐字节不变(测试锁定);空清单不建目录(测试锁定)
</verification>

<success_criteria>
- 用户复核存疑帧/连拍点名帧时,只需打开 review_suspects/ 或 review_burst_flagged/ 一个文件夹:原图、两颗检测器标记图、逐张中文原因齐备,零跨目录翻找
- review_sb_failed、marked_copies、mosaic 等既有产物与测试全部不受影响
- 代码注释、CLI 文案、测试注释全部通俗中文,术语首现有白话解释
</success_criteria>

<output>
完成后在 .planning/quick/261017-pub-calib-corners/ 下写 261017-pub-SUMMARY.md(改了什么、断言结果、新 run 目录名)
</output>
