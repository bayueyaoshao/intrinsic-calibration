---
quick: 261017-mmj-corners-sb-classic-fallback
plan: 261017-mmj
type: execute
wave: 1
depends_on: []
files_modified:
  - src/calib/io/config.py
  - config/calibration.yaml
  - src/calib/corners/artifacts.py
  - src/calib/visualization/overlay.py
  - src/calib/cli/main.py
  - tests/test_io_config.py
  - tests/test_corners_artifacts.py
  - tests/test_visualization.py
autonomous: true
execution_note: 必须在主树执行,禁止隔离工作副本(worktree)——重跑产物写在 git 忽略的 output/ 下,清理动作也针对主树路径
must_haves:
  truths:
    - "SB 未检出而 classic 检出的帧,SB 路线 npz 里该帧有角点、detected=True、source='classic_fallback'"
    - "classic 路线 npz 完全没有 source 键,内容与现状一致(经典路线保持纯净)"
    - "配置 sb_fallback_classic: false 时,SB 路线行为与现在完全一致(缺就缺,source 只有 'sb' 与 '')"
    - "新 run 的 review_sb_failed/ 下有 41 张原图副本 + 41 张标记图(40 张 classic 兜底角点叠加 + 1 张 MISS 红框),并带 DIAGNOSIS.md"
    - "detection_summary.json 的 SB 段同时报 n_detected(含兜底)、n_pure_sb、n_classic_fallback;sb meta.json 记 n_classic_fallback"
    - "旧 run 的 experiments/ 与 review_failed_sb/ 已删除,其余产物(npz×4、meta×4、marked_copies、mosaic、detection_summary.json)一个不动"
  artifacts:
    - path: "src/calib/io/config.py"
      provides: "CornersConfig.sb_fallback_classic 开关 + 中文校验"
      contains: "sb_fallback_classic"
    - path: "config/calibration.yaml"
      provides: "corners.sb_fallback_classic: true 配置项(通俗中文注释)"
      contains: "sb_fallback_classic"
    - path: "src/calib/corners/artifacts.py"
      provides: "save_corners 可选 source 数组参数"
      contains: "source"
    - path: "src/calib/visualization/overlay.py"
      provides: "render_marked 可选标注参数 + render_sb_failed_review 失败帧独立产物"
      contains: "render_sb_failed_review"
    - path: "src/calib/cli/main.py"
      provides: "corners 命令的 classic 兜底填槽、来源留痕、detection_summary 三口径、review_sb_failed 生成"
      contains: "classic_fallback"
  key_links:
    - from: "src/calib/cli/main.py corners() 主循环"
      to: "SB 槽位"
      via: "detect_sb 返回 None 且开关开启且 classic 有结果 → classic 结果填入 sb 槽位并记 source"
      pattern: "classic_fallback"
    - from: "src/calib/cli/main.py"
      to: "save_corners(source=...)"
      via: "仅 SB 路线传 source 数组,classic 调用签名不变"
      pattern: "source=sources\\[rows\\]"
    - from: "src/calib/cli/main.py"
      to: "render_sb_failed_review"
      via: "source 为 classic_fallback 或 '' 的帧逐张传入"
      pattern: "render_sb_failed_review"
---

<objective>
角点检测最终方案落地:SB 为主、classic 兜底(方案点 1),配置开关(方案点 2),SB 路线 npz 逐帧来源留痕 + 三口径统计(方案点 3),SB 失败帧独立产物 review_sb_failed/(方案点 4),主树重跑 calib corners 并验收 705/705(方案点 5),清理三轮实验的中间产物(方案点 6)。

依据:output/corners/run_20261016_085253/review_failed_sb/DIAGNOSIS.md 三轮实验结论——SB 全分辨率 664/705,classic 705/705;41 张 SB 失败帧是大倾斜/远距视角下 SB 几何校验的系统性拒绝(非 bug);"缩小档救援 + 全分辨率精化"与 classic 完全重合(偏差 0.01px 级),故产线不引入多尺度重试,classic 兜底更简单且多覆盖 17 张穷举也救不回的帧。

**执行环境约束(方案点 7,硬性):全部任务在主树执行,executor 不带隔离参数**——重跑产物写在 git 忽略的 output/ 下,隔离副本会在清理时连同产物一起删掉;清理动作也针对主树路径。

Purpose: 角点阶段从"实验对比"转为"生产定案",coverage 705/705,每帧来源可审计,失败帧证据按设计约定独立留存。
Output: 兜底产线代码 + 全绿测试 + 新 run 产物(含 review_sb_failed)+ 清理后的旧 run。
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@./CLAUDE.md
@.planning/STATE.md
@src/calib/cli/main.py            # corners 命令在 236-401 行,本次主改动区
@src/calib/corners/detect.py      # detect_classic / detect_sb:保持两函数纯净,本任务**不改此文件**
@src/calib/corners/artifacts.py   # save_corners/load_corners
@src/calib/io/config.py           # CornersConfig(92 行附近)、_parse_corners(266 行附近)
@src/calib/visualization/overlay.py  # render_marked/render_all,失败帧标记复用这里的绘制逻辑
@config/calibration.yaml          # corners 段(36 行附近)
@output/corners/run_20261016_085253/review_failed_sb/DIAGNOSIS.md  # 三轮实验结论(方案依据,Task 3 迁移它)
@tests/test_corners_artifacts.py
@tests/test_io_config.py
@tests/test_visualization.py
@tests/conftest.py                # make_chessboard_image 合成棋盘夹具(带真值角点)
</context>

<decision_coverage>
| 锁定方案点 | 落在哪个任务 |
|---|---|
| 1. SB 主 + classic 兜底填槽,产线不做多尺度 | Task 2 |
| 2. CornersConfig.sb_fallback_classic 开关 + yaml 配置 | Task 1 |
| 3. npz source 数组 + meta n_classic_fallback + summary 三口径 | Task 1(契约)/ Task 2(接线)|
| 4. review_sb_failed/original + marked 失败帧独立产物 | Task 2 |
| 5. 主树重跑验收(705/705 = 664 纯 SB + 41 兜底) | Task 3 |
| 6. 迁移 DIAGNOSIS.md + 删 experiments/、review_failed_sb/ 余量、/tmp 脚本 | Task 3 |
| 7. 主树执行、不用隔离工作副本 | 全 plan(见 objective 与 execution_note)|
</decision_coverage>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: 兜底配置开关 + SB 路线 npz 的 source 数组契约(含测试)</name>
  <files>src/calib/io/config.py, config/calibration.yaml, src/calib/corners/artifacts.py, tests/test_io_config.py, tests/test_corners_artifacts.py</files>
  <behavior>
    - 测试 1(test_io_config.py):yaml 缺 sb_fallback_classic 键时,解析结果的 corners.sb_fallback_classic 为 True(默认开)
    - 测试 2:显式写 sb_fallback_classic: false 解析为 False
    - 测试 3:写 sb_fallback_classic: 1(整数而非布尔)→ 抛 ValueError,中文报错含字段名 corners.sb_fallback_classic
    - 测试 4(test_corners_artifacts.py):save_corners 传 source=np.array(["sb","classic_fallback",""]) 落盘后 load_corners 读回,三个字符串逐位一致且走 allow_pickle=False 路径(numpy 的定长字符串数组不需要 pickle,反序列化安全测试不破)
    - 测试 5:save_corners 不传 source(经典路线现状调用)→ 读回的 npz 里**没有** source 键(classic 产物保持纯净,方案点 1/3)
  </behavior>
  <action>
    方案点 2 + 3 的契约部分。
    1. src/calib/io/config.py:CornersConfig 增加 frozen 字段 sb_fallback_classic: bool = True(放在 sb_flags 之后);_parse_corners 里解析 corners 段同名键,缺省取 CornersConfig.sb_fallback_classic;校验必须是布尔值(python 里 isinstance(True, int) 也为真,所以先判 bool 再拒其余一切,包括 1/0 整数),不合法走 _fail("corners.sb_fallback_classic", ..., "必须是 true 或 false(SB 未检出的帧是否用经典检测器结果兜底)");构造 CornersConfig 时传入该字段。
    2. config/calibration.yaml:corners 段 sb_flags 行之后加 sb_fallback_classic: true,配通俗中文注释——说明"SB 未检出的帧用经典检测器的结果填进 SB 路线的槽位,每帧来源记在角点文件的 source 数组里;false = 关闭兜底,行为与只有 SB 时完全一致(缺就缺)",并提一句依据(DIAGNOSIS 三轮实验:多尺度救援产物与 classic 完全重合,classic 兜底更简单)。
    3. src/calib/corners/artifacts.py:save_corners 增加可选参数 source: np.ndarray | None = None(放在 detected 之后、lineage 之前);非 None 时 np.savez 多存一个 source 数组(np.asarray 原样,定长 unicode dtype),None 时**完全不写该键**——经典路线调用一个字不改,文件名也不变。模块顶部 docstring 的布局说明里补一行 source 的含义("sb" | "classic_fallback" | "",仅 SB 生产路线有此键)。load_corners 不用改(它按 data.files 全量返回)。
    4. 按上面 behavior 补测试:test_io_config.py 用 tmp_path 写最小 yaml(参照文件内既有测试的写法);test_corners_artifacts.py 复用 burst_of_three 夹具(2 张检出 + 1 张未检出正好对应 "sb"/"classic_fallback"/"" 三种值)。既有测试文件里的老调用不传 source,必须原样通过(向后兼容即测试 5 的另一面)。
    5. 注释/docstring/报错全部通俗中文,术语首现加白话解释,遵循文件既有风格。
    提交:`feat(corners): add sb_fallback_classic switch and npz source array contract`
  </verify>
  <automated>uv run pytest tests/test_io_config.py tests/test_corners_artifacts.py -q</automated>
  </verify>
  <done>
    默认 True;显式 false 生效;非布尔值给带字段名的中文报错;source 数组往返逐位一致且不破 allow_pickle=False 安全测试;不传 source 的调用落盘后无 source 键;老测试全部通过。
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: corners 命令 classic 兜底填槽 + 来源留痕 + review_sb_failed 失败帧独立产物(含测试)</name>
  <files>src/calib/cli/main.py, src/calib/visualization/overlay.py, tests/test_visualization.py</files>
  <behavior>
    - 测试 1:render_marked 传可选标注短文(ASCII)时,副本正常生成、尺寸与原图一致、默认不传时行为与现状逐字节同路径(既有 render_marked 测试不破)
    - 测试 2:render_sb_failed_review 对 2 帧失败清单(1 帧 classic 兜底带角点 + 1 帧两法都未检出)生成 review_sb_failed/original/ 2 张原图副本 + review_sb_failed/marked/ 2 张标记图(兜底帧 = 角点叠加 + 短标;双失败帧 = MISS 红框),summary.json 计数正确且含通俗中文说明
    - 测试 3:render_sb_failed_review 传空清单 → 返回 None,且**不创建** review_sb_failed 目录(方案点 4:没有失败帧不建目录)
  </behavior>
  <action>
    方案点 1 + 3 接线 + 方案点 4。src/calib/corners/detect.py 不改(两颗检测器保持纯净)。
    1. src/calib/visualization/overlay.py:
       a. render_marked 增加可选参数 note: str | None = None——非 None 时在画面左上角画红色 ASCII 短标(字号/线宽沿用 MISS 分支那套按画面尺寸比例推导的写法,不写死像素)。docstring 说明:cv2.putText 不支持中文字形,图内标注只能是 ASCII 短标,完整中文说明写在 summary.json 与控制台(沿用文件头已有的这条承诺)。
       b. 新增函数 render_sb_failed_review(corners_run_dir, frames, board, quality) -> dict | None:frames 是"SB 本体未检出"的帧清单,每项含 image_id、path(原图)、uv_or_none(classic 兜底角点,两法都未检出时为 None)。清单为空直接返回 None、不建目录。非空时:逐帧把原图**复制**(shutil.copy2,保持原文件名与格式)到 review_sb_failed/original/;调 render_marked 把角点叠加(或 MISS 红框)写到 review_sb_failed/marked/<image_id>.jpg,note 用 ASCII 短标,如兜底帧 "SB MISS - CLASSIC FALLBACK"、双失败帧 "SB+CLASSIC MISS";最后写 review_sb_failed/summary.json——含通俗中文说明(本文件夹由 calib corners 自动生成;标记图是"SB 未检出,已用 classic 兜底"的逐帧证据;双失败帧画 MISS 红框)与计数(总数、classic 兜底张数、两法都未检出张数)。返回 summary 字典供控制台汇总。原图只读不改写(真实性承诺 T-05-01 一致)。
    2. src/calib/cli/main.py corners() 主循环(约 277-289 行)重排:每帧先跑 classic 并把结果留在局部变量;SB 检出 → 照常填 sb 槽位,sources[i] = "sb";SB 未检出且 cfg.corners.sb_fallback_classic 为 True 且 classic 本帧有结果 → 把 classic 的 uv/sq_px/win_size 填进 sb 槽位、detected=True、sources[i] = "classic_fallback";两法都未检出 → sources[i] = ""。sources 是 np.full(n, "", dtype="<U16")(仅 SB 路线用;classic 槽位逻辑一字不动)。主循环外层加一个 np 数组 sources,别放进 arrays 字典的 classic 槽。
    3. 落盘段(约 317-327 行):save_corners 仅对 SB 路线多传 source=sources[rows];SB 的 meta_extra 追加 "sb_fallback_classic": cfg.corners.sb_fallback_classic 与 "n_classic_fallback": 该划分内 source=="classic_fallback" 的帧数(方案点 3 的 meta 留痕);classic 的 meta_extra 与调用完全不变。
    4. detection_summary.json(约 330-344 行):SB 段在现有字段外增加 "n_pure_sb"(source=="sb" 计数)与 "n_classic_fallback";n_detected 保持生产口径(detected 数组求和,天然含兜底)。开关关闭时这两个新字段照报(n_pure_sb == n_detected、n_classic_fallback == 0),口径稳定不分支。
    5. 在 render_all 之后调 render_sb_failed_review:frames 取 sources[i] 为 "classic_fallback" 或 "" 的帧(classic_fallback 帧带 classic 的 uv;"" 帧 uv_or_none=None)。控制台中文汇总补两行:SB 生产口径(纯 SB x 张 + classic 兜底 y 张,开关状态),review_sb_failed 路径与张数;corners 命令的 help 文案也补上"SB 未检出时用经典检测器兜底"一句(通俗中文)。
    6. 测试加在 tests/test_visualization.py(overlay 模块的老家):合成棋盘夹具造兜底帧角点,make_tiny_jpg 夹具/临时文件造原图与双失败帧,对照 behavior 三条。
    提交:`feat(corners): classic fallback for SB route with per-frame provenance and review_sb_failed artifacts`
  </verify>
  <automated>uv run pytest tests/test_visualization.py tests/test_corners.py tests/test_corners_artifacts.py tests/test_io_config.py -q</automated>
  </verify>
  <done>
    兜底帧在 SB npz 里有角点、detected=True、source="classic_fallback";classic npz 无 source 键;detection_summary SB 段三口径齐;review_sb_failed 有原图副本 + 标记图 + 中文 summary.json,无失败帧不建目录;cv2.putText 只画 ASCII 短标、中文说明在 summary.json 与控制台;全部相关测试绿。
  </done>
</task>

<task type="auto">
  <name>Task 3: 主树重跑 calib corners 验收 + 清理三轮实验中间产物</name>
  <files>无源码改动;动作对象全部在 git 忽略的 output/ 与 /tmp(本任务**无 git 提交**)</files>
  <action>
    方案点 5 + 6 + 7。全部在主树路径执行(重跑与清理都不进隔离副本)。
    1. 代码已全绿(Task 1/2),在仓库根目录跑 `uv run calib corners`(用最新一轮 ingest 产物 output/ingest/run_20261016_085253/manifest.json,705 张 × 2 检测器,分钟级,耐心等完成)。
    2. 找到新 run 目录(output/corners/ 下时间戳最新的 run_*),核对并断言(用 python 单行脚本读 detection_summary.json 与 npz,数字对不上立即停下排查、绝不带病通过):
       - classic:train 640/640、test 65/65(共 705/705)
       - SB 生产路线:train n_detected 640(n_pure_sb 600 + n_classic_fallback 40)、test n_detected 65(n_pure_sb 64 + n_classic_fallback 1)
       - corners_train_sb.npz 有 source 数组且计数 {sb:600, classic_fallback:40};corners_train_classic.npz **没有** source 键;corners_train_sb.meta.json 记 n_classic_fallback=40
       - review_sb_failed/ 有 41 张原图副本 + 41 张标记图,summary.json 计数一致(40 兜底 + 1 双未检出,双未检出那张是 test 帧 20260722_111500 的 MISS 红框)
       - detection_summary.json 的 burst_check 段照常输出(键存在、失败清单留痕,失败数与上轮口径一致即可)
    3. 迁移诊断记录:把 output/corners/run_20261016_085253/review_failed_sb/DIAGNOSIS.md 复制到新 run 的 review_sb_failed/DIAGNOSIS.md,并在文件开头加一行说明:「本文件夹由 calib corners 自动生成,DIAGNOSIS.md 是实验诊断记录」(方案点 6 原文)。
    4. 删除中间实验产物(用户已明确批准):
       - 删旧 run 整个 experiments/ 目录(4 个实验 json + sb_rescue_marked/)
       - 删旧 run 整个 review_failed_sb/ 目录(DIAGNOSIS.md 已迁移,余下 marked_sb/、original/ 人工补救版已被代码产物取代)
       - rm -f /tmp/exp_a_sb_rescue.py /tmp/exp_b_windows.py /tmp/exp_c_gated.py /tmp/exp_d_coarse2fine.py(-f 幂等,a/b 可能已不在)
    5. 核对旧 run 其余产物一个不动:ls 确认 4 份 npz、4 份 meta.json、detection_summary.json、marked_copies/、mosaic/ 都还在(它们是阶段一验证证据)。
    本任务无 git 提交(全部动作对象在 git 忽略目录与 /tmp);执行结果数字在最终回复里原样报告给用户做人工验收。
  </verify>
  <automated>uv run python -c "import json,glob,numpy as np; r=sorted(glob.glob('output/corners/run_*'))[-1]; s=json.load(open(r+'/detection_summary.json')); z=np.load(r+'/corners_train_sb.npz',allow_pickle=False); c=np.load(r+'/corners_train_classic.npz',allow_pickle=False); import os; d=s['detection']; assert d['train']['classic']['n_detected']==640 and d['test']['classic']['n_detected']==65; assert d['train']['sb']['n_detected']==640 and d['train']['sb']['n_pure_sb']==600 and d['train']['sb']['n_classic_fallback']==40; assert d['test']['sb']['n_detected']==65 and d['test']['sb']['n_classic_fallback']==1; assert 'source' in z.files and int((z['source']=='classic_fallback').sum())==40; assert 'source' not in c.files; assert len(os.listdir(r+'/review_sb_failed/original'))==41 and len(os.listdir(r+'/review_sb_failed/marked'))==41; assert os.path.isfile(r+'/review_sb_failed/DIAGNOSIS.md'); assert not os.path.exists('output/corners/run_20261016_085253/experiments') and not os.path.exists('output/corners/run_20261016_085253/review_failed_sb'); print('OK', r)"</automated>
    <human-check>人工验收清单(用户):① 翻看新 run review_sb_failed/marked/ 的 41 张标记图,确认兜底角点贴得准、双未检出帧红框留痕;② 打开 review_sb_failed/DIAGNOSIS.md 确认迁移完整且开头有说明行;③ 核对控制台输出的 SB 生产口径 705/705(664 纯 SB + 41 兜底)。</human-check>
  </verify>
  <done>
    新 run:classic 705/705,SB 生产路线 705/705(664 + 41),source 留痕与三口径统计正确,review_sb_failed 41+1 套齐全且带 DIAGNOSIS.md;旧 run 只少了 experiments/ 与 review_failed_sb/,其余证据产物原样;/tmp 实验脚本已删;断言脚本输出 OK。
  </done>
</task>

</tasks>

<verification>
1. `uv run pytest -q`(排除依赖真实全量数据的慢冒烟,如 test_smoke_realdata 单跑核实可跑则一并跑)全绿。
2. Task 3 的 python 断言脚本输出 OK(生产口径、留痕、失败帧产物、清理、旧 run 保全六项一次验完)。
3. 不改 src/calib/corners/detect.py(grep 确认 detect_classic/detect_sb 未动,test_detect_sb_source_guard_single_refinement_call 继续通过即为旁证)。
</verification>

<success_criteria>
- 兜底开:SB 路线 coverage 705/705,每帧来源可审计(npz source + meta n_classic_fallback + summary 三口径)
- 兜底关(calibration.yaml 改 false 即可验证,不必真跑 705 张):单测锁定行为等同现状
- review_sb_failed/ 失败帧证据独立留存(原图 + 标记 + 中文 summary + 迁移的 DIAGNOSIS.md),设计约定补账完成
- classic 路线产物零变化;旧 run 阶段一验证证据零损失;实验中间产物(/tmp 脚本、experiments/、人工补救目录)清零
</success_criteria>

<output>
全部任务完成后向用户报告:新 run 路径、三口径数字、review_sb_failed 清单、清理明细,供人工验收。
</output>
