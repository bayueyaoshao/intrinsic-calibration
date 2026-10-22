# 阶段六:报告组装 + CLI 全流水线 + 可复用性硬化 - 研究

**研究日期:** 2026-10-22
**领域:** 报告组装(jinja2 模板)/ CLI 全流水线编排 / 换数据可复用性
**置信度:** HIGH(全部结论来自本仓库代码与最新轮产物实地盘点;jinja2 版本经 uv 实测解析)

<phase_requirements>
## 阶段需求(来自 REQUIREMENTS.md 原文)

| ID | 原文摘要 | 研究支撑 |
|----|---------|---------|
| REP-01 | 可视化原语——角点叠加/残差热图/CV 曲线/CRLB 对比/敏感度排序/摆位覆盖图,matplotlib 统一 PNG | 九成 PNG 已在各阶段落盘(见"数据源盘点"),报告层主要是引用;残差热图无现成图(见开放问题 1) |
| REP-02 | 证据链报告组装(HTML/MD)——各阶段结论、图表、血缘指纹汇总,含最终内参+协方差+选型结论+误差预算表 | report/ 目录空壳;数据全部齐备,组装方案见第一节 |
| REP-03 | CLI 全流水线一条命令(run-all)+ 各阶段独立子命令 + Python 库接口 | run_all 现只串①–⑤;各命令本就是库函数(可直接调用,现有 run_all 即在用) |
| ITER-02 | 问题视图/姿态标记与剔除清单机制,剔除后重标定一键执行 | exclude_images 配置机制已在(calibration.yaml),缺"清单生成"与"重标入口" |
| ITER-03 | 运行清单(run manifest)+ 跨运行对比——报告含本次 vs 上次关键指标 diff | ingest 已有跨轮存疑对比表先例(write_round_diff);报告级需新落 report_data.json |
| ITER-04 | 回环触发条件表内置,报告自动给出"回哪一步、做什么" | pipeline.md 已有回环路标;规则引擎为纯函数,输入 analyze/evaluate 判定段 |
| REU-01 | 姿态数/重复张数/测试集张数全部数据驱动,不硬编码;张数不齐不影响正确性 | grep 证实 src/ 无数据集数字硬编码;不齐路径已实现(见第三节残留清单) |
| REU-02 | 新数据符合目录契约放入后,仅调整 config 即可 run-all 一键完成 | 迁移清单见第三节;验收载体 = 既有 test_smoke_realdata 缩小数据集模式 |
| REU-03 | board.yaml 是棋盘参数唯一来源;角点参数扫描范围随实测方格边长自适应 | 已实现(阶段一);阶段六只需回归验证不被新代码破坏 |

ROADMAP 阶段六成功标准 5 条原文(规划必读):①run-all 从 input_data 直达 output/report(HTML+MD)且每个数字可溯源;②报告每节以判行开头(三判据 PASS/FAIL)+ 内置回环建议;③剔除清单 + 一键重标 + 跨运行 diff;④缩小/变体数据集仅改 config 跑通;⑤各阶段独立运行于既有上游产物(--from/--until 语义)+ Python 库接口有测试。
</phase_requirements>

## 一、报告组装(report 层)

### 现状盘点
- `src/calib/report/` 只有空 `__init__.py`,是阶段一预留的占位;`calib report` 命令在 main.py 里是 `_not_implemented(8, ...)` 占位。
- 结论素材已随段落账:evaluate.json 的 selection 段(选型走查/判行表/两处一致性)、analyze.json 六段(各带中文判读键 verdict_note / extrapolation_note / plain_language / advice / notes),引用时不需要复述 findings.md 的固定数字。
- 图表已齐:corners 拼图 2 张、noise 传递曲线 1 张、fisher 特征谱 16 张、evaluate 3 张、analyze 9 张,全部 PNG 在各轮 figures/ 下。
- jinja2 未进依赖(pyproject 无,venv 实测 ModuleNotFoundError);CLAUDE.md 推荐栈早已指定 jinja2 ≥3.1。

### 推荐做法
1. **两层结构:组装与排版分离。** `report/build.py` 纯函数层负责"读全部上游 JSON → 组装一个 context 字典"(所有中文文案、排名、物理量级换算在此定稿);`report/templates/` 下 jinja2 双模板(report.html.j2 / report.md.j2)只做排版不做计算。好处:模板渲染冒烟测试可以喂合成 context,不碰真实轮。
2. **产物契约:** `output/report/run_<时间戳>/{report.html, report.md, report_data.json, figures/, meta血缘}`。report_data.json 是组装 context 的原样快照——它同时是 ITER-03 跨运行 diff 的数据基础和"每个数字可溯源"的机器可读底账。
3. **报告骨架(证据链叙事,按 pipeline.md 八步顺序):** ①结论速览(当选模型 + 最终内参表 + 不确定度表 + 三判据 PASS/FAIL 判行 + 误差物理量级一句话)→ ②数据与体检(manifest 统计、存疑计数)→ ③角点(双检测器检出率、连拍断言、寻优定案)→ ④噪声底(双口径 σ)→ ⑤理论极限(CRLB 中位数、条件数分档、退化预警)→ ⑥防作弊考试(test RMS 走查表 = 排名、判行表、CV/bootstrap/分组/留一)→ ⑦误差归因(预算比值、覆盖足迹、三分量、敏感度建议、各向异性)→ ⑧回环建议表(ITER-04)→ ⑨产物索引与血缘链 → 附录:上游 figures 复制进 report/figures/ 并编号引用。
4. **物理量级参照零硬编码:** 换算尺 = board.square_size_mm ÷ corners 检测摘要的 sq_px 中位数(本机 25mm/47px ≈ 0.53mm/px),"发丝 ≈ 0.1mm"类参照词写在模板文案里,数值全部 build 层现算——新数据自动得到自己的换算尺。
5. **最终内参 + "协方差":** result.json 没有协方差矩阵,报告给"内参值 + 三列不确定度"合表:CRLB se(crlb.json 理论下界,带 Σ 敏感性带)、bootstrap 经验 σ、重复张分组经验 σ——三个口径并排正是本项目"参数稳定"判据的完整呈现,比裸协方差更可读。
6. **失败轮与缺席段:** 上游 failure_note 非空或 analyze 段为 None → 对应节渲染"缺席 + 原因原文",绝不冒充数值;血缘登记的上游文件已被及时清理删除 → 产物索引节如实披露"该轮上游已清理"(阶段五 deferred-items 的既有教训),只有**直接取数的最新轮主文件缺失**才报错退出。

### 坑与对策
- jinja2 需加入 pyproject 依赖(3.1.6,uv dry-run 实测可解析);HTML 渲染必须开 autoescape,防止产物里的 `<`/`&` 类字符破坏排版。
- 模板内不写任何本数据集数字(findings 结论 1–11 的数字一律从本轮产物读),否则换数据集即出错——这正是 REU-02 的反面。
- analyze 六段可能缺席(meta.missing_sections),模板必须条件渲染;报告层不新画 matplotlib 图(沿用各阶段 PNG),避开中文字体与样式二次踩坑。
- 术语通俗中文硬性:CRLB(理论精度下界)、血缘(产物来源指纹)等首次出现带一句白话——建议模板内置固定术语小节。

## 二、CLI 全流水线(run-all + report 命令)

### 现状盘点
- 子命令 9 个:ingest/corners/noise/calibrate/fisher/evaluate/analyze + report(占位)+ run-all。run-all 现只串 ingest→corners→calibrate→noise→fisher,直接函数调用,靠 typer.Exit 异常冒泡停链,没有逐段中文播报与失败指引;evaluate/analyze/report 未接入。
- 失败契约已统一成形:noise/fisher/evaluate/analyze 失败时"照常落盘失败报告(数值 null + 中文 failure_note)再退出码 1";下游命令读上游时检查 failure_note 与文件存在性,中文报错(不静默在坏输入上跑)。
- 各命令上游定位 = `latest_run_dir`(找最新轮)+ 血缘优先防串轮,已全部实现——"各阶段独立运行于既有上游产物"事实上成立,缺的只是 --from/--until 的流程范围开关。
- run-all 目前没有任何测试(test_cli_main.py 无 run_all 用例)。

### 推荐做法
1. **顺序对齐文档:** run-all 改为 pipeline.md 的 ①ingest→②corners→③noise→④calibrate→⑤fisher→⑥evaluate→⑦analyze→⑧report(现 calibrate 在 noise 前,与文档顺序不一致,统一掉免得报告叙事对不上)。
2. **--from/--until 实现为步骤名切片:** `calib run-all --from calibrate --until report` 只跑区间;每步运行前本就检查上游最新轮(中文报错提示先跑哪步),天然满足成功标准 5。注意 D-12 硬规则"命令行零算法参数"不冲突——这是流程范围参数,不是算法参数。每步仍是直接函数调用(现有模式),不做子进程池(过度设计)。
3. **失败即停 + 断点指引:** 任一步抛 Exit(1) 即停;run-all 在失败时播报"已完成到第几步、产物在哪、恢复用哪条单步命令"(从最新轮目录现读,不额外记账)。"跳过已完成阶段"不建议做自动断点续跑——血缘防串轮已保证重跑安全,自动跳过反而引入"上游变了下游没跑"的静默风险;--from 手动指定区间就是本项目需要的断点续跑粒度。
4. **退出码收口(小改):** ingest/corners/calibrate 现在遇坏输入直接抛 ValueError(traceback + 退出码 1,能用不体面);统一包一层"中文一句话 + Exit(1)",对齐后四步体例。

### 坑与对策
- typer.Exit 在函数直调链中的冒泡行为是 Click 标准行为但未在本仓库被 run-all 实测过(evaluate/analyze 的 Exit(1) 路径目前只在子进程测试里走过)——计划里安排一条"中途失败 run-all 退出码 1"的合成测试验证。
- 全链真数据时长:evaluate quick 本轮 3211 秒 + analyze 15–40 分钟 + calibrate 15–25 分钟,合计约 1.5–2.5 小时;run-all 的 pytest 验收必须走缩小数据集(第五节),不许把 2.5 小时塞进常规测试。
- ITER-02"一键重标"建议形态:报告生成 `suggest_exclude.json`(候选清单 + 每项中文理由,来自 lopo 点名/连拍断言/存疑复核),用户确认后把 image_id 填进 calibration.yaml 的 exclude_images,再跑 `calib run-all --from calibrate`——"一键"= 一条命令重标,不新造剔除文件格式(exclude_images 已是唯一真相源,别做第二套清单机制)。

## 三、可复用性硬化(换数据/换棋盘)

### 现状盘点(grep 实测)
- 数据路径、棋盘规格、全部算法参数只在 config;姿态数/重复张数/测试张数全程数据驱动;张数不齐的既有处理:重复张分组按最少重复数截断、噪声按张数加权/单帧退化、ingest 不要求齐——REU-01 的行为已大体实现。
- 残留的本数据集痕迹(全部可控):
  1. `crlb.json` 观测集键名 `views640`/`poses80`(fisher 模块常量)——换数据后键名字面数字失真,含义是"全帧版/姿态聚合版";改名会破坏 analyze/crosscheck 的读取键与旧轮可比,**建议不改名,报告层加白话注解**。
  2. `evaluate.bic_n_poses: 80`(config 默认)——新数据姿态数不同必须改,进迁移清单。
  3. 各阈值初值(blur_z 3.5、stability_disp_px_max 0.05、radial_bin_edges)标注"待首轮分布校准"——新数据首轮需人工复核,进迁移清单。
  4. 47 个 realdata 测试断言 640/80/65 具体数字——属测试资产非产品代码,但新数据集全链验收不能靠它们,要靠缩小版。
  5. docs/findings.md 的结论数字只属于本数据集——报告层已按"数字只从产物读"设计,天然规避。

### 推荐:新数据迁移清单(REU-02 验收即按它走)
1. 图像放入 `input_data/train|test`(train 每姿态一目录、test 平铺;同一相机同参数);
2. `config/board.yaml` 改棋盘规格(cols/rows/square_size_mm);
3. `config/calibration.yaml` 改:data.train/test 路径、bic_n_poses=有效姿态数、新数据首轮 evaluate/analyze 用 full 档、exposures_per_pose 维持 1(findings 结论 1);
4. `exclude_images` 清空;
5. 旧 `output/` 移走或换新工作目录(路径相对当前目录解析是硬契约);
6. 首轮跑完人工复核四件事:体检分位线、噪声双口径量级、CRLB 条件数分档、选型走查。
建议把这份清单落进 docs/(如 docs/new-dataset.md),报告"产物索引"节也附一句指引。

### 坑与对策
- 张数不齐的极端:单姿态数据会让交叉验证折数(默认 5 或 quick 3)大于姿态数而失败——迁移清单写明"姿态数至少 ≥ 折数×2",报告披露实际姿态数;不必为此改代码。
- 成功标准 4 的"变体"验收:现有 test_smoke_realdata 的缩小集是 3 姿态 × 8 张(齐)——需新增"张数不齐"变体(复制姿态目录后删部分连拍张),断言:run-all 退出码 0、slices 组数 = 最少重复数、noise 单帧折算姿态计数 > 0、标定不崩。

## 四、报告数据源盘点(全部最新轮,均已实地核验键结构)

| 产物(轮次) | 报告消费的关键键 | 加工需求 |
|---|---|---|
| ingest run_20261018_163039/manifest.json | summary{n_train,n_test,n_poses,repeats_distribution,resolution,suspect_counts,detector_success_rates} | 无,直接读 |
| corners run_20261020_102709/detection_summary.json + mosaic/*.png | detection{split}{classic,sb}{success_rate,sq_px}, burst_check | sq_px.median → 物理换算尺(build 层算) |
| noise run_20261018_165412/noise.json + transfer_curve.png | sigma_detector_corner / sigma_temporal_corner / per_source_counts / outlier_poses | 无 |
| fisher run_20261019_025859/crlb.json + figures/spectrum_* | models.{m}.observation_sets.{views640,poses80}.{detector,temporal}{se,condition_number,flag,top_directions,band_*} | 键名白话注解;只引当选模型 2 张谱图 |
| calibration run_20261018_163713/{comparison.json, m2__scipy/result.json} | comparison 12 行;result.json 的 params / rms_overall / per_view_rms / converged | 不确定度合表(拼 CRLB se + bootstrap/slices σ) |
| evaluate run_20261022_071708/evaluate.json + figures/ 3 张 | selection{selected,test_rms_px,walk,verdicts,agreement,three_criteria} / delivery.models.{m}{final_params,rms_px,bic} / cv、bootstrap、slices、lopo、source_separation | 判行表排版;排名表从 test_rms_px 排序 |
| analyze run_20261022_044356/analyze.json + figures/ 9 张 | budget{rows,verdict,lines,recheck} / residual_structure{bands,coverage,extrapolation_note} / test_xyz{per_direction,observability,notes} / sensitivity{advice,uniqueness,px_equivalents} / corner_opt{stats._self,_prod_sb,verdict_note,threshold_px} / anisotropy{summary,plain_language} | 回环建议表(ITER-04 规则引擎);剔除清单候选(ITER-02) |
| report run_*/report_data.json(本阶段新建) | 上一次的 selected / test_rms_px / 预算比值 / sigma_detector | 跨运行 diff(ITER-03);上一份不存在时 diff 节写"首次运行" |

**血缘披露规则(实地核验):当前七个最新轮的 lineage.upstream 指向文件全部在盘,链是完整的。** 报告仍须按"指纹对得上 → 在盘;不在盘 → 如实写已清理"处理,因为及时清理是常态(阶段五 deferred-items 已有断链先例)。对账容差注意:新旧轮实验规模可能不同(findings 结论 3/5 的 bootstrap 50 次 vs 10 次先例),diff 时按 meta 里记录的规模解释差异,不冒充回归。

## 五、测试与验收

- **报告层:** ①模板渲染冒烟(合成 context → 双模板渲染,断言小节标题与关键数字字符串出现、缺席段不崩);②关键数字抽查 = 报告内数字与 report_data.json 逐位一致(不锁固定值,锁"同源一致");③真实最新轮渲染一条,打 realdata 标记。
- **全流水线:** 扩展 test_smoke_realdata 的"临时目录 + 软链接缩小数据集 + 独立进程接力"模式,接到 evaluate→analyze→report→run-all(quick 档);另加"张数不齐变体"档;再加一条"中途失败退出码 1"的合成用例。真数据 2.5 小时全链 run-all 验收由人工 checkpoint 执行,不进 pytest。
- **命令:** 白天 `uv run pytest -m "not realdata"`(609 个,分钟级);夜间/合并前 `uv run pytest`(全量 656)。realdata 标记体例已在 pyproject 注册,沿用。

## 标准栈与包审计

| 库 | 版本 | 用途 | 依据 |
|---|---|---|---|
| jinja2 | 3.1.6(uv dry-run 实测解析) | HTML/MD 模板渲染 | CLAUDE.md 推荐栈指定;PyPI 老牌包(pallets/jinja)slopcheck 判 [OK] |
| (无其他新依赖) | — | 报告不新画图、不引 pandas(表格直接模板排版) | matplotlib/typer/pyyaml 均已在栈 |

## 环境可用性

| 依赖 | 需要 | 在盘 | 版本 | 兜底 |
|---|---|---|---|---|
| Python + uv | 全部 | ✓ | 3.12 / uv 0.11.32 | — |
| jinja2 | 报告渲染 | ✗(venv 无) | 3.1.6 可解析 | 无兜底,须加依赖 |
| pytest realdata 标记 | 验收 | ✓ | 已注册 | — |

## 安全域(ASVS L1,离线本地工具)

| 类别 | 适用 | 处置 |
|---|---|---|
| V5 输入校验 | 最小适用 | jinja2 开 autoescape;JSON 只用 json.load 定长读取,不 eval 产物内容;路径固定在 output/ 下拼装 |
| V2/V3/V4/V6 | 不适用 | 无认证/会话/访问控制/密码学(离线单机 CLI) |

## 假设登记(需用户/讨论确认)

| # | 假设 | 风险若错 |
|---|---|---|
| A1 | typer.Exit 在 run-all 直调链中冒泡即停(Click 标准行为,本仓库未实测过直调链) | 失败即停契约失效;已安排合成测试兜底验证 |
| A2 | crlb.json 键名 views640/poses80 保持不改、报告加注解即可 | 若用户要求改名,需动 fisher/analyze/crosscheck 三处读取键并牺牲旧轮可比 |
| A3 | "残差热图"用现有 radial_bands(残差随半径分带曲线)+ coverage_footprint(空间分带)即可覆盖 | 若用户要逐像素热图,REP-01 需新增一张图(半小时级工作量) |
| A4 | ITER-02"一键"= 填 exclude_images 后 `run-all --from calibrate` 一条命令 | 若用户期望全自动改配置重跑,需扩 CLI 开关(建议拒绝:人工拍板是既定纪律) |

## 开放问题(2026-10-22 用户已裁定,RESOLVED)

1. REP-01 的"残差热图":**现有图算数**——分带曲线 + r⁴ 覆盖足迹图已能看出边缘变差与外推区无观测,报告直接引用,不新画逐像素热图(裁定 A3)
2. 跨运行 diff 的"上次"定义:**取上一份 report 轮**(report_data.json 就是为它建的,与建议一致)
3. ITER-02"一键重标"形态:**配置填剔除名单 + `run-all --from calibrate`**,复用角点/噪声产物(裁定 A4)

## 阶段六计划切分建议(4 个计划)

| 计划 | 目标 | 主要文件 | 依赖 |
|---|---|---|---|
| 06-01 报告组装库层与模板 | report/build.py 组装纯函数 + jinja2 双模板 + `calib report` 落地(HTML/MD/report_data.json/figures 复制)+ 缺席段与失败轮披露 + 报告层测试(合成冒烟 + 真实轮 realdata 一条) | src/calib/report/(build.py、templates/)、cli/main.py report 命令、pyproject(加 jinja2)、tests/test_report_* | 无(纯消费既有产物);先行 |
| 06-02 回环三件(ITER-02/03/04) | 回环建议表规则引擎(纯函数:判行/预算/覆盖/一致性 → 建议)+ 剔除清单生成(suggest_exclude.json)+ 跨运行 diff(读上一份 report_data.json)+ 三块接进报告 | src/calib/report/loopback.py、diff.py、tests/ | 06-01 的 report_data.json 契约 |
| 06-03 CLI 全流水线收口 | `calib report` 接入 run-all(顺序对齐 pipeline.md)+ --from/--until + 失败断点中文指引 + ingest/corners/calibrate 退出码收口 + run-all 合成测试(含失败退出码用例) | cli/main.py、tests/test_cli_run_all.py | 06-01(report 命令存在);与 06-02 可并行 |
| 06-04 可复用性硬化与全链验收 | 张数不齐变体缩小集测试(仅改 config 跑通 run-all)+ 新数据迁移清单文档 + REU-01/02/03 逐条对账 + docs/pipeline.md/REQUIREMENTS 状态同步 + 真数据全链 run-all 人工 checkpoint(四要素指引) | tests/test_run_all_variant.py、docs/new-dataset.md、docs/pipeline.md | 06-01/02/03 全部;末位 |

Wave 结构:06-01 → (06-02 ∥ 06-03) → 06-04。

## Sources

- 本仓库实地盘点:src/calib/cli/main.py(2385 行全读)、src/calib/report/(空)、src/calib/io/manifest.py、config/{board,calibration}.yaml、pyproject.toml、tests/conftest.py、tests/test_smoke_realdata.py、output/ 七个最新轮 JSON 键结构(python json 实读 keys)
- .planning/ROADMAP.md 阶段六节、REQUIREMENTS.md REP/ITER/REU 九行、docs/pipeline.md、docs/findings.md 结论 1–11
- .planning/phases/05-*/deferred-items.md(血缘断链教训)、.planning/config.json(nyquist/security 开关)
- jinja2 3.1.6:slopcheck 0.6.1 判 [OK] + `uv pip install --dry-run` 实测解析(本会话)

## Metadata

**置信度分解:** 现状盘点 HIGH(全部实地读取)/ 方案推荐 HIGH(沿用项目已验证体例:失败契约、缩小集测试、血缘披露)/ typer 直调冒泡 MEDIUM(标准行为但未实测,已排测试)
**研究日期:** 2026-10-22 | **有效期:** 稳定(30 天;产物轮次变化只影响第四节轮名,键结构稳定)
