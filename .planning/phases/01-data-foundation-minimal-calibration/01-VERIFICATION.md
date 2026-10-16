---
phase: 01-data-foundation-minimal-calibration
verified: 2026-10-16T09:07:00Z
status: human_needed
score: 5/5
overrides_applied: 0
human_verification:
  - test: "D-05 步骤1:打开 output/corners/run_20261016_085253/mosaic/overview_mosaic.png(164 格),抽查最糊几张的 marked_copies/classic/ 与 marked_copies/sb/ 同名文件并排:角点是否贴准?有无整排系统性偏移?"
    expected: "角点落在黑白格交点上,classic 与 sb 两套基本一致;无整排同向偏移"
    why_human: "角点视觉贴合度只能人眼判断;合成图测试(<0.5px)与重投影 RMS(0.08~0.14px)是间接证据,不能替代真实照片目检"
  - test: "D-05 步骤2:打开 mosaic/suspect_mosaic.png(89 格)与 output/ingest/run_20261016_085253/suspect_originals/ 任意 2-3 张原图:存疑判定有无误杀(清晰被标)/漏杀(明显糊没标)?"
    expected: "存疑清单与人工判断大体一致,误杀/漏杀在可接受范围;有异议则调 config 阈值重跑"
    why_human: "判定阈值是否合理是人工审美与领域判断,程序无法自证"
  - test: "D-05 步骤3:在 marked_copies/ 翻看连拍断言点名的 3 个姿态(20260717_174622 / 20260720_134806 / 20260720_134914)的 8 连拍:组内波动是真实手抖还是排序翻转污染?"
    expected: "确认是连拍间棋盘真实移动(手抖),而非镜像翻转污染"
    why_human: "区分'真实移动'与'排序污染'需要人眼对比连拍序列画面"
  - test: "D-05 步骤4:查看 2×2 对比表(classic_pinhole 0.140 / classic_fisheye 0.133 / sb_pinhole 1.242 / sb_fisheye 0.083 px):量级是否可接受?sb_pinhole 1.24px 是否符合这支广角镜头的直觉?SB 41 张 MISS 是否可接受?"
    expected: "用户认可基线量级,Phase 1 收手;或提出 config 调整走回环"
    why_human: "收手标准(D-05)明文是用户认可;RMS 量级是否合理依赖用户对镜头的领域知识"
---

# Phase 1: 数据基座 + 最小完整标定流程 — 验证报告

**Phase Goal:** 用户在真实数据(input_data/ 640 train + 65 test)上一条命令跑通"入库体检 → 找角点 → 出内参",得到一个基线标定结果与角点叠加可视化,废帧剔除留痕
**Verified:** 2026-10-16T09:07:00Z
**Status:** human_needed(全部 5 条成功标准自动化验证通过;D-05 人工核查待用户答复)
**Re-verification:** No — initial verification

## 核心验证方式说明

验证时 `output/` 在主检出上**不存在**(执行期产物留在已清理的 worktree 中,gitignore 不进 git)。因此本验证不采信任何 SUMMARY 中的产物陈述,而是在当前 main 代码上**重新完整跑通真实数据全链**(ingest 705 张 → corners 705 张×双检测器 → calibrate 2×2,退出码 0,总耗时约 18 分钟),对新鲜产物逐条断言。所有下述证据均来自本次独立重跑。

## Goal Achievement

### Observable Truths(ROADMAP 成功标准逐条)

| #  | Truth                                                                                                        | Status     | Evidence(本次独立重跑实测)                                                                                                                                                     |
| -- | ------------------------------------------------------------------------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | `calib ingest` 产出 manifest.json:每张图带 pose_id/repeat_idx/split 与预检结论(指标比例化),废帧剔除留痕可查   | ✓ VERIFIED | 新产物 `output/ingest/run_20261016_085253/`:705 条(640+65)、每条含 image_id/split/pose_id/repeat_idx/n_repeats/precheck(blur_raw/blur_norm/expo_hi/expo_lo/visibility/suspect_sources/verdict);存疑 89 张 == suspect_originals/ 89 张副本精确相等;跨轮 diff 表在盘;blur_norm = blur_raw/实测方格边长²(precheck.py:169,比例化);repeats_distribution {'8': 80} 全部来自扫描 |
| 2  | `calib corners` 产出 corners_train.npz/corners_test.npz:亚像素角点,窗口从实测方格像素边长推导;连拍排序经规范化断言 | ✓ VERIFIED | 4 份 npz 精确命名契约;train 640/test 65 行对齐 manifest;全部 1409 个检出帧满足 2*win_size+1 < sq_px(自动化断言通过);未检出帧 NaN 占行不消失;window.py 推导公式 max(3, floor(sq_px/2)−1) 无写死像素;canonicalize 对 classic/sb 两路线统一(detect.py:78,104);连拍断言 80/75 组,6 组失败逐帧点名进 detection_summary.json(spread 3.7~6.2px,远低于翻转污染特征);合成测试锁定四翻转变体归一 + 翻转帧必被断言抓住 |
| 3  | 最小标定命令在真实 train 数据上产出基线内参与重投影 RMS,落盘 output/calibration/                               | ✓ VERIFIED | 新产物 `output/calibration/run_20261016_090004/`:4 组合全收敛 — classic_pinhole 0.1401px / classic_fisheye 0.1332px / sb_pinhole 1.2421px / sb_fisheye 0.0827px;K_fx≈1855~1872;4 份 residuals.npz shape (N,88,2) 可 allow_pickle=False 读取;comparison.json 恰 4 条;**血缘闭合实测:4 份 calibration.json 记录的 corners_sha256 与实际 npz 文件逐字节哈希相符** |
| 4  | 重复张数不齐或单张姿态不导致崩溃(数据驱动,无硬编码张数)                                                       | ✓ VERIFIED | dataset.py:103 n_repeats=len(jpgs) 纯数据驱动;单张姿态 n_repeats=1 合法存续;真实扫描 145 个姿态组(80×8 + 65 单张)全部处理;单元测试锁定 3张+1张不齐/单张/空目录/坏文件名路径(83 项非集成测试全过);源码无硬编码 640/705/8(grep 证实,仅注释提及) |
| 5  | 角点叠加可视化图(sampling + 最差若干视图)可人工核查角点是否偏移                                                | ✓ VERIFIED(产物就位;目检待人工) | marked_copies/classic/ 与 /sb/ 各 705 张(实测计数);overview_mosaic.png 2560×4326(164 格:89 存疑 + 每姿态 1 张 + 模糊度最差 20);suspect_mosaic.png 2560×2472(89 格);mosaic_index.json 逐格索引;最差视图抽样含 blur_norm 0.005~0.007 的 test 连拍。像素级正确性由合成真值测试(<0.5px)与重投影 RMS 亚像素量级间接保证;**人眼核查为 D-05 人工项,见 human_verification** |

**Score:** 5/5 truths verified

### Required Artifacts(三 level:存在/有实质/被接线)

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `src/calib/io/dataset.py` (178行) | 姿态数据模型与韧性扫描 | ✓ VERIFIED | ImageRecord/scan_dataset_dirs/imread_checked/verify_consistent_resolution;被 main.py:99 调用 |
| `src/calib/io/config.py` (369行) | 配置解析校验(safe_load) | ✓ VERIFIED | BoardSpec/CalibrationConfig 数据类;中文报错;test_io_config 锁定 safe_load |
| `src/calib/io/manifest.py` (302行) | manifest 写盘 + latest_run_dir + 跨轮 diff | ✓ VERIFIED | 嵌 make_lineage(实测 lineage 键齐);被 corners/calibrate 消费 |
| `src/calib/io/precheck.py` (292行) | 体检三指标 + 双来源判定 | ✓ VERIFIED | 真实 Laplacian/裁剪占比/双检测器探针;judge_suspects 稳健 z 分数(MAD);无删除路径 |
| `src/calib/corners/window.py` (113行) | 方格边长估计 + 窗口推导断言 | ✓ VERIFIED | 推导公式 + 2w+1<sq_px 逐帧断言(中文消息);无模块级魔法数字 |
| `src/calib/corners/canonical.py` (81行) | 排序规范化 + 连拍断言 | ✓ VERIFIED | 四翻转变体归一(测试锁定);burst_consistency 返回 ok/spread 键 |
| `src/calib/corners/detect.py` (108行) | 双检测器统一封装 | ✓ VERIFIED | cv2.cornerSubPix( 全文件恰 1 次(源码计数断言锁定 SB 隔离);两路线都过 canonicalize |
| `src/calib/corners/artifacts.py` (84行) | npz 落盘/读取 | ✓ VERIFIED | load_corners 强制 allow_pickle=False;真实 npz 往返验证 |
| `src/calib/solve/baseline.py` (340行) | 2×2 基线 + 剔除过滤 + 落盘 | ✓ VERIFIED | square_size 仅 4 处引用全在本文件(corners/precheck 0 处,IO-03 解耦);fisheye try/except 驯服;exclude_images 过滤(测试锁定) |
| `src/calib/visualization/overlay.py` (130行) | 标准叠加 + 全量副本 | ✓ VERIFIED | cv2.drawChessboardCorners;读 load_corners;真实 705×2 副本产出 |
| `src/calib/visualization/mosaic.py` (283行) | 拼图 + 逐格索引 | ✓ VERIFIED (⚠ WR-01) | 两拼图 + mosaic_index.json 真实产出;但 suspect_mosaic 找副本的路径断链(见 Anti-Patterns) |
| `src/calib/visualization/suspects.py` (35行) | 存疑原图副本 | ✓ VERIFIED | shutil.copy2;真实 89 张副本与存疑数精确相等 |
| `src/calib/cli/main.py` (544行) | typer 九子命令 | ✓ VERIFIED | 九命令全注册;ingest/corners/calibrate 完整实现;帮助文字通俗中文 |
| `config/calibration.yaml` | 参数唯一来源 + exclude_images | ✓ VERIFIED | exclude_images: [] 在盘;比例化阈值键(blur_z_threshold/quantile)在盘 |
| `SKELETON.md` / `docs/pipeline.md` / `REQUIREMENTS.md` | 骨架契约/措辞同步/需求状态 | ✓ VERIFIED | SKELETON 含"架构决策"与"刻意不做的事";pipeline.md 无"自动剔除"字样;REQUIREMENTS 四行 Shipped (Phase 1) |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | --- | --- | ------ | ------- |
| cli/main.py | io/dataset.py | scan_dataset_dirs 调用(main.py:99) | ✓ WIRED | 真实 ingest 产出 705 条即证明 |
| io/manifest.py | util/lineage.py | make_lineage(manifest.py:169) | ✓ WIRED | manifest lineage 键实测在盘 |
| cli/main.py corners | ingest manifest | latest_run_dir(main.py:250) | ✓ WIRED | 本次 corners 消费的正是本轮 ingest 产物 |
| corners/detect.py | corners/window.py | derive_window_frame/per_corner(detect.py:65-68,101) | ✓ WIRED | 真实数据窗口不等式对 1409 帧全成立 |
| cli/main.py corners | util/lineage.py | make_lineage(upstream=manifest) → save_corners meta | ✓ WIRED | meta.json 含 manifest 上游哈希;冒烟测试锁定闭合 |
| cli/main.py calibrate | corners/artifacts.py | load_corners(main.py:446) | ✓ WIRED | 标定实际消费 npz 并哈希闭合 |
| solve/baseline.py | io/config.py | cfg.exclude_images(baseline.py:197) | ✓ WIRED | 剔除过滤测试锁定 |
| test_smoke_realdata.py | cli 三命令 | subprocess 接力(test:87-89) | ✓ WIRED | 3 integration 测试实跑通过(46s) |
| io/precheck.py | 检测器原始调用 | findChessboardCorners/SB(precheck.py:120-121) | ✓ WIRED | 探针成功率与角点阶段逐帧一致 |
| cli/main.py ingest | io/precheck.py | compute_metrics→judge_suspects(main.py:110-116) | ✓ WIRED | 705 条全带 precheck.verdict |
| cli/main.py ingest | visualization/suspects.py | write_suspect_originals(main.py:129) | ✓ WIRED | 89 副本 == 89 存疑 |
| cli/main.py corners | overlay/mosaic | render_all + write_mosaics(main.py:363-366) | ✓ WIRED | 副本与拼图实测产出 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| manifest.json images[].precheck | blur_norm/expo/verdict | 逐张 imread → compute_metrics → judge_suspects | 是(89/705 存疑,分布与 SUMMARY 一致,确定性可复现) | ✓ FLOWING |
| corners npz | uv/sq_px/win_size/detected | 真实照片 × 双检测器 + canonicalize | 是(classic 100%/sb 93.8%,方格边长实测 ~45px) | ✓ FLOWING |
| calibration.json | K/D/rms/per_view_rms | corners npz × 640 视图 → calibrateCameraExtended / fisheye.calibrate | 是(0.083~1.24px,无静态返回) | ✓ FLOWING |
| marked_copies/mosaic | 像素 | input_data 原图 + npz 角点 | 是(705×2 副本;1.2GB 量级) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| 全链真实数据一条命令接力 | `calib ingest && calib corners && calib calibrate`(后台完整跑) | 退出码 0,三阶段产物齐全 | ✓ PASS |
| manifest 全字段断言 | python 断言脚本(705 条 × 12 键) | ALL PASS | ✓ PASS |
| 窗口不等式对全部检出帧 | `2*win+1 < sq_px` numpy 断言 | 1409/1409 帧 PASS | ✓ PASS |
| 标定 schema + 血缘哈希闭合 | corners_sha256 vs 实际文件 sha256 | 4/4 组合全等 | ✓ PASS |
| 单元测试套件 | `uv run pytest -m "not integration"` | 83 passed | ✓ PASS |
| 真实数据集成冒烟(独立进程三命令接力+血缘断言) | `uv run pytest -m integration` | 3 passed (46s) | ✓ PASS |
| lint | `uv run ruff check src tests` | All checks passed | ✓ PASS |
| corners 在"只有 train 无 test"的合法输入上 | 临时树(仅 train)ingest→corners | corners 崩溃(IndexError, main.py:336)— **CR-01 复现** | ✗ FAIL(见 Anti-Patterns) |

### Probe Execution

无 `scripts/*/tests/probe-*.sh` 约定探针,计划也未声明独立探针脚本。探针角色由 `tests/test_smoke_realdata.py`(3 个 integration 用例:子进程三命令接力 + 血缘两环指纹闭合断言)承担,本次实跑 3/3 通过。

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| IO-01 | 01-01, 01-04 | 姿态 ID 数据模型 + manifest 记录姿态/重复/分布 | ✓ SATISFIED | 705 条记录、145 姿态组、{'8': 80} 分布、只读扫描 |
| IO-02 | 01-04, 01-05 | 入库预检三指标比例化 + 废帧留痕 | ✓ SATISFIED | precheck 全字段、blur_norm 按方格边长²归一、89 存疑只标记 + 副本 + 跨轮 diff |
| IO-03 | 01-01 | 配置解析校验,square_size 与内参解耦 | ✓ SATISFIED | safe_load + 中文报错(测试锁定);square_size 0 处泄漏进 corners/precheck |
| COR-01 | 01-02, 01-03, 01-05 | 角点检测亚像素精化,候选可配置,窗口比例化 | ✓ SATISFIED | 双检测器(config 选 flags)、窗口从实测边长推导(1409 帧断言全过)、canonicalize + 连拍断言 |

**ORPHANED 检查:** REQUIREMENTS.md 映射到 Phase 1 的恰为 IO-01/IO-02/IO-03/COR-01 四项,与五个 PLAN frontmatter 的 requirements 并集完全一致,无遗漏、无多余。四项均已标记 Shipped (Phase 1),与实测证据相符。

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| src/calib/cli/main.py | 333-336 | **CR-01(代码审查确认,本次实测复现):空划分(仅 train 无 test)时 `np.array([])` 推断为 float64 做花式索引 → IndexError,corners 命令崩溃** | ⚠️ Warning | 不触及本阶段真实数据(640+65 双划分齐全),也不属于成功标准 4 的字面范围(张数不齐/单张姿态——两者均实测不崩);但违背 dataset.py 明文承诺的合法输入("只有训练集是合法输入"),应尽快修(`dtype=int` 一行 + 补测试) |
| src/calib/visualization/mosaic.py | 237 | **WR-01:存疑拼图在 corners run 目录找 suspect_originals/,实际副本在 ingest run 目录 → 真实流水线永远走"退回原图路径"分支(本次运行即如此)。画面内容等价(均为未标记原图),拼图正常产出,但 mosaic_index 的指引文案落空** | ⚠️ Warning | 跨阶段路径断链;测试夹具按错误布局构造掩盖了它 |
| src/calib/io/config.py | 169-175 | **WR-02:yaml 语法错误抛 ScannerError(非 ValueError),击穿中文报错防线,--help 也会裸栈崩溃** | ⚠️ Warning | 正常配置下不触发 |
| src/calib/solve/baseline.py | 94-95 | **WR-03:calibrate_pinhole 不捕获 cv2.error,与鱼眼"失败记录不中断"契约不对称;病态输入可致四组合全断** | ⚠️ Warning | 本次真实数据 4 组合全收敛,未触发 |
| src/calib/io/precheck.py | 80-86 | **WR-04:board_roi 回退硬编码检测器选项(违反 D-12);classic 未检出而 SB 检出的照片模糊度通道失明(本数据 classic 全检出,未触发)** | ⚠️ Warning | 当前真实数据下无实际影响 |
| src/calib/cli/main.py | 406,505-520 | noise/fisher/evaluate/analyze/report 五命令为计划内占位(D-11 骨架契约,打印中文说明正常退出) | ℹ️ Info | 属 Phase 3-6 范围,非本阶段目标;非违规桩 |
| 全仓 | - | IN-01~IN-07(格式不统一、blur_z_threshold 名不副实、CWD 相对路径防线测试、裸 assert 防线、SB 窗口记录语义、计数逻辑双实现、image_id 下游不复验) | ℹ️ Info | 见 01-REVIEW.md,均不影响本阶段目标 |

**债务标记检查:** 全部源码/测试/配置 grep 无 TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER —— 无未审计的债务标记,BLOCKER 级反模式为零。

### Human Verification Required

见 frontmatter `human_verification` 四项(即 01-05 PLAN 的 D-05 收手标准清单,路径已按本次重跑的最新产物更新为 `output/corners/run_20261016_085253/` 与 `output/calibration/run_20261016_090004/`)。01-05 SUMMARY 明确记录人工核查"待用户答复"——D-05 收手标准是用户认可,程序无法代替。

### Gaps Summary

**无阻断性缺口。** 五条成功标准全部在当前 main 代码 + 真实数据上独立复现通过:

1. 执行期产物未随 worktree 合并保留(output/ 不进 git)——本次验证通过完整重跑消除了对 SUMMARY 陈述的依赖,且新产物数值与 SUMMARY 记录逐位一致(89 存疑、classic 100%/SB 93.8%、RMS 0.1401/0.1332/1.2421/0.0827),证明流水线确定性可复现,这本身是比"产物在盘"更强的证据。
2. CR-01(corners 空划分崩溃)为真实缺陷但**不阻断本阶段目标**:本阶段真实数据双划分齐全,成功标准 4 的字面范围(张数不齐/单张姿态)实测不崩。建议随后以小修复闭环(一行 `dtype=int` + 回归测试),不值得为此打回阶段。
3. WR-01~WR-04 均为审查确认的 Warning 级,当前真实数据下不触发或影响等价,记录在案供后续处理。
4. 状态为 human_needed 的唯一原因:D-05 收手标准 = 用户人工核查角点叠加图/存疑清单/2×2 对比后认可。自动化证据(合成真值 <0.5px、重投影 RMS 亚像素、血缘闭合)已就位,等待用户按清单四步答复。

---

_Verified: 2026-10-16T09:07:00Z_
_Verifier: Claude (gsd-verifier)_
