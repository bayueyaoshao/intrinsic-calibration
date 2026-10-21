---
phase: 5
slug: error-attribution-corner-tuning
status: planned
nyquist_compliant: true
wave_0_complete: true
created: 2026-10-21
plans: 6
---

# Phase 5 — Validation Strategy

> 执行期的反馈采样契约(每个任务提交后跑什么、多久能知道对不对)。逐任务对照表在计划定稿后回填。

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest(既有,Phase 1 起落地) |
| **Config file** | pyproject.toml(test 配置段)+ tests/conftest.py |
| **Quick run command** | `uv run pytest tests -q -x --ignore=tests/test_evaluation_realdata.py --ignore=tests/test_corner_realdata.py` |
| **Full suite command** | `uv run pytest tests -q` |
| **Estimated runtime** | 快速环 ~10–30 秒(纯单测);全量含真实数据集成,分钟级(集成测试复用血缘指纹对得上的历史产物,见 Phase 3/4 体例) |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/python -m pytest tests -q -m "not integration"` + `.venv/bin/ruff check src tests`
- **After every plan wave:** Run `.venv/bin/python -m pytest tests -q -m "not integration"`(489 基线全绿零回退);每计划另跑 `.venv/bin/python -m pytest tests/test_analysis_realdata.py -q -m integration`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 秒(快速环)

---

## Per-Task Verification Map

<!-- 计划定稿后回填:每行 = 任务 ID | 计划 | 波次 | 需求 | 威胁编号 | 安全行为 | 测试类型 | 自动命令 | 文件状态 | 进度 -->

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-T1 | 05-01 | 1 | ANA-01 | T-05-03 | yaml safe_load + 类型化校验 | unit | `.venv/bin/python -m pytest tests/test_io_config.py -x -q` | ✅ 既有 | ⬜ |
| 05-01-T2 | 05-01 | 1 | ANA-01 | T-05-01/04 | allow_pickle=False + 失败契约 | unit | `.venv/bin/python -m pytest tests/test_analysis_artifacts.py -x -q` | ❌ 随任务建 | ⬜ |
| 05-01-T3 | 05-01 | 1 | ANA-01 | T-05-01/02 | 复核一致位 + 硬报错 | unit | `.venv/bin/python -m pytest tests/test_analysis_budget.py -x -q` | ❌ 随任务建 | ⬜ |
| 05-02-T1 | 05-02 | 2 | ANA-01 | T-05-06/07 | 血缘登记 + 失败契约退出码 1 | unit(smoke) | `.venv/bin/python -m pytest tests/test_analysis_cli.py -x -q` | ❌ 随任务建 | ⬜ |
| 05-02-T2 | 05-02 | 2 | ANA-01 | — | 图失败不拦主产物 | unit | `.venv/bin/python -m pytest tests/test_visualization.py -x -q` | ✅ 扩展 | ⬜ |
| 05-02-T3 | 05-02 | 2 | ANA-01 | T-05-06 | 真实数据对账(指纹门) | integration | `uv run calib analyze && .venv/bin/python -m pytest tests/test_analysis_realdata.py -q -m integration` | ❌ 随任务建 | ⬜ |
| 05-03-T1 | 05-03 | 3 | ANA-02 | T-05-09/11 | npz 防线 + 失败占行 | unit | `.venv/bin/python -m pytest tests/test_analysis_residual.py -x -q` | ❌ 随任务建 | ⬜ |
| 05-03-T2 | 05-03 | 3 | ANA-02 | — | 空间图足迹叠加 | unit | `.venv/bin/python -m pytest tests/test_visualization.py -x -q` | ✅ 扩展 | ⬜ |
| 05-03-T3 | 05-03 | 3 | ANA-02 | T-05-10 | 归一化基准披露对账 | unit+integration | `uv run calib analyze && .venv/bin/python -m pytest tests/test_analysis_realdata.py -q -m integration` | ✅ 扩展 | ⬜ |
| 05-04-T1 | 05-04 | 4 | ANA-03 | T-05-12/14 | 反解恒等式 + 病态占行 | unit | `.venv/bin/python -m pytest tests/test_analysis_test_xyz.py -x -q` | ❌ 随任务建 | ⬜ |
| 05-04-T2 | 05-04 | 4 | ANA-04 | T-05-13/15 | what-if 合成声明 + 建议不自动执行 | unit | `.venv/bin/python -m pytest tests/test_analysis_sensitivity.py -x -q` | ❌ 随任务建 | ⬜ |
| 05-04-T3 | 05-04 | 4 | ANA-03/04 | T-05-12 | 四段齐 + uniqueness 对账 | unit+integration | `uv run calib analyze && .venv/bin/python -m pytest tests/test_analysis_realdata.py -q -m integration` | ✅ 扩展 | ⬜ |
| 05-05-T1 | 05-05 | 5 | COR-02 | T-05-18/19 | 检测唯一入口 + 退化参数校验 | unit | `.venv/bin/python -m pytest tests/test_analysis_corner_opt.py -x -q` | ❌ 随任务建 | ⬜ |
| 05-05-T2 | 05-05 | 5 | COR-02 | T-05-16/17 | TaskFailed 信封 + 串行并行逐位一致 | unit | `.venv/bin/python -m pytest tests/test_analysis_corner_opt.py -x -q` | ✅ 扩展 | ⬜ |
| 05-05-T3 | 05-05 | 5 | COR-02 | — | 双口径曲线真实首跑 | unit+integration | `uv run calib analyze && .venv/bin/python -m pytest tests/test_analysis_realdata.py -q -m integration` | ✅ 扩展 | ⬜ |
| 05-06-T1 | 05-06 | 6 | COR-03 | T-05-20/21 | temporal 过滤 + clamp 防零除 | unit | `.venv/bin/python -m pytest tests/test_analysis_anisotropy.py -x -q` | ❌ 随任务建 | ⬜ |
| 05-06-T2 | 05-06 | 6 | COR-03 | — | 六段全链 + 单块兜底 | unit+integration | `uv run calib analyze && .venv/bin/python -m pytest tests/test_analysis_realdata.py -q -m integration` | ✅ 扩展 | ⬜ |
| 05-06-T3 | 05-06 | 6 | 全部 | T-05-22 | 契约/账本同步 + 人工核查 | automated grep + human-check | `grep -c "calib analyze" docs/pipeline.md && grep -c "Shipped (Phase 5)" .planning/REQUIREMENTS.md` + checkpoint | — | ⬜ |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_analysis_budget.py` / `test_analysis_artifacts.py`(05-01)、`test_analysis_cli.py` / `test_analysis_realdata.py`(05-02)、`test_analysis_residual.py`(05-03)、`test_analysis_test_xyz.py` / `test_analysis_sensitivity.py`(05-04)、`test_analysis_corner_opt.py`(05-05)、`test_analysis_anisotropy.py`(05-06)——全部随对应实现任务同计划创建(每任务 verify 含该文件 pytest 命令,无独立 Wave 0)
- [x] 合成棋盘 fixture:`tests/conftest.py` 的 `make_chessboard_image`(自带 degrade 参数)与 `make_true_views` 已覆盖 COR-02 单测与 ANA-02/03 恒等式测试,零新基建
- [x] 无需新装框架/依赖(pytest/ruff 已在 dev 组,阶段零新依赖)

*无独立 Wave 0:测试文件与实现同任务交付,采样节奏见 Sampling Rate 与 Per-Task Verification Map。*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 空间图表带覆盖足迹、人话结论可读性 | ANA-02 | 图表可读性与人话措辞需人眼判断 | 打开 output/analysis/ 最新轮 figures/,核对每张空间图有覆盖足迹叠加、结论行无行话 |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies(05-06-T3 为 checkpoint,另有 grep 自动位 + human-check)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify(全部任务均有 automated 位)
- [x] Wave 0 covers all MISSING references(无独立 Wave 0,测试随任务建)
- [x] No watch-mode flags
- [x] Feedback latency < 30s(单文件 pytest 快速环;集成与 corner_opt 首跑为例外,标 integration 不进快速环)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** planned(2026-10-21,随 6 份 PLAN 定稿回填)
