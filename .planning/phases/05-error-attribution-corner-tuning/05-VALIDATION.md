---
phase: 5
slug: error-attribution-corner-tuning
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-10-21
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

- **After every task commit:** Run `uv run pytest tests -q -x --ignore=tests/test_evaluation_realdata.py`
- **After every plan wave:** Run `uv run pytest tests -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 秒(快速环)

---

## Per-Task Verification Map

<!-- 计划定稿后回填:每行 = 任务 ID | 计划 | 波次 | 需求 | 威胁编号 | 安全行为 | 测试类型 | 自动命令 | 文件状态 | 进度 -->

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| (待计划定稿后回填) | | | ANA-01~04, COR-02/03 | | | | | | ⬜ |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_analysis_*.py` — 误差预算/方向分解/敏感度各段单测存根(按计划实际文件名回填)
- [ ] `tests/test_corner_degradation.py` — 合成退化实验单测存根(按计划实际文件名回填)
- [ ] `tests/conftest.py` — 既有共享夹具扩展(小棋盘合成观测集)

*按计划定稿回填;若无新增:"Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 空间图表带覆盖足迹、人话结论可读性 | ANA-02 | 图表可读性与人话措辞需人眼判断 | 打开 output/analysis/ 最新轮 figures/,核对每张空间图有覆盖足迹叠加、结论行无行话 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
