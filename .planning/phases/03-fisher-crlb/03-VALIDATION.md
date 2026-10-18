---
phase: 3
slug: fisher-crlb
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-10-18
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | none — 若仓库尚无 pytest 配置则 Wave 0 补齐 |
| **Quick run command** | `uv run pytest -x -q tests/ -k "noise or fisher or crlb"` |
| **Full suite command** | `uv run pytest -q tests/` |
| **Estimated runtime** | ~30–120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -x -q tests/ -k "noise or fisher or crlb"`
- **After every plan wave:** Run `uv run pytest -q tests/`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| (由 PLAN.md 生成后回填) | — | — | NOI-01 / NOI-02 / NOI-03 | — | N/A | unit + numeric-parity | `uv run pytest -q tests/` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

> 注:具体任务到测试的映射需在 PLAN.md 生成后回填;规划阶段的维度 8 检查以 RESEARCH.md「Validation Architecture」为准。

---

## Wave 0 Requirements

- [ ] `tests/` 下新增 noise/fisher 对应测试桩(若尚无)
- [ ] `tests/conftest.py` — 共享 fixtures(小图合成数据、固定随机种子)
- [ ] pytest 框架已装(uv dev 依赖)

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CRLB 敏感性带呈现是否可读(图/表) | NOI-03 | 图表可读性属人工判断 | 运行 `calib fisher` 后人工检查 output/figures 下图表标注是否含敏感性带、是否用通俗中文 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
