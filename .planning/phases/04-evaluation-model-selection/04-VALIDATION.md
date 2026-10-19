---
phase: 4
slug: evaluation-model-selection
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-10-19
---

# Phase 4 — Validation Strategy

> 执行期的反馈采样契约(每个任务提交后跑什么、多久能知道对不对)。逐任务对照表在计划定稿后回填。

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest(既有,Phase 1 起落地) |
| **Config file** | pyproject.toml(test 配置段)+ tests/conftest.py |
| **Quick run command** | `uv run pytest tests -q -x --ignore=tests/test_evaluation_realdata.py` |
| **Full suite command** | `uv run pytest tests -q` |
| **Estimated runtime** | 快速环 ~10–30 秒(纯单测);全量含真实数据集成,分钟级(集成测试复用血缘指纹对得上的历史产物,见 Phase 3 体例) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests -q -x --ignore=tests/test_evaluation_realdata.py`
- **After every plan wave:** Run `uv run pytest tests -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 秒(快速环)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| (计划定稿后回填) | 01 | 1 | EVA-01 | — | N/A | unit | `uv run pytest tests/test_evaluation_splits.py -q` | ❌ W0 | ⬜ pending |
| (计划定稿后回填) | 02 | 2 | EVA-02 | — | N/A | unit | `uv run pytest tests/test_evaluation_cv.py -q` | ❌ W0 | ⬜ pending |
| (计划定稿后回填) | 03 | 2 | EVA-03 | — | N/A | unit | `uv run pytest tests/test_evaluation_slices.py -q` | ❌ W0 | ⬜ pending |
| (计划定稿后回填) | 04 | 3 | EVA-04/05 | — | N/A | unit | `uv run pytest tests/test_evaluation_resample.py -q` | ❌ W0 | ⬜ pending |
| (计划定稿后回填) | 05 | 3 | SOL-02 | — | N/A | unit | `uv run pytest tests/test_optimizer_comparison.py -q` | ❌ W0 | ⬜ pending |
| (计划定稿后回填) | 06 | 4 | EVA-02 | — | N/A | integration | `uv run pytest tests/test_evaluation_realdata.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Phase 4 特有的校验红线(来自成功标准与 CONTEXT 决策):**

1. **防泄漏断言是硬门禁**——任何切分函数的测试必须断言:任意切分下训练/验证姿态交集为空(EVA-01;D-01/D-03,seed 42 固定可复现)
2. **同输入同输出**——四类实验函数的确定性断言(同参数重跑逐位一致,随机源全部走 seed)
3. **"实测 < CRLB"硬告警测试**——构造经验 std 低于理论下界的合成用例,断言告警触发不静默(成功标准 3;D-05/D-06,0.42×σ_CRLB 判定线)
4. **多起点一致性**——扰动初值 N 次收敛到同一解的断言(m4 重点;D-15)
5. **真实数据集成测试**——`calib evaluate` 全链对账(CV 并排表/判行表/test 评分落盘,血缘闭环)

---

## Wave 0 Requirements

- [ ] `tests/test_evaluation_splits.py` — 切分防泄漏 + 固定种子复现(EVA-01)
- [ ] `tests/test_evaluation_experiments.py` — 四类实验函数确定性(共用基座)
- [ ] `tests/test_evaluation_realdata.py` — 真实数据集成(随对应计划补)

*注:以上为按成功标准预估的桩位;实际文件名以 PLAN.md 定稿为准,回填时更正。joblib 需在本阶段首个计划内 `uv add joblib`(研究结论:PyPI 1.6.0)。*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 判行表与选型结论的人工核查(四要素:看哪个文件/作用/看哪里/好坏判据) | EVA-02 | 选型结论是"可论证性"交付,需人眼核对判行表与证据链指向 | 打开 output/evaluate/ 最新轮的 selection.json 与判行表,核对三判据 PASS/FAIL 与 held-out 并排表一致;CV 曲线误差棒不重叠/重叠的视觉判断 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending(计划定稿回填对照表后复核)
