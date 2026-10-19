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
| **Quick run command** | `uv run pytest tests -q -x --ignore=tests/test_optimizer_realdata.py --ignore=tests/test_evaluation_realdata.py` |
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
| 01-T1 依赖+EvaluateConfig+parallel | 01 | 1 | EVA-01 | T-04-01-01/04 | config 区间校验,中文报错 | unit | `uv run pytest tests/test_io_config.py -q` | ✅ 既有文件扩展 | ⬜ pending |
| 01-T2 splitting 切分纯函数 | 01 | 1 | EVA-01 | T-04-01-02/03 | 零重叠断言 + 确定性 | unit | `uv run pytest tests/test_evaluation_split.py -q` | ❌ W0 | ⬜ pending |
| 02-T1 optimizer.json 契约 | 02 | 2 | SOL-02 | T-04-02-02 | 契约键校验/失败契约 | unit | `uv run pytest tests/test_optimizer_comparison.py -q` | ❌ W0 | ⬜ pending |
| 02-T2 16 组合网格+推荐配置 | 02 | 2 | SOL-02 | T-04-02-01/03 | 初值行序对齐;失败占行 | unit | `uv run pytest tests/test_optimizer_comparison.py -q` | ❌ W0 | ⬜ pending |
| 02-T3 多起点双容差 | 02 | 2 | SOL-02 | T-04-02-03 | 坏解记录不静默 | unit | `uv run pytest tests/test_optimizer_comparison.py -q` | ❌ W0 | ⬜ pending |
| 02-T4 真实数据优化器试车 | 02 | 2 | SOL-02 | T-04-02-04/05 | 血缘登记;只写 output/evaluate/ | integration | `uv run pytest tests/test_optimizer_realdata.py -m integration -q` | ❌ W0 | ⬜ pending |
| 03-T1 配对三分支+BIC+MAD | 03 | 2 | EVA-02/05 | T-04-03-01 | 输入 isfinite 检查 | unit | `uv run pytest tests/test_selection_stats.py -q` | ❌ W0 | ⬜ pending |
| 03-T2 卡方告警线+判行表 | 03 | 2 | EVA-02 | T-04-03-02/03 | 阈值随结论落盘 | unit | `uv run pytest tests/test_selection_stats.py -q` | ❌ W0 | ⬜ pending |
| 04-T1 CV 实验 | 04 | 3 | EVA-01 | T-04-04-02 | 验证侧外参不训练 | unit | `uv run pytest tests/test_evaluation_experiments.py -k cv -q` | ❌ W0 | ⬜ pending |
| 04-T2 LOPO+分组+CRLB 告警 | 04 | 3 | EVA-03/05 | T-04-04-01 | σ 口径标识;硬告警不静默 | unit | `uv run pytest tests/test_evaluation_experiments.py -k "lopo or slice" -q` | ❌ W0 | ⬜ pending |
| 04-T3 bootstrap+来源分离 | 04 | 3 | EVA-04 | T-04-04-04/05 | 同种子逐位一致 | unit | `uv run pytest tests/test_evaluation_experiments.py -q` | ❌ W0 | ⬜ pending |
| 05-T1 全量重训+test 评分 | 05 | 4 | EVA-02/SOL-02 | T-04-05-01 | test 集行序契约装载 | unit | `uv run pytest tests/test_evaluation_experiments.py -k delivery -q` | ❌ W0 | ⬜ pending |
| 05-T2 evaluate.json 契约 | 05 | 4 | EVA-02 | T-04-05-02/05 | save/load 往返;npz allow_pickle=False | unit | `uv run pytest tests/test_evaluation_smoke.py -q` | ❌ W0 | ⬜ pending |
| 05-T3 判行表三图+全链冒烟 | 05 | 4 | EVA-02 | T-04-05-03 | 图失败不拦主产物 | unit(冒烟) | `uv run pytest tests/test_evaluation_smoke.py -q` | ❌ W0 | ⬜ pending |
| 06-T1 `calib evaluate` 填充 | 06 | 5 | EVA-02/SOL-02 | T-04-06-01/04 | 失败契约落盘+退出码 1 | unit | `uv run pytest -m "not integration" -q` | ✅ 既有文件扩展 | ⬜ pending |
| 06-T2 真实数据全链对账 | 06 | 5 | EVA-02/SOL-02 | T-04-06-02 | 血缘一致才复用历史轮 | integration | `uv run pytest tests/test_evaluation_realdata.py -m integration -q` | ❌ W0 | ⬜ pending |
| 06-T3 人工核查判行表 | 06 | 5 | EVA-02 | T-04-06-03 | 四要素指引 | manual | —(checkpoint:human-verify) | — | ⬜ pending |
| 06-T4 需求状态+契约收口 | 06 | 5 | 全部 | — | N/A | docs+回归 | `uv run pytest -m "not integration" -q` | ✅ 既有文件 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Phase 4 特有的校验红线(来自成功标准与 CONTEXT 决策):**

1. **防泄漏断言是硬门禁**——任何切分函数的测试必须断言:任意切分下训练/验证姿态交集为空(EVA-01;D-01/D-03,seed 42 固定可复现)
2. **同输入同输出**——四类实验函数的确定性断言(同参数重跑逐位一致,随机源全部走 seed)
3. **"实测 < CRLB"硬告警测试**——构造经验 std 低于理论下界的合成用例,断言告警触发不静默(成功标准 3;D-05/D-06,0.42×σ_CRLB 判定线)
4. **多起点一致性**——扰动初值 N 次收敛到同一解的断言(m4 重点;D-15)
5. **真实数据集成测试**——`calib evaluate` 全链对账(CV 并排表/判行表/test 评分落盘,血缘闭环)

---

## Wave 0 Requirements

- [ ] `uv add joblib` — 04-01 Task 1(阶段唯一新依赖,合法性审计 [OK])
- [ ] `tests/test_evaluation_split.py` — 切分防泄漏 + 固定种子复现(04-01 Task 2,EVA-01 硬门禁,先于全部实验函数)
- [ ] `tests/test_evaluation_experiments.py` — 四类实验函数确定性(04-04 共用基座;05-T1 delivery 用例也落此文件)
- [ ] `tests/test_optimizer_comparison.py` / `tests/test_selection_stats.py` / `tests/test_evaluation_smoke.py` — 分别随 04-02 / 04-03 / 04-05 落地
- [ ] `tests/test_optimizer_realdata.py` / `tests/test_evaluation_realdata.py` — 真实数据集成(04-02 / 04-06,integration marker 默认跳过)

*注:实际文件名已按 6 份 PLAN.md 定稿回填(与 04-RESEARCH.md Wave 0 Gaps 一致;原预估桩位 test_evaluation_splits/cv/slices/resample 合并为上述文件)。快速环命令相应更正为 `--ignore=tests/test_optimizer_realdata.py --ignore=tests/test_evaluation_realdata.py`。*

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
