---
phase: 2
slug: m1-m4
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-10-17
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> 验证维度来源:02-RESEARCH.md §Validation Architecture(6 维度)。

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x(uv 管理,阶段一已就位;106 条现有测试为回归基线) |
| **Config file** | pyproject.toml [tool.pytest.ini_options](markers: integration,阶段一已建) |
| **Quick run command** | `uv run pytest tests/ -x -q -m "not integration"` |
| **Full suite command** | `uv run pytest tests/ -q` |
| **Estimated runtime** | 快速 ~30–60s;全套 ~1–2min(新增模型/求解测试后增加约 1–2min) |

**新增测试基建(计划落实时补):** dev 依赖 `numdifftools`(数值雅可比基准,仅测试路径,研究已审计);模型测试参数采样夹具(典型/边界/种子随机,见容差矩阵)。

---

## Sampling Rate

- After every task commit: Run `uv run pytest tests/ -x -q -m "not integration"`
- After every plan wave: Run `uv run pytest tests/ -q`(含真实数据集成冒烟)
- Before `/gsd-verify-work`: Full suite must be green
- Max feedback latency: 60 秒(快速命令;单次 OpenCV 标定实测 ~100s,真实数据试车类任务以"单组合冒烟"替代全矩阵)

---

## 验证维度(源自研究,阶段成功标准对齐)

| # | 维度 | 对应成功标准 | 自动化形态 |
|---|------|-------------|-----------|
| 1 | 投影一致性(parity) | 标准 1 | M1/M2/M3 vs `cv2.projectPoints`、M4 vs `cv2.fisheye.projectPoints`,float64 atol 1e-9;OpenCV float32 路径 1e-3;典型/边界/随机种子 ≥3 组参数 |
| 2 | 解析雅可比正确性 | 标准 1 | 每模型每参数列,对 numdifftools 中心差分,列相对误差阈值(研究建议 1e-6,含典型+边界参数) |
| 3 | 双后端同构与可互换 | 标准 2 | 同一 ObservationSet × 任一注册模型,两后端返回同一 CalibrationResult 结构;ScipySolver 残差源码断言(仅由 models.project 构成) |
| 4 | 真实数据收敛不劣化 | 标准 3 | 全矩阵试车落盘(scipy 精化 RMS ≤ 纯 OpenCV RMS + 容差);"实用收敛"纯函数判定) |
| 5 | 血缘与独立重跑 | 标准 4 | 产物 lineage 含配置哈希+上游哈希+时间戳;新进程只凭产物重跑指定组合 |
| 6 | 注册机制零侵入 | 标准 5 | 测试内定义玩具模型,一处注册,两求解器均可解 |

**容差矩阵(研究 §Common Pitfalls + §Code Examples 实测):**
- 自实现投影 vs OpenCV(float64 输入):atol 1e-9 px(实测达 4.5e-13)
- OpenCV float32 绑定路径(calibrateCamera 输入输出):1e-3 px 量级
- 解析雅可比 vs 中心差分:列相对误差 1e-6
- 跨后端残差比较:一律用 models.project 自算(禁用 cv2.projectPoints 的 jacobian 输出,float32 下已实测不可靠)

---

## Per-Task Verification Map

*(计划生成后按 PLAN.md 任务填充——阶段一家法,见 01-VALIDATION.md 样式)*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | — | — | — | — | — | — | — | — | — |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `uv run add --group dev numdifftools` — 数值雅可比测试基准(唯一新依赖,研究已审计 [OK])
- [ ] 新测试文件按计划落地(parity / 雅可比 / 求解器 / 注册机制)
- [ ] 现有 106 条测试保持全绿(回归基线,任何阶段二改动不得破坏阶段一行为)

*测试框架与夹具基建阶段一已就位,无框架安装需求。*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 全矩阵试车结果量级认可(收敛表 + 精化增益方向符合直觉) | SOL-01 | 数值达标可自动判,但"结果合理"需标定工程师目检 | 看 output/calibration/ 试车产物汇总表:RMS 量级、精化后误差变化方向、失败组合的中文 failure_note |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
