---
phase: 3
slug: fisher-crlb
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-10-18
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x(uv 管理,Phase 1 已建成) |
| **Config file** | pytest marker 配置(`integration` marker 已在 Phase 2 定义) |
| **Quick run command** | `uv run pytest -m "not integration" -q` |
| **Full suite command** | `uv run pytest -q tests/` |
| **Estimated runtime** | ~30–120 seconds |

---

## Sampling Rate

- **After every task commit:** Run 该任务 `<automated>` 命令(见下表,均含 `-m "not integration"` 快跑)
- **After every plan wave:** Run `uv run pytest -q tests/`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

任务为各 PLAN.md 内的位置序号(计划内 Task 1/2/3)。威胁登记见各 PLAN.md `<threat_model>` 块。

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-T1 | 01 | 1 | NOI-02 | T-01 配置白名单 | noise: 段参数白名单解析,未知键中文报错 | unit | `uv run pytest tests/test_io_config.py -q && uv run pytest -m "not integration" -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-01-T2 | 01 | 1 | NOI-02 | T-01 | Immerkaer 常数回收测试锁死(A2) | unit | `uv run pytest tests/test_noise_immerkaer.py -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-01-T3 | 01 | 1 | NOI-02, NOI-01 | T-01 注入网格资源上限 | σ 网格/档数/并发上限只从 config 读 | unit | `uv run pytest tests/test_noise_injection.py tests/test_noise_immerkaer.py -q && uv run pytest -m "not integration" -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-02-T1 | 02 | 1 | NOI-03 | T-02 D-10 第二份数学静态锁 | J 只经 make_jacobian_fn/pack_theta | unit | `uv run pytest tests/test_noise_fisher.py -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-02-T2 | 02 | 1 | NOI-03 | T-02 奇异防护 | 条件数 ≥1e12 中文报错不静默 pinv | unit | `uv run pytest tests/test_noise_fisher.py -q && uv run pytest -m "not integration" -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-03-T1 | 03 | 2 | NOI-01 | T-03 npz 禁 pickle | 时域统计逐姿态来源标注 | unit | `uv run pytest tests/test_noise_temporal.py -q && uv run pytest -m "not integration" -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-03-T2 | 03 | 2 | NOI-01 | T-03 产物目录越界 | noise.json 契约 + make_lineage | unit | `uv run pytest -m "not integration" -q && uv run python -c "from calib.cli.main import noise; print('noise 命令已填充')"` | ❌ W0→任务自建 | ⬜ pending |
| 03-03-T3 | 03 | 2 | NOI-01 | T-03 | 真实数据首跑(test 65 张单帧路径) | integration | `uv run pytest tests/test_noise_realdata.py -m integration -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-04-T1 | 04 | 2 | NOI-03 | T-04 异常不炸主流程 | cv2.error → status=failed + 中文 failure_note | unit | `uv run pytest tests/test_noise_crosscheck.py -q && uv run pytest -m "not integration" -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-05-T1 | 05 | 3 | NOI-03 | T-05 配置白名单 | FisherConfig + 80 姿态聚合观测集 | unit | `uv run pytest tests/test_noise_fisher.py tests/test_io_config.py -q && uv run pytest -m "not integration" -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-05-T2 | 05 | 3 | NOI-03 | T-05 npz 禁 pickle + 血缘 | run_fisher_matrix 16 份编排 + crlb.json | unit | `uv run pytest -m "not integration" -q && uv run python -c "from calib.cli.main import fisher; from calib.noise.fisher import run_fisher_matrix; print('fisher 编排已填充')"` | ❌ W0→任务自建 | ⬜ pending |
| 03-05-T3 | 05 | 3 | NOI-03 | T-05 | 真实数据试车 + 成功标准 3/4/5 对账 | integration | `uv run pytest tests/test_fisher_realdata.py -m integration -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-06-T1 | 06 | 4 | NOI-01/02/03 | T-06 血缘链闭合 | 五步接力结构键全 + 数值 null 契约 | integration | `uv run pytest tests/test_smoke_realdata.py -m integration -q` | ❌ W0→任务自建 | ⬜ pending |
| 03-06-T2 | 06 | 4 | NOI-01/02/03 | T-06 | run-all 直达 crlb.json + 文档/需求同步 | integration | `uv run pytest tests/test_cli_main.py -q && uv run pytest -m "not integration" -q && uv run calib run-all` | ❌ W0→任务自建 | ⬜ pending |
| 03-06-T3 | 06 | 4 | NOI-01/02/03 | — | 人工核查(见 Manual-Only) | manual | —(blocking checkpoint) | — | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

> 注:测试文件由各任务自建(计划 files_modified 已含对应 tests/ 文件),"File Exists" 列的 ❌ W0 表示随任务创建,非 Wave 0 预建——基础设施(pytest/conftest/integration marker)Phase 1/2 已就绪。

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* — pytest 框架、conftest fixtures、`integration` marker 均在 Phase 1/2 建成;本阶段新测试文件随所属任务自建。

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 成功标准 1–5 终审:噪声底报告区分 detector/位姿抖动、传递曲线可读、CRLB 谱/条件数退化方向标注、敏感性带呈现 | NOI-01/02/03 | 图表可读性与结论对账属人工判断 | 执行 03-06 Task 3(blocking checkpoint):审阅 output/figures 下图表含敏感性带与通俗中文标注,对账五条成功标准后人工确认 |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies(03-06-T3 为人工核查,已在 Manual-Only 登记且为 blocking checkpoint)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references(基础设施已就绪,无 MISSING)
- [x] No watch-mode flags
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-10-18(计划校验后回填)
