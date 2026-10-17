---
phase: 2
slug: m1-m4
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-10-17
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> 验证维度来源:02-RESEARCH.md §Validation Architecture(6 维度);任务行已按 02-01~02-06 计划填充。

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x(uv 管理,阶段一已就位;106 条现有测试为回归基线) |
| **Config file** | pyproject.toml [tool.pytest.ini_options](markers: integration,阶段一已建) |
| **Quick run command** | `uv run pytest tests/ -x -q -m "not integration"` |
| **Full suite command** | `uv run pytest tests/ -q` |
| **Estimated runtime** | 快速 ~30–90s(新增模型/求解单测后);全套含真实数据试车类 integration 约 5–10 分钟(全矩阵实跑 15–25 分钟单独后台执行,不进 pytest) |

**新增测试基建(随计划落地):** dev 依赖 `numdifftools`(02-01 Task 1,仅测试路径,研究已审计 [OK]);合成真值视角夹具上提 conftest(02-03 Task 3)。

---

## Sampling Rate

- After every task commit: Run `uv run pytest tests/ -x -q -m "not integration"`
- After every plan wave: Run `uv run pytest tests/ -q`(含真实数据集成)
- Before `/gsd-verify-work`: Full suite must be green
- Max feedback latency: 90 秒(快速命令;真实数据试车类任务以"单组合冒烟(m4 最快)+ 后台全矩阵"替代,见 02-05 Task 3)

---

## 验证维度(源自研究,阶段成功标准对齐)

| # | 维度 | 对应成功标准 | 自动化形态 |
|---|------|-------------|-----------|
| 1 | 投影一致性(parity) | 标准 1 | M1/M2/M3 vs `cv2.projectPoints`、M4 vs `cv2.fisheye.projectPoints`,float64 atol 1e-9;OpenCV float32 路径 1e-3;典型/边界/随机种子 ≥3 组参数(02-01 Task 3) |
| 2 | 解析雅可比正确性 | 标准 1 | 每模型每参数列,对 numdifftools 列相对误差 < 1e-6(含近轴点;M1–M3 另加 cv2 雅可比 float64 辅助交叉)(02-01 Task 3) |
| 3 | 双后端同构与可互换 | 标准 2 | 同一 ObservationSet × 任一注册模型,两后端 CalibrationResult 字段名单一致;失败契约逐键(02-03 Task 2/3、02-04 Task 2) |
| 4 | 真实数据收敛不劣化 | 标准 3 | 全矩阵试车落盘;m1/m2 各至少一条 scipy 路径(含 soft_l1)RMS ≤ 同模型 opencv + 1e-6(02-05 Task 3) |
| 5 | 血缘与独立重跑 | 标准 4 | lineage 三要素齐 + upstream 哈希重算对账;新进程重跑数值一致 < 1e-12(02-05 Task 3、02-06 Task 2) |
| 6 | 注册机制零侵入 | 标准 5 | 测试内玩具模型一处注册,两求解器均可解;求解器源码无具体模型 import(02-04 Task 3) |

**容差矩阵(研究 §Common Pitfalls + §Code Examples 实测):**
- 自实现投影 vs OpenCV(float64 输入):atol 1e-9 px(实测达 4.5e-13)
- OpenCV float32 绑定路径(calibrateCamera 输入输出):1e-3 px 量级
- 解析雅可比 vs 数值基准:列相对误差 1e-6
- 跨后端残差比较:一律用 models.project 自算(禁用 cv2.projectPoints 的 jacobian 输出,float32 下已实测不可靠)

---

## Per-Task Verification Map

*(已按 02-01~02-06 计划填充;File Exists 列:❌ = 测试文件由该任务创建,✅ = 修改既有文件)*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-1 | 02-01 | 1 | MOD-01 | T-02-01-SC/01 | numdifftools 仅 dev 组;注册表中文报错 | unit | `uv run python -c "import numdifftools; from calib.models.base import CameraModel" && uv run pytest -m "not integration" -q` | ✅ pyproject | ⬜ pending |
| 02-01-2 | 02-01 | 1 | MOD-02 | T-02-01-01/02 | 近轴除零保护;适配器 D 切片 | unit | `uv run python -c "from calib.models import create_model, BUILTIN_MODEL_IDS; [create_model(m) for m in BUILTIN_MODEL_IDS]" && uv run pytest -m "not integration" -q` | ❌ 新建 | ⬜ pending |
| 02-01-3 | 02-01 | 1 | MOD-02 | T-02-01-01 | parity 1e-9 / float32 1e-3;雅可比列相对 1e-6;src 无 numdifftools | unit | `uv run pytest tests/test_models_parity.py tests/test_models_jacobian.py -q && ! grep -rn "numdifftools" src/` | ❌ 新建 | ⬜ pending |
| 02-02-1 | 02-02 | 1 | SOL-01 | T-02-02-01/02/03 | npz allow_pickle=False;行序逐项断言;NaN 防线 | unit | `uv run python -c "from calib.solve.observation import ObservationSet, load_observation_set" && uv run pytest -m "not integration" -q` | ❌ 新建 | ⬜ pending |
| 02-02-2 | 02-02 | 1 | SOL-01 | T-02-02-02/03 | 装载契约六条(行序/剔除/未检出/NaN/下限/来源) | unit | `uv run pytest tests/test_solve_observation.py -q` | ❌ 新建 | ⬜ pending |
| 02-03-1 | 02-03 | 2 | SOL-01 | T-02-03-01 | yaml 白名单 + 中文校验(models/solvers/losses/f_scale/method) | unit | `uv run pytest tests/test_io_config.py -q && uv run python -c "from calib.io.config import load_calibration; load_calibration('config/calibration.yaml')"` | ✅ 修改 | ⬜ pending |
| 02-03-2 | 02-03 | 2 | SOL-01 | T-02-03-02 | 三件套落盘;失败契约 JSON null;回代统一尺 | unit | `uv run pytest tests/test_solve_solvers.py -q` | ❌ 新建 | ⬜ pending |
| 02-03-3 | 02-03 | 2 | SOL-01, MOD-01 | T-02-03-03/04 | float32 集中装包;失败不抛异常;求解器无具体模型 import | unit | `uv run pytest tests/test_solve_solvers.py tests/test_solve_baseline.py -q` | ❌/✅ | ⬜ pending |
| 02-04-1 | 02-04 | 3 | SOL-01 | T-02-04-01/02 | 残差仅 models.project;行序与数值雅可比双锁;向量化 | unit | `uv run pytest tests/test_solve_scipy_residuals.py -q` | ❌ 新建 | ⬜ pending |
| 02-04-2 | 02-04 | 3 | SOL-01 | T-02-04-03 | 实用收敛判定不静默(依据全进 solver_meta) | unit | `uv run pytest tests/test_solve_solvers.py -q` | ✅ 扩展 | ⬜ pending |
| 02-04-3 | 02-04 | 3 | SOL-01, MOD-01 | T-02-04-01/04 | D-08 行为+静态双锁;玩具模型零侵入 + 注册清理 | unit | `uv run pytest tests/test_models_registry.py tests/test_solve_scipy_residuals.py -q` | ❌ 新建 | ⬜ pending |
| 02-05-1 | 02-05 | 4 | SOL-01, ITER-01 | T-02-05-01/02 | 血缘 upstream 含初值链;时间戳目录不覆盖 | unit | `uv run python -c "from calib.solve.matrix import run_calibration_matrix" && uv run calib calibrate --help && uv run pytest -m "not integration" -q` | ❌/✅ | ⬜ pending |
| 02-05-2 | 02-05 | 4 | ITER-01 | T-02-05-01/04 | 临时树端到端;血缘 sha256 重算对账 | unit | `uv run pytest tests/test_calibrate_matrix.py -q -m "not integration"` | ❌ 新建 | ⬜ pending |
| 02-05-3 | 02-05 | 4 | SOL-01, ITER-01 | T-02-05-03/04/05 | 成功标准 3 判定;重跑一致 < 1e-12;12 组合血缘齐 | integration | `uv run pytest tests/test_calibrate_matrix.py -m integration -q` + 全矩阵后台实跑核对清单 | ✅ 扩展 | ⬜ pending |
| 02-06-1 | 02-06 | 5 | ITER-01 | T-02-06-04 | 位姿小求解模型无关(仅 CameraModel 接口);单帧失败记 NaN 不中断 | unit | `uv run pytest tests/test_solve_scoring.py -q` | ❌ 新建 | ⬜ pending |
| 02-06-2 | 02-06 | 5 | ITER-01 | T-02-06-01/02/03 | 705 行 join 硬断言;判定阈值随结论落盘;PNG 非空 | unit + integration | `uv run pytest tests/test_solve_scoring.py -q` + 真实 suspect_review 产物核对(705 行/89+616/与首轮 m2 一致) | ✅ 扩展 | ⬜ pending |
| 02-06-3 | 02-06 | 5 | ITER-01 | — | 文档契约与现实一致;需求状态可追溯 | doc + 全量 | `grep -q "__scipy" docs/pipeline.md && grep -q "Shipped (Phase 2)" .planning/REQUIREMENTS.md && uv run pytest -q` | ✅ 修改 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `uv add --group dev numdifftools`(0.11.1)—— 02-01 Task 1(唯一新依赖,研究已审计 [OK])
- [ ] 新测试文件按计划落地:parity/雅可比(02-01 Task 3)、observation(02-02 Task 2)、solvers(02-03 Task 2/3)、scipy_residuals + registry(02-04 Task 1/3)、calibrate_matrix(02-05 Task 2/3)、scoring(02-06 Task 1/2)—— 上表 ❌ 列即 Wave 0 缺口,全部有归属任务
- [ ] 现有 106 条测试保持全绿(回归基线,每任务提交前快速命令把关)

*测试框架与夹具基建阶段一已就位,无框架安装需求;合成真值视角夹具上提由 02-03 Task 3 完成。*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 全矩阵试车结果量级认可(收敛表 + 精化增益方向符合直觉) | SOL-01 | 数值达标可自动判,但"结果合理"需标定工程师目检 | 看 output/calibration/ 最新 run 的 comparison.json 与 CLI 汇总表:RMS 量级(m4 ≈0.08px、针孔族 ≈1px 量级)、精化后误差变化方向、失败组合的中文 failure_note |
| 存疑复核结论走向的人工认可(走向 A/B 皆算交付,但措辞与数字要人看过) | ITER-01(D-04) | 结论用于后续剔除决策,须人确认数字与措辞对得上 | 看 suspect_review/suspect_vs_normal.json 的 conclusion 与两组中位数,对照 .png 分布图 |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references(全部 ❌ 文件有归属任务)
- [x] No watch-mode flags
- [x] Feedback latency < 90s(快速命令;integration 与全矩阵实跑单独标注耗时预期)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending execution
