<!-- GSD:project-start source:PROJECT.md -->

## Project

**相机内参标定与评估框架 (intrinsic-calibration)**

一个评估驱动的相机内参标定与精度评估框架:输入预先采集好的标定图像(`input_data/`),输出当前数据条件下可论证的最优内参,以及完整的证据链报告——包括噪声底与可达精度预测(Fisher 信息/CRLB)、角点提取寻优证据、抗过拟合的模型选型、误差预算与参数敏感度诊断。用户是做相机标定研究的工程师,要求每一步结论都用数据证明。

**Core Value:** 对给定的一批标定图像,产出**可论证为当前数据下最优**的内参:held-out 泛化误差最小 + 参数稳定 + 误差逼近 CRLB 理论下界,三条判据同时达标,并量化"距离数据极限还有多远"。

### Constraints

- **Tech stack**: Python 3.12 + OpenCV 4.x(锁 <5.0) + numpy/scipy — 统计分析/可视化/快速迭代需要,离线场景性能足够;OpenCV 5.0 刚发布,churn 风险与可审计数值目标冲突
- **全局架构原则**: 阶段间仅通过带血缘的落盘产物通信(文件是唯一真相源),每阶段可独立运行/重跑/检查,输入输出契约显式 — 防止"分析无从下手"和迭代全量重跑
- **文档用语(硬性)**: 所有文档、报告、代码注释、CLI 帮助使用通俗中文;避免软件工程行话(如"切片/闭包/runner/语义"这类英语直译词),专业术语(如 Fisher 信息、雅可比)首次出现必须就地用一句白话解释;能用文字说清楚的不要堆术语 — 可读性优先于简洁
- **数据契约(硬性)**: input_data 全部图像必须来自同一相机、相同拍摄参数环境 — 单相机标定项目的前提
- **目录契约**: `input_data/train|test` 只读输入;`config/` 存放 board.yaml 与算法配置;运行产物写 `output/`(不进 git)
- **模块布局**: src/calib/ 下 io / corners / noise / models / solve / evaluation / analysis / visualization / report,与流水线阶段一一对应
- **单相机**: 所有标定、评估、报告均针对一台相机的一次标定任务

<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.12 | Runtime | Full wheel coverage across every dependency below; 3.13 works too but 3.12 is the conservative choice for a precision/numerics tool (longest testing tail). Not 3.11 — its support window closes sooner under SPEC 0 |
| opencv-python | 4.14.x, **pinned `<5.0`** | Corner detection (`findChessboardCorners`/`SB` + `cornerSubPix`), calibration solvers (`calibrateCamera` with `CALIB_RATIONAL_MODEL`, `cv2.fisheye.calibrate` for KB4), undistort primitives | Natively covers all 4 candidate models M1–M4. **Stay on 4.x**: OpenCV 5.0.0 shipped Aug 2026 and is only weeks old — calib3d is exactly the module where major-version behavior churn lands, and this project's whole value is reproducible, auditable numerics. Revisit 5.x after it accumulates a few point releases |
| numpy | 2.5.x | All array math, FIM assembly, Jacobians, artifact arrays (npz) | Latest stable line (2.5.4, Oct 2026). Note: 2.5.0 was a "transitional" release affecting compiled deps — pin numpy and scipy as a verified pair, see compatibility table |
| scipy | 1.18.x | `optimize.least_squares` (trust-region/LM + robust loss) for the generic `ScipySolver`; `stats` for bootstrap/CV statistics; sparse structures for FIM assembly | The generic solver over the model registry cannot be OpenCV (registry models must all share one residuals path); scipy is the standard. Robust kernels (soft_l1, Huber, Cauchy) are built into `least_squares` — no extra dependency |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | 3.10.x | All figure primitives: corner overlays, residual heatmaps, CV curves, CRLB-vs-actual bars, sensitivity rankings, pose coverage maps | Always — `visualization/` layer is matplotlib-only by design (architecture decision); PNG output into `output/figures/` |
| typer | ≥0.15 | CLI (`calib ingest|corners|noise|fisher|calibrate|evaluate|report|run-all`) | Always. Type-hint-driven subcommands map 1:1 onto pipeline stages; over `click` because the stage commands take typed options (model_id, k, seed) and typer generates help/validation for free. Runs on click underneath |
| pyyaml | ≥6.0 | `config/board.yaml` + `config/calibration.yaml` load & validation | Always. Read-only config; no round-trip editing, so ruamel.yaml is unnecessary |
| pandas | 2.3.x | Results tables: per-view error stats, CV fold tables, LOPO rankings, sensitivity rankings; CSV export for the report | Always for tabular experiment output; skip in the hot numeric path (numpy there) |
| jinja2 | ≥3.1 | Report assembly (HTML/MD evidence-chain report from `report_data.json` + figures) | `report/` stage only |
| joblib | ≥1.4 | Process-parallel experiment grid (4 models × solvers × 5 folds + 8 slices + N bootstraps + 80 LOPO ≈ 1.7k solves) | `evaluation/` runners; pure functions make folds embarrassingly parallel |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| uv | Env + dependency management, lockfile | Fast, deterministic locks matter because OpenCV/numpy/scipy must stay a pinned trio |
| pytest | Unit tests — especially model parity tests (project vs OpenCV `projectPoints`/`fisheye.projectPoints`) and numeric-Jacobian checks | The parity/Jacobian test suite is the enforcement mechanism for the shared-math architectural constraint |
| ruff | Lint + format | Standard for new Python projects |
| mypy | Type checking | Recommended given the `CameraModel` ABC / `Protocol` heavy design; optional if it slows iteration |
| numdifftools (dev-only) | Finite-difference Jacobian oracle | **Test oracle only** — never in the solve/FIM path. Validates analytic Jacobians in `models/` |

## Installation

# Core (pin the trio together)

# Supporting

# Dev dependencies

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| opencv-python 4.14.x | opencv-python 5.0.0 (Aug 2026) | Only after 5.0.x stabilizes (≥2–3 point releases) and a migration audit of calib3d/fisheye changelog is done. No current benefit |
| typer | click | If the team strongly prefers explicit decorators over type hints; functionally equivalent (typer wraps click) |
| scipy `least_squares` | Ceres (via pybind) / g2o | Never for this project — C++ toolchain, breaks the pure-Python deliverable, and its auto-diff Jacobians would bypass the shared analytic Jacobian constraint |
| matplotlib | plotly / bokeh | Only if the report becomes interactive (out of scope); matplotlib PNGs are the right artifact for an evidence-chain PDF/HTML |
| pandas CSV + JSON/npz artifacts | SQLite / DuckDB artifact store | If experiment grids grow to 10k+ runs with query needs; current scale (~1.7k solves) is fine with files |
| pyyaml | ruamel.yaml | Only if config must be programmatically edited with comment preservation — not the case here (configs are hand-authored, read-only) |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| OpenCV 5.0.x (for now) | Released Aug 2026; major-version churn risk in exactly the modules used (calib3d, fisheye); auditable numerics require a stable, documented baseline | opencv-python 4.14.x pinned `<5.0` |
| jax / torch / autograd for Jacobians | Project *requires* hand-derivable analytic Jacobians shared across models/noise/evaluation; autodiff adds GPU-sized dependency weight, tracing complexity, and a second math implementation to keep in sync | Analytic Jacobians in `models/`, validated against `numdifftools` in tests |
| mrcal / Kalibr as dependencies | Both are C++-centric toolchains with different deliverable shapes (mrcal: splined models, no CRLB prediction; Kalibr: ROS-entangled). Useful as *references*, poison as *deps* | Own implementation; consult mrcal docs for uncertainty methodology |
| `cv2.optflow`, contrib-only modules | opencv-contrib adds nothing for chessboard-only detection; `findChessboardCornersSB` is in main calib3d | opencv-python (non-contrib). If ChArUco arrives later (v2+), `cv2.aruco` has been in the main module since 4.7 |
| numba / cython hot-path optimization | One solve is seconds (113k residuals, <500 params); the bottleneck is experiment count, solved with process parallelism, not per-solve speed | numpy + scipy as-is, joblib for grids |
| Windows/macOS-specific tooling | Linux dev environment; OpenCV/numpy/scipy all first-class on Linux | uv + pip standard flow |

## Stack Patterns by Variant

- Keep OpenCVSolver as ground truth and treat ScipySolver anomalies as findings to document (multi-optimizer comparison is itself a project deliverable), not as a reason to add Ceres.
- Use scipy `least_squares(jac_sparsity=...)` and block-assembly in numpy (each residual touches 4+D intrinsics + 6 pose params) before considering any compiled extension.
- First try OpenCV drawing + homography warps (already in-stack); reach for a renderer dependency only if evidence demands sub-texture/defocus physics (unlikely to clear the complexity bar).

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| opencv-python 4.14.x | numpy 2.x (≥2.0) | 4.x line has supported numpy 2 since 4.10; verified pair 4.14 + 2.5 is the current safe set |
| numpy 2.5.x | scipy ≥1.17 (1.18.1 current) | numpy 2.5.0 was explicitly "transitional" for compiled deps like scipy — install scipy/numpy as a tested pair, not independently latest |
| Python 3.12 | all of the above | Full wheel matrix (manylinux); 3.13 also covered by current releases but 3.12 has the longest burn-in |
| typer ≥0.15 | click 8.1+ | Typer pins its own click range; no manual pin needed |

## Sources

- [NumPy news](https://numpy.org/news) — 2.5.0 transitional release (Jun 2026), 2.5.4 (Oct 10, 2026) — HIGH
- [SciPy homepage](https://scipy.org) — 1.18.1 (Aug 21, 2026); SPEC 0 dependency window policy — HIGH
- [opencv-python on PyPI](https://pypi.org/project/opencv-python) — 5.0.0.93 wrapper; [opencv-python releases](https://github.com/opencv/opencv-python/releases) — 4.14.0 (Jul 28, 2026), Python 3.10–3.13 support — HIGH
- [OpenCV releases](https://opencv.org/releases/) — OpenCV 5.0.0 official release Aug 19, 2026 — HIGH
- OpenCV 4.x calib3d/fisheye API docs (docs.opencv.org/4.x) — M1–M4 model coverage: `calibrateCamera` + `CALIB_RATIONAL_MODEL`, `cv2.fisheye` KB4, `findChessboardCornersSB` in main module — HIGH (verified in PITFALLS/ARCHITECTURE research)
- Local pip availability check — FAILED (sandbox network restriction); exact version resolution deferred to first `uv lock` — flagged MEDIUM until then

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
