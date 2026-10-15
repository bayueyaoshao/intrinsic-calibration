# Architecture Research

**Domain:** Camera intrinsic calibration & evaluation framework (offline Python pipeline, single camera, evidence-chain reporting)
**Researched:** 2026-10-15
**Confidence:** HIGH (core patterns verified against Kalibr/aslam, basalt-headers, COLMAP, OpenCV/scipy docs; a few design points flagged MEDIUM below)

## Standard Architecture

### System Overview

Offline batch pipeline — no users, no servers. The "scale axis" is experiment count (models × solvers × folds), not concurrency. The canonical shape for this domain is a **stage DAG over persisted artifacts**, with a small set of pure-core libraries (models, solvers, experiment runners) that every stage calls.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          CLI / pipeline (run-all)                         │
│      calib ingest|corners|noise|fisher|calibrate|evaluate|report          │
├──────────────────────────────────────────────────────────────────────────┤
│                              Stage Libraries                              │
│  ┌────────┐ ┌─────────┐ ┌───────┐ ┌──────────────┐ ┌──────────┐          │
│  │  io    │→│ corners │→│ noise │→│    solve     │→│evaluation│          │
│  │precheck│ │ detect  │ │ + FIM │ │ OpenCV/scipy │ │ 4 experi-│          │
│  │manifest│ │ refine  │ │ CRLB  │ │  (interface) │ │  ments   │          │
│  └────────┘ └─────────┘ └───────┘ └──────────────┘ └──────────┘          │
│       ┌──────────────────────────────────────────────────┐               │
│       │   analysis (error budget, sensitivity, coupling) │               │
│       └──────────────────────────────────────────────────┘               │
├──────────────────────────────────────────────────────────────────────────┤
│                       Shared Core (single source of truth)                │
│  ┌───────────────────────────────────┐  ┌──────────────────────────┐     │
│  │ models: CameraModel ABC + registry│  │ _types: ObservationSet,  │     │
│  │ project/unproject/jacobian        │  │ CalibrationResult,       │     │
│  │ ← used by solve, noise, analysis  │  │ BoardSpec, configs       │     │
│  └───────────────────────────────────┘  └──────────────────────────┘     │
├──────────────────────────────────────────────────────────────────────────┤
│  visualization (matplotlib primitives)   report (assembly: HTML/MD)      │
├──────────────────────────────────────────────────────────────────────────┤
│              output/ (artifact store: JSON scalars + npz arrays)          │
│   ingest/  corners/  noise/  fisher/  calibration/  evaluation/  report/  │
└──────────────────────────────────────────────────────────────────────────┘
         ↑ read-only                    config/ (board.yaml, calibration.yaml)
   input_data/train|test
```

This matches how mature calibration systems are built. Kalibr separates projection math (`aslam_cameras`), residuals + solver (Ceres), and experiment scripts; basalt-headers ships camera models as a standalone reusable layer with a `project/unproject + optional Jacobians` contract; COLMAP isolates camera models behind per-model parameter traits. None of them let distortion math live in more than one place — which is exactly this project's shared-Jacobian constraint.

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `io` | Load `input_data/train\|test` under the directory convention (pose dir × burst index), run ingest prechecks (blur via Laplacian variance, exposure histogram, board visibility), emit `ObservationSet` manifests | Plain functions + `@dataclass`; JSON manifest per split |
| `corners` | Detector abstraction → per-image subpixel corner extraction; burst aggregation candidates (per-pose mean vs all-frames); writes the canonical corners npz | `cv2.findChessboardCorners` + `cornerSubPix` behind a `Detector` protocol; npz + JSON report |
| `noise` | Estimate noise floor σ from same-pose bursts (temporal) with Immerkaer single-frame fallback; then compute Fisher information / CRLB per intrinsic using `models.jacobian` at a linearization point; pose-degeneracy detection (eigen-spectrum, fx–cy coupling) | numpy; FIM = marginalized JᵀJ/σ² |
| `models` | **Owns all projection math.** `CameraModel` ABC: `project`, `unproject`, `project_jacobian`, param-vector layout (`param_names`); registry for M1–M4 + zero-intrusion extension | 4 model classes; registry dict `model_id → class`; parity + numeric-Jacobian tests |
| `solve` | Solver interface with two interchangeable impls: `OpenCVSolver` (wraps `calibrateCamera` / `fisheye.calibrate`, maps model_id → flags) and `ScipySolver` (generic residuals over any registry model via `least_squares`, robust kernels). Both return the same `CalibrationResult` | Protocol class + two implementations; residuals built **only** from `models.project` |
| `evaluation` | Experiment runners as **pure functions over a calibrator**: K-fold CV by pose, repeat-index slice ×8, bootstrap (by pose), leave-one-pose-out ×80; metrics (RMS/max held-out reproj error, param spread, NEES vs CRLB) | `def kfold_pose_cv(obs, calibrate_fn, k, seed) -> Result`; serializable results |
| `analysis` | Error budget: actual σ vs CRLB per parameter; error-direction decomposition (per view / image region / radial mode); parameter sensitivity ranking; corner-extraction optimization experiments (synthetic degradation ladder) | numpy/matplotlib; consumes `models.jacobian` + solve residuals |
| `visualization` | Figure primitives reused by all stages: corner overlays, residual heatmaps, CV curves, CRLB-vs-actual bars, sensitivity ranking, pose coverage maps | Thin matplotlib wrappers, figure → png into `output/figures/` |
| `report` | Assemble stage artifacts + figures into evidence-chain report (HTML or MD); every conclusion links to its artifact | Jinja2 template + `report_data.json` |
| `cli`/`pipeline` | Per-stage entry points + `run-all` orchestrator; config load/validate; seed management | argparse (or typer); thin — logic lives in libraries |

## Recommended Project Structure

```
src/calib/
├── _types.py            # ObservationSet, CornerObs, ImageRef, BoardSpec,
│                        # CalibrationResult, ExperimentResult — dataclasses only
├── config.py            # load/validate board.yaml + calibration.yaml → dataclasses
├── io/                  # ingest: directory walk, prechecks, manifest
│   └── precheck.py
├── corners/             # Detector protocol, cv2 impl, subpixel refine,
│   └── optimize.py      # burst aggregation, corner-expt optimization runner
├── noise/               # temporal σ, Immerkaer fallback, FIM/CRLB, degeneracy
│   └── fisher.py
├── models/              # THE shared projection/Jacobian layer
│   ├── base.py          #   CameraModel ABC + registry
│   ├── pinhole_brown.py #   M1 (k1k2p1p2), M2 (k1k2k3p1p2), M3 (rational k1–k6p1p2)
│   ├── kb4.py           #   M4 fisheye equidistant KB4
│   └── _math.py         #   private: rotation, distort/unDistort kernels
├── solve/               # Solver protocol, residual builder (generic),
│   ├── opencv.py        #   OpenCVSolver (flags per model_id)
│   ├── scipy_solver.py  #   ScipySolver (trf/lm, robust loss, sparse jac)
│   └── residuals.py
├── evaluation/          # metrics + 4 pure-function experiment runners
│   ├── experiments.py   #   kfold_pose_cv / repeat_slice / bootstrap / loop
│   └── diagnostics.py
├── analysis/            # error budget, sensitivity ranking, direction
│   └── budget.py        #   decomposition, corner-expt synthesis
├── visualization/
│   └── plots.py
├── report/
│   └── build.py
├── pipeline.py          # run-all: stage order + skip-if-fresh
└── cli.py               # python -m calib <stage> ...
tests/
├── test_model_parity.py     # our project() ≡ cv2.projectPoints / fisheye maps
├── test_jacobian_numeric.py # analytic vs central-difference Jacobians
├── test_experiments.py      # split correctness: no pose leaks across folds
└── ...
config/
├── board.yaml
└── calibration.yaml
```

### Structure Rationale

- **`models/` as a leaf module:** depends only on numpy. `solve`, `noise` (FIM), and `analysis` (diagnostics) import it — never the reverse. This *structurally enforces* the shared-Jacobian constraint: if analysis needs a Jacobian, there is exactly one place to get it. (Verified pattern: Kalibr/basalt keep projection in a standalone layer precisely so solver and consumers can't drift.)
- **`_types.py` before everything:** every stage speaks in `ObservationSet` / `CalibrationResult`. The data model is the API between stages; get it right first.
- **`evaluation/` separate from `analysis/`:** evaluation *runs experiments* (CV, slices, bootstrap) and produces tables; analysis *interprets* them (budget, sensitivity). Report consumes both. Mixing them is how "runners" accrete plotting code.
- **Leading-underscore `_math.py`:** signals "private to models/" — the only place rotation/distortion kernels may live.
- **`pipeline.py` thin by design:** orchestration is a list of stage calls, not logic. Stage independence comes from the artifact store, not from the orchestrator.

## Architectural Patterns

### Pattern 1: Single-Source Projection/Jacobian Layer (the shared-Jacobian constraint)

**What:** All projection math — distortion kernels, Jacobians — lives in `models/`. Residuals (`solve/residuals.py`), Fisher information (`noise/fisher.py`), and diagnostics (`evaluation/diagnostics.py`, `analysis/budget.py`) all call the same `CameraModel` methods. No consumer re-implements or approximates them.

**When to use:** Always, here — this is the project's core architectural constraint (PROJECT.md: "理论界与实际求解必须严格可比"). The CRLB is only a meaningful benchmark against actual solver error if both use identical math.

**Trade-offs:** Slightly slower than a hand-inlined Fisher Jacobian (negligible at 56k residuals); requires the Jacobian of residuals w.r.t. *intrinsics only* to fall out of `project_jacobian` via chain rule through the pose transform.

**Example:**
```python
# models/base.py — the contract
class CameraModel(ABC):
    model_id: str
    param_names: tuple[str, ...]          # ordered → param vector layout

    @abstractmethod
    def project(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        """(N,3) camera-frame points + (P,) params -> (N,2) pixels."""

    @abstractmethod
    def project_jacobian(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        """d pixel / d param, shape (N, 2, P) — analytic."""

    @abstractmethod
    def unproject(self, pts_px: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Pixels -> normalized/rays. Needed for test-set scoring & init."""

def create_model(model_id: str) -> CameraModel: ...   # registry lookup

# solve/residuals.py — residuals use ONLY this
def residuals(theta: np.ndarray, model, obs, n_int: int) -> np.ndarray:
    r = np.empty(2 * obs.n_points)
    for view in obs.views:                      # extrinsics block of theta
        R, t = exp_se3(theta[n_int + 6*view.i : n_int + 6*view.i + 6])
        pts = (R @ obs.object_points.T).T + t   # world -> camera
        uv = model.project(pts, theta[:n_int])
        r[2*view.sl : 2*view.sl+2*view.n] = (uv - obs.image_points[view.sl]).ravel()
    return r

# noise/fisher.py — SAME jacobian, marginalized over poses
# F = J_θᵀ J_θ − J_θᵀ J_β (J_βᵀ J_β)⁻¹ J_βᵀ J_θ,  CRLB σ_k = sqrt((F⁻¹)_kk / σ²)
```

Verification hooks that keep this honest (build these as tests, not conventions):
1. **Parity test:** `our_model.project` ≡ `cv2.projectPoints` (M1–M3) and ≡ `cv2.fisheye.projectPoints` (M4) on identical params, within float tolerance. This is what makes OpenCV-solver results and scipy-solver results comparable at all.
2. **Numeric-Jacobian test:** analytic `project_jacobian` vs central differences, per model, per parameter.
3. **Cross-check:** OpenCV's `calibrateCameraExtended` returns `stdDeviationsIntrinsics` (itself an inverse-Jacobian estimate) — compare against our FIM diagonal at σ=1 as an independent sanity anchor.

### Pattern 2: Solver Interface with Canonical CalibrationResult (OpenCV ⇄ scipy interchangeable)

**What:** A `Solver` protocol both OpenCV and scipy backends satisfy; both emit the same `CalibrationResult` in the model's canonical `param_names` ordering, so `evaluation/` and `analysis/` are solver-agnostic.

**When to use:** Whenever multiple optimizers must be compared (a project requirement). The interface is the comparison's validity condition.

**Trade-offs:** OpenCV is battle-tested with excellent homography-based initialization but is a black box internally (no custom robust kernels; the fisheye solver is notoriously brittle with outliers and needs `CALIB_CHECK_COND` + failure handling). scipy `least_squares` (stable API since 1.0: `fun → residual vector`, `jac → (N, P)` dense or sparse, methods `trf/lm/dogbox`, losses `linear/soft_l1/huber/cauchy/arctan`) is fully transparent but needs initialization and its own Jacobian assembly — which Pattern 1 provides for the intrinsics block. Recommended hybrid: **OpenCV initializes → scipy refines/polishes** (Kalibr-style), and the optimizer-comparison experiment is then meaningful because both paths share the model layer.

**Example:**
```python
class Solver(Protocol):
    name: str
    def solve(self, obs: ObservationSet, model: CameraModel,
              cfg: SolverConfig, init: CalibrationResult | None = None
              ) -> CalibrationResult: ...

class OpenCVSolver:
    name = "opencv"
    _FLAGS = {  # model_id -> calibrateCamera flags
      "pinhole_brown_k1k2p1p2": 0,
      "pinhole_brown_k1k2k3p1p2": 0,
      "pinhole_rational_k1_k6p1p2": cv2.CALIB_RATIONAL_MODEL,
    }  # kb4 routes to cv2.fisheye.calibrate

class ScipySolver:
    name = "scipy"
    def solve(self, obs, model, cfg, init=None):
        theta0 = pack(init or initial_guess(obs, model))     # intrinsics + per-view (rvec,tvec)
        jac_sparsity = build_block_sparsity(obs, model)      # each residual: 4+D intrinsics + 6 pose
        res = least_squares(residuals, theta0, jac=analytic_jac, jac_sparsity=jac_sparsity,
                            loss=cfg.loss, f_scale=cfg.f_scale, method=cfg.method)
        cov = gauss_newton_covariance(res)                   # σ̂² (JᵀJ)⁻¹ → empirical σ vs CRLB
        return CalibrationResult.from_theta(res, model, obs, cov=cov, meta=cfg)
```

### Pattern 3: Experiment Runners as Pure Functions over a Calibrator (sklearn `GroupKFold` shape)

**What:** The four evaluation experiments are pure functions taking `(observations, calibrate_fn, config, seed)` where `calibrate_fn: (ObservationSubset) -> CalibrationResult` closes over the chosen model + solver. No runner mutates state; results are plain serializable tables.

**When to use:** Always — this is what makes the experiment grid (4 models × N solvers × K folds) tractable and parallelizable (each fold is an independent process), and what makes results reproducible from `(config, seed)` alone.

**Trade-offs:** A closure/protocol boundary more than a script would have — but scripts are exactly what makes these experiments silently inconsistent (different init or flags per experiment).

**Key correctness rule (the project's data-leakage invariant):** all splitting happens on `pose_id`, never on images. Same-pose bursts are near-duplicates; image-level splits leak. This is the sklearn `GroupKFold(groups=pose_id)` / `LeaveOneGroupOut` pattern applied to calibration — structural, not procedural.

**Example:**
```python
def kfold_pose_cv(obs: ObservationSet, calibrate: Calibrator,
                  k: int = 5, seed: int = 0) -> CVResult:
    poses = np.unique(obs.pose_ids)
    folds = stratified_shuffle_split(poses, k, seed)      # split pose ids, not images
    rows = []
    for train_poses, val_poses in folds:
        res = calibrate(obs.select(poses=train_poses))    # pure: subset → fit
        rows.append(score(val=res, val_set=obs.select(poses=val_poses),
                          metrics=["rms", "max", "per_param_sigma"]))
    return CVResult(folds=rows, summary=aggregate(rows))  # JSON-serializable
```

`repeat_slice` (fix poses, sweep repeat index 0–7 → pure-noise parameter covariance → the CRLB empirical anchor), `bootstrap` (resample *poses*), and `leave_one_pose_out` (problem-pose localization) are the same function shape with different split generators.

## Data Flow

### Stage Flow (pipeline = linear chain with two joins)

```
input_data/train|test (read-only)
    ↓ io.ingest: walk pose dirs, prechecks, manifest
output/ingest/manifest.json
    ↓ corners.detect: Detector → subpixel refine → burst aggregation
output/corners/corners_{train,test}.npz  (+ detector_report.json)
    ↓                                  ↘
noise.estimate (temporal σ from         │ (corners only)
bursts | Immerkaer fallback)            ↓
    ↓                        solve.calibrate (initial pass: default model+solver)
output/noise/noise_report.json   →  output/calibration/init/result.json  ──┐
    ↓                                                                      │ linearization
noise.fisher: FIM/CRLB at linearization point ← models.project_jacobian    │ point
    ↓                                                                      │
output/fisher/fisher_report.json (per-param σ_floor, eigen spectrum,       │
        coupling/degeneracy warnings)                                      │
                                                                           ↓
        solve.calibrate (experiment grid: M1–M4 × solvers, CV-selected)  ←─┘
    ↓
output/calibration/<model>__<solver>/{result.json, params.npz, residuals.npz}
    ↓                                     ↓
evaluation.experiments (4 runners)   analysis.budget (actual σ vs CRLB,
    ↓                                   direction decomposition, sensitivity)
output/evaluation/{cv,repeat_slice,bootstrap,leave_one_pose_out}.json
output/analysis/{budget,sensitivity}.json
    ↓
visualization.plots → output/figures/*.png
    ↓
report.build → output/report/index.html
```

### Artifact Store (what each stage persists, and why)

| Artifact | Format | Consumed by | Content |
|----------|--------|-------------|---------|
| `ingest/manifest.json` | JSON | corners, report | Every image ref: path, pose_id, burst_idx, split, precheck verdicts |
| `corners/corners_train.npz` (+test) | npz + JSON sidecar | noise, solve, corner-expt | Long-form arrays: `pose_ids`, `repeat_idx`, `image_ids`, `uv (M,2)`, `point_ids`, detector config hash |
| `noise/noise_report.json` | JSON | fisher, report | σ_temporal (source-labeled), σ_immerkaer fallback, which was used |
| `fisher/fisher_report.json` | JSON | analysis, report | Per-param CRLB σ, FIM eigen-spectrum, condition number, coupling warnings |
| `calibration/<run>/result.json` | JSON | evaluation, analysis, report | Params (named), RMS, solver meta, convergence, BIC inputs |
| `calibration/<run>/residuals.npz` | npz | analysis, visualization | Per-view residual matrix (enables heatmaps + budget without re-solving) |
| `calibration/<run>/covariance.npz` | npz | analysis | Empirical param covariance (from JᵀJ) |
| `evaluation/*.json` | JSON | analysis, report | Fold/slice/loop tables + summaries |
| `output/manifest.json` | JSON | everything | Config-content hashes + seed ledger — provenance for "which config produced this" |

**Rules that make stage independence work:**
- JSON for scalars/small tables, npz for arrays. A stage reads only its predecessors' artifacts — never re-derives from images.
- Every artifact carries the config hash that produced it (skip-if-fresh for `run-all`; explicit re-run otherwise).
- CLI per stage: `python -m calib corners --config calibration.yaml --out output/` must work on a fresh process against existing upstream artifacts. `run-all` is just the stages in order.

### Key Data Flows

1. **Corner observations → residuals → everything.** `ObservationSet` is loaded once from npz; every downstream consumer (solver residuals, CV subsetting, FIM assembly) is a masked view over the same long-form arrays (`pose_ids`, `repeat_idx` are the join keys). One load, zero re-parsing.
2. **σ noise → FIM → CRLB → budget.** The noise floor estimate is the *only* coupling between the measurement side and the theory side; it flows as a single scalar (per-axis optional) into `noise/fisher.py`, and the resulting CRLB vector is later compared against empirical σ from `repeat_slice` and the solver covariance in `analysis/budget.py`. Keeping this chain scalar-and-JSON is what makes the "distance to data limit" claim auditable.
3. **CV selection → final refit.** Model/hyperparameter selection reads only CV summaries; the delivered intrinsics come from one final refit on *all* train data with the selected config (PROJECT.md decision), then scored once on the held-out `test/` split.

## Suggested Build Order (dependency-driven)

```
1. _types + config          ← zero deps; the API between stages
2. models/ (4 models + registry + parity & Jacobian tests)
3. io/ (ingest, precheck, manifest)
4. corners/ (cv2 detector behind protocol → corners.npz)      ┐ can start with
5. noise/ σ estimation (temporal + Immerkaer)                 ┘ stage-1 data
6. solve/ (interface + OpenCVSolver + ScipySolver + residuals)  needs 1,2
7. noise/fisher (FIM/CRLB, degeneracy)                          needs 2,5,6(init)
8. evaluation/ (metrics + 4 pure runners)                       needs 6
9. analysis/ (budget, sensitivity, direction decomposition)     needs 7,8
10. corners/optimize (synthetic-degradation expt)               needs 8 (reuses runners)
11. visualization + report                                       consumes artifacts
12. pipeline.py + cli.py finalization (stub CLI from step 3 onward)
```

**Why this order:**
- **models before solve** — the solver's residuals are built from `model.project`; building solve first guarantees duplicated math (the cardinal sin of this domain). Parity tests land in step 2, when they can still block bad design.
- **noise σ before fisher** — FIM/CRLB is meaningless without the σ that scales it.
- **solve before fisher** — FIM needs a linearization point; the cheapest source is an initial calibration result (avoids a chicken-and-egg stage).
- **evaluation before corners-opt** — the corner-extraction experiment is just another grid of `calibrate_fn` variants; building it after the runners means it reuses them instead of spawning a bespoke harness.
- **visualization/report last** — pure consumers; but write figure primitives opportunistically per stage so report assembly is trivial at the end.
- CLI stubs should exist from step 3 (each stage runnable independently the moment it exists) — only the final `run-all` polish waits.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Single calibration task (current: 640 imgs × 88 pts ≈ 113k residuals, <500 params) | Nothing needed. One solve is seconds; the whole pipeline is minutes. JSON+npz artifact store is the right size — no database |
| Experiment grid blow-up (4 models × 3 solvers × (5 folds + 8 slices + N bootstraps + 80 LOPO)) | Parallelize runners with `multiprocessing`/joblib — pure functions make folds embarrassingly parallel; each worker loads corners npz once |
| 100+ bootstrap iters or finer corner-param grids | Cache per-fold sub-`ObservationSet`s as npz to avoid re-masking; keep solver `meta` in results so partial reruns can be diffed |
| Future multi-camera / new models (out of scope) | Registry already leaves the door open: a new model is one class + one registry entry; nothing else changes. Do not pre-build multi-camera plumbing |

### Scaling Priorities

1. **First bottleneck:** experiment count, not solver speed. A 4×3×(5+8+50+80) grid ≈ 1.7k solves → parallelize folds/slices across processes before optimizing solver code.
2. **Second bottleneck:** FIM assembly with dense per-view Jacobians (views × 88 pts × 2 × ~500 params). Use the block sparsity (each residual touches 4+D intrinsic + 6 pose params) — `jac_sparsity` in scipy, block assembly in numpy. Memory stays in the tens of MB.

## Anti-Patterns

### Anti-Pattern 1: Duplicating projection math in the Fisher/diagnostic code

**What people do:** Copy the (concise) distortion Jacobian into the Fisher module "for speed" or because "it's just the diagonal."
**Why it's wrong:** The instant the two implementations differ by a normalization or a k3 sign, the CRLB benchmark is computed against a *different model* than the solver optimized — the project's central claim ("误差逼近 CRLB") becomes unfalsifiable, and nothing fails loudly.
**Do this instead:** `models/` is the only file that contains projection math; imports are the enforcement; parity + numeric-Jacobian tests make drift a red test.

### Anti-Pattern 2: Using OpenCV's distortion implementation for solving and the project's for theory without a parity check

**What people do:** Trust `cv2.calibrateCamera` and write an independent Fisher module, assuming the models coincide.
**Why it's wrong:** OpenCV's rational model, fisheye KB4, and its parameter ordering are not guaranteed to match a hand-rolled implementation; parameter vectors would be silently non-comparable, and optimizer comparisons would measure implementation differences, not optimizer differences.
**Do this instead:** Parity tests (`project` vs `cv2.projectPoints` / `cv2.fisheye.projectPoints`) and a canonical named-parameter vector that OpenCV results are unpacked into at the solver boundary.

### Anti-Pattern 3: Splitting by image instead of by pose

**What people do:** K-fold over the 640 training images (natural, since images are the file unit).
**Why it's wrong:** 8 same-pose bursts are near-duplicates; a val set containing a sibling of a train image reports optimistic generalization — the anti-overfitting machinery itself gets fooled.
**Do this instead:** All splits operate on `pose_id` (GroupKFold semantics). Encode the invariant once in `ObservationSet.select(poses=...)` and never expose image-level splitting.

### Anti-Pattern 4: Persisting only final parameters

**What people do:** Each stage writes its headline result (intrinsic matrix, RMS) and recomputes or re-solves when a later stage needs detail.
**Why it's wrong:** Diagnostics need per-view residuals, budgets need covariances, reports need fold curves — re-solving per consumer makes stages non-independent, slow, and irreproducible.
**Do this instead:** Persist `result.json + residuals.npz + covariance.npz` per calibration run (artifact table above); downstream stages are pure readers.

### Anti-Pattern 5: Global mutable solver/model state ("current model" module variable)

**What people do:** Set module-level globals before calling a calibrate function in a loop.
**Why it's wrong:** Experiment runners become order-dependent; parallel workers share state; results can't be attributed to (config, seed).
**Do this instead:** `calibrate_fn` closures over (model, solver, cfg); runners are pure functions; the seed ledger lives in `output/manifest.json`.

### Anti-Pattern 6: Naive FIM that ignores pose nuisance parameters

**What people do:** FIM = JᵀJ/σ² over intrinsics only, evaluated at nominal poses.
**Why it's wrong:** Ignoring extrinsic uncertainty yields an optimistic lower bound; the fx–cy-style coupling the project wants to detect hides in the marginalization term. The predicted floor would flatter the data.
**Do this instead:** Marginalize extrinsics (Schur complement over the per-view pose block) in `noise/fisher.py`. *(MEDIUM confidence design point — standard estimation theory, but the exact treatment of σ estimation and burst correlation deserves a focused pass when implementing the Fisher phase.)*

## Integration Points

### External Libraries

| Library | Integration Pattern | Notes |
|---------|---------------------|-------|
| OpenCV (`opencv-python`, 4.11 present on system) | `corners/`: detection; `solve/opencv.py`: solver backend | Fisheye lives in a separate namespace (`cv2.fisheye.*`) with its own flags — route by `model_id`. Per-view errors/std-devs come from the `*Extended` calibrate bindings (verify availability with a smoke test — not introspectable on the current system python due to missing numpy). The fisheye solver is brittle with bad corners: wrap in try/except and record failures as first-class experiment results |
| scipy `optimize.least_squares` | `solve/scipy_solver.py` | Stable documented contract: `fun(x)→residual vector`, callable `jac`, `jac_sparsity`, methods `trf/lm/dogbox`, losses incl. `soft_l1/huber` |
| numpy | data model, persistence (npz), linear algebra | Long-form observation arrays; FIM/covariance math |
| matplotlib | `visualization/` primitives only | No plotting outside this module — report imports figures, never plots |
| pandas (optional, recommended) | experiment result tables | The four runners emit tables; pandas makes the analysis/JSON boundary painless. Keep it out of the core data model |
| jinja2 (optional) | `report/` HTML assembly | Markdown fallback if dependency budget is tight |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| stages ⇄ shared core | `models` API + `_types` dataclasses | models is a leaf: nothing may add math there, only consume |
| solve ⇄ models | `project` / `project_jacobian` / `param_names` | OpenCV results are unpacked into the canonical param vector at the boundary |
| noise(FIM) ⇄ models | `project_jacobian` only | FIM assembly never touches OpenCV distortion code |
| evaluation ⇄ solve | `Calibrator` protocol (pure `fn(obs)→result`) | runners must not import concrete solvers |
| visualization ⇄ stages | functions taking arrays/tables → png paths | stages never pass domain objects to plotting |
| report ⇄ everything | reads artifacts from `output/` only | report must be rebuildable without re-running any stage |
| config ⇄ stages | `config.py` dataclasses | yaml → validated dataclasses once; stages never see raw dicts |

## Config Architecture

Two files, both validated into dataclasses by `config.py`; experiments reference solver/model configs **by name**, resolved against the same registries the code uses.

```yaml
# config/board.yaml
board:
  type: checkerboard
  inner_corners: [11, 8]        # 88 points
  square_size_mm: 25.0

# config/calibration.yaml
data: {train: input_data/train, test: input_data/test}
seed: 42

corners:
  detector: {type: cv2_cornerSubPix, window: [5, 5]}
  aggregation: candidate         # per-pose mean vs all-frames — decided by slice experiment

noise: {method: temporal, fallback: immerkaer}

models: [pinhole_brown_k1k2p1p2, pinhole_brown_k1k2k3p1p2,
         pinhole_rational_k1_k6p1p2, kb4_fisheye]

solvers:
  opencv_default: {type: opencv}
  scipy_trf_softl1: {type: scipy, method: trf, loss: soft_l1, f_scale: 1.0}
  scipy_lm:         {type: scipy, method: lm, loss: linear}
defaults: {model: pinhole_brown_k1k2k3p1p2, solver: opencv_default}

evaluation:
  kfold: {k: 5}
  repeat_slice: {slices: 8}
  bootstrap: {n: 50}
  leave_one_pose_out: {}
analysis: {report_degeneracy_eigens: true}
report: {format: html}
```

Design points:
- **Experiments are names, not inline specs:** a grid entry like `(model=kb4_fisheye, solver=scipy_trf_softl1)` resolves through `create_model()` / solver registry — config can never name a model the code doesn't implement, and experiment provenance is a pair of strings.
- **One seed, derived sub-seeds:** every experiment gets `seed = base + hash(experiment_name)`; the ledger goes in `output/manifest.json`.
- **board.yaml stays separate** because it describes the physical artifact (shared across experiments), while calibration.yaml describes the algorithm choices (iterated on).

## Sources

- [ethz-asl/kalibr](https://github.com/ethz-asl/kalibr) and [supported models wiki](https://github.com/ethz-asl/kalibr/wiki/supported-models) — pluggable projection models (pinhole/radtan/equidistant/MEI/DS/EUCM) behind one camera abstraction (HIGH)
- [kalibr aslam_cv_python CameraProjections.cpp](https://github.com/ethz-asl/kalibr/blob/master/aslam_cv/aslam_cv_python/src/CameraProjections.cpp) — project-to-keypoint with explicit Jacobian out-params: the canonical `project+jacobian` contract (HIGH)
- [basalt-headers KannalaBrandtCamera4](https://vladyslavusenko.gitlab.io/basalt-headers/classbasalt_1_1KannalaBrandtCamera4.html) — "Project the point and optionally compute Jacobians. Unproject… " as the documented model API (HIGH)
- [COLMAP camera models](https://colmap.github.io/cameras.html) / [models.h](https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h) — per-model parameter-layout traits, shared radtan/fisheye kernels (HIGH)
- [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) — residual-vector contract, callable `jac`, sparse `jac_sparsity`, `trf/lm`, robust losses (HIGH; stable API; local introspection blocked — system python lacks numpy)
- [OpenCV calib3d](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) — `calibrateCamera` flag surface (`CALIB_RATIONAL_MODEL`, `CALIB_FIX_K*`, …), `cv2.fisheye.calibrate` namespace (HIGH for API shape; MEDIUM for `*Extended` per-view-error/std-dev bindings — smoke-test at implementation)
- [Visibility-aware Fisher Information optimization (Sci. Reports, 2026)](https://www.nature.com/articles/s41598-026-55054-2) — current practice of FIM→CRLB achievable-accuracy prediction with marginalization (MEDIUM — adjacent domain, confirms methodology)
- sklearn `GroupKFold` / `LeaveOneGroupOut` / `cross_validate` — the grouped-CV API shape the experiment runners mirror (HIGH, stable)
- Immerkaer, "Fast Noise Variance Estimation" (1996) — single-frame noise σ fallback (MEDIUM, from training data; trivially verified at implementation)
- Zhang, "A Flexible New Technique for Camera Calibration" (2000) — homography initialization, the basis of OpenCV's init that the hybrid solver exploits (HIGH, classic)

---
*Architecture research for: camera intrinsic calibration & evaluation framework (offline Python pipeline)*
*Researched: 2026-10-15*
