# Research Summary: intrinsic-calibration

**Domain:** Evaluation-driven camera intrinsic calibration & precision evaluation framework
**Researched:** 2026-10-15
**Overall confidence:** HIGH

## Executive Summary

The domain has a stable, mature core and an open differentiator lane. The core — chessboard corner
detection with subpixel refinement, Brown/rational/fisheye-KB4 distortion models, LM-style
reprojection-error minimization — is fully served by the mandated stack: Python 3.12 + OpenCV 4.x +
numpy/scipy. OpenCV alone natively covers all four candidate models (M1–M4); no external calibration
dependency is needed. Key stack decision: pin opencv-python <5.0 (5.0.0 shipped Aug 2026; calib3d
churn risk undermines auditable numerics).

The competitive landscape shows a clear gap this project fills: no mainstream tool delivers
noise-floor measurement, pre-calibration Fisher/CRLB prediction, systematic corner-extractor
optimization, pose-level cross-validated model selection, or parameter-level error budgets as
first-class outputs. mrcal is closest on uncertainty but takes a splined-model route without CRLB
prediction or selection closure.

Architecture canon for this domain is a stage DAG over persisted artifacts with a shared numeric
core: models/ owns all projection math and Jacobians; solve/ has two interchangeable backends
(OpenCVSolver + ScipySolver with residuals built only from models.project); evaluation experiments
are pure functions over a calibrator. This mirrors Kalibr/basalt/COLMAP and is the only structure
that makes "actual error approaches CRLB" falsifiable.

The pitfall profile is dominated by two dataset-specific traps: (1) cornerSubPix search windows
exceeding the 15–25 px square size — systematic localization bias invisible in reprojection error;
(2) treating 80-pose × 8-burst as 640 independent views — image-level splits, BIC counts, and
covariance interpretations all leak and inflate precision ~√8. Both are preventable structurally,
not by discipline.

## Key Findings

**Stack:** Python 3.12 + opencv-python 4.14.x (pinned <5.0) + numpy 2.5.x / scipy 1.18.x; typer CLI,
matplotlib, pandas, jinja2, joblib (~1.7k-solve experiment grid); analytic Jacobians with
numdifftools as test oracle only.
**Architecture:** Stage DAG over JSON+npz artifacts; models/ as single source of truth; dual
solvers behind one CalibrationResult; FIM extrinsics-marginalized via Schur complement.
**Critical pitfall:** subpixel window vs square size + burst-as-independent-views leakage, plus
selection-on-train-error and distortion extrapolation beyond corner coverage.
**Features:** all table stakes are LOW complexity on OpenCV; every differentiator is unclaimed
territory among competitors; complexity peaks are Fisher/CRLB machinery and the corner-optimization
synthetic testbed.

## Implications for Roadmap

Suggested phase structure (consolidating the dependency-driven build order in ARCHITECTURE.md):

1. **Foundations: _types + config + models/** — 4 model classes + registry with parity and
   numeric-Jacobian tests *before* any solving. Avoids duplicated projection math at birth.
2. **Ingest + corners + noise σ** — pose-ID data model, prechecks, detector protocol → corners.npz,
   temporal + Immerkaer noise floor. Anti-leakage structure lives here. Parallelizable against 1.
3. **Solve + Fisher/CRLB** — solver interface + both backends; FIM/CRLB + pose-degeneracy detection
   (needs σ from 2, linearization point from 3).
4. **Evaluation experiments** — pose-level K-fold CV, repeat-slice ×8, bootstrap, LOPO ×80 as pure
   functions, joblib-parallel; held-out-only model selection harness.
5. **Analysis + corner optimization** — error budget (actual vs CRLB, direction/radial decomposition),
   sensitivity ranking; corner synthetic-degradation experiment reuses evaluation runners (build after 4).
6. **Visualization + report + run-all CLI** — figure primitives accumulated per phase; report assembly.

**Phase ordering rationale:** models → solve → fisher (shared math, then linearization point, then
theory); evaluation before corner-optimization (the corner experiment is another calibrate_fn grid);
CLI stubs from phase 2 onward.

**Research flags for phases:**
- Phase 3: validate marginalization math against OpenCV `stdDeviationsIntrinsics` (built-in cross-check).
- Phase 5: corner-optimization experiment design (degradation grid, ground truth, metrics) is the
  least-standardized piece — focused design pass needed.
- Phases 1, 2, 4, 6: standard patterns, unlikely to need further research.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Mandated + domain standard; versions web-verified Oct 2026; exact locks deferred to first `uv lock` |
| Features | HIGH | Competitor cross-check (OpenCV/Kalibr/MATLAB/libCalib/mrcal), multi-source |
| Architecture | HIGH | Verified against Kalibr/aslam, basalt-headers, COLMAP, OpenCV/scipy docs; few points MEDIUM (flagged in file) |
| Pitfalls | HIGH | OpenCV facts from official 4.x docs; estimation theory standard; engineering claims individually flagged (some MEDIUM) |

## Gaps to Address

- Sandbox pip verification failed — resolve and smoke-test the opencv/numpy/scipy trio day one of phase 1.
- CRLB anchoring with measured structured Σ (8-sample bursts) — validate in phase 3.
- Corner-optimization experiment design — design pass at phase 5 start.
- OpenCV 5.x migration audit — deliberately deferred.

Sources:
- [NumPy news](https://numpy.org/news) — 2.5.0 transitional release, 2.5.4 (Oct 2026)
- [SciPy](https://scipy.org) — 1.18.1 (Aug 2026), SPEC 0 policy
- [opencv-python (PyPI)](https://pypi.org/project/opencv-python) — 5.0.0.93 wrapper
- [opencv-python releases (GitHub)](https://github.com/opencv/opencv-python/releases) — 4.14.0 (Jul 2026)
- [OpenCV releases](https://opencv.org/releases/) — OpenCV 5.0.0 (Aug 19, 2026)
