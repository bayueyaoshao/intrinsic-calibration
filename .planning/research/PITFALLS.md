# Pitfalls Research

**Domain:** 相机内参标定与评估框架 (evaluation-driven camera intrinsic calibration & precision evaluation)
**Researched:** 2026-10-15
**Confidence:** HIGH (OpenCV API facts verified against official 4.x docs; statistical/Fisher claims grounded in standard estimation theory literature and cross-checked across multiple sources; corner-detection engineering claims from detector papers + community issue threads — individually flagged below)

Pipeline stages referenced throughout use the project's module names: **io → noise → corners → models → solve → evaluation → analysis → report**.

---

## Critical Pitfalls

### Pitfall 1: cornerSubPix window larger than the checker square

**What goes wrong:**
With squares of only 15–25 px, a `cornerSubPix` search window following the classic tutorial default of `winSize=(11,11)` (a 23×23 px region) routinely crosses into the neighboring black/white cell. The "subpixel refinement" then converges toward the wrong gradient evidence — corners are pulled toward square centers or neighboring corners, producing a systematic (not random) localization bias of up to fractions of a pixel. Because the bias is nearly identical for all corners, it does **not** average out in calibration: it silently distorts fx/fy/cx/cy and especially the distortion coefficients. This is the single most likely precision bottleneck for this dataset and it is invisible in reprojection error, because the optimizer fits the (biased) corners consistently.

**Why it happens:**
Tutorials and most blog posts use `winSize=(11,11)` on large boards (squares ≥ 50 px). Developers copy the call without rescaling to square size. Also, `winSize` in OpenCV is a **half-window**: the search region is `(2*winSize.x+1) × (2*winSize.y+1)` — people who compute "window = 11 px" and pass 11 actually get 23 px.

**How to avoid:**
- Rule of thumb: `winSize ≈ floor(squareSize_px / 2) − 1`, capped so the total window `(2w+1)` stays strictly inside one square (i.e. `2w+1 < squareSize_px`). For 15–25 px squares that means winSize in the 3–7 range.
- Make winSize a **candidate in the corner-extraction optimization experiment** (the project's planned synthetic-degradation study), not a fixed constant. Compare localization error on synthetic images with known ground-truth corners per winSize; report the curve in the report stage.
- Prefer `findChessboardCornersSB` (with `CALIB_CB_ACCURACY`) as the primary candidate: it performs its own localized subpixel refinement and does **not** need a follow-up `cornerSubPix` at all — running `cornerSubPix` after SB is a documented anti-pattern that can move corners off their SB-optimal positions.
- `zeroZone=(-1,-1)` (auto) is usually right; a dead zone larger than ~half the square suppresses the true gradient center.

**Warning signs:**
- Corner positions quantized/stair-stepped as winSize crosses the square boundary.
- Repeatability across the 8-burst looks great (sub-0.05 px) but held-out reprojection error plateaus high — a signature of correlated systematic bias, not noise.
- Refined corners visually drift off the visual corner in zoomed overlays (the report's corner-overlay figure exists precisely to catch this).
- Reprojection residuals show a strong low-order radial pattern mirroring the square grid rather than lens radial direction.

**Phase to address:** corners (candidate search); verified visually in report.

---

### Pitfall 2: Treating 8 near-identical burst shots as 8 independent poses

**What goes wrong:**
The obvious-but-fatal statistical error: feeding all 640 images to `calibrateCamera` as if they were 640 observations, or splitting train/val by image instead of by pose. The 8 burst shots share the same board pose, so all pose information is duplicated 8×. Consequences:
- Standard errors from the optimizer are understated by up to ~√8 ≈ 2.8× (false precision).
- Image-level CV split leaks: validation images are near-copies of training images → model selection sees artificially low held-out error and overfit models (rational k4–k6, KB4) win.
- BIC/AIC computed with n=640 instead of the effective n=80 poses → complexity penalty ~8× too weak → over-parameterized models selected.
- The CRLB "anchor" comparison becomes meaningless because the empirical parameter uncertainty was computed under a false independence assumption.

**Why it happens:**
The data contract (`input_data/train/` = 80 poses × 8 shots) looks like ordinary "more data is better." Nothing in OpenCV stops you; `calibrateCamera` happily accepts 640 views. The leakage is invisible to training reprojection error, which always decreases.

**How to avoid:**
- Enforce pose-grouped splitting **structurally in io/evaluation**: the dataset object stores pose IDs, and every splitter (K-fold, holdout, leave-one-out, bootstrap) operates on the set of 80 pose IDs, never on image indices. Unit-test: assert no pose appears in both train and val.
- For BIC/information criteria, use n = number of independent pose units (80), and residual DOF consistent with that.
- Decide per-experiment whether burst images are (a) averaged into one super-corner observation per pose, (b) stacked as repeated observations of one pose with a correlated-noise covariance, or (c) used purely for noise estimation. The planned repeat-index slice experiment is the arbiter — do not fix the choice before running it.

**Warning signs:**
- `stdDeviationsIntrinsics` values 2–3× smaller than the spread observed across resampling experiments (the evaluation stage cross-checks exactly this).
- Held-out error much lower than test-set (65 independent images) error — classic leak signature.
- K-fold CV variance across folds suspiciously tiny.
- Cross-validation curve monotonically favors the most complex model with no minimum.

**Phase to address:** io (data model with pose IDs); evaluation (splitters); noise (repeats usage).

---

### Pitfall 3: Model selection on training reprojection error

**What goes wrong:**
Choosing among M1 (k1k2p1p2) / M2 (k1k2k3p1p2) / M3 (rational k1–k6p1p2) / M4 (KB4 fisheye) by whichever minimizes training reprojection error. Training error decreases monotonically with model complexity by construction, so M3/M4 always "win" — including when they are fitting corner-detection noise. The delivered intrinsics then carry extra DOF with inflated variance and degraded generalization, which the project's own three-criteria optimality definition would reject. This pitfall is especially corrosive here because the whole product is a *proof* of optimality; a selection procedure that provably favors overfitting invalidates the central claim.

**Why it happens:**
`calibrateCamera(...).reprojError` is the number sitting right in the return value, and lower looks better. Also, classic calibration literature reports only training RMS, so copying convention reproduces the mistake.

**How to avoid:**
- Selection metric = held-out reprojection error via pose-grouped K-fold CV (primary), with BIC (n=80 effective) and parameter identifiability (condition number of the reduced FIM, relative SEs) as secondary evidence — exactly as planned in PROJECT.md.
- Report all four models' train vs held-out error side by side so the generalization gap is itself evidence in the report.
- Final delivery: re-fit the selected model on all data (CV only selects; full-data refit delivers), per the project's stated convention.

**Warning signs:**
- Held-out error U-shaped or flat while train error monotonically decreases → overfit region visible.
- M3/M4 distortion coefficients with relative SE > 50% (or unbounded/floating to huge magnitudes like |k4| ≫ |k1|).
- k3/k4–k6 estimates flip sign or magnitude across folds while held-out error barely changes → fitting noise.

**Phase to address:** models (candidate definitions); solve (fit API must expose per-model config); evaluation (CV harness).

---

### Pitfall 4: Extrapolating distortion outside corner coverage

**What goes wrong:**
The board occupies mostly the central region; corners may be absent in the high-distortion periphery (2560×1440 wide-angle). Radial polynomial distortion is only *constrained* where corners exist. Predicting pixel positions (or undistorting points) in the periphery **extrapolates** the polynomial, and high-order terms (k3, k4–k6, KB's θ⁶/θ⁸) dominate exactly there. Undistorted outputs can bend backward (non-monotonic r→θ mapping), produce NaN/Inf, or silently invent image content. Any downstream consumer — and any honest "error budget" — must distinguish interpolation from extrapolation regimes.

**Why it happens:**
Polynomials look globally defined; nothing warns you. The reprojection error on the *covered* region can be excellent while periphery predictions are garbage. Wide-angle + small board is the textbook configuration for this failure.

**How to avoid:**
- io pre-check: compute per-image corner coverage footprint (convex hull of corners normalized by image extent); flag images with coverage radius below a threshold; report aggregate coverage as a coverage map figure.
- Fisher/CRLB analysis (noise stage) makes this quantitative *before* solving: the FIM for k3/k4–k6 given the actual corner distribution reveals near-singularity — this is precisely the "fx–cy coupling"-style degeneracy detection the project mandates.
- In report: always overlay the coverage footprint on residual/undistortion figures; never present periphery predictions without the footprint.
- If periphery matters: the answer is new data (more poses placing the board at edges), not a bigger polynomial. Say so in the error budget.

**Warning signs:**
- Undistortion of full-image grid produces folds (r mapping non-monotonic) or NaN at large radius.
- k3 (or k4–k6) relative SE huge; FIM smallest eigenvalue near zero with eigenvector loading on high-order radial terms.
- Held-out error on test images whose boards sit closer to edges is markedly worse.

**Phase to address:** io (coverage pre-check); noise/FIM (observability); analysis + report (coverage figures, error budget).

---

### Pitfall 5: CRLB comparison run with violated assumptions → "error beats the bound"

**What goes wrong:**
The project's headline criterion is "empirical error approaches CRLB." The CRLB is only valid for (a) an **unbiased** estimator, (b) the assumed noise model (here: corner localization noise with the covariance used in the FIM), (c) the same parameterization. Violations are routine:
- Distortion model truncation is a *bias*: the true lens has infinite-order radial distortion; a fitted M1/M2 has systematic residual, so its error is not bound by the CRLB computed under the assumed model — empirical RMSE can legitimately fall **below** the computed bound, which in a report reads as "we beat information theory," i.e. proof the comparison is broken.
- Heavy-tailed or outlier corner noise (JPEG artifacts, occasional corner misses) inflates FIM mismatch.
- Using a scalar corner σ measured on flat synthetic patches while real corner noise is anisotropic and correlated along the square edge direction.

**Why it happens:**
CRLB is quoted as a magic number; the fine print (unbiasedness, correct noise covariance, matched model) is treated as decoration. The bias-vs-variance decomposition is skipped.

**How to avoid:**
- Frame the CRLB as **model-conditional**: "CRLB of estimating θ under model M given noise covariance Σ." State it in the report next to each bound.
- Empirically verify the noise model in the noise stage using the 8-bursts: per-corner mean/variance, edge-direction vs cross-edge variance, temporal correlation. Feed the *measured* Σ (possibly diagonal or shrunk — see Pitfall 8) into the FIM, not an assumed scalar.
- Decompose empirical error into bias² + variance (bias estimable via the repeat-slice and test-set differences); compare the **variance** part to CRLB, which is what the bound actually governs.
- Treat "empirical < CRLB" as a hard red flag in the evaluation stage's sanity checks, never as a triumph.

**Warning signs:**
- RMSE below CRLB on any parameter → stop and audit assumptions.
- Bias term dominates variance for simple models → model truncation, move up model complexity (via CV, not by default).
- Repeat-slice parameter spread >> CRLB prediction → covariance mismatch (often: correlations ignored).

**Phase to address:** noise (Σ measurement); models (bias framing); evaluation (bound comparison + sanity flags); analysis (bias/variance decomposition).

---

### Pitfall 6: Fisher information computed with extrinsics treated as known

**What goes wrong:**
A naive FIM for intrinsics that ignores the 6·N pose (extrinsic) DOF, or that inverts only the intrinsic block of the full FIM, produces bounds wildly off. Extrinsics are unknown nuisance parameters jointly estimated with intrinsics; their uncertainty inflates intrinsic uncertainty substantially (e.g. fronto-parallel-heavy pose sets couple fx with tz). The CRLB "anchor" then overstates achievable precision by large factors, and the project's "distance to data limit" metric is fiction.

**Why it happens:**
Textbook FIM derivations for intrinsics often fix the pose for clarity; code follows the formula. The correct treatment — profiling or marginalizing nuisance parameters — is a Schur-complement away but easy to skip.

**How to avoid:**
- Build the **full** FIM over (intrinsics ⊕ all extrinsics) using the shared projection/Jacobian implementation (the architectural constraint already in PROJECT.md — it exists precisely so the FIM and solver use identical math).
- Marginalize extrinsics: `F_intrinsic = F_θθ − F_θβ · F_ββ⁻¹ · F_βθ` (Schur complement), or equivalently read intrinsic uncertainty from the full inverse covariance. Profiling (equivalent here for Gaussian noise) is also acceptable; document which was used.
- Regularize: if `F_ββ` is singular (degenerate pose), detect it via smallest-eigenvalue analysis and report the degenerate direction (this is the planned degeneracy detection — fx–cy coupling etc.), rather than emitting garbage bounds.
- Unit test: FIM-derived parameter SEs vs `stdDeviationsIntrinsics` from `calibrateCamera` on the same data must agree to a few percent when the same model/covariance is used. This cross-check is the single strongest validator that models/noise/evaluation truly share math.

**Warning signs:**
- CRLB SEs several times tighter than resampling spread → nuisance handling wrong or Σ optimistic.
- FIM inversion unstable / condition number astronomic → pose degeneracy, not a code bug per se.
- FIM-based SEs disagree with `stdDeviationsIntrinsics` → the "shared implementation" constraint is being violated somewhere.

**Phase to address:** noise (FIM construction); models (shared Jacobians); evaluation (cross-check vs solver SEs).

---

### Pitfall 7: fisheye (KB4) applied to a lens that is not actually fisheye — and vice versa

**What goes wrong:**
Two symmetric mistakes:
- Running `cv2.fisheye::calibrate` (equidistant KB4) on a ordinary wide-angle lens: the equidistant projection assumption is violated; θ–r mapping misfit shows up as huge k1–k4 compensation terms, poor convergence, or `CALIB_CHECK_COND` errors; results are silently bad if the check is off.
- Trusting Brown k1–k3 on a true wide-angle/KB lens: polynomial radial terms cannot represent near-θ-linear projections at FOV approaching/above ~120–140°; residuals explode at the periphery while the center fits, or distortion terms blow up compensating each other.

Also specific to OpenCV's fisheye API: it fits **fx, fy, cx, cy, k1–k4 with skew fixed** by default (`CALIB_FIX_SKEW` semantics — α=0 in u = fx(x'+αy')+cx), so silently assuming skew is estimated is wrong.

**Why it happens:**
"广角" (wide-angle) and "鱼眼" (fisheye) get conflated. Barrel distortion is *not* fisheye projection. The correct model is an empirical question the project is explicitly built to answer via CV — the pitfall is prejudging it, or misconfiguring the KB run.

**How to avoid:**
- Let M1–M4 compete on held-out error with identical corner data — that is the framework's job; do not pre-select by lens folklore.
- For KB4 runs: feed float32 arrays with shape (N,1,3)/(N,1,2) (see Integration Gotchas); check per-view balance (the API demands all views see the board well — very asymmetric coverage triggers bad extrinsics that poison everything); consider `CALIB_CHECK_COND` on and interpret its errors as "this view's distortion is unobservable," not as crashes to bypass.
- Report θ FOV implied by the fitted KB model and the max radius where corners constrain it; state interpolation vs extrapolation region (ties to Pitfall 4).
- Sanity-constrain model comparisons: same data, same splits, same convergence criteria, and check each model actually converged (status/termCriteria), so a non-converged fit doesn't masquerade as a model loser/winner.

**Warning signs:**
- KB4 k-coefficients orders of magnitude above Brown's, or signs alternating wildly across folds.
- Fisheye run silently returns with most extrinsics at init values (focal stuck at default max(w,h)/π) → it never actually optimized.
- Brown M3 residuals strongly azimuthal (not purely radial) → projection-family mismatch, favoring KB.

**Phase to address:** models (KB4 candidate config); solve (per-model solver wrappers); evaluation (fair-comparison harness).

---

### Pitfall 8: Noise covariance estimated from 8 samples and used as if accurate

**What goes wrong:**
The FIM needs corner-noise covariance Σ. The 8-bursts give only **8 samples per corner**; the sample standard deviation then carries a relative standard error of ≈ 1/√(2(n−1)) = 1/√14 ≈ **27%** per corner, far worse for a full covariance (which has O(p²) entries — with p=2 coordinates per corner plus cross-correlations, 8 samples cannot fill a dense covariance at all; it is rank-deficient). Feeding a noisy or rank-deficient Σ into the FIM propagates ~27%+ error into every CRLB number, silently. Worse: Σ estimated from corners includes *detection noise and board/label jitter* — if the board itself micro-rotated between burst shots, "noise" includes pose jitter misattributed as corner noise, inflating Σ and hence CRLBs.

**Why it happens:**
Estimating covariance needs ~10× more samples than estimating a mean; 8 feels like enough because per-corner scatter plots look plausible. The pose-jitter contamination is conceptually subtle: same-pose ≠ perfectly static board.

**How to avoid:**
- Default to **diagonal or block-sparse Σ** (per-corner σx, σy, optionally per-edge correlation), never dense per-corner covariance from 8 samples. Optionally apply shrinkage toward a structured target (diagonal, common-σ).
- Propagate uncertainty honestly: since CRLBs inherit Σ's ~27% error, report the bound as "CRLB ± (Σ uncertainty)" or run the FIM with Σ scaled by 0.8/1.25 to show bound sensitivity. This costs one line in the report and converts a false-precision disaster into stated uncertainty.
- Detect pose jitter: per-burst, fit pose-only (intrinsics fixed from a pooled fit) and check residual board-motion magnitude; if board motion is non-negligible relative to corner σ, either compensate (re-fit pose per burst) or document the inflation.
- The repeat-slice experiment design (slice by repeat index, poses constant) is exactly the right control — keep pose fixed within each slice-set and verify empirically that slice-to-slice parameter spread matches FIM prediction *given the Σ actually used*.

**Warning signs:**
- Per-corner σ from burst A differs from burst B by >2× (sampling noise or non-stationarity).
- Dense Σ matrix with 8 samples → singular; any multivariate Gaussian math on it produces garbage silently via pseudo-inverse.
- CRLB shifts materially when Σ rescaled → bound is Σ-dominated; report must say so.

**Phase to address:** noise (Σ estimation strategy + jitter audit); evaluation (bound sensitivity reporting).

---

### Pitfall 9: Robust loss left on for the final Gaussian-noise fit

**What goes wrong:**
Using Huber/Cauchy/IRLS throughout "for robustness" and reporting the result as the accuracy answer. Robust M-estimators trade efficiency for immunity to outliers: under (approximately) Gaussian corner noise — which the burst data should verify — a Huber fit has **higher** parameter variance than plain least squares, so it sits measurably further from the CRLB. The evaluation then wrongly concludes "data limit not reached" when the gap is self-inflicted. Conversely, switching to robust loss because residuals look heavy-tailed, without diagnosing *why* (usually: a few bad corners or a bad view), masks exactly the problems the evaluation stage exists to find.

**Why it happens:**
"Robust is safer" intuition; robust loss also lowers the *training* error, which (per Pitfall 3) people still read as improvement. The bias-variance cost of robustness is not widely internalized.

**How to avoid:**
- Two-phase protocol in solve: (1) robust pass (Huber or Cauchy IRLS) purely as an *outlier detector*; (2) final pass with plain least squares on inliers, which is the reported/evaluated fit. Log which views/corners were down-weighted — that list is diagnostic gold for the error budget.
- Verify noise Gaussianity from bursts (kurtosis per corner, Q-Q plot in report) so the efficiency argument for LS is data-backed, not assumed.
- If heavy tails are real and persistent, then robust loss is justified *for delivery*, but the CRLB comparison must then use the appropriate bound or be explicitly qualified — do not compare a robust estimator to a Gaussian CRLB and call the gap "room to improve."

**Warning signs:**
- Robust final fit with reprojection error slightly *lower* but held-out error *higher* than LS-on-inliers → efficiency loss at work.
- Few or zero outliers flagged, yet robust chosen → pure efficiency loss.
- Residual kurtosis >> 3 → investigate corners/views first (evaluation's leave-one-pose ×80 will name the culprits).

**Phase to address:** solve (two-pass strategy as default); evaluation (fair CRLB comparison, LOO diagnostics).

---

### Pitfall 10: Corner index/order flips across the 8-burst poison noise statistics

**What goes wrong:**
Checkerboard corner lists are only defined up to the lattice's symmetries: classic `findChessboardCorners` and `findChessboardCornersSB` can return corners with different orientation — a 90°/180° rotation of the ordering, or row-major vs SB's different traversal (documented behavior differences; SB can even return a different corner *count* convention in rare cases). For calibration itself a consistent per-image flip is benign (pose absorbs it). But the noise stage aggregates corners **by index across the 8 same-pose shots**: if shot 3's corner list is rotated 180° relative to shot 1, "corner (i) across repeats" compares *different physical corners* → per-corner σ becomes huge and spatially structured (mirrored patterns), the measured Σ is garbage, and the CRLB anchor is wrong. Same failure hits any per-corner temporal analysis.

**Why it happens:**
The API returns an (N,1,2) array with no orientation guarantee documented as stable across images or detector versions; blends of detectors across candidates (classic vs SB) almost guarantee mixed conventions. It fails silently — arrays still have the right shape and findChessboardCorners returns True.

**How to avoid:**
- Canonicalize ordering in corners stage: for each image, compute the corner-set orientation signature (e.g. vector from first to last point, or grid axis directions) and rotate/flip all detections into a canonical frame **based on content** (e.g. the asymmetric cue: which grid axis is longer is useless here — 11×8 is asymmetric by count, so use axis lengths; 11-axis vs 8-axis disambiguates 90°; remaining 180° fix by projecting through a pooled initial pose and minimizing index-wise residual).
- Assert consistency: within each pose's 8 shots, index-wise distances between shot-mean corners and each shot's corners must all be sub-pixel; a shot violating this by ~half the board is an ordering flip — auto-correct or quarantine it.
- Never aggregate across *different detectors* by index; each detector candidate gets its own aggregation.

**Warning signs:**
- Per-corner σ shows point-symmetric "hot corners" pattern (mirrored clusters) — the flip signature.
- One or two shots per pose contribute enormous index-wise residuals while their reprojection overlays look perfectly fine visually.
- Noise σ differs implausibly (5–10×) between detector candidates on identical images.

**Phase to address:** corners (canonicalization + consistency assert); noise (guards against poisoned aggregation); report (quarantined-shot listing).

---

### Pitfall 11: Board flatness and imaging systematic errors mistaken for lens properties

**What goes wrong:**
Calibration assumes a strictly planar board. A printed/A-mounted board with even 0.5–1 mm bow over 275×200 mm injects fake "distortion" that the optimizer happily absorbs into k-terms. Similarly: JPEG compression (artifacts bias corner centroids by a few hundredths of a pixel, direction-structured), focus/defocus blur varying across frame (asymmetric corner bias), rolling-shutter-like effects if any board/hand motion during burst exposure, and vignetting changing effective corner contrast. All of these are *systematic* — they do not average out over 640 images; they become part of the delivered "intrinsics" and cap the achievable accuracy below what any Fisher analysis of random noise predicts. The evaluation will then report a persistent gap to CRLB that no algorithmic improvement can close.

**Why it happens:**
Planarity is an untestable-from-data-alone assumption; teams verify focus and exposure but never the board. With data already collected (input_data is read-only), these errors are baked in — the framework can only *bound and disclose* them.

**How to avoid:**
- Estimate the planarity ceiling in the analysis stage: fit per-view residual patterns; a low-order smooth non-radial residual field common across views is the board-bow signature. Estimate the equivalent Z-surface and report "accuracy floor attributable to non-planarity ≈ X."
- Compare repeat-burst corner means (temporal-random) against cross-pose parameter spread (total): the gap is systematic — quantify it in the error budget rather than hiding it.
- Characterize JPEG/blur bias once via the planned synthetic-degradation experiment (apply known JPEG quality/blur to synthetic renders, measure corner bias curve) — this converts an unknown systematic into a stated bias term.
- Since input_data is immutable, frame these as **error-budget line items** with detection evidence, per the project's "quantify distance to data limit" mandate. This is a feature of the framework, not a failure — as long as it is measured and disclosed.

**Warning signs:**
- Residuals that reproduce the same spatial pattern (in board coordinates) across many views → board defect, not lens.
- Accuracy plateaus well above CRLB with no algorithmic lever moving it.
- Per-corner residuals correlated with local blur intensity or JPEG block grid.

**Phase to address:** io (exposure/blur pre-checks); corners (synthetic degradation study quantifying detector bias); analysis (board-bow estimation, systematic vs random split); report (error budget table).

---

### Pitfall 12: Bad initialization and parameter scaling stall or silently mislead the optimizer

**What goes wrong:**
Two coupled numerical failures:
- Initialization: `calibrateCamera`'s built-in homography-based init is decent, but custom pipelines (and the project's own solver must exist for shared Jacobians) often init distortion at zero — fine — but init focal poorly (e.g. default fisheye max(w,h)/π ≈ 1128 px here may be far from truth for a 2560 px wide FOV sensor), landing in a local minimum where the fit looks converged with poor parameters. Fisheye especially: with a too-small initial focal, KB runs get stuck or trigger `CALIB_CHECK_COND` failures.
- Scaling: fx~2000–4000 (pixels), cx~1280, k1~−0.1, k3~0.01, KB k4~0.001 — a gradient/Jacobian-based solver over raw parameters is badly scaled; convergence is slow or the trust region misbehaves, and termination criteria trip on the large-magnitude parameters while small ones are unconverged.

**Why it happens:**
Termination criteria (ε or count) hide partial convergence; residual curves look flat enough; nothing fails loudly. The fisheye default focal is a silent constant chosen for robustness, not accuracy.

**How to avoid:**
- Multi-start policy in solve: init focal from a rough FOV assumption sweep (e.g. FOV ∈ {100°, 120°, 140°} → focal = W/(2·tan(FOV/2))), pick start by training error, verify the winner from ≥2 distinct starts agrees.
- Use `scipy.optimize.least_squares(..., x_scale='jac')` (or equivalent scaling) in the custom solver; for OpenCV paths, parameters are pre-scaled internally, but keep TermCriteria explicit (`(COUNT+EPS, 200, eps)`) rather than defaults, and record the returned convergence status.
- After every solve, confirm: gradient norm small *jointly* (not per-parameter block), parameters stable under perturbed init, and per-view extrinsics plausible (board in front, sane tilt).

**Warning signs:**
- Same data + different init → materially different intrinsics (beyond SEs) → local minima.
- Solver converges but Jacobian column norms differ by 6+ orders of magnitude → scaling problem.
- Fisheye results identical across datasets except cx/cy → focal stuck at init.

**Phase to address:** solve (init strategy, scaling, convergence checks); evaluation (multi-start verification as part of harness).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoding winSize / detector params in call sites | Quick first calibration | Corner-optimization experiment can't sweep them; results unreproducible across configs | Never — config-driven from day one (project already mandates `config/`) |
| Splitting CV by image index instead of pose ID | 5-line splitter | Leaked validation invalidates every downstream selection claim (Pitfall 2) | Never |
| Using `calibrateCamera` return RMS as "the error" | Zero extra code | Selection and reporting both lie; framework's core value destroyed (Pitfall 3) | Only as a logged diagnostic alongside held-out metrics |
| Scalar σ assumed for all corners in FIM | FIM in 10 lines | CRLB off by 2–5× in structured ways; "distance to limit" metric meaningless | For early smoke tests only; replace before any report claim |
| Copying projections/Jacobians into noise & evaluation modules instead of sharing | Feels faster than refactoring | FIM and solver silently diverge after any model tweak; every CRLB claim void (violates explicit architectural constraint) | Never — this constraint is load-bearing for the product's claims |
| Skipping per-view coverage/quality logging at ingest | Ingest runs faster | Degenerate views discovered only after confusing solver failures; reruns needed | Never — ingest pre-checks are cheap |
| Storing intermediate results as ad-hoc dicts instead of typed artifacts | Rapid prototyping | Slice/bootstrap/LOO experiments can't be joined reliably; report stage drowns in plumbing | Spike code only; refactor before the evaluation stage |

## Integration Gotchas

OpenCV API traps verified against official 4.x documentation and issue tracker.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `cv2.calibrateCamera` | Passing `patternSize` as (rows, cols); passing imageSize as (h, w) | `patternSize = (points_per_row, points_per_col)` = **(11, 8)** for this board; `imageSize = gray.shape[::-1]` = (2560, 1440) |
| `cv2.calibrateCamera` | Assuming `perViewErrors` are mean errors | They are **per-view RMS** reprojection errors (documented); aggregate conventions differ from naive averaging of means — compute your own from residuals for full control |
| `cv2.calibrateCamera` flags | Enabling `CALIB_RATIONAL_MODEL` and forgetting it changes k-term semantics; stacking `CALIB_THIN_PRISM_MODEL`/`CALIB_TILTED_MODEL` into a "more is better" soup | Each flag adds parameters that must be selected/justified by CV; map M1–M3 to explicit flag sets, documented per model in `config/` |
| `cv2.fisheye::calibrate` | Passing float64 (N,3) or int arrays → assertion `objectPoints.type() == CV_32FC3 || CV_64FC3` fails, and (N,1,3) shape required | Cast to float32/float64 and reshape to (N,1,3) / (N,1,2); same for `fisheye::distortPoints` (which takes *undistorted normalized* points and needs an explicit K) |
| `cv2.fisheye::calibrate` | Not setting `K`/`D` input with `CALIB_USE_INTRINSIC_GUESS`, focal silently initialized to max(w,h)/π | Provide a FOV-derived init and verify the returned focal is not stuck at 1128.4 (= 2560/π) — the classic stuck-init telltale |
| `cv2.fisheye::calibrate` | Views with very different board coverage → singular Hessian per view | Use `CALIB_RECOMPUTE_EXTRINSIC` for better convergence on hard poses; treat `CALIB_CHECK_COND` errors as "this view's distortion unobservable," drop/replace that view — do not blanket-disable the check |
| `findChessboardCorners` + `cornerSubPix` | Copy-pasting `winSize=(11,11)`; forgetting winSize is a half-window | Scale to square size (Pitfall 1); total window `(2w+1) < square_px` |
| `findChessboardCornersSB` | Running `cornerSubPix` afterwards "for extra precision" | SB with `CALIB_CB_ACCURACY` already does localized subpixel refinement; post-refinement can degrade it. Also: SB's corner ordering convention differs from the classic detector (GitHub issue #22083) — never index-mix across detectors (Pitfall 10) |
| `cv2.undistort` / `undistortPoints` | Assuming valid output everywhere | Output outside the corner-coverage footprint is extrapolation (Pitfall 4); check for NaN/folds, restrict reported maps to interpolated region |
| `cv2.getOptimalNewCameraMatrix` | Using it in evaluation pipelines | It *changes* intrinsics (alpha blending); an evaluation framework must report under the *calibrated* matrix — keep it out of metric paths |
| OpenCV version drift | Results differ across machines | Pin OpenCV version in project config; detector behaviors (esp. SB) have changed across 4.x releases; log `cv2.__version__` into every report artifact |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Re-detecting corners on every experiment run | Each CV/slice/bootstrap rerun takes hours; 640 images × repeated detection | Detect once per detector candidate, cache with (image hash, detector, params) key; experiments operate on cached corners | Immediately at evaluation stage (dozens of refits × 640 images) |
| Bootstrap/repeat-slice refits in pure Python loops | Each refit ~seconds–minutes → total runs × N explodes | Vectorize residual computation with numpy (all views batched); refit is a small dense NLLS — keep it in scipy; parallelize folds/slices with multiprocessing | At ×8 slices × K folds × 4 models × bootstrap N — easily 1000+ solves |
| Full-rank covariance ops in FIM for 640×2×8 observations | O(n³) inversion of giant matrices, memory spikes | Exploit block structure: Schur complement on extrinsic block (Pitfall 6); per-pose blocks are only 6×6 | As soon as all views' FIM contributions are stacked naively |
| Dense per-corner covariance from bursts | Singular matrices, pseudo-inverse silently used | Structured/diagonal Σ (Pitfall 8) | Immediately at noise stage |
| Rendering overlay figures for all 640 images in report | Report generation takes 10+ minutes, multi-GB artifacts | Figure selection policy: all views for summary stats, overlays for worst-K views by error + one random sample; render at target DPI | At report stage |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Writing report/intermediate artifacts with predictable paths into shared locations | Overwrite/clobber of user files by batch runs | All writes under `output/` per directory contract; run IDs subdirectories; never write outside project root |
| Pickle-loading cached corners/artifacts from `output/` | Arbitrary code execution if cache dir is shared/tampered | Cache with safer formats (npz/json + explicit schema); document trust boundary |
| `cv2.setNumThreads(0)` / global state leaks between library calls in one process | Wrong results when library API used in user notebooks alongside their own OpenCV code | Document and scope thread/global settings inside CLI entry only; library functions leave global state untouched |

(Offline, single-user research tool — no network attack surface; the above are the realistic local-safety issues.)

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| One mega-run CLI with no stage entry points | A single failed stage (e.g. one quarantined pose) forces a full multi-hour rerun | CLI supports `--from <stage>` / `--until <stage>` with cached stage artifacts (matches "各阶段可独立调用" requirement) |
| Report dumps every number with no verdicts | Engineer can't see which of the three optimality criteria passed | Every report section opens with a verdict line (e.g. "CRLB gap: 1.18× — PASS"); evidence follows |
| No machine-readable summary of the run | Results can't be compared across experiments/detector candidates | Emit `summary.json` per run alongside HTML/PDF; report stage is a *renderer*, per the visualization/report split already decided |
| Silent quarantining of failed images/poses | User's effective dataset differs from `input_data` without their knowledge | Quarantine is always logged and reported with reasons and counts; exit code reflects quarantines |

## "Looks Done But Isn't" Checklist

- [ ] **Corner optimization:** Often missing the winSize sweep *including detector candidates* on synthetic degraded data — verify the report shows localization error vs parameters, not just a chosen config.
- [ ] **Noise measurement:** Often missing ordering-canonicalization assert per pose — verify per-corner σ maps look spatially smooth, not point-symmetric (Pitfall 10).
- [ ] **CRLB comparison:** Often missing the bias/variance split and Σ-uncertainty propagation — verify report states bound conditions and shows sensitivity to Σ scaling (Pitfalls 5, 8).
- [ ] **Fisher analysis:** Often missing extrinsic marginalization cross-check vs `stdDeviationsIntrinsics` — verify agreement to a few percent on identical model/Σ (Pitfall 6).
- [ ] **Model selection:** Often missing train-vs-held-out gap table for all four models — verify M1–M4 appear side by side with per-fold spread (Pitfall 3).
- [ ] **Final delivery refit:** Often missing the "CV selects, full data refits" step — verify delivered intrinsics come from an all-data fit of the CV-selected model, and that test/65 evaluation uses it, not a fold model.
- [ ] **Evaluation harness:** Often missing multi-start verification — verify final parameters stable across ≥2 inits (Pitfall 12).
- [ ] **Coverage disclosure:** Often missing corner-coverage footprint on every spatial figure — verify coverage map exists and periphery claims are fenced (Pitfall 4).

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| cornerSubPix window bias (1) | LOW | Re-run corner stage with swept winSize from cached images; downstream stages re-run from cache |
| Image-level CV leak (2) | MEDIUM | Fix splitter; **re-run all selection experiments** — every prior selection verdict is void |
| Model chosen on train error (3) | LOW | Re-run selection with held-out metric from cached fits; delivery refit re-run |
| Distortion extrapolation claims (4) | LOW | Re-scope figures/statements to coverage footprint; if periphery needed → new data acquisition (HIGH, out of project scope) |
| Broken CRLB comparison (5, 6, 8) | MEDIUM | Fix Σ/FIM treatment; re-run noise + analysis stages; CRLB claims re-derived — selection verdicts unaffected (independent of bounds) |
| Ordering flips in noise stats (10) | LOW | Add canonicalization; recompute noise-stage artifacts from cached detections |
| Robust-loss efficiency loss (9) | LOW | Re-fit final model LS-on-inliers from cached corners; evaluation re-run |
| Local minimum from bad init (12) | LOW–MEDIUM | Multi-start re-solve; if accepted results change, downstream evaluation re-runs |
| Board bow / systematic cap (11) | HIGH (unavoidable) | Cannot fix post-hoc — quantify, disclose in error budget; new board + reshoot only option |

## Pitfall-to-Phase Mapping

Project stages: io / noise / corners / models / solve / evaluation / analysis / report.

| Pitfall | Prevention Stage | Verification |
|---------|------------------|--------------|
| 1. cornerSubPix window vs square size | corners (parameter sweep) | Synthetic localization-error curve in report; zoomed corner overlays pass visual check |
| 2. Bursts treated as independent poses | io (pose-ID data model) + evaluation (grouped splitters) | Unit test: zero pose overlap across folds; resampling spread ≈ solver SEs |
| 3. Selection on train error | evaluation (CV harness, held-out only) | Report shows held-out minimum distinct from train monotonicity |
| 4. Distortion extrapolation | io (coverage check) + noise (FIM degeneracy) + report (footprints) | Coverage map present; no figure claims outside footprint |
| 5. CRLB assumptions violated | noise (measured Σ) + evaluation (sanity flags) | No "empirical < CRLB" events pass silently; bias/variance split shown |
| 6. FIM ignores extrinsics | noise (Schur marginalization) + models (shared Jacobians) | FIM SEs match `stdDeviationsIntrinsics` within a few % |
| 7. KB4 vs Brown misapplication | models + solve (per-model wrappers) | Fair-comparison table; no stuck-init fisheye fits |
| 8. Σ from 8 samples | noise (structured Σ, shrinkage) + report (bound sensitivity) | Report states Σ-derived uncertainty on CRLBs |
| 9. Robust loss efficiency loss | solve (two-pass: detect then LS) | Outlier log exists; final fit is LS-on-inliers unless justified |
| 10. Corner order flips | corners (canonicalization + asserts) | Per-corner σ maps spatially smooth; quarantine log clean or justified |
| 11. Board bow / systematic errors | io (pre-checks) + analysis (systematic split) + report (error budget) | Error budget table separates random vs systematic with evidence |
| 12. Init/scaling convergence | solve (multi-start, x_scale, explicit criteria) | Multi-start agreement logged per final fit |

## Sources

- OpenCV 4.x official calibration & fisheye documentation (via web reader): camera calibration tutorial (docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) — cornerSubPix (11,11) default, winSize half-window semantics, "at least 10 test patterns", patternStarted/ordering, RMS meanings; fisheye module page (docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html) — KB4 distortion model θ_d = θ(1+k1θ²+k2θ⁴+k3θ⁶+k4θ⁸), projection u = fx(x'+αy')+cx, flag bit values (FIX_SKEW, CHECK_COND, RECOMPUTE_EXTRINSIC, FIX_K1–K4, USE_INTRINSIC_GUESS), TermCriteria(100, DBL_EPSILON), focal init max(w,h)/π — **HIGH confidence**
- OpenCV calibrateCamera API reference (docs.opencv.org/4.x/d9/d0c/group__calib3d.html): 11-output signature incl. stdDeviationsIntrinsics/Extrinsics and perViewErrors (per-view RMS); CALIB_RATIONAL_MODEL / THIN_PRISM / TILTED flag semantics; patternStarted orientation convention — **HIGH confidence**
- Duda & Frese, "Accurate Detection and Localization of Checkerboard Corners for Calibration," BMVC 2018 (bmvc2018.org/contents/papers/0508.pdf) — SB detector algorithm, localized Radon transform, ACCURACY flag subpixel refinement — **HIGH confidence**
- OpenCV GitHub issue #22083 / forum threads (forum.opencv.org t/22083, bugs 15712, 22288): SB vs classic detector corner-ordering and count differences — **MEDIUM confidence** (community-reported, consistent across threads)
- fisheye::calibrate type/shape assertion failures (forum.opencv.org t/8049; GitHub issues #9150, #20603): CV_32FC3/CV_64FC3 and (N,1,3) shape requirements — **MEDIUM-HIGH confidence** (official assert string + multiple corroborating threads)
- perViewErrors semantics & stdDeviationsIntrinsics interpretation (stackoverflow.com/q/43878684; calib.io knowledge base; forum.opencv.org t/15808) — **MEDIUM confidence** (multiple independent sources agree; consistent with official docs)
- Estimation theory: Cramér-Rao bound conditions (unbiasedness, noise model match), nuisance-parameter elimination via Schur complement / profiling (standard texts: Kay, *Fundamentals of Statistical Signal Processing*; Triggs et al., "Bundle Adjustment — A Modern Synthesis") — **HIGH confidence** as theory, applied claims **MEDIUM** (application specifics depend on this project's Σ)
- Standard-error of a standard deviation, rel. SE ≈ 1/√(2(n−1)) (standard normal-theory result) — **HIGH confidence**
- Robust M-estimator efficiency loss under Gaussian noise (Huber, *Robust Statistics*; standard robust-regression literature) — **HIGH confidence**
- Engineering practices (winSize scaling to square size, multi-start, two-pass robust strategy, caching corners): community calibration guides (calib.io knowledge base) + known best practice — **MEDIUM confidence**

---
*Pitfalls research for: evaluation-driven camera intrinsic calibration & precision evaluation framework*
*Researched: 2026-10-15*
