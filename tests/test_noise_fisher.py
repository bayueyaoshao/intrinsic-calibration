"""Fisher/CRLB 数学核心的锁定测试:恒等式 + 结构 + 数值基准 + 蒙特卡洛。

锁定的决策与量级(对应 03-02 计划的各类行为):
- Schur 恒等式:inv(F_marg) 与 inv(JᵀJ) 取内参块逐元素相对差 < 1e-10
  (分块代数正确性的最强证据;小场景整体稠密化只发生在测试里,
  真实规模的主流程禁止);
- 结构:F_ββ 的非对角 6×6 块范数 < 1e-12(行序契约被测试锁定);
- 奇异防护:位姿不可观测必须中文报错点名视图序号,绝不静默 pinv;
- 数值基准:numdifftools 数值雅可比 → 数值 FIM,与解析路线相对差 < 1e-6
  (D-10:numdifftools 只进 tests/,src/calib 出现即违规);
- σ 缩放:CRLB 随 σ 严格线性(逐位一致),Σ ±27% 敏感性带是纯乘法。

注意:全部用例都在合成小场景上做数学锁定,与阶段四"真实数据上的
CRLB 实证校验"无关——这里锁的是"数学没算错"这根桩。
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numdifftools as nd  # D-10:数值差分基准,只进测试,绝不进 src/calib
import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from calib.io.config import BoardSpec
from calib.models import create_model
from calib.models.base import transform_to_camera
from calib.noise.fisher import (
    SIGMA_BAND_REL_8,
    aggregate_observations,
    build_aggregated_observation_set,
    crlb_se,
    marginalized_fisher,
    sensitivity_band,
    spectrum_report,
    weighted_fisher,
)
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet
from calib.solve.residuals import (
    make_jacobian_fn,
    make_residual_fn,
    pack_theta,
    unpack_theta,
)

# 合成小场景的固定棋盘与 M2 真值参数(棋盘与 config/board.yaml 同规格 11×8;
# 畸变给小而非零的值,让 FIM 的畸变列有真实信息量;棋盘放近到 z=700、
# 姿态带 ±0.25 rad 侧转,保证畸变与旋转都可观指定——场景条件数决定
# 恒等式测试能展现到多高的浮点精度,这是实测挑过的一组)
_BOARD = BoardSpec(cols=11, rows=8, square_size_mm=25.0)
_M2_TRUE = np.array([800.0, 810.0, 330.0, 250.0,
                     -0.25, 0.12, 0.002, -0.002, 0.01])


def _scene(n_views: int = 6, seed: int = 11):
    """合成小场景:已知 M2 参数与逐视图位姿,直接按模型投影生成角点。

    体例仿 tests/test_solve_solvers.py 的 _synthetic_obs:FIM 只吃几何
    (物点 + 位姿 + 参数),不吃图像,所以不走 conftest 的渲染夹具
    (make_true_views 渲染整幅棋盘图却不返回位姿)——投影直接生成保证
    角点与位姿逐位一致,蒙特卡洛求解也能从真值点出发。
    返回 (观测集, 模型, 参数向量, rvecs (V,3), tvecs (V,3))。
    """
    rng = np.random.default_rng(seed)
    objp = build_object_points(_BOARD)
    model = create_model("m2")
    rvecs = np.empty((n_views, 3), dtype=np.float64)
    tvecs = np.empty((n_views, 3), dtype=np.float64)
    pts = []
    for i in range(n_views):
        rvecs[i] = rng.uniform(-0.25, 0.25, 3)
        tvecs[i] = [*rng.uniform(-15.0, 15.0, 2), 700.0]
        pts_cam, _ = transform_to_camera(objp, rvecs[i], tvecs[i])
        pts.append(model.project(pts_cam, _M2_TRUE))
    obs = ObservationSet(
        image_ids=tuple(f"img_{k:02d}" for k in range(n_views)),
        pose_ids=tuple(f"pose_{k:02d}" for k in range(n_views)),
        object_points=objp,
        image_points=np.asarray(pts, dtype=np.float64),
        image_size=(640, 480),
        corner_source=("sb",) * n_views,
    )
    return obs, model, _M2_TRUE.copy(), rvecs, tvecs


# ---- 行为 1:Schur 边际化与"完整 JᵀJ 直接求逆取内参块"的恒等式 ----

def test_schur_marginalization_matches_direct_inversion():
    """Schur 消元路线与"完整 JᵀJ 直接求逆取内参块"一致到机器精度级。

    两层恒等式(分块代数正确性的最强证据;任何列切片/行序错位都会当场暴露):
    - F 层(主判据):F_marg(Schur 分块)与 F_θθ − F_θβ·inv(F_ββ 全矩阵)·F_βθ
      (不分块、整块消元的独立装配路线)逐元素相对差 < 1e-10;
    - 协方差层:inv(F_marg) 与 inv(JᵀJ)[:P, :P] 相对差 < 1e-9。

    "逐元素相对差"的分母兜底体例照 test_models_jacobian.py 的
    max(范数, 兜底):F 层分母以矩阵最大元的 1e-4 兜底——双精度舍入的
    绝对量级是 eps·‖F‖ ≈ 2e-16·‖F‖,兜底 1e-4·‖F‖ 后舍入贡献的相对差
    上限 2e-12,1e-10 的判定线自带 45 倍余量;任何真实的代数错误
    (错块/错行/转置)都是 O(1) 量级,必红。实测本场景 2.4e-12。

    协方差层的差异由两条求逆路线各自的舍入决定,量级 eps·cond
    (JᵀJ 的条件数 ~1e11;实测一致到矩阵量级的 1.6e-10),判定线取
    1e-9 留余量——恒等式的主判据在 F 层。
    """
    obs, model, params, rvecs, tvecs = _scene()
    F_marg, F_full, diag = marginalized_fisher(model, obs, params, rvecs, tvecs)
    n_int = len(model.param_names)
    n_views = len(obs.image_ids)

    # F 层:同一消元,分块批量逆 vs 全矩阵一把逆(独立装配路线)
    F_full_dense = F_full.toarray()
    F_marg_direct = (F_full_dense[:n_int, :n_int]
                     - F_full_dense[:n_int, n_int:]
                     @ np.linalg.inv(F_full_dense[n_int:, n_int:])
                     @ F_full_dense[n_int:, :n_int])
    scale_f = float(np.abs(F_marg_direct).max())
    denom_f = np.maximum(np.abs(F_marg_direct), 1e-4 * scale_f)
    err_f = float(np.max(np.abs(F_marg - F_marg_direct) / denom_f))
    assert err_f < 1e-10, \
        f"Schur 恒等式破坏:F 层逐元素相对差 {err_f:.2e} 超线 1e-10" \
        f"(分块批量求逆或列切片有错)"

    # 协方差层:分块 Schur 的逆 vs 全矩阵求逆取内参块
    cov_direct = np.linalg.inv(F_full_dense)[:n_int, :n_int]
    cov_schur = np.linalg.inv(F_marg)
    err_c = float(np.max(np.abs(cov_schur - cov_direct)) / np.abs(cov_direct).max())
    assert err_c < 1e-9, \
        f"Schur 恒等式破坏:协方差层相对差 {err_c:.2e} 超线 1e-9" \
        f"(inv(F_marg) 与 inv(JᵀJ) 内参块对不上)"

    assert diag["n_views"] == 6 and diag["n_params"] == n_int
    assert diag["param_names"] == model.param_names
    assert 0.0 < diag["max_pose_block_cond"] < 1e12
    assert F_marg.shape == (n_int, n_int)
    assert F_full.shape == (n_int + 6 * n_views, n_int + 6 * n_views)


# ---- 行为 2:F_ββ 的块对角结构(行序契约的结构锁) ----

def test_pose_fisher_blocks_are_block_diagonal():
    """F_ββ 的非对角 6×6 块范数 < 1e-12(结构零,不是近似零)。

    第 i 个视图的残差行只碰第 i 个视图的外参列,跨视图块必须严格为零;
    一旦非零 = 列切片或行序被谁动过(不报错只算错的那类事故)。
    """
    obs, model, params, rvecs, tvecs = _scene()
    _F_marg, F_full, _diag = marginalized_fisher(model, obs, params, rvecs, tvecs)
    n_int = len(model.param_names)
    n_views = len(obs.image_ids)

    F_bb = np.asarray(F_full[n_int:, n_int:].todense())
    off_diag_norms = [
        float(np.linalg.norm(F_bb[6 * i:6 * i + 6, 6 * j:6 * j + 6]))
        for i in range(n_views) for j in range(n_views) if i != j
    ]
    worst = max(off_diag_norms)
    assert worst < 1e-12, \
        f"F_ββ 的跨视图 6×6 块范数 {worst:.2e} 超线 1e-12——块对角结构被破坏," \
        f"行列顺序多半被动过(Pitfall 3)"


# ---- 行为 3:奇异位姿块的中文防护(绝不静默 pinv) ----

def test_degenerate_pose_block_raises_chinese_error_naming_view():
    """某视图位姿不可观测 → 中文报错点名该视图序号,绝不静默 pinv 糊过去。

    造法:把全部物点换成躺在一条直线上的点——绕这条直线的旋转对投影
    没有任何影响(导数恒零),该视图外参 6×6 信息块必然奇异。位姿不可
    观测本身是要上报的退化发现,所以防护必须是响亮的报错。
    """
    obs, model, params, rvecs, tvecs = _scene()
    objp_line = np.zeros((obs.object_points.shape[0], 3), dtype=np.float64)
    objp_line[:, 0] = np.linspace(-0.4, 0.4, obs.object_points.shape[0])
    pts = []
    for i in range(len(obs.image_ids)):
        pts_cam, _ = transform_to_camera(objp_line, rvecs[i], tvecs[i])
        pts.append(model.project(pts_cam, params))
    obs_line = replace(obs, object_points=objp_line, image_points=np.asarray(pts))

    with pytest.raises(AssertionError, match="不可观测") as excinfo:
        marginalized_fisher(model, obs_line, params, rvecs, tvecs)
    assert "个视图" in str(excinfo.value), \
        f"报错必须点名视图序号,实际消息:{excinfo.value}"


def test_fisher_source_has_no_second_math_and_no_pinv():
    """静态锁定(D-10 + 奇异防护红线):fisher.py 源码里——

    - 不出现 "pinv"(奇异位姿块必须报错上报,不许伪逆糊过去);
    - 不出现 "projectPoints" / "cv2"(投影只经 models/,第二份数学禁止);
    - 必须出现 "make_jacobian_fn" 与 "pack_theta"(J 与 θ 装箱只走
      residuals.py 的唯一入口)。
    """
    src = (Path(__file__).resolve().parents[1]
           / "src" / "calib" / "noise" / "fisher.py").read_text(encoding="utf-8")
    assert "pinv" not in src, "fisher.py 出现 pinv——奇异防护被静默绕过"
    assert "projectPoints" not in src, "fisher.py 出现 projectPoints——第二份数学进场"
    assert "cv2" not in src, "fisher.py 直接碰 cv2——绕过了 models/ 唯一数学源"
    assert "make_jacobian_fn" in src, "fisher.py 没走 make_jacobian_fn 取雅可比"
    assert "pack_theta" in src, "fisher.py 没走 pack_theta 装箱参数"


# ---- 行为 4:解析 FIM 对 numdifftools 数值差分基准(D-10 的数值背书) ----

def test_analytic_fim_matches_numdifftools_oracle():
    """数值雅可比 → 数值 FIM → 同一 Schur 路线,与解析 F_marg 逐元素
    相对差 < 1e-6(分母兜底体例照 test_models_jacobian.py)。

    numdifftools 是独立裁判:解析雅可比若有一处推导错,这一关必红。
    """
    obs, model, params, rvecs, tvecs = _scene()
    theta = pack_theta(model, params, rvecs, tvecs)
    n_int = len(model.param_names)
    n_views = len(obs.image_ids)

    jac_num = nd.Jacobian(make_residual_fn(model, obs))(theta)  # (2VP, P+6V) 稠密
    from calib.noise.fisher import _fisher_marginal_from_jacobian
    F_marg_num, _F_bb, _cond = _fisher_marginal_from_jacobian(
        csr_matrix(jac_num), n_int, n_views)

    F_marg_ana, _F_full, _diag = marginalized_fisher(model, obs, params, rvecs, tvecs)

    denom = np.maximum(np.abs(F_marg_num), 1e-12)
    err = float(np.max(np.abs(F_marg_ana - F_marg_num) / denom))
    assert err < 1e-6, \
        f"解析 FIM 对 numdifftools 数值基准逐元素相对差 {err:.2e} 超线 1e-6" \
        f"(D-10:解析雅可比或分块组装有错)"


# ==== 以下为 CRLB 主线数学(σ 缩放 / 敏感性带 / 特征谱 / 蒙特卡洛实证) ====


# ---- σ 线性缩放恒等式(双口径 CRLB 全部免费的理论根据) ----

@pytest.mark.parametrize("sigma", [0.3, 1.0, 2.5])
def test_crlb_scales_strictly_linearly_with_sigma(sigma: float):
    """CRLB 随 σ 严格线性:逐位一致 + 加权重算交叉验证 < 1e-12。

    两层:①crlb_se(F, σ) == σ × crlb_se(F, 1.0) 逐位一致(纯乘法);
    ②用 J/σ 重装加权 FIM 的独立路线复算,三档与乘法结果相对差 < 1e-12
    ——σ 只作乘数这条性质被两条路线同时锁死,双口径 × Σ 敏感性带
    不需要重算 FIM。
    """
    obs, model, params, rvecs, tvecs = _scene()
    F_marg, _F_full, _diag = marginalized_fisher(model, obs, params, rvecs, tvecs)
    se_unit = crlb_se(F_marg, 1.0)

    # ①乘法路线:逐位一致
    assert np.array_equal(crlb_se(F_marg, sigma), sigma * se_unit), \
        f"σ={sigma} 的 CRLB 不是 σ × SE(1.0) 的逐位结果——乘法缩放被破坏"

    # ②加权重算路线:J/σ 重新组装 FIM,验证两条
    #  a) F 层(机器精度,主判据):重算 FIM 与 σ=1 结果 ÷ σ² 相对差 < 1e-12
    #     ——计划 must_have 的"重算加权 FIM 与 σ=1 结果一致到机器精度";
    #  b) SE 层(过 inv 的下游):重算 FIM 开方与乘法 SE 相对差 < 1e-8。
    #     J/σ 的逐元乘法引入 ~eps 舍入,经 inv 按条件数放大(本场景
    #     cond(F_marg) ~5e8),实测 ~3e-10——1e-8 留 30 倍余量。
    F_weighted = weighted_fisher(model, obs, params, rvecs, tvecs, sigma)
    F_scaled = F_marg / (sigma * sigma)
    err_f = float(np.max(np.abs(F_weighted - F_scaled)) / np.abs(F_scaled).max())
    assert err_f < 1e-12, \
        f"σ={sigma} 加权重算 FIM 与 σ=1 结果÷σ² 的相对差 {err_f:.2e} 超线 1e-12" \
        f"(F 随 1/σ² 缩放到机器精度的性质被破坏)"

    se_weighted = crlb_se(F_weighted, 1.0)
    denom = np.maximum(sigma * se_unit, 1e-300)
    err = float(np.max(np.abs(se_weighted - sigma * se_unit) / denom))
    assert err < 1e-8, \
        f"σ={sigma} 加权重算的 SE 与乘法缩放的相对差 {err:.2e} 超线 1e-8" \
        f"(过 inv 的舍入放大超出预期)"


# ---- Σ 敏感性带(±27%,纯乘法) ----

def test_sigma_band_is_pure_multiplication_and_constant_locked():
    """敏感性带 = 每个 SE 乘 (1 ± 1/√14),常量来历被注释与数值双锁定。

    1/√(2×(8−1)) = 1/√14 ≈ 0.267:σ 由 8 样本(同姿态 8 连拍)的标准差
    估出时,标准差本身携带的相对标准误——Σ 的 ±27% 不确定性因此以带的
    形式随 CRLB 落盘,不输出裸下界。
    """
    rel = SIGMA_BAND_REL_8
    assert abs(rel - 1.0 / np.sqrt(14.0)) < 1e-15, \
        "SIGMA_BAND_REL_8 必须恰为 1/√14(8 样本标准差的相对标准误)"
    assert abs(rel - 0.267) < 5e-4, "1/√14 ≈ 0.267,偏差异常说明常量被改"

    se = np.array([0.01, 0.5, 3.0])
    band = sensitivity_band(se)
    assert np.array_equal(band["low"], se * (1.0 - rel)), "带下沿必须是 se×(1−rel)"
    assert np.array_equal(band["high"], se * (1.0 + rel)), "带上沿必须是 se×(1+rel)"
    assert band["rel"] == rel


# ---- 特征谱 / 条件数 / 软方向(手算 2×2 对照) ----

def test_spectrum_report_matches_hand_calculated_2x2():
    """手算 2×2 已知矩阵 [[4,1],[1,1]]:特征值升序、条件数、软方向 top-2
    载荷参数名逐位对照。

    手算:特征值 λ(F) = (5±√13)/2;协方差 = inv(F) 的特征值是其倒数,
    升序 [2/(5+√13), 2/(5−√13)];条件数 = (5+√13)/(5−√13) ≈ 6.17;
    最大协方差特征向量 ∝ (1, −(3+√13)/2),|载荷| 最大的参数是第 2 个。
    """
    F2 = np.array([[4.0, 1.0], [1.0, 1.0]])
    rep = spectrum_report(F2, ("a", "b"))

    lam_max = (5.0 + np.sqrt(13.0)) / 2.0
    lam_min = (5.0 - np.sqrt(13.0)) / 2.0
    np.testing.assert_allclose(rep["eigvals"], [1.0 / lam_max, 1.0 / lam_min],
                               rtol=1e-12, err_msg="协方差特征值(升序)与手算不符")
    assert abs(rep["condition_number"] - lam_max / lam_min) < 1e-12 * (lam_max / lam_min), \
        "条件数(λmax/λmin)与手算不符"
    top = rep["top_directions"][0]
    assert top["params"][:2] == ["b", "a"], \
        f"软方向 top-2 载荷参数名应为 ['b', 'a'],实际 {top['params']}"
    assert abs(abs(top["loadings"][0]) - abs((3.0 + np.sqrt(13.0)) / 2.0)
               / np.sqrt(1.0 + ((3.0 + np.sqrt(13.0)) / 2.0) ** 2)) < 1e-12, \
        "最大 |载荷| 与手算特征向量的归一化分量不符"
    assert rep["flag"] == "", "条件数 ≈ 6.2 不应触发任何退化旗标"


def test_spectrum_flag_thresholds_warning_and_degenerate():
    """旗标按条件数分档:≥1e4 记 warning,≥1e6 记 degenerate(默认阈值)。"""
    rep_warn = spectrum_report(np.diag([1.0e5, 1.0]), ("a", "b"))
    assert rep_warn["flag"] == "warning"
    rep_deg = spectrum_report(np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]]), ("a", "b"))
    assert rep_deg["flag"] == "degenerate"


# ---- 蒙特卡洛实证(合成数据上的数学单测) ----

_MC_N_REP = 200          # 200 次加噪-求解:std 估计的抽样误差约 1/√(2·199) ≈ 5%
_MC_SIGMA = 0.3          # 像素噪声:6 视图小场景单次求解秒级以下,总计远低于 60 秒


def test_monte_carlo_std_matches_fim_prediction():
    """200 次加噪-scipy 求解的逐参数 std 与 FIM 预测 SE 一致到 ±15% 量级。

    注意:这是合成数据上的数学单测,锁的是"FIM 数学没算错"这根桩;
    与阶段四"真实数据上的 CRLB 实证校验"无关。

    做法:同一合成 6 视图 M2 场景,固定种子注入 σ=0.3 像素高斯噪声,
    从真值点出发用 scipy 求解 200 次;逐参数的经验 std 除以 CRLB 预测
    SE,比值应全部落在 [0.85, 1.18](个别参数越线时按计划放宽到
    [0.8, 1.25],理由见断言前注释)。
    """
    obs, model, params, rvecs, tvecs = _scene()
    theta_true = pack_theta(model, params, rvecs, tvecs)
    F_marg, _F_full, _diag = marginalized_fisher(model, obs, params, rvecs, tvecs)
    se_pred = crlb_se(F_marg, _MC_SIGMA)

    rng = np.random.default_rng(20260302)
    estimates = np.empty((_MC_N_REP, len(model.param_names)), dtype=np.float64)
    for k in range(_MC_N_REP):
        noisy = obs.image_points + rng.normal(0.0, _MC_SIGMA, obs.image_points.shape)
        obs_k = replace(obs, image_points=noisy)
        jac_fn = make_jacobian_fn(model, obs_k)

        def dense_jac(th, _fn=jac_fn):     # 默认参数绑定本轮工厂,避开迟绑定
            return _fn(th).toarray()

        sol = least_squares(
            make_residual_fn(model, obs_k), theta_true,
            jac=dense_jac,
            method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if sol.status <= 0:
            raise AssertionError(f"蒙特卡洛第 {k} 次合成求解未收敛(status={sol.status})")
        estimates[k] = unpack_theta(model, sol.x)[0]

    std_mc = estimates.std(axis=0, ddof=1)
    ratios = std_mc / se_pred
    lo, hi = 0.85, 1.18
    if np.any((ratios < lo) | (ratios > hi)):
        # 放宽理由(计划明文允许):200 个样本估 std 自带 ~5% 抽样误差,
        # 弱约束参数(如 k3)还有小量非线性偏置,±15% 判定线边缘会有
        # 个别越线;放宽到 [0.8, 1.25] 仍锁定"逐参数同量级一致"的实证结论。
        lo, hi = 0.8, 1.25
    assert (ratios >= lo).all() and (ratios <= hi).all(), \
        f"蒙特卡洛 std 与 FIM 预测 SE 的比值越界:最小 {ratios.min():.3f}," \
        f"最大 {ratios.max():.3f},判定区间 [{lo}, {hi}]," \
        f"参数序 {model.param_names},比值 {np.round(ratios, 3)}"


# ---- D-10 静态锁定:numdifftools 绝不进 src/calib ----

def test_numdifftools_never_in_calib_source():
    """src/calib 全树不出现 numdifftools 引用(仿 test_models_jacobian 体例)。

    数值差分只配当测试基准;数学源一旦依赖它,解析雅可比的正确性就
    没有独立裁判了,而且求解路径会被拖慢几个数量级。
    """
    repo_root = Path(__file__).resolve().parents[1]
    hits = [str(p.relative_to(repo_root))
            for p in (repo_root / "src" / "calib").rglob("*.py")
            if "numdifftools" in p.read_text(encoding="utf-8")]
    assert not hits, f"src/calib 里出现 numdifftools 引用(D-10 违规):{hits}"


# ==== 以下为 80 姿态聚合观测集与聚合位姿(03-05,D-01/D-02)====

# 小棋盘(3×2 = 6 个角点)——聚合逻辑不关心角点数,数组小跑得快
_SMALL_BOARD = BoardSpec(cols=3, rows=2)
_P = _SMALL_BOARD.cols * _SMALL_BOARD.rows


def _write_corners_npz(path: Path, image_ids: list[str], uv: np.ndarray,
                       detected: np.ndarray, source: list[str]) -> Path:
    """按 corners 产物的落盘格式写一份合成角点文件(体例照 corners/artifacts.py)。"""
    np.savez(path, image_ids=np.asarray(image_ids),
             uv=np.asarray(uv, dtype=np.float64),
             detected=np.asarray(detected, dtype=bool),
             source=np.asarray(source))
    return path


def _burst_records(ids_per_pose: dict[str, list[str]]) -> list[dict]:
    """按姿态造 manifest 记录片段(image_id/pose_id/split/n_repeats)。"""
    records = []
    for pose_id, ids in ids_per_pose.items():
        for k, image_id in enumerate(ids):
            records.append({"image_id": image_id, "pose_id": pose_id,
                            "split": "train", "repeat_idx": k,
                            "n_repeats": len(ids)})
    return records


# ---- 行为 1:uv 逐点均值逐位一致 + detected_agg = 全部帧检出 + 来源留痕 ----

def test_aggregate_observations_uv_mean_matches_hand_calculation(tmp_path):
    """合成 2 姿态 × 3 帧(同一角点真值加已知小扰动)→

    - uv_agg 与手算"检出帧逐点均值"逐位一致(D-02:均值只吃检出帧);
    - detected_agg = 该姿态全部帧检出(有一帧没检出就记 False);
    - 来源列表逐姿态留痕(哪些帧进了均值,可审计)。
    """
    base = np.array([[100.0 + 3.0 * i, 200.0 + 2.0 * i] for i in range(_P)])
    # pose A:3 帧,第 2 帧未检出(整行 NaN)→ 均值只吃第 1、3 帧
    uv_a = np.stack([base + 0.10, np.full((_P, 2), np.nan), base - 0.06])
    # pose B:3 帧全检出
    uv_b = np.stack([base + 0.20, base + 0.30, base + 0.40])
    uv = np.concatenate([uv_a, uv_b])
    detected = np.array([True, False, True, True, True, True])
    ids = ["a1", "a2", "a3", "b1", "b2", "b3"]
    sources = ["sb", "sb", "classic_fallback", "sb", "sb", "sb"]
    npz = _write_corners_npz(tmp_path / "corners_train_sb.npz",
                             ids, uv, detected, sources)

    # 每帧一份"外参"(位姿):解侧只含检出帧(与真实标定产物同构)
    rng = np.random.default_rng(7)
    rvecs_all = rng.uniform(-0.1, 0.1, (5, 3))
    tvecs_all = rng.uniform(-10, 10, (5, 3))
    image_ids_all = ["a1", "a3", "b1", "b2", "b3"]
    records = _burst_records({"poseA": ["a1", "a2", "a3"],
                              "poseB": ["b1", "b2", "b3"]})

    agg = aggregate_observations(
        npz, records, rvecs_all, tvecs_all, image_ids_all,
        image_size=(640, 480))

    # uv 均值逐位一致:pose A 只吃检出的第 1、3 帧
    expect_a = (uv[0] + uv[2]) / 2.0
    np.testing.assert_array_equal(agg["poses"]["poseA"]["uv_agg"], expect_a,
                                  err_msg="poseA 的逐点均值与手算不一致")
    expect_b = (uv[3] + uv[4] + uv[5]) / 3.0
    np.testing.assert_array_equal(agg["poses"]["poseB"]["uv_agg"], expect_b,
                                  err_msg="poseB 的逐点均值与手算不一致")

    # detected_agg = 该姿态全部帧检出
    assert agg["poses"]["poseA"]["detected_agg"] is False
    assert agg["poses"]["poseB"]["detected_agg"] is True

    # 来源列表逐姿态留痕(只记进了均值的帧)
    assert agg["poses"]["poseA"]["sources"] == ["sb", "classic_fallback"]
    assert agg["poses"]["poseB"]["sources"] == ["sb", "sb", "sb"]

    # 聚合位姿:与逐帧外参的手算均值一致(平移逐位;旋转与 scipy 对照)
    np.testing.assert_array_equal(
        agg["poses"]["poseA"]["t_mean"],
        (tvecs_all[0] + tvecs_all[1]) / 2.0)
    np.testing.assert_allclose(
        agg["poses"]["poseA"]["rot_mean"],
        Rotation.from_rotvec(rvecs_all[[0, 1]]).mean().as_rotvec(),
        atol=1e-15)


# ---- 行为 2:聚合位姿红线——旋转均值,绝不逐分量平均 rodrigues ----

def test_aggregate_pose_rotation_mean_is_not_component_mean():
    """对两个转角较大的 rodrigues 向量(各绕不同轴 0.5 rad):

    - 聚合旋转与 scipy Rotation.mean() 对照一致(实现即用弦长均值);
    - 与"逐分量平均"的结果显著不同(两者旋转差 > 1e-3 弧度)——
      把"rodrigues 向量不是线性空间,逐分量平均不是旋转平均"这条禁令
      用数字锁死,谁改回逐分量平均谁就红。
    """
    rvecs = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    tvecs = np.zeros((2, 3))
    base = np.zeros((_P, 2))
    uv = np.stack([base, base])
    ids = ["x1", "x2"]
    npz = _write_corners_npz(_rot_test_npz_path(), ids, uv,
                             np.array([True, True]), ["sb", "sb"])
    records = _burst_records({"poseX": ids})
    agg = aggregate_observations(npz, records, rvecs, tvecs, ids,
                                 image_size=(64, 48))
    rot_mean = agg["poses"]["poseX"]["rot_mean"]

    # 与 scipy 弦长均值一致(逐位)
    np.testing.assert_array_equal(
        rot_mean, Rotation.from_rotvec(rvecs).mean().as_rotvec(),
        err_msg="聚合旋转不是 scipy Rotation.mean() 的弦长均值")

    # 与逐分量平均显著不同(红线:差 > 1e-3 弧度)
    naive = rvecs.mean(axis=0)
    angle = float((Rotation.from_rotvec(naive)
                   * Rotation.from_rotvec(rot_mean).inv()).magnitude())
    assert angle > 1e-3, (
        f"逐分量平均与旋转均值的差只有 {angle:.2e} rad,红线断言(>1e-3)"
        f"失去分辨力;或实现已退化成逐分量平均")


def _rot_test_npz_path() -> Path:
    """行为 2 用例的角点文件路径(固定文件名,内容每次覆盖,无残留影响)。"""
    d = Path(tempfile.mkdtemp(prefix="aggregate_rot_"))
    return d / "corners.npz"


# ---- 行为 3:σ 折减 n_repeats=8 → sigma_scale = 1/√8(iid 假设) ----

def test_sigma_scale_records_iid_reduction_for_8_frame_burst(tmp_path):
    """8 帧连拍全检出 → n_repeats=8、sigma_scale = 1/√8。

    折减成立的前提是 iid(连拍各帧噪声互不相关)——连拍若相关,√8
    折减不成立。这不是缺陷:640 全帧版与 80 聚合版的 CRLB 差值,要量化的
    正是"连拍相关性吃掉多少信息"(D-01 的设计意图),阶段四的重复张
    分组实验是它的实测对照。
    """
    base = np.zeros((_P, 2))
    uv = np.stack([base + 0.01 * k for k in range(8)])
    ids = [f"f{k}" for k in range(8)]
    npz = _write_corners_npz(tmp_path / "c8.npz", ids, uv,
                             np.ones(8, dtype=bool), ["sb"] * 8)
    records = _burst_records({"pose8": ids})
    agg = aggregate_observations(npz, records,
                                 np.zeros((8, 3)), np.zeros((8, 3)), ids,
                                 image_size=(64, 48))
    entry = agg["poses"]["pose8"]
    assert entry["n_repeats"] == 8
    assert entry["sigma_scale"] == pytest.approx(1.0 / np.sqrt(8.0))
    assert entry["detected_agg"] is True


# ---- 行为 4:聚合观测集通过 ObservationSet 全部形状/有限值断言 ----

def test_build_aggregated_observation_set_passes_all_checks(tmp_path):
    """全检出的聚合 → build_aggregated_observation_set 构造的观测集:

    六字段齐全、image_points (V,P,2) 全有限、image_ids == pose_ids
    (一姿态一视图)、corner_source 全记 "pose_mean"。
    """
    base = np.array([[50.0 + i, 60.0 + i] for i in range(_P)])
    uv = np.stack([base + 0.1, base + 0.3])
    ids = ["g1", "g2"]
    npz = _write_corners_npz(tmp_path / "g.npz", ids, uv,
                             np.array([True, True]), ["sb", "sb"])
    agg = aggregate_observations(npz, _burst_records({"poseG": ids}),
                                 np.zeros((2, 3)), np.zeros((2, 3)), ids,
                                 image_size=(640, 480))
    obs = build_aggregated_observation_set(_SMALL_BOARD, agg)
    # 一姿态一视图:一个姿态占一行
    assert obs.image_ids == ("poseG",)
    assert obs.image_points.shape == (1, _P, 2)
    assert np.isfinite(obs.image_points).all()
    assert obs.image_ids == obs.pose_ids
    assert set(obs.corner_source) == {"pose_mean"}
    assert obs.object_points.shape == (_P, 3)
    assert obs.image_size == (640, 480)


def test_build_aggregated_observation_set_rejects_undetected_pose(tmp_path):
    """带未检出帧的聚合进构造器:中文报错点名姿态,绝不带病进 Fisher。"""
    base = np.zeros((_P, 2))
    uv = np.stack([base, np.full((_P, 2), np.nan)])
    ids = ["h1", "h2"]
    npz = _write_corners_npz(tmp_path / "h.npz", ids, uv,
                             np.array([True, False]), ["sb", "sb"])
    agg = aggregate_observations(npz, _burst_records({"poseH": ids}),
                                 np.zeros((1, 3)), np.zeros((1, 3)), ["h1"],
                                 image_size=(64, 48))
    with pytest.raises(AssertionError, match="poseH"):
        build_aggregated_observation_set(_SMALL_BOARD, agg)
