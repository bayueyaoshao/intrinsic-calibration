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

from dataclasses import replace
from pathlib import Path

import numdifftools as nd  # D-10:数值差分基准,只进测试,绝不进 src/calib
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calib.io.config import BoardSpec
from calib.models import create_model
from calib.models.base import transform_to_camera
from calib.noise.fisher import marginalized_fisher
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet
from calib.solve.residuals import (
    make_jacobian_fn,
    make_residual_fn,
    pack_theta,
    unpack_theta,
)

# 合成小场景的固定棋盘与 M2 真值参数(棋盘与 config/board.yaml 同规格 11×8;
# 畸变给小而非零的值,让 FIM 的畸变列有真实信息量)
_BOARD = BoardSpec(cols=11, rows=8, square_size_mm=25.0)
_M2_TRUE = np.array([800.0, 810.0, 330.0, 250.0,
                     -0.05, 0.02, 0.001, -0.001, 0.0005])


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
        rvecs[i] = rng.uniform(-0.1, 0.1, 3)
        tvecs[i] = [*rng.uniform(-30.0, 30.0, 2), 800.0]
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
    """inv(F_marg) 与 inv(JᵀJ)[:P, :P] 逐元素相对差 < 1e-10。

    两条独立路线(分块 Schur 消元 vs 全矩阵求逆取块)必须一致到机器
    精度级——这是分块代数正确性的最强证据;任何列切片/行序错位都会
    在这里当场暴露。诊断字典一并核对键与量级。
    """
    obs, model, params, rvecs, tvecs = _scene()
    F_marg, F_full, diag = marginalized_fisher(model, obs, params, rvecs, tvecs)
    n_int = len(model.param_names)

    cov_direct = np.linalg.inv(F_full.toarray())[:n_int, :n_int]
    cov_schur = np.linalg.inv(F_marg)
    denom = np.maximum(np.abs(cov_direct), 1e-12)
    err = float(np.max(np.abs(cov_schur - cov_direct) / denom))
    assert err < 1e-10, \
        f"Schur 恒等式破坏:inv(F_marg) 与 inv(JᵀJ) 内参块逐元素相对差 " \
        f"{err:.2e} 超线 1e-10(分块代数或列切片有错)"

    assert diag["n_views"] == 6 and diag["n_params"] == n_int
    assert diag["param_names"] == model.param_names
    assert 0.0 < diag["max_pose_block_cond"] < 1e12
    assert F_marg.shape == (n_int, n_int)
    assert F_full.shape == (n_int + 6 * 6, n_int + 6 * 6)


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

    with pytest.raises(AssertionError, match="位姿不可观测") as excinfo:
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
