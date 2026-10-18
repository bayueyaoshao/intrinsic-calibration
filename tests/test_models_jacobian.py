"""模型层解析雅可比的数值校验:逐模型逐参数对 numdifftools 基准。

锁定的决策与量级:
- D-10(硬约束):numdifftools 是 dev 依赖、**只出现在 tests/ 目录**
  (本文件 import 它),src/calib 里出现对它的引用即违规——parity 文件
  有配套的 grep 锁定测试与提交前检查;
- 判定统一用**列相对误差 < 1e-6**(研究遗留问题 3 的定夺):解析列与
  数值列各求范数后比相对差,不逐列设绝对值——中心差分地板 ~7e-10,
  numdifftools 带理查森外推更准,1e-6 余量充足;
- M1–M3 另用 cv2.projectPoints 的 jacobian 输出(float64)做辅助交叉
  验证:相对误差 < 1e-4(cv2 的雅可比自身精度 ~2e-7,只作旁证不作
  唯一基准;float32 下它是完全错的,绝不许用);
- 点集必须含近轴点(含正原点):锁 KB4 的除零保护不影响导数正确性。
"""

from __future__ import annotations

import cv2
import numdifftools as nd  # D-10:数值差分基准,只进测试,绝不进 src/calib
import numpy as np
import pytest

from calib.models import BUILTIN_MODEL_IDS, create_model

# 与 parity 测试同一套参数采样(边界参数:导数非零才比得出意义)
_PARAMS = {
    "m1": (1872.0, 1872.5, 1280.2, 721.8, -0.35, 0.18, 0.003, -0.0025),
    "m2": (1872.0, 1872.5, 1280.2, 721.8, -0.35, 0.18, 0.003, -0.0025, 0.06),
    "m3": (1872.0, 1872.5, 1280.2, 721.8, -0.35, 0.18, 0.003, -0.0025, 0.06,
           0.05, -0.08, 0.03),
    "m4": (410.0, 410.5, 1279.6, 720.9, 0.35, -0.28, 0.16, -0.05),
}


def _points(model_id: str, seed: int = 7, n_random: int = 300) -> np.ndarray:
    """点集:随机 + 近轴(含正原点)+ 大半径;比 parity 少些点,数值差分快。"""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, size=n_random)
    if model_id == "m4":
        theta = rng.uniform(0.0, 1.2, size=n_random)
        pts = np.stack([np.sin(theta) * np.cos(ang), np.sin(theta) * np.sin(ang),
                        np.cos(theta)], axis=1)
        near = np.array([[1e-6, 0.0, np.sqrt(1.0 - 1e-12)],
                         [0.0, 0.0, 1.0]])  # 正原点:近轴保护下的导数也必须对
        big = np.array([[np.sin(1.3) * np.cos(a), np.sin(1.3) * np.sin(a), np.cos(1.3)]
                        for a in (0.3, 1.1, 2.2)])
    else:
        rad = rng.uniform(0.0, 1.2, size=n_random)
        z = rng.uniform(0.3, 5.0, size=n_random)
        pts = np.stack([rad * np.cos(ang) * z, rad * np.sin(ang) * z, z], axis=1)
        near = np.array([[1e-6, 0.0, 1.0], [0.0, -1e-6, 1.0]])
        big = np.array([[1.2 * np.cos(a), 1.2 * np.sin(a), 1.0]
                        for a in (0.4, 1.3, 2.6)])
    return np.vstack([pts, near, big])


# 逐模型逐参数参数化:M1×8 + M2×9 + M3×12 + M4×8 = 37 组合
_CASES = [(m, name) for m in BUILTIN_MODEL_IDS for name in create_model(m).param_names]


@pytest.mark.parametrize(("model_id", "param_name"), _CASES)
def test_analytic_jacobian_vs_numdifftools(model_id: str, param_name: str):
    """解析雅可比逐参数列对 numdifftools 数值基准:列相对误差 < 1e-6。

    列相对误差 = ‖解析列 − 数值列‖ / max(‖数值列‖, 1e-12):
    分母兜底 1e-12 防止近零列(如正原点处的 k 列)除出天文数字。
    """
    model = create_model(model_id)
    params = np.array(_PARAMS[model_id], dtype=np.float64)
    pts = _points(model_id)
    col = model.param_names.index(param_name)

    jac_ana = model.project_jacobian(pts, params).reshape(-1, len(params))
    jac_num = nd.Jacobian(lambda p: model.project(pts, p).ravel())(params)

    denom = max(float(np.linalg.norm(jac_num[:, col])), 1e-12)
    err = float(np.linalg.norm(jac_ana[:, col] - jac_num[:, col])) / denom
    assert err < 1e-6, \
        (f"模型 {model_id} 参数 {param_name} 的解析雅可比列相对误差 {err:.2e} "
         f"超线 1e-6(数值基准:numdifftools 中心差分 + 理查森外推)")


# cv2.projectPoints 雅可比的列布局(实测 cv2 4.14):(2N, 10+畸变数),
# 前 6 列是 rvec/tvec,第 6..9 列 = fx,fy,cx,cy,第 10 列起按 D 顺序 k1,k2,p1,p2,(k3),(k4,k5,k6)
_CV_COLUMN_OF = {0: 6, 1: 7, 2: 8, 3: 9}


@pytest.mark.parametrize("model_id", ("m1", "m2", "m3"))
def test_jacobian_cross_check_opencv_float64(model_id: str):
    """M1–M3 解析雅可比对 cv2.projectPoints 的 jacobian(float64)旁证:< 1e-4。

    cv2 的雅可比自身精度 ~2e-7(对中心差分),只作辅助交叉验证;
    float32 输入下它是完全错的(实测误差 60)——这里强制 float64。
    """
    model = create_model(model_id)
    params = np.array(_PARAMS[model_id], dtype=np.float64)
    pts = _points(model_id, seed=11)
    jac_ana = model.project_jacobian(pts, params)  # (N,2,P)

    n_dist = {"m1": 4, "m2": 5, "m3": 8}[model_id]
    d = np.zeros(8, dtype=np.float64)
    d[:n_dist] = params[4:]
    _uv, jac_cv = cv2.projectPoints(
        pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        model.intrinsics_matrix(params), d)
    jac_cv = jac_cv.reshape(-1, 2, jac_cv.shape[1])  # (N,2,10+8)

    for j, name in enumerate(model.param_names):
        cv_col = _CV_COLUMN_OF.get(j, 10 + (j - 4))  # 内参前 4 个映射,畸变接第 10 列起
        ours = jac_ana[:, :, j].ravel()
        theirs = jac_cv[:, :, cv_col].ravel()
        denom = max(float(np.linalg.norm(theirs)), 1e-12)
        err = float(np.linalg.norm(ours - theirs)) / denom
        assert err < 1e-4, \
            (f"模型 {model_id} 参数 {name} 对 cv2 雅可比(第 {cv_col} 列)"
             f"相对误差 {err:.2e} 超线 1e-4——先确认两侧都是 float64")


def test_numdifftools_never_in_src():
    """D-10 静态锁定:src/calib 里绝不出现 numdifftools 引用。

    数值差分只配当测试基准;数学源一旦依赖它,解析雅可比的正确性
    就没有独立裁判了,而且求解路径会被拖慢几个数量级。
    (提交前的命令行检查 `grep -rn numdifftools src/` 是同一道防线的
    命令行版,这里给日常开发一个 pytest 内的等价锁定。)
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    hits = [str(p.relative_to(repo_root)) for p in (repo_root / "src").rglob("*.py")
            if "numdifftools" in p.read_text(encoding="utf-8")]
    assert not hits, f"src/ 里出现 numdifftools 引用(D-10 违规):{hits}"
