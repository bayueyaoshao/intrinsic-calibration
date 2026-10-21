"""ANA-04 参数敏感度库层的行为测试:对角谱手算 + 建议规则表 + what-if 合成声明。

测的是什么(PLAN 05-04 Task 2 的 behavior + notes 断言):
1. 理论谱只消费不重算:合成对角特征谱(m1,8 参数,cy 最软)→ 最大方向
   载荷全部落在 cy、σ 级步长像素当量手算 = sqrt(4e-8)×800 = 0.16px;
   纯单参数方向不触发任何建议分支(advice 为空)。
2. 无量纲还原恒等式:raw ÷ scale == v_dimless(2 的幂刻度逐位还原)。
3. 建议规则表三分支(裁定③):k2/k3 集中 + 外推区间 → 补边缘姿态;
   覆盖段缺席 → 覆盖判据未知;像素参数耦合 → 补转角;p1/p2 可见但像素
   当量可忽略 → 不治。每条建议必带「不自动执行」声明。
4. uniqueness 如实透传:bootstrap 最大位移当量逐位原样、多起点判定
   原样、固定声明点名「解不唯一」。
5. what-if 档位(裁定②):quick 档 what_if 为 None + 明细空数组;
   full 档产出且合成声明必须含「合成」「非实拍」。
6. F 合并恒等式:4 个相同视角的 F 加 1 个相同视角 = 5F → 压缩比恰为
   sqrt(4/5)(线性代数恒等式,不允许"差不多")。
7. 源码红线(D-10):FIM 组装只许调 marginalized_fisher,本模块绝不
   重写 JᵀJ 组装、绝不 import OpenCV。
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from calib.analysis import sensitivity as sens
from calib.analysis.sensitivity import SCHEMA_SENSITIVITY, run_sensitivity
from calib.io.config import AnalyzeConfig, BoardSpec, CalibrationConfig
from calib.models import create_model
from calib.solve.baseline import build_object_points

_COLS, _ROWS, _SQUARE_MM = 11, 8, 20.0


def _m1_true_params() -> dict:
    """m1 真值内参(零畸变)——像素当量换算的基准参数。"""
    return {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}


def _m1_params_vec(model) -> np.ndarray:
    return np.array([float(_m1_true_params()[n]) for n in model.param_names])


def _write_synth_world(tmp_path):
    """合成 m1 世界:对角谱 fisher npz + bootstrap/slices 底账 npz + 报告字典。

    对角谱的含义:eigvecs = 单位阵 → 每个特征方向都是单一参数方向,
    像素当量可以手算(cy 方向 σ = sqrt(4e-8),×800 像素刻度 = 0.16px)。
    """
    model = create_model("m1")
    names = list(model.param_names)
    fisher_path = tmp_path / "fisher_m1.npz"
    eigvals = np.full(len(names), 1.0e-9)
    eigvals[names.index("cy")] = 4.0e-8  # cy 最软:最大特征方向必是纯 cy
    np.savez(
        fisher_path,
        eigvals_cov=eigvals,
        eigvecs_cov=np.eye(len(names)),
        scale_dimless=np.array([800.0, 800.0, 800.0, 800.0, 1.0, 1.0, 1.0, 1.0]),
        param_names=np.array(names),
        F_marg_views640=np.diag([1.0e6, 1.0e6, 1.0e6, 1.0e6,
                                 1.0e4, 1.0e4, 1.0e2, 1.0e2]),
    )
    rng = np.random.default_rng(2026)
    boot_path = tmp_path / "bootstrap.npz"
    np.savez(boot_path, m1__params=rng.normal(0.0, 1e-3, size=(50, 8)))
    slices_path = tmp_path / "slices.npz"
    np.savez(slices_path, m1__group_params=rng.normal(0.0, 1e-3, size=(8, 8)))
    report = {
        "lineage": {"round_id": "run_test"},
        "delivery": {"models": {"m1": {"final_params": _m1_true_params()}}},
        "slices": {"models": {"m1": {"empirical_std": {"p1": 1.0e-5, "p2": 1.5e-5}}}},
        "bootstrap": {"models": {"m1": {"max_disp_px": 1.9943}}},
        "optimizer": {"multistart": {"models": {"m1": {
            "param_disp_px": 0.205, "verdict": "inconsistent"}}}},
    }
    return {"model": model, "fisher": fisher_path, "boot": boot_path,
            "slices": slices_path, "report": report}


def _run_quick(tmp_path, coverage_section=None):
    """quick 档跑一遍完整编排(所有测试共用的取数路径)。"""
    world = _write_synth_world(tmp_path)
    cfg = CalibrationConfig()
    board = BoardSpec(cols=_COLS, rows=_ROWS, square_size_mm=_SQUARE_MM)
    return run_sensitivity(
        "m1", world["report"], world["fisher"], world["boot"], world["slices"],
        world["model"], build_object_points(board),
        np.zeros(3), np.array([0.0, 0.0, 800.0]),
        coverage_section, cfg,
    ), world


# ---------------------------------------------------------------------------
# Test 1:对角谱手算(理论谱只消费不重算的执行位)
# ---------------------------------------------------------------------------

def test_theory_spectrum_diag_handcheck(tmp_path):
    """对角谱:eigvecs=单位阵 → 最大方向必是纯 cy,载荷占比 1.0、像素当量
    手算 sqrt(4e-8)×800 = 0.16px(主点平移逐点等移,无近似);纯单参数
    方向不满足任何建议分支的判据,advice 必须为空(规则表不许乱开火)。"""
    (section, _detail), _world = _run_quick(tmp_path)
    assert section["schema"] == SCHEMA_SENSITIVITY

    top = section["theory_spectrum"]["top_directions"][0]
    assert next(iter(top["loads"])) == "cy", "最大特征方向必须是纯 cy"
    assert top["loads"]["cy"] == pytest.approx(1.0)
    assert top["px_equiv"] == pytest.approx(0.16, rel=1e-6), \
        f"手算 0.16px,实际 {top['px_equiv']}"

    # 纯单参数方向:分支一(k2/k3)不火、分支二(≥2 个像素参数)不火、
    # 分支三(p1/p2 可见)不火 → 规则表一次都不该开火
    assert section["advice"] == []

    # 经验谱与载荷并排表:结构在盘、数值有限
    empirical = section["empirical_spectrum"]
    assert 0.0 <= empirical["cosine_theory_vs_bootstrap"] <= 1.0
    assert 0.0 <= empirical["cosine_theory_vs_slices"] <= 1.0
    assert len(empirical["loads_table"]) == 8
    for row in empirical["loads_table"]:
        assert all(np.isfinite(row[k]) for k in ("theory_raw", "bootstrap", "slices"))
    # 近零参数像素当量:σ=1e-5 的 p1 投影位移远低于 0.05px 影响线
    assert section["px_equivalents"]["p1_px_equiv"] < 0.05


def test_raw_direction_roundtrip_bit_exact():
    """无量纲还原恒等式:v_raw ÷ scale == v_dimless 逐位还原(2 的幂刻度
    下乘除是精确的浮点移位,不许有任何损耗)——原始坐标与无量纲坐标
    往返一致的契约由本例锁定。"""
    rng = np.random.default_rng(5)
    v_dimless = rng.normal(size=8)
    scale = np.array([1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0])
    v_raw = sens._raw_direction(v_dimless, scale)
    assert np.array_equal(v_raw / scale, v_dimless)


# ---------------------------------------------------------------------------
# Test:uniqueness 如实透传(findings 结论 3)
# ---------------------------------------------------------------------------

def test_uniqueness_passthrough_as_is(tmp_path):
    """uniqueness 三个数全部原样透传不许改写:bootstrap 最大位移当量
    1.9943 逐位、多起点位移与判定原样;报告无 source_separation 段时
    pose_share_max 如实给 None;固定声明必须点名「解不唯一」。"""
    (section, _detail), _world = _run_quick(tmp_path)
    uniq = section["uniqueness"]
    assert uniq["bootstrap_max_disp_px"] == 1.9943
    assert uniq["multistart_disp_px"] == 0.205
    assert uniq["multistart_verdict"] == "inconsistent"
    assert uniq["pose_share_max"] is None
    assert "解不唯一" in uniq["note"]


# ---------------------------------------------------------------------------
# Test:建议规则表三分支(直接对 build_advice 喂构造输入)
# ---------------------------------------------------------------------------

def _dir(rank: int, loads: dict, px: float) -> dict:
    return {"rank": rank, "eigval": 1.0, "loads": loads,
            "loads_dimless": loads, "px_equiv": px}


def test_advice_branch1_edge_poses(tmp_path):
    """分支一:最大方向载荷集中在 k2/k3(≥0.5)且有外推区间 → 建议补
    够到画面四角的边缘姿态,文案带姿态数与目标 r,末尾必带「不自动执行」
    声明(裁定③)。"""
    cfg = CalibrationConfig()
    coverage = {"train": {"coverage": {"extrapolation_interval": [0.685, 0.797]}}}
    advice = sens.build_advice(
        [_dir(1, {"k2": 0.55, "k3": 0.25, "fx": 0.20}, 0.4)],
        {"p1": None, "p2": None}, coverage, cfg)
    assert len(advice) >= 1
    first = advice[0]
    assert "k2" in first["direction"] and "k3" in first["direction"]
    assert "0.685" in first["evidence"] and "0.797" in first["evidence"]
    assert f"补 {cfg.analyze.what_if_n_poses} 个" in first["advice"]
    assert f"{cfg.analyze.what_if_target_r}" in first["advice"]
    assert "不自动执行" in first["advice"]


def test_advice_branch1_coverage_unknown():
    """分支一的覆盖判据退化:residual_structure 段缺席(传 None)→
    文案必须如实说「覆盖判据未知」并指引先补覆盖足迹,不许装作知道。"""
    advice = sens.build_advice(
        [_dir(1, {"k2": 0.7, "k3": 0.1, "fx": 0.2}, 0.4)],
        {"p1": None, "p2": None}, None, CalibrationConfig())
    assert "覆盖判据未知" in advice[0]["advice"]
    assert "不自动执行" in advice[0]["advice"]


def test_advice_branch2_corner_diversity():
    """分支二:像素参数间强耦合(fx/fy 合计 ≥0.5 且各 ≥0.15)→ 建议
    补转角多样性;分支一(k2/k3 <0.5)不火,分支顺序不串。"""
    advice = sens.build_advice(
        [_dir(1, {"fx": 0.40, "fy": 0.35, "k2": 0.25}, 0.09)],
        {"p1": None, "p2": None}, None, CalibrationConfig())
    assert len(advice) == 1
    assert "转角" in advice[0]["advice"]
    assert "不自动执行" in advice[0]["advice"]


def test_advice_branch3_nearzero_untreated():
    """分支三:p1/p2 在载荷表可见(≥0.05)但 σ 级像素当量低于 0.05px
    影响线 → 判「不治」,不拿相对百分比说事(findings 结论 1 反面教材)。"""
    advice = sens.build_advice(
        [_dir(1, {"k2": 0.60, "p1": 0.08, "fx": 0.32}, 0.3)],
        {"p1": 0.009, "p2": 0.004}, None, CalibrationConfig())
    last = advice[-1]
    assert "不治" in last["advice"]
    assert "0.0090px" in last["evidence"] and "0.0040px" in last["evidence"]
    assert "不自动执行" in last["advice"]


# ---------------------------------------------------------------------------
# Test:what-if 档位与合成声明(裁定②)
# ---------------------------------------------------------------------------

def test_quick_what_if_is_none_with_empty_detail(tmp_path):
    """quick 档:what_if 必须是 None + 固定文案说明为什么不做什么档做;
    明细里合成位姿数组为空(quick 不许偷偷做合成预演)。"""
    (section, detail), _world = _run_quick(tmp_path)
    assert section["what_if"] is None
    assert "quick 档" in section["what_if_note"]
    assert "full" in section["what_if_note"]
    assert detail["synthetic_rvecs"].shape == (0, 3)
    assert detail["synthetic_tvecs"].shape == (0, 3)
    assert detail["synthetic_r_max"].shape == (0,)


def test_full_what_if_synthetic_declaration(tmp_path):
    """full 档:what-if 产出且合成声明必须白纸黑字写「合成」「非实拍」
    (合成假设不是实测,证据链性质不许含糊);压缩比全部有限且 ≤1
    (往信息矩阵里加信息,边际化标准差不许变大)。"""
    world = _write_synth_world(tmp_path)
    cfg = dataclasses.replace(
        CalibrationConfig(),
        analyze=dataclasses.replace(AnalyzeConfig(), profile="full"))
    board = BoardSpec(cols=_COLS, rows=_ROWS, square_size_mm=_SQUARE_MM)
    section, detail = run_sensitivity(
        "m1", world["report"], world["fisher"], world["boot"], world["slices"],
        world["model"], build_object_points(board),
        np.zeros(3), np.array([0.0, 0.0, 800.0]), None, cfg)

    what_if = section["what_if"]
    assert what_if is not None, "full 档必须产出 what-if 补拍预演"
    assert "合成" in what_if["synthetic"] and "非实拍" in what_if["synthetic"]
    assert section["what_if_note"] is None
    assert what_if["n_poses_selected"] >= 1
    assert len(what_if["compression_ratio"]) == 8
    for name, ratio in what_if["compression_ratio"].items():
        assert np.isfinite(ratio) and ratio <= 1.0 + 1e-12, \
            f"{name} 压缩比 {ratio} 非法:加信息不许放大标准差"
    assert detail["synthetic_rvecs"].shape[0] == what_if["n_poses_selected"]


def test_what_if_f_merge_identity(monkeypatch):
    """F 合并恒等式:基准 F0 = 4F、预演补 1 份 F → F_new = 5F,逐参数
    压缩比恰为 sqrt(4/5)(线性代数恒等式,rel=1e-9 量级)。

    Fisher 用任取的对称正定矩阵替身:零畸变真值下单平面视角的真实
    边际化信息天然奇异(内参与外参的经典耦合,阶段四已知的退化),
    恒等式验证的是 _what_if_block 的合并数学(F0 当不透明输入,
    F_new = F0 + ΣF_i),替身反而把被测对象隔离得更干净;
    真实 marginalized_fisher 路径由 full 档端到端例覆盖。"""
    model = create_model("m1")
    params = _m1_params_vec(model)
    objp = build_object_points(BoardSpec(cols=_COLS, rows=_ROWS,
                                         square_size_mm=_SQUARE_MM))
    rvec = np.array([0.05, -0.03, 0.01])
    tvec = np.array([5.0, -4.0, 800.0])
    rng = np.random.default_rng(9)
    m = rng.normal(size=(8, 8))
    f1 = m @ m.T + 8.0 * np.eye(8)

    calls: list[int] = []

    def _fake_fisher(model_, obs_, params_, rvecs_, tvecs_):
        calls.append(len(rvecs_))
        return f1.copy(), None, {}

    monkeypatch.setattr(sens, "marginalized_fisher", _fake_fisher)
    monkeypatch.setattr(
        sens, "_synthesize_what_if_poses",
        lambda *a, **k: ([(rvec, tvec, 0.9)], 1))
    section, detail = sens._what_if_block(
        model, objp, params, 4.0 * f1, CalibrationConfig())

    assert calls == [1], "每个合成位姿恰好调一次 marginalized_fisher"
    assert detail["synthetic_rvecs"].shape == (1, 3)
    expected = float(np.sqrt(4.0 / 5.0))
    for name, ratio in section["compression_ratio"].items():
        assert ratio == pytest.approx(expected, rel=1e-9), \
            f"{name} 压缩比 {ratio} vs 恒等式 {expected}"


# ---------------------------------------------------------------------------
# Test:源码红线(D-10 单一 FIM 组装源)
# ---------------------------------------------------------------------------

def test_source_single_fim_source_red_line():
    """源码红线:FIM 组装只许调 noise.fisher.marginalized_fisher,本模块
    自己写的只有线性代数(经验谱的协方差特征分解);绝不 import OpenCV、
    绝不手写 JᵀJ 组装。"""
    source = inspect.getsource(sens)
    assert "marginalized_fisher" in source, "FIM 组装必须只调既有接口"
    assert "import cv2" not in source, "analysis 层绝不 import OpenCV"
    for banned in ("make_jacobian_fn", "csr_matrix", "projectPoints",
                   "Rodrigues", "solvePnP"):
        assert banned not in source, f"analysis 层不许出现 {banned}"
