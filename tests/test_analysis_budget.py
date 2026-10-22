"""误差预算 run_budget 的测试(05-01 ANA-01,体例照 test_analysis_artifacts.py)。

核心手法:合成一份"自洽"的底账——slices.npz 与 evaluate 报告里落账的
统计量比值从同一组数组算出,复核必过;然后按三种破坏方向逐个击破
(落账比值被改 / 逐行实测低于下界 / 统计量跌破硬线),断言每条都当场
中文硬报错。统计量键名是 evaluate 侧 crlb_alert 的落账契约,本地照抄
一份(与 budget.py 各存一份,不跨文件借私有件)。
"""

from __future__ import annotations

import numpy as np
import pytest

from calib.analysis.budget import CRLB_CALIBER_NOTE, SCHEMA_BUDGET, run_budget
from calib.io.config import BoardSpec
from calib.models import create_model
from calib.models.base import transform_to_camera
from calib.solve.baseline import build_object_points

# 检测器级 σ(实测 0.008708px,口径铁律:只认 noise.json 顶层键的量级)
SIGMA = 0.008708
HARD_LINE = 0.4207
SOFT_LINE = 0.5564
_STAT_KEYS = ("重投影位移中位", "重投影位移95分位", "重投影位移最大")

# 合成 m2 参数的"真值"与逐组步长:组参数按 g·步长 等差排开,
# 8 组的样本标准差(ddof=1)= 步长 × sqrt(6),可手算核对
_BASE_PARAMS = {
    "fx": 800.0, "fy": 810.0, "cx": 640.0, "cy": 480.0,
    "k1": -0.1, "k2": 0.05, "p1": 0.001, "p2": -0.001, "k3": 0.01,
}
_STEPS = {
    "fx": 1.0, "fy": 1.2, "cx": 0.8, "cy": 0.9,
    "k1": 2e-4, "k2": 1e-4, "p1": 5e-5, "p2": 3e-5, "k3": 4e-5,
}
REF_RVEC = np.array([0.1, -0.05, 0.02])
REF_TVEC = np.array([10.0, -20.0, 800.0])


def _build_case(tmp_path, *, bound_factor: float = 0.5) -> dict:
    """合成一份自洽底账:8 组 × 4 视图 × 5 角点的重投影 + 逐行下界 +
    evaluate 报告骨架(复核比值从同一组数组算出,必过)。

    bound_factor 是下界相对实测的比例:<1 正常通过(比值 = 1/factor);
    1~2.37 之间只触发逐行铁律(比值仍在硬线上);>2.37 连统计量一起
    跌破硬线。
    """
    model = create_model("m2")
    names = list(model.param_names)
    n_groups, n_views, n_points = 8, 4, 5

    rng = np.random.default_rng(7)
    delta = 0.02 + 0.01 * rng.random((n_views, n_points, 2))
    uv_base = rng.uniform(100.0, 600.0, (n_views, n_points, 2))
    group_uv = uv_base[None] + np.arange(n_groups)[:, None, None, None] * delta[None]

    meas_rows = np.std(group_uv, axis=0, ddof=1).ravel()
    bound_rows = bound_factor * meas_rows
    group_bound_px = np.broadcast_to(bound_rows, (n_groups, bound_rows.size)).copy()
    group_params = np.array(
        [[_BASE_PARAMS[n] + g * _STEPS[n] for n in names] for g in range(n_groups)]
    )

    npz = tmp_path / "slices.npz"
    np.savez(
        npz,
        m2__group_uv=group_uv,
        m2__group_bound_px=group_bound_px,
        m2__group_params=group_params,
        m2__param_names=np.array(names),
    )

    stat_fns = {
        "重投影位移中位": np.median,
        "重投影位移95分位": lambda a: np.percentile(a, 95),
        "重投影位移最大": np.max,
    }
    per_param = {
        name: {"ratio": float(fn(meas_rows) / fn(bound_rows))}
        for name, fn in stat_fns.items()
    }
    crlb_se_mean = {n: 0.5 for n in names}
    crlb_se_mean["fx"] = 0.6
    crlb_se_mean["fy"] = 0.7
    report = {
        "lineage": {"round_id": "run_test"},
        "slices": {"models": {"m2": {
            "crlb_alert": {
                "per_param": per_param,
                "lines": {"hard": HARD_LINE, "soft": SOFT_LINE},
            },
            "crlb_se_mean": crlb_se_mean,
        }}},
        "delivery": {"models": {"m2": {"final_params": dict(_BASE_PARAMS)}}},
        "bootstrap": {"models": {"m2": {
            "param_std": {n: 0.1 * (i + 1) for i, n in enumerate(names)},
        }}},
        "source_separation": {"m2": {
            n: {"var_pose_sampling": (0.04 * (i + 1)) ** 2}
            for i, n in enumerate(names)
        }},
        "optimizer": {"multistart": {"models": {"m2": {"param_disp_px": 0.15}}}},
    }
    return {
        "model": model,
        "report": report,
        "npz": npz,
        "object_points": build_object_points(BoardSpec()),
        "group_uv": group_uv,
        "group_bound_px": group_bound_px,
        "group_params": group_params,
    }


def test_budget_rows_values_and_utilization(tmp_path):
    """预算行取数与派生逐项核对:empirical_std 手算 = 步长×√6,利用率比值
    只给 fx/fy/cx/cy(k1..k3 与 p1/p2 置 None),其余列照抄上游对应键。"""
    case = _build_case(tmp_path)
    section, detail = run_budget(
        "m2", case["report"], case["npz"], SIGMA, case["model"],
        case["object_points"], REF_RVEC, REF_TVEC,
    )

    assert section["schema"] == SCHEMA_BUDGET
    assert section["model_id"] == "m2"
    assert section["sigma_detector_corner"] == SIGMA
    assert section["verdict"] == "pass"
    assert section["n_pointwise_violations"] == 0
    assert section["lines"] == {"hard": HARD_LINE, "soft": SOFT_LINE}
    assert section["sources"] == {
        "evaluate_round": "run_test",
        "n_groups": 8,
        "sigma_source": "noise.json:sigma_detector_corner",
    }

    # 复核块:重算比值 = 2.0(下界是实测的一半),与落账一致
    # (两次独立重算允许浮点末位抖动,与 run_budget 的 isclose 同一松紧)
    assert section["recheck"]["consistent"] is True
    for name in _STAT_KEYS:
        assert section["recheck"]["ratios"][name] == pytest.approx(2.0, rel=1e-12)
        assert section["recheck"]["evaluate_ratios"][name] == pytest.approx(
            section["recheck"]["ratios"][name], rel=1e-12
        )

    # 逐参数行
    rows = section["rows"]
    assert len(rows) == 9
    for i, row in enumerate(rows):
        name = row["param"]
        assert name == case["model"].param_names[i]
        assert row["empirical_std"] == pytest.approx(
            _STEPS[name] * np.sqrt(6.0), rel=1e-12
        )
        assert row["bootstrap_std"] == pytest.approx(0.1 * (i + 1))
        assert row["pose_sampling_std"] == pytest.approx(0.04 * (i + 1))
        assert row["multistart_disp_px"] == pytest.approx(0.15)
        if name in ("fx", "fy", "cx", "cy"):
            se = 0.6 if name == "fx" else (0.7 if name == "fy" else 0.5)
            assert row["utilization_ratio"] == pytest.approx(
                row["empirical_std"] / se
            )
        else:
            assert row["utilization_ratio"] is None
    by_name = {r["param"]: r for r in rows}
    assert by_name["fx"]["crlb_se"] == pytest.approx(0.6)
    assert by_name["fy"]["crlb_se"] == pytest.approx(0.7)
    assert by_name["k1"]["crlb_se"] == pytest.approx(0.5)

    # 明细旁车数组:形状契约(40 行 = 4 视图 × 5 角点 × 2 分量)
    assert detail["meas_rows"].shape == (40,)
    assert detail["bound_rows"].shape == (40,)
    assert detail["ratios_rows"].shape == (40,)
    assert detail["group_params"].shape == (8, 9)
    assert list(detail["param_names"]) == list(case["model"].param_names)


def test_budget_pixel_equivalent_matches_manual_projection(tmp_path):
    """像素当量列必须真是"±1 个标准差扰动的投影位移最大值":同一参考位姿
    下手动投影一遍逐角点位移对照(接线校验:取对参数、取对参考位姿)。"""
    case = _build_case(tmp_path)
    section, _ = run_budget(
        "m2", case["report"], case["npz"], SIGMA, case["model"],
        case["object_points"], REF_RVEC, REF_TVEC,
    )

    pts_cam, _ = transform_to_camera(case["object_points"], REF_RVEC, REF_TVEC)
    model = case["model"]
    base_vec = np.array([_BASE_PARAMS[n] for n in model.param_names])
    for row in section["rows"]:
        unit = np.zeros_like(base_vec)
        unit[model.param_names.index(row["param"])] = 1.0
        p0 = model.project(pts_cam, base_vec)
        p1 = model.project(pts_cam, base_vec + row["empirical_std"] * unit)
        manual = float(np.linalg.norm(p1 - p0, axis=1).max())
        assert row["empirical_std_px"] == pytest.approx(manual, rel=1e-9)


def test_budget_carries_crlb_caliber_note(tmp_path):
    """口径声明必须原样随表(k1..k3 利用率无意义的判定口径不重写第二份)。"""
    case = _build_case(tmp_path)
    section, _ = run_budget(
        "m2", case["report"], case["npz"], SIGMA, case["model"],
        case["object_points"], REF_RVEC, REF_TVEC,
    )
    assert section["crlb_caliber_note"] == CRLB_CALIBER_NOTE
    assert "k1" in section["crlb_caliber_note"]
    assert "无意义" in section["crlb_caliber_note"]


def test_budget_tampered_ratio_raises_drift(tmp_path):
    """落账比值被改 1%:独立重算对不上 = 口径漂移,当场中文硬报错。"""
    case = _build_case(tmp_path)
    key = "重投影位移中位"
    case["report"]["slices"]["models"]["m2"]["crlb_alert"]["per_param"][key][
        "ratio"
    ] *= 1.01
    with pytest.raises(AssertionError) as exc_info:
        run_budget(
            "m2", case["report"], case["npz"], SIGMA, case["model"],
            case["object_points"], REF_RVEC, REF_TVEC,
        )
    message = str(exc_info.value)
    assert "口径漂移" in message
    assert key in message


def test_budget_pointwise_violation_raises(tmp_path):
    """逐行实测低于理论下界(比值仍在硬线上,只触发逐行铁律这一条):
    物理上不可能,当场硬报错并点名口径错配/污染的排查方向。"""
    case = _build_case(tmp_path, bound_factor=1.5)
    with pytest.raises(AssertionError) as exc_info:
        run_budget(
            "m2", case["report"], case["npz"], SIGMA, case["model"],
            case["object_points"], REF_RVEC, REF_TVEC,
        )
    message = str(exc_info.value)
    assert "散布低于理论下界" in message
    assert "40/40" in message  # 每一行都低于
    assert "口径错配" in message
    assert "sigma_detector_corner" in message  # 排查指引第①条点名 σ 口径


def test_budget_stat_below_hard_line_raises(tmp_path):
    """统计量比值跌破硬告警线:同一条判定线,硬报错点名统计量与告警线值。"""
    case = _build_case(tmp_path, bound_factor=10.0)
    with pytest.raises(AssertionError) as exc_info:
        run_budget(
            "m2", case["report"], case["npz"], SIGMA, case["model"],
            case["object_points"], REF_RVEC, REF_TVEC,
        )
    message = str(exc_info.value)
    assert "低于硬告警线" in message
    for name in _STAT_KEYS:  # 三个统计量比值全为 0.1,全部跌破
        assert name in message
    assert "0.4207" in message


def test_budget_rejects_bad_sigma():
    """σ 校验先于一切取数:非正 / 非有限 / 非数一律中文报错,点名
    noise.json 的 sigma_detector_corner 口径。"""
    model = create_model("m2")
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(AssertionError, match="必须是正的有限数"):
            run_budget(
                "m2", {}, "x.npz", bad, model,
                np.zeros((1, 3)), REF_RVEC, REF_TVEC,
            )
    with pytest.raises(AssertionError, match="sigma_detector 不是数"):
        run_budget(
            "m2", {}, "x.npz", "abc", model,
            np.zeros((1, 3)), REF_RVEC, REF_TVEC,
        )


def test_budget_missing_slices_raises_chinese(tmp_path):
    """底账缺失(slices.npz 不在 / 换错模型缺键):当场中文报错指路,
    提示核对 model_id 与 selection.selected。"""
    case = _build_case(tmp_path)
    with pytest.raises(AssertionError, match="slices.npz 不在"):
        run_budget(
            "m2", case["report"], tmp_path / "nope.npz", SIGMA, case["model"],
            case["object_points"], REF_RVEC, REF_TVEC,
        )
    model3 = create_model("m3")
    with pytest.raises(AssertionError) as exc_info:
        run_budget(
            "m3", case["report"], case["npz"], SIGMA, model3,
            case["object_points"], REF_RVEC, REF_TVEC,
        )
    message = str(exc_info.value)
    assert "m3__group_uv" in message
    assert "selection.selected" in message


def test_budget_missing_lineage_raises_chinese(tmp_path):
    """evaluate.json 缺 lineage 段、或 lineage 缺 round_id(旧轮产物/手改
    文件):当场中文报错指路,不许抛裸 KeyError——上游取数一律走 _require。"""
    case = _build_case(tmp_path)
    report_no_lineage = {k: v for k, v in case["report"].items() if k != "lineage"}
    with pytest.raises(AssertionError, match="lineage"):
        run_budget(
            "m2", report_no_lineage, case["npz"], SIGMA, case["model"],
            case["object_points"], REF_RVEC, REF_TVEC,
        )
    report_no_round = dict(case["report"])
    report_no_round["lineage"] = {}
    with pytest.raises(AssertionError, match="round_id"):
        run_budget(
            "m2", report_no_round, case["npz"], SIGMA, case["model"],
            case["object_points"], REF_RVEC, REF_TVEC,
        )


def test_budget_shape_mismatch_raises(tmp_path):
    """形状契约:group_bound_px 与 group_uv 对不上当场报错(逐行下界必须
    与逐角点重投影一一对应)。"""
    case = _build_case(tmp_path)
    model = case["model"]
    np.savez(
        case["npz"],
        m2__group_uv=case["group_uv"],
        m2__group_bound_px=case["group_bound_px"][:, :39],  # 蓄意少一列
        m2__group_params=case["group_params"],
        m2__param_names=np.array(list(model.param_names)),
    )
    with pytest.raises(AssertionError, match="形状"):
        run_budget(
            "m2", case["report"], case["npz"], SIGMA, model,
            case["object_points"], REF_RVEC, REF_TVEC,
        )
