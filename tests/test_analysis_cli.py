"""calib analyze 命令体的行为测试(monkeypatch.chdir 造假上游树,进程内直调)。

与 test_cli_main.py 同一体例:命令函数是普通函数,直接调用;工作目录切到
tmp_path 下的临时树,产物落 tmp,真实 output/ 不受影响。上游六轮(corners /
noise / calibration / fisher / evaluate + 数据清单)各放最小文件——其中预算
底账(slices.npz 与 evaluate.json 落账统计量)从同一组数组算出,复核必过
(手法同 test_analysis_budget.py 的"自洽合成底账")。

锁的行为:最小上游树跑通且 budget 段有值、其余五段显式缺位;铁律违例走
失败契约(退出码 1,不套逐块兜底);缺上游 / 失败轮当场中文报错。
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import typer

import calib.cli.main as cli_main
from calib.analysis.artifacts import ANALYZE_KEYS
from calib.cli.main import analyze
from calib.models import create_model

# 合成 m2 参数"真值"与逐组步长(与 test_analysis_budget.py 同一套:
# 8 组等差排开,样本标准差 = 步长 × √6 可手算;cx/cy 按 640×480 画面中心,
# 与 conftest 合成棋盘夹具的默认内参几何一致)
_BASE_PARAMS = {
    "fx": 800.0, "fy": 810.0, "cx": 320.0, "cy": 240.0,
    "k1": -0.1, "k2": 0.05, "p1": 0.001, "p2": -0.001, "k3": 0.01,
}
_STEPS = {
    "fx": 1.0, "fy": 1.2, "cx": 0.8, "cy": 0.9,
    "k1": 2e-4, "k2": 1e-4, "p1": 5e-5, "p2": 3e-5, "k3": 4e-5,
}


def _build_upstream_tree(tmp_path: Path) -> Path:
    """最小上游产物树:六轮各放最小文件,预算底账自洽(复核必过)。

    返回 evaluate 轮目录(失败轮测试要改它的 evaluate.json)。
    """
    model = create_model("m2")
    names = list(model.param_names)
    n_groups, n_views, n_points = 8, 4, 5

    # ---- 预算底账:落账比值与 slices.npz 从同一组数组算出,复核必过 ----
    rng = np.random.default_rng(7)
    delta = 0.02 + 0.01 * rng.random((n_views, n_points, 2))
    uv_base = rng.uniform(100.0, 600.0, (n_views, n_points, 2))
    group_uv = uv_base[None] + np.arange(n_groups)[:, None, None, None] * delta[None]
    meas_rows = np.std(group_uv, axis=0, ddof=1).ravel()
    bound_rows = 0.5 * meas_rows  # 下界是实测的一半 → 比值 2.0,远高于硬线
    group_bound_px = np.broadcast_to(bound_rows, (n_groups, bound_rows.size)).copy()
    group_params = np.array(
        [[_BASE_PARAMS[n] + g * _STEPS[n] for n in names] for g in range(n_groups)]
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

    def _dump_json(path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # ---- config(其余键全有默认值,analyze 档位默认 quick)----
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "board.yaml").write_text(
        "board:\n  board_type: chessboard\n  cols: 11\n  rows: 8\n"
        "  square_size_mm: 25.0\n  unit: mm\n", encoding="utf-8")
    (config_dir / "calibration.yaml").write_text(
        "data:\n  train: input_data/train\n  test: input_data/test\n",
        encoding="utf-8")

    # ---- ingest(数据清单)----
    manifest_path = (tmp_path / "output" / "ingest" / "run_20260101_000000"
                     / "manifest.json")
    _dump_json(manifest_path, {"images": []})

    # ---- corners(角点双 npz + 血缘 meta 指向同轮清单)----
    corners_run = tmp_path / "output" / "corners" / "run_20260101_000001"
    corners_run.mkdir(parents=True)
    np.savez(corners_run / "corners_train_sb.npz",
             image_ids=np.array(["img_00"]), uv=np.zeros((1, n_points, 2)))
    np.savez(corners_run / "corners_test_sb.npz",
             image_ids=np.array(["img_t0"]), uv=np.zeros((1, n_points, 2)))
    _dump_json(corners_run / "corners_train_sb.meta.json", {
        "lineage": {"upstream": {"manifest.json": {"path": str(manifest_path)}}},
    })

    # ---- noise(检测器级 σ 的唯一入口 + 连拍逐角点明细)----
    noise_run = tmp_path / "output" / "noise" / "run_20260101_000002"
    noise_run.mkdir(parents=True)
    _dump_json(noise_run / "noise.json", {
        "schema_version": "noise-v1",
        "failure_note": None,
        "sigma_detector_corner": 0.008708,
    })
    np.savez(noise_run / "temporal_sigma_per_corner.npz",
             pose_ids=np.array(["p0", "p1"]),
             sigma_per_corner_all=np.zeros((2, n_points, 2)),
             sources=np.array(["temporal", "temporal"]))

    # ---- calibration(选定模型三件套;位姿与合成棋盘默认几何一致)----
    calib_run = (tmp_path / "output" / "calibration" / "run_20260101_000003"
                 / "m2__scipy")
    calib_run.mkdir(parents=True)
    _dump_json(calib_run / "result.json", {"model_id": "m2", "converged": True})
    np.savez(calib_run / "extrinsics.npz",
             image_ids=np.array(["img_00", "img_01"]),
             rvecs=np.array([[0.0, 0.05, 0.0], [0.0, 0.06, 0.01]]),
             tvecs=np.array([[0.0, 0.0, 800.0], [1.0, -2.0, 810.0]]))
    np.savez(calib_run / "residuals.npz", residuals=np.zeros((2, n_points, 2)))

    # ---- fisher(crlb.json + 选定模型特征谱明细)----
    fisher_run = tmp_path / "output" / "fisher" / "run_20260101_000004"
    fisher_run.mkdir(parents=True)
    _dump_json(fisher_run / "crlb.json",
               {"schema_version": "crlb-v1", "failure_note": None})
    np.savez(fisher_run / "fisher_m2.npz",
             eigvals_cov=np.arange(9, dtype=np.float64) + 1.0,
             eigvecs_cov=np.eye(9))

    # ---- evaluate(json 主契约 + 实验底账双 npz)----
    evaluate_run = tmp_path / "output" / "evaluate" / "run_20260101_000005"
    (evaluate_run / "experiments").mkdir(parents=True)
    np.savez(evaluate_run / "experiments" / "slices.npz",
             m2__group_uv=group_uv,
             m2__group_bound_px=group_bound_px,
             m2__group_params=group_params,
             m2__param_names=np.array(names))
    np.savez(evaluate_run / "experiments" / "bootstrap.npz",
             m2__rep=np.arange(5),
             m2__params=np.zeros((5, len(names))))
    _dump_json(evaluate_run / "evaluate.json", {
        "schema_version": "evaluate-v1",
        "failure_note": None,
        "lineage": {"round_id": "run_eval_test"},
        "selection": {"selected": "m2"},
        "slices": {"models": {"m2": {
            "crlb_alert": {
                "per_param": per_param,
                "lines": {"hard": 0.4207, "soft": 0.5564},
            },
            "crlb_se_mean": {n: 0.001 * (i + 1) for i, n in enumerate(names)},
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
    })
    return evaluate_run


def _latest_analyze_run(tmp_path: Path) -> Path:
    runs = sorted((tmp_path / "output" / "analyze").glob("run_*"))
    assert len(runs) == 1, f"应恰好产出一个 analyze 运行目录,实际 {len(runs)}"
    return runs[0]


def test_analyze_minimal_tree_runs_budget_and_leaves_five_sections_none(
        tmp_path, monkeypatch):
    """最小上游树直调 analyze():正常返回(退出码 0);analyze.json 11 键
    全在,budget 段有值且 pass,其余五段显式 None 并进 meta.missing_sections;
    明细旁车 budget.npz 与证据图 budget_bars.png 落盘。"""
    _build_upstream_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    analyze()  # 不抛即成功(直调无 typer.Exit = 退出码 0)

    out_dir = _latest_analyze_run(tmp_path)
    doc = json.loads((out_dir / "analyze.json").read_text(encoding="utf-8"))
    assert doc["schema_version"] == "analyze-v1"
    assert set(doc) == set(ANALYZE_KEYS), "analyze.json 顶层契约键必须 11 个不少"

    budget = doc["budget"]
    assert isinstance(budget, dict) and budget["verdict"] == "pass"
    assert budget["rows"] and budget["recheck"]
    assert budget["model_id"] == "m2"
    for key in ("residual_structure", "test_xyz", "sensitivity",
                "corner_opt", "anisotropy"):
        assert doc[key] is None, f"{key} 段应显式缺位(None),不是空字典"
    assert set(doc["meta"]["missing_sections"]) == {
        "residual_structure", "test_xyz", "sensitivity", "corner_opt", "anisotropy"}
    assert doc["meta"]["selected_model"] == "m2"
    assert (out_dir / "analysis" / "budget.npz").is_file()
    assert (out_dir / "figures" / "budget_bars.png").is_file()
    # 血缘:上游六轮全部登记(防串轮的原始证据)
    upstream = doc["lineage"]["upstream"]
    for name in ("evaluate.json", "experiments/slices.npz", "experiments/bootstrap.npz",
                 "corners_train_sb.npz", "corners_test_sb.npz", "manifest.json",
                 "noise.json", "crlb.json", "m2__scipy/result.json"):
        assert name in upstream, f"血缘缺上游 {name}"


def test_analyze_budget_failure_contract(tmp_path, monkeypatch):
    """budget 内部铁律违例(AssertionError)必须冒泡:命令抛 typer.Exit
    退出码 1,analyze.json 落失败契约(failure_note 非空中文、六段全 None)
    ——刻意不套逐块兜底,坏结论不许静默出表。"""
    _build_upstream_tree(tmp_path)
    monkeypatch.chdir(tmp_path)

    def _boom(*_args, **_kwargs):
        raise AssertionError("预算铁律违例(测试注入)")

    monkeypatch.setattr(cli_main, "run_budget", _boom)
    with pytest.raises(typer.Exit) as exc_info:
        analyze()
    assert exc_info.value.exit_code == 1

    out_dir = _latest_analyze_run(tmp_path)
    doc = json.loads((out_dir / "analyze.json").read_text(encoding="utf-8"))
    assert doc["failure_note"], "失败契约的 failure_note 必须非空"
    assert "失败" in doc["failure_note"]
    for key in ("budget", "residual_structure", "test_xyz",
                "sensitivity", "corner_opt", "anisotropy"):
        assert doc[key] is None, f"失败契约里 {key} 段必须是 None"
    assert doc["lineage"]["upstream"], "失败报告同样要带全血缘"


def test_analyze_missing_upstream_raises_chinese(tmp_path, monkeypatch):
    """缺任一上游文件:ValueError 且消息以「上游产物缺失:」开头、
    点名该先跑哪条命令。"""
    evaluate_run = _build_upstream_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    (evaluate_run.parents[1] / "noise" / "run_20260101_000002" / "noise.json").unlink()
    with pytest.raises(ValueError) as exc_info:
        analyze()
    message = str(exc_info.value)
    assert message.startswith("上游产物缺失:")
    assert "calib noise" in message


def test_analyze_rejects_failed_round_upstream(tmp_path, monkeypatch):
    """上游是失败轮(evaluate.json failure_note 非空):当场中文拒绝,
    不拿空数据做对账。"""
    evaluate_run = _build_upstream_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    doc = json.loads((evaluate_run / "evaluate.json").read_text(encoding="utf-8"))
    doc["failure_note"] = "评估中途失败(测试注入)"
    (evaluate_run / "evaluate.json").write_text(
        json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        analyze()
    assert "失败轮" in str(exc_info.value)


def test_analyze_signature_zero_params_and_lineage_covers_rounds():
    """D-12 源码断言:命令签名零算法参数;上游六轮全部登记进
    make_lineage 的 upstream 字典(血缘一次登记防串轮)。"""
    assert list(inspect.signature(analyze).parameters) == []
    source = inspect.getsource(cli_main.analyze)
    assert "make_lineage" in source
    for name in ("evaluate.json", "experiments/slices.npz",
                 "experiments/bootstrap.npz", "corners_train_sb.npz",
                 "corners_test_sb.npz", "manifest.json", "noise.json",
                 "crlb.json", "temporal_sigma_per_corner.npz",
                 "fisher_{model_id}.npz", "{model_id}__scipy/result.json"):
        assert name in source, f"血缘登记缺上游 {name}"
