"""优化器对比(16 组合网格 + 推荐配置 + 多起点)的单元测试(小样合成数据)。

小样证明什么(真实数据 640 视图的试车在 tests/test_optimizer_realdata.py):
- optimizer.json 契约(Task 1):合法报告 save → load 逐键还原;缺键当场
  中文报错;失败契约结构键全且中文说明强制非空;私有明细键转 npz 旁车、
  json 只留引用,npz 读回强制 allow_pickle=False;
- 16 组合网格构造(Task 2,D-13/D-14):组合行键一个不少(含 init_rms 与
  f_scale_actual)、单组合失败转契约占行不中断、推荐规则"最小 RMS 胜出、
  1e-3 相对差内并列取更简核";
- 多起点扰动一致性(Task 3,D-15):扰动种子确定性(同种子重跑逐位一致)、
  双容差判定缺一不可(RMS 一致但参数位移超限也判不一致)、卡坏解照实记录。

合成观测集用 conftest 的 make_true_views(已知真值内参的合成棋盘角点,
6 姿态 × 2 张),配置对象直构造绕过 yaml(体例照 test_noise_injection.py);
并行上限写 1 走串行回退,monkeypatch 才能生效。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calib.evaluation.artifacts import (
    OPTIMIZER_KEYS,
    SCHEMA_VERSION,
    load_optimizer_report,
    make_failure_optimizer,
    save_optimizer_report,
)

# ---------------------------------------------------------------------------
# Task 1:optimizer.json 契约(合法往返 / 缺键中文报错 / 失败契约 / npz 旁车)
# ---------------------------------------------------------------------------

def _valid_report() -> dict:
    """一份键全的合法报告(数值随意,契约测试只看结构与往返一致)。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "created": "2026-10-19T00:00:00+00:00",
        "lineage": {"round_id": "run_test"},
        "recommended_solver": {"m1": {"loss": "linear"}},
        "combos": [{"name": "m1__scipy", "model": "m1", "solver": "scipy",
                    "loss": "linear", "n_views": 12, "rms_overall": 0.1,
                    "converged": True, "failure_note": None,
                    "solver_status": 1, "init_rms": 0.2,
                    "f_scale_actual": None}],
        "multistart": None,
        "meta": {"n_combos": 1, "n_failed": 0},
        "failure_note": None,
    }


def test_report_roundtrip_preserves_all_keys(tmp_path):
    """合法报告 save → load:顶层契约键逐键还原,数值逐位一致。"""
    report = _valid_report()
    lineage = {"round_id": "run_test", "created": report["created"]}
    paths = save_optimizer_report(tmp_path, report, lineage)
    assert paths["optimizer_json"] == tmp_path / "optimizer.json"
    assert paths["detail_npz"] is None  # 没给明细键就不该有 npz

    loaded = load_optimizer_report(tmp_path)
    doc = loaded["report"]
    assert loaded["detail"] is None
    for key in OPTIMIZER_KEYS:
        assert key in doc, f"读回的报告缺契约键:{key}"
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["recommended_solver"] == {"m1": {"loss": "linear"}}
    assert doc["combos"][0]["rms_overall"] == 0.1
    assert doc["combos"][0]["init_rms"] == 0.2
    assert doc["meta"]["n_combos"] == 1
    assert doc["failure_note"] is None
    # 血缘由 save 强制落位(参数传入的那份)
    assert doc["lineage"]["round_id"] == "run_test"


def test_missing_contract_key_raises_chinese(tmp_path):
    """缺契约键:写盘当场中文报错,指名缺了哪些键(不许静默落半成品)。"""
    report = _valid_report()
    del report["multistart"]
    with pytest.raises(AssertionError, match="契约键缺失"):
        save_optimizer_report(tmp_path, report, {"round_id": "run_test"})
    # 报错信息里要能直接看到缺的键名
    with pytest.raises(AssertionError, match="multistart"):
        save_optimizer_report(tmp_path / "again", report, {"round_id": "r"})


def test_missing_json_load_raises_chinese(tmp_path):
    """主 json 缺失:读回直接抛中文错,不静默返回空壳。"""
    with pytest.raises(FileNotFoundError, match="optimizer.json 不在"):
        load_optimizer_report(tmp_path)


def test_failure_optimizer_contract_complete_and_note_forced():
    """失败契约:结构键一个不少、数值字段 None、中文说明强制非空。"""
    failure = make_failure_optimizer({"round_id": "run_x"}, "网格中途失败:ValueError——初值缺失")
    for key in OPTIMIZER_KEYS:
        assert key in failure, f"失败契约缺结构键:{key}"
    assert failure["failure_note"] and "网格中途失败" in failure["failure_note"]
    assert failure["recommended_solver"] == {}
    assert failure["combos"] == []
    assert failure["multistart"] is None
    # 空说明 = 伪造证据,直接拒绝
    with pytest.raises(ValueError, match="非空的中文失败说明"):
        make_failure_optimizer({}, "   ")


def test_detail_key_goes_to_npz_sidecar_not_json(tmp_path):
    """私有明细键 _optimizer_detail:写盘转 npz 旁车 + 血缘 meta,json 只留引用。"""
    report = _valid_report()
    report["_optimizer_detail"] = {
        "per_start_rms": np.array([0.1, 0.2, 0.3]),
        "start_labels": np.array(["s0", "s1", "s2"]),
    }
    paths = save_optimizer_report(tmp_path, report, {"round_id": "run_npz"})
    assert paths["detail_npz"] == tmp_path / "optimizer_detail.npz"
    # npz 旁车与血缘 meta 都在
    assert (tmp_path / "optimizer_detail.meta.json").is_file()
    with open(tmp_path / "optimizer_detail.meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["file"] == "optimizer_detail.npz"
    assert meta["n_arrays"] == 2
    assert meta["lineage"]["round_id"] == "run_npz"
    # json 里只留引用,明细键本体绝不进 json
    assert "_optimizer_detail" not in report
    loaded = load_optimizer_report(tmp_path)
    assert loaded["report"]["meta"]["detail_npz"] == "optimizer_detail.npz"
    assert loaded["detail"] is not None
    np.testing.assert_array_equal(loaded["detail"]["per_start_rms"],
                                  [0.1, 0.2, 0.3])
    assert list(loaded["detail"]["start_labels"]) == ["s0", "s1", "s2"]


def test_numpy_scalars_are_jsonable(tmp_path):
    """numpy 标量/数组混进报告也能落盘(转原生类型),读回是纯 Python 值。"""
    report = _valid_report()
    report["combos"][0]["rms_overall"] = np.float64(0.25)
    report["combos"][0]["n_views"] = np.int64(12)
    save_optimizer_report(tmp_path, report, {"round_id": "run_json"})
    with open(tmp_path / "optimizer.json", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["combos"][0]["rms_overall"] == 0.25
    assert raw["combos"][0]["n_views"] == 12


# ---------------------------------------------------------------------------
# Task 2:16 组合网格构造 + 推荐规则(D-13/D-14;小样 1 模型 × 2 核)
# ---------------------------------------------------------------------------

from calib.evaluation import optimizer as optimizer_mod
from calib.evaluation.optimizer import (
    recommend_optimization_config,
    run_optimizer_grid,
)
from calib.io.config import (
    BoardSpec,
    EvaluateConfig,
    ScipyConfig,
)
from calib.models import create_model
from calib.solve.baseline import build_object_points
from calib.solve.matrix import combo_dir_name
from calib.solve.observation import ObservationSet
from calib.solve.opencv import solve_with_opencv
from calib.solve.result import (
    compute_result_metrics,
    save_result,
)

_COMBO_ROW_KEYS = ("name", "model", "solver", "loss", "n_views",
                   "rms_overall", "converged", "failure_note",
                   "solver_status", "init_rms", "f_scale_actual")


def _write_corners_inputs(tmp_path, obs: ObservationSet):
    """把合成观测集写成角点 npz + 数据清单(走真实装载路径 load_observation_set)。"""
    corners_path = tmp_path / "corners_train_sb.npz"
    np.savez(corners_path,
             image_ids=np.asarray(obs.image_ids),
             uv=np.asarray(obs.image_points, dtype=np.float64),
             detected=np.ones(len(obs.image_ids), dtype=bool),
             source=np.asarray(["sb"] * len(obs.image_ids)))
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"images": [{"image_id": img, "pose_id": pose,
                               "split": "train"}
                              for img, pose in zip(obs.image_ids,
                                                   obs.pose_ids)],
                   "summary": {"resolution": [640, 480]}}, f)
    return corners_path, manifest_path


@pytest.fixture
def small_grid_setup(make_true_views, tmp_path):
    """小网格地基:6 姿态 × 2 张合成观测集 + 伪标定轮(m1__opencv 初值产物)。

    初值产物按正式口径落盘(save_result 三件套)——网格从文件读回初值,
    正好覆盖 _load_init_result 的装载与行序对齐断言(T-04-02-01)。
    """
    views, _k = make_true_views(n_views=12)  # 6 姿态 × 2 张连拍
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    obs = ObservationSet(
        image_ids=tuple(f"img_{k:02d}" for k in range(12)),
        pose_ids=tuple(f"pose_{k // 2:02d}" for k in range(12)),
        object_points=build_object_points(board),
        image_points=np.asarray(views, dtype=np.float64),
        image_size=(640, 480),
        corner_source=("sb",) * 12,
    )
    corners_path, manifest_path = _write_corners_inputs(tmp_path, obs)

    run_dir = tmp_path / "calibration" / "run_fake"
    model = create_model("m1")
    init = solve_with_opencv(obs, model)
    assert init.converged, f"合成数据上 m1 OpenCV 初值应收敛:{init.failure_note}"
    _rms, _per_view, residuals = compute_result_metrics(
        obs, np.array([init.params[n] for n in model.param_names]),
        init.rvecs, init.tvecs, model)
    save_result(run_dir / "m1__opencv", init, residuals=residuals)

    eval_cfg = EvaluateConfig(models=("m1",),
                              optimizer_losses=("linear", "soft_l1"),
                              evaluate_max_workers=1)  # 串行:monkeypatch 才生效
    return {"obs": obs, "board": board, "corners": corners_path,
            "manifest": manifest_path, "run_dir": run_dir,
            "eval_cfg": eval_cfg}


def test_small_grid_runs_and_rows_are_complete(small_grid_setup):
    """小网格跑通:组合行键一个不少,init_rms / f_scale_actual 有值,
    推荐配置产出且 meta 记 linear 参照(鲁棒核的实用收敛参照落了账)。"""
    s = small_grid_setup
    report = run_optimizer_grid(s["corners"], s["manifest"], s["board"],
                                ScipyConfig(), s["eval_cfg"], s["run_dir"])
    assert report["schema_version"] == "optimizer-v1"
    assert report["meta"]["n_combos"] == 2
    assert report["meta"]["calibration_run"] == "run_fake"
    assert report["multistart"] is None

    combos = report["combos"]
    assert [row["name"] for row in combos] == [
        combo_dir_name("m1", "scipy", "linear"),
        combo_dir_name("m1", "scipy", "soft_l1"),
    ]
    for row in combos:
        for key in _COMBO_ROW_KEYS:
            assert key in row, f"组合行缺键:{key}(行 {row['name']})"
        assert row["converged"], f"{row['name']} 在干净合成数据上应收敛:{row['failure_note']}"
        assert row["rms_overall"] is not None and row["rms_overall"] < 0.5
        assert row["init_rms"] is not None and row["init_rms"] > 0, \
            "每行必须带共同初值的起点 RMS(Pitfall 8:没有它提升归因不到核)"
        assert row["n_views"] == 12
    # auto 档的 f_scale 实际生效值记进行里(Pitfall 4:口径不是 bug,可查)
    assert combos[1]["f_scale_actual"] is not None and combos[1]["f_scale_actual"] > 0
    # 推荐配置:四键形状对(小样只有 m1),loss 在白名单内
    rec = report["recommended_solver"]
    assert rec["m1"]["loss"] in ("linear", "soft_l1")
    if rec["m1"]["loss"] == "soft_l1":
        assert rec["m1"]["f_scale_px"] is not None
    assert report["meta"]["rms_linear_ref"]["m1"] is not None, \
        "linear 核的参照 RMS 没落账,鲁棒核的实用收敛判定就无从对账"


def test_robust_combo_carries_rms_linear_ref(small_grid_setup, monkeypatch):
    """鲁棒核组合必带 rms_linear_ref(运行时实证):拦一层 spy 看实参,
    漏传会让 is_practically_converged 判定失真(Anti-Patterns 第 5 条)。"""
    s = small_grid_setup
    captured: list[tuple[str, float | None]] = []
    real_solve = optimizer_mod.solve_with_scipy

    def spy(obs, model, cfg, init=None, loss="linear", rms_linear_ref=None):
        captured.append((loss, rms_linear_ref))
        return real_solve(obs, model, cfg, init=init, loss=loss,
                          rms_linear_ref=rms_linear_ref)

    monkeypatch.setattr(optimizer_mod, "solve_with_scipy", spy)
    run_optimizer_grid(s["corners"], s["manifest"], s["board"],
                       ScipyConfig(), s["eval_cfg"], s["run_dir"])
    robust_calls = [(loss, ref) for loss, ref in captured if loss != "linear"]
    assert robust_calls, "鲁棒核组合没有被调到,测试装配有误"
    assert all(ref is not None for _loss, ref in robust_calls), \
        f"鲁棒核调用的 rms_linear_ref 缺失:{robust_calls}"
    linear_calls = [ref for loss, ref in captured if loss == "linear"]
    assert linear_calls == [None], "linear 核自己不该带参照(它就是参照)"


def test_combo_failure_becomes_contract_row_not_crash(small_grid_setup,
                                                      monkeypatch):
    """单组合异常转失败契约占行:soft_l1 注入爆炸,linear 照常收敛,
    网格整体照常返回(一个组合病了不拖死整批,T-04-02-03)。"""
    s = small_grid_setup
    real_solve = optimizer_mod.solve_with_scipy

    def boom(obs, model, cfg, init=None, loss="linear", rms_linear_ref=None):
        if loss != "linear":
            raise RuntimeError("人为注入的求解爆炸")
        return real_solve(obs, model, cfg, init=init, loss=loss,
                          rms_linear_ref=rms_linear_ref)

    monkeypatch.setattr(optimizer_mod, "solve_with_scipy", boom)
    report = run_optimizer_grid(s["corners"], s["manifest"], s["board"],
                                ScipyConfig(), s["eval_cfg"], s["run_dir"])
    by_loss = {row["loss"]: row for row in report["combos"]}
    assert by_loss["linear"]["converged"] is True
    failed = by_loss["soft_l1"]
    assert failed["converged"] is False
    assert failed["rms_overall"] is None and failed["f_scale_actual"] is None
    assert failed["failure_note"] and "未跑成" in failed["failure_note"] \
        and "RuntimeError" in failed["failure_note"]
    assert failed["name"] == combo_dir_name("m1", "scipy", "soft_l1")
    assert report["meta"]["n_failed"] == 1


def test_missing_opencv_init_raises_chinese(small_grid_setup, tmp_path):
    """初值产物缺失:D-14 锁死共同初值,读不回来就中文报错,不许换初值凑数。"""
    s = small_grid_setup
    empty_run = tmp_path / "calibration" / "run_empty"
    with pytest.raises(ValueError, match="初值产物不可用"):
        run_optimizer_grid(s["corners"], s["manifest"], s["board"],
                           ScipyConfig(), s["eval_cfg"], empty_run)


# ---- 推荐规则(纯函数直测,假组合行构造已知 RMS,两种走向都锁)----

def _fake_row(model: str, loss: str, rms: float | None,
              f_scale: float | None = None, converged: bool = True) -> dict:
    """构造一行已知 RMS 的假组合行(推荐规则的单测不真解算)。"""
    return {"name": combo_dir_name(model, "scipy", loss), "model": model,
            "solver": "scipy", "loss": loss, "n_views": 12,
            "rms_overall": rms, "converged": converged,
            "failure_note": None if converged else "假行:未收敛占位",
            "solver_status": 1 if converged else 0,
            "init_rms": 0.2, "f_scale_actual": f_scale}


def test_recommend_min_rms_wins_even_against_simpler_kernel():
    """走向一:最小 RMS 胜出——linear 虽更简但差 1%(远超并列线),
    soft_l1 以更小的 RMS 胜,且带 f_scale_px。"""
    combos = [
        _fake_row("m1", "linear", 0.1010),
        _fake_row("m1", "soft_l1", 0.1000, f_scale=0.166),
        _fake_row("m1", "huber", 0.1050, f_scale=0.2),
    ]
    rec = recommend_optimization_config(combos, EvaluateConfig(models=("m1",)))
    assert rec["m1"] == {"loss": "soft_l1", "f_scale_px": 0.166}


def test_recommend_tie_prefers_simpler_kernel():
    """走向二:数值并列取简——soft_l1 只比 linear 好 0.05%(千分之一并列线
    之内),判并列,取更简的 linear(更少非线性加权,结论更可辩护)。"""
    combos = [
        _fake_row("m1", "linear", 0.1000),
        _fake_row("m1", "soft_l1", 0.10005, f_scale=0.166),  # 相对差 5e-4 < 1e-3
    ]
    rec = recommend_optimization_config(combos, EvaluateConfig(models=("m1",)))
    assert rec["m1"] == {"loss": "linear"}, \
        f"并列应取更简核,实际 {rec['m1']}(并列线 1e-3 内不许引入复杂度)"


def test_recommend_skips_unconverged_and_handles_all_failed():
    """未收敛组合不进候选;全体未收敛时记 loss=None + 中文说明,不静默。"""
    combos = [
        _fake_row("m1", "linear", None, converged=False),
        _fake_row("m1", "soft_l1", 0.09, f_scale=0.2),  # 只有它收敛 → 直接胜出
    ]
    rec = recommend_optimization_config(combos, EvaluateConfig(models=("m1",)))
    assert rec["m1"] == {"loss": "soft_l1", "f_scale_px": 0.2}

    all_failed = [_fake_row("m2", "linear", None, converged=False),
                  _fake_row("m2", "cauchy", None, converged=False)]
    rec2 = recommend_optimization_config(all_failed,
                                         EvaluateConfig(models=("m2",)))
    assert rec2["m2"]["loss"] is None
    assert rec2["m2"]["note"].strip(), "全败模型的推荐占位必须带中文说明"

