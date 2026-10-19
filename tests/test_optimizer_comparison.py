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
    # 网格返回值直接组齐契约(created / failure_note 由这里给,省得每个
    # 调用方手工补——真实数据轮漏补就被落盘校验拦下);lineage 是唯一
    # 例外:由调用方经 make_lineage 组装,save_optimizer_report 落位
    for key in OPTIMIZER_KEYS:
        if key != "lineage":
            assert key in report, f"网格返回值缺契约键:{key}"
    assert report["failure_note"] is None
    assert report["created"] is not None
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


# ---------------------------------------------------------------------------
# Task 3:多起点扰动一致性(D-15;小样 N=3,串行执行保证注入可控)
# ---------------------------------------------------------------------------

from dataclasses import replace as dc_replace

from calib.evaluation.optimizer import (
    _multistart_n_for,
    _perturbed_init,
    run_multistart,
)
from calib.evaluation.splitting import _seed_for
from calib.io.config import CalibrationConfig
from calib.solve.result import CalibrationResult


def _multistart_cfg(models=("m1",), n=3) -> CalibrationConfig:
    """小样多起点配置:1 个模型 × N 起点、串行(注入按调用序可控)。"""
    base = CalibrationConfig()
    return dc_replace(
        base,
        evaluate=dc_replace(base.evaluate, models=tuple(models),
                            multistart_n=n, multistart_n_fisheye=n,
                            evaluate_max_workers=1),
    )


@pytest.fixture
def small_multistart_setup(make_true_views):
    """小样多起点地基:6 姿态 × 2 张合成观测集(与网格夹具同款拼装)。"""
    views, _k = make_true_views(n_views=12)
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    obs = ObservationSet(
        image_ids=tuple(f"img_{k:02d}" for k in range(12)),
        pose_ids=tuple(f"pose_{k // 2:02d}" for k in range(12)),
        object_points=build_object_points(board),
        image_points=np.asarray(views, dtype=np.float64),
        image_size=(640, 480),
        corner_source=("sb",) * 12,
    )
    return obs, board


def _fake_converged(obs: ObservationSet, model_id: str, params: dict,
                    rms: float) -> CalibrationResult:
    """构造一份收敛的假结果(判定单测用;位姿给个深度 800 的合规摆位,
    让投影位移当量的计算有正常分母)。"""
    n = len(obs.image_ids)
    return CalibrationResult(
        converged=True, failure_note=None, model_id=model_id, solver="scipy",
        solver_meta={"status": 1, "loss": "linear", "f_scale": 1.0},
        params=dict(params), rvecs=np.zeros((n, 3)),
        tvecs=np.tile(np.array([0.0, 0.0, 800.0]), (n, 1)),
        rms_overall=rms, per_view_rms=None,
        image_ids=obs.image_ids, image_size=obs.image_size,
        n_views=n, n_points_per_view=int(obs.object_points.shape[0]),
        init_from=None, lineage={})


def test_multistart_seed_determinism(small_multistart_setup):
    """扰动种子确定性:同配置两次 run_multistart(同种子派生)逐位一致——
    每起点种子 = _seed_for(f"mstart|{模型}|{i}", cfg.seed),重跑不掷新骰子。"""
    obs, board = small_multistart_setup
    cfg = _multistart_cfg(models=("m1",), n=3)
    first = run_multistart(obs, board, cfg)
    second = run_multistart(obs, board, cfg)
    assert first["models"]["m1"]["n_starts"] == 3
    assert first["models"]["m1"] == second["models"]["m1"], \
        "同种子重跑必须逐位一致(随机性只从 cfg.seed 派生)"
    # 种子派生公式与计划文本一致(逐起点可复算)
    for row in first["models"]["m1"]["starts"]:
        assert row["seed"] == _seed_for(f"mstart|m1|{row['start']}", cfg.seed)
    # 每起点的扰动起点彼此不同(扰动真生效了,不是 N 份同一初值)
    model = create_model("m1")
    baseline = solve_with_opencv(obs, model)
    for row in first["models"]["m1"]["starts"]:
        assert row["converged"], f"干净合成数据上起点应收敛:{row['failure_note']}"
        assert row["params"]["fx"] != baseline.params["fx"], \
            "扰动后的起点参数应偏离基线,原样解等于没扰动"


def test_multistart_rms_deviation_verdict_inconsistent(small_multistart_setup,
                                                       monkeypatch):
    """注入一个 RMS 明显偏离的起点:RMS 极差超容差 → inconsistent(双容差
    判定的第一支)。"""
    obs, board = small_multistart_setup
    cfg = _multistart_cfg(models=("m1",), n=3)
    real_solve = optimizer_mod.solve_with_scipy
    calls = {"n": 0}

    def fake_solve(obs_, model, scipy_cfg, init=None, loss="linear",
                   rms_linear_ref=None):
        result = real_solve(obs_, model, scipy_cfg, init=init, loss=loss,
                            rms_linear_ref=rms_linear_ref)
        calls["n"] += 1
        if calls["n"] == 2:   # 第 2 个起点人为抬 0.1 px(容差 0.001 的百倍)
            return dc_replace(result, rms_overall=result.rms_overall + 0.1)
        return result

    monkeypatch.setattr(optimizer_mod, "solve_with_scipy", fake_solve)
    out = run_multistart(obs, board, cfg)["models"]["m1"]
    assert out["n_starts"] == 3 and out["n_converged"] == 3
    assert out["verdict"] == "inconsistent"
    assert out["rms_spread_px"] >= cfg.evaluate.multistart_rms_tol_px
    assert "RMS 极差" in out["note"]
    assert out["bad_solutions"] == []   # 都收敛了:坏解名单应为空


def test_multistart_param_drift_verdict_inconsistent(small_multistart_setup,
                                                     monkeypatch):
    """RMS 一致但参数位移超限也判 inconsistent(双容差缺一不可:
    参数不同 RMS 相同 = 可辨识性问题本身,Anti-Patterns 第 6 条)。"""
    obs, board = small_multistart_setup
    cfg = _multistart_cfg(models=("m1",), n=3)
    baseline = solve_with_opencv(obs, create_model("m1"))
    calls = {"n": 0}

    def fake_solve(obs_, model, scipy_cfg, init=None, loss="linear",
                   rms_linear_ref=None):
        calls["n"] += 1
        params = dict(baseline.params)
        if calls["n"] == 2:   # 第 2 个起点 fx 偏 1%(画面边缘位移远超 0.1 px)
            params["fx"] = params["fx"] * 1.01
        return _fake_converged(obs_, model.model_id, params, rms=0.05)

    monkeypatch.setattr(optimizer_mod, "solve_with_scipy", fake_solve)
    out = run_multistart(obs, board, cfg)["models"]["m1"]
    assert out["n_converged"] == 3
    assert out["rms_spread_px"] == 0.0, "三个起点的 RMS 被构造成全等"
    assert out["param_disp_px"] >= cfg.evaluate.multistart_param_tol_px
    assert out["verdict"] == "inconsistent"
    assert "参数位移当量" in out["note"]


def test_multistart_start_failure_recorded_not_fatal(small_multistart_setup,
                                                     monkeypatch):
    """单起点求解失败转契约记录:坏解进 bad_solutions 点名,其余起点照常
    判定(卡坏解是观察对象不是事故,D-15)。"""
    obs, board = small_multistart_setup
    cfg = _multistart_cfg(models=("m1",), n=3)
    real_solve = optimizer_mod.solve_with_scipy
    calls = {"n": 0}

    def fake_solve(obs_, model, scipy_cfg, init=None, loss="linear",
                   rms_linear_ref=None):
        calls["n"] += 1
        if calls["n"] == 1:   # 第 1 个起点注入爆炸
            raise RuntimeError("人为注入的起点爆炸")
        return real_solve(obs_, model, scipy_cfg, init=init, loss=loss,
                          rms_linear_ref=rms_linear_ref)

    monkeypatch.setattr(optimizer_mod, "solve_with_scipy", fake_solve)
    out = run_multistart(obs, board, cfg)["models"]["m1"]
    assert out["n_starts"] == 3 and out["n_converged"] == 2
    assert len(out["bad_solutions"]) == 1
    assert out["bad_solutions"][0]["start"] == 0
    assert "RuntimeError" in out["bad_solutions"][0]["failure_note"]
    assert out["verdict"] in ("consistent", "inconsistent")  # 剩余 2 起点照常判定


def test_fisheye_model_gets_doubled_start_count(small_multistart_setup,
                                                monkeypatch):
    """m4(鱼眼)起点数取 multistart_n_fisheye(D-15 重点观察对象);
    基线与精化都打假,只数起点数与契约形状。"""
    obs, board = small_multistart_setup
    cfg = _multistart_cfg(models=("m4",), n=3)
    assert cfg.evaluate.multistart_n_fisheye == 3
    assert _multistart_n_for(create_model("m4"), cfg.evaluate) == 3
    assert _multistart_n_for(create_model("m1"), cfg.evaluate) == 3
    # 正式配置下:普通模型 10 次、鱼眼 20 次
    full_ev = CalibrationConfig().evaluate
    assert _multistart_n_for(create_model("m1"), full_ev) == full_ev.multistart_n
    assert _multistart_n_for(create_model("m4"), full_ev) == full_ev.multistart_n_fisheye

    m4_model = create_model("m4")
    m4_params = dict.fromkeys(m4_model.param_names, 0.01)
    m4_params.update(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    fake_baseline = _fake_converged(obs, "m4", m4_params, rms=0.1)

    monkeypatch.setattr(
        optimizer_mod, "solve_with_opencv",
        lambda obs_, model: fake_baseline)
    monkeypatch.setattr(
        optimizer_mod, "solve_with_scipy",
        lambda obs_, model, scipy_cfg, init=None, loss="linear",
        rms_linear_ref=None: _fake_converged(obs_, "m4", dict(init.params),
                                             rms=0.05))
    out = run_multistart(obs, board, cfg)["models"]["m4"]
    assert out["n_starts"] == 3
    assert len(out["starts"]) == 3
    # 扰动按 m4 的参数名单分组执行(k 族是畸变组),不许有参数被静默跳过
    for row in out["starts"]:
        assert set(row["params"]) == set(m4_model.param_names)


def test_perturbation_respects_amplitude_config():
    """扰动幅度按配置生效:同种子下 fx 只在 ±intr_rel 内偏离;未知参数名
    当场中文报错(不静默跳过)。"""
    base = CalibrationConfig()
    ev = dc_replace(base.evaluate, multistart_intr_rel=0.02,
                    multistart_dist_rel=0.10)
    init = _fake_converged(
        ObservationSet(
            image_ids=("a",), pose_ids=("p",),
            object_points=np.zeros((88, 3)), image_points=np.zeros((1, 88, 2)),
            image_size=(640, 480), corner_source=("sb",)),
        "m1", {"fx": 1000.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0,
               "k1": 0.1, "k2": 0.0, "p1": 0.0, "p2": 0.0},
        rms=0.1)
    init = dc_replace(init, rvecs=np.zeros((1, 3)), tvecs=np.ones((1, 3)))
    for trial in range(20):
        rng = np.random.default_rng(trial)
        out = _perturbed_init(init, ev, rng)
        assert abs(out.params["fx"] - 1000.0) <= 1000.0 * ev.multistart_intr_rel
        assert abs(out.params["k1"] - 0.1) <= 0.1 * ev.multistart_dist_rel
        assert out.params["k2"] == 0.0   # 乘性扰动:0 还是 0(不发明偏移)
        assert out.params["fx"] != init.params["fx"] or \
            out.params["fy"] != init.params["fy"] or \
            out.params["cx"] != init.params["cx"]
        # 旋转:旋转向量加了 ±rot_deg 内的扰动,范数不为零
        assert np.linalg.norm(out.rvecs[0]) > 0
        # 平移:范数在 ±trans_rel 内
        assert abs(np.linalg.norm(out.tvecs[0]) - 3**0.5) <= 3**0.5 * ev.multistart_trans_rel + 1e-12

    bad = dc_replace(init, params={"u1": 1.0, "fx": 800.0})
    with pytest.raises(AssertionError, match="不认识参数名"):
        _perturbed_init(bad, ev, np.random.default_rng(0))

