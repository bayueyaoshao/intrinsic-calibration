"""评估交付(evaluate.json 主契约 + 合成数据全链冒烟)的测试。

本文件两段(体例照 test_optimizer_comparison.py 与 test_evaluation_experiments.py):
- 契约单元段(Task 2):evaluate.json 13 键写盘前校验、save → load 往返
  逐键还原、明细数组只进 experiments/*.npz 旁车(json 只留 detail_npz
  引用)、失败契约结构键全、npz 旁车 meta 带血缘;
- 全链冒烟段(Task 3):合成 6 姿态小样一条龙(优化器小网格 → 四实验 →
  全量重训交付 → 来源分离组装 → 判行表与选型 → 契约落盘 → 三图),
  04-06 的命令层只是把这里的调用序列接上取数与中文播报。

合成观测集用 conftest 的 make_true_views(已知真值内参的合成棋盘角点),
配置对象直构造绕过 yaml;并行上限写 1 走串行回退,monkeypatch 才能生效。
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pytest

from calib.evaluation.artifacts import (
    EVALUATE_KEYS,
    SCHEMA_VERSION_EVALUATE,
    load_evaluate_report,
    make_failure_evaluate,
    save_evaluate_report,
)
from calib.evaluation.experiments import (
    bootstrap_pose,
    cv_pose_grouped,
    lopo,
    pose_sampling_variance,
    run_delivery,
    slices_by_index,
)
from calib.evaluation.optimizer import run_optimizer_grid
from calib.evaluation.selection import build_verdict_table, select_model
from calib.io.config import BoardSpec, CalibrationConfig, ScipyConfig
from calib.models import create_model
from calib.solve.observation import load_observation_set
from calib.solve.opencv import solve_with_opencv
from calib.solve.result import compute_result_metrics, save_result
from calib.util.lineage import make_lineage
from calib.visualization.evaluate_figs import (
    plot_cv_errorbars,
    plot_dispersion_compare,
    plot_lopo_pose_errors,
)

# ---------------------------------------------------------------------------
# Task 2:evaluate.json 契约(合法往返 / 缺键中文报错 / 失败契约 / npz 旁车)
# ---------------------------------------------------------------------------


def _minimal_report() -> dict:
    """一份键全的最小报告(各段都是字典,数值随意,契约测试只看结构)。"""
    return {
        "schema_version": SCHEMA_VERSION_EVALUATE,
        "created": "2026-10-19T00:00:00+00:00",
        "lineage": {},
        "optimizer": {"recommended_solver": {"m1": {"loss": "linear"}}},
        "cv": {"models": {"m1": {"mean_heldout_px": 0.1}}},
        "bootstrap": {"models": {"m1": {"max_disp_px": 0.01}}},
        "slices": {"models": {"m1": {"slice_max_disp_px": 0.02}}},
        "lopo": {"models": {"m1": {"mean_lopo_px": 0.12}}},
        "source_separation": {"m1": {"fx": {"pose_share": 0.5}}},
        "selection": {"selected": "m1"},
        "delivery": {"models": {"m1": {"rms_px": 0.1}}},
        "meta": {},
        "failure_note": None,
    }


def _all_details() -> dict:
    """五份私有明细键(含嵌套 {模型: {数组}} 的分组/bootstrap 形态)。"""
    return {
        "_cv_detail": {
            "assignment_round": np.array([0, 0], dtype=np.int64),
            "detail_rms": np.array([0.1, np.nan], dtype=np.float64),
            "detail_pose": np.array(["pose_00", "pose_01"]),
        },
        "_bootstrap_detail": {
            "m1": {
                "rep": np.array([0, 1], dtype=np.int64),
                "params": np.array([[1.0, 2.0], [3.0, 4.0]]),
            },
        },
        "_slice_detail": {
            "m1": {
                "group_index": np.array([0, 1], dtype=np.int64),
                "sigma_used": 0.30,
            },
        },
        "_lopo_detail": {
            "detail_val_pose": np.array(["pose_00", "pose_01"]),
            "detail_rms": np.array([0.2, 0.3], dtype=np.float64),
        },
        "_delivery_detail": {
            "test_model": np.array(["m1", "m1"]),
            "test_image": np.array(["timg_00", "timg_01"]),
            "test_rms": np.array([0.15, 0.16], dtype=np.float64),
        },
    }


def test_evaluate_contract_has_thirteen_keys():
    """13 键契约本身是红线:多一键少一键都算破坏下游消费面,直接显形。"""
    assert len(EVALUATE_KEYS) == 13
    assert SCHEMA_VERSION_EVALUATE == "evaluate-v1"


def test_evaluate_roundtrip_preserves_all_keys(tmp_path):
    """合法报告(带全部五份明细)save → load:顶层契约键逐键还原,数值
    逐位一致;明细数组在 npz 不在 json(json 各段只剩 detail_npz 引用)。"""
    report = _minimal_report()
    for key, arrays in _all_details().items():
        report[key] = arrays
    lineage = {"round_id": "run_test", "created": report["created"]}
    paths = save_evaluate_report(tmp_path, report, lineage)

    assert paths["evaluate_json"] == tmp_path / "evaluate.json"
    assert set(paths["detail_npz"]) == {"cv", "bootstrap", "slices", "lopo", "delivery"}
    for section, npz_path in paths["detail_npz"].items():
        assert npz_path == tmp_path / "experiments" / f"{section}.npz"
        assert npz_path.is_file()

    loaded = load_evaluate_report(tmp_path)
    doc = loaded["report"]
    for key in EVALUATE_KEYS:
        assert key in doc, f"读回的报告缺契约键:{key}"
    assert doc["schema_version"] == SCHEMA_VERSION_EVALUATE
    assert doc["lineage"]["round_id"] == "run_test"
    assert doc["selection"] == {"selected": "m1"}
    assert doc["failure_note"] is None

    # json 各段只留引用:private 明细键被摘走,段内出现 detail_npz 指路
    for section in ("cv", "bootstrap", "slices", "lopo", "delivery"):
        assert doc[section]["detail_npz"] == f"experiments/{section}.npz"
    raw = json.loads((tmp_path / "evaluate.json").read_text(encoding="utf-8"))
    assert "assignment_round" not in raw["cv"], "明细数组不许进 json(汇总只是索引)"
    assert "_cv_detail" not in raw, "私有明细键不许留在顶层"

    # 明细数组逐键还原(嵌套键拍平成 模型__数组名)
    details = loaded["details"]
    assert set(details) == {"cv", "bootstrap", "slices", "lopo", "delivery"}
    assert np.array_equal(details["cv"]["detail_rms"][0], np.float64(0.1))
    assert np.isnan(details["cv"]["detail_rms"][1])
    assert np.array_equal(details["bootstrap"]["m1__params"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert details["slices"]["m1__sigma_used"] == np.float64(0.30)
    assert np.array_equal(details["delivery"]["test_image"], np.array(["timg_00", "timg_01"]))


def test_evaluate_missing_contract_key_raises_chinese(tmp_path):
    """缺契约键:写盘当场中文报错,指名缺了哪些键(不许静默落半成品)。"""
    report = _minimal_report()
    del report["selection"]
    with pytest.raises(AssertionError, match="契约键缺失"):
        save_evaluate_report(tmp_path, report, {"round_id": "run_test"})
    with pytest.raises(AssertionError, match="selection"):
        save_evaluate_report(tmp_path / "again", report, {"round_id": "r"})


def test_evaluate_detail_without_dict_section_raises(tmp_path):
    """明细键有数组、但对应段不是字典:组装顺序错了当场报错(明细旁车必须
    挂在实验段下,悬空明细读回的人找不到目录)。"""
    report = _minimal_report()
    report["lopo"] = None  # 失败段的形态:None 段不该挂明细
    report["_lopo_detail"] = {"detail_rms": np.array([0.1])}
    with pytest.raises(AssertionError, match="lopo"):
        save_evaluate_report(tmp_path, report, {"round_id": "run_test"})


def test_evaluate_failure_contract_complete_and_note_forced():
    """失败契约:结构键一个不少、各实验段 None、中文说明强制非空。"""
    failure = make_failure_evaluate({"round_id": "run_x"}, "评估中途失败:ValueError——上游缺失")
    for key in EVALUATE_KEYS:
        assert key in failure, f"失败契约缺结构键:{key}"
    assert failure["failure_note"] and "评估中途失败" in failure["failure_note"]
    for section in (
        "optimizer",
        "cv",
        "bootstrap",
        "slices",
        "lopo",
        "source_separation",
        "selection",
        "delivery",
    ):
        assert failure[section] is None, f"失败契约的 {section} 段应为 None"
    # 空说明 = 伪造证据,直接拒绝
    with pytest.raises(ValueError, match="非空的中文失败说明"):
        make_failure_evaluate({}, "   ")


def test_evaluate_npz_sidecar_meta_carries_lineage(tmp_path):
    """每个明细 npz 配同名 .meta.json 且血缘必带(证据链不断,T-04-05-04)。"""
    report = _minimal_report()
    report["_cv_detail"] = {"detail_rms": np.array([0.1, 0.2])}
    save_evaluate_report(tmp_path, report, {"round_id": "run_meta"})
    meta_path = tmp_path / "experiments" / "cv.meta.json"
    assert meta_path.is_file()
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["file"] == "experiments/cv.npz"
    assert meta["section"] == "cv"
    assert meta["lineage"]["round_id"] == "run_meta"


def test_evaluate_missing_json_load_raises_chinese(tmp_path):
    """主 json 缺失:读回直接抛中文错,不静默返回空壳。"""
    with pytest.raises(FileNotFoundError, match="evaluate.json 不在"):
        load_evaluate_report(tmp_path)


def test_evaluate_broken_npz_reference_raises_chinese(tmp_path):
    """json 引用了明细 npz 但文件不在:读回中文报错(引用断裂 = 产物不完整,
    静默吞掉等于撒谎)。"""
    report = _minimal_report()
    report["_cv_detail"] = {"detail_rms": np.array([0.1])}
    save_evaluate_report(tmp_path, report, {"round_id": "run_broken"})
    (tmp_path / "experiments" / "cv.npz").unlink()
    with pytest.raises(FileNotFoundError, match="明细旁车"):
        load_evaluate_report(tmp_path)


# ---------------------------------------------------------------------------
# Task 3:合成数据全链冒烟(优化器小网格 → 四实验 → 交付 → 选型 → 契约 → 三图)
# ---------------------------------------------------------------------------

# 实验返回值里不进 json 的键:私有明细键(转 npz 旁车)与行级大表(逐姿态 /
# 逐张明细同样在 npz 数组里,json 只留汇总与引用——D-13 汇总只是索引)
_JSON_EXCLUDED_KEYS = {
    "_cv_detail",
    "_bootstrap_detail",
    "_slice_detail",
    "_lopo_detail",
    "_delivery_detail",
    "per_pose_rows",
    "group_records",
    "rows",
    "test_rows",
}


def _json_section(section: dict) -> dict:
    """实验返回值 → evaluate.json 的段:摘掉明细键,只留汇总(契约形状)。"""
    return {k: v for k, v in section.items() if k not in _JSON_EXCLUDED_KEYS}


def _assert_reports_equal(a, b, path="报告") -> None:
    """递归逐位比较(数组用 np.array_equal,浮点 NaN 视为相等)——
    "同种子重跑逐位一致"的硬断言工具(体例照 test_evaluation_experiments.py)。"""
    if isinstance(a, dict):
        assert isinstance(b, dict), path
        assert set(a) == set(b), f"{path}: 键集合不一致 {set(a) ^ set(b)}"
        for k in a:
            _assert_reports_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_reports_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.shape == b.shape, (
            f"{path}: 数组形状不一致 {a.shape} vs {getattr(b, 'shape', None)}"
        )
        if a.dtype.kind == "f":
            assert np.array_equal(a, b, equal_nan=True), f"{path}: 数组数值不一致"
        else:
            assert np.array_equal(a, b), f"{path}: 数组数值不一致"
    elif isinstance(a, float):
        if np.isnan(a):
            assert isinstance(b, float) and np.isnan(b), f"{path}: NaN 不对称"
        else:
            assert a == b, f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture
def smoke_setup(make_true_views, tmp_path):
    """全链冒烟地基:15 视角合成数据(12 训练 = 6 姿态 × 2 连拍,3 考试)、
    训练/考试两套角点文件与数据清单(走真实装载路径)、伪标定轮
    (m1/m2__opencv 初值产物,网格从文件读回共同初值,D-14)。

    配置小样化:2 模型 × 2 核 × 2 折 × 2 轮 × 4 次 bootstrap;并行上限写 1
    走串行回退(体例照 test_optimizer_comparison.py)。
    """
    views, _k = make_true_views(n_views=15)
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)

    def _write_split(stem, image_ids, pose_ids, uv, split):
        """一套角点 npz + 数据清单(体例照 test_optimizer_comparison.py)。"""
        corners_path = tmp_path / f"corners_{stem}_sb.npz"
        np.savez(
            corners_path,
            image_ids=np.asarray(image_ids),
            uv=np.asarray(uv, dtype=np.float64),
            detected=np.ones(len(image_ids), dtype=bool),
            source=np.asarray(["sb"] * len(image_ids)),
        )
        manifest_path = tmp_path / f"manifest_{stem}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "images": [
                        {"image_id": img, "pose_id": pose, "split": split}
                        for img, pose in zip(image_ids, pose_ids)
                    ],
                    "summary": {"resolution": [640, 480]},
                },
                f,
            )
        return corners_path, manifest_path

    corners_train, manifest_train = _write_split(
        "train",
        [f"img_{k:02d}" for k in range(12)],
        [f"pose_{k // 2:02d}" for k in range(12)],
        np.asarray(views[:12], dtype=np.float64),
        "train",
    )
    corners_test, manifest_test = _write_split(
        "test",
        ["timg_00", "timg_01", "timg_02"],
        ["tpose_00", "tpose_01", "tpose_02"],
        np.asarray(views[12:], dtype=np.float64),
        "test",
    )

    obs_train = load_observation_set(corners_train, manifest_train, board, split="train")
    run_dir = tmp_path / "calibration" / "run_fake"
    for mid in ("m1", "m2"):
        model = create_model(mid)
        init = solve_with_opencv(obs_train, model)
        assert init.converged, f"合成数据上 {mid} OpenCV 初值应收敛:{init.failure_note}"
        _rms, _per_view, residuals = compute_result_metrics(
            obs_train,
            np.array([init.params[n] for n in model.param_names]),
            init.rvecs,
            init.tvecs,
            model,
        )
        save_result(run_dir / f"{mid}__opencv", init, residuals=residuals)

    base = CalibrationConfig()
    cfg = replace(
        base,
        evaluate=replace(
            base.evaluate,
            models=("m1", "m2"),
            optimizer_losses=("linear", "soft_l1"),
            cv_n_folds=2,
            cv_n_rounds=2,
            bootstrap_n=4,
            evaluate_max_workers=1,
            # 训练侧保持每姿态 2 张(2026-10-22 降载前口径):本冒烟考的是
            # 全链接线不断架,小样合成数据上每姿态 1 张(3 行训练)会让 m2
            # 触发"实用收敛"误报(小样数据现象);旋钮语义由
            # test_evaluation_experiments / test_evaluation_split 专项测试另考
            exposures_per_pose=2,
        ),
    )
    return {
        "board": board,
        "cfg": cfg,
        "corners_train": corners_train,
        "manifest_train": manifest_train,
        "corners_test": corners_test,
        "manifest_test": manifest_test,
        "run_dir": run_dir,
    }


def test_full_chain_smoke_optimizer_to_contract(smoke_setup):
    """全链冒烟:优化器小网格 → 四实验 → 全量重训交付 → 来源分离组装 →
    判行表与选型 → evaluate.json 落盘 → 三张判行表图,一条龙跑通不散架。
    04-06 的命令层只是把这里的调用序列接上取数与中文播报。"""
    s = smoke_setup
    models = tuple(s["cfg"].evaluate.models)
    t0 = time.perf_counter()

    # ① 优化器小网格:推荐配置从这里出(D-13:四实验统一用推荐配置)
    grid = run_optimizer_grid(
        s["corners_train"],
        s["manifest_train"],
        s["board"],
        ScipyConfig(),
        s["cfg"].evaluate,
        s["run_dir"],
    )
    assert grid["failure_note"] is None
    optimizer_config = grid["recommended_solver"]
    assert set(optimizer_config) == set(models), "推荐配置应覆盖全部参选模型"

    # ② 四实验(观测集走真实装载路径,split 行序契约在入口内置断言)
    obs_train = load_observation_set(
        s["corners_train"], s["manifest_train"], s["board"], split="train"
    )
    obs_test = load_observation_set(s["corners_test"], s["manifest_test"], s["board"], split="test")
    cv = cv_pose_grouped(obs_train, s["board"], s["cfg"], optimizer_config)
    boot = bootstrap_pose(obs_train, s["board"], s["cfg"], optimizer_config)
    slices = slices_by_index(
        obs_train,
        s["board"],
        s["cfg"],
        optimizer_config,
        {"sigma_detector_corner": 0.30},
    )
    lopo_rep = lopo(obs_train, s["board"], s["cfg"], optimizer_config)
    for m in models:
        assert cv["models"][m]["mean_heldout_px"] is not None
        assert cv["models"][m]["n_failed_folds"] == 0, "干净合成数据上不该有失败折"
        assert boot["models"][m]["max_disp_px"] is not None
        assert slices["models"][m]["slice_max_disp_px"] is not None
        assert lopo_rep["models"][m]["mean_lopo_px"] is not None

    # ③ 全量重训交付(CV 只管选型,交付一律全量重训;test 逐张并排评分)
    delivery = run_delivery(obs_train, obs_test, s["board"], s["cfg"], optimizer_config)
    for m in models:
        entry = delivery["models"][m]
        assert entry["converged"], f"{m} 全量重训在合成数据上应收敛:{entry['failure_note']}"
        assert entry["bic"] is not None and np.isfinite(entry["bic"])
    assert len(delivery["test_rows"]) == 2 * 3, "test 评分行 = 模型数 × 考试照片数"
    assert all(row["rms_px"] is not None for row in delivery["test_rows"])

    # ④ 来源分离组装(填 bootstrap 的 var_slice / pose_sampling_var 占位键,
    # D-07:相减的是方差,分组侧经验标准差先平方)
    source_separation: dict[str, dict] = {}
    stability_summary: dict[str, dict] = {}
    crlb_summary: dict[str, dict] = {}
    for m in models:
        var_slice = {n: float(v) ** 2 for n, v in slices["models"][m]["empirical_std"].items()}
        sep = pose_sampling_variance(boot["models"][m]["var_bootstrap"], var_slice)
        boot["models"][m]["var_slice"] = var_slice
        boot["models"][m]["pose_sampling_var"] = sep
        source_separation[m] = sep
        stability_summary[m] = {
            "max_disp_px": boot["models"][m]["max_disp_px"],
            "slice_max_disp_px": slices["models"][m]["slice_max_disp_px"],
        }
        # CRLB 对照缺席时的兜底(冒烟小样组数少,告警可能算不成;不冒充数值)
        crlb_summary[m] = slices["models"][m]["crlb_alert"] or {
            "n_below_hard": 0,
            "verdict_note": "冒烟小样 CRLB 对照缺席,硬告警按 0 计",
        }

    # ⑤ 判行表 + BIC 写入 + 选型(BIC 在交付里算好,这里只落账,D-10)
    verdicts = build_verdict_table(
        cv["models"],
        stability_summary,
        crlb_summary,
        {},
        {
            "gen_overfit_ratio_max": s["cfg"].evaluate.gen_overfit_ratio_max,
            "stability_disp_px_max": s["cfg"].evaluate.stability_disp_px_max,
        },
    )
    for row in verdicts["verdicts"]:
        row["bic"] = delivery["models"][row["model_id"]]["bic"]
    selection = select_model(verdicts["verdicts"], cv["models"], s["cfg"])
    assert selection["selected"] in models or (
        selection["selected"] is None and selection["reasons"]
    ), "选型结论要么给模型名,要么 None + 逐模型未当选原因"
    assert selection["selection_rule"], "选型规则说明必须随结论落盘(可考)"

    # ⑥ 组装 evaluate 报告落盘(明细转 npz 旁车,json 只留汇总与引用)
    lineage = make_lineage(
        {},
        {
            "corners_train": s["corners_train"],
            "manifest_train": s["manifest_train"],
            "corners_test": s["corners_test"],
            "manifest_test": s["manifest_test"],
        },
        "run_smoke",
    )
    report = {
        "schema_version": SCHEMA_VERSION_EVALUATE,
        "created": lineage["created"],
        "lineage": {},  # save_evaluate_report 落位
        "optimizer": optimizer_config,
        "cv": _json_section(cv),
        "bootstrap": _json_section(boot),
        "slices": _json_section(slices),
        "lopo": _json_section(lopo_rep),
        "source_separation": source_separation,
        "selection": selection,
        "delivery": _json_section(delivery),
        "meta": {"round_id": "run_smoke"},
        "failure_note": None,
    }
    report["_cv_detail"] = cv["_cv_detail"]
    report["_bootstrap_detail"] = boot["_bootstrap_detail"]
    report["_slice_detail"] = slices["_slice_detail"]
    report["_lopo_detail"] = lopo_rep["_lopo_detail"]
    report["_delivery_detail"] = delivery["_delivery_detail"]

    out_dir = s["run_dir"].parent / "evaluate" / "run_smoke"
    save_evaluate_report(out_dir, report, lineage)
    doc = load_evaluate_report(out_dir)["report"]
    for key in EVALUATE_KEYS:
        assert key in doc, f"读回的报告缺契约键:{key}"
    assert doc["failure_note"] is None
    for section in ("cv", "bootstrap", "slices", "lopo", "delivery"):
        assert doc[section]["detail_npz"] == f"experiments/{section}.npz"
    assert doc["lineage"]["upstream"], "血缘 upstream 不能为空(T-04-05-04)"
    # 来源分离:每个参数的位姿占比都是 [0, 1] 内的实数
    for m in models:
        for name, block in source_separation[m].items():
            share = block["pose_share"]
            assert isinstance(share, float) and 0.0 <= share <= 1.0, (
                f"{m}.{name} 的位姿占比越界:{share}"
            )

    # ⑦ 三张判行表图(库函数只管画图;冒烟里要真画出来且非空)
    fig_dir = out_dir / "figures"
    fig_paths = (
        plot_cv_errorbars(cv["models"], fig_dir / "cv_errorbars.png"),
        plot_lopo_pose_errors(lopo_rep, fig_dir / "lopo_pose_errors.png"),
        plot_dispersion_compare(
            boot["models"], slices["models"], fig_dir / "dispersion_compare.png"
        ),
    )
    for p in fig_paths:
        assert p.is_file() and p.stat().st_size > 0, f"冒烟图没画出来:{p}"

    # 冒烟也有时间上限:超 300 秒说明小样配置退化成了正式口径,装配有误
    elapsed = time.perf_counter() - t0
    assert elapsed < 300.0, f"全链冒烟耗时 {elapsed:.1f} 秒,超过 300 秒上限"


def test_full_chain_rerun_numeric_sections_bit_identical(smoke_setup):
    """同种子重跑:cv / bootstrap 两段逐位一致——随机性只从全局种子按标签
    确定性派生的承诺(04-02 教训:线程锁与派生规则任一失守这里先显形)。"""
    s = smoke_setup
    obs = load_observation_set(s["corners_train"], s["manifest_train"], s["board"], split="train")
    optimizer_config = {"m1": {"loss": "linear"}, "m2": {"loss": "linear"}}
    first_cv = cv_pose_grouped(obs, s["board"], s["cfg"], optimizer_config)
    first_boot = bootstrap_pose(obs, s["board"], s["cfg"], optimizer_config)
    second_cv = cv_pose_grouped(obs, s["board"], s["cfg"], optimizer_config)
    second_boot = bootstrap_pose(obs, s["board"], s["cfg"], optimizer_config)
    _assert_reports_equal(first_cv, second_cv)
    _assert_reports_equal(first_boot, second_boot)
