"""真实数据试车:优化器对比(D-13)16 组合 + 多起点一致性(D-15)首轮落地(SOL-02)。

整文件标记 integration(需要真实 input_data 与 corners/noise/calibration
产物,日常开发用 `uv run pytest -m "not integration"` 自动跳过;前置产物
缺失时整文件跳过并写明原因,不误报失败)。

测的是什么(对应 PLAN 04-02 验收):
1. 结构:optimizer.json 契约键全、非失败轮;combos 恰 16 行
   (4 模型 × 4 损失核:linear 与三种鲁棒核——鲁棒核 = 把离群残差的
   权重压低、不让个别坏点拖偏整体的损失函数),逐行汇总键齐;
2. 推荐配置:recommended_solver 四模型键齐,loss 在四核白名单内
   (D-14);某模型全军覆没时记 {"loss": None, "note": 中文},不静默;
3. 已知事实对账(Pitfall 5):m1 的 soft_l1 组合与 Phase 2 历史轮
   (run_20261018_163713,status=0 未过实用收敛)对照——本次收敛与
   不收敛都是合法发现,两头都要显式记录,绝不静默;
4. 多起点:四模型 models 键齐,每模型起点行数 = N(鱼眼 m4 翻倍),
   verdict 数值(rms_spread_px / param_disp_px)落盘;
5. 血缘:optimizer.json 的 lineage.upstream 含角点 npz + 数据清单 +
   噪声底 + 逐模型标定产物,且登记的文件真实存在(T-04-02-04);
6. 落盘纪律:每轮写全新 output/evaluate/run_<时间戳>/ 目录,不覆盖
   旧轮(T-04-02-05);本轮把对照用的历史轮目录名记进 meta。

体例照 tests/test_fisher_realdata.py:定位最新上游轮、直接调库函数
(subprocess CLI 在 04-06 才存在)、模块级夹具只跑一轮(16 组合 ×
640 视图 + 多起点合计几十次求解,分钟级),断言共享其产物。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from calib.evaluation.artifacts import (
    OPTIMIZER_KEYS,
    save_optimizer_report,
)
from calib.evaluation.optimizer import run_multistart, run_optimizer_grid
from calib.io.config import load_board, load_calibration
from calib.solve.observation import load_observation_set
from calib.util.lineage import make_lineage

pytestmark = [pytest.mark.integration, pytest.mark.realdata]

REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = REPO_ROOT / "config" / "board.yaml"
CALIBRATION_PATH = REPO_ROOT / "config" / "calibration.yaml"

# Phase 2 历史轮:m1__scipy_softl1 status=0(未过实用收敛)的出处,
# 只作 meta 对照引用(04-RESEARCH §Sources;本轮血缘不依赖它)
_HISTORY_RUN = "run_20261018_163713"

_EXPECTED_MODELS = ("m1", "m2", "m3", "m4")
_ALLOWED_LOSSES = ("linear", "soft_l1", "huber", "cauchy")

# 每行组合汇总的键(照 evaluation/optimizer.py 的 _combo_row;
# 失败占行同结构,数值字段 None)
_COMBO_ROW_KEYS = ("name", "model", "solver", "loss", "n_views",
                   "rms_overall", "converged", "failure_note",
                   "solver_status", "init_rms", "f_scale_actual")


def _latest_run(stage_dir: Path) -> Path:
    runs = sorted(p for p in stage_dir.iterdir()
                  if p.is_dir() and p.name.startswith("run_"))
    assert runs, f"{stage_dir} 下没有任何 run 目录"
    return runs[-1]


def _manifest_path_for(corners_run: Path, corner_route: str) -> Path:
    """定位与角点产物同一轮的数据清单(与 calibrate 命令同体例):
    优先角点 meta 里记录的上游清单路径(血缘链闭合,防止之后重跑
    ingest 串轮),缺记录才退回最新一轮 ingest 产物。"""
    meta_path = corners_run / f"corners_train_{corner_route}.meta.json"
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        recorded = (meta.get("lineage", {}).get("upstream", {})
                    .get("manifest.json", {}).get("path"))
        if recorded and Path(recorded).is_file():
            return Path(recorded)
    return _latest_run(REPO_ROOT / "output" / "ingest") / "manifest.json"


@pytest.fixture(scope="module")
def optimizer_tree() -> dict:
    """真实数据跑一轮优化器对比 + 多起点,产物落 output/evaluate/,
    返回读回的报告与各上游目录,供全部断言共享。

    前置产物(真实照片、corners / noise / calibration)缺失时整文件
    跳过;每轮写全新 run_<时间戳> 目录,重跑不覆盖旧轮。
    """
    if not (REPO_ROOT / "input_data" / "train").is_dir():
        pytest.skip("真实 input_data 不在,跳过优化器对比集成测试")
    for stage in ("corners", "noise", "calibration"):
        if not (REPO_ROOT / "output" / stage).is_dir():
            pytest.skip(f"还没有 output/{stage} 产物,先跑对应上游命令再来")

    board = load_board(BOARD_PATH)
    cfg = load_calibration(CALIBRATION_PATH)
    route = cfg.calibrate.corner_route
    corners_run = _latest_run(REPO_ROOT / "output" / "corners")
    noise_run = _latest_run(REPO_ROOT / "output" / "noise")
    calib_run = _latest_run(REPO_ROOT / "output" / "calibration")
    corners_npz = corners_run / f"corners_train_{route}.npz"
    manifest_path = _manifest_path_for(corners_run, route)

    def _grid_progress(row: dict) -> None:
        if row["converged"]:
            flag = f"收敛,RMS {row['rms_overall']:.4f} px"
        else:
            flag = f"未收敛({(row['failure_note'] or '')[:40]})"
        print(f"\n[组合] {row['name']}:{flag}")

    def _ms_progress(row: dict) -> None:
        print(f"\n[多起点] {row['model']} {row['done']}/{row['total']}")

    # ---- 16 组合网格(共同初值 = 各模型 OpenCV 解,从标定产物读回)----
    report = run_optimizer_grid(
        corners_npz, manifest_path, board, cfg.calibrate.scipy,
        cfg.evaluate, calib_run,
        exclude_images=tuple(cfg.exclude_images), progress=_grid_progress)

    # ---- 多起点一致性(全量观测集;N 与幅度从 evaluate 配置读)----
    obs = load_observation_set(corners_npz, manifest_path, board,
                               split="train",
                               exclude_images=tuple(cfg.exclude_images))
    report["multistart"] = run_multistart(obs, board, cfg,
                                          progress=_ms_progress)

    # meta 记录对照引用的历史轮(m1-softl1 已知事实的出处,Pitfall 5)
    report.setdefault("meta", {})["history_reference_run"] = _HISTORY_RUN

    # ---- 血缘(照 calib fisher 命令的 upstream 登记体例)+ 落盘 ----
    started = datetime.now(UTC)
    round_id = f"run_{started.strftime('%Y%m%d_%H%M%S')}"
    out_dir = REPO_ROOT / "output" / "evaluate" / round_id
    upstream = {
        f"corners_train_{route}.npz": corners_npz,
        "manifest.json": manifest_path,
        "noise.json": noise_run / "noise.json",
    }
    for model_id in cfg.evaluate.models:
        # 标定产物按组合子目录存放(m{X}__scipy/result.json),不在 run 根下
        for rel in (f"{model_id}__scipy/result.json",
                    f"{model_id}__scipy/extrinsics.npz",
                    f"{model_id}__opencv/result.json"):
            candidate = calib_run / rel
            if candidate.is_file():
                upstream[rel] = candidate
    lineage = make_lineage(
        config_files={"board.yaml": BOARD_PATH,
                      "calibration.yaml": CALIBRATION_PATH},
        upstream=upstream, round_id=round_id)

    paths = save_optimizer_report(out_dir, report, lineage)
    assert not out_dir.name.startswith(_HISTORY_RUN), "写盘轮次标识撞了历史轮"
    with open(paths["optimizer_json"], encoding="utf-8") as f:
        saved = json.load(f)

    # ---- 中文摘要(逐模型一行:推荐配置 + 多起点判定)----
    print(f"\n优化器对比完成,产物目录:{out_dir}"
          f"(上游 corners {corners_run.name} / noise {noise_run.name} / "
          f"calibration {calib_run.name},历史对照轮 {_HISTORY_RUN})")
    for model_id in saved["recommended_solver"]:
        rec = saved["recommended_solver"][model_id]
        if rec.get("loss") is None:
            print(f"  {model_id}:没有收敛组合可推荐({rec.get('note', '')[:50]})")
        else:
            scale = (f",f_scale {rec['f_scale_px']:.3f} px"
                     if "f_scale_px" in rec else "")
            print(f"  {model_id}:推荐 {rec['loss']} 核{scale}")
        ms = saved["multistart"]["models"][model_id]
        print(f"  {model_id}:多起点 {ms['n_converged']}/{ms['n_starts']} 收敛,"
              f"判定 {ms['verdict']}"
              + (f",RMS 极差 {ms['rms_spread_px']:.4f} px"
                 if ms["rms_spread_px"] is not None else ""))

    return {"report": saved, "run_dir": out_dir, "calib_run": calib_run,
            "n_views": report["meta"]["n_views"],
            "evaluate_cfg": cfg.evaluate}


def test_optimizer_contract_and_16_rows(optimizer_tree: dict):
    """结构对账:契约键全、非失败轮、combos 恰 16 行(4 模型 × 4 核),
    逐行汇总键齐;robust 行带 rms_linear_ref 所需的对照信息。"""
    report = optimizer_tree["report"]
    for key in OPTIMIZER_KEYS:
        assert key in report, f"optimizer.json 缺契约键:{key}"
    assert report["failure_note"] is None, \
        f"本轮是失败轮:{report['failure_note']}"

    combos = report["combos"]
    assert len(combos) == 16, \
        f"组合数应为 16(4 模型 × 4 核),实际 {len(combos)}"
    for model_id in _EXPECTED_MODELS:
        rows = [r for r in combos if r["model"] == model_id]
        assert len(rows) == 4, \
            f"{model_id} 应有 4 行组合,实际 {len(rows)}"
        assert {r["loss"] for r in rows} == set(_ALLOWED_LOSSES), \
            f"{model_id} 的核名单不对:{sorted({r['loss'] for r in rows})}"
        for row in rows:
            for key in _COMBO_ROW_KEYS:
                assert key in row, \
                    f"{model_id} × {row['loss']} 行缺汇总键:{key}"
            assert row["solver"] == "scipy"
            assert row["n_views"] == optimizer_tree["n_views"], \
                f"{model_id} × {row['loss']} 视图数与全量口径不符"
            if not row["converged"]:
                assert row["failure_note"], \
                    (f"{model_id} × {row['loss']} 未收敛却没写原因——"
                     f"静默失败等于撒谎")
    meta = report["meta"]
    assert meta["n_combos"] == 16
    assert meta["n_failed"] == sum(1 for r in combos if not r["converged"])
    assert meta["history_reference_run"] == _HISTORY_RUN, \
        "meta 里必须记录对照用的历史轮目录名(已知事实对账的出处)"
    assert meta["calibration_run"] == optimizer_tree["calib_run"].name, \
        "初值来源轮登记与实际读回的标定轮不一致(串轮风险,T-04-02-01)"


def test_recommended_solver_keys_and_whitelist(optimizer_tree: dict):
    """推荐配置:四模型键齐;loss 在四核白名单内(D-14);有推荐的必给
    出最小 RMS 的组合,全军覆没的记 {"loss": None, "note"} 不静默。"""
    report = optimizer_tree["report"]
    recommended = report["recommended_solver"]
    assert set(recommended) == set(_EXPECTED_MODELS), \
        f"recommended_solver 应为四模型键,实际 {tuple(recommended)}"
    for model_id in _EXPECTED_MODELS:
        rec = recommended[model_id]
        if rec.get("loss") is None:
            assert rec.get("note"), \
                f"{model_id} 无推荐配置却没写中文说明(不静默被破坏)"
            continue
        assert rec["loss"] in _ALLOWED_LOSSES, \
            f"{model_id} 推荐 loss 越白名单:{rec['loss']}(D-14 四核之外)"
        # 推荐必须出自收敛组合:其 RMS 不得劣于该模型任何收敛行
        rows = [r for r in report["combos"]
                if r["model"] == model_id and r["converged"]]
        assert rows, f"{model_id} 有收敛行却记了无推荐,自相矛盾"
        best = min(r["rms_overall"] for r in rows)
        chosen = next(r for r in rows if r["loss"] == rec["loss"])
        assert abs(chosen["rms_overall"] - best) <= best * 1e-3, \
            (f"{model_id} 推荐的 {rec['loss']} 核 RMS({chosen['rms_overall']})"
             f"比最优 {best} 差出千分之一并列线,推荐规则被改坏")


def test_m1_softl1_reconciliation_explicit(optimizer_tree: dict):
    """已知事实对账(Pitfall 5):m1 × soft_l1 与 Phase 2 历史轮
    (status=0 未过实用收敛)对照——两种结果都合法,都必须显式记录:
    仍未收敛 → 与历史一致,写明原因;本次收敛 → 记录差异(新发现)。"""
    row = next(r for r in optimizer_tree["report"]["combos"]
               if r["model"] == "m1" and r["loss"] == "soft_l1")
    if row["converged"]:
        print(f"\n[对账] m1 × soft_l1 本次收敛(RMS {row['rms_overall']:.4f}"
              f" px,solver_status={row['solver_status']})——与历史轮"
              f" {_HISTORY_RUN} 的 status=0 不同,差异已显式记录为新发现")
    else:
        assert row["failure_note"], \
            "m1 × soft_l1 未收敛必须写明原因(对账结论的正文)"
        print(f"\n[对账] m1 × soft_l1 本次仍未收敛,与历史轮 {_HISTORY_RUN} "
              f"的 status=0 一致:solver_status={row['solver_status']},"
              f"原因:{(row['failure_note'] or '')[:80]}")


def test_lineage_upstream_files_exist(optimizer_tree: dict):
    """血缘(T-04-02-04):lineage.upstream 含角点 npz + 数据清单 +
    噪声底 + 逐模型标定产物,登记的每个文件真实存在(产物链可追溯)。"""
    upstream = optimizer_tree["report"]["lineage"]["upstream"]
    for key in ("manifest.json", "noise.json"):
        assert key in upstream, f"血缘缺 {key} 上游:{tuple(upstream)}"
    assert any(k.startswith("corners_train_") for k in upstream), \
        f"血缘缺角点 npz 上游:{tuple(upstream)}"
    for model_id in _EXPECTED_MODELS:
        assert f"{model_id}__opencv/result.json" in upstream, \
            f"血缘缺 {model_id} 的 OpenCV 初值产物(共同初值来源要可考)"
        assert f"{model_id}__scipy/result.json" in upstream, \
            f"血缘缺 {model_id} 的 scipy 产物:{tuple(upstream)}"
    for key, entry in upstream.items():
        path = Path(entry["path"])
        assert path.is_file(), f"血缘登记的上游文件不在:{key} → {path}"


def test_multistart_models_complete(optimizer_tree: dict):
    """多起点(D-15):四模型键齐;起点行数 = N(鱼眼 m4 翻倍);
    verdict 判定与数值(rms_spread_px / param_disp_px)落盘;
    未收敛的起点必须留原因。"""
    report = optimizer_tree["report"]
    ms_models = report["multistart"]["models"]
    assert set(ms_models) == set(_EXPECTED_MODELS), \
        f"multistart.models 应为四模型键,实际 {tuple(ms_models)}"
    eval_cfg = optimizer_tree["evaluate_cfg"]
    n_expect = {"m1": eval_cfg.multistart_n, "m2": eval_cfg.multistart_n,
                "m3": eval_cfg.multistart_n,
                "m4": eval_cfg.multistart_n_fisheye}
    for model_id in _EXPECTED_MODELS:
        block = ms_models[model_id]
        assert block["n_starts"] == n_expect[model_id], \
            (f"{model_id} 起点数应为 {n_expect[model_id]},"
             f"实际 {block['n_starts']}")
        assert len(block["starts"]) == block["n_starts"], \
            f"{model_id} 的起点行数与 n_starts 对不上(有起点被静默吞掉)"
        assert block["verdict"] in ("consistent", "inconsistent"), \
            f"{model_id} 的 verdict 越枚举:{block['verdict']!r}"
        if block["verdict"] == "consistent":
            for field in ("rms_spread_px", "param_disp_px"):
                assert block[field] is not None, \
                    f"{model_id} 判一致却缺 {field} 数值(判定要有数可查)"
            assert block["rms_spread_px"] < eval_cfg.multistart_rms_tol_px
            assert block["param_disp_px"] < eval_cfg.multistart_param_tol_px
        for row in block["starts"]:
            if not row["converged"]:
                assert row["failure_note"], \
                    f"{model_id} 起点号 {row['start']} 未收敛却没写原因"
    # m4 重点观察(D-15):判定与数值无论好坏都必须在案
    m4 = ms_models["m4"]
    assert m4["note"], "m4 的判定说明为空(重点观察对象的结论要写明)"
    print(f"\n[多起点] m4(鱼眼,重点观察):{m4['n_converged']}/"
          f"{m4['n_starts']} 收敛,判定 {m4['verdict']},"
          f"坏解 {len(m4['bad_solutions'])} 个——{m4['note'][:80]}")


def test_rerun_writes_new_run_dir(optimizer_tree: dict, tmp_path):
    """落盘纪律(T-04-02-05):再跑一轮会得到全新目录、互不覆盖——
    落盘只写调用方指定的目录(本轮与探针轮都在,内容互不串)。"""
    first = optimizer_tree["run_dir"]
    probe = tmp_path / "run_probe_next_round"
    report = {"schema_version": "optimizer-v1", "created": None,
              "recommended_solver": {}, "combos": [], "multistart": None,
              "meta": {}, "failure_note": "探针:验证再跑一轮不覆盖旧轮"}
    save_optimizer_report(probe, report, {})
    assert (probe / "optimizer.json").is_file(), "探针轮没落盘"
    # 旧轮原封不动:契约键还在、组合行数未被探针轮冲掉
    with open(first / "optimizer.json", encoding="utf-8") as f:
        first_doc = json.load(f)
    assert len(first_doc["combos"]) == 16, "旧轮 optimizer.json 被冲掉了"
