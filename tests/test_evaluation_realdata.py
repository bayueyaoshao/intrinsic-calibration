"""真实数据全链对账:`calib evaluate` 一条命令 × Phase 4 成功标准 1–5(阶段四收官)。

整文件标记 integration(需要真实 input_data 与 corners/noise/calibration/
fisher 上游产物,日常开发用 `uv run pytest -m "not integration"` 自动跳过;
前置产物缺失时整文件跳过并写明原因,不误报失败)。

测的是什么(对应 ROADMAP Phase 4 成功标准 1–5 与 PLAN 04-06 Task 2;
2026-10-21 选型重构后按"档位感知"对账:quick 档四类实验只考当选模型、
重复缩水,full 档全模型全量,参试模型与次数以 meta 落账为准):
0. 取数体例(先于一切断言):最新 output/evaluate/run_*/evaluate.json 的
   血缘指纹(配置哈希 + 全部上游文件哈希)与当前上游产物一致就直接复用
   (全链千余次求解、小时级,不盲目重跑);指纹对不上才 subprocess 重跑
   一条 `calib evaluate`(T-04-06-02:比对不严会复用错轮产物)。
1. 标准 1:四实验段(cv/bootstrap/slices/lopo)全部在;参试模型与
   meta.experiment_models 落账一致(quick 档 = 当选模型);交叉验证每模型
   fold_heldout 恰 = 轮数 × 折数个样本;折分配明细 npz 键齐,并逐折复核
   训练/验证姿态零交集、同轮每姿态恰好考一次(姿态级防泄漏的产物侧
   复核;函数级门禁在 04-01 单测,这里对账的是真实落盘产物)。
2. 标准 2:cv 段参试模型"训练误差 vs 考题误差"并排数值;判行表行 =
   参试模型,每行三判据(泛化/稳定/逼近 CRLB)各带数值、PASS/FAIL 与
   判定式;test 整体 RMS 选型(method/walk/规则)与三判据第二意见、
   两处一致性落账;判行表逐行带 BIC 数值(平局裁决要用它)。
3. 标准 3:重复张分组每组口径标识(views80_single_frame + detector,
   匹配口径 2026-10-20 定案)随明细旁车落盘(下游不可能拿错口径对账);
   CRLB 硬告警线用记录的显著性 alpha 与自由度反推核验(不写死;df=7 时
   硬线约 0.42);低于硬线的统计量计数、判定与说明相互一致。
4. 标准 4:bootstrap 逐次参数明细 npz(档位次数 × 参试模型)可读;来源
   分离段数值可读(方差相减、钳 0 生效、位姿占比在 0–1);留一姿态
   逐姿态表 + 问题姿态点名名单可读(单模型也过,lopo 是诊断实验)。
5. 标准 5:优化器 16 组合行齐 + 推荐配置四模型键齐 + 多起点判定落盘;
   交付段四模型 final_params/rms/BIC 齐(交付与档位无关,恒全模型),
   test 独立照片评分行数 = 模型数 × 65;Fisher 谱退化预警随判行表
   notes 标注(退化模型必带)。
6. 血缘与图:evaluate.json 的 lineage.upstream 各文件真实存在;判行表
   三张 PNG(cv 误差棒 / LOPO 点名 / 离散对比)齐。

体例照 tests/test_optimizer_realdata.py / test_fisher_realdata.py:模块级
夹具只取一轮产物,全部断言共享。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from calib.evaluation.artifacts import EVALUATE_KEYS, load_evaluate_report
from calib.io.config import load_calibration

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = REPO_ROOT / "config" / "board.yaml"
CALIBRATION_PATH = REPO_ROOT / "config" / "calibration.yaml"
EVALUATE_DIR = REPO_ROOT / "output" / "evaluate"

_EXPECTED_MODELS = ("m1", "m2", "m3", "m4")
_ALLOWED_LOSSES = ("linear", "soft_l1", "huber", "cauchy")
_FIGURE_NAMES = ("cv_errorbars.png", "lopo_pose_errors.png", "dispersion_compare.png")
# subprocess 全链重跑的上限:真实数据千余次求解小时级,给足 6 小时
_SUBPROCESS_TIMEOUT_S = 6 * 3600


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_runs_newest_first() -> list[Path]:
    if not EVALUATE_DIR.is_dir():
        return []
    return sorted(
        (p for p in EVALUATE_DIR.iterdir()
         if p.is_dir() and p.name.startswith("run_")),
        reverse=True,
    )


def _fingerprint_problem(run_dir: Path) -> str | None:
    """检查一轮产物能不能复用:返回 None 表示血缘指纹完全吻合。

    复用条件(T-04-06-02,从严):evaluate.json 在、非失败轮、配置哈希
    一致、且登记的**每一个**上游文件(角点双 npz / 数据清单 / noise /
    crlb / 逐模型标定产物)都还在且内容逐位一致——上游任何一个重跑过,
    旧轮结论就不能冒充当前数据的结论。
    """
    json_path = run_dir / "evaluate.json"
    if not json_path.is_file():
        return "缺 evaluate.json"
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("failure_note"):
        return "失败轮(实验段为 null)不可复用"
    lineage = doc.get("lineage") or {}
    for name, sha in (lineage.get("config_sha256") or {}).items():
        path = REPO_ROOT / "config" / name
        if not path.is_file() or _sha256(path) != sha:
            return f"配置 {name} 与当前不一致"
    for name, entry in (lineage.get("upstream") or {}).items():
        path = Path(entry.get("path", ""))
        if not path.is_file():
            return f"上游 {name} 文件不在:{path}"
        if _sha256(path) != entry.get("sha256"):
            return f"上游 {name} 内容已变(上游重跑过,旧轮不能复用)"
    return None


@pytest.fixture(scope="module")
def evaluate_tree() -> dict:
    """取一轮真实数据评估产物:血缘指纹吻合就复用,否则重跑一条命令。

    前置产物缺失时整文件跳过;重跑走 subprocess(与用户手敲完全同一条
    命令,命令层的行为也被顺带测到);取回后 load_evaluate_report 读回
    主契约与全部明细 npz,供全部断言共享。
    """
    if not (REPO_ROOT / "input_data" / "train").is_dir():
        pytest.skip("真实 input_data 不在,跳过评估全链集成测试")
    for stage in ("corners", "noise", "calibration", "fisher"):
        if not (REPO_ROOT / "output" / stage).is_dir():
            pytest.skip(f"还没有 output/{stage} 产物,先跑对应上游命令再来")

    cfg = load_calibration(CALIBRATION_PATH)

    run_dir: Path | None = None
    last_problem = "没有任何历史轮"
    for candidate in _evaluate_runs_newest_first():
        last_problem = _fingerprint_problem(candidate)
        if last_problem is None:
            run_dir = candidate
            print(f"\n[取数] 血缘指纹吻合,复用历史轮:{candidate.name}(不重跑)")
            break
    if run_dir is None:
        print(f"\n[取数] 无可复用轮(最近一轮问题:{last_problem});"
              f"subprocess 重跑 calib evaluate(小时级,请等)……")
        proc = subprocess.run(
            ["uv", "run", "calib", "evaluate"],
            cwd=REPO_ROOT,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
        assert proc.returncode == 0, (
            f"calib evaluate 退出码 {proc.returncode}:命令中途失败"
            "(失败契约应已落盘,先看最新 output/evaluate/run_*/evaluate.json"
            " 的 failure_note)。"
        )
        for candidate in _evaluate_runs_newest_first():
            if _fingerprint_problem(candidate) is None:
                run_dir = candidate
                break
        assert run_dir is not None, "重跑完成却没有指纹吻合的新轮,落盘血缘与实际取数不一致"

    loaded = load_evaluate_report(run_dir)
    report = loaded["report"]
    for key in EVALUATE_KEYS:
        assert key in report, f"evaluate.json 缺契约键:{key}"
    assert report["failure_note"] is None, \
        f"本轮是失败轮:{report['failure_note']}"
    print(f"\n[取数] 产物目录:{run_dir}"
          f"(meta:{json.dumps(report.get('meta') or {}, ensure_ascii=False)})")
    return {
        "report": report,
        "details": loaded["details"],
        "run_dir": run_dir,
        "eval_cfg": cfg.evaluate,
    }


def _exp_profile_and_counts(report: dict, eval_cfg) -> tuple:
    """按报告档位推"四类实验实际参试模型与重复次数"(2026-10-21 选型重构)。

    quick 档:四类实验只考当选模型、重复缩水(quick_* 覆盖);full 档:
    全部模型、正式档次数。实验参试模型以 meta.experiment_models 落账为准
    (缺键的老轮按全模型处理,但老轮血缘已对不上、不会被复用)。"""
    meta = report.get("meta") or {}
    profile = str(meta.get("profile", "full"))
    exp_models = tuple(meta.get("experiment_models") or _EXPECTED_MODELS)
    if profile == "quick":
        counts = (int(eval_cfg.quick_cv_n_rounds), int(eval_cfg.quick_cv_n_folds),
                  int(eval_cfg.quick_bootstrap_n))
    else:
        counts = (int(eval_cfg.cv_n_rounds), int(eval_cfg.cv_n_folds),
                  int(eval_cfg.bootstrap_n))
    assert profile in ("quick", "full"), f"档位越枚举:{profile!r}"
    assert exp_models and set(exp_models) <= set(_EXPECTED_MODELS), \
        f"实验参试模型越名单:{exp_models}"
    return profile, exp_models, counts


# ---------------------------------------------------------------------------
# 成功标准 1:防泄漏切分的产物侧对账 + 四实验段全在
# ---------------------------------------------------------------------------


def test_criterion_1_four_sections_and_cv_samples(evaluate_tree: dict):
    """标准 1:四实验段全部在且无缺席;四类实验的参试模型与档位落账一致
    (quick 只考当选模型);cv 每模型 fold_heldout 恰 = 轮数 × 折数;
    折分配明细 npz 键齐可读。"""
    report = evaluate_tree["report"]
    for section in ("cv", "bootstrap", "slices", "lopo", "delivery"):
        assert isinstance(report.get(section), dict), \
            f"{section} 段缺席(实验没跑成或组装被跳过):成功标准 1–5 无从对账"
    assert report["meta"]["partial_sections"] == [], \
        f"有实验段缺席:{report['meta']['partial_sections']}"

    eval_cfg = evaluate_tree["eval_cfg"]
    profile, exp_models, (n_rounds, n_folds, _n_boot) = \
        _exp_profile_and_counts(report, eval_cfg)
    if profile == "quick":
        selected = (report.get("selection") or {}).get("selected")
        assert tuple(exp_models) == (selected,), \
            f"quick 档实验应只考当选模型 {selected!r},实际:{exp_models}"
    n_samples = n_rounds * n_folds
    cv_models = report["cv"]["models"]
    assert set(cv_models) == set(exp_models), \
        f"cv 段模型键 {tuple(cv_models)} 与实验参试名单 {exp_models} 不一致"
    for model_id, block in cv_models.items():
        assert len(block["fold_heldout"]) == n_samples, \
            (f"{model_id} fold_heldout 应为 {n_samples} 个样本"
             f"({n_rounds} 轮 × {n_folds} 折,{profile} 档),"
             f"实际 {len(block['fold_heldout'])}")
        assert len(block["fold_train"]) == n_samples, \
            f"{model_id} 训练侧样本数与考题侧对不上"

    detail = evaluate_tree["details"].get("cv") or {}
    for key in ("assignment_round", "assignment_fold",
                "assignment_train_poses", "assignment_val_poses"):
        assert key in detail, f"cv 明细旁车缺折分配键:{key}(分配表是零泄漏的原始证据)"


def test_criterion_1_pose_partition_zero_overlap(evaluate_tree: dict):
    """标准 1(产物侧复核):逐折训练/验证姿态零交集;同一轮里每个姿态
    恰好进一个验证折(划分唯一),且验证集合 = 该轮全部姿态。"""
    detail = evaluate_tree["details"]["cv"]
    eval_cfg = evaluate_tree["eval_cfg"]
    rounds = [int(v) for v in detail["assignment_round"]]
    _profile, _exp, (n_rounds, n_folds, _n_boot) = \
        _exp_profile_and_counts(evaluate_tree["report"], eval_cfg)
    assert sorted(set(rounds)) == list(range(n_rounds)), "折分配轮号不齐"

    for r in range(n_rounds):
        val_seen: set[str] = set()
        all_poses: set[str] = set()
        for i, row_round in enumerate(rounds):
            if row_round != r:
                continue
            train_p = set(str(detail["assignment_train_poses"][i]).split(","))
            val_p = set(str(detail["assignment_val_poses"][i]).split(","))
            fold = int(detail["assignment_fold"][i])
            overlap = train_p & val_p
            assert not overlap, \
                f"第 {r} 轮第 {fold} 折训练/验证姿态重叠:{sorted(overlap)[:4]}——泄漏"
            clash = val_p & val_seen
            assert not clash, \
                f"第 {r} 轮姿态 {sorted(clash)[:4]} 进了多个验证折(划分不唯一)"
            val_seen |= val_p
            all_poses |= train_p | val_p
        assert val_seen == all_poses, \
            f"第 {r} 轮验证姿态集合与全部姿态不一致(有姿态漏考或漏训)"
    print(f"\n[标准 1] 折分配复核通过:{n_rounds} 轮 × {n_folds} 折,"
          f"逐折零交集、每姿态每轮恰考一次")


# ---------------------------------------------------------------------------
# 成功标准 2:训练 vs held-out 并排 + 判行表四行 + 选型结论
# ---------------------------------------------------------------------------


def test_criterion_2_train_vs_heldout_and_verdicts(evaluate_tree: dict):
    """标准 2:cv 参试模型训练/考题并排数值;判行表行 = 参试模型
    (quick 档只考当选模型),每行三判据带数值 + PASS/FAIL + 判定式;
    test RMS 选型(method/walk/rule)与三判据第二意见、两处一致性落账。"""
    report = evaluate_tree["report"]
    eval_cfg = evaluate_tree["eval_cfg"]
    _profile, exp_models, _counts = _exp_profile_and_counts(report, eval_cfg)
    cv_models = report["cv"]["models"]
    for model_id in exp_models:
        block = cv_models[model_id]
        for key in ("mean_train_px", "mean_heldout_px", "n_failed_folds"):
            assert key in block, f"{model_id} 的 cv 摘要缺 {key}(并排表缺一列)"
        assert block["mean_train_px"] > 0 and block["mean_heldout_px"] > 0, \
            f"{model_id} 的误差均值非正:{block}"

    selection = report["selection"]
    assert isinstance(selection, dict) and selection, "selection 段缺席"
    verdicts = selection.get("verdicts") or []
    assert len(verdicts) == len(exp_models), \
        f"判行表应 {len(exp_models)} 行(参试模型),实际 {len(verdicts)}"
    assert {v["model_id"] for v in verdicts} == set(exp_models)
    for row in verdicts:
        m = row["model_id"]
        for criterion in ("generalization", "stability", "crlb"):
            blk = row[criterion]
            assert isinstance(blk["pass"], bool), f"{m} 的 {criterion} 判定缺布尔"
            assert blk["note"], f"{m} 的 {criterion} 判定缺判定式(结论要有数可查)"
        assert row["overall"] in ("PASS", "FAIL"), f"{m} 的总判定越枚举"
        # 判行表数值与 cv 段同源一致(两条组装路径不得各说各话)
        assert row["generalization"]["mean_heldout_px"] == \
            pytest.approx(cv_models[m]["mean_heldout_px"]), \
            f"{m} 判行表的 held-out 与 cv 段不一致"
    selected = selection.get("selected")
    assert selected in _EXPECTED_MODELS, f"选型结论不可读:{selected!r}"
    assert selection.get("method") == "test_rms_simplicity", \
        f"选型方法标识不对:{selection.get('method')!r}"
    assert selection.get("rule"), "选型缺规则记录(凭空下结论)"
    assert selection.get("reason"), "选型结论缺理由说明"
    assert isinstance(selection.get("walk"), list) and selection["walk"], \
        "选型缺逐档走查流水(walk):每步比较数字必须落账"
    test_rms = selection.get("test_rms_px") or {}
    assert set(test_rms) == set(_EXPECTED_MODELS), \
        f"test 整体 RMS 表缺模型:{tuple(test_rms)}"
    # 当选模型的 test 整体 RMS 必须有成绩(选型原料)
    assert test_rms.get(selected) is not None, \
        f"当选模型 {selected} 的 test 整体 RMS 无成绩,选型结论无凭"
    # 三判据第二意见与两处一致性落账
    tc = selection.get("three_criteria") or {}
    assert tc, "三判据第二意见缺席"
    assert tc.get("selected") in _EXPECTED_MODELS or tc.get("selected") is None
    agreement = selection.get("agreement") or {}
    assert isinstance(agreement.get("consistent"), bool) and agreement.get("note"), \
        "两处选型的一致性判定与说明缺席"
    print(f"\n[标准 2] 选型:{selected}(test RMS 简单优先);"
          f"三判据第二意见:{tc.get('selected') or '无'};"
          f"{'两处一致' if agreement.get('consistent') else '两处打架'}")


def test_criterion_2_verdict_rows_carry_bic(evaluate_tree: dict):
    """标准 2(平局裁决的弹药在案):判行表逐行带有限 BIC 数值,且与
    交付段的 BIC 同源一致(quick 档判行表只有参试模型行,逐行对账)。"""
    report = evaluate_tree["report"]
    delivery_models = report["delivery"]["models"]
    for row in report["selection"]["verdicts"]:
        m = row["model_id"]
        bic = row.get("bic")
        assert bic is not None and np.isfinite(bic), \
            f"{m} 判行表缺 BIC 数值(平局裁决将无从谈起)"
        assert bic == pytest.approx(delivery_models[m]["bic"]), \
            f"{m} 判行表 BIC 与交付段不一致"
    print("\n[标准 2] BIC 逐行在案:"
          + ", ".join(f"{r['model_id']}={r['bic']:.1f}"
                      for r in report["selection"]["verdicts"]))


# ---------------------------------------------------------------------------
# 成功标准 3:重复张分组口径标识 + CRLB 硬告警线反推核验
# ---------------------------------------------------------------------------


def test_criterion_3_slices_markers_and_alert_lines(evaluate_tree: dict):
    """标准 3:每组口径标识(views80_single_frame + detector,匹配口径
    2026-10-20 定案)随明细旁车落盘;硬告警线用记录的 alpha 与自由度反推
    (df=7 时约 0.42);低于硬线的计数、判定与说明相互一致,不静默。"""
    report = evaluate_tree["report"]
    slices = report["slices"]
    eval_cfg = evaluate_tree["eval_cfg"]
    assert slices["meta"]["sigma_source"] == "detector", "分组实验 σ 口径标识不对"
    alpha_hard = float(slices["meta"]["alpha_hard"])
    alpha_soft = float(slices["meta"]["alpha_soft"])
    assert alpha_hard == pytest.approx(float(eval_cfg.crlb_alert_alpha_hard))
    assert alpha_soft == pytest.approx(float(eval_cfg.crlb_alert_alpha_soft))

    detail = evaluate_tree["details"]["slices"]
    _profile, exp_models, _counts = _exp_profile_and_counts(report, eval_cfg)
    for model_id in exp_models:
        block = slices["models"][model_id]
        assert block["n_groups"] >= 2, f"{model_id} 分组数 {block['n_groups']} < 2,经验抖动无从谈起"
        assert block["slice_max_disp_px"] is not None, \
            f"{model_id} 缺分组参数位移当量(稳定判据缺一条腿)"
        # 明细旁车的每组口径标识(组记录不进 json,D-13;标识必须在旁车)
        markers = detail[f"{model_id}__observation_set"]
        assert {str(v) for v in markers} == {"views80_single_frame"}, \
            f"{model_id} 每组口径标识异常:{sorted(set(map(str, markers)))}"
        assert {str(v) for v in detail[f"{model_id}__sigma_source"]} == {"detector"}, \
            f"{model_id} 每组 σ 来源标识异常"

        alert = block["crlb_alert"]
        if alert is None:
            assert block["crlb_note"], \
                f"{model_id} CRLB 对照缺席却没写原因(静默跳过等于撒谎)"
            print(f"\n[标准 3] {model_id}:CRLB 对照缺席({block['crlb_note'][:60]})")
            continue
        # 匹配口径标识(2026-10-20 定案):像素空间 + 检测器级 σ;
        # 逐行铁律:实测低于下界的行数必须为 0(出现即口径错配 bug)
        assert alert.get("caliber") == "pixel_reproj_detector", \
            f"{model_id} CRLB 对照口径标识缺失或不对:{alert.get('caliber')!r}"
        assert alert.get("n_pointwise_violations") == 0, \
            f"{model_id} 逐行铁律核查违例 {alert.get('n_pointwise_violations')} 行" \
            "(实测低于理论下界,物理上不可能,口径必有错配)"
        # 告警线反推核验:s = sqrt(chi2.ppf(alpha, df)/df),df 自适应不写死
        df = int(alert["df"])
        expected_hard = float(np.sqrt(chi2.ppf(alpha_hard, df) / df))
        assert float(alert["lines"]["hard"]) == pytest.approx(expected_hard, rel=1e-9), \
            f"{model_id} 硬告警线与 alpha/df 反推值不符"
        expected_soft = float(np.sqrt(chi2.ppf(alpha_soft, df) / df))
        assert float(alert["lines"]["soft"]) == pytest.approx(expected_soft, rel=1e-9)
        if df == 7:
            assert expected_hard == pytest.approx(0.42, abs=0.01), \
                "df=7 时硬线应约 0.42(卡方双线基准)"
        n_below = sum(1 for p in alert["per_param"].values() if p["below_hard"])
        assert n_below == alert["n_below_hard"], f"{model_id} 低于硬线计数与逐参数记录不符"
        assert (alert["verdict"] == "fail") == (n_below > 0), \
            f"{model_id} 告警判定与计数矛盾"
        assert alert["verdict_note"], f"{model_id} 告警结论缺说明"
        print(f"\n[标准 3] {model_id}:df={df},硬线 {expected_hard:.4f},"
              f"实测/理论全查 {alert['n_checked']} 参数,"
              f"低于硬线 {alert['n_below_hard']} 个,判 {alert['verdict']}")


# ---------------------------------------------------------------------------
# 成功标准 4:bootstrap 明细 + 来源分离 + 留一姿态点名
# ---------------------------------------------------------------------------


def test_criterion_4_bootstrap_detail_and_separation(evaluate_tree: dict):
    """标准 4:bootstrap 逐次参数明细 npz(次数 = 档位配置值 × 参试模型);
    来源分离段数值可读(方差相减、负值钳 0、位姿占比 0–1);组装侧
    var_slice 与分组经验标准差平方一致(D-07:方差才可加)。"""
    report = evaluate_tree["report"]
    eval_cfg = evaluate_tree["eval_cfg"]
    _profile, exp_models, (_r, _f, n_boot) = _exp_profile_and_counts(report, eval_cfg)
    detail = evaluate_tree["details"]["bootstrap"]
    slices_models = report["slices"]["models"]

    for model_id in exp_models:
        reps = detail.get(f"{model_id}__rep")
        params = detail.get(f"{model_id}__params")
        assert reps is not None and params is not None, \
            f"{model_id} 缺 bootstrap 逐次明细(npz 旁车缺列)"
        assert len(reps) == n_boot, \
            f"{model_id} bootstrap 次数应为 {n_boot},实际 {len(reps)}"
        assert params.shape[0] == n_boot, f"{model_id} 逐次参数矩阵行数不对"

        sep = (report["source_separation"] or {}).get(model_id)
        assert isinstance(sep, dict) and sep, \
            f"{model_id} 缺来源分离数值(纯噪声/位姿抽样两来源没分开)"
        emp = slices_models[model_id]["empirical_std"]
        assert emp, f"{model_id} 缺分组经验标准差(来源分离的减数)"
        boot_block = report["bootstrap"]["models"][model_id]
        for name, var_slice_val in boot_block["var_slice"].items():
            assert var_slice_val == pytest.approx(float(emp[name]) ** 2, rel=1e-9), \
                f"{model_id}.{name} 的分组侧方差 ≠ 经验标准差平方(D-07 口径错)"
        for name, entry in sep.items():
            assert entry["var_noise"] > 0, f"{model_id}.{name} 纯噪声方差非正"
            assert entry["var_pose_sampling"] >= 0, \
                f"{model_id}.{name} 位姿抽样方差为负(钳 0 失效)"
            assert 0.0 <= entry["pose_share"] <= 1.0, \
                f"{model_id}.{name} 位姿占比越界:{entry['pose_share']}"
            assert entry["var_total"] == pytest.approx(
                entry["var_noise"] + entry["var_pose_sampling"], rel=1e-9), \
                f"{model_id}.{name} 总方差 ≠ 两来源之和(方差可加被改坏)"
    print(f"\n[标准 4] bootstrap {n_boot} 次 × {len(exp_models)} 参试模型明细在案;"
          f"来源分离逐参数核验通过")


def test_criterion_4_lopo_table_and_outliers(evaluate_tree: dict):
    """标准 4(留一姿态):逐姿态表(lopo-v1 契约:per_pose 为 姿态号→
    误差 的字典,失败姿态记 None,行级明细在 rows)参试模型姿态集一致
    (2026-10-21 提速裁定:lopo 只考一个模型,单模型也过);问题姿态
    点名名单可读,且点到的姿态确实在逐姿态表里。"""
    report = evaluate_tree["report"]
    lopo_models = report["lopo"]["models"]
    assert lopo_models, "lopo 段没有模型"
    assert set(lopo_models) <= set(_EXPECTED_MODELS), \
        f"lopo 模型越名单:{tuple(lopo_models)}"
    table_ids: set[str] | None = None
    for model_id, block in lopo_models.items():
        per_pose = block["per_pose"]
        assert isinstance(per_pose, dict) and per_pose, \
            f"{model_id} 留一姿态表为空(应为 姿态号→误差 字典)"
        ids = set(per_pose)
        assert table_ids is None or ids == table_ids, \
            f"{model_id} 逐姿态表姿态集与其他模型不一致"
        table_ids = ids
        failed = {p for p, v in per_pose.items() if v is None}
        assert len(failed) == int(block["n_failed_poses"]), \
            f"{model_id} 失败姿态数与 n_failed_poses 对不上"
        assert isinstance(block["outlier_poses"], list), \
            f"{model_id} 缺问题姿态名单"
        for item in block["outlier_poses"]:
            assert item["pose_id"] in ids, \
                f"{model_id} 点名的姿态 {item['pose_id']} 不在逐姿态表里"
            assert item["pose_id"] not in failed, \
                f"{model_id} 点名的姿态 {item['pose_id']} 是失败姿态(失败不参与点名)"
            assert np.isfinite(item["mad_z"]), f"{model_id} 点名缺 z 值"
        # 行级明细与逐姿态表同源:rows 覆盖全部姿态且数值一致
        rows = block["rows"]
        assert len(rows) == len(ids), f"{model_id} 行级明细行数与逐姿态表不一致"
        for row in rows:
            assert row["val_pose"] in ids, f"{model_id} 行级明细混入未知姿态"
            v = row["heldout_px"]
            assert (v is None) == (row["val_pose"] in failed), \
                f"{model_id} 行 {row['val_pose']} 的失败标记与逐姿态表矛盾"
            if v is not None:
                assert v == pytest.approx(per_pose[row["val_pose"]], rel=1e-9), \
                    f"{model_id} 行 {row['val_pose']} 误差与逐姿态表不一致"
    n_poses = len(table_ids)
    print(f"\n[标准 4] 留一姿态:{n_poses} 姿态 × {len(lopo_models)} 模型;点名名单:"
          + "; ".join(
              f"{m}:{len(lopo_models[m]['outlier_poses'])} 个"
              for m in lopo_models))


# ---------------------------------------------------------------------------
# 成功标准 5:优化器/多起点/交付 + test 并排评分 + Fisher 预警
# ---------------------------------------------------------------------------


def test_criterion_5_optimizer_and_multistart(evaluate_tree: dict):
    """标准 5(优化器对比):16 组合行齐;推荐配置四模型键齐且核在
    白名单;多起点判定与数值落盘。"""
    report = evaluate_tree["report"]
    optimizer = report["optimizer"]
    assert isinstance(optimizer, dict) and optimizer, "optimizer 段缺席"
    combos = optimizer["combos"]
    assert len(combos) == 16, f"组合数应为 16,实际 {len(combos)}"
    assert set(optimizer["recommended_solver"]) == set(_EXPECTED_MODELS)
    for model_id, rec in optimizer["recommended_solver"].items():
        if rec.get("loss") is None:
            assert rec.get("note"), f"{model_id} 无推荐却没写原因"
            continue
        assert rec["loss"] in _ALLOWED_LOSSES, f"{model_id} 推荐核越白名单:{rec['loss']}"
    ms_models = (optimizer.get("multistart") or {}).get("models") or {}
    assert set(ms_models) == set(_EXPECTED_MODELS), \
        f"multistart 四模型键不全:{tuple(ms_models)}"
    for model_id, block in ms_models.items():
        assert block["verdict"] in ("consistent", "inconsistent"), \
            f"{model_id} 多起点判定越枚举:{block['verdict']!r}"
        if block["verdict"] == "consistent":
            for field in ("rms_spread_px", "param_disp_px"):
                assert block[field] is not None and np.isfinite(block[field]), \
                    f"{model_id} 判一致却缺 {field} 数值"
    print("\n[标准 5] 多起点判定:"
          + ", ".join(f"{m}={ms_models[m]['verdict']}" for m in _EXPECTED_MODELS))


def test_criterion_5_delivery_and_test_rows(evaluate_tree: dict):
    """标准 5(交付):四模型全量重训 final_params/rms/BIC 齐;test 独立
    照片评分行数 = 模型数 × test 张数;逐行明细 npz 可读。"""
    report = evaluate_tree["report"]
    delivery = report["delivery"]
    models = delivery["models"]
    assert set(models) == set(_EXPECTED_MODELS)
    n_test = int(delivery["meta"]["n_test_images"])
    assert n_test == int(delivery["meta"]["n_test_rows"]) // len(models), \
        "test 行数与张数对不上"
    for model_id, entry in models.items():
        assert entry["converged"], f"{model_id} 全量重训未收敛:{entry['failure_note']}"
        assert entry["final_params"], f"{model_id} 缺最终内参(交付物主体)"
        assert entry["rms_px"] > 0, f"{model_id} 全量重训 RMS 非正"
        assert np.isfinite(entry["bic"]), f"{model_id} 缺 BIC"
        assert entry["n_views"] == int(delivery["meta"]["n_views"]), \
            f"{model_id} 重训视图数与全量口径不符"

    detail = evaluate_tree["details"]["delivery"]
    test_model = [str(v) for v in detail["test_model"]]
    test_rms = np.asarray(detail["test_rms"], dtype=np.float64)
    assert len(test_model) == len(_EXPECTED_MODELS) * n_test, \
        f"test 评分行应 {len(_EXPECTED_MODELS) * n_test},实际 {len(test_model)}"
    summary_parts = []
    for model_id in _EXPECTED_MODELS:
        rms_m = test_rms[np.asarray(test_model) == model_id]
        n_failed = int(np.isnan(rms_m).sum())
        assert n_failed == 0, f"{model_id} 有 {n_failed} 张 test 评分失败(不应静默)"
        summary_parts.append(
            f"{model_id}:中位 {np.median(rms_m):.4f} / 最大 {rms_m.max():.4f} px")
    print(f"\n[标准 5] test {n_test} 张并排评分(中位/最大):" + "; ".join(summary_parts))


def test_criterion_5_fisher_warnings_in_notes(evaluate_tree: dict):
    """标准 5(Fisher 预警,D-11/T-04-06-03):crlb.json 谱退化模型在
    判行表 notes 里必带预警;非退化模型不强加。预警来源直接对账
    crlb.json 原件,不信转抄。"""
    report = evaluate_tree["report"]
    fisher_run = report["meta"]["fisher_run"]
    crlb_path = REPO_ROOT / "output" / "fisher" / fisher_run / "crlb.json"
    assert crlb_path.is_file(), f"crlb.json 不在:{crlb_path}"
    with open(crlb_path, encoding="utf-8") as f:
        crlb = json.load(f)

    flagged: set[str] = set()
    for model_id in _EXPECTED_MODELS:
        rec = (((crlb.get("models") or {}).get(model_id) or {})
               .get("observation_sets", {}).get("views640", {}))
        if (rec.get("detector") or rec.get("temporal") or {}).get("flag"):
            flagged.add(model_id)

    rows = {row["model_id"]: row for row in report["selection"]["verdicts"]}
    for model_id, row in rows.items():
        notes = row.get("notes") or []
        has_warning = any("Fisher" in str(note) for note in notes)
        if model_id in flagged:
            assert has_warning, \
                f"{model_id} 谱体检记了退化(crlb.json flag 非空),判行表却没带 Fisher 预警"
        else:
            assert not has_warning, \
                f"{model_id} 谱体检无退化标记,判行表却带预警(预警不能瞎标)"
    print(f"\n[标准 5] Fisher 预警对账:谱退化模型 {sorted(flagged) or '无'},"
          f"判行表 notes 标注一致")


# ---------------------------------------------------------------------------
# 血缘与图
# ---------------------------------------------------------------------------


def test_lineage_upstream_and_three_figures(evaluate_tree: dict):
    """血缘:evaluate.json 登记的每个上游文件真实存在;判行表三张 PNG
    (CV 误差棒 / LOPO 点名 / 离散对比)齐。"""
    report = evaluate_tree["report"]
    run_dir = evaluate_tree["run_dir"]
    for key in ("corners_train_sb.npz", "corners_test_sb.npz", "manifest.json",
                "noise.json", "crlb.json"):
        assert key in report["lineage"]["upstream"], f"血缘缺 {key} 上游"
    for name, entry in report["lineage"]["upstream"].items():
        assert Path(entry["path"]).is_file(), f"血缘登记的上游文件不在:{name}"
    figures = run_dir / "figures"
    for name in _FIGURE_NAMES:
        assert (figures / name).is_file(), f"缺判行表图:{figures / name}"
    print(f"\n[图] 三张判行表图在位:{run_dir / 'figures'}")
