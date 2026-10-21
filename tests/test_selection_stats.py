"""选型统计纯函数的测试:合成数据把每个判定分支锁死。

小样证明什么(全部固定 numpy 种子,逐位可复现):
- paired_comparison 三分支(D-09/D-10):差值远小于等价带 → 判"实际等价"
  (取简);置信区间含 0 但太宽 → 判"统计平局"(交 BIC 裁决);区间不含 0
  → 判"真实差异"(held-out 小者胜)。非正态差值走 Wilcoxon 符号秩兜底,
  所用方法与正态性 p 值记进返回值。
- bic_full_fit(D-10):对已知 rss / 参数数 / 姿态数给出手算期望值
  (写死在断言里);rss=0 返回 inf 不崩溃,非法输入中文报错。
- mad_outlier_poses(D-08):注入的离群姿态被点名、全体相同(MAD=0)零误报。

真实数据上的结论只信本文件证明过的分支逻辑;阈值全部从参数读,
函数不做任何 config 解析(D-12)。卡方告警线、三判据判行表与选型规则
的用例在本文件下半部(任务 2 段)。
"""

from __future__ import annotations

import numpy as np
import pytest

from calib.evaluation.selection import (
    bic_full_fit,
    build_verdict_table,
    crlb_alert_check,
    mad_outlier_poses,
    paired_comparison,
    select_by_test_rms,
    select_model,
)
from calib.io.config import CalibrationConfig

# 固定种子:每个分支的合成样本逐位可复现,期望值都经手工核过
SEED_EQ, SEED_TIE, SEED_DIFF, SEED_EXP, SEED_MAD = 11, 22, 33, 44, 55


class TestPairedComparison:
    """配对检验三分支判定(D-09/D-10)与输入防线(T-04-03-01)。"""

    def test_equivalent_branch_ci_inside_band(self):
        # 25 个差值 ~ N(0, 0.001):均值差远小于等价带 0.005 → 判等价(取简)
        rng = np.random.default_rng(SEED_EQ)
        diff = rng.normal(0.0, 0.001, 25)
        out = paired_comparison(diff, tie_band_px=0.005, tie_band_rel=0.02)
        assert out["verdict"] == "equivalent"
        assert out["normal"] is True
        assert out["method"] == "ttest_rel"
        assert out["ci_high"] <= out["delta"]
        assert out["ci_low"] >= -out["delta"]
        # meta 记录:正态性 p 值可读出(0.51,无证据说它不是正态)
        assert out["shapiro_p"] >= 0.05

    def test_delta_uses_relative_band_when_simpler_mean_given(self):
        # δ = max(0.005, 2% × 0.3) = 0.006:相对档生效
        rng = np.random.default_rng(SEED_EQ)
        diff = rng.normal(0.0, 0.001, 25)
        out = paired_comparison(diff, 0.005, 0.02, simpler_mean=0.3)
        assert out["delta"] == pytest.approx(max(0.005, 0.02 * 0.3))
        assert out["verdict"] == "equivalent"
        # 不给 simpler_mean 时只用绝对档
        out_abs = paired_comparison(diff, 0.005, 0.02)
        assert out_abs["delta"] == pytest.approx(0.005)

    def test_tie_branch_ci_contains_zero_but_too_wide(self):
        # 差值 ~ N(0, 1.0):零均值大方差,区间含 0 但宽度远超 ±0.005 → 统计平局
        rng = np.random.default_rng(SEED_TIE)
        diff = rng.normal(0.0, 1.0, 25)
        out = paired_comparison(diff, tie_band_px=0.005, tie_band_rel=0.02)
        assert out["verdict"] == "tie"
        assert out["ci_low"] <= 0.0 <= out["ci_high"]
        assert out["ci_high"] > out["delta"] or out["ci_low"] < -out["delta"]
        # 平局不靠 p 值判:meta 里的方法与数值齐备,结论可辩护
        assert out["method"] == "ttest_rel"

    def test_different_branch_ci_excludes_zero(self):
        # 差值 ~ N(0.5, 0.1):区间整体在带外且不含 0 → 真实差异
        rng = np.random.default_rng(SEED_DIFF)
        diff = rng.normal(0.5, 0.1, 25)
        out = paired_comparison(diff, tie_band_px=0.005, tie_band_rel=0.02)
        assert out["verdict"] == "different"
        assert out["ci_low"] > 0.0 or out["ci_high"] < 0.0
        assert abs(out["mean_diff"]) > out["delta"]

    def test_nonnormal_diff_falls_back_to_wilcoxon(self):
        # 指数分布差值(n=25):Shapiro-Wilk 拒绝正态(p ≈ 0.005)→ Wilcoxon 兜底
        rng = np.random.default_rng(SEED_EXP)
        diff = rng.exponential(1.0, 25)
        out = paired_comparison(diff, 0.005, 0.02)
        assert out["normal"] is False
        assert out["method"] == "wilcoxon"
        assert out["shapiro_p"] < 0.05
        # meta 记录 Wilcoxon 的实际路径(精确法/正态近似)与 p 值
        assert out["wilcoxon_method"] in ("exact", "approx")
        assert 0.0 <= out["wilcoxon_p"] <= 1.0
        # 全部差值为正 → 区间必须整体为正(方向判得出来,不会误判平局)
        assert out["ci_low"] > 0.0

    def test_note_is_chinese_one_liner_with_numbers(self):
        # 呈现体例:判定式与阈值随结论落盘(照 scoring.py 三分支体例)
        rng = np.random.default_rng(SEED_DIFF)
        diff = rng.normal(0.5, 0.1, 25)
        out = paired_comparison(diff, 0.005, 0.02)
        assert "0.5" in out["note"] or "0.4" in out["note"]  # 数值进结论
        assert "等价带" in out["note"]

    def test_rejects_nonfinite_input_in_chinese(self):
        diff = np.array([0.1, 0.2, np.nan, 0.0, 0.3])
        with pytest.raises(ValueError, match="非有限"):
            paired_comparison(diff, 0.005, 0.02)

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="至少"):
            paired_comparison(np.array([0.1, -0.1]), 0.005, 0.02)


class TestBicFullFit:
    """BIC 平局裁决器(D-10):已知值手算对照 + 非法输入防线。"""

    def test_known_value_matches_hand_calculation(self):
        # rss = 80×(0.1)² = 0.8,n = 80 个有效姿态,k = 8 内参 + 6×10 外参 = 68
        rss, n, k = 0.8, 80, 68
        expected = n * np.log(rss / n) + k * np.log(n)
        assert bic_full_fit(rss, n_params=k, n_poses=n) == pytest.approx(expected, rel=1e-12)
        # 手算期望写死:-70.43580372122335(80·ln(0.008) + 68·ln(80))
        assert bic_full_fit(0.8, 68) == pytest.approx(-70.43580372122335, rel=1e-12)

    def test_zero_rss_returns_inf_without_crash(self):
        # 完美拟合对数无定义且物理上可疑:返回 +inf,让它在 BIC 比较里必输
        assert bic_full_fit(0.0, 68) == float("inf")

    def test_invalid_inputs_raise_in_chinese(self):
        with pytest.raises(ValueError, match="rss"):
            bic_full_fit(-1.0, 68)
        with pytest.raises(ValueError, match="不合法"):
            bic_full_fit(0.8, 0)
        with pytest.raises(ValueError, match="不合法"):
            bic_full_fit(0.8, 68, n_poses=0)


class TestMadOutlierPoses:
    """留一姿态的问题姿态点名(D-08):MAD 稳健 z 分数。"""

    def test_injected_outlier_named_with_z_score(self):
        # 80 个 N(0,1) 里注入一个 10.0:恰好它被点名,且只点它
        rng = np.random.default_rng(SEED_MAD)
        vals = {f"pose_{i:03d}": float(v) for i, v in enumerate(rng.normal(0.0, 1.0, 80))}
        vals["pose_007"] = 10.0
        out = mad_outlier_poses(vals, threshold=3.5)
        assert len(out) == 1
        pose_id, z = out[0]
        assert pose_id == "pose_007"
        assert z == pytest.approx(8.221306127032793, rel=1e-9)

    def test_all_identical_values_no_false_alarm(self):
        # MAD = 0(全体值相同)→ 零点名:全体一个样无从谈离群,不误报
        vals = {f"pose_{i:03d}": 0.123 for i in range(80)}
        assert mad_outlier_poses(vals, threshold=3.5) == []

    def test_threshold_read_from_parameter(self):
        # 阈值从参数读(config 化):提到 100 → 同一批数据零点名
        rng = np.random.default_rng(SEED_MAD)
        vals = {f"pose_{i:03d}": float(v) for i, v in enumerate(rng.normal(0.0, 1.0, 80))}
        vals["pose_007"] = 10.0
        assert mad_outlier_poses(vals, threshold=100.0) == []

    def test_results_sorted_by_z_descending(self):
        rng = np.random.default_rng(SEED_MAD)
        vals = {f"pose_{i:03d}": float(v) for i, v in enumerate(rng.normal(0.0, 1.0, 80))}
        vals["pose_007"] = 10.0
        vals["pose_042"] = 5.0  # 次强离群(z ≈ 3.95,过 3.5 线;4.0 的 z ≈ 3.0 过不了)
        out = mad_outlier_poses(vals, threshold=3.5)
        assert [p for p, _ in out] == ["pose_007", "pose_042"]
        assert out[0][1] > out[1][1]

    def test_rejects_nonfinite_in_chinese(self):
        vals = {f"pose_{i:03d}": float(i) for i in range(10)}
        vals["pose_003"] = float("nan")
        with pytest.raises(ValueError, match="非有限"):
            mad_outlier_poses(vals, threshold=3.5)

    def test_empty_input_returns_empty(self):
        assert mad_outlier_poses({}, threshold=3.5) == []


# ---------------------------------------------------------------------------
# 任务 2:卡方硬告警线(D-05/D-06)+ 三判据判行表 + 选型规则(D-11/D-12)
# ---------------------------------------------------------------------------

# 判行表阈值(从 EvaluateConfig 取的两个执行位;测试里直接传字面量)
TH = {"gen_overfit_ratio_max": 1.5, "stability_disp_px_max": 0.05}

# 卡方告警线的 venv 实测期望值(df=7 与 df=3,alpha=0.01):
#   线 = sqrt(chi2.ppf(alpha, df) / df),随组数自适应
HARD_LINE_DF7 = 0.42072086191744923
SOFT_LINE_DF7 = 0.5564363536057004
HARD_LINE_DF3 = 0.19564576995096442


def _sigma_named() -> dict[str, float]:
    """8 个参数的合成 CRLB 下界(每个 0.01,量级随意,判定只看比值)。"""
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")
    return {nm: 0.01 for nm in names}


def _cv_entry(held: float, train: float, folds=None, n_failed: int = 0) -> dict:
    """判行表输入的 CV 摘要(接口先行,形状照 04-03 计划定死)。"""
    return {
        "mean_heldout_px": held,
        "mean_train_px": train,
        "fold_heldout": list(folds) if folds is not None else [held] * 25,
        "n_failed_folds": n_failed,
    }


def _stab_entry(boot_disp: float, slice_disp: float) -> dict:
    """判行表输入的稳定性摘要(bootstrap 与分组离散的像素位移当量)。"""
    return {"max_disp_px": boot_disp, "slice_max_disp_px": slice_disp}


def _passing_crlb() -> dict:
    """一份全 PASS 的 crlb_alert_check 返回值(比值 0.9,高于两条线)。"""
    emp = {nm: 0.009 for nm in _sigma_named()}
    return crlb_alert_check(emp, _sigma_named(), 8, 0.01, 0.05)


def _row(mid: str, gen: bool, stab: bool, crlb: bool, bic: float | None = None) -> dict:
    """构造一行判行表结果(只带 select_model 消费的判定键)。"""
    row = {
        "model_id": mid,
        "generalization": {"pass": gen},
        "stability": {"pass": stab},
        "crlb": {"pass": crlb},
        "notes": [],
        "overall": "PASS" if (gen and stab and crlb) else "FAIL",
    }
    if bic is not None:
        row["bic"] = bic
    return row


class TestCrlbAlertCheck:
    """「实测好于理论下界」两档告警(D-05/D-06):卡方线数学 + 不静默。"""

    def test_all_above_lines_pass(self):
        # 8 个经验标准差全部 0.9 × σ_CRLB(高于软线 0.56)→ 零告警,判 pass
        emp = {nm: 0.009 for nm in _sigma_named()}
        out = crlb_alert_check(emp, _sigma_named(), n_groups=8, alpha_hard=0.01, alpha_soft=0.05)
        assert out["verdict"] == "pass"
        assert out["n_below_hard"] == 0
        assert out["n_below_soft"] == 0
        assert out["n_no_bound"] == 0
        # 卡方线数学:alpha=0.01/df=7 → 0.4207…(venv 实测值写死对照)
        assert out["lines"]["hard"] == pytest.approx(HARD_LINE_DF7, rel=1e-12)
        assert out["lines"]["soft"] == pytest.approx(SOFT_LINE_DF7, rel=1e-12)
        assert out["df"] == 7

    def test_below_hard_line_fails_loudly(self):
        # 某参数经验 std = 0.3 × σ_CRLB(低于 0.42 硬线)→ FAIL 且不静默:
        # 告警说明必须含参数名与两个数值(实测比值、告警线)
        emp = {nm: 0.009 for nm in _sigma_named()}
        emp["k1"] = 0.003
        out = crlb_alert_check(emp, _sigma_named(), 8, 0.01, 0.05)
        assert out["verdict"] == "fail"
        assert out["n_below_hard"] == 1
        assert out["per_param"]["k1"]["below_hard"] is True
        assert "k1" in out["verdict_note"]
        assert "0.30" in out["verdict_note"]
        assert "0.42" in out["verdict_note"]

    def test_soft_line_counts_but_does_not_fail(self):
        # 经验 std = 0.50 × σ_CRLB(在 0.42 与 0.56 之间)→ 软提示计数,不拦
        emp = {nm: 0.005 for nm in _sigma_named()}
        out = crlb_alert_check(emp, _sigma_named(), 8, 0.01, 0.05)
        assert out["verdict"] == "pass"
        assert out["n_below_hard"] == 0
        assert out["n_below_soft"] == 8

    def test_inf_crlb_skipped_and_counted(self):
        # σ_CRLB = inf 的参数(数据定不住的方向)跳过检查,计入 n_no_bound:
        # 不进 FAIL 也不进 PASS 分母
        emp = {nm: 0.009 for nm in _sigma_named()}
        emp["k6"] = 1.0
        crlb = _sigma_named()
        crlb["k6"] = float("inf")
        out = crlb_alert_check(emp, crlb, 8, 0.01, 0.05)
        assert out["verdict"] == "pass"
        assert out["n_no_bound"] == 1
        assert out["per_param"]["k6"]["ratio"] is None
        assert out["n_below_hard"] == 0

    def test_df_adapts_to_group_count(self):
        # 4 组经验 std(自由度 3):硬线 = sqrt(chi2.ppf(0.01, 3)/3),手算对照
        emp = {nm: 0.009 for nm in ("fx", "fy", "cx", "cy")}
        crlb = {nm: 0.01 for nm in emp}
        out = crlb_alert_check(emp, crlb, 4, 0.01, 0.05)
        assert out["df"] == 3
        assert out["lines"]["hard"] == pytest.approx(HARD_LINE_DF3, rel=1e-12)

    def test_too_few_groups_raises_in_chinese(self):
        with pytest.raises(ValueError, match="至少"):
            crlb_alert_check({"fx": 0.009}, {"fx": 0.01}, 1, 0.01, 0.05)


class TestBuildVerdictTable:
    """三判据判行表(D-12):行结构、判定式落盘、Fisher 预警与失败率进 notes(D-11)。"""

    def test_row_structure_and_overall(self):
        cv = {
            "m1": _cv_entry(0.10, 0.12),  # 比值 0.833 < 1.5 → 泛化 PASS
            "m2": _cv_entry(0.10, 0.12),
        }
        stab = {"m1": _stab_entry(0.01, 0.02), "m2": _stab_entry(0.09, 0.02)}
        crlb = {"m1": _passing_crlb(), "m2": _passing_crlb()}
        table = build_verdict_table(cv, stab, crlb, {}, TH)
        assert table["schema"] == "evaluate-verdict-v1"
        assert table["thresholds"] == TH  # 实际用值随表落盘(T-04-03-02)
        rows = {r["model_id"]: r for r in table["verdicts"]}
        assert rows["m1"]["overall"] == "PASS"
        assert rows["m2"]["overall"] == "FAIL"
        assert rows["m2"]["stability"]["pass"] is False
        assert rows["m2"]["generalization"]["pass"] is True
        assert rows["m1"]["crlb"]["pass"] is True
        # 每条判据的判定式随结论落盘:数值 + 阈值都能从 note 读出来
        assert "0.833" in rows["m1"]["generalization"]["note"]
        assert "1.5" in rows["m1"]["generalization"]["note"]
        assert "0.05" in rows["m2"]["stability"]["note"]
        assert "pass" in rows["m1"]["crlb"]["note"]

    def test_fisher_warning_and_failed_folds_go_into_notes(self):
        cv = {"m3": _cv_entry(0.15, 0.12, n_failed=4)}
        stab = {"m3": _stab_entry(0.01, 0.02)}
        crlb = {"m3": _passing_crlb()}
        warning = "Fisher 预警:高阶畸变方向谱退化(条件数 5.7e11)"
        table = build_verdict_table(cv, stab, crlb, {"m3": warning}, TH)
        row = table["verdicts"][0]
        # Fisher 预警原样进 notes(D-11),失败折数说明同进
        assert warning in row["notes"]
        assert any("4" in s for s in row["notes"])
        assert row["generalization"]["n_failed_folds"] == 4

    def test_missing_sibling_summary_raises_in_chinese(self):
        cv = {"m1": _cv_entry(0.10, 0.12), "m2": _cv_entry(0.11, 0.12)}
        stab = {"m1": _stab_entry(0.01, 0.02)}  # m2 缺稳定性摘要
        with pytest.raises(ValueError, match="m2"):
            build_verdict_table(cv, stab, {"m1": _passing_crlb(), "m2": _passing_crlb()}, {}, TH)

    def test_nonfinite_cv_numbers_rejected(self):
        cv = {"m1": _cv_entry(float("nan"), 0.12)}
        with pytest.raises(ValueError, match="非有限"):
            build_verdict_table(
                cv, {"m1": _stab_entry(0.01, 0.02)}, {"m1": _passing_crlb()}, {}, TH
            )


class TestSelectModel:
    """选型规则(D-09/D-10/D-12):全 PASS 才当选、取简、BIC 裁决、无当选给原因。"""

    def test_single_passing_candidate_selected(self):
        verdicts = [_row("m1", True, True, True), _row("m3", True, False, True)]
        cv = {"m1": _cv_entry(0.10, 0.12), "m3": _cv_entry(0.11, 0.12)}
        out = select_model(verdicts, cv, CalibrationConfig())
        assert out["selected"] == "m1"
        assert "唯一" in out["reason"]
        assert "全 PASS" in out["selection_rule"]

    def test_equivalent_takes_simpler_over_heldout_min(self):
        # m2 的 held-out 均值最小,但与 m1 的差距整体落在等价带内 → 取更简的 m1
        rng = np.random.default_rng(67)
        n1 = rng.normal(0.0, 1e-4, 25)
        n2 = rng.normal(0.0, 1e-4, 25)
        folds_m1 = (0.1000 + n1).tolist()
        folds_m2 = (0.10001 + n2).tolist()
        cv = {
            "m1": _cv_entry(float(np.mean(folds_m1)), 0.12, folds_m1),
            "m2": _cv_entry(float(np.mean(folds_m2)), 0.12, folds_m2),
        }
        out = select_model(
            [_row("m1", True, True, True), _row("m2", True, True, True)], cv, CalibrationConfig()
        )
        assert out["comparison"]["verdict"] == "equivalent"
        assert out["selected"] == "m1"  # m1 = 8 参,m2 = 9 参(D-09 取简)
        assert "m1" in out["reason"] and "等价" in out["reason"]

    def test_tie_resolved_by_bic(self):
        # 折级差距太大 → 统计平局;BIC 更小者胜:更复杂的 m2(9 参)凭
        # BIC 1000 < 1010 胜出。取简兜底会给 m1(8 参),本断言证明走的
        # 真是 BIC 裁决(D-10),不是拿"取简"冒充
        rng = np.random.default_rng(66)
        d = rng.normal(0.0, 1.0, 25)
        folds_m1 = (0.100 - d / 2.0).tolist()  # 均值小 → held-out 第一名
        folds_m2 = (0.100 + d / 2.0).tolist()
        cv = {
            "m1": _cv_entry(float(np.mean(folds_m1)), 0.12, folds_m1),
            "m2": _cv_entry(float(np.mean(folds_m2)), 0.12, folds_m2),
        }
        verdicts = [
            _row("m1", True, True, True, bic=1010.0),
            _row("m2", True, True, True, bic=1000.0),
        ]
        out = select_model(verdicts, cv, CalibrationConfig())
        assert out["comparison"]["verdict"] == "tie"
        assert out["selected"] == "m2"
        assert "BIC" in out["reason"]

    def test_tie_without_bic_falls_back_to_simpler(self):
        # 行内没带 BIC 数值:退回 D-09 取简,且理由里说明 BIC 缺席
        rng = np.random.default_rng(66)
        d = rng.normal(0.0, 1.0, 25)
        folds_m1 = (0.100 - d / 2.0).tolist()
        folds_m2 = (0.100 + d / 2.0).tolist()
        cv = {
            "m1": _cv_entry(float(np.mean(folds_m1)), 0.12, folds_m1),
            "m2": _cv_entry(float(np.mean(folds_m2)), 0.12, folds_m2),
        }
        out = select_model(
            [_row("m1", True, True, True), _row("m2", True, True, True)], cv, CalibrationConfig()
        )
        assert out["comparison"]["verdict"] == "tie"
        assert out["selected"] == "m1"
        assert "BIC" in out["reason"]

    def test_no_passing_candidate_returns_null_with_reasons(self):
        verdicts = [_row("m3", True, False, True), _row("m4", False, True, False)]
        cv = {"m3": _cv_entry(0.10, 0.12), "m4": _cv_entry(0.10, 0.12)}
        out = select_model(verdicts, cv, CalibrationConfig())
        assert out["selected"] is None
        assert set(out["reasons"]) == {"m3", "m4"}
        assert "稳定" in out["reasons"]["m3"]
        assert "泛化" in out["reasons"]["m4"] and "CRLB" in out["reasons"]["m4"]


class TestSelectByTestRms:
    """test 整体 RMS 贪心选型(2026-10-21 定案,依据 docs/findings.md 结论 2)。"""

    def test_real_data_shape_picks_m2(self):
        """本批数据的真实形状(m1→m4 递复杂):m2 提升 46.7% 升级,m3/m4
        只好 1.2%–1.9% 不够 10% 门槛 → 选 m2;走查流水逐步落账。"""
        out = select_by_test_rms(
            {"m1": 0.3020, "m2": 0.1598, "m3": 0.1569, "m4": 0.1587},
            ("m1", "m2", "m3", "m4"),
            0.10,
        )
        assert out["selected"] == "m2"
        assert out["test_rms_px"] == {"m1": 0.3020, "m2": 0.1598, "m3": 0.1569, "m4": 0.1587}
        assert [w["moved"] for w in out["walk"]] == [True, False, False]
        assert out["walk"][0]["improvement"] == pytest.approx((0.3020 - 0.1598) / 0.3020)
        assert out["walk"][1]["improvement"] == pytest.approx((0.1598 - 0.1569) / 0.1598)
        assert "m2" in out["reason"]

    def test_not_adjacent_only_bad_middle_does_not_bury_good_later(self):
        """走查与当前模型比(不与上一步比):中间档 m2 不行不埋没 m3。"""
        out = select_by_test_rms(
            {"m1": 0.30, "m2": 0.29, "m3": 0.15}, ("m1", "m2", "m3"), 0.10
        )
        assert out["selected"] == "m3"
        assert [w["moved"] for w in out["walk"]] == [False, True]
        # m3 的提升按"相对当前模型 m1"算,不是相对 m2
        assert out["walk"][1]["improvement"] == pytest.approx((0.30 - 0.15) / 0.30)

    def test_stops_when_nothing_beats_current(self):
        """全部后续档都不够门槛 → 停在最简模型(简单优先)。"""
        out = select_by_test_rms(
            {"m1": 0.20, "m2": 0.19, "m3": 0.185}, ("m1", "m2", "m3"), 0.10
        )
        assert out["selected"] == "m1"
        assert all(not w["moved"] for w in out["walk"])
        assert "最简" in out["reason"]

    def test_none_scores_skipped_and_recorded(self):
        """无成绩(None)的模型跳过、记进走查流水,不参与升级比较。"""
        out = select_by_test_rms(
            {"m1": 0.30, "m2": None, "m3": 0.15}, ("m1", "m2", "m3"), 0.10
        )
        assert out["selected"] == "m3"
        assert out["walk"][0]["improvement"] is None and not out["walk"][0]["moved"]
        assert "没有" in out["walk"][0]["note"]

    def test_all_fail_no_forced_selection(self):
        """全部模型无成绩 → 不硬选,给中文理由。"""
        out = select_by_test_rms(
            {"m1": None, "m2": None}, ("m1", "m2"), 0.10
        )
        assert out["selected"] is None
        assert out["walk"] == []
        assert "不硬选" in out["reason"]

    def test_single_model_walk_empty(self):
        """只有一个候选:直接当选,无走查。"""
        out = select_by_test_rms({"m1": 0.25}, ("m1",), 0.10)
        assert out["selected"] == "m1" and out["walk"] == []

    def test_zero_gate_takes_any_improvement(self):
        """门槛 0 = 任何提升都升级(合法边界)。"""
        out = select_by_test_rms(
            {"m1": 0.30, "m2": 0.299}, ("m1", "m2"), 0.0
        )
        assert out["selected"] == "m2"

    def test_inputs_validated_in_chinese(self):
        """缺条目 / 非有限值 / 门槛越界 / 空名单 → 全部中文报错。"""
        with pytest.raises(ValueError, match="缺"):
            select_by_test_rms({"m1": 0.2}, ("m1", "m2"), 0.10)
        with pytest.raises(ValueError, match="非有限"):
            select_by_test_rms({"m1": float("nan")}, ("m1",), 0.10)
        with pytest.raises(ValueError, match="improve_min"):
            select_by_test_rms({"m1": 0.2}, ("m1",), 1.0)
        with pytest.raises(ValueError, match="空表"):
            select_by_test_rms({}, (), 0.10)
