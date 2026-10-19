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
    mad_outlier_poses,
    paired_comparison,
)

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
