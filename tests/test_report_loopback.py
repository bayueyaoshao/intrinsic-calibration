"""回环建议规则引擎测试(合成最小判定段,白天快速档,秒级)。

测的是 06-02 落地的 loopback.build_loopback_advice 纯函数:
1. 逐条单触发:十类触发条件(三判据各一条、误差预算、残差外推区、
   敏感度建议、test 三分量解不出、角点寻优更优候选、连拍各向异性)
   各自单独满足时,恰好产出一条对应「回哪一步」的建议行;
2. 全部达标:所有判定段齐且全部通过 → 只给一行「无需回环」;
3. 判据缺席:判定段落账为 null → 该条规则跳过并记「判据缺席」行,
   不冒充「无需回环」;
4. 步骤名纪律:建议行指向的步骤名必须出自 docs/pipeline.md 八步
   (①–⑧ 中文名),不允许自造路标。

输入全部是合成字典(照 evaluate.json / analyze.json 真实键名缩小),
函数零文件读写。
"""

from __future__ import annotations

from calib.report.loopback import (
    PIPELINE_STEPS,
    build_loopback_advice,
)

# ---------------------------------------------------------------- 合成判定段


def _eval_doc(gen_pass: bool = True, stab_pass: bool = True,
              crlb_pass: bool = True, selected: str | None = "m2") -> dict:
    """最小 evaluate 判定段:selection.selected + verdicts 当选模型行。"""
    return {
        "selection": {
            "selected": selected,
            "verdicts": [{
                "model_id": "m2",
                "generalization": {"pass": gen_pass},
                "stability": {"pass": stab_pass},
                "crlb": {"pass": crlb_pass},
            }],
        },
    }


def _corner_opt(production_rms: float, rival: tuple[str, float] | None = None,
                production: str = "sb") -> dict:
    """最小 corner_opt 判定段:生产口径 + 可选的更稳候选。

    每个方法的 12 个非校准单元自参照 RMS 全取同一个值,让「未显著
    劣化单元自参照平均」退化成这个值本身(阈值 0.08,低于它的单元
    全部计入)。
    """
    thr = 0.08
    stats = {"_self": {}}
    methods = {"sb": production_rms}
    if rival:
        methods[rival[0]] = rival[1]
    for name, rms in methods.items():
        stats["_self"][name] = {
            f"unit_{i}": {"rms_px": rms} for i in range(12)
        }
    return {
        "production_method": production,
        "calibration_unit": "noise_0.0",
        "threshold_px": thr,
        "stats": stats,
    }


def _analyze_doc(*, budget_fail: bool = False, has_gap: bool = False,
                 advice: list | None = None, solve_failed: int = 0,
                 corner_opt: dict | None = None,
                 aniso_median: float = 1.0) -> dict:
    """最小 analyze 六段:全部默认「达标」,按参数逐项改成触发态。"""
    return {
        "budget": {"verdict": "fail" if budget_fail else "pass"},
        "residual_structure": {"train": {"coverage": {
            "r_max_obs": 0.48 if has_gap else 0.8,
            "r_corner_frame": 0.8,
        }}},
        "sensitivity": {"advice": advice or []},
        "test_xyz": {"n_solve_failed": solve_failed},
        "corner_opt": corner_opt if corner_opt is not None else _corner_opt(0.02),
        "anisotropy": {"summary": {"median": aniso_median,
                                   "significant_line": 1.5}},
    }


# 全达标参照:三判据 PASS + analyze 六段全部不触发
_ALL_PASS_ANALYZE = _analyze_doc()


def _rows_of_kind(rows: list[dict], kind: str) -> list[dict]:
    return [r for r in rows if r["kind"] == kind]


# =====================================================================
# 1. 逐条单触发
# =====================================================================
def test_each_rule_triggers_individually():
    cases = [
        # (本条单独改动的判定段, 期望建议行数, 期望步骤名, 期望触发文案关键词)
        ("泛化FAIL", _eval_doc(gen_pass=False), _ALL_PASS_ANALYZE,
         1, "⑥ 防作弊考试", "泛化"),
        ("稳定FAIL", _eval_doc(stab_pass=False), _ALL_PASS_ANALYZE,
         1, "① 照片入库体检", "稳定"),
        ("逼近CRLB FAIL", _eval_doc(crlb_pass=False), _ALL_PASS_ANALYZE,
         1, "③ 测噪声底", "CRLB"),
        ("预算未达标", _eval_doc(), _analyze_doc(budget_fail=True),
         1, "③ 测噪声底", "预算"),
        ("外推区", _eval_doc(), _analyze_doc(has_gap=True),
         1, "① 照片入库体检", "外推"),
        ("敏感度建议", _eval_doc(),
         _analyze_doc(advice=[{"direction": "k2/k3 高阶径向",
                               "evidence": "边角零观测",
                               "advice": "补 6 个够到四角的姿态"}]),
         1, "① 照片入库体检", "k2/k3 高阶径向"),
        ("test解不出", _eval_doc(),
         _analyze_doc(solve_failed=2),
         1, "② 找棋盘角点", "解不出"),
        ("寻优更优候选", _eval_doc(),
         _analyze_doc(corner_opt=_corner_opt(0.06, rival=("classic_w7", 0.02))),
         1, "② 找棋盘角点", "classic_w7"),
        ("各向异性", _eval_doc(), _analyze_doc(aniso_median=2.0),
         1, "① 照片入库体检", "各向异性"),
    ]
    for name, eval_doc, analyze_doc, n, step, keyword in cases:
        rows = _rows_of_kind(
            build_loopback_advice(eval_doc, analyze_doc), "建议")
        assert len(rows) == n, f"{name}:应恰有 {n} 条建议行,实际 {len(rows)}"
        assert rows[0]["step"] == step, \
            f"{name}:步骤名应为「{step}」,实际「{rows[0]['step']}」"
        assert keyword in rows[0]["trigger"], \
            f"{name}:触发判据应含「{keyword}」,实际「{rows[0]['trigger']}」"
        assert rows[0]["action"], f"{name}:做什么一栏不能为空"


def test_sensitivity_advice_each_item_one_row():
    """敏感度 advice 有几条就转几行,每行都指向回①补拍。"""
    advice = [
        {"direction": "方向1 k2/k3", "evidence": "e1", "advice": "补边角姿态"},
        {"direction": "方向2 fx/fy", "evidence": "e2", "advice": "补转角姿态"},
    ]
    rows = _rows_of_kind(
        build_loopback_advice(_eval_doc(), _analyze_doc(advice=advice)), "建议")
    assert len(rows) == 2
    assert all(r["step"] == "① 照片入库体检" for r in rows)
    for item, row in zip(advice, rows):
        assert item["direction"] in row["trigger"]
        assert item["advice"] in row["action"]


# =====================================================================
# 2. 全部达标 → 单行「无需回环」
# =====================================================================
def test_all_pass_gives_single_no_loopback_row():
    rows = build_loopback_advice(_eval_doc(), _ALL_PASS_ANALYZE)
    assert len(rows) == 1, f"全达标只应有一行,实际 {len(rows)}:{rows}"
    assert rows[0]["kind"] == "达标"
    assert "无需回环" in rows[0]["action"]
    assert "三判据达标" in rows[0]["action"]


def test_triggered_state_has_no_all_pass_row():
    """有触发行时绝不能再给「无需回环」(自相矛盾的建议等于没建议)。"""
    rows = build_loopback_advice(_eval_doc(stab_pass=False), _ALL_PASS_ANALYZE)
    assert not _rows_of_kind(rows, "达标")


# =====================================================================
# 3. 判据缺席 → 跳过该条规则并记缺席,不冒充「无需回环」
# =====================================================================
def test_absent_sections_recorded_and_no_all_pass():
    rows = build_loopback_advice({}, {})
    assert not _rows_of_kind(rows, "达标"), "判据全缺席时不得给「无需回环」"
    assert not _rows_of_kind(rows, "建议")
    absent = _rows_of_kind(rows, "判据缺席")
    # evaluate 三判据 1 处 + analyze 六段 6 处 = 7 处缺席
    assert len(absent) == 7, f"应记 7 处判据缺席,实际 {len(absent)}:{absent}"
    joined = " ".join(r["trigger"] for r in absent)
    for keyword in ("三判据", "误差预算", "残差", "敏感度",
                    "test 三分量", "角点寻优", "各向异性"):
        assert keyword in joined, f"缺席记录缺「{keyword}」:{joined}"


def test_null_sections_same_as_missing():
    """上游落账为 null(而不是键缺失)走同一条缺席路径。"""
    rows = build_loopback_advice({"selection": None},
                                 {"budget": None, "residual_structure": None,
                                  "sensitivity": None, "test_xyz": None,
                                  "corner_opt": None, "anisotropy": None})
    assert len(_rows_of_kind(rows, "判据缺席")) == 7
    assert not _rows_of_kind(rows, "达标")


# =====================================================================
# 4. 步骤名纪律:路标必须出自 docs/pipeline.md 八步
# =====================================================================
def test_advice_steps_are_pipeline_step_names():
    all_rows = build_loopback_advice(_eval_doc(gen_pass=False, stab_pass=False,
                                               crlb_pass=False),
                                     _analyze_doc(budget_fail=True,
                                                  has_gap=True,
                                                  solve_failed=1,
                                                  corner_opt=_corner_opt(
                                                      0.06,
                                                      rival=("classic_w7", 0.02)),
                                                  aniso_median=2.0))
    allowed = set(PIPELINE_STEPS) | {"—"}
    for r in all_rows:
        assert r["step"] in allowed, \
            f"建议行步骤名「{r['step']}」不在八步路标内:{PIPELINE_STEPS}"


def test_pipeline_steps_match_docs():
    """八步路标常量与 docs/pipeline.md 的步骤名逐字一致(硬性纪律)。"""
    assert tuple(PIPELINE_STEPS) == (
        "① 照片入库体检",
        "② 找棋盘角点",
        "③ 测噪声底",
        "④ 求解相机内参",
        "⑤ 算理论精度极限",
        "⑥ 防作弊考试",
        "⑦ 误差归因",
        "⑧ 汇总证据出报告",
    )
