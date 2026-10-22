"""回环建议规则引擎(ITER-04):判定段 → 「回哪一步、做什么」的自动建议。

输入是 evaluate.json 与 analyze.json 两个判定文档(evaluate 轮的三判据
判行 + analyze 轮六段判定),输出一列建议行——每行 =
{kind, trigger(触发判据,判据原文中文), step(回哪一步), action(做什么)}。
步骤名出自 docs/pipeline.md 的八步路标(①–⑧),全部走常量表,不允许
自造路标——报告里给的回头路必须和项目流程图对得上。

三条硬规矩:
1. 纯函数零 IO:只读传入的字典,不碰任何文件;
2. 判定段缺席(落账为 null / 键缺失)→ 对应规则跳过,并记一行
   「判据缺席」如实披露,绝不冒充「无需回环」;
3. 全部判定段齐且全部达标 → 只给一行「无需回环」结论,不硬凑建议。

「回哪一步」的白话:流水线每一步的结论都落盘、可以单独重跑,回环就是
改了上游之后只重跑受影响的下游环节,不必从头再来(docs/pipeline.md)。
"""

from __future__ import annotations

# 回环路标:步骤名与 docs/pipeline.md 八步逐字一致(硬性纪律,有测试锁定)
PIPELINE_STEPS = (
    "① 照片入库体检",
    "② 找棋盘角点",
    "③ 测噪声底",
    "④ 求解相机内参",
    "⑤ 算理论精度极限",
    "⑥ 防作弊考试",
    "⑦ 误差归因",
    "⑧ 汇总证据出报告",
)

STEP_INGEST = PIPELINE_STEPS[0]
STEP_CORNERS = PIPELINE_STEPS[1]
STEP_NOISE = PIPELINE_STEPS[2]
STEP_CALIBRATE = PIPELINE_STEPS[3]
STEP_FISHER = PIPELINE_STEPS[4]
STEP_EVALUATE = PIPELINE_STEPS[5]
STEP_ANALYZE = PIPELINE_STEPS[6]
STEP_REPORT = PIPELINE_STEPS[7]

# 建议行类型:建议(触发了一条规则)/ 判据缺席(判定段不在,如实记)/
# 达标(全部判定齐且全过,单行「无需回环」)
ROW_ADVICE = "建议"
ROW_ABSENT = "判据缺席"
ROW_ALL_PASS = "达标"

# 角点寻优换候选的两条门槛(口径照 analyze 轮判读原文「未显著劣化单元
# 自参照平均」):候选要同时满足 ①未显著劣化单元数 ≥ 3(不能靠个别单元
# 的侥幸);②自参照平均比生产口径低 ≥ 30%(肉眼可辨的差距,不是抖动)
_CORNER_OPT_MIN_OK_UNITS = 3
_CORNER_OPT_BETTER_RATIO = 0.7

# 残差外推区的判定余量:最远观测离画面四角差超过这个比例才算「够不到」
_COVER_GAP_MIN = 0.01


def _absent_row(section_cn: str) -> dict:
    return {
        "kind": ROW_ABSENT,
        "trigger": f"{section_cn}(本轮缺席)",
        "step": "—",
        "action": (f"判定段「{section_cn}」本轮缺席(落账为空),"
                   f"对应的回环建议给不出来;先补跑产出该判定的命令,再重出报告。"),
    }


def _advice_row(trigger: str, step: str, action: str) -> dict:
    return {"kind": ROW_ADVICE, "trigger": trigger, "step": step,
            "action": action}


def _selection_verdict_row(evaluate_report: dict | None) -> dict | None:
    """从 evaluate 判定段取「当选模型」的判行行;选型/判行缺席给 None。"""
    selection = ((evaluate_report or {}).get("selection") or {})
    selected = selection.get("selected")
    if not selected:
        return None
    for row in selection.get("verdicts") or []:
        if row.get("model_id") == selected:
            return row
    return None


def _three_criteria_rows(verdict_row: dict) -> list[dict]:
    """三判据各一条规则:泛化 FAIL 回⑥复核,稳定 FAIL 回①补拍,
    逼近 CRLB FAIL 回③核对口径(路标照 docs/pipeline.md 回头路)。"""
    rows: list[dict] = []
    gen = verdict_row.get("generalization") or {}
    stab = verdict_row.get("stability") or {}
    crlb = verdict_row.get("crlb") or {}
    if gen.get("pass") is False:
        rows.append(_advice_row(
            "泛化判据 FAIL:留出考试成绩明显差于训练成绩,有死记硬背(过拟合)迹象",
            STEP_EVALUATE,
            "先用 full 档回⑥复核考试(日常 quick 档考得少,结论存疑时按配置"
            "注释回 full 全量复核);确认过拟合后回④求解相机内参换更简单的"
            "模型重标,或回①补拍姿态更多样的照片扩大覆盖。"))
    if stab.get("pass") is False:
        rows.append(_advice_row(
            "稳定判据 FAIL:换一批照片重新标定,参数挪动超过门槛(参数在晃)",
            STEP_INGEST,
            "先看报告第⑦节误差归因定晃动的大头:换姿态主导时回①补拍更多"
            "摆拍姿态(覆盖面上去晃动自然下来);检测噪声主导时回③测噪声底"
            "核对噪声口径,再决定要不要重拍。"))
    if crlb.get("pass") is False:
        rows.append(_advice_row(
            "逼近 CRLB 判据 FAIL:实测晃动离理论下界还有大段距离,"
            "数据潜力没用足",
            STEP_NOISE,
            "回③测噪声底核对检测器口径 σ(理论下界全靠它乘出来,口径错配会"
            "同时冤枉或放过模型),再回⑤算理论精度极限复核观测集口径;"
            "口径都对还差得远,按报告第⑦节建议回①补拍。"))
    return rows


def _budget_row(analyze_report: dict | None) -> list[dict]:
    """误差预算判行:fail = 实测贴近甚至低于理论下界,物理上不可能,
    几乎总是噪声口径错配(CRLB 绝不允许大于实测——铁律)。"""
    budget = (analyze_report or {}).get("budget")
    if budget is None:
        return [_absent_row("误差预算(analyze 轮)")]
    if budget.get("verdict") != "fail":
        return []
    return [_advice_row(
        "误差预算未达标:对账告警——实测晃动贴近甚至低于理论下界,"
        "物理上不可能,几乎总是口径错配",
        STEP_NOISE,
        "回③测噪声底核对检测器口径 σ(拿更松的时域口径会算出虚低、"
        "更「准」的下界),回⑤算理论精度极限复核观测集口径;口径对齐之前,"
        "这轮的预算数字不许进任何结论。")]


def _residual_row(analyze_report: dict | None) -> list[dict]:
    """残差方向分解的外推区:最远观测够不到画面四角 → 补拍边缘覆盖位姿。"""
    residual = (analyze_report or {}).get("residual_structure")
    if residual is None:
        return [_absent_row("残差方向分解(analyze 轮)")]
    coverage = ((residual.get("train") or {}).get("coverage") or {})
    r_obs = coverage.get("r_max_obs")
    r_corner = coverage.get("r_corner_frame")
    if r_obs is None or r_corner is None:
        return []
    if float(r_corner) - float(r_obs) <= _COVER_GAP_MIN:
        return []
    return [_advice_row(
        f"残差外推区:最远观测 r={float(r_obs):.3f},画面四角 r="
        f"{float(r_corner):.3f},中间没有照片,畸变在这段全靠外推",
        STEP_INGEST,
        "回①补拍能够到画面边缘与四角的摆拍位姿(参考第⑦节建议:目标"
        "最远观测 r 往 0.75 靠,每姿态 1–2 张即可,人工决定;补拍后照片按"
        "目录契约放进 input_data,从②起重跑受影响环节)。")]


def _sensitivity_rows(analyze_report: dict | None) -> list[dict]:
    """参数敏感度建议:每条原文转成一行「回①补拍哪类位姿」。"""
    sensitivity = (analyze_report or {}).get("sensitivity")
    if sensitivity is None:
        return [_absent_row("参数敏感度(analyze 轮)")]
    rows = []
    for item in sensitivity.get("advice") or []:
        direction = str(item.get("direction") or "未命名方向")
        rows.append(_advice_row(
            f"参数敏感度建议:{direction}",
            STEP_INGEST,
            f"第⑦节建议原文:「{item.get('advice') or '—'}」(证据:"
            f"{item.get('evidence') or '—'})。若采纳,回①补拍对应类型的"
            f"位姿(人工决定),补拍后从②起重跑受影响环节;不采纳则维持"
            f"现状,并在下一份报告里说明理由。"))
    return rows


def _test_xyz_row(analyze_report: dict | None) -> list[dict]:
    """test 三分量:有考题帧解不出 → 回查该批帧的角点质量。"""
    test_xyz = (analyze_report or {}).get("test_xyz")
    if test_xyz is None:
        return [_absent_row("test 三分量(analyze 轮)")]
    n_failed = test_xyz.get("n_solve_failed") or 0
    if not n_failed:
        return []
    return [_advice_row(
        f"test 三分量有 {n_failed} 帧解不出(位姿反解失败,考题作废)",
        STEP_CORNERS,
        "回②复核这批照片的角点质量(看角点叠加副本与存疑复核包),"
        "确认是检出问题就换角点方法重提;个别照片本身拍坏的人工确认后"
        "按剔除名单处理。")]


def _corner_opt_row(analyze_report: dict | None) -> list[dict]:
    """角点寻优:存在明显更稳的候选方法 → 回②换候选重提。

    口径照 analyze 轮判读原文:每方法取「未显著劣化单元(自参照 RMS
    低于随段落盘的阈值)」的自参照平均,候选比生产口径低 ≥ 30% 且
    未显著劣化单元数不少于 3,才算「非当前配置最优」。
    """
    corner_opt = (analyze_report or {}).get("corner_opt")
    if corner_opt is None:
        return [_absent_row("角点寻优(analyze 轮)")]
    production = corner_opt.get("production_method")
    threshold = corner_opt.get("threshold_px")
    self_stats = ((corner_opt.get("stats") or {}).get("_self") or {})
    calib_unit = corner_opt.get("calibration_unit") or "noise_0.0"
    if not production or not threshold or production not in self_stats:
        return []

    def _undegraded_mean(method: str) -> tuple[float | None, int]:
        units = self_stats.get(method) or {}
        vals = [float(u.get("rms_px"))
                for name, u in units.items()
                if name != calib_unit and u.get("rms_px") is not None
                and float(u.get("rms_px", 0)) < float(threshold)]
        if not vals:
            return None, 0
        return sum(vals) / len(vals), len(vals)

    prod_mean, _prod_n = _undegraded_mean(production)
    if prod_mean is None:
        return []
    best_name, best_mean, best_n = None, None, 0
    for method in sorted(self_stats):
        if method == production:
            continue
        mean, n_ok = _undegraded_mean(method)
        if mean is None or n_ok < _CORNER_OPT_MIN_OK_UNITS:
            continue
        if mean < prod_mean * _CORNER_OPT_BETTER_RATIO and \
                (best_mean is None or mean < best_mean):
            best_name, best_mean, best_n = method, mean, n_ok
    if best_name is None:
        return []
    return [_advice_row(
        f"角点寻优显示候选 {best_name} 比当前生产口径 {production} 更稳"
        f"(未显著劣化单元自参照平均 {best_mean:.4f}px 对 "
        f"{prod_mean:.4f}px,{best_n} 个单元计入)",
        STEP_CORNERS,
        f"回②找棋盘角点把方法候选换成 {best_name} 重提角点,再从④起重标;"
        f"换之前先对照第③节寻优曲线确认差距在意的参数方向上成立"
        f"(只提示方向,不自动改配置)。")]


def _anisotropy_row(analyze_report: dict | None) -> list[dict]:
    """连拍各向异性:横纵定位散布之比的中位越过显著线 → 拍摄端稳有问题。"""
    anisotropy = (analyze_report or {}).get("anisotropy")
    if anisotropy is None:
        return [_absent_row("连拍各向异性(analyze 轮)")]
    summary = anisotropy.get("summary") or {}
    median = summary.get("median")
    line = summary.get("significant_line")
    if median is None or line is None or float(median) <= float(line):
        return []
    return [_advice_row(
        f"连拍各向异性显著:横纵定位散布之比中位 {float(median):.2f},"
        f"越过显著线 {float(line):.1f}(横竖不对称,常见原因是拍摄瞬间"
        f"有方向性的抖动)",
        STEP_INGEST,
        "回①复核拍摄稳定性(三脚架/延时快门,排除一个方向的晃动),"
        "必要时重拍散布最大的几个姿态;也可先回③测噪声底确认两个口径"
        "的差有没有异常放大。")]


def build_loopback_advice(evaluate_report: dict | None,
                          analyze_report: dict | None) -> list[dict]:
    """回环建议主入口:依序过内置规则表,收集触发行。

    参数:
        evaluate_report: evaluate.json 的内容(可 None / 判定段可缺席);
        analyze_report:  analyze.json 的内容(可 None / 六段可各自缺席)。

    返回:建议行列表,每行 {kind, trigger, step, action}:
        - kind「建议」:触发了某条规则,step 指向回哪一步;
        - kind「判据缺席」:判定段不在,该条规则跳过并如实记录;
        - kind「达标」:全部判定齐且全过时唯一一行「无需回环」。
    """
    rows: list[dict] = []

    # ---- evaluate 三判据判行(三选型判据各一条规则)----
    verdict_row = _selection_verdict_row(evaluate_report)
    if verdict_row is None:
        rows.append(_absent_row("三判据判行(evaluate 轮选型判定)"))
    else:
        rows.extend(_three_criteria_rows(verdict_row))

    # ---- analyze 六段判定 ----
    rows.extend(_budget_row(analyze_report))
    rows.extend(_residual_row(analyze_report))
    rows.extend(_sensitivity_rows(analyze_report))
    rows.extend(_test_xyz_row(analyze_report))
    rows.extend(_corner_opt_row(analyze_report))
    rows.extend(_anisotropy_row(analyze_report))

    # ---- 全部达标(无触发行、无缺席行)→ 单行「无需回环」----
    if not any(r["kind"] in (ROW_ADVICE, ROW_ABSENT) for r in rows):
        rows.append({
            "kind": ROW_ALL_PASS,
            "trigger": "全部判定达标",
            "step": "—",
            "action": "无需回环:三判据达标且误差逼近理论下界,角点、覆盖、"
                      "噪声各段也没有需要回头处理的问题;维持当前配置继续用。",
        })
    return rows
