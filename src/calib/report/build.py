"""报告组装纯函数层:读七路上游最新轮,把全部中文文案定稿成一个 context 字典。

流水线第⑧步「汇总证据出报告」采用「组装与排版分离」的两层结构:
- 本文件只负责「组装」:逐段找各阶段最新一轮产物(只读不重算),把全部
  中文文案、排名与换算在这里定稿成一个字典(context);本层不写任何盘。
- report/templates/ 下的双模板(HTML/Markdown)只负责「排版」:模板里
  不写任何本数据集的数字,换数据集不用改模板(可复用性 REU-02)。

context 的两条硬规矩:
1. 报告里出现的每个数字都是本层定稿的字符串——报告与快照底账
   report_data.json 逐位同源,「报告里的数」永远能在底账里原样找到;
   metrics_raw 一小节保留机器可读的原始数(跨运行对比专用,模板不渲染它)。
2. 缺席段、失败轮、血缘已清理的上游都如实记进 disclosures 披露清单,
   绝不冒充数值;只有「直接取数的主文件整个缺失」才抛中文错误退出——
   拿不存在的数据出报告等于伪造证据。
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from calib.io.config import load_board
from calib.io.manifest import latest_run_dir

# 棋盘规格的唯一来源(数据契约:board.yaml 是棋盘参数唯一来源,REU-03)
BOARD_PATH = Path("config") / "board.yaml"

# 双模板所在目录(随包分发;模板只做排版,数字一律来自 context)
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# 七路上游:阶段目录名 → 主文件名 + 通俗名。七份主文件都是报告的直接取数
# 主文件,缺任何一份都中文报错退出;轮内小段的缺席才走披露(不冒充数值)。
_UPSTREAM = (
    ("ingest", "manifest.json", "照片入库体检"),
    ("corners", "detection_summary.json", "找棋盘角点"),
    ("noise", "noise.json", "测噪声底"),
    ("calibration", "comparison.json", "求解相机内参"),
    ("fisher", "crlb.json", "算理论精度极限"),
    ("evaluate", "evaluate.json", "防作弊考试"),
    ("analyze", "analyze.json", "误差归因"),
)

# 九节标题(报告骨架,照 docs/pipeline.md 八步叙事排布;HTML/MD 模板共用)
SECTION_TITLES = (
    "① 结论速览",
    "② 数据与体检",
    "③ 角点",
    "④ 噪声底",
    "⑤ 理论极限",
    "⑥ 防作弊考试",
    "⑦ 误差归因",
    "⑧ 回环建议",
    "⑨ 产物索引与血缘链",
)


def _fmt(value, nd: int = 4) -> str:
    """数字 → 定稿字符串:None/非有限值一律「—」;常规值定点小数,
    很小的值(绝对值 < 0.001,如 p1/p2 的标准差)转科学计数,避免
    「0.0000」这种抹掉量级信息的显示。"""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "—"
    if abs(f) >= 0.001:
        return f"{f:.{nd}f}"
    return f"{f:.{nd - 1}e}"


def _pct(value) -> str:
    """占比 → 百分比字符串(None → 「—」)。"""
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _flag_cn(flag: str | None) -> str:
    """Fisher 谱体检的档位记号 → 中文档位(陌生取值原样透出,不吞信息)。"""
    return {"": "正常", "warning": "警戒", "degenerate": "退化"}.get(
        flag or "", str(flag or "—"))


def _verdict_word(passed) -> str:
    """布尔判定 → PASS/FAIL(缺素材给 None,由调用方决定披露方式)。"""
    if passed is None:
        return None
    return "PASS" if passed else "FAIL"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _as_dict(doc) -> dict:
    """上游文档统一按字典读:comparison.json 这类行列表自动变空字典,
    让失败轮/血缘检查对七种产物形状都安全。"""
    return doc if isinstance(doc, dict) else {}


def _median_text(values) -> str:
    """一组数的中位数 → 定稿字符串(空表给「—」)。"""
    if not values:
        return "—"
    return _fmt(statistics.median(float(v) for v in values if v is not None))


def _obs_set_note() -> str:
    """crlb.json 观测集键名的白话注解(键名不改名,注解就地讲清,REU-01)。"""
    return (
        "键名注解:views640 = 全帧版(把训练照片逐张全部计入的口径),"
        "poses80 = 姿态聚合版(每个姿态聚成一处的对照口径);"
        "键名里的数字是历史命名,不代表本数据的张数,换数据后含义不变。"
    )


def _plan_figures(rounds: dict, selected: str | None) -> list[dict]:
    """编排附录图表清单:全部直接编号引用各上游轮既有 PNG(报告层不新画图),
    源文件不在盘的条目直接不进清单(如实缺席,不放空链接)。
    残差热图 = 分带曲线 + 覆盖足迹两件既有图(2026-10-22 用户裁定①)。"""
    corners_run = rounds["corners"]["run"]
    noise_run = rounds["noise"]["run"]
    fisher_run = rounds["fisher"]["run"]
    eval_run = rounds["evaluate"]["run"]
    analyze_run = rounds["analyze"]["run"]
    plan = [
        (corners_run / "mosaic" / "overview_mosaic.png", "③",
         "角点叠加:角点标记总览拼图(常态抽样 + 最差格 + 全部存疑,细节回单张副本放大看)"),
        (corners_run / "mosaic" / "suspect_mosaic.png", "③",
         "角点叠加:存疑照片拼图(未标记原貌,与总览拼图对照看)"),
        (noise_run / "transfer_curve.png", "④",
         "噪声底:像素噪声 → 角点误差的传递曲线(注入实验实测)"),
        (analyze_run / "figures" / "corner_opt_self.png", "③",
         "角点寻优曲线(主口径:各方法对自家干净检出的偏移,量退化敏感度)"),
        (analyze_run / "figures" / "corner_opt_prodsb.png", "③",
         "角点寻优曲线(旁证口径:对生产 SB 干净检出的绝对偏移,跨方法可比)"),
        (fisher_run / "figures" / f"spectrum_{selected}_views640_detector.png", "⑤",
         "CRLB 对比:当选模型全帧版(views640)检测器口径的特征谱"),
        (fisher_run / "figures" / f"spectrum_{selected}_views640_temporal.png", "⑤",
         "CRLB 对比:当选模型全帧版(views640)时域口径的特征谱"),
        (eval_run / "figures" / "cv_errorbars.png", "⑥",
         "CV 曲线:按姿态分组交叉验证的 held-out 误差(留出考试的成绩棒)"),
        (eval_run / "figures" / "dispersion_compare.png", "⑥",
         "离散对比:bootstrap 重抽样与重复张分组的参数晃动并排"),
        (eval_run / "figures" / "lopo_pose_errors.png", "⑥",
         "留一姿态:逐姿态剔除重标的误差散点(点名问题姿态)"),
        (analyze_run / "figures" / "budget_bars.png", "⑦",
         "误差预算:逐参数实测晃动 vs 理论下界柱状对比"),
        (analyze_run / "figures" / "radial_bands.png", "⑦",
         "残差分带:残差随半径分带的曲线(残差热图的替代证据,2026-10-22 裁定①)"),
        (analyze_run / "figures" / "coverage_footprint.png", "⑦",
         "摆位覆盖:观测足迹与画面四角的覆盖图(r⁴ 覆盖足迹,裁定①的另一件)"),
        (analyze_run / "figures" / "test_xyz.png", "⑦",
         "test 三分量对比:逐方向位移与 z 分数"),
        (analyze_run / "figures" / "sensitivity_ranking.png", "⑦",
         "敏感度排序:参数组合方向按像素当量排序"),
        (analyze_run / "figures" / "anisotropy_map.png", "⑦",
         "各向异性地图:横纵定位散布之比在画面上的分布"),
        (analyze_run / "figures" / "anisotropy_hist.png", "⑦",
         "各向异性直方图:逐角点横纵比的分布"),
    ]
    figures: list[dict] = []
    for source, section, caption in plan:
        if not source.is_file():
            continue
        figures.append({
            "idx": f"图{len(figures) + 1}",
            "file": source.name,
            "caption": caption,
            "source": f"{source.parent.parent.name} 轮 {source.parent.name}/",
            "section": section,
            "_src": str(source),  # 私有键:命令层复制用,模板不渲染
        })
    return figures


def build_report_context(out_root: Path) -> dict:
    """组装报告 context:逐段定位七路上游最新轮,只读不重算,产出全中文
    且可直接 json 序列化的字典。缺直接取数主文件时抛中文 ValueError。

    参数:
        out_root: 阶段输出根(约定 output/,相对当前工作目录)。
    """
    out_root = Path(out_root)
    disclosures: list[dict] = []

    def _disclose(kind: str, where: str, text: str) -> None:
        disclosures.append({"kind": kind, "where": where, "text": text})

    # ---- 七路上游定位与读取(缺主文件当场中文报错,不静默出假报告)----
    rounds: dict[str, dict] = {}
    for stage, fname, _cn in _UPSTREAM:
        run = latest_run_dir(out_root, stage)
        main = run / fname
        if not main.is_file():
            raise ValueError(
                f"上游产物缺失:{main}。报告要从它取数,"
                f"请先运行 calib {stage} 再来。")
        rounds[stage] = {"run": run, "name": run.name, "main": main,
                         "doc": _load_json(main)}

    manifest_doc = rounds["ingest"]["doc"]
    corners_doc = rounds["corners"]["doc"]
    noise_doc = rounds["noise"]["doc"]
    crlb_doc = rounds["fisher"]["doc"]
    eval_doc = rounds["evaluate"]["doc"]
    analyze_doc = rounds["analyze"]["doc"]

    # ---- 棋盘规格(读不了不拦报告:物理换算尺缺席并披露)----
    board = None
    try:
        board = load_board(BOARD_PATH)
    except (ValueError, OSError) as exc:
        _disclose("配置缺席", SECTION_TITLES[0],
                  f"config/board.yaml 读不了({exc}),物理换算尺缺席,"
                  f"报告中的像素数字不折算实物。")

    # ---- 物理换算尺(零硬编码:方格实物边长 ÷ 照片里方格像素边长中位)----
    sq_median = None
    det_train_sb = ((corners_doc.get("detection") or {}).get("train") or {}).get("sb") or {}
    det_test_sb = ((corners_doc.get("detection") or {}).get("test") or {}).get("sb") or {}
    for block in (det_train_sb, det_test_sb):
        sq = block.get("sq_px") or {}
        if sq.get("median") is not None:
            sq_median = float(sq["median"])
            break
    mm_per_px = None
    if board is not None and sq_median:
        mm_per_px = float(board.square_size_mm) / sq_median

    # ---- 失败轮登记(失败轮数值字段是 null,相应各节走缺席披露)----
    # 注意 comparison.json 是行列表不是字典,统一经 _as_dict 守卫
    failures: dict[str, str] = {}
    for stage, _fname, cn in _UPSTREAM:
        note = _as_dict(rounds[stage]["doc"]).get("failure_note")
        if note:
            failures[stage] = str(note)
            _disclose("失败轮", SECTION_TITLES[8],
                      f"上游「{cn}」({stage})本轮是失败轮:{note}")

    # =====================================================================
    # 第①节 结论速览:当选模型 + 内参/不确定度表 + 三判据判行 + 误差量级
    # =====================================================================
    selection = (eval_doc.get("selection") or {}) if "evaluate" not in failures else {}
    selected = selection.get("selected") or None
    delivery_models = ((eval_doc.get("delivery") or {}).get("models") or {}) \
        if "evaluate" not in failures else {}
    delivery_entry = delivery_models.get(selected) or {} if selected else {}

    # 内参表:三列不确定度并排(CRLB 理论下界 / bootstrap 经验 / 重复张分组)
    crlb_model = ((crlb_doc.get("models") or {}).get(selected) or {}) if selected else {}
    crlb_rec = ((crlb_model.get("observation_sets") or {}).get("views640")
                or {}).get("detector") or {}
    boot_entry = ((eval_doc.get("bootstrap") or {}).get("models") or {}).get(selected) or {}
    slices_entry = ((eval_doc.get("slices") or {}).get("models") or {}).get(selected) or {}

    params: dict | None = None
    params_source = None
    if delivery_entry.get("final_params"):
        params = dict(delivery_entry["final_params"])
        params_source = (
            f"evaluate 轮 {rounds['evaluate']['name']} 全量重训交付"
            f" delivery.models.{selected}.final_params(evaluate.json)"
        )
    elif selected:
        # 交付缺席时回退标定轮同模型的全量精化解(来源照实披露,不冒充交付)
        cand = rounds["calibration"]["run"] / f"{selected}__scipy" / "result.json"
        if cand.is_file():
            params = (_load_json(cand).get("params") or None)
            if params:
                params_source = (
                    f"calibration 轮 {rounds['calibration']['name']}"
                    f" {selected}__scipy/result.json"
                    f"(交付轮缺席,回退标定轮全量精化解)"
                )
        if not params:
            _disclose("数据缺席", SECTION_TITLES[0],
                      f"当选模型 {selected} 的内参缺席:交付轮无 final_params,"
                      f"标定轮 {cand} 也不在,内参表无法给出。")
    else:
        _disclose("数据缺席", SECTION_TITLES[0],
                  "evaluate 轮没有选型结论(selection.selected 为空),"
                  "内参表与三判据判行缺席。")

    se_by_param: dict = {}
    if crlb_rec.get("status") == "ok" and crlb_rec.get("param_names"):
        se_by_param = dict(zip(crlb_rec["param_names"], crlb_rec["se"]))

    order = list(se_by_param or params or {})
    param_rows = []
    for name in order:
        param_rows.append({
            "param": name,
            "value": _fmt((params or {}).get(name)),
            "crlb_se": _fmt(se_by_param.get(name)),
            "bootstrap_std": _fmt((boot_entry.get("param_std") or {}).get(name)),
            "slice_std": _fmt((slices_entry.get("empirical_std") or {}).get(name)),
        })

    # 三判据判行(素材 = 判行表当选模型行;缺席就不判,不冒充)
    verdict_row = None
    for row in selection.get("verdicts") or []:
        if row.get("model_id") == selected:
            verdict_row = row
            break
    verdict_items: list[dict] = []
    overall_state = None
    if verdict_row is not None:
        for cn_name, key in (("泛化", "generalization"), ("稳定", "stability"),
                             ("逼近 CRLB", "crlb")):
            verdict_items.append({
                "name": cn_name,
                "state": _verdict_word((verdict_row.get(key) or {}).get("pass")),
                "note": (verdict_row.get(key) or {}).get("note") or "",
            })
        states = [item["state"] for item in verdict_items]
        overall_state = (verdict_row.get("overall")
                         or ("PASS" if all(s == "PASS" for s in states)
                             else "FAIL" if any(s == "FAIL" for s in states) else None))

    # 误差物理量级一句话(换算尺现算,零硬编码)
    test_rms = (selection.get("test_rms_px") or {}).get(selected) if selected else None
    physical_line = None
    if mm_per_px is not None and test_rms is not None:
        physical_line = (
            f"换算尺:照片里方格像素边长中位 {_fmt(sq_median, 1)}px 对应实物 "
            f"{board.square_size_mm}{board.unit},即 1px ≈ {_fmt(mm_per_px, 3)}mm;"
            f"当选模型 test 整体 RMS 折到实物约 {_fmt(test_rms * mm_per_px, 3)}mm"
            f"——头发丝直径的量级(一根头发丝约 0.05~0.1mm)。"
        )
    elif test_rms is not None:
        physical_line = (
            f"换算尺缺席(见披露),test 整体 RMS 保留像素口径 {_fmt(test_rms)}px。")

    s1 = {
        "title": SECTION_TITLES[0],
        "selected": selected,
        "verdict_line": (
            f"三判据判行(模型 {selected}):"
            + " / ".join(f"{i['name']} {i['state']}" for i in verdict_items)
            + f" → 综合 {overall_state}"
        ) if overall_state else None,
        "verdict_items": verdict_items,
        "overall_state": overall_state,
        "params_table": param_rows,
        "params_source": params_source,
        "crlb_caliber_note": (
            "「CRLB 理论下界」列取全帧版(views640)× 检测器级噪声口径;"
            + _obs_set_note()),
        "test_rms_line": (
            f"test 整体 RMS {_fmt(test_rms)}px"
            + (f"(独立考题 {manifest_doc.get('summary', {}).get('n_test')} 张"
               f"没参加训练的照片)" if manifest_doc.get("summary", {}).get("n_test") else "")
        ) if test_rms is not None else None,
        "physical_line": physical_line,
        "reason": selection.get("reason"),
        "walk_notes": [step.get("note", "") for step in selection.get("walk") or []],
        "agreement_note": (selection.get("agreement") or {}).get("note"),
        "absent_note": (
            f"evaluate 轮是失败轮,本节缺席:{failures['evaluate']}"
            if "evaluate" in failures else
            (None if selected else "evaluate 轮没有选型结论,本节的判行与内参表缺席。")),
    }

    # =====================================================================
    # 第②节 数据与体检(ingest 清单汇总)
    # =====================================================================
    m_sum = manifest_doc.get("summary") or {}
    sus = m_sum.get("suspect_counts") or {}
    rates = m_sum.get("detector_success_rates") or {}
    dist = m_sum.get("repeats_distribution") or {}
    dist_text = "、".join(f"连拍 {k} 张的姿态有 {v} 个" for k, v in dist.items()) or "—"
    s2 = {
        "title": SECTION_TITLES[1],
        "rows": [
            {"k": "训练照片", "v": f"{m_sum.get('n_train', '—')} 张"},
            {"k": "摆拍姿态", "v": f"{m_sum.get('n_poses', '—')} 个({dist_text})"},
            {"k": "测试照片", "v": f"{m_sum.get('n_test', '—')} 张(独立拍摄,每张一题)"},
            {"k": "统一分辨率",
             "v": "x".join(str(r) for r in m_sum.get("resolution") or []) or "—"},
            {"k": "棋盘规格",
             "v": (f"{board.cols}x{board.rows} 内角点,方格 {board.square_size_mm}"
                   f"{board.unit}") if board else "—(board.yaml 缺席,见披露)"},
            {"k": "体检存疑",
             "v": (f"{sus.get('suspect_total', 0)} 张"
                   f"(同姿态相对比较点名 {sus.get('pose_relative', 0)} 张、"
                   f"全局线兜底点名 {sus.get('global', 0)} 张;"
                   f"只标记不剔除,照常留在数据流中)")},
            {"k": "扫描警告", "v": f"{len(m_sum.get('warnings') or [])} 条(原文在 manifest)"},
        ],
        "visibility_rows": [
            {"split": split,
             "classic": _pct((rates.get(split) or {}).get("classic")),
             "sb": _pct((rates.get(split) or {}).get("sb"))}
            for split in ("train", "test") if split in rates
        ],
        "absent_note": (f"ingest 轮是失败轮,本节缺席:{failures['ingest']}"
                        if "ingest" in failures else None),
    }

    # =====================================================================
    # 第③节 角点(双检测器检出率 + 连拍断言 + 寻优定案)
    # =====================================================================
    detection = corners_doc.get("detection") or {}
    det_rows = []
    for split in ("train", "test"):
        slot = detection.get(split) or {}
        if not slot:
            continue
        cells = []
        for det_name, det_cn in (("classic", "经典 classic"), ("sb", "新款 sb")):
            stat = slot.get(det_name) or {}
            rate = stat.get("success_rate")
            cells.append(
                f"{det_cn} {stat.get('n_detected', '—')}/{stat.get('n_images', '—')}"
                f"({_pct(rate)})")
        det_rows.append({"split": split, "cells": cells})
    sb_train = (detection.get("train") or {}).get("sb") or {}
    sb_test = (detection.get("test") or {}).get("sb") or {}
    n_pure = sum(b.get("n_pure_sb") or 0 for b in (sb_train, sb_test))
    n_fb = sum(b.get("n_classic_fallback") or 0 for b in (sb_train, sb_test))

    burst = corners_doc.get("burst_check") or {}
    groups = burst.get("n_groups_checked") or {}
    burst_rows = [
        {"detector": f.get("detector"), "pose_id": f.get("pose_id"),
         "spread": (f"组内波动中位 {_fmt(f.get('spread_median_px'), 2)}px / "
                    f"最大 {_fmt(f.get('spread_max_px'), 2)}px"
                    f"(容差 {f.get('threshold_px') or '—'})")}
        for f in burst.get("failures") or []
    ]

    # 寻优定案(原料 = analyze 轮 corner_opt 段的中文判读,原文引用)
    corner_opt = (analyze_doc.get("corner_opt") or {}) if "analyze" not in failures else {}
    s3 = {
        "title": SECTION_TITLES[2],
        "det_rows": det_rows,
        "sb_production_line": (
            f"SB 生产口径:纯 SB 检出 {n_pure} 张 + 经典检测器兜底 {n_fb} 张"
            f"(逐帧来源记在角点文件的 source 数组,兜底不与 SB 本体混淆)"
            if (n_pure or n_fb) else None),
        "burst_line": (
            f"连拍一致性断言:每颗检测器检查 {groups.get('classic', '—')}"
            f"(classic)/ {groups.get('sb', '—')}(sb)组,失败 "
            f"{burst.get('n_failures', '—')} 组(点名存疑供人工审阅,不自动剔除)"
            if burst else None),
        "burst_rows": burst_rows,
        "corner_opt_note": corner_opt.get("verdict_note"),
        "corner_opt_absent_note": (
            f"analyze 轮是失败轮,角点寻优定案缺席:{failures['analyze']}"
            if "analyze" in failures else
            (None if corner_opt else
             "角点寻优段缺席(analyze 轮落账为 null),寻优定案暂缺,见披露清单。")),
        "absent_note": (f"corners 轮是失败轮,本节缺席:{failures['corners']}"
                        if "corners" in failures else None),
    }

    # =====================================================================
    # 第④节 噪声底(双口径 σ;检测器级口径是 CRLB 与预算的唯一乘数)
    # =====================================================================
    sigma_det = noise_doc.get("sigma_detector_corner")
    sigma_tmp = noise_doc.get("sigma_temporal_corner")
    per_source = noise_doc.get("per_source_counts") or {}
    outliers = noise_doc.get("outlier_poses") or []
    s4 = {
        "title": SECTION_TITLES[3],
        "rows": [
            {"k": "检测器口径(角点检测自身带来的误差)",
             "v": (f"{_fmt(sigma_det)}px"
                   + (f" ≈ {_fmt(sigma_det * mm_per_px, 4)}mm" if mm_per_px and sigma_det else ""))},
            {"k": "时域口径(同一位置连拍的角点波动,含拍摄瞬间的位姿抖动)",
             "v": (f"{_fmt(sigma_tmp)}px"
                   + (f" ≈ {_fmt(sigma_tmp * mm_per_px, 4)}mm" if mm_per_px and sigma_tmp else ""))},
            {"k": "两个口径的关系",
             "v": "时域比检测器口径大出的那一截,就是拍摄瞬间相机/棋盘没端稳的抖动;"
                  "后续理论下界(CRLB)与误差预算只认检测器口径,拿时域口径算会"
                  "把「天生最准能到多少」虚高数十倍(口径错配等于算错)。"},
            {"k": "来源计数",
             "v": (f"{per_source.get('temporal', 0)} 个姿态连拍实测(temporal)+ "
                   f"{per_source.get('immerkaer_transfer', 0)} 个姿态单帧折算"
                   f"(immerkaer_transfer)")},
            {"k": "波动最大的姿态(只点名不剔除)",
             "v": ("、".join(f"{o.get('pose_id')}({_fmt(o.get('sigma_pose'), 3)}px)"
                             for o in outliers[:3]) or "无")},
        ],
        "absent_note": (f"noise 轮是失败轮,本节缺席:{failures['noise']}"
                        if "noise" in failures else None),
    }

    # =====================================================================
    # 第⑤节 理论极限(CRLB 中位数 + 条件数分档 + 退化预警;全模型一表)
    # =====================================================================
    crlb_models = (crlb_doc.get("models") or {}) if "fisher" not in failures else {}
    theory_rows = []
    for model_id in sorted(crlb_models):
        rec = ((crlb_models[model_id].get("observation_sets") or {})
               .get("views640") or {}).get("detector") or {}
        if rec.get("status") == "ok":
            theory_rows.append({
                "model": model_id,
                "median_se": _median_text(rec.get("se")),
                "cond": _fmt(rec.get("condition_number"), 2),
                "flag": _flag_cn(rec.get("flag")),
            })
        else:
            theory_rows.append({
                "model": model_id, "median_se": "—", "cond": "—",
                "flag": rec.get("failure_note") or "记录失败",
            })
    sel_rec = {}
    if selected:
        sel_rec = ((crlb_models.get(selected) or {}).get("observation_sets")
                   or {}).get("views640") or {}
    sel_det = sel_rec.get("detector") or {}
    sel_tmp_rec = sel_rec.get("temporal") or {}
    lin_check = (crlb_models.get(selected) or {}).get("linearization_check") or {}
    crosscheck = (crlb_models.get(selected) or {}).get("opencv_crosscheck") or {}
    s5 = {
        "title": SECTION_TITLES[4],
        "intro": (
            "CRLB(理论精度下界)回答「这批数据天生最准能到多少」:只考虑角点"
            "检测噪声时,每个内参的标准差有一个物理上压不破的下界;实际标定"
            "结果越贴近下界,说明数据潜力被用得越足。条件数是「最软方向与最硬"
            "方向之比」:越大说明有些参数组合方向上精度越差,达到退化线意味着"
            "这批摆位定不住那些参数组合(要上报的发现)。"),
        "rows": theory_rows,
        "obs_note": _obs_set_note(),
        "selected_lines": ([
            (f"当选模型 {selected} 全帧版(views640):检测器口径 CRLB 中位数 "
             f"{_median_text(sel_det.get('se'))}px,条件数 {_fmt(sel_det.get('condition_number'), 2)}"
             f"(档位:{_flag_cn(sel_det.get('flag'))})"),
            (f"当选模型全帧版时域口径对照:CRLB 中位数 {_median_text(sel_tmp_rec.get('se'))}px"
             f"(σ 口径更松,界自然更松——双口径并排防口径混用)"),
            (f"线性化检查:{lin_check.get('note') or '—'}(实测最大相对差 "
             f"{_fmt(lin_check.get('rel_diff'))})"),
            (f"OpenCV 标准差交叉验证:{crosscheck.get('status') or '—'}"
             f"({crosscheck.get('verdict_note') or crosscheck.get('failure_note') or '—'})"),
        ] if selected else []),
        "absent_note": (f"fisher 轮是失败轮,本节缺席:{failures['fisher']}"
                        if "fisher" in failures else
                        (None if crlb_models else "crlb.json 里没有任何模型记录,本节缺席。")),
    }

    # =====================================================================
    # 第⑥节 防作弊考试(test 走查排名 + 判行表 + 四类实验规模)
    # =====================================================================
    verdicts = selection.get("verdicts") or []
    test_rms_px = selection.get("test_rms_px") or {}
    ranking = sorted(
        ((m, v) for m, v in test_rms_px.items() if v is not None),
        key=lambda kv: kv[1])
    rank_rows = []
    for m, v in ranking:
        rank_rows.append({
            "model": m, "test_rms": _fmt(v),
            "mark": "  ← 当选" if m == selected else "",
        })
    thresholds = selection.get("verdict_thresholds") or {}
    verdict_rows = []
    for row in verdicts:
        gen = row.get("generalization") or {}
        stab = row.get("stability") or {}
        crlb_v = row.get("crlb") or {}
        failed_names = [cn for cn, blk in (
            ("泛化", gen), ("稳定", stab), ("逼近 CRLB", crlb_v))
            if blk and blk.get("pass") is False]
        verdict_rows.append({
            "model": row.get("model_id"),
            "heldout": _fmt(gen.get("mean_heldout_px")),
            "train": _fmt(gen.get("mean_train_px")),
            "overfit": _fmt(gen.get("overfit_ratio"), 3),
            "disp": _fmt(stab.get("max_disp_px_overall")),
            "bic": _fmt(row.get("bic"), 1),
            "overall": row.get("overall") or "—",
            "failed": ("FAIL 于:" + "、".join(failed_names)) if failed_names else "",
        })
    cv_blk = (eval_doc.get("cv") or {}) if "evaluate" not in failures else {}
    boot_blk = (eval_doc.get("bootstrap") or {}) if "evaluate" not in failures else {}
    slices_blk = (eval_doc.get("slices") or {}) if "evaluate" not in failures else {}
    lopo_blk = (eval_doc.get("lopo") or {}) if "evaluate" not in failures else {}
    optimizer = (eval_doc.get("optimizer") or {}) if "evaluate" not in failures else {}
    cv_meta = cv_blk.get("meta") or {}
    boot_meta = boot_blk.get("meta") or {}
    slices_meta = slices_blk.get("meta") or {}
    exp_lines = []
    if cv_meta:
        exp_lines.append(
            f"按姿态分组交叉验证:{cv_meta.get('n_folds', '—')} 折 x "
            f"{cv_meta.get('n_rounds', '—')} 轮,求解 {cv_meta.get('n_solves', '—')} 次"
            f"(留出一部分姿态不训练、只考试,防死记硬背)")
    if boot_meta:
        exp_lines.append(
            f"bootstrap 重抽样:{boot_meta.get('n_resamples', '—')} 次"
            f"(反复换姿势抽样重新标定,看参数挪动多大)")
    if slices_meta:
        exp_lines.append(
            f"重复张分组:{slices_meta.get('n_groups', '—')} 组"
            f"(同一姿态的连拍轮换分组,量纯噪声引起的参数晃动)")
    if lopo_blk:
        exp_lines.append(
            f"留一姿态:逐个剔除姿态重标,参试模型 "
            f"{'、'.join(sorted(lopo_blk.get('models') or {})) or '—'}"
            f"(诊断实验,不进判行表)")
    rec_solver = optimizer.get("recommended_solver") or {}
    if rec_solver:
        exp_lines.append(
            "优化器对比推荐配置:"
            + ";".join(f"{m} 用 {rec.get('loss') or '无收敛组合'} 核"
                       for m, rec in sorted(rec_solver.items())))
    multistart = (optimizer.get("multistart") or {}).get("models") or {}
    for m, blk in sorted(multistart.items()):
        exp_lines.append(
            f"{m} 多起点一致性:{blk.get('n_converged', '—')}/"
            f"{blk.get('n_starts', '—')} 收敛,判定 {blk.get('verdict', '—')}")
    delivery_lines = []
    for m in sorted(delivery_models):
        entry = delivery_models.get(m) or {}
        if entry.get("rms_px") is not None:
            delivery_lines.append(
                f"{m}:全量重训 train RMS {_fmt(entry.get('rms_px'))}px,"
                f"BIC {_fmt(entry.get('bic'), 1)}(BIC = 给模型复杂度记罚分的选型尺,"
                f"越小越好;交付 = 全体训练照片上重训的最终内参)")
        else:
            delivery_lines.append(
                f"{m}:重训未跑成({entry.get('failure_note') or '原因未记录'})")
    s6 = {
        "title": SECTION_TITLES[5],
        "verdict_line": (
            "判行表判定:" + ";".join(
                f"{r['model']} {r['overall']}" for r in verdict_rows)
            if verdict_rows else None),
        "intro": (
            "「防作弊考试」= 不让模型只看训练题自夸:held-out(留出考试)成绩"
            "用没参加训练的数据打分,过拟合比 = 考试成绩 ÷ 训练成绩(越接近 1 "
            "越没死记硬背),位移当量 = 换一组数据重标后参数在画面上挪动多少像素。"),
        "thresholds_line": (
            f"判行门槛:过拟合比 < {_fmt(thresholds.get('gen_overfit_ratio_max'), 2)},"
            f"位移当量 < {_fmt(thresholds.get('stability_disp_px_max'), 2)}px"
            if thresholds else None),
        "rank_rows": rank_rows,
        "rank_note": (
            "排名按 test 整体 RMS(独立考题成绩)升序;选型规则 = 复杂度逐档走查,"
            "更复杂一档的提升达到门槛才升级,否则取更简者。"),
        "verdict_rows": verdict_rows,
        "exp_lines": exp_lines,
        "delivery_lines": delivery_lines,
        "agreement_note": (selection.get("agreement") or {}).get("note"),
        "three_criteria_note": (
            ((selection.get("three_criteria") or {}).get("reason") or None)
            if selection.get("three_criteria") else None),
        "absent_note": (
            f"evaluate 轮是失败轮,本节缺席:{failures['evaluate']}"
            if "evaluate" in failures else
            (None if (verdicts or test_rms_px) else
             "evaluate 轮里判行表与 test 成绩都缺席(轮内落账为 null),本节大部缺席。")),
    }

    # =====================================================================
    # 第⑦节 误差归因(预算判行 + 覆盖足迹 + 三分量 + 敏感度 + 各向异性)
    # =====================================================================
    budget = (analyze_doc.get("budget") or {}) if "analyze" not in failures else {}
    residual = (analyze_doc.get("residual_structure") or {}) \
        if "analyze" not in failures else {}
    test_xyz = (analyze_doc.get("test_xyz") or {}) if "analyze" not in failures else {}
    sensitivity = (analyze_doc.get("sensitivity") or {}) \
        if "analyze" not in failures else {}
    anisotropy = (analyze_doc.get("anisotropy") or {}) \
        if "analyze" not in failures else {}
    missing_sections = ((analyze_doc.get("meta") or {}).get("missing_sections") or [])
    sec_cn = {"budget": "误差预算", "residual_structure": "残差方向分解",
              "test_xyz": "test 三分量", "sensitivity": "参数敏感度",
              "corner_opt": "角点寻优", "anisotropy": "连拍各向异性"}
    for sec_key in missing_sections:
        _disclose("分析段缺席", SECTION_TITLES[6],
                  f"分析段「{sec_cn.get(sec_key, sec_key)}」在本轮落账为 null"
                  f"(单块失败不中断其余块的兜底设计,原因见当轮命令播报)。")

    budget_verdict = None
    if budget:
        raw = budget.get("verdict")
        budget_verdict = {"pass": "PASS", "fail": "FAIL"}.get(raw, raw)
    ratios = ((budget.get("recheck") or {}).get("ratios") or {})

    def _ratio(key: str) -> str:
        v = ratios.get(key)
        return f"{float(v):.0f}" if v is not None else "—"
    budget_rows = [
        {"param": r.get("param"),
         "empirical_px": _fmt(r.get("empirical_std_px")),
         "bootstrap": _fmt(r.get("bootstrap_std")),
         "pose_sampling": _fmt(r.get("pose_sampling_std")),
         "multistart": _fmt(r.get("multistart_disp_px")),
         "crlb_se": _fmt(r.get("crlb_se")),
         "utilization": (_fmt(r.get("utilization_ratio"), 1) + " 倍"
                         if r.get("utilization_ratio") is not None else "—")}
        for r in budget.get("rows") or []
    ]
    coverage = (residual.get("train") or {}).get("coverage") or {}
    coverage_line = None
    if coverage:
        coverage_line = (
            f"最远观测 r = {_fmt(coverage.get('r_max_obs'), 3)},画面四角 r = "
            f"{_fmt(coverage.get('r_corner_frame'), 3)}(r = 离画面中心的距离占"
            f"对角线一半的比例);r > 0.5 的观测占 "
            f"{_pct(coverage.get('frac_beyond_05'))},却承载 k2(二阶径向畸变)"
            f"信息的 {_pct(coverage.get('info_share_beyond_05'))}——"
            f"边角观测稀少但畸变信息恰恰藏在边角。")
    bands = (residual.get("train") or {}).get("bands") or []
    band_rows = [
        {"band": f"[{_fmt(b.get('band_lo'), 1)}, {_fmt(b.get('band_hi'), 1)})",
         "n": str(b.get("n", "—")),
         "rms": _fmt(b.get("rms_px")),
         "max": _fmt(b.get("max_px"))}
        for b in bands
    ]
    obs_blk = test_xyz.get("observability") or {}
    dir_blk = (test_xyz.get("per_direction") or {}).get("dX_abs_mm") or {}
    z_blk = (test_xyz.get("per_direction") or {}).get("abs_zscore") or {}
    advice_items = [
        {"direction": a.get("direction"), "evidence": a.get("evidence"),
         "advice": a.get("advice")}
        for a in sensitivity.get("advice") or []
    ]
    uniq = sensitivity.get("uniqueness") or {}
    aniso_sum = anisotropy.get("summary") or {}
    s7 = {
        "title": SECTION_TITLES[6],
        "verdict_line": (
            f"误差预算判行:{budget_verdict}"
            + (f"(逐行核查违例 {budget.get('n_pointwise_violations')} 个,"
               f"逐行最小实测/下界比 {_fmt(budget.get('min_ratio_pointwise'), 1)})"
               if budget else ""))
            if budget_verdict else None,
        "intro": (
            "误差预算回答「实际晃动离理论极限还有多远」:逐参数把 bootstrap、"
            "换姿态、多起点三路晃动与同口径 CRLB 下界并排;实测低于理论下界"
            "物理上不可能,出现即是口径错配的信号(命令会当场报错)。"
            f"预算判定线(硬告警线):{_fmt((budget.get('lines') or {}).get('hard'))}。"
            if budget else None),
        "budget_rows": budget_rows,
        "ratios_line": (
            "三统计量实测是同口径下界的 "
            f"{_ratio('重投影位移中位')} / {_ratio('重投影位移95分位')} / "
            f"{_ratio('重投影位移最大')} 倍"
            if ratios else None) if budget else None,
        "recheck_line": (
            f"独立重算对账:{'一致' if (budget.get('recheck') or {}).get('consistent') else '不一致(口径漂移,要排查)'}"
            if budget.get("recheck") else None) if budget else None,
        "coverage_line": coverage_line,
        "extrapolation_note": residual.get("extrapolation_note"),
        "band_rows": band_rows,
        "test_xyz_lines": (
            [(f"test 三分量分解:{test_xyz.get('n_valid', '—')}/"
              f"{test_xyz.get('n_frames', '—')} 帧有效(病态剔除 "
              f"{test_xyz.get('n_cond_excluded', '—')}、解不出 "
              f"{test_xyz.get('n_solve_failed', '—')})"),
             (f"逐方向位移绝对值中位:x {_fmt((dir_blk.get('x') or {}).get('median'))}mm,"
              f"y {_fmt((dir_blk.get('y') or {}).get('median'))}mm,"
              f"z {_fmt((dir_blk.get('z') or {}).get('median'))}mm;"
              f"z 分数中位 x {_fmt((z_blk.get('x') or {}).get('median'), 2)},"
              f"y {_fmt((z_blk.get('y') or {}).get('median'), 2)},"
              f"z {_fmt((z_blk.get('z') or {}).get('median'), 2)}"
              "(Z 分量大不等于误差主要在深度:单视角深度可观测性差,判读看 z 分数列)"),
             (f"深度放大倍数中位 {_fmt(obs_blk.get('depth_amplification_median'), 2)} 倍"
              if obs_blk.get("depth_amplification_median") is not None else
              "深度放大倍数缺席")]
            if test_xyz else []),
        "test_xyz_notes": list(test_xyz.get("notes") or []),
        "advice_items": advice_items,
        "uniqueness_line": (
            f"解不唯一证据:bootstrap 最大位移当量 {_fmt(uniq.get('bootstrap_max_disp_px'), 2)}px,"
            f"多起点判定 {uniq.get('multistart_verdict', '—')}"
            if uniq else None),
        "uniqueness_note": uniq.get("note"),
        "anisotropy_line": (
            f"连拍各向异性:横纵定位散布之比中位 {_fmt(aniso_sum.get('median'), 2)}"
            f"(接近 1 = 横竖对称),显著各向异性(>{_fmt(aniso_sum.get('significant_line'), 1)})"
            f"{aniso_sum.get('n_aniso_gt_1p5', '—')} 处"
            if aniso_sum else None),
        "anisotropy_plain": anisotropy.get("plain_language"),
        "absent_note": (
            f"analyze 轮是失败轮,本节缺席:{failures['analyze']}"
            if "analyze" in failures else
            (None if (budget or residual or test_xyz or sensitivity or anisotropy) else
             "analyze 轮六段全部落账为 null,本节缺席。")),
    }

    # =====================================================================
    # 第⑧节 回环建议(占位:由后续版本接入规则引擎)
    # =====================================================================
    s8 = {
        "title": SECTION_TITLES[7],
        "placeholder": (
            "本节是「回哪一步、做什么」的自动建议表,由后续版本接入"
            "(输入就是本报告第⑥⑦节的判定段)。当前请先看第⑦节各分析段的"
            "建议原文(advice):它们只提示方向,不自动执行任何补拍或重标定。"),
    }

    # =====================================================================
    # 第⑨节 产物索引与血缘链 + 附录图表(编号引用各上游轮既有 PNG)
    # =====================================================================
    chain = []
    for stage, fname, cn in _UPSTREAM:
        entry = rounds[stage]
        lineage = _as_dict(entry["doc"]).get("lineage") or {}
        upstream = lineage.get("upstream") or {}
        missing = [name for name, up in upstream.items()
                   if not Path((up or {}).get("path", "")).is_file()]
        chain.append({
            "stage": cn,
            "run": entry["name"],
            "main_file": str(entry["main"]),
            "created": lineage.get("created") or "—",
            "upstream_status": ("完整(登记的上游文件都在盘)" if not missing
                                else f"该轮上游已清理:{'、'.join(missing)}"),
        })
        if missing:
            _disclose("上游已清理", SECTION_TITLES[8],
                      f"{cn}轮({entry['name']})的血缘登记指向已不在盘的文件:"
                      f"{'、'.join(missing)}。及时清理是常态,照实披露不影响本轮结论。")

    figures = _plan_figures(rounds, selected)
    s9 = {
        "title": SECTION_TITLES[8],
        "chain": chain,
        "figures": figures,
        "reuse_line": (
            "换一批数据时:把旧 output/ 移走,新照片按目录契约放进 input_data/,"
            "改好 config/ 两个配置文件后按流水线顺序重跑即可——本报告的全部"
            "数字与换算尺都按新数据现算,不需要改任何代码或模板。"),
    }

    # ---- 机器可读关键指标(跨运行对比专用;模板不渲染本段)----
    cv_m2 = ((cv_blk.get("models") or {}).get(selected) or {}) if selected else {}
    metrics_raw = {
        "selected": selected,
        "test_rms_px": float(test_rms) if test_rms is not None else None,
        "sigma_detector_corner_px": float(sigma_det) if sigma_det is not None else None,
        "mm_per_px": float(mm_per_px) if mm_per_px is not None else None,
        "budget_ratio_median": (float(ratios.get("重投影位移中位"))
                                if ratios.get("重投影位移中位") is not None else None),
        "crlb_median_se_views640_detector_px": (
            float(statistics.median(sel_det["se"]))
            if sel_det.get("status") == "ok" and sel_det.get("se") else None),
        "cv_mean_heldout_px": (
            float(cv_m2["mean_heldout_px"])
            if cv_m2.get("mean_heldout_px") is not None else None),
        "rounds": {stage: rounds[stage]["name"] for stage, _f, _c in _UPSTREAM},
    }

    return {
        "schema": "report-context-v1",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "board": ({
            "cols": board.cols, "rows": board.rows,
            "square_size_mm": board.square_size_mm, "unit": board.unit,
        } if board else None),
        "scale": {
            "sq_px_median": _fmt(sq_median, 1),
            "mm_per_px": _fmt(mm_per_px, 3),
            "note": ("物理换算尺 = 方格实物边长 ÷ 照片里方格像素边长中位;"
                     "报告里的毫米数全部由它折算,换数据自动跟随。"),
        },
        "sections": [s["title"] for s in (s1, s2, s3, s4, s5, s6, s7, s8, s9)],
        "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5,
        "s6": s6, "s7": s7, "s8": s8, "s9": s9,
        "disclosures": disclosures,
        "metrics_raw": metrics_raw,
    }


def run_report(out_root: Path) -> dict:
    """顶层一步到位:组装 context → 渲染双模板 → 落盘本轮全部产物。

    命令层(calib report)只调这一个函数——06-02 往 context 加段时,
    命令体与模板骨架都不用改。落盘四件:
    report.html(排版证据链报告)/ report.md(同内容纯文本版)/
    report_data.json(context 原样快照,机器可读底账)/ figures/(从各上游
    轮原样复制来的既有 PNG,报告只引用不重画)。

    返回:{"run_dir", "round_id", "n_figures", "selected", "n_disclosures"},
    供命令层播报与清理旧轮。
    """
    out_root = Path(out_root)
    context = build_report_context(out_root)

    round_id = "run_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / "report" / round_id
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 图表:按清单从各上游轮原样复制(文件名各自不同,不会互相覆盖)
    n_figures = 0
    for fig in context["s9"]["figures"]:
        src = fig.get("_src")
        if src and Path(src).is_file():
            shutil.copy2(src, fig_dir / fig["file"])
            n_figures += 1

    # HTML 必须开自动转义(autoescape):上游产物里若混入 < 和 &,
    # 不能让它们破坏页面结构(威胁 T-06-01);StrictUndefined 让模板
    # 引用不存在的键时当场报错,而不是悄悄渲染成空。
    env_html = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
        undefined=StrictUndefined,
    )
    # Markdown 是纯文本格式,开转义反而会把正文里的 & 改写成 &amp;——
    # 不开;防篡改同样成立:模板零本数据集数字,数字全部来自 context。
    env_md = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=False,
        undefined=StrictUndefined,
    )
    (run_dir / "report.html").write_text(
        env_html.get_template("report.html.j2").render(**context),
        encoding="utf-8")
    (run_dir / "report.md").write_text(
        env_md.get_template("report.md.j2").render(**context),
        encoding="utf-8")

    # 底账快照:context 原样落盘——报告里的每个数字都能在这里逐位找到
    # (这也是 06-02 跨运行对比的取数基础)。
    (run_dir / "report_data.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8")

    return {
        "run_dir": run_dir,
        "round_id": round_id,
        "n_figures": n_figures,
        "selected": context["metrics_raw"]["selected"],
        "n_disclosures": len(context["disclosures"]),
    }
