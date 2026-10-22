"""报告层跨轮纯函数:剔除候选清单收集(ITER-02)+ 跨运行对比(ITER-03)。

两个函数都零 IO——只读传入的字典,不碰文件;落盘与读盘全部在 build 层
组装与命令层完成。这样合成输入就能单测每一路,不必搭真实轮。

剔除候选(ITER-02,裁定①)的机制边界:本模块只「收集 + 说理由」,
绝不自动改配置。人工确认后把 image_id 填进 config/calibration.yaml 的
exclude_images(唯一剔除名单),再跑 calib run-all --from calibrate。
"""

from __future__ import annotations


def _fmt_px(value) -> str:
    """像素数 → 白话字符串(None/非数给「—」)。"""
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}px"
    except (TypeError, ValueError):
        return "—"


def _candidate(image_id: str | None, pose_id: str | None,
               source: str, reason: str) -> dict:
    """一条剔除候选:照片名 + 摆拍位置 + 来源 + 中文理由(给人工确认用)。"""
    return {"image_id": image_id, "pose_id": pose_id,
            "source": source, "reason": reason}


def _images_of_pose(manifest: dict | None, pose_id: str) -> list[dict]:
    """manifest 里某摆拍位置的全部照片(按连拍序排,训练集才有连拍)。"""
    imgs = [im for im in (manifest or {}).get("images") or []
            if im.get("pose_id") == pose_id]
    return sorted(imgs, key=lambda im: im.get("repeat_idx") or 0)


def _lopo_candidates(evaluate_report: dict | None,
                     manifest: dict | None) -> list[dict]:
    """第一路:留一姿态点名(逐个撤掉某摆拍位置重算、看结果晃不晃)。

    点名清单直接用 evaluate 轮算好的 outlier_poses(稳健 z 分数超过
    门槛的摆拍位置);候选落到该姿态的每一张连拍上——剔除名单是按张
    记的,想去掉一个姿态就得整组填。理由给白话:晃了多少、门槛多少。
    """
    lopo = (evaluate_report or {}).get("lopo") or {}
    models = lopo.get("models") or {}
    selected = (((evaluate_report or {}).get("selection") or {})
                .get("selected"))
    blk = models.get(selected) if selected else None
    if blk is None and len(models) == 1:
        blk = next(iter(models.values()))
    if not blk:
        return []
    threshold = (lopo.get("meta") or {}).get("mad_z_threshold")
    mad_z = blk.get("mad_z") or {}
    z_txt = lambda z: f"{float(z):.1f}" if z is not None else "—"
    thr_txt = (f"{float(threshold):.1f}" if threshold is not None else "3.5")
    out: list[dict] = []
    for item in blk.get("outlier_poses") or []:
        # 生产端(evaluation/experiments.py)落的是字典 {"pose_id", "mad_z"};
        # 兼容裸姿态名字符串,防旧轮产物或手改产物串格式
        if isinstance(item, dict):
            pose = item.get("pose_id")
            z_value = item.get("mad_z")
        else:
            pose = item
            z_value = None
        if not pose:
            continue
        if z_value is None:
            z_value = mad_z.get(pose)
        reason = (
            f"留一姿态实验点名:撤掉这个摆拍位置重新标定时误差明显变大"
            f"(稳健 z 分数 {z_txt(z_value)},门槛 {thr_txt}),"
            f"这个姿态的连拍整组都值得人工复核")
        imgs = _images_of_pose(manifest, pose)
        if imgs:
            out.extend(_candidate(im.get("image_id"), pose, "留一姿态点名",
                                  reason) for im in imgs)
        else:
            out.append(_candidate(None, pose, "留一姿态点名",
                                  reason + "(manifest 里找不到该姿态的照片,"
                                  "请对照照片目录人工补全名单)"))
    return out


def _burst_candidates(corners_report: dict | None,
                      manifest: dict | None) -> list[dict]:
    """第二路:连拍一致性断言失败张(同姿态连拍里同名角点位置波动超容差)。

    断言本身在 corners 轮 detection_summary.json 的 burst_check 段;同一
    姿态可能被两颗检测器各点名一次,按姿态去重只收一组。候选同样落
    到该姿态的每一张连拍上。
    """
    burst = ((corners_report or {}).get("burst_check") or {})
    out: list[dict] = []
    seen_poses: set[str] = set()
    for item in burst.get("failures") or []:
        pose = item.get("pose_id")
        if not pose or pose in seen_poses:
            continue
        seen_poses.add(pose)
        reason = (
            f"连拍一致性断言点名:同一姿态的连拍里,同名角点的位置波动超出"
            f"容差(组内波动中位 {_fmt_px(item.get('spread_median_px'))} / "
            f"最大 {_fmt_px(item.get('spread_max_px'))},容差 "
            f"{_fmt_px(item.get('threshold_px'))}),常见原因是角点排序被"
            f"镜像/翻转污染或拍摄瞬间晃动,请对照人工复核包确认")
        imgs = _images_of_pose(manifest, pose)
        if imgs:
            out.extend(_candidate(im.get("image_id"), pose, "连拍断言点名",
                                  reason) for im in imgs)
        else:
            out.append(_candidate(None, pose, "连拍断言点名",
                                  reason + "(manifest 里找不到该姿态的照片,"
                                  "请对照照片目录人工补全名单)"))
    return out


def _ingest_candidates(manifest: dict | None) -> list[dict]:
    """第三路:入库体检存疑里「复核计数高」的照片。

    「计数高」= 被至少两类判定来源同时点名(比如模糊度和曝光同时越线);
    只被一类来源点名的照片属于例行标记(只标记不剔除的既定纪律),
    不进候选表,免得名单被 89 张例行存疑淹没。候选按来源数降序、
    同数按照片名排,最可疑的排前面。
    """
    scored: list[tuple[int, str, dict]] = []
    for im in (manifest or {}).get("images") or []:
        precheck = im.get("precheck") or {}
        if precheck.get("verdict") != "suspect":
            continue
        sources = list(precheck.get("suspect_sources") or [])
        if len(sources) < 2:
            continue
        reason = (
            f"入库体检被 {len(sources)} 类判定来源同时点名"
            f"({'、'.join(sources)}),多个指标同时越线,值得优先人工复核"
            f"(存疑只标记不剔除;对照存疑原图副本 suspect_originals/ 确认,"
            f"确属拍坏再进剔除名单)")
        image_id = im.get("image_id") or ""
        scored.append((len(sources), image_id,
                       _candidate(image_id, im.get("pose_id"),
                                  "入库体检存疑", reason)))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [entry for _n, _i, entry in scored]


def collect_exclude_candidates(evaluate_report: dict | None = None,
                               analyze_report: dict | None = None,
                               manifest: dict | None = None,
                               corners_report: dict | None = None) -> dict:
    """收集剔除候选清单(ITER-02):三路候选取并集,零候选落空表。

    参数(全部是各轮 JSON 读回的字典,缺席给 None):
        evaluate_report: evaluate.json(留一姿态点名来源);
        analyze_report:  analyze.json(本版三路候选不读它,参数保留与
                         计划签名一致;连拍断言实际在 corners 轮);
        manifest:        manifest.json(照片清单 + 体检判定,候选落张);
        corners_report:  detection_summary.json(连拍断言点名来源)。

    返回:{"candidates": [{image_id, pose_id, source, reason}, ...]},
    顺序 = 留一点名 → 连拍断言 → 入库体检(多来源按计数降序)。
    """
    candidates: list[dict] = []
    candidates += _lopo_candidates(evaluate_report, manifest)
    candidates += _burst_candidates(corners_report, manifest)
    candidates += _ingest_candidates(manifest)
    return {"candidates": candidates}


# ---------------------------------------------------------------- 跨运行对比

# 对比键清单(锚点 = 上一份 report 轮的 report_data.json,裁定②):
# (报告里显示的指标名, metrics_raw 键, 变好方向)
# 方向语义按指标含义写死:test RMS、预算比值、噪声底都是越低越好;
# 当选模型是名字,没有好坏,只报「持平 / 换成谁」。
_DIFF_KEYS = (
    ("当选模型", "selected", "name"),
    ("test 整体 RMS(px)", "test_rms_px", "lower"),
    ("预算比值·重投影位移中位(实测 ÷ 理论下界)", "budget_ratio_median", "lower"),
    ("预算比值·重投影位移95分位(实测 ÷ 理论下界)", "budget_ratio_p95", "lower"),
    ("预算比值·重投影位移最大(实测 ÷ 理论下界)", "budget_ratio_max", "lower"),
    ("噪声底·检测器口径 σ(px)", "sigma_detector_corner_px", "lower"),
)

# 数值「持平」的相对容差:变化小于千分之一视为持平(四位小数显示下
# 本来就看不出差别,报「变好/变差」反而误导)
_TIE_REL = 1e-3


def _fmt_value(key: str, value) -> str:
    """对比值 → 定稿字符串(组装层定稿纪律:模板只排版)。"""
    if value is None:
        return "—"
    if key == "selected":
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _change_word(kind: str, current, previous) -> str:
    """变化一栏的方向词(语义按指标含义写死,见 _DIFF_KEYS 第三列)。"""
    if previous is None:
        return "上次无此数据"
    if current is None:
        return "本次无此数据"
    if kind == "name":
        return "持平" if current == previous else f"换成 {current}"
    try:
        cur_f, prev_f = float(current), float(previous)
    except (TypeError, ValueError):
        return "持平" if current == previous else "有变化"
    if abs(cur_f - prev_f) <= _TIE_REL * max(abs(prev_f), 1e-12):
        return "持平"
    went_down = cur_f < prev_f
    if kind == "lower":
        return "变好(降低)" if went_down else "变差(升高)"
    return "变好(升高)" if not went_down else "变差(降低)"


def compare_with_previous(current: dict, previous: dict | None) -> dict:
    """本次 vs 上次的关键指标对比(ITER-03),纯函数零 IO。

    参数是两份 report_data.json 底账快照(context 原样落盘的那份);
    previous 给 None 表示没有上一份报告轮(首次运行)。上一份是旧版
    底账、缺某个键时,那一行照出、变化栏记「上次无此数据」,不崩也不
    冒充持平。

    双方实验规模不同(如 bootstrap 重抽样次数)时给规模备注,提醒
    「规模不同,数值差异不能直接当回归读」——不冒充回归,也不冒充持平。

    返回:{"status": "ok" | "首次运行", "note", "rows", "scale_note"},
    rows 每行 {label, current, previous, change},全部是定稿字符串。
    """
    if not previous:
        return {
            "status": "首次运行",
            "note": ("这是本项目第一份报告(没有上一份报告轮可比);"
                     "从下一份报告开始,这里会出现本次与上次的关键指标对比,"
                     "先看本次防作弊考试的成绩,再对照看比上次好没好。"),
            "rows": [],
            "scale_note": None,
        }

    cur_raw = current.get("metrics_raw") or {}
    prev_raw = previous.get("metrics_raw") or {}
    rows = []
    for label, key, kind in _DIFF_KEYS:
        cur_v, prev_v = cur_raw.get(key), prev_raw.get(key)
        if cur_v is None and prev_v is None:
            continue  # 两边都没有,没有可比的内容,不排空行
        rows.append({
            "label": label,
            "current": _fmt_value(key, cur_v),
            "previous": _fmt_value(key, prev_v),
            "change": _change_word(kind, cur_v, prev_v),
        })

    # 实验规模差异备注(如实解释,不冒充回归)
    parts = []
    cur_boot = cur_raw.get("n_bootstrap_resamples")
    prev_boot = prev_raw.get("n_bootstrap_resamples")
    if cur_boot is not None and prev_boot is not None and cur_boot != prev_boot:
        parts.append(f"bootstrap 重抽样次数不同(上次 {prev_boot} 次 / "
                     f"本次 {cur_boot} 次)")
    cur_profile = cur_raw.get("profile")
    prev_profile = prev_raw.get("profile")
    if cur_profile and prev_profile and cur_profile != prev_profile:
        parts.append(f"实验档位不同(上次 {prev_profile} / 本次 {cur_profile})")
    scale_note = (f"实验规模差异备注:{';'.join(parts)}——规模不同,两组数字"
                  f"不能直接当回归读,先对齐规模再比(规模原文见各自底账的 "
                  f"metrics_raw)。") if parts else None

    return {"status": "ok", "note": None,
            "rows": rows, "scale_note": scale_note}
