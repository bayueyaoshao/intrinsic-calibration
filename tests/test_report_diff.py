"""报告层候选清单与跨运行对比测试(合成最小判定段,白天快速档,秒级)。

测的是 06-02 落地的 diff 模块两件纯函数,以及它们接进报告层的整链:
1. collect_exclude_candidates:三路剔除候选(留一姿态点名 / 连拍断言
   失败 / 入库体检多来源存疑)逐路单收、零候选落空表、来源与理由齐;
2. build 层 ⑧ 节组装与 suggest_exclude.json 落盘(复用 test_report_build
   的合成七路树):回环建议表、候选清单、固定操作指引三块都进 context
   与双模板渲染产物,空名单也照常落盘。

全部合成输入,不碰真实 output/。
"""

from __future__ import annotations

import json

import test_report_build

from calib.report.build import build_report_context
from calib.report.diff import collect_exclude_candidates, compare_with_previous
from calib.report.loopback import PIPELINE_STEPS

# ---------------------------------------------------------------- 合成判定段


def _eval_doc(outlier_poses: list[str] | None = None,
              mad_z: dict | None = None) -> dict:
    """最小 evaluate 判定段:选型 + 留一姿态点名(mad_z 门槛 3.5)。"""
    return {
        "selection": {"selected": "m2", "verdicts": [], "test_rms_px": {}},
        "lopo": {
            "models": {"m2": {
                "outlier_poses": outlier_poses or [],
                "mad_z": mad_z or {},
            }},
            "meta": {"mad_z_threshold": 3.5},
        },
    }


def _corners_doc(failures: list[dict] | None = None) -> dict:
    """最小 corners 判定段:连拍一致性断言点名清单。"""
    return {"burst_check": {"n_failures": len(failures or []),
                            "failures": failures or []}}


def _manifest(images: list[dict] | None = None) -> dict:
    """最小 manifest:照片清单(image_id / pose_id / split / precheck)。"""
    return {"images": images or []}


def _train(image_id: str, pose_id: str, suspect: bool = False,
           sources: list[str] | None = None) -> dict:
    precheck = ({"verdict": "suspect", "suspect_sources": sources or ["global:blur"]}
                if suspect else {"verdict": "ok", "suspect_sources": []})
    return {"image_id": image_id, "pose_id": pose_id, "split": "train",
            "repeat_idx": 0, "precheck": precheck}


# =====================================================================
# 1. collect_exclude_candidates:三路候选逐路单收
# =====================================================================
def test_lopo_outlier_pose_expands_to_its_images():
    """留一点名展开成该姿态全部连拍;喂生产端真实格式(字典带 pose_id/mad_z)。"""
    manifest = _manifest([
        _train("img_a0", "poseX"), _train("img_a1", "poseX"),
        _train("img_b0", "poseY"),
    ])
    ev = _eval_doc(outlier_poses=[{"pose_id": "poseX", "mad_z": 4.2}])
    out = collect_exclude_candidates(ev, None, manifest, None)
    cands = out["candidates"]
    assert len(cands) == 2, f"姿态 poseX 有 2 张连拍,应各出一条候选:{cands}"
    assert {c["image_id"] for c in cands} == {"img_a0", "img_a1"}
    for c in cands:
        assert c["pose_id"] == "poseX"
        assert "留一姿态" in c["source"]
        assert "4.2" in c["reason"], "理由应带稳健 z 分数原文数字"
        assert "3.5" in c["reason"], "理由应带门槛数字(白话对照)"


def test_lopo_outlier_pose_tolerates_bare_string_entries():
    """兼容裸姿态名字符串条目:z 分数退回 blk 级 mad_z 对照表取数。"""
    manifest = _manifest([_train("img_a0", "poseX")])
    ev = _eval_doc(outlier_poses=["poseX"], mad_z={"poseX": 4.2})
    out = collect_exclude_candidates(ev, None, manifest, None)
    cands = out["candidates"]
    assert len(cands) == 1 and "4.2" in cands[0]["reason"]


def test_burst_failures_expand_and_dedupe_by_pose():
    """连拍断言失败张:同一姿态被两颗检测器各点名一次只收一组。"""
    manifest = _manifest([_train("img_x0", "poseB"), _train("img_x1", "poseB")])
    corners = _corners_doc(failures=[
        {"detector": "classic", "pose_id": "poseB",
         "spread_median_px": 0.9, "spread_max_px": 1.4,
         "threshold_px": [0.35, 1.4]},
        {"detector": "sb", "pose_id": "poseB",
         "spread_median_px": 0.8, "spread_max_px": 1.2,
         "threshold_px": [0.35, 1.4]},
    ])
    out = collect_exclude_candidates(None, None, manifest, corners)
    cands = out["candidates"]
    assert len(cands) == 2, f"同姿态两颗检测器点名应去重成一组 2 张:{cands}"
    assert {c["image_id"] for c in cands} == {"img_x0", "img_x1"}
    assert "连拍" in cands[0]["source"]
    assert "1.4" in cands[0]["reason"], "理由应带组内波动数字"
    # 容差锁真实形状(生产端落 [中位容差, 最大容差] 二元列表,不是标量)
    assert "中位容差 0.35px" in cands[0]["reason"], cands[0]["reason"]
    assert "最大容差 1.40px" in cands[0]["reason"], cands[0]["reason"]


def test_burst_suspect_image_ids_win_over_whole_pose_expansion():
    """生产端已点名到帧(suspect_image_ids)时逐张收,不整组展开全部连拍。

    corners 轮断言失败时按「偏离超最大容差」点名具体跳位帧;整组展开会把
    大张数好照片混进人工名单,稀释剔除名单的指向性。
    """
    manifest = _manifest([_train("img_x0", "poseB"), _train("img_x1", "poseB"),
                          _train("img_x2", "poseB")])
    corners = _corners_doc(failures=[
        {"detector": "classic", "pose_id": "poseB",
         "suspect_image_ids": ["img_x2"],
         "spread_median_px": 0.9, "spread_max_px": 1.4,
         "threshold_px": [0.35, 1.4]},
    ])
    out = collect_exclude_candidates(None, None, manifest, corners)
    cands = out["candidates"]
    assert [c["image_id"] for c in cands] == ["img_x2"], \
        f"逐帧点名优先,不该整组展开 3 张:{cands}"
    assert cands[0]["pose_id"] == "poseB"
    assert "连拍" in cands[0]["source"]
    assert "1.40px" in cands[0]["reason"], "逐帧点名的理由沿用整组版的数字"


def test_same_image_collected_once_across_routes():
    """同一张照片被多路同时点名只收一条(重复上榜对人工确认是噪声)。

    poseB 既被留一姿态整组展开、又被连拍断言整组展开,img_x0 还被
    入库体检多来源点名——三路重叠后每张只留先到的一条。
    """
    manifest = _manifest([
        _train("img_x0", "poseB", suspect=True,
               sources=["pose_relative:blur", "pose_relative:exposure"]),
        _train("img_x1", "poseB"),
    ])
    ev = _eval_doc(outlier_poses=[{"pose_id": "poseB", "mad_z": 4.2}])
    corners = _corners_doc(failures=[
        {"detector": "classic", "pose_id": "poseB",
         "spread_median_px": 0.9, "spread_max_px": 1.4,
         "threshold_px": [0.35, 1.4]},
    ])
    out = collect_exclude_candidates(ev, None, manifest, corners)
    ids = [c["image_id"] for c in out["candidates"]]
    assert len(ids) == len(set(ids)), f"同一张照片不得重复上榜:{ids}"
    assert ids == ["img_x0", "img_x1"], f"三路重叠后各张只留一条:{ids}"


def test_ingest_multi_source_suspects_only():
    """入库体检存疑:只有被多类判定来源同时点名的才算「计数高」。"""
    manifest = _manifest([
        _train("img_multi", "poseC", suspect=True,
               sources=["pose_relative:blur", "pose_relative:exposure"]),
        _train("img_single", "poseD", suspect=True,
               sources=["pose_relative:blur"]),
    ])
    out = collect_exclude_candidates(None, None, manifest, None)
    cands = out["candidates"]
    assert [c["image_id"] for c in cands] == ["img_multi"], \
        f"单来源存疑不算计数高,只收 2 类来源同时点名的:{cands}"
    assert "2 类判定来源" in cands[0]["reason"]
    assert "pose_relative:blur" in cands[0]["reason"], "理由应带判定来源原文"


def test_zero_candidates_empty_table():
    out = collect_exclude_candidates(_eval_doc(), _corners_doc(), _manifest(), None)
    assert out == {"candidates": []}, f"零候选应落空表:{out}"


# =====================================================================
# 2. build 层 ⑧ 节组装 + suggest_exclude.json 落盘(合成七路树整链)
# =====================================================================
def test_s8_advice_candidates_and_suggest_file(tmp_path, monkeypatch):
    """合成树整链:⑧ 节出建议行与候选行,双模板渲染出三块,名单落盘。

    合成树(test_report_build.build_minimal_tree)里稳定判据 FAIL + 预算
    verdict=fail + 外推区 + 敏感度建议 → 至少 4 条建议行;连拍断言点名
    1 组 → 候选清单 1 条(姿态不在 manifest,走整组候选回退路径)。
    """
    test_report_build.build_minimal_tree(tmp_path)
    monkeypatch.chdir(tmp_path)

    ctx = build_report_context(tmp_path / "output")
    s8 = ctx["s8"]

    # 建议表:合成树触发 稳定 FAIL / 预算 fail / 外推区 / 敏感度建议
    assert len(s8["rows"]) >= 4, f"合成树应触发至少 4 条建议:{s8['rows']}"
    steps_used = {r["step"] for r in s8["rows"]}
    assert steps_used <= set(PIPELINE_STEPS), "建议行步骤名必须出自八步路标"
    # 触发态下不得给「无需回环」
    assert s8["all_pass_line"] is None
    # corner_opt 在合成树缺席 → 判据缺席行如实记录
    assert any("角点寻优" in r["trigger"] for r in s8["absent_rows"])

    # 候选清单:合成树 lopo 无 outlier、存疑 1 张是单来源(不算计数高)、
    # 连拍断言点名 1 组且该姿态不在 manifest → 整组候选(image_id 缺,
    # 理由指引人工补全);固定操作指引照样给出(裁定①)
    assert len(s8["candidates"]) == 1, f"应恰有 1 条连拍断言候选:{s8['candidates']}"
    cand = s8["candidates"][0]
    assert cand["source"] == "连拍断言点名"
    assert cand["image_id"] is None and cand["pose_id"] == 'p<x>&"q'
    assert "人工补全名单" in cand["reason"]
    # 容差按二元列表渲染成白话(不是裸列表「[0.35, 1.4]」)
    assert "中位容差 0.35px / 最大容差 1.40px" in cand["reason"], cand["reason"]
    assert "exclude_images" in s8["guidance_line"]
    assert "calib run-all --from calibrate" in s8["guidance_line"]

    # 双模板都渲染出建议表、候选空表与操作指引
    html = test_report_build._render(ctx, "report.html.j2", autoescape=True)
    md = test_report_build._render(ctx, "report.md.j2", autoescape=False)
    for text in (s8["title"], s8["rows"][0]["trigger"], s8["rows"][0]["action"],
                 s8["guidance_line"], "suggest_exclude.json"):
        assert text in html, f"HTML ⑧节缺:{text[:40]}"
        assert text in md, f"Markdown ⑧节缺:{text[:40]}"

    # 命令层落盘:suggest_exclude.json 与四件产物一起出现,内容 = context 候选
    test_report_build.report_cmd()
    runs = sorted((tmp_path / "output" / "report").glob("run_*"))
    run_dir = runs[-1]
    suggest = json.loads(
        (run_dir / "suggest_exclude.json").read_text(encoding="utf-8"))
    assert suggest == {"candidates": s8["candidates"]}
    assert (run_dir / "report_data.json").is_file()


# =====================================================================
# 3. compare_with_previous:跨运行对比(ITER-03)
# =====================================================================
def _snapshot(selected="m2", test_rms=0.17, sigma=0.0087,
              ratios=(101.0, 1100.0, 1400.0), boot=10, profile="quick") -> dict:
    """合成一份 report_data.json 底账快照(只含 diff 读的键)。"""
    return {
        "created": "2026-10-23T00:00:00+00:00",
        "metrics_raw": {
            "selected": selected,
            "test_rms_px": test_rms,
            "sigma_detector_corner_px": sigma,
            "budget_ratio_median": ratios[0],
            "budget_ratio_p95": ratios[1],
            "budget_ratio_max": ratios[2],
            "n_bootstrap_resamples": boot,
            "profile": profile,
        },
    }


def test_compare_all_keys_with_direction_words():
    current = _snapshot(test_rms=0.16, ratios=(95.0, 1050.0, 1350.0))
    out = compare_with_previous(current, _snapshot())
    assert out["status"] == "ok"
    assert len(out["rows"]) == 6, f"六个对比键一行不少:{out['rows']}"
    by_label = {r["label"]: r for r in out["rows"]}
    # 方向语义按指标含义写死:越低越好的指标,降 = 变好
    assert by_label["当选模型"]["change"] == "持平"
    assert by_label["当选模型"]["current"] == "m2"
    assert by_label["test 整体 RMS(px)"]["change"] == "变好(降低)"
    assert by_label["test 整体 RMS(px)"]["current"] == "0.1600"
    assert by_label["test 整体 RMS(px)"]["previous"] == "0.1700"
    for label in tuple(by_label):
        if label.startswith("预算比值"):
            assert by_label[label]["change"] == "变好(降低)", label
    assert by_label["噪声底·检测器口径 σ(px)"]["change"] == "持平"
    assert out["scale_note"] is None, "规模相同时不给规模备注"


def test_compare_model_change_reports_swap():
    out = compare_with_previous(_snapshot(selected="m3"), _snapshot(selected="m2"))
    row = next(r for r in out["rows"] if r["label"] == "当选模型")
    assert row["change"] == "换成 m3", f"换模型要如实报换了谁:{row}"


def test_compare_missing_previous_key_not_crash():
    """上一份是旧版底账、缺某个键:那一行照出,变化栏记「上次无此数据」。"""
    prev = _snapshot()
    del prev["metrics_raw"]["budget_ratio_p95"]
    prev["metrics_raw"]["budget_ratio_max"] = None
    out = compare_with_previous(_snapshot(), prev)
    by_label = {r["label"]: r for r in out["rows"]}
    row = by_label["预算比值·重投影位移95分位(实测 ÷ 理论下界)"]
    assert row["previous"] == "—" and row["change"] == "上次无此数据"
    row_max = by_label["预算比值·重投影位移最大(实测 ÷ 理论下界)"]
    assert row_max["change"] == "上次无此数据"
    assert len(out["rows"]) == 6, "缺键不崩,其余键照常对比"


def test_compare_first_run_branch():
    out = compare_with_previous(_snapshot(), None)
    assert out["status"] == "首次运行"
    assert out["note"], "首次运行要给一句中文说明"
    assert out["rows"] == [] and out["scale_note"] is None


def test_compare_scale_difference_note_not_regression():
    """实验规模不同(bootstrap 次数)→ 备注解释差异,不冒充回归。"""
    out = compare_with_previous(_snapshot(boot=30, profile="full"),
                                _snapshot(boot=10, profile="quick"))
    note = out["scale_note"]
    assert note, "规模不同必须给备注"
    assert "10 次 / 本次 30 次" in note
    assert "quick / 本次 full" in note
    assert "不能直接当回归读" in note


# =====================================================================
# 4. 「与上次对比」节接入报告(合成树整链:上一轮锚点 → 真实数字对比)
# =====================================================================
def test_report_diff_section_against_previous_round(tmp_path, monkeypatch):
    """预置一份上一轮底账 → 新报告的「与上次对比」节出现真实数字对比。"""
    test_report_build.build_minimal_tree(tmp_path)
    monkeypatch.chdir(tmp_path)

    prev_dir = tmp_path / "output" / "report" / "run_20250101_000000"
    prev_dir.mkdir(parents=True)
    (prev_dir / "report_data.json").write_text(
        json.dumps(_snapshot(test_rms=0.17, ratios=(101.0, 1100.0, 1400.0),
                             boot=10, profile="quick"), ensure_ascii=False),
        encoding="utf-8")

    test_report_build.report_cmd()
    runs = sorted((tmp_path / "output" / "report").glob("run_*"))
    run_dir = runs[-1]
    data = json.loads(
        (run_dir / "report_data.json").read_text(encoding="utf-8"))
    s_diff = data["s_diff"]
    assert s_diff["status"] == "ok"
    assert s_diff["previous_round"] == "run_20250101_000000"
    assert s_diff["previous_created"] == "2026-10-23T00:00:00+00:00"

    by_label = {r["label"]: r for r in s_diff["rows"]}
    assert len(s_diff["rows"]) == 6
    # 合成树 m2 test RMS 0.31 vs 上一轮 0.17 → 如实报变差(不粉饰)
    row = by_label["test 整体 RMS(px)"]
    assert row["current"] == "0.3100" and row["previous"] == "0.1700"
    assert row["change"] == "变差(升高)"
    # bootstrap 规模差异备注:合成树 30 次 vs 上一轮 10 次
    assert "10 次 / 本次 30 次" in s_diff["scale_note"]

    html = (run_dir / "report.html").read_text(encoding="utf-8")
    md = (run_dir / "report.md").read_text(encoding="utf-8")
    for text in (s_diff["title"], "run_20250101_000000",
                 row["current"], row["change"], "变差"):
        assert text in html, f"HTML「与上次对比」节缺:{text}"
        assert text in md, f"Markdown「与上次对比」节缺:{text}"


def test_report_first_run_renders_first_time_note(tmp_path, monkeypatch):
    """没有任何上一轮 → 「与上次对比」节渲染首次运行说明,不崩。"""
    test_report_build.build_minimal_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    ctx = build_report_context(tmp_path / "output")
    assert ctx["s_diff"]["status"] == "首次运行"
    html = test_report_build._render(ctx, "report.html.j2", autoescape=True)
    md = test_report_build._render(ctx, "report.md.j2", autoescape=False)
    for text in (ctx["s_diff"]["title"], "第一份报告"):
        assert text in html, f"HTML 缺首次运行说明:{text}"
        assert text in md, f"Markdown 缺首次运行说明:{text}"
