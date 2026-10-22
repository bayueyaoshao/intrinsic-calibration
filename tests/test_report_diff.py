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
from calib.report.diff import collect_exclude_candidates
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
    manifest = _manifest([
        _train("img_a0", "poseX"), _train("img_a1", "poseX"),
        _train("img_b0", "poseY"),
    ])
    ev = _eval_doc(outlier_poses=["poseX"], mad_z={"poseX": 4.2})
    out = collect_exclude_candidates(ev, None, manifest, None)
    cands = out["candidates"]
    assert len(cands) == 2, f"姿态 poseX 有 2 张连拍,应各出一条候选:{cands}"
    assert {c["image_id"] for c in cands} == {"img_a0", "img_a1"}
    for c in cands:
        assert c["pose_id"] == "poseX"
        assert "留一姿态" in c["source"]
        assert "4.2" in c["reason"], "理由应带稳健 z 分数原文数字"
        assert "3.5" in c["reason"], "理由应带门槛数字(白话对照)"


def test_burst_failures_expand_and_dedupe_by_pose():
    """连拍断言失败张:同一姿态被两颗检测器各点名一次只收一组。"""
    manifest = _manifest([_train("img_x0", "poseB"), _train("img_x1", "poseB")])
    corners = _corners_doc(failures=[
        {"detector": "classic", "pose_id": "poseB",
         "spread_median_px": 0.9, "spread_max_px": 1.4, "threshold_px": 1.0},
        {"detector": "sb", "pose_id": "poseB",
         "spread_median_px": 0.8, "spread_max_px": 1.2, "threshold_px": 1.0},
    ])
    out = collect_exclude_candidates(None, None, manifest, corners)
    cands = out["candidates"]
    assert len(cands) == 2, f"同姿态两颗检测器点名应去重成一组 2 张:{cands}"
    assert {c["image_id"] for c in cands} == {"img_x0", "img_x1"}
    assert "连拍" in cands[0]["source"]
    assert "1.4" in cands[0]["reason"], "理由应带组内波动数字"


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
