"""张数不齐变体数据集全链测试:仅改 config,缩水数据上的全链契约。

整文件标记 integration + realdata(需要真实 input_data 数据与分钟级算力,
日常开发用 `uv run pytest -m "not realdata"` 自动跳过;夜间/合并前再跑)。

测的是什么(REU-01/02 的验收载体):把"缩小版新数据集"做成**张数不齐**的
变体——config 从仓库复制后只改数据路径写法与 bic_n_poses(80→4),其余
零改动;input_data/train 放 4 个姿态目录,连拍张数故意删成 8/8/5/2 张
(不齐),test 平铺 5 张。然后像真实用户一样跑 `calib run-all`,断言
四层契约:

1. ①–⑥ 步全跑通(--until evaluate,退出码 0):照片清单如实、重复张
   分组组数 = 最少重复张数、噪声单帧折算路径被行使、选型选出已收敛
   的模型——张数不齐 + 仅改 config,考试环节完整成立(REU-01/02);
2. ⑦ 步误差归因按设计正确拒绝(--from analyze,退出码 1):缩水数据
   上误差预算的 ANA-01 铁律守卫(任何一个格子的实测重投影散布低于
   理论下界即硬报错)当场拦下,中文点名原因,失败报告照常落盘、各段
   为 null——绝不冒充一份站不住的证据链;
3. ⑧ 步报告对失败链如实披露(calib report,退出码 0):分析轮缺席进
   披露清单、五件产物齐、回环建议的剔除候选照常生成——证据链工具
   遇到坏链先说实话(REU-02)。

为什么变体全链(含⑦)跑不通、而本测试把守卫当契约来锁:重复张分组
组数 = 最少重复张数(2 组),而误差预算的逐格铁律要求**每一个**
(视图,角点)格子的组间实测散布不低于理论下界——组间散布来自分组
解之间的参数差在小数据上方向有限(2 组只有 1 个差方向),总有格子
与该方向近乎正交、实测近乎零,而缩水数据的理论下界又偏松,铁律必然
触发。这不是缺陷:真数据(8 组、80 姿态)靠方向丰富 + 下界极紧才能
全格通过;缩水数据被拦下正是"实测低于理论下界物理上不可能"口径铁律
在正确工作。变体全链放行(放宽守卫/加大变体)是需要用户拍板的口径
决定,已登记计划 SUMMARY 交主会话人工核查时裁决。

另核一道血缘闭合(防串轮,T-06-09):角点产物登记的照片清单指纹与实际
文件一致、噪声产物登记的全部上游指纹逐环核对——临时树从零建起,不存在
旧轮冒充本轮的可能,链对不上就是代码坏了。

成本取舍(照 test_smoke_realdata 体例在 docstring 写明):变体集只有
28 张照片,三段子进程(--until evaluate 全链最重、--from analyze 秒级、
report 秒级)合计分钟级,允许进 pytest 但只挂 realdata 夜间档标记;
真数据全量(705 张)run-all 约两小时,只由人工 checkpoint 执行,绝不进
测试。软链接只读不复制,尊重 input_data 只读契约。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.realdata]

REPO_ROOT = Path(__file__).resolve().parents[1]

# 4 个姿态的连拍保留张数:8/8/5/2,故意不齐。
# 最少 2 张是下限:重复张分组组数 = 最少重复张数,误差预算段要求
# 组数 >= 2(少于 2 组谈不了组间散布,中文报错),2 组恰好踩住边界;
# 连拍删到 1 张会在更早处被同一道守卫拦下,不属于本测试的范围。
REPEAT_COUNTS = (8, 8, 5, 2)
N_TEST_IMAGES = 5
# 各段子进程的超时上限(秒):①–⑥ 全链里注入实验最重,给足余量
RUN_TIMEOUTS_S = {"to_evaluate": 7200, "analyze": 1800, "report": 600}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_run(stage_dir: Path) -> Path:
    runs = sorted(p for p in stage_dir.iterdir()
                  if p.is_dir() and p.name.startswith("run_"))
    assert runs, f"{stage_dir} 下没有任何 run 目录"
    return runs[-1]


def _run_calib(args: list[str], tmp_root: Path, timeout: int) -> subprocess.CompletedProcess:
    """在变体树里以真实用户方式跑一条 calib 命令(独立进程,只靠落盘
    文件传递状态)。"""
    return subprocess.run(
        ["uv", "run", "--project", str(REPO_ROOT), "calib", *args],
        check=False,
        cwd=tmp_root, capture_output=True, text=True,
        timeout=timeout,
        env={**os.environ},
    )


@pytest.fixture(scope="module")
def variant_tree(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """搭建张数不齐变体数据集,跑三段命令,返回产物路径与运行结果。

    构造纪律:config 从仓库复制后仅改数据路径写法与 bic_n_poses,其余零
    改动——"仅改 config 即可跑通"本身就是要断言的事(REU-02),这里改的
    每一行都会在下方测试里逐项对账。input_data 全部软链接单张照片(姿态
    目录本身用真实目录承载,才能只挑部分连拍而不碰仓库原图)。

    三段命令:①–⑥ 全链(--until evaluate,必须退出码 0)→ ⑦ 单段
    (--from analyze,预期被守卫拦下退出码 1)→ ⑧ 报告(对失败链如实
    披露,必须退出码 0)。
    """
    tmp_root = tmp_path_factory.mktemp("run_all_variant")

    # config:从仓库原样复制,只改三处——数据路径写成显式相对写法
    # (证明路径由 config 决定)、bic_n_poses 80→4(有效姿态数)。
    cfg_dir = tmp_root / "config"
    cfg_dir.mkdir()
    for yaml_name in ("board.yaml", "calibration.yaml"):
        text = (REPO_ROOT / "config" / yaml_name).read_text(encoding="utf-8")
        if yaml_name == "calibration.yaml":
            text = text.replace('train: "input_data/train"',
                                'train: "./input_data/train"')
            text = text.replace('test: "input_data/test"',
                                'test: "./input_data/test"')
            text = text.replace("bic_n_poses: 80",
                                f"bic_n_poses: {len(REPEAT_COUNTS)}")
        (cfg_dir / yaml_name).write_text(text, encoding="utf-8")

    # input_data/train:字典序前 4 个真实姿态目录,各自只软链接前 N 张连拍
    real_train = REPO_ROOT / "input_data" / "train"
    real_test = REPO_ROOT / "input_data" / "test"
    pose_dirs = sorted(p for p in real_train.iterdir()
                       if p.is_dir())[:len(REPEAT_COUNTS)]
    test_imgs = sorted(p for p in real_test.iterdir()
                       if p.suffix == ".jpg")[:N_TEST_IMAGES]
    assert len(pose_dirs) == len(REPEAT_COUNTS) and len(test_imgs) == N_TEST_IMAGES, \
        "真实数据不足,变体测试前提被破坏"
    (tmp_root / "input_data" / "train").mkdir(parents=True)
    (tmp_root / "input_data" / "test").mkdir(parents=True)
    for pose_dir, n_keep in zip(pose_dirs, REPEAT_COUNTS):
        dst = tmp_root / "input_data" / "train" / pose_dir.name
        dst.mkdir()
        frames = sorted(pose_dir.glob("*.jpg"))[:n_keep]
        assert len(frames) == n_keep, f"{pose_dir.name} 连拍不足 {n_keep} 张"
        for img in frames:
            os.symlink(img, dst / img.name)
    for img in test_imgs:
        os.symlink(img, tmp_root / "input_data" / "test" / img.name)

    # 第一段:①–⑥ 全链(quick 档)。失败带完整输出报错,一眼可见。
    to_evaluate = _run_calib(["run-all", "--until", "evaluate"], tmp_root,
                             RUN_TIMEOUTS_S["to_evaluate"])
    if to_evaluate.returncode != 0:
        raise AssertionError(
            f"变体 ①–⑥ 全链断裂(退出码 {to_evaluate.returncode})\n"
            f"--- 标准输出 ---\n{to_evaluate.stdout}\n"
            f"--- 标准错误 ---\n{to_evaluate.stderr}"
        )

    # 第二段:⑦ 误差归因单段,预期被 ANA-01 铁律守卫拦下(退出码 1)。
    analyze = _run_calib(["run-all", "--from", "analyze"], tmp_root,
                         RUN_TIMEOUTS_S["analyze"])
    if analyze.returncode != 1:
        raise AssertionError(
            f"变体 ⑦ 预期被守卫拦下(退出码 1),实际退出码 "
            f"{analyze.returncode}——守卫行为或数据前提变了,请排查\n"
            f"--- 标准输出 ---\n{analyze.stdout}\n"
            f"--- 标准错误 ---\n{analyze.stderr}"
        )

    # 第三段:⑧ 报告,对失败链如实披露(退出码 0)。
    report = _run_calib(["report"], tmp_root, RUN_TIMEOUTS_S["report"])
    if report.returncode != 0:
        raise AssertionError(
            f"变体报告对失败链渲染断裂(退出码 {report.returncode})\n"
            f"--- 标准输出 ---\n{report.stdout}\n"
            f"--- 标准错误 ---\n{report.stderr}"
        )

    out = tmp_root / "output"
    return {
        "tmp_root": tmp_root,
        "to_evaluate": to_evaluate,
        "analyze": analyze,
        "report": report,
        "ingest_run": _latest_run(out / "ingest"),
        "corners_run": _latest_run(out / "corners"),
        "calibration_run": _latest_run(out / "calibration"),
        "noise_run": _latest_run(out / "noise"),
        "evaluate_run": _latest_run(out / "evaluate"),
        "analyze_root": out / "analyze",
        "report_root": out / "report",
    }


def test_variant_config_only_chain_to_evaluate(variant_tree: dict):
    """仅改 config 的变体集,①–⑥ 全链跑通且各路断言全过。

    退出码 0(夹具已保证,这里再补无堆栈与收尾播报);照片清单如实;
    重复张分组组数 = 最少重复张数;噪声单帧折算路径被行使;当选模型
    标定解收敛。
    """
    combined = variant_tree["to_evaluate"].stdout + variant_tree["to_evaluate"].stderr
    assert "Traceback" not in combined, "①–⑥ 输出里出现了报错堆栈"
    assert "全部完成" in combined, "没看到区间全跑通的收尾播报"

    # 照片清单如实:张数不齐的 train + 5 张 test,分布逐项对上
    manifest = json.loads(
        (variant_tree["ingest_run"] / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest["summary"]
    assert summary["n_train"] == sum(REPEAT_COUNTS)
    assert summary["n_test"] == N_TEST_IMAGES
    assert summary["n_poses"] == len(REPEAT_COUNTS)
    assert summary["repeats_distribution"] == dict(
        Counter(str(n) for n in REPEAT_COUNTS)), \
        "重复张数分布应是不齐的 8/8/5/2 各一个姿态(8 张的有两个姿态)"

    # 重复张分组组数 = 最少重复张数(张数不齐按最少截断,数据驱动)
    evaluate_doc = json.loads(
        (variant_tree["evaluate_run"] / "evaluate.json").read_text(encoding="utf-8"))
    assert evaluate_doc["failure_note"] is None, \
        f"evaluate 本轮是失败轮:{evaluate_doc['failure_note']}"
    min_repeats = min(REPEAT_COUNTS)
    slice_models = evaluate_doc["slices"]["models"]
    assert slice_models, "slices 实验没有任何模型的记录"
    for model_id, block in slice_models.items():
        assert block["n_groups"] == min_repeats, \
            f"{model_id} 的重复张分组组数应为最少重复张数 {min_repeats}," \
            f"实际 {block['n_groups']}"

    # 单帧折算路径被真实行使:test 独立照没有连拍可减,逐张走
    # "单帧像素噪声经传递曲线折算"路径(来源逐张标注);train 侧全部
    # 姿态连拍 >=2 张,按来源标注走连拍波动口径
    noise_doc = json.loads(
        (variant_tree["noise_run"] / "noise.json").read_text(encoding="utf-8"))
    assert noise_doc["failure_note"] is None, \
        f"noise 本轮是失败轮:{noise_doc['failure_note']}"
    assert noise_doc["n_poses_temporal"] == len(REPEAT_COUNTS)
    assert noise_doc["n_poses_fallback"] == 0, \
        "train 全部姿态连拍 >=2 张,不应有姿态退化到单帧折算"
    test_split = noise_doc["splits"]["test"]
    assert test_split["n_images"] == N_TEST_IMAGES
    per_image = test_split["per_image"]
    assert len(per_image) == N_TEST_IMAGES
    assert all(rec["source"] == "immerkaer_transfer" for rec in per_image), \
        "test 独立照应逐张标注单帧折算来源"

    # 选型选出的模型,其标定解已收敛(主路线 scipy 精化组合)
    selected = evaluate_doc["selection"]["selected"]
    assert selected, f"选型没有选出模型:{evaluate_doc['selection']}"
    selected_doc = json.loads(
        (variant_tree["calibration_run"] / f"{selected}__scipy" / "result.json")
        .read_text(encoding="utf-8"))
    assert selected_doc["converged"] is True, \
        f"当选模型 {selected} 的标定解未收敛:{selected_doc['failure_note']}"


def test_variant_analyze_guard_is_the_contract(variant_tree: dict):
    """⑦ 误差归因在缩水数据上按设计正确拒绝:ANA-01 铁律中文硬报错、
    退出码 1、不吐堆栈、失败报告照常落盘且各分析段为 null(不冒充)。"""
    combined = variant_tree["analyze"].stdout + variant_tree["analyze"].stderr
    assert "Traceback" not in combined, "守卫拒绝应给中文一句,不应吐堆栈"
    assert "误差预算硬报错" in combined and "ANA-01 铁律" in combined, \
        f"没看到 ANA-01 铁律守卫的中文点名:{combined[-600:]}"

    analyze_runs = sorted(p for p in variant_tree["analyze_root"].glob("run_*")
                          if p.is_dir())
    assert analyze_runs, "守卫拒绝也应照常落失败报告"
    analyze_doc = json.loads(
        (analyze_runs[-1] / "analyze.json").read_text(encoding="utf-8"))
    assert analyze_doc["failure_note"], "失败轮必须写非空中文失败说明"
    assert "ANA-01" in str(analyze_doc["failure_note"])
    for section in ("budget", "residual_structure", "test_xyz",
                    "sensitivity", "corner_opt", "anisotropy"):
        assert analyze_doc.get(section) is None, \
            f"失败轮的 {section} 段必须为 null,不冒充数值"


def test_variant_report_discloses_failed_analyze(variant_tree: dict):
    """⑧ 报告对失败链如实披露:五件产物齐、缺席披露点名误差归因失败轮、
    回环建议的剔除候选照常生成——证据链工具遇坏链先说实话(REU-02)。"""
    combined = variant_tree["report"].stdout + variant_tree["report"].stderr
    assert "Traceback" not in combined, "报告输出里出现了报错堆栈"

    report_runs = sorted(p for p in variant_tree["report_root"].glob("run_*")
                         if p.is_dir())
    assert report_runs, "run-all 没有落任何报告轮"
    report_run = report_runs[-1]
    for name in ("report.html", "report.md", "report_data.json",
                 "suggest_exclude.json"):
        assert (report_run / name).is_file(), f"缺报告产物:{name}"
    figures = report_run / "figures"
    assert figures.is_dir() and any(figures.iterdir()), "报告图目录为空"

    context = json.loads(
        (report_run / "report_data.json").read_text(encoding="utf-8"))
    disclosures = context.get("disclosures") or []
    assert any("误差归因" in str(d) for d in disclosures), \
        f"失败的分析轮应进披露清单,实际披露:{disclosures}"

    candidates = context.get("s8", {}).get("candidates") or []
    assert candidates, "回环建议的剔除候选不应为空(留一点名了问题姿态)"
    assert any("留一姿态" in str(c.get("source", "")) for c in candidates), \
        f"候选应含留一姿态点名来源:{[c.get('source') for c in candidates][:5]}"


def test_variant_lineage_chain_closed(variant_tree: dict):
    """血缘链闭合(防串轮,T-06-09):角点登记的照片清单指纹、噪声产物的
    全部上游指纹,逐环与实际文件核对——链上任何一环对不上当场显形。"""
    tmp_root = variant_tree["tmp_root"]
    manifest_path = variant_tree["ingest_run"] / "manifest.json"

    # 第一环:角点产物记录的照片清单路径与指纹 = 本轮实际清单
    meta = json.loads(
        (variant_tree["corners_run"] / "corners_train_sb.meta.json")
        .read_text(encoding="utf-8"))
    recorded = meta["lineage"]["upstream"]["manifest.json"]
    manifest_from_lineage = tmp_root / recorded["path"]
    assert manifest_from_lineage.resolve() == manifest_path.resolve()
    assert recorded["sha256"] == _sha256(manifest_path), \
        "角点产物记录的清单指纹与实际文件不符,血缘链断裂"

    # 第二环:噪声产物登记的全部上游(角点/清单/逐姿态明细)逐项指纹核对
    noise_path = variant_tree["noise_run"] / "noise.json"
    noise_doc = json.loads(noise_path.read_text(encoding="utf-8"))
    upstream = noise_doc.get("lineage", {}).get("upstream")
    assert upstream, "noise.json 没带上游血缘"
    for name, entry in upstream.items():
        f = tmp_root / entry["path"]
        assert f.is_file(), f"noise.json 血缘登记的上游 {name} 不在:{f}"
        assert entry["sha256"] == _sha256(f), \
            f"noise.json 记录的 {name} 指纹与实际文件不符,血缘链断裂"
