"""真实数据噪声阶段端到端集成测试:独立进程跑 calib noise 全流程(NOI-01/NOI-02 收口)。

整文件标记 integration(需要真实 input_data 与 corners 产物,日常开发用
`uv run pytest -m "not integration"` 自动跳过;数据或上游产物缺失时整文件
跳过并写明原因,不误报失败)。

测的是什么:
1. calib noise 一条命令跑通,noise.json 顶层契约键一个不少,双口径 σ 是
   有限正值;
2. 来源计数对账:n_poses_temporal + n_poses_fallback == 数据清单里的
   train 姿态总数(数据驱动不写死);test 切分每张独立照都走折算路径;
3. 合理性:时域口径 ≥ 检测器口径(时域含拍摄瞬间的位姿抖动,不应小于
   纯检测器口径;相等时给出 clamped 比例提示);
4. 注入产物形状与 meta 一致(σ 档数 / 重复数从产物读,不写死);
   σ=0 校准档的角点位移恰为零(基线对齐的硬标准);
5. 血缘第一环:noise.json 记录的 corners npz 上游文件真实存在
   (03-06 会把全链指纹断言扩展到 fisher)。

复用规则(实测成本说明):注入实验在这批 1440p 数据上实测约两小时
(计划预估 5–15 分钟严重偏小)。每个注入副本的噪声种子由
(图像, σ 档, 重复) 三元组确定性推导,同输入逐位复现,所以只要
output/noise 下已有一份"完整(非失败轮)且血缘指纹与当前 corners
产物对得上"的轮子,就直接复用它做断言,不重跑两小时的确定性计算;
corners 产物一变(重跑/换参数)指纹就对不上,自动触发全新一轮。
没有可复用轮时才真正跑命令(独立进程,超时上限 3 小时)。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from calib.noise.artifacts import REPORT_KEYS, load_noise_report

pytestmark = [pytest.mark.integration, pytest.mark.realdata]

REPO_ROOT = Path(__file__).resolve().parents[1]
NOISE_TIMEOUT_S = 10800  # 注入实验实测约两小时,超时上限给足 3 小时


def _latest_run(stage_dir: Path) -> Path:
    runs = sorted(p for p in stage_dir.iterdir()
                  if p.is_dir() and p.name.startswith("run_"))
    assert runs, f"{stage_dir} 下没有任何 run 目录"
    return runs[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reusable_run(corners_run: Path) -> Path | None:
    """找最近一份"完整且血缘对得上当前 corners 产物"的噪声轮,没有则 None。

    完整 = noise.json 在且 failure_note 为空(失败轮的数值字段是 null,
    断言不了);指纹对得上 = 记录的 corners npz sha256 与实际文件一致
    (上游一变就自动触发全新一轮,绝不拿旧轮的产物充数)。
    """
    noise_root = REPO_ROOT / "output" / "noise"
    if not noise_root.is_dir():
        return None
    corners_npz = corners_run / "corners_train_sb.npz"
    actual_sha = _sha256(corners_npz)
    for run in sorted((p for p in noise_root.iterdir()
                       if p.is_dir() and p.name.startswith("run_")),
                      reverse=True):
        doc_path = run / "noise.json"
        if not doc_path.is_file():
            continue
        with open(doc_path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("failure_note") is not None:
            continue
        recorded = (doc.get("lineage", {}).get("upstream", {})
                    .get("corners_train_sb.npz", {}))
        if recorded.get("sha256") == actual_sha:
            return run
    return None


@pytest.fixture(scope="module")
def noise_tree() -> dict:
    """在仓库根目录独立进程跑 calib noise,返回运行结果与读回的产物。

    前置数据(真实照片、最新一轮 corners 产物)缺失时整文件跳过;
    已有完整且血缘对得上的历史轮则直接复用(复用规则见模块 docstring);
    命令失败带着完整输出报错,让失败原因一眼可见。
    """
    if not (REPO_ROOT / "input_data" / "train").is_dir() \
            or not (REPO_ROOT / "input_data" / "test").is_dir():
        pytest.skip("真实 input_data 不在,跳过噪声阶段集成测试")
    corners_out = REPO_ROOT / "output" / "corners"
    if not corners_out.is_dir():
        pytest.skip("还没有 corners 产物,先跑 calib corners 再来")

    corners_run = _latest_run(corners_out)
    noise_run = _reusable_run(corners_run)
    result = None
    if noise_run is not None:
        print(f"\n复用完整噪声轮(血缘指纹与当前 corners 对得上):{noise_run}")
    else:
        result = subprocess.run(
            ["uv", "run", "--project", str(REPO_ROOT), "calib", "noise"],
            check=False,
            cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=NOISE_TIMEOUT_S, env={**os.environ},
        )
        if result.returncode != 0:
            raise AssertionError(
                f"calib noise 退出码 {result.returncode}\n"
                f"--- 标准输出 ---\n{result.stdout}\n--- 标准错误 ---\n{result.stderr}"
            )
        noise_run = _latest_run(REPO_ROOT / "output" / "noise")
        assert (noise_run / "noise.json").is_file(), \
            f"命令退出码 0 但最新轮没有 noise.json:{noise_run}"
    loaded = load_noise_report(noise_run)
    lineage = loaded["report"]["lineage"]
    manifest_path = REPO_ROOT / lineage["upstream"]["manifest.json"]["path"]
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    return {"result": result, "noise_run": noise_run, "loaded": loaded,
            "manifest": manifest}


def test_report_contract_and_dual_sigma_finite(noise_tree: dict):
    """noise.json 存在、契约键一个不少;双口径 σ 是有限正值;本轮非失败轮。"""
    report = noise_tree["loaded"]["report"]
    for key in REPORT_KEYS:
        assert key in report, f"noise.json 缺契约键:{key}"
    assert report["failure_note"] is None, f"本轮是失败轮:{report['failure_note']}"

    for key in ("sigma_detector_corner", "sigma_temporal_corner"):
        value = report[key]
        assert value is not None and np.isfinite(value) and value > 0, \
            f"{key} 应为有限正值,实际 {value!r}"


def test_source_counts_cover_all_poses_and_test_split(noise_tree: dict):
    """来源计数与数据清单对账:全部 train 姿态都被标注;test 每张独立照
    都走折算路径(独立照天然无连拍)。"""
    report = noise_tree["loaded"]["report"]
    images = noise_tree["manifest"]["images"]
    train_poses = {r["pose_id"] for r in images if r["split"] == "train"}
    test_ids = [r["image_id"] for r in images if r["split"] == "test"]

    counts = report["per_source_counts"]
    assert report["n_poses_temporal"] + report["n_poses_fallback"] \
        == len(train_poses), \
        (f"来源计数 {report['n_poses_temporal']} + {report['n_poses_fallback']}"
         f" != manifest 里的 train 姿态总数 {len(train_poses)}")
    assert counts["temporal"] == report["n_poses_temporal"]
    assert counts["immerkaer_transfer"] == report["n_poses_fallback"]

    per_image = report["splits"]["test"]["per_image"]
    assert report["splits"]["test"]["n_images"] == len(test_ids)
    assert [e["image_id"] for e in per_image] == test_ids
    assert all(e["source"] == "immerkaer_transfer" for e in per_image), \
        "test 独立照必须逐条走折算路径,不许混进别的来源"


def test_temporal_not_below_detector_caliber(noise_tree: dict):
    """合理性(双口径防混线):时域口径 ≥ 检测器口径——时域含位姿抖动,
    不应小于纯检测器口径;相等时提示 clamped 比例(差距全在测量不确定度内)。"""
    report = noise_tree["loaded"]["report"]
    temporal = report["sigma_temporal_corner"]
    detector = report["sigma_detector_corner"]
    assert temporal >= detector, \
        (f"时域口径 {temporal:.4f} 小于检测器口径 {detector:.4f}:"
         f"两口径疑似混线,请检查传递曲线折算路径")
    if temporal == detector:
        clamped = noise_tree["loaded"]["temporal"]["clamped"]
        share = float(np.mean(clamped)) if clamped.size else 0.0
        print(f"提示:双口径相等,clamped 姿态占比 {share * 100:.0f}%")


def test_injection_artifacts_shape_and_zero_reference_level(noise_tree: dict):
    """注入产物:位移数组形状与 noise.json 记录的规模一致(σ 档数/重复数
    从产物读,不写死);transfer_curve.json 含 σ=0 档且该档位移恰为零。"""
    report = noise_tree["loaded"]["report"]
    images = noise_tree["manifest"]["images"]
    n_train = sum(1 for r in images if r["split"] == "train")

    injection = report["injection"]
    grid = injection["grid"]
    n_levels, n_repeats = len(grid), injection["n_repeats"]
    with np.load(noise_tree["noise_run"] / injection["displacements_npz"],
                 allow_pickle=False) as data:
        disp = data["displacements"]
        assert disp.ndim == 5 and disp.shape == \
            (n_train, n_levels, n_repeats, 88, 2), \
            f"位移数组形状 {disp.shape} 与 meta 记录的规模对不上"

    with open(noise_tree["noise_run"] / report["transfer_curve"]["path"],
              encoding="utf-8") as f:
        curve = json.load(f)
    assert curve["grid"][0] == 0.0, "传递曲线网格第一档必须是 σ=0 校准行"
    zero_level = next(e for e in curve["levels"] if e["sigma"] == 0.0)
    assert zero_level["rms_euclidean"] == pytest.approx(0.0, abs=1e-9), \
        ("σ=0 校准档位移不为零:基线角点没有对齐,"
         f"实际 rms_euclidean = {zero_level['rms_euclidean']}")
    assert zero_level.get("role") == "reference"


def test_lineage_first_ring_and_detail_readback(noise_tree: dict):
    """血缘第一环:noise.json 记录的每个上游文件真实存在(指纹闭合的
    前提);明细 npz 读回行数与 json 汇总一致(汇总只是索引不是替代)。"""
    report = noise_tree["loaded"]["report"]
    lineage = report["lineage"]
    assert lineage["upstream"], "noise.json 没带上游血缘"
    for name, entry in lineage["upstream"].items():
        path = REPO_ROOT / entry["path"]
        assert path.is_file(), \
            f"血缘记录的上游 {name} 不在:{path}(血缘链第一环断裂)"
    upstream_names = set(lineage["upstream"])
    assert "corners_train_sb.npz" in upstream_names
    assert "manifest.json" in upstream_names

    n_poses = report["n_poses_temporal"] + report["n_poses_fallback"]
    temporal = noise_tree["loaded"]["temporal"]
    assert temporal is not None, "成功轮必须有逐姿态明细 npz"
    assert temporal["pose_ids"].shape == (n_poses,)
    assert temporal["sigma_per_corner_all"].shape == (n_poses, 88, 2)
    assert temporal["sigma_pose"].shape == (n_poses,)

    images = noise_tree["manifest"]["images"]
    n_images = len(images)
    immerkaer = noise_tree["loaded"]["immerkaer"]
    assert immerkaer is not None and immerkaer["image_ids"].shape == (n_images,)
    assert set(immerkaer["split"].tolist()) == {"train", "test"}
