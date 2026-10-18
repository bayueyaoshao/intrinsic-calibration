"""真实数据端到端冒烟测试:干净临时目录上三命令独立进程接力贯通。

整文件标记 integration(需要真实 input_data 数据,日常开发用
`uv run pytest -m "not integration"` 自动跳过)。

测的是什么:临时目录就是一份"缩小版新数据集"——config 从仓库复制,
input_data/train 软链接真实前 3 个姿态目录(24 张)、test 软链接前 5 张,
然后像真实用户一样依次运行 calib ingest → calib corners → calib calibrate
(每条命令独立进程)。断言三件事:
1. 三命令退出码全 0,产物齐全(manifest / 4 份角点 npz / 矩阵组合目录);
2. 血缘链闭合:标定产物记录的角点文件指纹与实际文件一致,角点产物记录的
   数据清单指纹与实际文件一致——链上任何一环被篡改都会在这里显形(T-03-01);
3. 未收敛组合(小样本上模型失配很常见)产物结构也必须完整(键一个不少)。

calibrate 一步跑的是配置矩阵(阶段二演进:模型 × 求解路径 × 损失核,
仓库默认配置 = 4 模型 × {opencv, scipy(linear), scipy(soft_l1)} 共 12 组合),
组合目录名与 result.json 键契约见 calib.solve.result / matrix。

软链接只读不复制,尊重 input_data 只读契约。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]

# 矩阵组合目录名(镜像仓库 config/calibration.yaml 默认矩阵:
# models [m1..m4] × solvers [opencv, scipy] × scipy.losses [linear, soft_l1])
MATRIX_COMBOS = tuple(
    f"{model}__{suffix}"
    for model in ("m1", "m2", "m3", "m4")
    for suffix in ("opencv", "scipy", "scipy_softl1")
)
# result.json 必须齐备的键(失败组合也一样,数值键为 null)
RESULT_JSON_KEYS = (
    "lineage", "model_id", "solver", "solver_meta", "params",
    "image_ids", "image_size", "n_views", "n_points_per_view",
    "init_from", "converged", "failure_note", "rms_overall", "per_view_rms",
)


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


@pytest.fixture(scope="module")
def smoke_tree(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """搭建缩小版数据集,跑通三命令,返回各产物路径与运行结果。

    任何一步失败都带着该命令的完整输出报错,让失败原因一眼可见。
    """
    tmp_root = tmp_path_factory.mktemp("smoke_realdata")

    # config:从仓库原样复制(参数唯一来源,缩小版数据集也要同一套配置)
    cfg_dir = tmp_root / "config"
    cfg_dir.mkdir()
    for yaml_name in ("board.yaml", "calibration.yaml"):
        shutil.copy(REPO_ROOT / "config" / yaml_name, cfg_dir / yaml_name)

    # input_data:train 取字典序前 3 个姿态目录、test 取前 5 张,全部软链接
    real_train = REPO_ROOT / "input_data" / "train"
    real_test = REPO_ROOT / "input_data" / "test"
    pose_dirs = sorted(p for p in real_train.iterdir() if p.is_dir())[:3]
    test_imgs = sorted(p for p in real_test.iterdir() if p.suffix == ".jpg")[:5]
    assert len(pose_dirs) == 3 and len(test_imgs) == 5, "真实数据不足,冒烟前提被破坏"
    (tmp_root / "input_data" / "train").mkdir(parents=True)
    (tmp_root / "input_data" / "test").mkdir(parents=True)
    for pose_dir in pose_dirs:
        os.symlink(pose_dir, tmp_root / "input_data" / "train" / pose_dir.name)
    for img in test_imgs:
        os.symlink(img, tmp_root / "input_data" / "test" / img.name)

    # 三命令独立进程接力(每条都是全新进程,只靠落盘文件传递状态)
    outputs: dict[str, subprocess.CompletedProcess] = {}
    for step in ("ingest", "corners", "calibrate"):
        result = subprocess.run(
            ["uv", "run", "--project", str(REPO_ROOT), "calib", step],
            check=False,
            cwd=tmp_root, capture_output=True, text=True, timeout=900,
            env={**os.environ},
        )
        outputs[step] = result
        if result.returncode != 0:
            raise AssertionError(
                f"冒烟链在 calib {step} 处断裂(退出码 {result.returncode})\n"
                f"--- 标准输出 ---\n{result.stdout}\n--- 标准错误 ---\n{result.stderr}"
            )

    out = tmp_root / "output"
    ingest_run = _latest_run(out / "ingest")
    corners_run = _latest_run(out / "corners")
    calibration_run = _latest_run(out / "calibration")
    return {
        "tmp_root": tmp_root,
        "outputs": outputs,
        "ingest_run": ingest_run,
        "corners_run": corners_run,
        "calibration_run": calibration_run,
    }


def test_three_commands_relay_and_artifacts_complete(smoke_tree: dict):
    """三命令退出码全 0,各阶段产物齐全且规模正确(3 姿态 × 8 + 5 张)。"""
    assert all(r.returncode == 0 for r in smoke_tree["outputs"].values())

    manifest_path = smoke_tree["ingest_run"] / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["n_train"] == 24
    assert manifest["summary"]["n_test"] == 5

    corners_run = smoke_tree["corners_run"]
    for split, n_rows in (("train", 24), ("test", 5)):
        for detector in ("classic", "sb"):
            npz_path = corners_run / f"corners_{split}_{detector}.npz"
            assert npz_path.is_file(), f"缺角点文件:{npz_path}"
            with np.load(npz_path, allow_pickle=False) as data:
                assert data["image_ids"].shape == (n_rows,)
                assert data["uv"].shape == (n_rows, 88, 2)

    calibration_run = smoke_tree["calibration_run"]
    for name in MATRIX_COMBOS:
        assert (calibration_run / name / "result.json").is_file(), \
            f"缺标定产物:{name}"
    comparison = json.loads(
        (calibration_run / "comparison.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in comparison] == list(MATRIX_COMBOS), \
        "comparison.json 的组合名/顺序与配置矩阵不一致"


def test_lineage_chain_closed(smoke_tree: dict):
    """血缘链闭合:标定 → 角点 → 数据清单,每一环的文件指纹都与实际文件一致。

    链上任何一环被篡改(换了角点文件、换了清单),指纹就对不上,当场显形。
    产物里记录的路径是"相对当时工作目录"的,临时树就是当时的工作目录;
    scipy 组合多一环:init_result.json 指向同轮同模型的 opencv 初值产物。
    """
    tmp_root = smoke_tree["tmp_root"]
    corners_run = smoke_tree["corners_run"]
    manifest_path = smoke_tree["ingest_run"] / "manifest.json"

    # 第一环:角点产物记录的数据清单指纹 = 实际清单文件
    meta = json.loads(
        (corners_run / "corners_train_sb.meta.json").read_text(encoding="utf-8"))
    recorded_manifest = meta["lineage"]["upstream"]["manifest.json"]
    manifest_from_lineage = tmp_root / recorded_manifest["path"]
    assert manifest_from_lineage.is_file()
    assert manifest_from_lineage.resolve() == manifest_path.resolve()
    assert recorded_manifest["sha256"] == _sha256(manifest_path), \
        "角点产物记录的清单指纹与实际文件不符,血缘链断裂"

    # 第二环:每个标定组合记录的角点文件路径与指纹 = 本轮实际角点文件;
    # scipy 组合另记初值产物指纹(同轮 <模型>__opencv/result.json)
    calibration_run = smoke_tree["calibration_run"]
    corners_npz = corners_run / "corners_train_sb.npz"
    for name in MATRIX_COMBOS:
        doc = json.loads(
            (calibration_run / name / "result.json").read_text(encoding="utf-8"))
        lineage = doc["lineage"]
        recorded = lineage["upstream"]["corners_train_sb.npz"]
        corners_file = tmp_root / recorded["path"]
        assert corners_file.resolve() == corners_npz.resolve(), \
            f"{name} 引用的角点文件不在本轮 corners 产物目录:{corners_file}"
        assert recorded["sha256"] == _sha256(corners_npz), \
            f"{name} 记录的角点指纹与实际文件不符,血缘链断裂"
        # 剔除名单留痕(T-03-02):键必须在,值为本轮实际剔除的照片编号
        assert "excluded_images" in lineage
        if name.endswith(("__scipy", "__scipy_softl1")):
            recorded_init = lineage["upstream"]["init_result.json"]
            init_file = tmp_root / recorded_init["path"]
            assert init_file.is_file(), f"{name} 的初值产物不在:{init_file}"
            assert init_file.parent.name == f"{doc['model_id']}__opencv", \
                f"{name} 的初值产物不是同模型的 opencv 结果:{init_file.parent.name}"
            assert recorded_init["sha256"] == _sha256(init_file), \
                f"{name} 记录的初值指纹与实际文件不符,血缘链断裂"


def test_matrix_structure_complete_even_if_not_converged(smoke_tree: dict):
    """12 个组合的产物结构一视同仁:键一个不少;失败组合数值字段为 null
    但 failure_note 非空,收敛组合必须有参数字典与位姿/残差两份 npz。"""
    calibration_run = smoke_tree["calibration_run"]
    for name in MATRIX_COMBOS:
        doc = json.loads(
            (calibration_run / name / "result.json").read_text(encoding="utf-8"))
        for key in RESULT_JSON_KEYS:
            assert key in doc, f"{name} 缺结构键:{key}"
        assert doc["n_points_per_view"] == 88
        assert doc["n_views"] >= 2
        if doc["converged"]:
            assert doc["rms_overall"] is not None and doc["failure_note"] is None
            assert doc["params"], f"{name} 收敛却没给任何参数"
            with np.load(calibration_run / name / "extrinsics.npz",
                         allow_pickle=False) as data:
                assert data["rvecs"].shape == (doc["n_views"], 3)
            with np.load(calibration_run / name / "residuals.npz",
                         allow_pickle=False) as data:
                assert data["residuals"].shape == (doc["n_views"], 88, 2)
                assert data["image_ids"].shape == (doc["n_views"],)
        else:
            note = doc["failure_note"]
            assert isinstance(note, str) and note.strip(), \
                f"{name} 未收敛却没写中文失败说明"
            assert doc["rms_overall"] is None and doc["per_view_rms"] is None
            assert doc["params"] == {}
