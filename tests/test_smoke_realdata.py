"""真实数据端到端冒烟测试:干净临时目录上五命令独立进程接力贯通。

整文件标记 integration(需要真实 input_data 数据,日常开发用
`uv run pytest -m "not integration"` 自动跳过)。

测的是什么:临时目录就是一份"缩小版新数据集"——config 从仓库复制,
input_data/train 软链接真实前 3 个姿态目录(24 张)、test 软链接前 5 张,
然后像真实用户一样依次运行 calib ingest → calib corners → calib calibrate
→ calib noise → calib fisher(每条命令独立进程,fisher 依赖 calibrate
与 noise 两家产物所以排最后)。断言三件事:
1. 五命令退出码全 0,产物齐全(manifest / 4 份角点 npz / 矩阵组合目录 /
   noise.json / crlb.json);
2. 血缘链闭合:标定产物记录的角点文件指纹与实际文件一致,角点产物记录的
   数据清单指纹与实际文件一致,noise 与 crlb 的上游指纹逐环核对——链上
   任何一环被篡改都会在这里显形(T-03-01 / T-03-06-01);
3. 未收敛组合(小样本上模型失配很常见)与 CRLB 失败组合的产物结构也必须
   完整(键一个不少,数值字段 null、中文失败说明非空)。

成本取舍(在 docstring 写明,照 03-03/03-05 集成测试复用规则的体例):
本测试不复用仓库 output/ 下的历史 noise/fisher 轮——临时目录自成一套
产物树,旧轮血缘指纹对不上,拿旧轮充数等于伪造证据;好在缩小版数据集的
注入实验只有 24 张 × 9 档 σ × 4 次重复 ≈ 864 个副本(8 进程并行,分钟级),
"全量注入两小时级"的成本问题只存在于仓库根的全量数据上,那里由
test_noise_realdata.py / test_fisher_realdata.py 的复用规则负责。
真相源仍是落盘产物:断言一律从临时树里的 noise.json / crlb.json 读回。

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

from calib.noise.artifacts import CRLB_KEYS, REPORT_KEYS

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

# 五步各自的超时上限(秒):ingest/corners 沿用原 900;calibrate 的 12 组合
# 矩阵、noise 的注入实验(缩小版数据集分钟级)、fisher 的逐模型交叉验证
# 重解各自给足余量,谁超时谁就是真断裂
STEP_TIMEOUTS_S = {
    "ingest": 900,
    "corners": 900,
    "calibrate": 1800,
    "noise": 3600,
    "fisher": 3600,
}


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
    """搭建缩小版数据集,跑通五命令,返回各产物路径与运行结果。

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

    # 五命令独立进程接力(每条都是全新进程,只靠落盘文件传递状态;
    # fisher 吃 calibrate 与 noise 两家产物,排在最后)
    outputs: dict[str, subprocess.CompletedProcess] = {}
    for step in ("ingest", "corners", "calibrate", "noise", "fisher"):
        result = subprocess.run(
            ["uv", "run", "--project", str(REPO_ROOT), "calib", step],
            check=False,
            cwd=tmp_root, capture_output=True, text=True,
            timeout=STEP_TIMEOUTS_S[step],
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
    noise_run = _latest_run(out / "noise")
    fisher_run = _latest_run(out / "fisher")
    return {
        "tmp_root": tmp_root,
        "outputs": outputs,
        "ingest_run": ingest_run,
        "corners_run": corners_run,
        "calibration_run": calibration_run,
        "noise_run": noise_run,
        "fisher_run": fisher_run,
    }


def test_five_commands_relay_and_artifacts_complete(smoke_tree: dict):
    """五命令退出码全 0,各阶段产物齐全且规模正确(3 姿态 × 8 + 5 张)。"""
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

    # noise:noise.json 本轮非失败轮,引用的传递曲线 json 真实在
    # (临时树从零建起,不存在旧轮产物冒充本轮的问题,T-03-06-01)
    noise_run = smoke_tree["noise_run"]
    noise_doc = json.loads(
        (noise_run / "noise.json").read_text(encoding="utf-8"))
    assert noise_doc["failure_note"] is None, \
        f"noise 本轮是失败轮:{noise_doc['failure_note']}"
    assert (noise_run / noise_doc["transfer_curve"]["path"]).is_file(), \
        "noise.json 引用的传递曲线 json 不在"

    # fisher:crlb.json 本轮新产物
    assert (smoke_tree["fisher_run"] / "crlb.json").is_file(), \
        "缺 CRLB 汇总产物 crlb.json"


def _assert_upstream_closed(doc: dict, tmp_root: Path, stage: str) -> None:
    """核对一份产物的全部上游登记:文件真实存在 + 指纹与实际文件一致。

    登记的路径是"相对当时工作目录"的,临时树就是当时的工作目录;
    每个上游条目都逐项核对,链上任何一环被篡改都会在这里显形。
    """
    upstream = doc.get("lineage", {}).get("upstream")
    assert upstream, f"{stage} 没带上游血缘"
    for name, entry in upstream.items():
        f = tmp_root / entry["path"]
        assert f.is_file(), f"{stage} 血缘登记的上游 {name} 不在:{f}"
        assert entry["sha256"] == _sha256(f), \
            f"{stage} 记录的 {name} 指纹与实际文件不符,血缘链断裂"


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

    # 第三环:noise.json 的上游 = 角点产物 + 同轮数据清单(test 进噪声
    # 划分时另登记 test 角点),逐项指纹核对
    noise_doc = json.loads(
        (smoke_tree["noise_run"] / "noise.json").read_text(encoding="utf-8"))
    _assert_upstream_closed(noise_doc, tmp_root, "noise.json")
    noise_upstream = noise_doc["lineage"]["upstream"]
    assert {"corners_train_sb.npz", "manifest.json"} <= set(noise_upstream), \
        f"noise.json 血缘缺角点/清单上游:{tuple(noise_upstream)}"
    assert "corners_test_sb.npz" in noise_upstream, \
        "仓库配置 noise_splits 含 test,血缘必须登记 test 角点文件"
    noise_corners_file = tmp_root / noise_upstream["corners_train_sb.npz"]["path"]
    assert noise_corners_file.resolve() == corners_npz.resolve(), \
        "noise 引用的角点文件不在本轮 corners 产物目录"
    noise_json_path = smoke_tree["noise_run"] / "noise.json"

    # 第四环:crlb.json 的上游 = 角点 + 清单 + noise.json + 逐模型标定产物
    # (result.json 失败组合也写盘所以必在;extrinsics 只在收敛组合有,
    # 血缘按"文件在不在"登记,收敛性逐模型对账)
    crlb_doc = json.loads(
        (smoke_tree["fisher_run"] / "crlb.json").read_text(encoding="utf-8"))
    _assert_upstream_closed(crlb_doc, tmp_root, "crlb.json")
    fisher_upstream = crlb_doc["lineage"]["upstream"]
    assert {"corners_train_sb.npz", "manifest.json"} <= set(fisher_upstream), \
        f"crlb.json 血缘缺角点/清单上游:{tuple(fisher_upstream)}"
    assert "noise.json" in fisher_upstream, \
        f"crlb.json 血缘缺 noise.json 上游:{tuple(fisher_upstream)}"
    noise_file_from_crlb = tmp_root / fisher_upstream["noise.json"]["path"]
    assert noise_file_from_crlb.resolve() == noise_json_path.resolve(), \
        "crlb 引用的 noise.json 不在本轮 noise 产物目录"
    for i in "1234":
        result_key = f"m{i}__scipy/result.json"
        assert result_key in fisher_upstream, \
            f"crlb.json 血缘缺 {result_key}:{tuple(fisher_upstream)}"
        model_doc = json.loads(
            (calibration_run / f"m{i}__scipy" / "result.json")
            .read_text(encoding="utf-8"))
        if model_doc["converged"]:
            assert f"m{i}__scipy/extrinsics.npz" in fisher_upstream, \
                f"m{i} 已收敛,血缘却没登记它的位姿产物"


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


def test_noise_and_fisher_structure_contract(smoke_tree: dict):
    """noise.json 与 crlb.json 的结构契约一视同仁:契约键一个不少。

    noise:失败时数值字段为 null、中文 failure_note 非空(失败也落盘的
    契约,T-03-03-04);成功时双口径 σ 必须是有限正值。
    crlb:models 下 m1–m4 四键全,16 份记录逐份可寻址;ok 记录数值齐,
    失败记录中文说明点名、数值为 null(单组合失败不拖死整批,T-03-05-04)。
    """
    noise_doc = json.loads(
        (smoke_tree["noise_run"] / "noise.json").read_text(encoding="utf-8"))
    for key in REPORT_KEYS:
        assert key in noise_doc, f"noise.json 缺结构键:{key}"
    if noise_doc["failure_note"] is None:
        for key in ("sigma_detector_corner", "sigma_temporal_corner"):
            value = noise_doc[key]
            assert value is not None and np.isfinite(value) and value > 0, \
                f"{key} 应为有限正值,实际 {value!r}"
    else:
        assert noise_doc["sigma_detector_corner"] is None \
            and noise_doc["sigma_temporal_corner"] is None, \
            "失败轮的双口径 σ 必须为 null"
        assert str(noise_doc["failure_note"]).strip(), \
            "失败轮必须写非空中文失败说明"

    crlb_doc = json.loads(
        (smoke_tree["fisher_run"] / "crlb.json").read_text(encoding="utf-8"))
    for key in CRLB_KEYS:
        assert key in crlb_doc, f"crlb.json 缺结构键:{key}"
    assert set(crlb_doc["models"]) == {"m1", "m2", "m3", "m4"}, \
        f"models 键应为 m1–m4,实际 {tuple(crlb_doc['models'])}"
    for model_id, block in crlb_doc["models"].items():
        for obs_name in ("views640", "poses80"):
            for sigma_name in ("detector", "temporal"):
                where = f"{model_id}×{obs_name}×{sigma_name}"
                record = block["observation_sets"][obs_name][sigma_name]
                for key in ("se", "band_low", "band_high", "condition_number",
                            "top_directions", "status", "failure_note"):
                    assert key in record, f"{where} 缺结构键:{key}"
                if record["status"] == "ok":
                    for key in ("se", "band_low", "band_high",
                                "condition_number"):
                        assert record[key] is not None, \
                            f"{where} 的 {key} 缺数值"
                    arr = np.asarray(record["se"], dtype=float)
                    assert not np.isnan(arr).any(), \
                        f"{where} 的 se 出现 NaN(数据缺口要显式记 None 或 inf)"
                else:
                    note = record["failure_note"]
                    assert isinstance(note, str) and note.strip(), \
                        f"{where} 未算成却没写中文失败说明"
                    assert record["se"] is None and \
                        record["condition_number"] is None, \
                        f"{where} 失败记录的数值字段必须为 null"
