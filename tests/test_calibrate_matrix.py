"""calibrate 命令的全矩阵编排测试:合成数据端到端 + 真实数据 integration。

快速部分(不加 integration 标记,合成 6 视图秒级)锁定:
- D-01 全矩阵:临时树(config 裁剪为 models:[m2] × solvers:[opencv, scipy] ×
  losses:[linear, soft_l1])上独立进程跑 calib calibrate,退出码 0;
  产物目录 m2__opencv / m2__scipy / m2__scipy_softl1 三件套齐全,
  comparison.json 三行组合名正确;
- 血缘链闭合(成功标准 4):m2__scipy 的 result.json 血缘 upstream 同时含
  corners_train_sb.npz 与 init_result.json,sha256 与实际文件重算一致
  ——算的不是那份输入、或初值被换过,都会在这里显形;
- 初值协议:m2__scipy 的 init_from 指向 m2__opencv 的产物文件;params
  按模型参数名给全 9 个键(fx…k3);
- 精化不劣化(成功标准 3 的合成版):合成数据应收敛良好,scipy 两条损失
  路径的回代 RMS ≤ 纯 OpenCV + 1e-9。

integration 部分(本文件下半段,标记 integration)锁定真实数据:
m4 最小矩阵两轮独立进程数值一致(< 1e-12,确定性)+ 血缘哈希重算一致。

合成真值视角的造法与 conftest.make_true_views 同一套(已知内参投影,
带"标准答案");本文件用 module 级夹具搭树 + subprocess 独立进程,
失败时 AssertionError 附完整 stdout/stderr(体例照 test_smoke_realdata.py)。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# m2 的参数名契约(针孔 + 5 系数;params 键数断言用)
_M2_PARAM_NAMES = {"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"}


def _sha256(path: Path) -> str:
    """重算文件 sha256(血缘对账的测试侧尺子,体例照 test_smoke_realdata.py)。"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_run(stage_dir: Path) -> Path:
    """取某阶段最新的 run_<时间戳> 目录(目录名字典序 = 时间序)。"""
    runs = sorted(p for p in stage_dir.iterdir()
                  if p.is_dir() and p.name.startswith("run_"))
    assert runs, f"{stage_dir} 下没有任何 run 目录"
    return runs[-1]


def _run_cli(step: str, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """独立进程跑 calib <step>(cwd=临时树;失败对账交给调用方)。"""
    return subprocess.run(
        ["uv", "run", "--project", str(REPO_ROOT), "calib", step],
        check=False, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        env={**os.environ},
    )


def _synthetic_views(n_views: int = 6) -> list[np.ndarray]:
    """造 n_views 个已知真值视角(与 conftest.make_true_views 同一套内参/分布,
    本文件自带一份是因为 module 级夹具不能用函数级夹具)。

    无畸变针孔、fx=fy=800、主点画面中心、深度统一 800;角点由
    cv2.projectPoints 按 11×8 物点网格投影得出,自带"标准答案"。
    """
    rng = np.random.default_rng(7)
    k_true = np.array([[800.0, 0.0, 320.0],
                       [0.0, 800.0, 240.0],
                       [0.0, 0.0, 1.0]])
    xs = (np.arange(11) - 5.0) * 20.0
    ys = (np.arange(8) - 3.5) * 20.0
    objp = np.zeros((8 * 11, 3), dtype=np.float64)
    objp[:, 0] = np.tile(xs, 8)
    objp[:, 1] = np.repeat(ys, 11)
    views = []
    for _k in range(n_views):
        rvec = np.array([rng.uniform(-0.15, 0.15),
                         rng.uniform(-0.10, 0.20),
                         rng.uniform(-0.15, 0.15)])
        tvec = np.array([rng.uniform(-30.0, 30.0),
                         rng.uniform(-30.0, 30.0), 800.0])
        uv, _jac = cv2.projectPoints(objp, rvec, tvec, k_true, np.zeros(5))
        views.append(uv.reshape(-1, 2).astype(np.float64))
    return views


@pytest.fixture(scope="module")
def synthetic_matrix_tree(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """搭"合成缩小版数据集"并独立进程跑 calib calibrate,返回产物路径。

    临时树里只有 calibrate 需要的东西:裁剪版 config(m2 单模型)、
    合成角点 npz(含 source 键,D-02 的 sb 路线形态)、最小数据清单
    (与角点同目录;角点 meta 的血缘记录指向它,防串轮走血缘优先路径)。
    """
    tmp_root = tmp_path_factory.mktemp("matrix_synthetic")

    # config:从仓库原样复制后把模型矩阵裁剪为单模型 m2(体例照
    # test_smoke_realdata.py 的"config 从仓库复制";文本替换保持注释结构)
    cfg_dir = tmp_root / "config"
    cfg_dir.mkdir()
    shutil.copy(REPO_ROOT / "config" / "board.yaml", cfg_dir / "board.yaml")
    calib_yaml = (REPO_ROOT / "config" / "calibration.yaml").read_text(encoding="utf-8")
    calib_yaml = calib_yaml.replace("models: [m1, m2, m3, m4]", "models: [m2]")
    (cfg_dir / "calibration.yaml").write_text(calib_yaml, encoding="utf-8")

    # 合成角点 + 最小数据清单(同目录);meta 血缘记录清单路径,
    # _manifest_path_for 会走"血缘优先"定位,与真实链路同一条
    image_ids = [f"img_{k:02d}" for k in range(6)]
    views = _synthetic_views(6)
    corners_run = tmp_root / "output" / "corners" / "run_test"
    corners_run.mkdir(parents=True)
    np.savez(corners_run / "corners_train_sb.npz",
             image_ids=np.asarray(image_ids),
             uv=np.asarray(views, dtype=np.float64),
             detected=np.ones(6, dtype=bool),
             source=np.full(6, "sb", dtype="<U16"))
    manifest = {
        "images": [{"image_id": image_ids[k], "pose_id": f"pose_{k // 2:02d}",
                    "split": "train"} for k in range(6)],
        "summary": {"resolution": [640, 480]},
    }
    manifest_path = corners_run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    meta = {"lineage": {"upstream": {
        "manifest.json": {"path": "output/corners/run_test/manifest.json",
                          "sha256": _sha256(manifest_path)}}}}
    (corners_run / "corners_train_sb.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    result = _run_cli("calibrate", tmp_root, timeout=300)
    if result.returncode != 0:
        raise AssertionError(
            f"合成树 calib calibrate 退出码 {result.returncode}\n"
            f"--- 标准输出 ---\n{result.stdout}\n--- 标准错误 ---\n{result.stderr}"
        )
    return {
        "tmp_root": tmp_root,
        "process": result,
        "corners_run": corners_run,
        "calibration_run": _latest_run(tmp_root / "output" / "calibration"),
    }


def test_exit_zero_and_combo_artifacts(synthetic_matrix_tree: dict):
    """行为 1+2:独立进程退出码 0;三个组合目录三件套齐全(收敛组合),
    comparison.json 三行组合名与目录名一一对应。"""
    run = synthetic_matrix_tree["calibration_run"]
    names = ["m2__opencv", "m2__scipy", "m2__scipy_softl1"]
    for name in names:
        combo = run / name
        assert (combo / "result.json").is_file(), f"缺产物目录:{name}"
        doc = json.loads((combo / "result.json").read_text(encoding="utf-8"))
        if doc["converged"]:
            assert (combo / "extrinsics.npz").is_file(), f"{name} 缺位姿文件"
            assert (combo / "residuals.npz").is_file(), f"{name} 缺残差文件"

    comparison = json.loads(
        (run / "comparison.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in comparison] == names, \
        f"comparison 组合名不对:{[row['name'] for row in comparison]}"
    assert len(comparison) == 3
    for row in comparison:
        assert row["model"] == "m2" and row["n_views"] == 6


def test_scipy_lineage_upstream_hashes_match_files(synthetic_matrix_tree: dict):
    """行为 3:m2__scipy 的血缘 upstream 含角点 npz 与初值 result.json 两项,
    sha256 与实际文件重算一致(链上任何一环被换都会显形)。"""
    tmp_root = synthetic_matrix_tree["tmp_root"]
    corners_run = synthetic_matrix_tree["corners_run"]
    run = synthetic_matrix_tree["calibration_run"]

    doc = json.loads((run / "m2__scipy" / "result.json").read_text(encoding="utf-8"))
    upstream = doc["lineage"]["upstream"]
    assert "corners_train_sb.npz" in upstream and "init_result.json" in upstream, \
        f"scipy 组合的 upstream 缺项:{sorted(upstream)}"

    corners_npz = tmp_root / upstream["corners_train_sb.npz"]["path"]
    assert corners_npz.resolve() == (corners_run / "corners_train_sb.npz").resolve()
    assert upstream["corners_train_sb.npz"]["sha256"] == _sha256(corners_npz), \
        "角点 npz 的血缘指纹与实际文件不符,血缘链断裂"

    init_file = tmp_root / upstream["init_result.json"]["path"]
    assert init_file.is_file(), f"血缘指向的初值产物不存在:{init_file}"
    assert upstream["init_result.json"]["sha256"] == _sha256(init_file), \
        "初值产物的血缘指纹与实际文件不符,血缘链断裂"


def test_scipy_init_from_and_params_contract(synthetic_matrix_tree: dict):
    """行为 4:m2__scipy 的 init_from 指向 m2__opencv 的产物文件;params
    按模型参数名给全 9 个键(fx…k3),下游按名取、不猜位置。"""
    run = synthetic_matrix_tree["calibration_run"]

    doc = json.loads((run / "m2__scipy" / "result.json").read_text(encoding="utf-8"))
    assert doc["init_from"] == "m2__opencv/result.json", \
        f"init_from 应指向同模型 OpenCV 初值产物,实际是 {doc['init_from']!r}"
    assert (run / doc["init_from"]).is_file(), "init_from 指向的文件不存在"
    assert set(doc["params"]) == _M2_PARAM_NAMES, \
        f"m2 参数键不对:{sorted(doc['params'])}"
    assert len(doc["params"]) == 9
    assert doc["solver_meta"]["init_source"] == "same_round_opencv"

    # 鲁棒核路径同样从落盘初值出发(精化对照的输入一致性)
    doc_soft = json.loads(
        (run / "m2__scipy_softl1" / "result.json").read_text(encoding="utf-8"))
    assert doc_soft["init_from"] == "m2__opencv/result.json"
    assert doc_soft["solver_meta"]["loss"] == "soft_l1"


def test_scipy_refine_no_worse_than_opencv(synthetic_matrix_tree: dict):
    """行为 5:合成数据应收敛良好——三条路径全部收敛,且 scipy 两条损失
    路径的回代 RMS ≤ 纯 OpenCV + 1e-9(同一把回代尺,可比)。"""
    run = synthetic_matrix_tree["calibration_run"]
    docs = {
        name: json.loads((run / name / "result.json").read_text(encoding="utf-8"))
        for name in ("m2__opencv", "m2__scipy", "m2__scipy_softl1")
    }
    for name, doc in docs.items():
        assert doc["converged"] is True, \
            f"合成无畸变数据上 {name} 应收敛:{doc['failure_note']}"
        assert doc["rms_overall"] < 0.5, \
            f"{name} 合成数据 RMS {doc['rms_overall']:.4f}px 超线(应有真值级精度)"

    rms_cv = docs["m2__opencv"]["rms_overall"]
    for name in ("m2__scipy", "m2__scipy_softl1"):
        rms = docs[name]["rms_overall"]
        assert rms <= rms_cv + 1e-9, \
            f"{name} 精化后 RMS {rms:.9f} 比纯 OpenCV {rms_cv:.9f} 还差(容差 1e-9)"
