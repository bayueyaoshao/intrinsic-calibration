"""受控注入实验与传递曲线的测试(小样单测 + 真实全量集成)。

小样(合成 2 张棋盘图 × 2 档 σ × 1 次重复)证明数据链本身是对的:
位移数组形状、同种子逐位复现、σ=0 档恒零、检测失败留痕不炸整体;
真实 640 张全量跑标注 integration(慢,默认跳过),由 calib noise 闭环执行。

加噪-检测-位移这条链的全部参数只从配置读(D-12):小样配置直接构造
NoiseConfig(绕过 yaml 解析的范围校验,只用 1 档 × 1 重复把测试压到秒级)。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from calib.corners.artifacts import save_corners
from calib.corners.detect import detect_sb
from calib.io.config import (
    CalibrationConfig,
    CornersConfig,
    NoiseConfig,
    load_board,
    load_calibration,
)
from calib.io.dataset import ImageRecord, imread_checked
from calib.noise.injection import (
    _anchor_grid,
    _seed_for,
    add_gaussian_noise,
    run_injection_experiment,
)
from calib.noise.transfer import (
    build_transfer_curve,
    fit_interp,
    interp_corner_sigma,
    save_transfer_curve,
)
from calib.util.lineage import make_lineage

COLS, ROWS = 11, 8  # 与真实棋盘同规格(88 个内角点)


def _write_chessboard(make_chessboard_image, path: Path, pose_shift: float) -> None:
    """写一张合成棋盘图(轻微改位姿让两张图不完全一样)。"""
    rvec = np.array([0.0, 0.05 + pose_shift, 0.0])
    img, _corners, _sq = make_chessboard_image(rvec=rvec)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"测试写图失败:{path}")


def _build_baseline_npz(out_dir: Path, records,
                        corners_override: dict[str, np.ndarray] | None = None) -> Path:
    """像真实 corners 阶段一样:对写好的图逐张跑 detect_sb,存成基线角点 npz。

    基线必须与真实产物同源(检测器输出,不是投影真值)——真值与检出本就差
    亚像素级的一点,σ=0 校准行的"位移恒 0"断言抓的正是基线错位。
    corners_override:个别图塞占位角点(如全黑图:检测必失败,但基线要标
    "已检出"才能走进"副本检测失败"的留痕路径,而不是"基线缺失"路径)。
    """
    ids, uv_rows, sq_rows, win_rows, det_rows, src_rows = [], [], [], [], [], []
    for r in records:
        gray = imread_checked(r.path)
        result = detect_sb(gray, COLS, ROWS, CornersConfig())
        ids.append(r.image_id)
        override = (corners_override or {}).get(r.image_id)
        if result is not None:
            uv_rows.append(result["uv"])
            sq_rows.append(result["sq_px"])
            win_rows.append(result["win_size"])
            det_rows.append(True)
            src_rows.append("sb")
        elif override is not None:
            uv_rows.append(override)
            sq_rows.append(20.0)
            win_rows.append(5.0)
            det_rows.append(True)
            src_rows.append("sb")
        else:
            uv_rows.append(np.full((COLS * ROWS, 2), np.nan))
            sq_rows.append(np.nan)
            win_rows.append(np.nan)
            det_rows.append(False)
            src_rows.append("")
    save_corners(out_dir, "train", "sb", ids, np.asarray(uv_rows),
                 np.asarray(sq_rows), np.asarray(win_rows),
                 np.asarray(det_rows, dtype=bool),
                 source=np.asarray(src_rows, dtype="<U16"),
                 lineage=make_lineage({}, {}, "run_test"))
    return out_dir / "corners_train_sb.npz"


def _small_noise_cfg() -> NoiseConfig:
    """小样配置:1 档 σ=4 × 1 次重复、串行(不开子进程)、网格写死保确定性。"""
    return NoiseConfig(injection_n_levels=1, injection_sigma_min=4.0,
                       injection_sigma_max=4.0, injection_n_repeats=1,
                       injection_max_workers=1)


@pytest.fixture
def small_experiment(make_chessboard_image, tmp_path):
    """小样注入地基:2 张合成棋盘图 + 基线角点 npz + 小样 NoiseConfig。"""
    img_dir = tmp_path / "input_data" / "train" / "pose_a"
    records = []
    for k in range(2):
        image_id = f"20260101_00000{k}"
        path = img_dir / f"{image_id}.jpg"
        _write_chessboard(make_chessboard_image, path, 0.01 * k)
        records.append(ImageRecord(image_id=image_id, path=path, split="train",
                                   pose_id="pose_a", repeat_idx=k, n_repeats=2))
    npz = _build_baseline_npz(tmp_path / "corners_run", records)
    cfg = CalibrationConfig(noise=_small_noise_cfg())
    return records, npz, cfg


# ---- 行为 1:小样注入——形状 / 同种子逐位复现 / σ=0 恒零 ----

def test_small_sample_shape_seed_repro_and_zero_level(small_experiment, tmp_path):
    """位移数组形状 (张, 档, 重复, 角点, 轴);同种子重跑逐位一致;σ=0 档恰为 0。"""
    records, npz, cfg = small_experiment
    out1 = tmp_path / "run1"
    result = run_injection_experiment(npz, records, cfg, COLS, ROWS, out1)

    with np.load(result["displacements_path"], allow_pickle=False) as data:
        disp = data["displacements"]
        source = data["source"]
    assert disp.shape == (2, 2, 1, COLS * ROWS, 2)
    # σ=0 参照档:同一检测器重跑干净原图,位移应恰为 0(基线对齐校验,Pitfall 4)
    assert np.all(disp[:, 0] == 0.0), "σ=0 参照档位移非零:基线没对齐或检测不确定"
    assert np.all(source[:, 0] == "sb")
    # σ=4 档:检测全部成功且位移确实非零(检测器对噪声有响应)
    assert np.isfinite(disp[:, 1]).all()
    assert np.abs(disp[:, 1]).max() > 0.0

    # 同种子完整重跑第二次:位移数组逐位一致(不落盘合成图像也能精确复现)
    out2 = tmp_path / "run2"
    result2 = run_injection_experiment(npz, records, cfg, COLS, ROWS, out2)
    with np.load(result2["displacements_path"], allow_pickle=False) as data2:
        disp2 = data2["displacements"]
    np.testing.assert_array_equal(disp, disp2)
    assert result2["meta"]["grid"] == result["grid"]


def test_meta_documents_no_disk_tradeoff(small_experiment, tmp_path):
    """meta 里写明"合成图像不落盘、靠种子三元组精确复现"的取舍说明。"""
    records, npz, cfg = small_experiment
    result = run_injection_experiment(npz, records, cfg, COLS, ROWS,
                                      tmp_path / "run_meta")
    note = result["meta"]["note"]
    assert "不落盘" in note and "种子" in note
    assert result["meta"]["base_seed"] == cfg.noise.injection_seed
    assert result["meta"]["n_repeats"] == 1
    assert result["meta"]["grid"] == [0.0, 4.0]


# ---- 种子三元组与网格锚定(注入可复现的两块基石)----

def test_seed_triple_derivation():
    """同 (图, 档, 重复) 同种子;任一元不同则种子不同;种子落在 32 位范围。"""
    s1 = _seed_for("img_a", 4.0, 0, 20261018)
    assert s1 == _seed_for("img_a", 4.0, 0, 20261018)
    assert _seed_for("img_a", 4.0, 1, 20261018) != s1
    assert _seed_for("img_b", 4.0, 0, 20261018) != s1
    assert _seed_for("img_a", 8.0, 0, 20261018) != s1
    assert 0 <= s1 < 2**32


def test_anchor_grid_explicit_override():
    """配置写死网格上下限:覆盖自动锚定,σ=0 参照档总在最前。"""
    cfg = NoiseConfig(injection_n_levels=2, injection_sigma_min=2.0,
                      injection_sigma_max=8.0)
    grid = _anchor_grid([3.0, 5.0], cfg)
    assert grid == [0.0, 2.0, 8.0]


def test_anchor_grid_auto_from_median():
    """配置留空:按实测中位数 σ 自动锚定(下限 max(0.25, σ/4)、上限 min(128, σ×32))。"""
    cfg = NoiseConfig(injection_n_levels=3)
    grid = _anchor_grid([4.0, 4.0, 4.0], cfg)
    assert grid[0] == 0.0
    assert grid[1] == pytest.approx(1.0)          # max(0.25, 4/4)
    assert grid[-1] == pytest.approx(128.0)       # min(128, 4×32)
    assert len(grid) == 4                         # 3 档 + σ=0 参照档


# ---- 行为 2:加噪函数的契约 ----

def test_add_gaussian_noise_contract():
    """输出 uint8、值域 [0,255]、形状不变;同 rng 状态两次逐位一致;clip 生效。"""
    img = np.full((60, 80), 128, dtype=np.uint8)

    out = add_gaussian_noise(img, 10.0, np.random.default_rng(7))
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert int(out.min()) >= 0 and int(out.max()) <= 255

    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    np.testing.assert_array_equal(
        add_gaussian_noise(img, 10.0, rng1), add_gaussian_noise(img, 10.0, rng2))

    big = add_gaussian_noise(img, 1000.0, np.random.default_rng(7))
    assert int(big.min()) == 0 and int(big.max()) == 255  # 大 σ 下 clip 真的截住了


# ---- 行为 3:检测必失败的场景——NaN 留痕、计数、不抛异常 ----

def test_failed_detection_recorded_not_raised(make_chessboard_image, tmp_path):
    """全黑图(检测必失败)混进小样:该副本位移 NaN、失败计数 +1、
    source 记 "",整体不抛异常(部分成功语义,研究 Pitfall 7)。"""
    img_dir = tmp_path / "input_data" / "train" / "pose_a"
    records = []
    for k in range(2):
        image_id = f"20260101_00000{k}"
        path = img_dir / f"{image_id}.jpg"
        _write_chessboard(make_chessboard_image, path, 0.01 * k)
        records.append(ImageRecord(image_id=image_id, path=path, split="train",
                                   pose_id="pose_a", repeat_idx=k, n_repeats=3))
    # 第三张:全黑图(检测必失败);基线塞占位角点并标"已检出",
    # 让它走进"副本检测失败"的留痕路径(行为 3 的本体)
    black_path = img_dir / "20260101_00002.jpg"
    black_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(black_path), np.zeros((480, 640), dtype=np.uint8))
    records.append(ImageRecord(image_id="20260101_00002", path=black_path,
                               split="train", pose_id="pose_a", repeat_idx=2, n_repeats=3))
    dummy = np.stack([np.arange(COLS * ROWS, dtype=np.float64) / 10.0,
                      np.arange(COLS * ROWS, dtype=np.float64) / 20.0], axis=1)
    npz = _build_baseline_npz(tmp_path / "corners_run", records,
                              corners_override={"20260101_00002": dummy})
    cfg = CalibrationConfig(noise=_small_noise_cfg())

    result = run_injection_experiment(npz, records, cfg, COLS, ROWS,
                                      tmp_path / "run_fail")  # 不抛异常即通过
    with np.load(result["displacements_path"], allow_pickle=False) as data:
        disp = data["displacements"]
        source = data["source"]
        detected = data["detected"]
    assert np.isnan(disp[2]).all()          # 全黑图全部副本位移记 NaN
    assert np.all(source[2] == "")          # 来源留痕:两法都没检出
    assert not detected[2].any()            # 检出掩码全 False

    per_image = json.loads(
        Path(result["per_image_path"]).read_text(encoding="utf-8"))
    failed = next(r for r in per_image["images"] if r["image_id"] == "20260101_00002")
    # 小样网格 = [σ=0 参照档, σ=4],共 2 档 × 1 重复 = 2 个副本,全部失败
    assert failed["n_fail"] == 2
    ok = next(r for r in per_image["images"] if r["image_id"] == "20260101_000000")
    assert ok["n_fail"] == 0                # 正常图不受失败副本牵连


# ---- 行为 4:传递曲线五键齐、σ=0 校准行、PCHIP 保形插值 ----

def _synth_curve():
    """构造已知答案的位移数据:每档 rms 恰等于该档 σ(便于保形断言)。"""
    grid = [0.0, 1.0, 4.0, 16.0]
    n, _L, r, p = 4, 4, 2, 88
    disp = np.zeros((n, 4, r, p, 2))
    for li, sigma in enumerate(grid[1:], start=1):
        disp[:, li, :, :, 0] = sigma       # x 方向位移 = σ → rms 恰为 σ
    detected = np.ones((n, 4, r), dtype=bool)
    source = np.full((n, 4, r), "sb", dtype="<U16")
    source[:, 3, :] = "classic_fallback"   # 最高档全部兜底
    detected[0, 3, 0] = False              # 最高档第 1 个副本检测失败
    return grid, disp, detected, source


def test_transfer_curve_keys_and_zero_reference():
    """每档五键齐;σ=0 档 rms 恰 0 且标 reference;失败率/兜底率分档统计正确。"""
    grid, disp, detected, source = _synth_curve()
    curve = build_transfer_curve(disp, grid, detected, source)
    assert curve["grid"] == grid
    for entry in curve["levels"]:
        for key in ("rms_euclidean", "scatter_low", "scatter_high",
                    "failure_rate", "fallback_rate"):
            assert key in entry, f"传递曲线第 {entry['sigma']} 档缺键 {key}"
    zero = curve["levels"][0]
    assert zero["role"] == "reference" and zero["rms_euclidean"] == 0.0
    assert curve["levels"][1]["rms_euclidean"] == pytest.approx(1.0)
    assert curve["levels"][2]["rms_euclidean"] == pytest.approx(4.0)
    top = curve["levels"][3]
    assert top["rms_euclidean"] == pytest.approx(16.0)  # 失败副本不计入 rms
    assert top["fallback_rate"] == pytest.approx(1.0)
    assert top["failure_rate"] == pytest.approx(1 / 8)  # 8 个副本里失败 1 个
    assert curve["levels"][0]["fallback_rate"] == 0.0


def test_interp_shape_preserving_and_sqrt2_conversion():
    """PCHIP 保形:网格两档之间的插值不越过相邻档曲线值;√2 换算逐位可对账。"""
    grid, disp, detected, source = _synth_curve()
    curve = build_transfer_curve(disp, grid, detected, source)
    interp = fit_interp(curve)

    mid = interp_corner_sigma(2.0, interp)      # 2.0 介于 1 与 4 档之间
    y1 = curve["levels"][1]["rms_euclidean"]
    y4 = curve["levels"][2]["rms_euclidean"]
    assert y1 < mid * np.sqrt(2.0) < y4, "插值越过相邻档曲线值:保形性被破坏"
    # 恰在网格点上:每轴 σ = 曲线值 / √2
    assert interp_corner_sigma(4.0, interp) == pytest.approx(4.0 / np.sqrt(2.0))


def test_save_transfer_curve_json(tmp_path):
    """transfer_curve.json 落盘:中文 meta 原样可读(ensure_ascii=False)。"""
    grid, disp, detected, source = _synth_curve()
    curve = build_transfer_curve(disp, grid, detected, source)
    path = save_transfer_curve(curve, tmp_path)
    assert path.name == "transfer_curve.json" and path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["grid"] == grid
    assert any("每轴" in v for v in str(loaded.get("meta", {})).split(";")) or \
        "sigma_corner_per_axis" in json.dumps(loaded, ensure_ascii=False)


# ---- 出图(visualization 层)----

def test_plot_transfer_curve_png(tmp_path):
    """误差棒图落 PNG:文件存在且非空。"""
    from calib.visualization.noise_figs import plot_transfer_curve

    grid, disp, detected, source = _synth_curve()
    curve = build_transfer_curve(disp, grid, detected, source)
    out = plot_transfer_curve(curve, tmp_path / "figures" / "transfer_curve.png")
    assert out.exists() and out.stat().st_size > 0


# ---- 真实全量集成(慢,默认跳过;`uv run pytest -m integration` 才跑)----

@pytest.mark.integration
def test_full_injection_real_data(tmp_path):
    """真实 640 张全量注入(D-08):端到端跑一遍全网格,产物结构齐全。"""
    repo = Path(__file__).resolve().parents[1]
    manifests = sorted((repo / "output" / "ingest").glob("run_*/manifest.json"))
    corners_npzs = sorted((repo / "output" / "corners").glob("run_*/corners_train_sb.npz"))
    if not manifests or not corners_npzs:
        # 前提数据缺失是"这里跑不了"(如运行产物不进 git 的独立检出),
        # 不是发现缺陷——跳过并说明,在主检出上显式运行才会真跑全量
        pytest.skip("真实数据或 corners 产物未就位:请先跑 calib ingest 与 calib corners")
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    board = load_board(repo / "config" / "board.yaml")
    cfg = load_calibration(repo / "config" / "calibration.yaml")

    result = run_injection_experiment(
        corners_npzs[-1], manifest["images"], cfg,
        board.cols, board.rows, tmp_path / "injection_full")

    n_train = sum(1 for img in manifest["images"] if img["split"] == "train")
    with np.load(result["displacements_path"], allow_pickle=False) as data:
        disp = data["displacements"]
        image_ids = [str(x) for x in np.asarray(data["image_ids"]).ravel()]
    expected_shape = (n_train, len(result["grid"]), cfg.noise.injection_n_repeats,
                      board.cols * board.rows, 2)
    assert disp.shape == expected_shape
    assert len(image_ids) == n_train
    assert len(json.loads(
        Path(result["per_image_path"]).read_text(encoding="utf-8"))["images"]) == n_train
    # σ=0 参照档:全量真实图上同样必须恰为 0(基线对齐在真实数据上的校验)
    assert np.all(disp[:, 0] == 0.0)
