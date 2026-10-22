"""COR-02 角点退化寻优库层测试(05-05 Task 1:退化原语/种子四元组/方法候选)。

全部用合成棋盘投影夹具(conftest 的 make_chessboard_image):图像自带真值
角点,退化是否恒等、候选窗口是否合规、检测是否找得准都能就地量出来。
另含一道源码防线测试:corner_opt.py 是调度与统计层,检测只能走
calib.corners.detect 两个入口——源码里绝不出现 cv2 角点函数字样。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import calib.analysis.corner_opt as corner_opt_module
from calib.analysis.corner_opt import (
    _seed_for,
    apply_degradation,
    build_method_candidates,
    detect_with,
    run_corner_opt,
)
from calib.corners.canonical import canonicalize
from calib.corners.detect import detect_classic, detect_sb
from calib.corners.window import estimate_sq_px
from calib.io.config import (
    AnalyzeConfig,
    CalibrationConfig,
    CornersConfig,
)
from calib.solve.observation import ObservationSet

# 检测器测试用配置:直接用数据类默认值(与仓库 calibration.yaml 一致)
_CFG = CornersConfig()


# ---------------------------------------------------------------------------
# 退化原语 apply_degradation
# ---------------------------------------------------------------------------


def test_apply_degradation_deterministic_same_seed():
    """同种子两次施加同档加噪,退化副本逐位一致(可复现承诺的根基)。"""
    rng0 = np.random.default_rng(7)
    img = rng0.integers(0, 256, size=(120, 160), dtype=np.uint8)
    out_a = apply_degradation(img, "noise", 8.0, np.random.default_rng(12345))
    out_b = apply_degradation(img, "noise", 8.0, np.random.default_rng(12345))
    assert np.array_equal(out_a, out_b)
    # 换种子必须变(否则"随机"是假的)
    out_c = apply_degradation(img, "noise", 8.0, np.random.default_rng(54321))
    assert not np.array_equal(out_a, out_c)


def test_apply_degradation_identity_levels_bitwise_equal_input():
    """恒等档(噪 0 / 糊 0 / 对比 1.0)输出逐位等于输入——校准档的零位移前提。"""
    img = np.random.default_rng(3).integers(0, 256, size=(96, 128), dtype=np.uint8)
    for family, level in (("noise", 0.0), ("blur", 0.0), ("contrast", 1.0)):
        out = apply_degradation(img, family, level, np.random.default_rng(0))
        assert np.array_equal(out, img), f"{family} 恒等档 {level} 改动了图像"
        assert out.dtype == np.uint8


def test_apply_degradation_noise64_changes_image_and_stays_in_range():
    """σ=64 强加噪:图像确实变了,且收尾量化后仍是 uint8、值域 [0, 255]。"""
    img = np.full((80, 100), 128, dtype=np.uint8)
    out = apply_degradation(img, "noise", 64.0, np.random.default_rng(9))
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img), "σ=64 的加噪不可能不改动图像"
    assert int(out.min()) >= 0 and int(out.max()) <= 255


def test_apply_degradation_unknown_family_chinese_error():
    """未知退化族当场中文报错(T-05-19:坏配置不许静默当恒等处理)。"""
    img = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="退化族"):
        apply_degradation(img, "jpeg", 5.0, np.random.default_rng(0))


def test_apply_degradation_invalid_level_chinese_error():
    """负数 / 非有限强度当场中文报错(强度没有负值,NaN 不许混进档位)。"""
    img = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(AssertionError, match="退化强度不合法"):
        apply_degradation(img, "noise", -1.0, np.random.default_rng(0))
    with pytest.raises(AssertionError, match="退化强度不合法"):
        apply_degradation(img, "blur", float("nan"), np.random.default_rng(0))


def test_apply_degradation_rejects_non_2d_input():
    """非二维输入(比如误传彩色图)当场中文报错,不喂进检测器。"""
    color = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(AssertionError, match="二维灰度图"):
        apply_degradation(color, "noise", 1.0, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# 种子四元组 _seed_for
# ---------------------------------------------------------------------------


def test_seed_quadruplet_deterministic_and_discriminating():
    """四元组任一元不同种子就不同;同四元组永远同种子;值域落在 32 位内。"""
    base = 20261021
    kw = {"image_id": "img_01", "unit_label": "noise_8.0",
          "method_label": "classic_w5", "rep": 0, "base_seed": base}
    ref = _seed_for(**kw)
    assert _seed_for(**kw) == ref, "同四元组必须永远同种子"
    assert 0 <= ref < 2**32
    for tweak in (
            {"image_id": "img_02"}, {"unit_label": "noise_16.0"},
            {"method_label": "sb"}, {"rep": 1}, {"base_seed": base + 1}):
        assert _seed_for(**{**kw, **tweak}) != ref, f"改 {tweak} 种子必须变"


# ---------------------------------------------------------------------------
# 方法候选 build_method_candidates / 检测调度 detect_with
# ---------------------------------------------------------------------------


def test_build_method_candidates_filters_ladder_by_sq_px():
    """窗口阶梯按 2w+1 < sq_px 过滤:大方格容纳大窗口,小方格自动缩档不报错。"""
    cfg = AnalyzeConfig()
    big = build_method_candidates(40.0, cfg)
    big_labels = [c["label"] for c in big]
    assert "classic_w13" in big_labels, "2×13+1=27 < 40,最大档必须进候选"
    assert big_labels.count("sb") == 1 and "sb_acc" in big_labels

    tiny = build_method_candidates(9.0, cfg)
    tiny_classic = [c["label"] for c in tiny if c["kind"] == "classic"]
    assert tiny_classic == ["classic_w3"], (
        f"sq_px=9 时只有 w3 合法(2×3+1=7<9),实际 {tiny_classic}")
    assert "sb" in [c["label"] for c in tiny], "SB 两档不受方格边长过滤"


def test_build_method_candidates_full_profile_adds_exhaustive():
    """full 档追加穷举搜索候选 sb_exh,quick 档没有。"""
    quick = [c["label"] for c in build_method_candidates(40.0, AnalyzeConfig())]
    full = [c["label"] for c in
            build_method_candidates(40.0, AnalyzeConfig(), profile="full")]
    assert "sb_exh" not in quick
    assert "sb_exh" in full


def test_detect_with_candidates_find_truth_under_1px(make_chessboard_image):
    """每个合法候选在干净合成图上都检出 88 个角点,对真值最大位移 < 1px。

    这是"方法 = 检测入口参数,不新增实现"的行为锁:候选经 detect_with
    调度后必须真的能检出——而不是返回 None 或错位点。
    """
    img, corners, _ = make_chessboard_image()
    sq_px = estimate_sq_px(corners, 11, 8)
    truth = canonicalize(corners, 11, 8)
    candidates = build_method_candidates(float(sq_px), AnalyzeConfig())
    assert candidates, "合成图方格边长下至少应有一个合法候选"
    for cand in candidates:
        result = detect_with(cand, img, 11, 8, _CFG)
        assert result is not None, f"候选 {cand['label']} 未检出合成棋盘"
        assert result["uv"].shape == (88, 2)
        max_err = float(np.linalg.norm(result["uv"] - truth, axis=1).max())
        assert max_err < 1.0, (
            f"候选 {cand['label']} 最大定位误差 {max_err:.3f}px 超过 1px")


def test_force_win_pins_window_and_unsafe_window_rejected(make_chessboard_image):
    """force_win 指定窗口生效(win_size 钉在指定档),违反安全断言当场报错。"""
    img, corners, _ = make_chessboard_image(square_px=20.0)
    sq_px = estimate_sq_px(corners, 11, 8)

    pinned = replace(_CFG, subpix=replace(_CFG.subpix, force_win=5))
    result = detect_classic(img, 11, 8, pinned)
    assert result is not None
    assert result["win_size"] == 5, "指定窗口必须被逐字采用,不被常规推导顶掉"

    # 2×12+1 = 25 不小于本帧方格像素边长(约 20):头号陷阱防线必须在场
    unsafe = replace(_CFG, subpix=replace(_CFG.subpix, force_win=12))
    with pytest.raises(AssertionError, match="指定窗口断言失败"):
        detect_classic(img, 11, 8, unsafe)
    assert 2 * 12 + 1 >= sq_px * 0.99  # 前提自检:这一档在本图上确实不安全


def test_corner_opt_source_has_no_cv2_corner_calls():
    """源码防线:corner_opt.py 里绝不出现 cv2 角点函数字样(检测唯一入口)。"""
    source = Path(__file__).resolve().parents[1].joinpath(
        "src", "calib", "analysis", "corner_opt.py").read_text(encoding="utf-8")
    assert "cornerSubPix" not in source, "二次亚像素精化只允许在 corners/detect.py"
    assert "findChessboardCorners" not in source, (
        "粗定位只允许在 corners/detect.py;corner_opt 只走 detect_classic/detect_sb")


# ---------------------------------------------------------------------------
# 主编排 run_corner_opt(小样本端到端;真实数据端到端留给集成测试)
# ---------------------------------------------------------------------------

# 检测器级 σ 口径照 noise.json 顶层键的量级(未显著劣化阈值 = 10 × 它)
_SIGMA = 0.008708


def _small_cfg(max_workers: int) -> CalibrationConfig:
    """小网格配置:2 帧 × 5 单元 × 4 方法,几分钟内跑完的端到端样本。"""
    small = AnalyzeConfig(
        noise_levels=(0.0, 8.0, 32.0),
        blur_levels=(1.0,),
        contrast_gains=(0.6,),
        classic_win_ladder=(3, 5),
        quick_corner_opt_frames=2,
        corner_opt_n_reps=1,
        analyze_max_workers=max_workers,
    )
    return replace(CalibrationConfig(), analyze=small)


def _build_world(make_chessboard_image, tmp_path: Path, n_frames: int = 2):
    """合成小世界:写盘棋盘照 + SB 检出作观测集 + evaluate 交付参数桩。

    观测集的 image_points 用生产 SB 干净检出填充(与生产管线同源);物点
    网格按夹具同一公式现造(square_px=20、z=800 → 物理方格 20 个单位)。
    """
    cols, rows = 11, 8
    ids, uv_rows, paths, poses = [], [], [], []
    for i in range(n_frames):
        img, _truth, _sq = make_chessboard_image(
            rvec=np.array([0.05, 0.05 + 0.02 * i, 0.0]),
            tvec=np.array([0.0, 0.0, 800.0 + 10.0 * i]))
        path = tmp_path / f"img_{i:02d}.png"
        if not cv2.imwrite(str(path), img):
            raise RuntimeError(f"测试夹具写图失败:{path}")
        res = detect_sb(img, cols, rows, CornersConfig())
        assert res is not None, f"合成棋盘第 {i} 帧 SB 未检出"
        ids.append(f"img_{i:02d}")
        uv_rows.append(res["uv"])
        paths.append(str(path))
        poses.append(f"pose_{i}")
    # 物点网格(与夹具同公式;run_corner_opt 不消费物点,只为观测集形状完整)
    square_world = 20.0
    xs = (np.arange(cols) - (cols - 1) / 2.0) * square_world
    ys = (np.arange(rows) - (rows - 1) / 2.0) * square_world
    objp = np.zeros((rows * cols, 3), dtype=np.float64)
    objp[:, 0] = np.tile(xs, rows)
    objp[:, 1] = np.repeat(ys, cols)
    obs = ObservationSet(
        image_ids=tuple(ids),
        pose_ids=tuple(poses),
        object_points=objp,
        image_points=np.stack(uv_rows),
        image_size=(640, 480),
        corner_source=tuple("sb" for _ in ids),
    )
    records = [{"image_id": i, "path": p} for i, p in zip(ids, paths)]
    report = {"delivery": {"models": {"linear": {"final_params": {
        "fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0}}}}}
    board = SimpleNamespace(cols=cols, rows=rows)
    return obs, records, report, board


def test_run_corner_opt_small_grid_end_to_end(make_chessboard_image, tmp_path):
    """小网格端到端:段结构、双口径并排、校准档零位移、加噪单调、明细行数。"""
    obs, records, report, board = _build_world(make_chessboard_image, tmp_path)
    section, detail = run_corner_opt(
        "linear", report, obs, records, board, _small_cfg(1), _SIGMA)

    assert section["schema"] == "analyze-corner-opt-v1"
    assert section["n_frames"] == 2
    assert section["calibration_unit"] == "noise_0.0"
    assert section["n_units"] == 5  # noise 0/8/32 + blur_1.0 + contrast_0.6
    assert set(section["stats"].keys()) == {"_self", "_prod_sb"}, "双口径必须并排"
    assert {"classic_w3", "classic_w5", "sb", "sb_acc"} <= set(section["methods"])

    # 校准档零位移:退化恒等 + 同方法同图同代码路径 → 逐位 0(裁定①主口径)
    self_sb = section["stats"]["_self"]["sb"]
    assert self_sb["noise_0.0"]["rms_px"] == pytest.approx(0.0, abs=1e-12)
    # 加噪越强偏得越多(随机噪声下 32 档不可能反超 8 档还更准)
    assert self_sb["noise_8.0"]["rms_px"] is not None
    assert self_sb["noise_32.0"]["rms_px"] >= self_sb["noise_8.0"]["rms_px"]
    # 阈值随段落账;判读文案双口径说明 + 生产方法点名
    assert section["threshold_px"] == pytest.approx(10.0 * _SIGMA)
    assert "生产" in section["verdict_note"] and "_self" in section["verdict_note"]

    # 明细大数组:行数 = 帧 × 方法 × 重复 × 单元,编号列同长;校准档逐位 0
    n_rows = 2 * len(section["methods"]) * 1 * section["n_units"]
    assert detail["disp_self"].shape == (n_rows, 88, 2)
    assert detail["disp_prodsb"].shape == (n_rows, 88, 2)
    for key in ("frame_idx", "method_idx", "rep_idx", "unit_idx"):
        assert detail[key].shape == (n_rows,)
    cal_u = section["units"].index("noise_0.0")
    sb_m = section["methods"].index("sb")
    cal_rows = np.flatnonzero(
        (detail["method_idx"] == sb_m) & (detail["unit_idx"] == cal_u))
    assert np.all(detail["disp_self"][cal_rows] == 0.0), (
        "校准档必须逐位零位移,否则参照对齐有假")


def test_run_corner_opt_same_seed_reproducible(make_chessboard_image, tmp_path):
    """同种子完整重跑:逐角点位移数组逐位一致(退化图不落盘的正当性根基)。"""
    obs, records, report, board = _build_world(make_chessboard_image, tmp_path)
    cfg = _small_cfg(1)
    sec_a, det_a = run_corner_opt(
        "linear", report, obs, records, board, cfg, _SIGMA)
    sec_b, det_b = run_corner_opt(
        "linear", report, obs, records, board, cfg, _SIGMA)
    for key in ("disp_self", "disp_prodsb"):
        assert np.array_equal(det_a[key], det_b[key], equal_nan=True), (
            f"{key} 同种子重跑出现逐位漂移,可复现承诺被破坏")
    assert np.array_equal(det_a["uv_clean"], det_b["uv_clean"], equal_nan=True)
    assert sec_a["stats"]["_self"]["sb"]["noise_32.0"]["rms_px"] == (
        sec_b["stats"]["_self"]["sb"]["noise_32.0"]["rms_px"])


def test_run_corner_opt_serial_parallel_bitwise_identical(
        make_chessboard_image, tmp_path):
    """串行(1 worker)与并行(2 worker)逐位一致(04-02 确定性红线)。"""
    obs, records, report, board = _build_world(make_chessboard_image, tmp_path)
    _sec_s, det_s = run_corner_opt(
        "linear", report, obs, records, board, _small_cfg(1), _SIGMA)
    _sec_p, det_p = run_corner_opt(
        "linear", report, obs, records, board, _small_cfg(2), _SIGMA)
    for key in ("disp_self", "disp_prodsb"):
        assert np.array_equal(det_s[key], det_p[key], equal_nan=True), (
            f"{key} 并行与串行出现末位漂移——cv2 线程锁未随任务进 worker")


def test_run_corner_opt_taskfailed_envelope_nan_and_count(
        make_chessboard_image, tmp_path, monkeypatch):
    """单任务异常包 TaskFailed 信封转 NaN 占行 + 计数留痕,其余方法照常出数。

    事故注在 worker 层(任务级),信封保护的正是这一层;阶段 A 参照检出
    在主流程里,不受影响。
    """
    real_worker = corner_opt_module._corner_opt_worker

    def flaky_worker(task):
        if task[4]["label"] == "classic_w3":  # 任务元组第 4 位是方法候选
            raise RuntimeError("注入的检测事故(测试)")
        return real_worker(task)

    monkeypatch.setattr(corner_opt_module, "_corner_opt_worker", flaky_worker)
    obs, records, report, board = _build_world(make_chessboard_image, tmp_path)
    section, detail = run_corner_opt(
        "linear", report, obs, records, board, _small_cfg(1), _SIGMA)

    # classic_w3 的 2 个任务(帧 × 重复)全部失败,计数如实落账
    assert section["n_failed_tasks"] == 2
    bad_m = section["methods"].index("classic_w3")
    assert np.isnan(detail["disp_self"][detail["method_idx"] == bad_m]).all()
    assert np.isnan(detail["disp_prodsb"][detail["method_idx"] == bad_m]).all()
    # 其余方法不受牵连:生产位校准档仍逐位零位移,统计照常
    assert section["stats"]["_self"]["sb"]["noise_0.0"]["rms_px"] == (
        pytest.approx(0.0, abs=1e-12))
    # 失败方法在该单元的统计为"空"而不是伪装的 0(n=0 + 数值 None)
    assert section["stats"]["_self"]["classic_w3"]["noise_8.0"]["n"] == 0
    assert section["stats"]["_self"]["classic_w3"]["noise_8.0"]["rms_px"] is None
