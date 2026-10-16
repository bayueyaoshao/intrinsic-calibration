"""2×2 基线标定的合成数据回归测试:已知真值的相机参数,断言求解数学正确。

全部用例跑在合成数据上(相机内参/位姿全已知,角点是带"标准答案"的):
- 针孔模型:8 个不同位姿的已知真值视角,断言 RMS < 0.5px 且 fx/fy/cx/cy
  对真值误差 < 1%;
- 鱼眼模型:锁定的行为是"绝不抛异常、结果结构完整"——收敛成功或带中文
  失败说明的未收敛,两种都算通过(KB4 对非鱼眼广角可能不收敛,D-10 预期内);
- 剔除清单:注入一张乱序坏角点照片,人工剔除后该照片确实不进标定;
- square_size 解耦:方格物理边长只允许出现在 solve/baseline.py 的
  build_object_points 一处,窗口推导等像素级逻辑绝不引用它(IO-03)。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from calib.io.config import BoardSpec, CalibrationConfig, FisheyeConfig
from calib.solve.baseline import (
    build_object_points,
    calibrate_fisheye,
    calibrate_pinhole,
    run_baselines,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_true_views(make_chessboard_image, n_views: int = 8):
    """造 n_views 个不同位姿的已知真值视角(默认内参,深度统一 800)。

    返回 (角点列表, 真值 K)。深度固定是必须的:合成夹具按"焦距×物理尺寸/距离"
    反推方格物理边长,各视角深度不一会让物点网格尺度自相矛盾。
    """
    rng = np.random.default_rng(7)
    k_true = np.array([[800.0, 0.0, 320.0],
                       [0.0, 800.0, 240.0],
                       [0.0, 0.0, 1.0]])
    views = []
    for _k in range(n_views):
        rvec = np.array([rng.uniform(-0.15, 0.15),
                         rng.uniform(-0.10, 0.20),
                         rng.uniform(-0.15, 0.15)])
        tvec = np.array([rng.uniform(-30.0, 30.0),
                         rng.uniform(-30.0, 30.0),
                         800.0])
        _img, corners, _sq = make_chessboard_image(
            K=k_true, rvec=rvec, tvec=tvec)
        views.append(corners)
    return views, k_true


def test_pinhole_recovers_known_truth(make_chessboard_image):
    """回归断言:8 个真值视角上,针孔标定的 RMS < 0.5px 且内参误差 < 1%。"""
    views, _k_true = _make_true_views(make_chessboard_image)
    # 合成几何里方格物理边长 = 20px × 800深度 / 800焦距 = 20(与 board 一致才不自相矛盾)
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)

    result = calibrate_pinhole(views, (640, 480), board)

    assert result["converged"] is True
    assert result["failure_note"] is None
    assert result["rms_overall"] < 0.5, f"整体 RMS {result['rms_overall']:.3f}px 超线"
    assert result["per_view_rms"].shape == (len(views),)
    assert float(result["per_view_rms"].max()) < 0.5

    k_est = result["K"]
    for name, true_value, est in (("fx", 800.0, k_est[0, 0]), ("fy", 800.0, k_est[1, 1]),
                                  ("cx", 320.0, k_est[0, 2]), ("cy", 240.0, k_est[1, 2])):
        assert abs(est - true_value) / true_value < 0.01, \
            f"{name} 对真值误差超 1%:真值 {true_value},估计 {est:.3f}"


def test_pinhole_image_size_is_width_height_order(make_chessboard_image):
    """陷阱 8 锁定:imageSize 参数是 (宽, 高) 语义。

    在 2560×1440 的"传感器"上投影真值角点(主点在 1280,720),
    标定恢复的主点必须贴着 宽/2=1280 与 高/2=720——
    若宽高顺序被写反,这条断言立刻显形。
    """
    rng = np.random.default_rng(11)
    k_true = np.array([[900.0, 0.0, 1280.0],
                       [0.0, 900.0, 720.0],
                       [0.0, 0.0, 1.0]])
    views = []
    for _k in range(8):
        rvec = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.16),
                         rng.uniform(-0.12, 0.12)])
        tvec = np.array([rng.uniform(-40.0, 40.0), rng.uniform(-40.0, 40.0), 800.0])
        # 方格物理边长 60:投影后约 900×60/800 ≈ 67px,棋盘半宽约 5×67≈340px,
        # 1280±340 与 720±340 都稳稳落在 2560×1440 画面内。
        _img, corners, _sq = make_chessboard_image(
            cols=11, rows=8, square_px=67.0, K=k_true, rvec=rvec, tvec=tvec,
            image_size=(1440, 2560))
        views.append(corners)

    board = BoardSpec(cols=11, rows=8, square_size_mm=60.0)
    result = calibrate_pinhole(views, (2560, 1440), board)  # (宽, 高),不是 (1440, 2560)

    k_est = result["K"]
    assert abs(k_est[0, 2] - 1280.0) / 2560.0 < 0.01, \
        f"cx 应贴着 宽/2=1280,实际 {k_est[0, 2]:.1f}——检查 imageSize 是否按 (宽,高) 传入"
    assert abs(k_est[1, 2] - 720.0) / 1440.0 < 0.01, \
        f"cy 应贴着 高/2=720,实际 {k_est[1, 2]:.1f}——检查 imageSize 是否按 (宽,高) 传入"


def test_fisheye_never_raises_and_structure_complete():
    """鱼眼 KB4:合成 KB4 角点上求解,锁定行为 = 不抛异常 + 结果结构键一个不少。

    合成数据按 KB4 的几何假设造(真广角视场:棋盘拉近到深度 250、
    焦距贴近 OpenCV 的内部初值 max(宽,高)/π ≈ 204),入射角最大约 26°,
    畸变信号真实可辨——回归检验的是"数据合假设时求解数学正确"。
    收敛成功(rms 合理、failure_note 为空)或未收敛(带中文 failure_note、
    数值字段全 None)都算通过——KB4 对非鱼眼广角可能不收敛,这是预期发现
    (实测提醒:若把棋盘放远变成近针孔数据,内部初值偏离真值过远,
    会收敛到 RMS 高达几十像素的假收敛,恰好说明 KB4 初值敏感性)。
    """
    k_true = np.array([[204.0, 0.0, 320.0],
                       [0.0, 204.0, 240.0],
                       [0.0, 0.0, 1.0]])
    d_true = np.array([0.05, -0.02, 0.005, 0.001])
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    objp = build_object_points(board).reshape(-1, 1, 3)

    rng = np.random.default_rng(23)
    views = []
    for _k in range(8):
        rvec = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.25),
                         rng.uniform(-0.2, 0.2)])
        tvec = np.array([rng.uniform(-15.0, 15.0), rng.uniform(-15.0, 15.0), 250.0])
        uv, _jac = cv2.fisheye.projectPoints(objp, rvec, tvec, k_true, d_true)
        assert np.isfinite(uv).all() and np.abs(uv - [320, 240]).max() < 300, \
            "合成鱼眼角点跑出画面,测试数据构造有误"
        views.append(uv.reshape(-1, 2))

    result = calibrate_fisheye(views, (640, 480), board, FisheyeConfig())

    # 结构键一个不少(失败的值语义:数值字段为 None,键必须在)
    for key in ("rms_overall", "K", "D", "rvecs", "tvecs", "per_view_rms",
                "flags_used", "converged", "failure_note"):
        assert key in result, f"鱼眼结果缺结构键:{key}"
    assert isinstance(result["converged"], bool)

    if result["converged"]:
        assert result["failure_note"] is None
        assert result["rms_overall"] < 0.5, \
            f"KB4 合成回归 RMS {result['rms_overall']:.3f}px 超线"
        assert result["per_view_rms"].shape == (len(views),)
    else:
        note = result["failure_note"]
        assert isinstance(note, str) and note.strip(), "未收敛时必须有中文失败说明"
        assert result["K"] is None and result["D"] is None
        assert result["rms_overall"] is None and result["per_view_rms"] is None


def test_fisheye_shape_assertion_raises_chinese_error():
    """陷阱 5 锁定:角点形状不合时,进函数先断言,报错一眼看懂(中文)。"""
    board = BoardSpec(cols=11, rows=8, square_size_mm=25.0)
    bad_views = [np.zeros((87, 2))] * 3  # 少了一个角点
    with pytest.raises(AssertionError, match="形状"):
        calibrate_fisheye(bad_views, (640, 480), board, FisheyeConfig())


def test_exclude_images_drops_bad_view(make_chessboard_image):
    """D-03 锁定:剔除清单里的照片被逻辑剔除——注入一张乱序坏角点照片,
    不剔除时标定被带坏,剔除后恢复干净,证明该照片确实没进标定。"""
    views, _k_true = _make_true_views(make_chessboard_image)
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    image_size = (640, 480)
    ids = [f"pose_{k:02d}_00{k}" for k in range(len(views))]

    def _pack(corner_list):
        return {
            detector: {
                "image_ids": np.asarray(ids),
                "uv": np.asarray(corner_list),
                "detected": np.ones(len(ids), dtype=bool),
            }
            for detector in ("classic", "sb")
        }

    # 坏数据:第 3 张的角点整体乱序(同名角点指向了棋盘对侧的典型污染形态)
    rng = np.random.default_rng(31)
    corrupted = list(views)
    corrupted[3] = corrupted[3][rng.permutation(corrupted[3].shape[0])]

    outcome_bad = run_baselines(_pack(corrupted), image_size,
                                CalibrationConfig(), board)
    rms_bad = outcome_bad["baselines"]["baseline_classic_pinhole"]["rms_overall"]
    assert rms_bad > 0.5, f"乱序角点理应把标定带坏(RMS {rms_bad:.3f}px 却很低,注入无效)"

    cfg_exclude = CalibrationConfig(exclude_images=(ids[3],))
    outcome_ok = run_baselines(_pack(corrupted), image_size, cfg_exclude, board)

    # 剔除名单留痕:被剔除的照片编号原样进返回值,供 lineage 落盘(T-03-02)
    assert outcome_ok["excluded_images"] == [ids[3]]
    pinhole_ok = outcome_ok["baselines"]["baseline_classic_pinhole"]
    assert ids[3] not in pinhole_ok["image_ids"]
    assert len(pinhole_ok["image_ids"]) == len(ids) - 1
    assert pinhole_ok["rms_overall"] < 0.5, \
        f"剔除坏照片后 RMS 应恢复到 0.5px 内,实际 {pinhole_ok['rms_overall']:.3f}px"


def test_run_baselines_four_combo_names_and_metadata(make_chessboard_image):
    """四个组合名精确 = classic/sb × pinhole/fisheye,且每个结果带落盘元信息。"""
    views, _k_true = _make_true_views(make_chessboard_image, n_views=4)
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    data = {
        "image_ids": np.asarray([f"img_{k}" for k in range(4)]),
        "uv": np.asarray(views),
        "detected": np.ones(4, dtype=bool),
    }
    outcome = run_baselines({"classic": data, "sb": data}, (640, 480),
                            CalibrationConfig(), board)

    assert set(outcome["baselines"]) == {
        "baseline_classic_pinhole", "baseline_classic_fisheye",
        "baseline_sb_pinhole", "baseline_sb_fisheye",
    }
    for name, result in outcome["baselines"].items():
        detector, model = name.removeprefix("baseline_").split("_")
        assert result["detector"] == detector and result["model"] == model
        assert result["image_size"] == (640, 480)
        assert result["n_points_per_view"] == 88
        assert result["objp"].shape == (88, 3)
    assert outcome["n_views"] == {"classic": 4, "sb": 4}


def test_too_few_usable_views_raises_chinese_error():
    """剔除后可用视图不足 2 个:直接报中文错误,不硬算。"""
    board = BoardSpec(cols=11, rows=8, square_size_mm=25.0)
    data = {
        "image_ids": np.asarray(["only_one"]),
        "uv": np.zeros((1, 88, 2)),
        "detected": np.ones(1, dtype=bool),
    }
    with pytest.raises(ValueError, match="可用视图"):
        run_baselines({"classic": data, "sb": data}, (640, 480),
                      CalibrationConfig(), board)


def test_square_size_decoupled_from_pixel_logic():
    """IO-03 解耦断言:square_size 只允许出现在 solve/baseline.py 一处。

    窗口推导(corners/window.py)与未来的体检(io/precheck.py)都不得引用它——
    方格物理边长填错只该影响"距离类指标",且能被这里抓住。
    """
    baseline_src = (REPO_ROOT / "src/calib/solve/baseline.py").read_text(encoding="utf-8")
    assert "square_size" in baseline_src, "build_object_points 应引用 square_size"

    window_src = (REPO_ROOT / "src/calib/corners/window.py").read_text(encoding="utf-8")
    assert "square_size" not in window_src, \
        "窗口推导是纯像素级逻辑,不得引用方格物理边长(square_size)"

    precheck = REPO_ROOT / "src/calib/io/precheck.py"
    if precheck.exists():  # 文件由后续计划(01-04)创建,存在即须合规
        precheck_src = precheck.read_text(encoding="utf-8")
        assert "square_size" not in precheck_src, \
            "体检指标应按实测方格像素边长归一,不得引用方格物理边长(square_size)"
