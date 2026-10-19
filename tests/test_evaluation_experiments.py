"""四类防作弊实验(交叉验证 / 留一姿态 / 重复张分组 / bootstrap)的单元测试(小样合成数据)。

小样证明什么(真实数据 640 视图的试车在 04-05/04-06 集成):
- cv_pose_grouped(Task 1,D-01/D-02/EVA-01):折级样本数 = 轮 × 折、每折
  训练/验证姿态零交集且并集 = 全体、验证误差按"先姿态级平均再进折级"聚合、
  验证侧外参绝不参与训练(只估位姿评分的计数断言)、同种子重跑逐位一致、
  单折失败转契约占行不中断;
- make_calibrate_fn(Task 1):折内精化必带 rms_linear_ref(Anti-Pattern 5),
  f_scale 只在推荐配置显式给出时才覆盖配置;
- _px_dispersion(Task 1):参数差换算成"像素位移当量"(参数数值差没有
  直观意义,近零畸变参数的相对百分比更无意义);
- lopo / slices_by_index(Task 2,D-05/D-06/D-08)与
  bootstrap_pose / pose_sampling_variance(Task 3,D-04/D-07)随对应任务追加。

合成观测集用 conftest 的 make_true_views(已知真值内参的合成棋盘角点,
6 姿态 × 2 张),配置对象直构造绕过 yaml(体例照 test_optimizer_comparison.py);
并行上限写 1 走串行回退,monkeypatch 才能生效。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from calib.evaluation import experiments as experiments_mod
from calib.evaluation.experiments import (
    _px_dispersion,
    cv_pose_grouped,
    make_calibrate_fn,
)
from calib.evaluation.splitting import assign_cv_folds, unique_poses
from calib.io.config import BoardSpec, CalibrationConfig
from calib.models import create_model
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet
from calib.evaluation.splitting import obs_rows, pose_row_mask


def _assert_reports_equal(a, b, path="报告") -> None:
    """递归逐位比较(数组用 np.array_equal,浮点 NaN 视为相等)——
    "同种子重跑逐位一致"的硬断言工具。直接 == 会碰上
    "数组真值不明确"的坑,必须按类型分派。"""
    if isinstance(a, dict):
        assert isinstance(b, dict), path
        assert set(a) == set(b), f"{path}: 键集合不一致 {set(a) ^ set(b)}"
        for k in a:
            _assert_reports_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_reports_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.shape == b.shape, \
            f"{path}: 数组形状不一致 {a.shape} vs {getattr(b, 'shape', None)}"
        assert np.array_equal(a, b, equal_nan=True), f"{path}: 数组数值不一致"
    elif isinstance(a, float):
        if np.isnan(a):
            assert isinstance(b, float) and np.isnan(b), f"{path}: NaN 不对称"
        else:
            assert a == b, f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture
def cv_setup(make_true_views):
    """小样 CV 地基:6 姿态 × 2 张合成观测集 + 2 折 × 2 轮 × 2 模型小配置。

    并行上限写 1(串行回退):monkeypatch 替换模块级函数才能生效,
    与 test_optimizer_comparison.py 同一体例。
    """
    views, _k = make_true_views(n_views=12)  # 6 姿态 × 2 张连拍
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    obs = ObservationSet(
        image_ids=tuple(f"img_{k:02d}" for k in range(12)),
        pose_ids=tuple(f"pose_{k // 2:02d}" for k in range(12)),
        object_points=build_object_points(board),
        image_points=np.asarray(views, dtype=np.float64),
        image_size=(640, 480),
        corner_source=("sb",) * 12,
    )
    base = CalibrationConfig()
    cfg = replace(base, evaluate=replace(base.evaluate,
                                         models=("m1", "m2"),
                                         cv_n_folds=2,
                                         cv_n_rounds=2,
                                         evaluate_max_workers=1))
    optimizer_config = {"m1": {"loss": "linear"}, "m2": {"loss": "linear"}}
    return {"obs": obs, "board": board, "cfg": cfg,
            "optimizer_config": optimizer_config}


# ---------------------------------------------------------------------------
# Task 1:cv_pose_grouped(行为 1–5)+ 求解胶水
# ---------------------------------------------------------------------------

def test_cv_fold_samples_and_zero_overlap(cv_setup):
    """行为 1:小样 2 轮 × 2 折 × 2 模型跑通;每模型折级样本数 = 轮 × 折 = 4;
    每折训练/验证姿态零交集且并集 = 全体(明细里的分配表逐折断言)。"""
    s = cv_setup
    report = cv_pose_grouped(s["obs"], s["board"], s["cfg"],
                             s["optimizer_config"])
    assert report["schema"] == "cv-v1"
    assert set(report["models"]) == {"m1", "m2"}

    poses = set(unique_poses(s["obs"]))
    detail = report["_cv_detail"]
    n_folds_total = len(detail["assignment_round"])
    assert n_folds_total == 4, "折级分配数必须是 轮 × 折 = 4"
    for k in range(n_folds_total):
        train = set(str(detail["assignment_train_poses"][k]).split(","))
        val = set(str(detail["assignment_val_poses"][k]).split(","))
        assert not (train & val), \
            f"第 {k} 折训练/验证姿态有交集:{sorted(train & val)}"
        assert train | val == poses, \
            f"第 {k} 折两侧并集必须是全体姿态(缺:{sorted(poses - train - val)})"

    for m in ("m1", "m2"):
        summary = report["models"][m]
        assert len(summary["fold_heldout"]) == 4
        assert len(summary["fold_train"]) == 4
        assert summary["n_failed_folds"] == 0, \
            f"{m} 在干净合成数据上不该有失败折:{summary['fold_rows']}"
        assert all(v is not None and np.isfinite(v)
                   for v in summary["fold_heldout"])
        assert summary["mean_heldout_px"] == pytest.approx(
            float(np.mean(summary["fold_heldout"])))
        assert summary["mean_train_px"] is not None
        assert summary["loss_used"] == "linear"
        # 折级行的键一个不少(消费面:04-03 判行表 + 04-05 落盘)
        for row in summary["fold_rows"]:
            for key in ("round", "fold", "model_id", "train_rms_px",
                        "heldout_px", "n_train_poses", "n_val_poses",
                        "n_train_views", "n_val_views", "f_scale_actual",
                        "failure_note"):
                assert key in row, f"折级行缺键:{key}"


def test_cv_pose_level_aggregation(cv_setup, monkeypatch):
    """行为 2(D-02 聚合口径):某姿态 2 张的 RMS 已知可算 → 姿态级误差 =
    2 张 RMS 的平均(不按角点数加权、不是 2 行都进折级);折级 held-out =
    验证姿态级误差的均值。用受控注册表直接认出每张照片,期望值手算。"""
    s = cv_setup
    # 注册表:行 k(姿态 p = k//2,第 j = k%2 张)→ 受控 RMS = 1.0·p + 0.1·j
    registry = {}
    for k in range(12):
        registry[s["obs"].image_points[k].tobytes()] = \
            1.0 * (k // 2) + 0.1 * (k % 2)

    def fake_solve_pose_only(uv_view, object_points, params, model):
        return np.zeros(3), np.zeros(3), registry[uv_view.tobytes()]

    monkeypatch.setattr(experiments_mod, "solve_pose_only",
                        fake_solve_pose_only)
    report = cv_pose_grouped(s["obs"], s["board"], s["cfg"],
                             s["optimizer_config"])

    # 期望值从确定性分配表手算:姿态级 = 两张 RMS 的平均(相距 0.1,均值在正中)
    assignment = assign_cv_folds(unique_poses(s["obs"]), s["cfg"])
    pose_level = {f"pose_{p:02d}": 1.0 * p + 0.05 for p in range(6)}
    for m in ("m1", "m2"):
        for i, entry in enumerate(assignment):
            expected = float(np.mean([pose_level[p]
                                      for p in entry["val_poses"]]))
            got = report["models"][m]["fold_heldout"][i]
            assert got == pytest.approx(expected), (
                f"{m} 折 {i}(验证姿态 {entry['val_poses']})的折级 held-out "
                f"应为姿态级误差的均值 {expected},实际 {got}")

    # per_pose_rows:每个(模型, 折, 验证姿态)恰一行,数值 = 姿态两张的平均
    rows = report["per_pose_rows"]
    assert len(rows) == 2 * 4 * 3, "每模型 4 折 × 3 验证姿态 = 12 行"
    for row in rows:
        p = int(str(row["pose_id"]).split("_")[1])
        assert row["rms_px"] == pytest.approx(1.0 * p + 0.05)
        assert row["n_views"] == 2
        assert row["failure_note"] is None


def test_cv_validation_extrinsics_never_trained(cv_setup, monkeypatch):
    """行为 3(防泄漏第二层):验证姿态评分只经 solve_pose_only(计数 =
    模型 × 折 × 每折验证姿态 × 每姿态张数);训练求解的观测集行数 =
    训练姿态 × 2、不含任何验证姿态,且每次精化都带 rms_linear_ref。"""
    s = cv_setup
    pnp_calls: list[bytes] = []
    real_pnp = experiments_mod.solve_pose_only

    def counting_pnp(uv_view, object_points, params, model):
        pnp_calls.append(uv_view.tobytes())
        return real_pnp(uv_view, object_points, params, model)

    monkeypatch.setattr(experiments_mod, "solve_pose_only", counting_pnp)

    train_snapshots: list[tuple[tuple[str, ...], float | None]] = []
    real_solve = experiments_mod.solve_with_scipy

    def spy_solve(obs, model, cfg, init=None, loss="linear",
                  rms_linear_ref=None):
        train_snapshots.append((tuple(obs.pose_ids), rms_linear_ref))
        return real_solve(obs, model, cfg, init=init, loss=loss,
                          rms_linear_ref=rms_linear_ref)

    monkeypatch.setattr(experiments_mod, "solve_with_scipy", spy_solve)

    report = cv_pose_grouped(s["obs"], s["board"], s["cfg"],
                             s["optimizer_config"])
    assert report["schema"] == "cv-v1"

    assignment = assign_cv_folds(unique_poses(s["obs"]), s["cfg"])
    n_val_scores = sum(len(e["val_poses"]) for e in assignment) * 2
    assert len(pnp_calls) == 2 * n_val_scores, (
        f"只估位姿评分次数应恰为 模型 × 验证打分数 = {2 * n_val_scores},"
        f"实际 {len(pnp_calls)}——训练侧绝不该走 solve_pose_only")
    assert len(set(pnp_calls)) == len(pnp_calls), \
        "同一张照片被重复打分,评分对账不上"

    assert len(train_snapshots) == 2 * len(assignment)
    for train_poses, rms_ref in train_snapshots:
        assert len(train_poses) == 6, \
            f"训练观测集行数应是 训练姿态 3 × 2 张 = 6,实际 {len(train_poses)}"
        assert len(set(train_poses)) == 3, "训练观测集应恰含 3 个姿态"
        assert rms_ref is not None, \
            "折内精化必带 rms_linear_ref(Anti-Pattern 5:漏传让收敛判定失真)"


def test_cv_same_seed_reruns_bit_identical(cv_setup):
    """行为 4:同种子两次运行,汇总表 + 明细数组逐位一致(确定性红线)。"""
    s = cv_setup
    r1 = cv_pose_grouped(s["obs"], s["board"], s["cfg"], s["optimizer_config"])
    r2 = cv_pose_grouped(s["obs"], s["board"], s["cfg"], s["optimizer_config"])
    _assert_reports_equal(r1, r2)


def test_cv_fold_failure_becomes_contract_not_crash(cv_setup, monkeypatch):
    """行为 5:注入 m2 必失败的求解 → m2 各折转失败契约占行(失败率进汇总,
    说明带异常类型),m1 照常收敛——一个组合病了不拖死整批(T-04-04-03)。"""
    s = cv_setup
    real_solve = experiments_mod.solve_with_scipy

    def boom(obs, model, cfg, init=None, loss="linear", rms_linear_ref=None):
        if model.model_id == "m2":
            raise RuntimeError("人为注入的求解爆炸")
        return real_solve(obs, model, cfg, init=init, loss=loss,
                          rms_linear_ref=rms_linear_ref)

    monkeypatch.setattr(experiments_mod, "solve_with_scipy", boom)
    report = cv_pose_grouped(s["obs"], s["board"], s["cfg"],
                             s["optimizer_config"])

    m2 = report["models"]["m2"]
    assert m2["n_failed_folds"] == 4
    assert all(v is None for v in m2["fold_heldout"])
    assert m2["mean_heldout_px"] is None, "全部折失败时均值不能冒充数值"
    for row in m2["fold_rows"]:
        assert row["failure_note"], "失败折必须带非空中文说明"
        assert "RuntimeError" in row["failure_note"]
        assert row["heldout_px"] is None and row["train_rms_px"] is None

    m1 = report["models"]["m1"]
    assert m1["n_failed_folds"] == 0
    assert all(v is not None for v in m1["fold_heldout"])


def test_make_calibrate_fn_cv_refine_carries_rms_linear_ref(cv_setup,
                                                            monkeypatch):
    """求解胶水:make_calibrate_fn 先 OpenCV 初值再 scipy 精化,精化实参
    必带 rms_linear_ref = OpenCV 初值的回代 RMS;f_scale 只在显式给出时
    才覆盖配置(推荐配置带 f_scale_px 的情形,D-13)。"""
    s = cv_setup
    model = create_model("m1")
    captured: dict = {}
    real_scipy = experiments_mod.solve_with_scipy
    real_opencv = experiments_mod.solve_with_opencv

    def spy_scipy(obs, model, cfg, init=None, loss="linear",
                  rms_linear_ref=None):
        captured["loss"] = loss
        captured["rms_linear_ref"] = rms_linear_ref
        captured["f_scale"] = cfg.f_scale
        return real_scipy(obs, model, cfg, init=init, loss=loss,
                          rms_linear_ref=rms_linear_ref)

    monkeypatch.setattr(experiments_mod, "solve_with_scipy", spy_scipy)

    train_obs = obs_rows(s["obs"], pose_row_mask(
        s["obs"], ["pose_00", "pose_01", "pose_02"]))
    calibrate_fn = make_calibrate_fn(model, s["cfg"].calibrate.scipy, "linear")
    result = calibrate_fn(train_obs)
    assert result.converged, f"合成数据上应收敛:{result.failure_note}"

    init = real_opencv(train_obs, model)  # 确定性求解,重解拿参照 RMS 对账
    assert captured["loss"] == "linear"
    assert captured["rms_linear_ref"] == pytest.approx(float(init.rms_overall))
    assert captured["f_scale"] == "auto", "没显式给 f_scale 就该保持配置原值"

    captured.clear()
    calibrate_fs = make_calibrate_fn(model, s["cfg"].calibrate.scipy,
                                     "soft_l1", f_scale=2.5)
    calibrate_fs(train_obs)
    assert captured["f_scale"] == pytest.approx(2.5), \
        "推荐配置显式给出 f_scale_px 时必须覆盖 auto 档"


def test_px_dispersion_pixel_equivalence():
    """像素位移当量:两组参数只差 fx 1.0 像素、正对位姿下 → 最大位移 =
    最远角点的归一化横坐标 × 1.0(= 0.125 px,手算锁死)——参数差用
    像素说话,不用相对百分比(近零畸变参数的相对值无意义)。"""
    board = BoardSpec(cols=11, rows=8, square_size_mm=20.0)
    model = create_model("m1")
    params_a = dict(zip(model.param_names,
                        (800.0, 800.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0)))
    params_b = dict(params_a, fx=801.0)
    objp = build_object_points(board)
    disp = _px_dispersion([params_a, params_b], model, objp,
                          np.zeros(3), np.array([0.0, 0.0, 800.0]))
    x_norm = np.abs(objp[:, 0]) / 800.0   # 正对位姿:深度 800,归一化 = x/Z
    assert disp == pytest.approx(float(x_norm.max()), abs=1e-9)
    assert disp == pytest.approx(0.125, abs=1e-9)
    # 单组参数无从谈"两两之间"的位移,返回 0 不崩
    assert _px_dispersion([params_a], model, objp,
                          np.zeros(3), np.array([0.0, 0.0, 800.0])) == 0.0
