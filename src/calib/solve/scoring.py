"""逐帧回代评分与只估位姿小求解:held-out 评分的地基(D-04 复核的引擎)。

白话:标定把内参算出来之后,"一张没参与标定的照片拍得准不准"怎么量?
把标定得到的内参**当尺子用**,只挪棋盘的位姿(6 个数:3 个旋转 + 3 个
平移),让棋盘投影到画面上的位置尽量贴近检测到的角点——挪完之后还剩下
的偏差就是这张照片的重投影误差。内参固定、只估位姿,解的是个 6 参数的
小最小二乘,每张毫秒级。任何注册模型都适用(玩具模型也不例外,D-09):
投影只走 model.project,初值的反投影只走 model.unproject,
本文件不出现任何具体模型的名字,也不出现 OpenCV 的投影函数。

两个入口:
- score_views:整批回代评分——位姿已知(train 帧标定时解过)时逐帧算
  RMS。实现直接委托 compute_result_metrics(同一把尺只此一份,严禁分叉);
- solve_pose_only:单视角只估位姿小求解——位姿未知(test 帧没参与标定)
  时先把位姿估出来再算 RMS。初值先用 model.unproject 把角点反投影成
  视线方向(反投影本身就完成了去畸变),再在归一化平面坐标里用
  cv2.solvePnP 解位姿(线性方法、模型无关),精化用 scipy 的 least_squares。

在两者之上还有一个编排入口 review_suspects:把"体检判存疑的照片算起来
是不是真的差"(阶段一人工验收遗留问题,D-04)做成带血缘的落盘结论——
train 帧直接读标定结果的逐帧 RMS,test 帧逐张只估位姿后评分,再与数据
清单的存疑判定逐行对齐,产出全量逐帧表、两组统计与中文结论。

失败语义:输入形状不对当场中文报错(不带病进求解);solvePnP 解不出
(点太少或退化)抛中文 ValueError,调用方决定单帧记 NaN 还是中断。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from calib.io.config import BoardSpec, CalibrationConfig
from calib.models import create_model
from calib.models.base import CameraModel, as_params, transform_to_camera
from calib.solve.matrix import combo_dir_name
from calib.solve.observation import ObservationSet, load_observation_set
from calib.solve.result import compute_result_metrics, load_result
from calib.util.lineage import make_lineage
from calib.visualization.suspect_compare import plot_suspect_compare

# solvePnP 迭代解法要求的最少点数(4 点以下方程组欠定,硬算必错)
_MIN_PNP_POINTS = 4

# 存疑组"明显更差"的判定阈值(D-04 走向 A 的线):存疑组误差中位数相对
# 正常组中位数的偏差(差的绝对值 / 正常组中位数)达到 30% 判"明显更差"。
# 阈值本身随结论落盘(T-02-06-02:判定式可追溯,不凭空下结论),
# 数据说话——不到线就是"两组无差别",不预设存疑帧有害。
_WORSE_DEVIATION_THRESHOLD = 0.30


def score_views(obs: ObservationSet,
                params: np.ndarray,
                rvecs: np.ndarray,
                tvecs: np.ndarray,
                model: CameraModel) -> np.ndarray:
    """整批回代评分:逐帧重投影 RMS,形状 (V,),单位像素。

    与 result.compute_result_metrics 是**同一把尺**(本函数直接委托它,
    不存在第二份实现——行为测试锁定逐位一致):残差 =
    model.project(物点经 transform_to_camera 变换到相机坐标系) − 实测
    角点,逐帧 RMS = sqrt(该帧全部角点平方欧氏残差的平均)。

    参数:
        obs:    观测集(物点网格 + 实测角点,行序契约见 observation.py)
        params: (P,) 内参参数向量(按 model.param_names 布局)
        rvecs / tvecs: (V,3) 逐视图位姿,与 obs 的视图逐行对齐
        model:  任一注册的 CameraModel 实例
    """
    _rms_overall, per_view_rms, _residuals = compute_result_metrics(
        obs, params, rvecs, tvecs, model)
    return per_view_rms


def solve_pose_only(uv_view: np.ndarray,
                    object_points: np.ndarray,
                    params: np.ndarray,
                    model: CameraModel) -> tuple[np.ndarray, np.ndarray, float]:
    """内参当尺子用、只挪棋盘的位姿,给没参与标定的照片打分。

    白话:给定一张照片的 88 个检测角点、棋盘物点网格、标定得到的内参,
    求这张照片里棋盘摆在哪(6 个位姿数),再看贴得有多紧(剩余 RMS)。
    内参全程固定不动——它是"尺子",不是被量对象。

    步骤(两段,全部只经 CameraModel 接口):
    1. 初值 = 角点先经 model.unproject 反投影成单位视线方向(反投影的
       牛顿迭代天然完成去畸变),取 x/z、y/z 转成归一化平面坐标,再
       cv2.solvePnP(物点, 归一化坐标, 单位阵 K, 零畸变)解位姿——
       线性方法、模型无关,且初值不带畸变偏差。直接拿像素坐标按零畸变
       解 PnP 的老路线在大倾角 + 强畸变 + 棋盘偏画面中心时会把第二段的
       精化带进错误盆地(真实案例:40° 倾角帧被误评 4.82px,正确解
       0.07px,四模型同坑——反投影路线在四模型上全部救回);
    2. 精化 = scipy least_squares(6 参数,残差仅
       model.project(transform_to_camera(物点, rvec, tvec)) − 角点,
       method="trf",x_scale="jac")。

    参数:
        uv_view:       (P,2) 单视角实测角点(像素),P ≥ 4;
        object_points: (P,3) 物点网格(与 uv_view 逐行对齐,Z=0 棋盘平面);
        params:        (P,) 标定得到的内参参数向量(固定不动);
        model:         任一注册的 CameraModel 实例。

    返回:(rvec (3,), tvec (3,), rms float)——精化后的位姿与回代 RMS(像素)。
    """
    uv = np.asarray(uv_view, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise AssertionError(
            f"uv_view 形状不合:期望 (P,2) 单视角实测角点,实际是 {uv.shape}。"
            f"只估位姿小求解一次处理一张照片,请检查调用方传入的数据排布。"
        )
    objp = np.asarray(object_points, dtype=np.float64)
    if objp.shape != (uv.shape[0], 3):
        raise AssertionError(
            f"object_points 形状不合:期望 ({uv.shape[0]},3) 与角点逐行对齐,"
            f"实际是 {objp.shape}。物点网格与检测角点必须一一对齐,请检查行序。"
        )
    if uv.shape[0] < _MIN_PNP_POINTS:
        raise ValueError(
            f"角点数 {uv.shape[0]} 少于位姿求解下限 {_MIN_PNP_POINTS} 个:"
            f"方程组欠定,位姿解不出来,请检查该帧的角点检出质量。"
        )
    p = as_params(params, model.model_id, model.param_names)

    # ---- 初值:反投影到归一化平面坐标后按单位阵 K 解 PnP ----
    # 零畸变直接解像素坐标的老路线会把精化带进错误盆地(见函数 docstring
    # 的真实案例),反投影路线对针孔与鱼眼一律成立,且只经 CameraModel 接口。
    rays = model.unproject(uv, p)
    if not bool((rays[:, 2] > 0).all()):
        raise ValueError(
            f"模型 {model.model_id} 的视线反投影出现 Z 分量非正的角点,"
            f"归一化平面坐标不存在,该帧几何可疑:"
            f"请按单帧失败记录,不影响其余帧。")
    ok, rvec0, tvec0 = cv2.solvePnP(
        objp, rays[:, :2] / rays[:, 2:3], np.eye(3, dtype=np.float64),
        np.zeros(5, dtype=np.float64))
    if not ok:
        raise ValueError(
            f"模型 {model.model_id} 的位姿初值求解失败(solvePnP 解不出):"
            f"该帧角点可能退化(近共线/分布过窄),复核时按单帧失败记录,不影响其余帧。"
        )

    # ---- 精化:6 参数最小二乘,内参加锁 ----
    def residual(theta: np.ndarray) -> np.ndarray:
        pts_cam, _pose_block = transform_to_camera(objp, theta[:3], theta[3:])
        return (model.project(pts_cam, p) - uv).ravel()

    theta0 = np.concatenate([np.asarray(rvec0, dtype=np.float64).ravel(),
                             np.asarray(tvec0, dtype=np.float64).ravel()])
    sol = least_squares(residual, theta0, method="trf", x_scale="jac")

    rvec = sol.x[:3].copy()
    tvec = sol.x[3:].copy()
    rms = float(np.sqrt(np.mean(np.sum(sol.fun.reshape(-1, 2) ** 2, axis=1))))
    return rvec, tvec, rms


def _empty_review(conclusion: str) -> dict:
    """空结论(不抛异常):没有可用标定结果等情况下返回带中文说明的空表。"""
    return {
        "lineage": {},
        "review_model": None,
        "result_used": None,
        "fallback": None,
        "threshold": None,
        "group_stats": {"suspect": None, "normal": None},
        "verdict": "not_generated",
        "conclusion": conclusion,
        "frame_table": [],
    }


def _group_stats(values: list[float | None]) -> dict:
    """一组的统计:样本数、有效数、中位数、均值、四分位数(无效值不计入)。"""
    valid = np.asarray([v for v in values
                        if v is not None and np.isfinite(v)], dtype=np.float64)
    stats = {"n": len(values), "n_valid": int(valid.size),
             "median": None, "mean": None, "q25": None, "q75": None}
    if valid.size:
        stats.update(median=float(np.median(valid)),
                     mean=float(np.mean(valid)),
                     q25=float(np.percentile(valid, 25)),
                     q75=float(np.percentile(valid, 75)))
    return stats


def _score_test_frames(corners_run: Path, manifest_path: Path, board: BoardSpec,
                       cfg: CalibrationConfig, params_vec: np.ndarray,
                       model: CameraModel) -> dict[str, tuple[float | None, str | None]]:
    """test 切分逐帧评分(位姿未知 → 只估位姿小求解),返回 编号 → (RMS, 备注)。

    单帧失败记 (None, 中文备注),不中断整批循环(T-02-06-04);test 角点
    文件不在时全部帧带中文备注留痕,同样不中断。
    """
    route = cfg.calibrate.corner_route
    test_npz = corners_run / f"corners_test_{route}.npz"
    scores: dict[str, tuple[float | None, str | None]] = {}
    if not test_npz.is_file():
        return scores  # 调用方按"文件不在"统一给备注

    obs = load_observation_set(test_npz, manifest_path, board, split="test")
    for i, image_id in enumerate(obs.image_ids):
        try:
            _rvec, _tvec, rms = solve_pose_only(
                obs.image_points[i], obs.object_points, params_vec, model)
            scores[image_id] = (rms, None)
        except (ValueError, AssertionError) as e:
            scores[image_id] = (None, f"只估位姿求解失败:{e}")
    return scores


def review_suspects(calibration_run: Path | str, corners_run: Path | str,
                    manifest_path: Path | str, board: BoardSpec,
                    cfg: CalibrationConfig) -> dict:
    """存疑帧复核编排(D-04 闭环):体检判"存疑"的照片算起来是不是真的差?

    做法:选本轮一个标定结果当尺子(优先 <模型>__scipy,缺失/未收敛回退
    <模型>__opencv 并留 fallback 记号)→ train 帧直接读它的逐帧 RMS、
    test 帧逐张只估位姿后评分 → 与数据清单按 image_id 逐行对齐(存疑判定
    与来源记号原样保留,陌生取值不吞)→ 全量逐帧表 + 两组统计 + 按阈值
    生成的中文结论,带血缘落盘 suspect_review/。

    返回:结论文档字典(与落盘的 suspect_vs_normal.json 同一内容,外加
    files 键记两个产物的路径)。没有可用结果或清单为空时返回带中文说明
    的空结论(不抛异常、不写盘),标定本身不受影响。
    """
    calibration_run = Path(calibration_run)
    corners_run = Path(corners_run)
    manifest_path = Path(manifest_path)
    model_id = cfg.calibrate.suspect_review_model

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        return _empty_review(
            f"存疑帧复核未生成:数据清单 {manifest_path.name} 里没有照片清单,"
            f"没有可复核的对象。请先跑 calib ingest。")

    # ---- 选结果:优先 scipy 主线,回退 opencv(fallback 留痕),都没有则空结论 ----
    # 组合命名只准走 combo_dir_name(一处定死);scipy 主线必须显式给
    # loss="linear"(缺省会拼成 _scipy_linear 之外的变体名,指错目录)
    scipy_combo = combo_dir_name(model_id, "scipy", "linear")
    opencv_combo = combo_dir_name(model_id, "opencv")
    doc: dict | None = None
    chosen_combo: str | None = None
    fallback_note: str | None = None
    for combo in (scipy_combo, opencv_combo):
        if not (calibration_run / combo / "result.json").is_file():
            continue
        candidate = load_result(calibration_run / combo)
        if candidate.get("converged") and candidate.get("rms_overall") is not None:
            doc, chosen_combo = candidate, combo
            break
    if doc is not None and chosen_combo == opencv_combo:
        fallback_note = (f"优先组合 {scipy_combo} 不存在或未收敛,"
                         f"回退用 {opencv_combo} 的结果做对照")
    if doc is None:
        return _empty_review(
            f"存疑帧复核未生成:模型 {model_id} 的 {scipy_combo} 与 "
            f"{opencv_combo} 两个组合都不存在或都未收敛,没有可用的内参当尺子。"
            f"请先跑完 calib calibrate 的模型矩阵。")

    model = create_model(model_id)
    params_by_name = doc.get("params") or {}
    missing = [name for name in model.param_names if name not in params_by_name]
    if missing:
        return _empty_review(
            f"存疑帧复核未生成:所选结果 {chosen_combo} 的参数字典缺 "
            f"{model.model_id} 的参数名 {missing},这份 result.json 可疑,"
            f"请重跑该组合。")
    params_vec = np.array([float(params_by_name[name])
                           for name in model.param_names], dtype=np.float64)

    # ---- 清单逐行对齐(image_id 一等键;存疑判定与来源记号原样保留)----
    manifest_by_id: dict[str, dict] = {}
    for entry in images:
        precheck = entry.get("precheck") or {}
        manifest_by_id[str(entry["image_id"])] = {
            "pose_id": entry.get("pose_id"),
            "split": entry.get("split"),
            "verdict": precheck.get("verdict"),
            "suspect_sources": list(precheck.get("suspect_sources", [])),
        }

    train_rms: dict = doc.get("per_view_rms") or {}
    test_scores = _score_test_frames(corners_run, manifest_path, board,
                                     cfg, params_vec, model)

    # ---- 全量逐帧表:清单里每一张都占一行,数值缺失给 NaN + 中文备注 ----
    frame_table: list[dict] = []
    test_npz = corners_run / f"corners_test_{cfg.calibrate.corner_route}.npz"
    for entry in images:
        image_id = str(entry["image_id"])
        info = manifest_by_id[image_id]
        split = info["split"]
        rms: float | None = None
        note: str | None = None
        if split == "train":
            if image_id in train_rms:
                rms = float(train_rms[image_id])
            else:
                note = "未进标定(人工剔除或角点未检出),没有该帧的 RMS"
        elif split == "test":
            if image_id in test_scores:
                rms, note = test_scores[image_id]
            elif not test_npz.is_file():
                note = f"test 角点文件不存在({test_npz.name}),无法评分"
            else:
                note = "未进 test 观测集(角点未检出或被剔除),没有该帧的 RMS"
        else:
            note = f"清单切分取值 {split!r} 不认识,原样保留,未评分"
        frame_table.append({
            "image_id": image_id,
            "pose_id": info["pose_id"],
            "split": split,
            "suspect": info["verdict"] == "suspect",
            "verdict": info["verdict"],
            "suspect_sources": info["suspect_sources"],
            "per_view_rms": rms,
            "note": note,
        })

    # ---- 两组统计 + 走向判定(阈值与判定式随结论落盘)----
    stats_suspect = _group_stats([r["per_view_rms"] for r in frame_table if r["suspect"]])
    stats_normal = _group_stats([r["per_view_rms"] for r in frame_table if not r["suspect"]])
    med_s, med_n = stats_suspect["median"], stats_normal["median"]
    threshold = _WORSE_DEVIATION_THRESHOLD
    if med_s is None or med_n is None or med_n <= 0:
        verdict = "undetermined"
        conclusion = (
            f"结论(无法判定):两组的有效数值不足(存疑组有效 "
            f"{stats_suspect['n_valid']} 张、正常组有效 {stats_normal['n_valid']} 张,"
            f"正常组中位数 {med_n}),请检查逐帧表的备注列。")
    else:
        deviation = abs(med_s - med_n) / med_n
        ratio = med_s / med_n
        invalid_note = ""
        n_invalid = (stats_suspect["n"] - stats_suspect["n_valid"]
                     + stats_normal["n"] - stats_normal["n_valid"])
        if n_invalid:
            invalid_note = f"(另有 {n_invalid} 张无有效数值,未计入统计,备注见逐帧表)"
        if deviation >= threshold and med_s > med_n:
            verdict = "worse"
            conclusion = (
                f"结论(走向 A):存疑组明显更差——存疑 {stats_suspect['n_valid']} 张"
                f"有效帧的误差中位数 {med_s:.3f}px 是正常组({stats_normal['n_valid']} 张,"
                f"中位数 {med_n:.3f}px)的 {ratio:.2f} 倍(偏差 "
                f"{deviation * 100:.1f}%,达到 {threshold * 100:.0f}% 阈值)。"
                f"体检指标与真实误差相关:后续如需剔除低质量帧,本目录逐帧表就是依据。"
                f"{invalid_note}")
        elif deviation >= threshold:
            verdict = "better"
            conclusion = (
                f"结论(反常发现):存疑组误差中位数 {med_s:.3f}px 反而低于正常组 "
                f"{med_n:.3f}px(偏差 {deviation * 100:.1f}%,超过 "
                f"{threshold * 100:.0f}% 阈值)——\"看着糊\"的帧算得更准,体检判定"
                f"可能过严,建议人工抽看复核。{invalid_note}")
        else:
            verdict = "similar"
            conclusion = (
                f"结论(走向 B):两组无差别——存疑 {stats_suspect['n_valid']} 张"
                f"有效帧的误差中位数 {med_s:.3f}px 与正常组({stats_normal['n_valid']} 张,"
                f"中位数 {med_n:.3f}px)偏差 {deviation * 100:.1f}%,未达 "
                f"{threshold * 100:.0f}% 阈值。\"看着糊\"≠\"算得差\":存疑帧照常"
                f"留在数据流中,不剔除(D-01)。{invalid_note}")

    # ---- 血缘:upstream = 所选 result.json + manifest.json;
    # 配置指纹直接继承所选结果的(复核与那次标定用的是同一套配置)----
    lineage = make_lineage(
        config_files={},
        upstream={
            "result.json": calibration_run / chosen_combo / "result.json",
            "manifest.json": manifest_path,
        },
        round_id=str(doc.get("lineage", {}).get("round_id") or calibration_run.name),
    )
    inherited = (doc.get("lineage") or {}).get("config_sha256") or {}
    if inherited:
        lineage["config_sha256"] = dict(inherited)

    document = {
        "lineage": lineage,
        "review_model": model_id,
        "result_used": chosen_combo,
        "fallback": fallback_note,
        "threshold": {
            "relative_deviation": threshold,
            "rule": ("存疑组误差中位数相对正常组中位数的偏差"
                     "(|中位数差| / 正常组中位数)≥ 0.30 且存疑组更高,判\"明显更差\""),
        },
        "group_stats": {"suspect": stats_suspect, "normal": stats_normal},
        "verdict": verdict,
        "conclusion": conclusion,
        "frame_table": frame_table,
    }

    out_dir = calibration_run / "suspect_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "suspect_vs_normal.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    png_path = plot_suspect_compare(frame_table, out_dir / "suspect_vs_normal.png")
    document["files"] = {"json": str(json_path), "png": str(png_path)}
    return document
