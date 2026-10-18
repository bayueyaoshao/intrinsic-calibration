"""时域噪声底:利用同姿态连拍的角点波动直接实测噪声,并给每条结果标注来源。

白话原理:相机对着同一个摆放位置连拍几张,画面里的"真实"棋盘没动,
角点位置却每次差一点——这点差异就来自噪声(像素噪声 + 拍摄瞬间的
微小位姿抖动)。对每张照片的同一编号角点算"重复间标准差",就得到
逐角点的噪声水平;再把 88×2 张小表汇成姿态级、全局级的标量。

两条计算路径(成功标准 1 的"来源逐条标注"):
- temporal:检出帧 ≥ 2 的姿态,直接算连拍波动(真实测量);
- immerkaer_transfer:检出帧不足 2 张的姿态(以及 test 切分的独立照,
  它们天然无连拍),退化为"Immerkaer 单帧像素噪声 → 经传递曲线折算成
  角点误差"的间接估计(传递曲线 = 注入实验量出的"像素噪声被检测器
  放大成多大角点误差"的经验关系)。

检测器噪声与位姿抖动的分离(成功标准 1):时域总波动² − 检测器口径² =
位姿抖动²。减法出负数说明两个测量量的差距在测量不确定度之内,不是错误——
截断为零并打 clamped 标记,绝不静默吞掉。

上游输入只有落盘产物(corners npz + 照片清单记录 + 逐张 Immerkaer 估计),
本模块纯读文件现算,不碰图像(库层不依赖 CLI 层,进度播报由调用方自理)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calib.io.config import NoiseConfig
from calib.noise.transfer import interp_corner_sigma

# Immerkaer 主线口径名 → 逐张估计字典里的键(与 injection.py 同一套配置口径)
_VARIANT_KEYS = {"std": "sigma_std", "robust": "sigma_robust"}

# 全局离群姿态点名的个数(只点名不剔除,研究 Pattern 1 的裁量项)
_N_OUTLIERS = 3


def _rec_get(rec, key: str):
    """从照片记录取字段:清单字典(CLI 传)与 ImageRecord(测试传)都认。"""
    if isinstance(rec, dict):
        return rec[key]
    return getattr(rec, key)


def pose_groups(records) -> dict[str, list[int]]:
    """把照片记录按姿态编号分组,只聚 train 切分,返回 {姿态: 行号列表}。

    records 的元素是清单字典或 ImageRecord(两种形态都认);
    返回的下标是"传入列表自己的行号",test 独立照不构成连拍,不进分组。
    """
    groups: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        if _rec_get(rec, "split") == "train":
            groups.setdefault(_rec_get(rec, "pose_id"), []).append(i)
    return groups


def load_corners_rows(corners_npz_path: Path | str) -> dict:
    """读回角点 npz 的整表数据,读前先验形状(上游文件坏了当场响)。

    返回 {"image_ids": [str], "uv": (N, 点数, 2) float64,
          "detected": (N,) bool, "source": [str]}。
    读取强制 allow_pickle=False(npz 的 pickle 模式可以执行任意代码,
    项目既定防线);形状/行数互相矛盾时给中文报错点名文件——行序错位
    不报错只算错,必须在入口拦下(威胁项 T-03-03-01)。
    """
    with np.load(Path(corners_npz_path), allow_pickle=False) as data:
        image_ids = [str(x) for x in np.asarray(data["image_ids"]).ravel()]
        uv = np.asarray(data["uv"], dtype=np.float64)
        detected = np.asarray(data["detected"], dtype=bool)
        source = [str(x) for x in np.asarray(data["source"]).ravel()]

    n = len(image_ids)
    if uv.ndim != 3 or uv.shape[-1] != 2 or uv.shape[0] != n \
            or detected.shape != (n,) or len(source) != n:
        raise AssertionError(
            f"角点文件 {Path(corners_npz_path).name} 的形状/行数对不上:"
            f"image_ids {n} 行、uv {uv.shape}、detected {detected.shape}、"
            f"source {len(source)} 条。期望 uv 是 (行数, 点数, 2) 且四者行数"
            f"一致。这份文件多半已损坏或与棋盘规格不配套,请重跑 calib corners。"
        )
    return {"image_ids": image_ids, "uv": uv, "detected": detected,
            "source": source}


def _main_sigma(image_id: str, immerkaer_sigmas: dict[str, dict],
                variant_key: str) -> float:
    """取一张照片的主线口径 Immerkaer 估计;缺失即中文报错点名。"""
    entry = immerkaer_sigmas.get(image_id)
    if entry is None or variant_key not in entry:
        raise ValueError(
            f"照片 {image_id} 没有可用的 Immerkaer 噪声估计(口径 {variant_key})。"
            f"请确认调用方已对该照片算过 immerkaer_sigma 并按 image_id 传入。")
    return float(entry[variant_key])


def temporal_noise(corners_npz_path: Path | str, records,
                   immerkaer_sigmas: dict[str, dict],
                   curve_interp, cfg: NoiseConfig) -> dict:
    """时域噪声底主入口:逐姿态连拍波动 + 来源标注 + 单帧退化 + 抖动分离。

    参数:
        corners_npz_path: train 切分的 SB 路线角点 npz(逐行带 detected 掩码);
        records:          照片记录清单(manifest["images"] 字典或 ImageRecord 均可),
                          用于姿态归属与 test 切分的独立照名单;
        immerkaer_sigmas: {image_id: {"sigma_std": .., "sigma_robust": ..}},
                          逐张 Immerkaer 像素噪声估计(主线口径由配置选);
        curve_interp:     传递曲线插值函数(对数 σ 进、欧氏角点误差出,
                          来自 transfer.fit_interp;测试可注入假曲线);
        cfg:              NoiseConfig(主线口径 / 池化统计量 / 切分名单)。

    返回(数值单位一律"像素/轴",与理论精度计算的标量 Σ 同口径):
        sigma_temporal_corner / sigma_detector_corner:全局双口径标量;
        sigma_temporal_rms / sigma_temporal_median:两种池化并报,主值按配置;
        n_poses_temporal / n_poses_fallback / per_source_counts:来源计数;
        outlier_poses:σ_pose 最大的至多 3 个姿态(只点名,数值不剔除);
        poses:逐姿态明细(sigma_per_corner 为 (点数, 2) 数组,退化姿态为 None);
        splits:train 汇总与 test 逐张折算记录。
    """
    variant_key = _VARIANT_KEYS.get(cfg.immerkaer_variant)
    if variant_key is None:
        raise ValueError(
            f"immerkaer_variant 配置不认识:{cfg.immerkaer_variant!r};"
            f"可选:{sorted(_VARIANT_KEYS)}。请检查 calibration.yaml 的 noise 段。")
    if "train" not in cfg.noise_splits:
        raise ValueError(
            f"noise.noise_splits = {list(cfg.noise_splits)} 不含 train:"
            f"时域噪声底必须有训练连拍,请把 train 加回划分名单。")

    rows = load_corners_rows(corners_npz_path)

    # 姿态归属:角点行的每张照片必须能在 train 记录里找到(行序对齐的
    # 轻量断言——找不到多半是角点文件与照片清单不是同一轮,串了轮)
    pose_of_row: list[str] = []
    train_pose_of_image: dict[str, str] = {}
    seen: set[str] = set()
    for rec in records:
        image_id = _rec_get(rec, "image_id")
        if _rec_get(rec, "split") == "train":
            if image_id in seen:
                raise ValueError(
                    f"照片记录清单里 image_id 重复:{image_id}。"
                    f"请检查 manifest 是否损坏。")
            seen.add(image_id)
            train_pose_of_image[image_id] = _rec_get(rec, "pose_id")
    for image_id in rows["image_ids"]:
        if image_id not in train_pose_of_image:
            raise ValueError(
                f"角点文件里的照片 {image_id} 在照片清单的 train 切分里找不到:"
                f"角点产物与照片清单多半不是同一轮,请重跑 calib corners 或"
                f"改用同一轮的 manifest.json。")
        pose_of_row.append(train_pose_of_image[image_id])

    # 按姿态聚角点行号(分组 helper 吃"行号 → 姿态"的轻量记录列表,
    # 与吃 manifest 记录是同一条代码路径)
    row_shims = [{"split": "train", "pose_id": p} for p in pose_of_row]
    groups = pose_groups(row_shims)

    pose_entries: list[dict] = []
    for pose_id in sorted(groups):
        row_idxs = groups[pose_id]
        det_idxs = [i for i in row_idxs if rows["detected"][i]]
        mean_imm = float(np.mean([
            _main_sigma(rows["image_ids"][i], immerkaer_sigmas, variant_key)
            for i in row_idxs]))

        if len(det_idxs) >= 2:
            # 主路径:连拍重复间标准差(逐角点逐轴,ddof=1 的无偏估计)
            sigma_per_corner = np.std(rows["uv"][det_idxs], axis=0, ddof=1)
            sigma_pose = float(np.sqrt(np.mean(sigma_per_corner ** 2)))
            sigma_detector = interp_corner_sigma(mean_imm, curve_interp)
            variance = sigma_pose ** 2 - sigma_detector ** 2
            # 减法出负 = 时域波动没有超过检测器自身噪声,差距在测量不确定度
            # 之内,不是错误:截断为零并打 clamped 标记留痕,绝不静默吞掉
            clamped = bool(variance < 0.0)
            sigma_jitter = float(np.sqrt(max(variance, 0.0)))
            pose_entries.append({
                "pose_id": pose_id,
                "n_frames": len(row_idxs),
                "n_detected": len(det_idxs),
                "source": "temporal",
                "sigma_per_corner": sigma_per_corner,
                "sigma_pose": sigma_pose,
                "sigma_detector_used": sigma_detector,
                "sigma_jitter": sigma_jitter,
                "clamped": clamped,
            })
        else:
            # 退化路径:检出帧不足 2 张(或 test 独立照)没法算波动,
            # 用该姿态各帧 Immerkaer σ 均值经传递曲线折算成角点误差
            sigma_transfer = interp_corner_sigma(mean_imm, curve_interp)
            pose_entries.append({
                "pose_id": pose_id,
                "n_frames": len(row_idxs),
                "n_detected": len(det_idxs),
                "source": "immerkaer_transfer",
                "sigma_per_corner": None,
                "sigma_pose": sigma_transfer,
                "sigma_detector_used": sigma_transfer,
                "sigma_jitter": 0.0,
                "clamped": False,
            })

    sigma_poses = np.asarray([p["sigma_pose"] for p in pose_entries],
                             dtype=np.float64)
    sigma_rms = float(np.sqrt(np.mean(sigma_poses ** 2)))
    sigma_median = float(np.median(sigma_poses))
    pooling = cfg.temporal_pooling
    if pooling not in ("rms", "median"):
        raise ValueError(
            f"temporal_pooling 配置不认识:{pooling!r};可选:['median', 'rms']。"
            f"请检查 calibration.yaml 的 noise 段。")
    sigma_global = sigma_rms if pooling == "rms" else sigma_median

    # 全局检测器口径:全部 train 图主线口径 Immerkaer σ 的中位数处查曲线
    all_imm = [_main_sigma(image_id, immerkaer_sigmas, variant_key)
               for image_id in rows["image_ids"]]
    sigma_detector_global = interp_corner_sigma(float(np.median(all_imm)),
                                                curve_interp)

    # 离群姿态:σ_pose 最大的至多 3 个,只点名不剔除(数值原样保留)
    order = np.argsort(sigma_poses)[::-1][:_N_OUTLIERS]
    outlier_poses = [{"pose_id": pose_entries[i]["pose_id"],
                      "sigma_pose": float(sigma_poses[i])} for i in order]

    # test 切分(配置含 test 时):独立照逐张走折算路径,天然无连拍
    test_per_image: list[dict] = []
    if "test" in cfg.noise_splits:
        for rec in records:
            if _rec_get(rec, "split") != "test":
                continue
            image_id = _rec_get(rec, "image_id")
            sigma_px = _main_sigma(image_id, immerkaer_sigmas, variant_key)
            test_per_image.append({
                "image_id": image_id,
                "sigma_immerkaer": sigma_px,
                "sigma_corner_per_axis": interp_corner_sigma(sigma_px,
                                                             curve_interp),
                "source": "immerkaer_transfer",
            })

    n_temporal = sum(1 for p in pose_entries if p["source"] == "temporal")
    n_fallback = len(pose_entries) - n_temporal
    return {
        "sigma_temporal_corner": sigma_global,
        "sigma_detector_corner": sigma_detector_global,
        "sigma_temporal_rms": sigma_rms,
        "sigma_temporal_median": sigma_median,
        "pooling": pooling,
        "n_poses_temporal": n_temporal,
        "n_poses_fallback": n_fallback,
        "per_source_counts": {"temporal": n_temporal,
                              "immerkaer_transfer": n_fallback},
        "outlier_poses": outlier_poses,
        "poses": pose_entries,
        "splits": {
            "train": {"n_poses": len(pose_entries),
                      "n_images": len(rows["image_ids"])},
            "test": {"n_images": len(test_per_image),
                     "per_image": test_per_image},
        },
    }
