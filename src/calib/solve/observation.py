"""观测集装载:把角点文件与数据清单对齐成进标定的唯一输入形态(ObservationSet)。

白话:观测集 = 进标定的那部分角点 + 对齐信息。角点文件里躺着某个切分的
全部照片的角点(未检出的整行是 NaN),数据清单记录每张照片属于哪个姿态、
哪个切分;装载就是把这两份落盘产物对齐、过滤,得到求解器要吃的干净数组。
双求解后端(OpenCV / scipy)与存疑帧复核(D-04 的 test 切分只估位姿评分)
共用这一个入口——谁的输入都一样,结果才可比。

行序契约(全链一条,就此定死):
ObservationSet 的视图顺序 = 求解结果里 rvecs / tvecs / per_view_rms 的顺序
= 落盘产物里 image_ids 的顺序。装载时断言角点文件的 image_ids 与数据清单
里该切分的照片逐项相等(连顺序也相等),任何错位直接报错——行序一旦漂移,
D-04 的清单对齐与阶段四按姿态切分会全部错行,而且不报错、只是悄悄算错
(研究 Pitfall 10)。

D-03 界定:本阶段只用 sb 路线的角点(SB 检测为主、经典检测兜底的那份,
逐帧带 source 来源字段,即 D-02 定版的输入)。SB 与 classic 两套角点之间
约 0.135px 的系统差不在这个阶段量化;如需评估,留到阶段四五用 classic
角点做对照。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from calib.io.config import BoardSpec
from calib.solve.baseline import build_object_points

# 装载允许的数据切分:train 进标定,test 供存疑帧复核(D-04)做只估位姿评分
_ALLOWED_SPLITS = {"train", "test"}


@dataclass(frozen=True)
class ObservationSet:
    """一次标定任务的观测集(装载产物,不可变)。

    五个长度为 V 的字段逐行对应同一张照片,行序契约由它们共同承载。
    """

    image_ids: tuple[str, ...]      # (V,) 照片编号,对齐键:D-04 对数据清单、落盘产物都用它逐行对齐
    pose_ids: tuple[str, ...]       # (V,) 姿态编号,阶段四按姿态切分(防数据泄漏)的连接键,现在就留
    object_points: np.ndarray       # (P,3) float64 物点网格(复用 baseline.build_object_points,square_size 只进那里)
    image_points: np.ndarray        # (V,P,2) float64 实测角点像素坐标,全部点检出且为有限值
    image_size: tuple[int, int]     # (宽, 高),取自数据清单 summary 的 resolution
    corner_source: tuple[str, ...]  # (V,) 每帧角点来源:"sb" 或 "classic_fallback",逐帧留痕(D-02)


def _fail(message: str) -> None:
    """抛出中文数据清单错误(统一走本函数,消息体例照 io/config.py 的 _fail)。"""
    raise ValueError(message)


def _read_manifest(manifest_path: Path) -> dict:
    """读数据清单文件:json 用 json.load,yaml 用 safe_load(绝不执行配置里的代码)。"""
    with open(manifest_path, encoding="utf-8") as f:
        if manifest_path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def _split_manifest(manifest: dict, split: str, manifest_path: Path) -> tuple[list[str], dict[str, str], tuple[int, int]]:
    """按切分过滤数据清单,返回 (照片编号列表、编号→姿态编号映射、(宽, 高))。

    照片编号列表的顺序就是清单里该切分的顺序——行序断言以此为"期望值"。
    """
    images = manifest.get("images")
    if not isinstance(images, list):
        _fail(
            f"数据清单 {manifest_path.name} 里没有 images 列表(或它不是列表),"
            f"无法按切分 {split!r} 过滤照片。清单由 calib ingest 生成,"
            f"请检查文件是否损坏或传错了路径。"
        )
    resolution = (manifest.get("summary") or {}).get("resolution")
    if not (isinstance(resolution, (list, tuple)) and len(resolution) == 2):
        _fail(
            f"数据清单 {manifest_path.name} 的 summary.resolution 缺失或不是"
            f"[宽, 高] 两个数,观测集无法确定图像尺寸。"
            f"清单由 calib ingest 生成,请检查文件是否损坏或传错了路径。"
        )
    ids: list[str] = []
    pose_map: dict[str, str] = {}
    for entry in images:
        if entry.get("split") != split:
            continue
        image_id = str(entry["image_id"])
        ids.append(image_id)
        pose_map[image_id] = str(entry["pose_id"])
    return ids, pose_map, (int(resolution[0]), int(resolution[1]))


def _read_npz_image_ids(corners_npz: Path) -> list[str]:
    """只读角点文件的照片编号列(强制 allow_pickle=False,防线照 load_corners)。"""
    with np.load(corners_npz, allow_pickle=False) as data:
        return [str(x) for x in np.asarray(data["image_ids"]).ravel()]


def _assert_row_order(npz_ids: list[str], split_ids: list[str], split: str) -> None:
    """行序锁:角点文件的照片编号必须与该切分的清单逐项相等(研究 Pitfall 10)。

    错位时给出第一处不一致的位置、清单侧与文件侧各自的照片编号、下一步动作。
    """
    if npz_ids == split_ids:
        return
    for k, (file_id, list_id) in enumerate(zip(npz_ids, split_ids)):
        if file_id != list_id:
            mismatch = (f"第一处不一致在第 {k} 行:清单里是「{list_id}」,"
                        f"角点文件里是「{file_id}」")
            break
    else:
        mismatch = (f"前 {min(len(npz_ids), len(split_ids))} 行完全一致,是行数不同:"
                    f"清单 {len(split_ids)} 行,角点文件 {len(npz_ids)} 行")
    raise AssertionError(
        f"角点文件的照片编号与数据清单(split={split})对不上:{mismatch}。"
        f"行序契约要求两者逐行一致(观测集顺序 = 求解结果顺序 = 落盘顺序),"
        f"这两份产物多半不是同一轮的:请重跑 calib corners,"
        f"或改用与角点文件同一轮 ingest 生成的 manifest.json。"
    )


def load_observation_set(corners_npz: Path | str, manifest_path: Path | str,
                         board: BoardSpec, split: str = "train",
                         exclude_images: tuple[str, ...] = ()) -> ObservationSet:
    """把 sb 路线角点文件与数据清单对齐、过滤,装成观测集(行序契约在此定死)。

    参数:
        corners_npz:   sb 路线角点文件路径(如 corners_train_sb.npz,D-02 定版输入)
        manifest_path: 数据清单路径(calib ingest 产出的 manifest.json,json/yaml 均可读)
        board:         棋盘规格(物点网格由此构造,复用 baseline.build_object_points)
        split:         "train"(进标定)或 "test"(供 D-04 存疑帧复核只估位姿评分)
        exclude_images: 人工剔除清单——逻辑剔除(不进标定,原数据不动,D-03);
                       本轮实际命中的名单用 excluded_image_ids 单独查询

    过滤规则(沿用阶段一 run_baselines 的约定):
        剔除清单里的照片跳过;未检出(detected=False,角点是 NaN)的帧跳过;
        检出标记为真但角点含 NaN/Inf 的帧直接抛错,绝不带着坏数进求解;
        幸存视图不足 2 个时报错——两三个视角定不了内参,硬算必错。
    """
    if split not in _ALLOWED_SPLITS:
        raise ValueError(
            f"split 参数不合法:只接受 train 或 test,实际是 {split!r}。"
            f"请检查调用处传入的切分名。"
        )

    manifest = _read_manifest(Path(manifest_path))
    split_ids, pose_map, image_size = _split_manifest(manifest, split, Path(manifest_path))

    # npz 读取强制 allow_pickle=False:npz 的 pickle 模式可以执行任意代码,
    # 关死它(阶段一 load_corners 的防线,此处延续)
    with np.load(Path(corners_npz), allow_pickle=False) as data:
        if "source" not in data.files:
            raise ValueError(
                f"角点文件 {Path(corners_npz).name} 里没有 source 键——它不是"
                f"sb 路线的角点文件(D-02:SB 为主 + classic 兜底那份才逐帧记录来源)。"
                f"请改用 corners_<切分>_sb.npz,或重跑 calib corners 生成 sb 路线产物。"
            )
        npz_ids = [str(x) for x in np.asarray(data["image_ids"]).ravel()]
        uv = np.asarray(data["uv"], dtype=np.float64)
        detected = np.asarray(data["detected"], dtype=bool)
        source = [str(x) for x in np.asarray(data["source"]).ravel()]

    # 行序锁:装载前先确认角点文件与清单逐行一致,错位即停(Pitfall 10)
    _assert_row_order(npz_ids, split_ids, split)

    n_points = board.cols * board.rows
    if uv.ndim != 3 or uv.shape[1:] != (n_points, 2):
        raise AssertionError(
            f"角点数组形状不合:期望 (行数,{n_points},2)(棋盘 {board.cols}×{board.rows}"
            f" 个内角点),实际是 {uv.shape}。多半是角点文件与 board.yaml 的棋盘规格"
            f"不是同一套:请核对 config/board.yaml 与生成角点文件时的配置。"
        )

    exclude = set(exclude_images)
    excluded_seen: list[str] = []
    views: list[np.ndarray] = []
    kept_ids: list[str] = []
    kept_poses: list[str] = []
    kept_sources: list[str] = []
    for i, image_id in enumerate(npz_ids):
        if image_id in exclude:
            excluded_seen.append(image_id)
            continue  # 人工剔除清单:逻辑剔除,原数据不动(D-03)
        if not detected[i]:
            continue  # 未检出的帧是 NaN 角点,进不了标定
        if not np.isfinite(uv[i]).all():
            raise AssertionError(
                f"照片 {image_id} 标记为已检出,但角点里混入了非数值(NaN/无穷),"
                f"角点文件可疑,请重新运行 calib corners。"
            )
        views.append(uv[i])
        kept_ids.append(image_id)
        kept_poses.append(pose_map[image_id])
        kept_sources.append(source[i])
    if len(views) < 2:
        raise ValueError(
            f"split={split} 剔除后只剩 {len(views)} 个可用视图"
            f"(检出且未被人工剔除,本轮实际剔除 {len(excluded_seen)} 张),"
            f"至少需要 2 个才能标定内参。请检查 exclude_images 清单与角点检出率。"
        )

    return ObservationSet(
        image_ids=tuple(kept_ids),
        pose_ids=tuple(kept_poses),
        object_points=build_object_points(board),
        image_points=np.asarray(views, dtype=np.float64),
        image_size=image_size,
        corner_source=tuple(kept_sources),
    )


def excluded_image_ids(corners_npz: Path | str, manifest_path: Path | str,
                       split: str = "train",
                       exclude_images: tuple[str, ...] = ()) -> tuple[str, ...]:
    """返回本轮实际被人工剔除的照片名单(按字典序去重,供血缘落盘留痕)。

    只统计"确实出现在该切分角点文件里、且在剔除清单上"的照片——
    剔除清单里写了别切分的照片不算命中(谁剔了什么可审计,T-03-02 体例)。
    行序校验与 load_observation_set 同一套(错位即抛错),两边口径一致。
    """
    if split not in _ALLOWED_SPLITS:
        raise ValueError(
            f"split 参数不合法:只接受 train 或 test,实际是 {split!r}。"
            f"请检查调用处传入的切分名。"
        )
    manifest = _read_manifest(Path(manifest_path))
    split_ids, _pose_map, _size = _split_manifest(manifest, split, Path(manifest_path))
    npz_ids = _read_npz_image_ids(Path(corners_npz))
    _assert_row_order(npz_ids, split_ids, split)

    exclude = set(exclude_images)
    return tuple(sorted({image_id for image_id in npz_ids if image_id in exclude}))
