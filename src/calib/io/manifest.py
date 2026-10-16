"""数据清单(manifest.json)写盘:入库体检阶段的唯一产物、后续阶段的唯一入口。

每次运行落独立目录 output/ingest/run_<UTC 时间戳>/,带血缘指纹,
同一批数据重跑多少次都互不覆盖、跨轮可比(迭代回环 D-04 的地基)。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from calib.io.dataset import ImageRecord
from calib.util.lineage import make_lineage


def _relpath_for_manifest(path: Path) -> str:
    """把照片路径转成相对仓库根(当前工作目录)的统一格式写进 manifest。"""
    path = Path(path)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(Path.cwd()))
        except ValueError:
            return str(path)
    return str(path)


def summarize(records: tuple[ImageRecord, ...] | list[ImageRecord],
              resolution: tuple[int, int],
              warnings: tuple[str, ...] | list[str]) -> dict:
    """从扫描结果汇总统计:全部数字来自实际记录,代码里没有任何写死的张数。"""
    train = [r for r in records if r.split == "train"]
    test = [r for r in records if r.split == "test"]

    # 重复数分布只对 train 姿态有意义:{"8": 80} 表示 80 个姿态各有 8 张连拍。
    # test 每张独立,没有"连拍"概念,单独计数。
    poses: dict[str, int] = {}
    for r in train:
        poses[r.pose_id] = r.n_repeats
    repeats_distribution: dict[str, int] = {}
    for n in poses.values():
        key = str(n)
        repeats_distribution[key] = repeats_distribution.get(key, 0) + 1

    width, height = resolution
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_poses": len(poses),
        "repeats_distribution": repeats_distribution,
        "n_images": len(records),
        "resolution": [width, height],
        "warnings": list(warnings),
    }


def write_manifest(records: tuple[ImageRecord, ...] | list[ImageRecord],
                   resolution: tuple[int, int],
                   warnings: tuple[str, ...] | list[str],
                   out_root: Path,
                   config_files: dict[str, Path]) -> Path:
    """写 manifest.json 到 output/ingest/run_<时间戳>/,返回该运行目录。

    参数:
        records:      scan_dataset 的照片记录
        resolution:   (宽, 高),来自 verify_consistent_resolution
        warnings:     扫描阶段的全部中文警告(留痕,不崩溃的承诺兑现处)
        out_root:     阶段输出根(如 output/ingest)
        config_files: 配置文件名 → 路径,写进血缘指纹
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    round_id = f"run_{timestamp}"
    run_dir = Path(out_root) / round_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        # 血缘指纹:这份清单是什么时候、用什么配置算出来的(ITER-01)
        "lineage": make_lineage(config_files=config_files, upstream={}, round_id=round_id),
        "summary": summarize(records, resolution, warnings),
        # schema 预留说明:后续的体检阶段会给每条 image 追加一个 precheck 对象
        # (模糊度/曝光/棋盘可见性等体检结论),当前版本尚无该键,下游读到缺失属正常。
        "schema_note": "后续阶段会给 images 里每条记录追加 precheck 对象(体检结论),当前版本不含该键。",
        "images": [
            {
                "image_id": r.image_id,
                "path": _relpath_for_manifest(r.path),
                "split": r.split,
                "pose_id": r.pose_id,
                "repeat_idx": r.repeat_idx,
                "n_repeats": r.n_repeats,
            }
            for r in records
        ],
    }

    target = run_dir / "manifest.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return run_dir


def latest_run_dir(output_root: Path, stage: str) -> Path:
    """返回某阶段最新的 run_<时间戳> 目录(目录名字典序 = 时间序)。

    下游命令(corners/calibrate/...)用它定位上游产物;
    没有任何运行目录时抛中文错误,提示先跑上游命令。
    """
    stage_dir = Path(output_root) / stage
    if not stage_dir.is_dir():
        raise ValueError(
            f"目录 {stage_dir} 不存在:还没有跑过「{stage}」这一步。"
            f"请先运行对应的上游命令(流水线顺序见 calib --help)。"
        )
    runs = sorted(p for p in stage_dir.iterdir()
                  if p.is_dir() and p.name.startswith("run_"))
    if not runs:
        raise ValueError(
            f"目录 {stage_dir} 下没有任何 run_<时间戳> 运行目录,"
            f"请先运行对应的上游命令。"
        )
    return runs[-1]
