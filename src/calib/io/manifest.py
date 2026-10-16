"""数据清单(manifest.json)写盘:入库体检阶段的唯一产物、后续阶段的唯一入口。

每次运行落独立目录 output/ingest/run_<UTC 时间戳>/,带血缘指纹,
同一批数据重跑多少次都互不覆盖、跨轮可比(迭代回环 D-04 的地基)。
"""

from __future__ import annotations

import json
import math
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


def _finite_or_none(value) -> float | None:
    """指标数值统一转成"有限浮点数或 null":缺失/算不出(如整图回退时的
    归一值)写 null,而不是 NaN——NaN 不是合法 JSON,下游工具读不动。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _precheck_object(entry: dict) -> dict:
    """把体检条目(指标数字 + 判定结论)整理成 manifest 里每张照片的 precheck 对象。

    只挑约定键,指标与可见性分开放:数字给机器比对,判定来源给人审阅(D-06
    ——manifest 存全部数字;suspect_sources 逐条列明判定来自谁,D-02)。
    """
    return {
        "blur_raw": _finite_or_none(entry.get("blur_raw")),
        "blur_norm": _finite_or_none(entry.get("blur_norm")),
        "expo_hi": _finite_or_none(entry.get("expo_hi")),
        "expo_lo": _finite_or_none(entry.get("expo_lo")),
        "visibility": {
            "classic_ok": bool(entry.get("classic_ok", False)),
            "sb_ok": bool(entry.get("sb_ok", False)),
            "box_area_ratio": _finite_or_none(entry.get("box_area_ratio")),
            "corners_count_ok": bool(entry.get("corners_count_ok", False)),
        },
        "suspect_sources": list(entry.get("suspect_sources", [])),
        "verdict": entry.get("verdict", "ok"),
    }


def _precheck_summary(records, precheck_results: list[dict]) -> dict:
    """体检汇总:存疑计数按来源前缀分桶 + 双检测器分划分成功率。

    suspect_counts 数的是"照片张数"(一张照片带同前缀的多条来源只计一次);
    detector_success_rates 是可见性探针的成败占比,分 train/test 与检测器,
    它顺带回答"经典检测器在这批数据的成功率"这个研究阶段留下的开放问题。
    """
    buckets = {"pose_relative": 0, "global": 0}
    n_suspect = 0
    splits: dict[str, dict[str, list[int]]] = {}
    for record, entry in zip(records, precheck_results):
        sources = list(entry.get("suspect_sources", []))
        if entry.get("verdict") == "suspect":
            n_suspect += 1
        if any(s.startswith("pose_relative:") for s in sources):
            buckets["pose_relative"] += 1
        if any(s.startswith("global:") for s in sources):
            buckets["global"] += 1
        slot = splits.setdefault(record.split, {"classic_ok": [], "sb_ok": []})
        slot["classic_ok"].append(bool(entry.get("classic_ok", False)))
        slot["sb_ok"].append(bool(entry.get("sb_ok", False)))

    rates: dict[str, dict[str, float | None]] = {}
    for split, slot in splits.items():
        total = len(slot["classic_ok"])
        rates[split] = {
            "classic": (sum(slot["classic_ok"]) / total) if total else None,
            "sb": (sum(slot["sb_ok"]) / total) if total else None,
        }
    return {
        "suspect_counts": {**buckets, "suspect_total": n_suspect},
        "detector_success_rates": rates,
    }


def write_manifest(records: tuple[ImageRecord, ...] | list[ImageRecord],
                   resolution: tuple[int, int],
                   warnings: tuple[str, ...] | list[str],
                   out_root: Path,
                   config_files: dict[str, Path],
                   precheck_results: list[dict] | None = None) -> Path:
    """写 manifest.json 到 output/ingest/run_<时间戳>/,返回该运行目录。

    参数:
        records:      scan_dataset 的照片记录
        resolution:   (宽, 高),来自 verify_consistent_resolution
        warnings:     扫描阶段的全部中文警告(留痕,不崩溃的承诺兑现处)
        out_root:     阶段输出根(如 output/ingest)
        config_files: 配置文件名 → 路径,写进血缘指纹
        precheck_results: 体检判定清单(与 records 等长同序,每条是指标数字 +
                      suspect_sources + verdict);缺省 None 表示本轮没做体检
                      (兼容旧入口),images 不带 precheck 键
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    round_id = f"run_{timestamp}"
    run_dir = Path(out_root) / round_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if precheck_results is not None:
        if len(precheck_results) != len(records):
            raise ValueError(
                f"体检结果条数({len(precheck_results)})与照片记录条数({len(records)})不一致,"
                f"无法逐张对应,请检查体检流程是否漏算或多算了照片。"
            )
        schema_note = (
            "images 每条带 precheck 对象:blur_raw/blur_norm/expo_hi/expo_lo(体检数字,"
            "缺失写 null)+ visibility(双检测器可见性)+ suspect_sources(判定来源逐条"
            "列明,pose_relative:*=同姿态连拍内比较,global:*=全局分位线/双检测器可见性)"
            "+ verdict(ok|suspect)。存疑只是标记,照片照常留在数据流中(D-01)。"
        )
    else:
        schema_note = "本轮未附带体检结论,images 不含 precheck 键(带体检的入口会逐张补齐)。"

    summary = summarize(records, resolution, warnings)
    if precheck_results is not None:
        summary.update(_precheck_summary(records, precheck_results))

    manifest = {
        # 血缘指纹:这份清单是什么时候、用什么配置算出来的(ITER-01)
        "lineage": make_lineage(config_files=config_files, upstream={}, round_id=round_id),
        "summary": summary,
        "schema_note": schema_note,
        "images": [
            {
                "image_id": r.image_id,
                "path": _relpath_for_manifest(r.path),
                "split": r.split,
                "pose_id": r.pose_id,
                "repeat_idx": r.repeat_idx,
                "n_repeats": r.n_repeats,
                **({"precheck": _precheck_object(entry)}
                   if precheck_results is not None else {}),
            }
            for r, entry in zip(records, precheck_results or [])
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
