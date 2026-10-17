"""calib 命令行入口:九个子命令 = 整条流水线的通俗中文地图(D-11 / D-13)。

八个阶段命令按 docs/pipeline.md 的顺序排列:
① ingest 照片入库体检 → ② corners 找棋盘角点 → ③ noise 测噪声底
→ ④ calibrate 求解相机内参 → ⑤ fisher 算理论精度极限 → ⑥ evaluate 防作弊考试
→ ⑦ analyze 误差归因 → ⑧ report 汇总证据出报告;外加 run-all 一键跑已实现的全流程。

硬性规则(D-12):命令行不接受任何算法参数,全部参数从 config/ 读——
参数来源唯一,每次运行产物记录所用配置,结果才可追溯、可复现。
路径一律相对当前工作目录解析(./config、./input_data、./output),
这是"换数据集只改配置不改代码"的基础。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import typer

from calib.corners.artifacts import CANONICAL_RULE_VERSION, load_corners, save_corners
from calib.corners.canonical import burst_consistency
from calib.corners.detect import detect_classic, detect_sb
from calib.io.config import load_board, load_calibration
from calib.io.dataset import imread_checked, scan_dataset_dirs, verify_consistent_resolution
from calib.io.manifest import latest_run_dir, write_manifest, write_round_diff
from calib.io.precheck import compute_metrics, judge_suspects
from calib.solve.baseline import DETECTORS, run_baselines, save_calibration
from calib.util.lineage import make_lineage, sha256_file
from calib.visualization.mosaic import write_mosaics
from calib.visualization.overlay import (
    render_all,
    render_flagged_review,
    render_sb_failed_review,
)
from calib.visualization.suspects import write_suspect_originals


def manifest_suspect_counts(judged: list[dict]) -> dict:
    """按来源前缀分桶统计存疑照片张数(控制台汇总用;同一来源族只计一张)。"""
    counts = {"pose_relative": 0, "global": 0, "suspect_total": 0}
    for j in judged:
        if j.get("verdict") != "suspect":
            continue
        counts["suspect_total"] += 1
        if any(s.startswith("pose_relative:") for s in j.get("suspect_sources", [])):
            counts["pose_relative"] += 1
        if any(s.startswith("global:") for s in j.get("suspect_sources", [])):
            counts["global"] += 1
    return counts

app = typer.Typer(
    name="calib",
    help="相机内参标定与评估框架:输入标定照片,输出可论证为当前数据下最优的内参与证据链。"
         "子命令即流水线八步(顺序即执行顺序),run-all 一键跑完已实现的全流程。",
    no_args_is_help=True,
)

# 约定的路径(相对当前工作目录;数据目录不写死——从 calibration.yaml 的 data 段读)
BOARD_PATH = Path("config") / "board.yaml"
CALIBRATION_PATH = Path("config") / "calibration.yaml"
OUTPUT_ROOT = Path("output")


@app.callback()
def _setup_runtime() -> None:
    """每个命令执行前的统一准备:按配置限制 OpenCV 线程数。

    OpenCV 默认会占满所有核,批量处理(几百张图)时会把机器拖垮;
    线程数从 calibration.yaml 的 cv2_threads 读(0 = 关闭 OpenCV 自带多线程)。
    """
    try:
        cfg = load_calibration(CALIBRATION_PATH)
        cv2.setNumThreads(cfg.cv2_threads)
    except (ValueError, OSError):
        # --help / 配置还没就位时不拦着用户:真正需要配置的命令自己会再校验
        pass


def _not_implemented(step: int, name: str) -> None:
    """未实现阶段的占位输出:说清楚这是第几步,由后续阶段实现。"""
    typer.echo(f"本命令对应流水线第{'①②③④⑤⑥⑦⑧'[step - 1]}步「{name}」,由后续阶段实现。")


@app.command(help="① 照片入库体检:全体检——模糊度/曝光/棋盘可见性逐张打分,存疑只标记不剔除(D-01),判定来源逐条写进 manifest,存疑原图另存副本,并生成与上一轮的存疑对比表")
def ingest() -> None:
    """流水线第①步:全体检 + 登记造册(一张照片都不漏、一张都不剔)。

    步骤:读配置 → 扫描 train/test 目录 → 逐张核对分辨率一致
    → 逐张算体检指标(棋盘框内模糊度、曝光裁剪比例、双检测器可见性探针)
    → 双来源存疑判定(同姿态相对比较为主,全局分位线兜底,D-02)
    → 写 manifest(每张照片带全部体检数字与判定来源,D-06)
    → 存疑原图复制副本(拼图缩略图看不清细节)→ 跨轮存疑对比表(D-04)
    → 打印中文汇总。
    """
    board = load_board(BOARD_PATH)          # 棋盘规格(体检探针按它问"看没看见")
    cfg = load_calibration(CALIBRATION_PATH)

    # 数据目录来自配置(相对当前工作目录),换数据集只改 calibration.yaml
    scan = scan_dataset_dirs(train_dir=Path(cfg.data.train), test_dir=Path(cfg.data.test))
    resolution = verify_consistent_resolution(scan.records)

    # ---- 逐张体检(双检测器探针是主要开销,705 张为分钟级)----
    n_total = len(scan.records)
    typer.echo(f"开始全体检:共 {n_total} 张照片 × 双检测器可见性探针(分钟级,请稍等)……")
    metrics_list: list[dict] = []
    for k, rec in enumerate(scan.records):
        gray = imread_checked(rec.path)
        entry = {"image_id": rec.image_id, "split": rec.split,
                 "pose_id": rec.pose_id, "n_repeats": rec.n_repeats}
        entry.update(compute_metrics(gray, board.cols, board.rows, cfg.corners))
        metrics_list.append(entry)
        if (k + 1) % 100 == 0:
            typer.echo(f"  已体检 {k + 1}/{n_total} 张……")

    # ---- 双来源判定(只标记,输入几张输出几张,D-01/D-02)----
    judged = judge_suspects(metrics_list, cfg.ingest.precheck)
    merged = [{**m, **j} for m, j in zip(metrics_list, judged)]

    run_dir = write_manifest(
        records=scan.records,
        resolution=resolution,
        warnings=scan.warnings,
        out_root=OUTPUT_ROOT / "ingest",
        config_files={"board.yaml": BOARD_PATH, "calibration.yaml": CALIBRATION_PATH},
        precheck_results=merged,
    )

    # ---- 存疑原图副本(归可视化层)与跨轮对比表 ----
    copied = write_suspect_originals(
        [{"image_id": r.image_id, "path": r.path, "precheck": j}
         for r, j in zip(scan.records, judged)],
        run_dir,
    )
    diff_path, diff_counts = write_round_diff(run_dir, OUTPUT_ROOT / "ingest")

    # ---- 中文汇总 ----
    train = sum(1 for r in scan.records if r.split == "train")
    test = sum(1 for r in scan.records if r.split == "test")
    poses = {r.pose_id for r in scan.records if r.split == "train"}
    # 每姿态连拍张数分布:按姿态统计(若按照片统计会把 80 个姿态数成 640)
    per_pose: dict[str, int] = {}
    for r in scan.records:
        if r.split == "train":
            per_pose[r.pose_id] = r.n_repeats
    dist: dict[str, int] = {}
    for n in per_pose.values():
        dist[str(n)] = dist.get(str(n), 0) + 1

    counts = manifest_suspect_counts(merged)
    n_suspect = counts["suspect_total"]
    typer.echo("入库体检完成,数据清单(带全部体检数字)已写盘。")
    typer.echo(f"  训练照片:{train} 张(姿态 {len(poses)} 个,每个姿态连拍张数分布 {dist})")
    typer.echo(f"  测试照片:{test} 张")
    typer.echo(f"  统一分辨率:{resolution[0]}x{resolution[1]}(宽x高)")
    typer.echo(f"  棋盘规格:{board.cols}x{board.rows} 内角点,方格 {board.square_size_mm}{board.unit}")
    typer.echo(f"  警告:{len(scan.warnings)} 条(全部已写进 manifest 留痕)")
    typer.echo(f"  存疑照片:{n_suspect} 张(只标记不剔除,照常留在数据流中,D-01)")
    typer.echo(f"    按来源分桶:同姿态相对比较 {counts['pose_relative']} 张,"
               f"全局线兜底 {counts['global']} 张(判定来源逐条在 manifest)")
    for split in ("train", "test"):
        slot = [m for m in metrics_list if m["split"] == split]
        if not slot:
            continue
        rate_c = sum(1 for m in slot if m["classic_ok"]) / len(slot)
        rate_s = sum(1 for m in slot if m["sb_ok"]) / len(slot)
        typer.echo(f"    棋盘可见性({split}):经典 classic {rate_c * 100:.1f}%,"
                   f"新款 sb {rate_s * 100:.1f}%(探针只问看没看见,精化是角点阶段的事)")
    typer.echo(f"  存疑原图副本:{len(copied)} 张 → {run_dir / 'suspect_originals'}")
    if diff_counts is not None:
        typer.echo(f"  跨轮对比(与上一轮):新增存疑 {diff_counts['new_suspect']} 张 / "
                   f"解除存疑 {diff_counts['resolved']} 张 / "
                   f"持续存疑 {diff_counts['persistent']} 张")
    typer.echo(f"  跨轮对比表:{diff_path}")
    typer.echo(f"  数据清单:{run_dir / 'manifest.json'}")


def _stats_block(values: np.ndarray) -> dict | None:
    """检出帧的统计(min/中位数/max);一个都没检出时返回 None。"""
    if values.size == 0:
        return None
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _burst_failures_for(uv: np.ndarray, sq_px: np.ndarray, detected: np.ndarray,
                        images: list[dict], detector: str,
                        burst_cfg) -> tuple[list[dict], int]:
    """对一个检测器的 train 角点按姿态分组做连拍一致性断言(D-09)。

    返回 (失败清单, 检查了多少组)。test 每张独立、单张姿态不构成连拍,跳过。
    断言失败不抛异常:失败组里逐帧与"连拍内逐点中位数参照"比偏差,
    偏差超限的帧点名进清单(疑似镜像/翻转污染),供人工审阅,不自动剔除(D-01)。
    """
    groups: dict[str, list[int]] = {}
    for i, rec in enumerate(images):
        if rec["split"] == "train":
            groups.setdefault(rec["pose_id"], []).append(i)

    failures: list[dict] = []
    for pose_id in sorted(groups):
        idxs = groups[pose_id]
        det_idxs = [i for i in idxs if detected[i]]
        if len(det_idxs) < 2:
            continue  # 连拍里可用的帧不足 2 张,没法比,跳过
        uvs = uv[det_idxs]
        sq = float(np.median(sq_px[det_idxs]))
        result = burst_consistency(uvs, sq,
                                   median_tol=burst_cfg.median_tol,
                                   max_tol=burst_cfg.max_tol)
        if result["ok"]:
            continue
        reference = np.median(uvs, axis=0)
        deviation = np.linalg.norm(uvs - reference, axis=2).max(axis=1)
        suspect_local = [j for j in range(len(det_idxs))
                         if deviation[j] > result["threshold_px"][1]]
        suspect_idxs = [det_idxs[j] for j in suspect_local] or det_idxs
        failures.append({
            "detector": detector,
            "split": "train",
            "pose_id": pose_id,
            "suspect_image_ids": [images[i]["image_id"] for i in suspect_idxs],
            "spread_median_px": result["spread_median_px"],
            "spread_max_px": result["spread_max_px"],
            "threshold_px": result["threshold_px"],
            "note": "同编号角点在连拍内跳位,疑似排序被镜像/翻转污染;"
                    "已点名存疑供人工审阅,未剔除(D-01)。",
        })
    return failures, sum(1 for g in groups.values()
                         if sum(1 for i in g if detected[i]) >= 2)


# 体检存疑来源的中文翻译与 ASCII 缩写(记号格式 "前缀:指标"):
# 前缀说判定来自哪条线,指标说卡在哪一项
_SUSPECT_SOURCE_CN = {"pose_relative": "同姿态相对比较", "global": "全局线兜底"}
_SUSPECT_METRIC_CN = {"blur": "模糊度", "exposure": "曝光"}
_SUSPECT_METRIC_ASCII = {"blur": "BLUR", "exposure": "EXPO"}


def _suspect_review_frames(images: list[dict], arrays: dict) -> list[dict]:
    """从数据清单筛出体检判"存疑"的帧,组装 review_suspects 复核清单。

    每帧 info 同时保留存疑来源的原始记号(对账 manifest 用)与中文翻译,
    外加三个体检指标(blur_norm/expo_hi/expo_lo)原值——复核时一个
    文件夹看完,不用翻 manifest。角点取各检测器路线的最终结果:
    detected 为 False 时给 None;sb 槽位的 detected 天然含 classic 兜底帧
    (与 marked_copies/sb 同判据),不需要在这里特殊处理。
    记号若出现陌生取值,原样保留不吞掉——宁可露出英文也不丢信息。
    """
    frames = []
    for i, rec in enumerate(images):
        precheck = rec.get("precheck", {})
        if precheck.get("verdict") != "suspect":
            continue
        tokens = list(precheck.get("suspect_sources", []))
        reason_parts = []
        note_parts = []
        for token in tokens:
            prefix, _, metric = token.partition(":")
            reason_parts.append(
                f"{_SUSPECT_SOURCE_CN.get(prefix, prefix)}:"
                f"{_SUSPECT_METRIC_CN.get(metric, metric)}")
            note_parts.append(_SUSPECT_METRIC_ASCII.get(metric, metric.upper()))
        # 图上画不了中文,note 用 ASCII 缩写(cv2.putText 不支持中文字形);
        # 同一指标被两条线同时点名时缩写去重(如 BLUR+EXPO 而不是 BLUR+BLUR+EXPO)
        note = "SUS " + "+".join(dict.fromkeys(note_parts)) if note_parts else "SUS"
        frames.append({
            "image_id": rec["image_id"],
            "path": rec["path"],
            "uv_classic": (arrays["classic"]["uv"][i]
                           if arrays["classic"]["detected"][i] else None),
            "uv_sb": (arrays["sb"]["uv"][i]
                      if arrays["sb"]["detected"][i] else None),
            "note": note,
            "info": {
                "pose_id": rec["pose_id"],
                "repeat_idx": rec["repeat_idx"],
                "split": rec["split"],
                "suspect_sources": tokens,
                "reason": "存疑原因:" + "、".join(reason_parts),
                "metrics": {
                    "blur_norm": precheck.get("blur_norm"),
                    "expo_hi": precheck.get("expo_hi"),
                    "expo_lo": precheck.get("expo_lo"),
                },
            },
        })
    return frames


def _burst_review_frames(images: list[dict], arrays: dict,
                         burst_failures: list[dict]) -> list[dict]:
    """把连拍一致性断言点名的帧(两颗检测器并集)组装成复核清单。

    同一张照片可能被两颗检测器同时点名,只保留一条,flagged_by 列出全部
    点名者;reason 按检测器各写一条:组内波动中位数/最大值与各自容差的
    数值和对比结论。超限与否按数值现算(中位数 ≥ 中位容差 即"超限"),
    不抄断言时的旧结论,保证文字与数字永远一致。
    """
    flagged: dict[str, dict] = {}
    for failure in burst_failures:
        detector = failure["detector"]
        median_px, max_px = failure["spread_median_px"], failure["spread_max_px"]
        median_tol, max_tol = failure["threshold_px"]
        median_word = "超限" if median_px >= median_tol else "未超"
        max_word = "超限" if max_px >= max_tol else "未超"
        reason = (f"连拍断言失败:同名角点组内波动中位数 {median_px:.2f}px"
                  f"(容差 {median_tol:.2f}px,{median_word});"
                  f"最大 {max_px:.2f}px(容差 {max_tol:.2f}px,{max_word})")
        for image_id in failure["suspect_image_ids"]:
            entry = flagged.setdefault(image_id, {"flagged_by": [],
                                                  "reason": {}})
            entry["flagged_by"].append(detector)
            entry["reason"][detector] = reason

    # 按数据清单的原始顺序输出(并集去重后顺序稳定,重跑结果可对账)
    frames = []
    for i, rec in enumerate(images):
        info = flagged.get(rec["image_id"])
        if info is None:
            continue
        frames.append({
            "image_id": rec["image_id"],
            "path": rec["path"],
            "uv_classic": (arrays["classic"]["uv"][i]
                           if arrays["classic"]["detected"][i] else None),
            "uv_sb": (arrays["sb"]["uv"][i]
                      if arrays["sb"]["detected"][i] else None),
            "note": "BURST FLAG",
            "info": {
                "pose_id": rec["pose_id"],
                "repeat_idx": rec["repeat_idx"],
                "split": rec["split"],
                "flagged_by": info["flagged_by"],
                "reason": info["reason"],
            },
        })
    return frames


@app.command(help="② 找棋盘角点:经典与新款两个检测器并跑、各存一份,"
                  "SB 未检出时用经典检测器兜底(可在配置关闭),"
                  "亚像素窗口按实测方格边长比例推导;"
                  "存疑帧与连拍点名帧各归拢一个专属复核文件夹"
                  "(原图副本 + 两颗检测器标记图 + 逐张中文原因)")
def corners() -> None:
    """流水线第②步:在数据清单登记的全部照片上找角点。

    步骤:读配置 → 找最新一轮 ingest 产物读照片清单 → 逐张读图,
    经典与新款(SB)两颗检测器各跑一次(D-07);SB 未检出的帧按配置开关
    用经典检测器的结果兜底填进 SB 槽位,逐帧来源记在 source 数组 →
    各自规范化、按姿态分组做连拍一致性断言(D-09)
    → 写 4 份角点 npz(train/test × classic/sb)与检测质量摘要
    → 全量标记副本(每张照片两套,D-06/D-07)
    → SB 失败帧独立证据 review_sb_failed/(原图副本 + 标记图)
    → 体检存疑帧复核文件夹 review_suspects/ 与连拍点名帧复核文件夹
    review_burst_flagged/(原图副本 + 两颗检测器标记图 + 逐张中文原因,
    人工一个文件夹看完,不跨目录翻找)
    → 汇总拼图两张 + 逐格索引(只是索引不是替代,D-06)→ 打印中文汇总。
    """
    started = datetime.now(UTC)
    board = load_board(BOARD_PATH)
    cfg = load_calibration(CALIBRATION_PATH)

    # 上游产物:最新一轮入库清单(文件是唯一真相源,阶段间只靠文件通信)
    ingest_run = latest_run_dir(OUTPUT_ROOT, "ingest")
    with open(ingest_run / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    images: list[dict] = manifest["images"]

    timestamp = started.strftime("%Y%m%d_%H%M%S")
    round_id = f"run_{timestamp}"
    out_dir = OUTPUT_ROOT / "corners" / round_id
    lineage = make_lineage(
        config_files={"board.yaml": BOARD_PATH, "calibration.yaml": CALIBRATION_PATH},
        upstream={"manifest.json": ingest_run / "manifest.json"},
        round_id=round_id,
    )

    n = len(images)
    detectors = {"classic": detect_classic, "sb": detect_sb}
    arrays = {
        name: {
            "uv": np.full((n, board.cols * board.rows, 2), np.nan),
            "sq_px": np.full(n, np.nan),
            "win_size": np.full(n, np.nan),
            "detected": np.zeros(n, dtype=bool),
        }
        for name in detectors
    }

    # SB 生产路线的逐帧来源留痕:"sb" = SB 本体检出;"classic_fallback" =
    # SB 未检出、经典检测器结果兜底填入;"" = 两法都未检出(兜底关闭时
    # SB 缺失帧也是 "")。经典路线没有兜底概念,不记来源、产物保持纯净。
    sources = np.full(n, "", dtype="<U16")

    typer.echo(f"开始找角点:共 {n} 张照片 × 2 颗检测器(经典较慢,请稍等)……")
    for i, rec in enumerate(images):
        gray = imread_checked(Path(rec["path"]))
        # 经典先跑:结果留在局部变量,SB 未检出时按开关兜底复用。
        # 未检出的帧照常占一行(NaN/False),绝不从数据里消失(D-01)
        classic_result = detect_classic(gray, board.cols, board.rows, cfg.corners)
        if classic_result is not None:
            slot = arrays["classic"]
            slot["uv"][i] = classic_result["uv"]
            slot["sq_px"][i] = classic_result["sq_px"]
            slot["win_size"][i] = classic_result["win_size"]
            slot["detected"][i] = True

        sb_result = detect_sb(gray, board.cols, board.rows, cfg.corners)
        sb = arrays["sb"]
        if sb_result is not None:
            sb["uv"][i] = sb_result["uv"]
            sb["sq_px"][i] = sb_result["sq_px"]
            sb["win_size"][i] = sb_result["win_size"]
            sb["detected"][i] = True
            sources[i] = "sb"
        elif classic_result is not None and cfg.corners.sb_fallback_classic:
            # SB 未检出、兜底开关开、classic 本帧有结果:classic 的角点连同
            # 方格边长/精化窗口一起填进 SB 槽位——对下游而言这帧"SB 路线
            # 已检出",真实来源单独记在 sources 里,绝不与 SB 本体检出混淆
            sb["uv"][i] = classic_result["uv"]
            sb["sq_px"][i] = classic_result["sq_px"]
            sb["win_size"][i] = classic_result["win_size"]
            sb["detected"][i] = True
            sources[i] = "classic_fallback"
        if (i + 1) % 100 == 0:
            typer.echo(f"  已处理 {i + 1}/{n} 张……")

    # 连拍一致性断言:两颗检测器各自做(同姿态分组,test 每张独立跳过)
    burst_failures: list[dict] = []
    groups_checked: dict[str, int] = {}
    for name in detectors:
        slot = arrays[name]
        fails, checked = _burst_failures_for(
            slot["uv"], slot["sq_px"], slot["detected"], images, name,
            cfg.corners.burst_check)
        burst_failures.extend(fails)
        groups_checked[name] = checked

    # 落盘:4 份 npz(train/test × classic/sb)+ 检测质量摘要
    meta_extra = {
        "classic_flags": list(cfg.corners.classic_flags),
        "sb_flags": cfg.corners.sb_flags,
        "subpix": {
            "window_granularity": cfg.corners.subpix.window_granularity,
            "min_win": cfg.corners.subpix.min_win,
            "max_iter": cfg.corners.subpix.max_iter,
            "epsilon": cfg.corners.subpix.epsilon,
        },
        "burst_check": {
            "median_tol": cfg.corners.burst_check.median_tol,
            "max_tol": cfg.corners.burst_check.max_tol,
        },
    }
    for split in ("train", "test"):
        rows = [i for i, rec in enumerate(images) if rec["split"] == split]
        for name in detectors:
            slot = arrays[name]
            source_arg = None
            extra = meta_extra
            if name == "sb":
                # 仅 SB 路线带 source 数组与兜底留痕;classic 的调用一字不变
                source_arg = sources[rows]
                extra = {
                    **meta_extra,
                    "sb_fallback_classic": cfg.corners.sb_fallback_classic,
                    "n_classic_fallback": int((source_arg == "classic_fallback").sum()),
                }
            save_corners(
                out_dir, split, name,
                image_ids=[images[i]["image_id"] for i in rows],
                uv=slot["uv"][rows], sq_px=slot["sq_px"][rows],
                win_size=slot["win_size"][rows], detected=slot["detected"][rows],
                source=source_arg,
                lineage=lineage, meta_extra=extra,
            )

    # 检测质量摘要:分 划分 × 检测器 的成功率与方格/窗口统计 + 存疑帧清单
    detection: dict[str, dict] = {}
    for split in ("train", "test"):
        # dtype=int 必须显式写:"只有 train 没有 test"是数据层承诺的合法输入,
        # 空列表会被 numpy 推断成 float64 数组,拿它做花式索引当场 IndexError,
        # 下面 rows.size 的 success_rate=None 保护根本没机会生效
        rows = np.array([i for i, rec in enumerate(images)
                         if rec["split"] == split], dtype=int)
        detection[split] = {}
        for name in detectors:
            slot = arrays[name]
            ok = slot["detected"][rows]
            n_det = int(ok.sum())
            detection[split][name] = {
                "n_images": int(rows.size),
                "n_detected": n_det,
                "success_rate": (n_det / rows.size) if rows.size else None,
                "sq_px": _stats_block(slot["sq_px"][rows][ok]),
                "win_size": _stats_block(slot["win_size"][rows][ok].astype(float)),
            }
            if name == "sb":
                # SB 段三口径:n_detected 是生产口径(detected 数组求和,天然含
                # 兜底帧);另报纯 SB 与 classic 兜底张数,来源可对账。
                # 兜底开关关闭时两字段照报(n_pure_sb == n_detected、兜底 0),
                # 口径稳定不随开关分支
                split_sources = sources[rows]
                detection[split]["sb"]["n_pure_sb"] = int((split_sources == "sb").sum())
                detection[split]["sb"]["n_classic_fallback"] = \
                    int((split_sources == "classic_fallback").sum())

    summary = {
        "lineage": lineage,
        "board": {"cols": board.cols, "rows": board.rows},
        "canonical_rule": CANONICAL_RULE_VERSION,
        "detection": detection,
        "burst_check": {
            "n_groups_checked": groups_checked,
            "n_failures": len(burst_failures),
            "failures": burst_failures,
        },
    }
    summary_path = out_dir / "detection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 全量标记副本(D-06/D-07):manifest 里每张照片、每颗检测器各一份,
    # 未检出画 MISS 红框留痕;重跑 corners 即整目录重新生成,幂等
    copy_summary = render_all(out_dir, ingest_run / "manifest.json", board, cfg)

    # SB 失败帧独立证据(设计约定"提取效果不好的图像,原图 + 标记图单独存"):
    # SB 本体未检出的帧逐张传入——classic 兜底帧带 classic 角点画叠加,
    # 两法都未检出的帧画 MISS 红框;没有失败帧则不建目录
    review_frames = [
        {
            "image_id": rec["image_id"],
            "path": rec["path"],
            "uv_or_none": (arrays["classic"]["uv"][i]
                           if sources[i] == "classic_fallback" else None),
        }
        for i, rec in enumerate(images)
        if sources[i] in ("classic_fallback", "")
    ]
    review_summary = render_sb_failed_review(
        out_dir, review_frames, board, quality=int(cfg.visualization.jpeg_quality))

    # 两个专属复核文件夹(与 review_sb_failed 同构):体检存疑帧、连拍点名帧
    # 各归拢一处——原图副本 + 两颗检测器标记图 + 逐张中文原因,人工复核
    # 不再跨目录按文件名翻找;清单为空时不建目录
    suspect_frames = _suspect_review_frames(images, arrays)
    suspect_review_summary = render_flagged_review(
        out_dir, "review_suspects", suspect_frames, board,
        quality=int(cfg.visualization.jpeg_quality),
        note="本文件夹由 calib corners 自动生成,汇总体检判\"存疑\"的帧"
             "(manifest 的 precheck.verdict == \"suspect\",只标记不剔除,D-01)。"
             "original/ 存原图副本(逐字节复制,源文件未改动);"
             "marked_classic/ 与 marked_sb/ 是两颗检测器各自的角点标记图"
             "(未检出画 MISS 红框留痕)。frames 数组逐张写明中文判定原因与"
             "体检指标原值,suspect_sources 保留原始记号,可与 manifest 对账。",
        extra=manifest_suspect_counts(
            [rec.get("precheck", {}) for rec in images]))

    burst_frames = _burst_review_frames(images, arrays, burst_failures)
    burst_poses = sorted({f["pose_id"] for f in burst_failures})
    burst_review_summary = render_flagged_review(
        out_dir, "review_burst_flagged", burst_frames, board,
        quality=int(cfg.visualization.jpeg_quality),
        note="本文件夹由 calib corners 自动生成,汇总连拍一致性断言点名的帧"
             "(两颗检测器各自的点名清单取并集,只点名不剔除,D-01)。"
             "original/ 存原图副本(逐字节复制,源文件未改动);"
             "marked_classic/ 与 marked_sb/ 是两颗检测器各自的角点标记图。"
             "frames 数组逐张写明每颗点名检测器的组内波动中位数/最大值与"
             "容差对比结论,数值与 detection_summary.json 的 burst_check 一致。",
        extra={"n_poses": len(burst_poses), "flagged_poses": burst_poses})

    # 汇总拼图(D-06):总览 + 存疑两张 + 逐格索引,只是索引不是替代;
    # 存疑原图副本在 ingest 运行目录,显式传入(不靠 mosaic 内部反推)
    mosaic_info = write_mosaics(out_dir, manifest, cfg, ingest_run_dir=ingest_run)

    # 中文汇总:两颗检测器成功率对比表 + 方格/窗口量级 + 断言结论 + 产物位置
    typer.echo("找角点完成,4 份角点文件与质量摘要已写盘。")
    typer.echo("  检出成功率对比(检出 / 总数):")
    typer.echo(f"    {'划分':<6}{'经典 classic':<22}{'新款 sb':<22}")
    for split in ("train", "test"):
        cells = []
        for name in ("classic", "sb"):
            stat = detection[split][name]
            rate = stat["success_rate"]
            cells.append(f"{stat['n_detected']}/{stat['n_images']}"
                         f" ({rate * 100:.1f}%)" if rate is not None else "无照片")
        typer.echo(f"    {split:<6}{cells[0]:<22}{cells[1]:<22}")
    for name in ("classic", "sb"):
        sq = detection["train"][name]["sq_px"] or detection["test"][name]["sq_px"]
        if sq:
            typer.echo(f"  {name} 方格像素边长(检出帧):中位数 {sq['median']:.1f}px"
                       f"(范围 {sq['min']:.1f} ~ {sq['max']:.1f})")
    n_pure_sb = int((sources == "sb").sum())
    n_fallback = int((sources == "classic_fallback").sum())
    fallback_state = "开" if cfg.corners.sb_fallback_classic else "关"
    typer.echo(f"  SB 生产口径(兜底开关:{fallback_state}):纯 SB {n_pure_sb} 张"
               f" + classic 兜底 {n_fallback} 张,合计 {n_pure_sb + n_fallback}/{n} 张"
               f"(逐帧来源在 corners_*_sb.npz 的 source 数组,meta 记兜底张数)")
    if review_summary is not None:
        typer.echo(f"  SB 失败帧独立证据:原图副本 + 标记图各 {review_summary['n_total']} 张"
                   f"(classic 兜底 {review_summary['n_classic_fallback']} 张、"
                   f"两法都未检出 {review_summary['n_both_missed']} 张)"
                   f" → {out_dir / 'review_sb_failed'}")
    else:
        typer.echo("  SB 失败帧独立证据:无 SB 未检出帧,未生成 review_sb_failed/")
    if suspect_review_summary is not None:
        typer.echo(f"  存疑帧复核文件夹:原图副本 + classic/sb 标记图各 "
                   f"{suspect_review_summary['n_total']} 张"
                   f"(来源分桶:同姿态相对比较 {suspect_review_summary['pose_relative']} 张、"
                   f"全局线兜底 {suspect_review_summary['global']} 张;"
                   f"逐张中文原因在 summary.json)→ {out_dir / 'review_suspects'}")
    else:
        typer.echo("  存疑帧复核文件夹:无体检存疑帧,未生成 review_suspects/")
    if burst_review_summary is not None:
        typer.echo(f"  连拍点名帧复核文件夹:原图副本 + classic/sb 标记图各 "
                   f"{burst_review_summary['n_total']} 张"
                   f"(点名姿态 {burst_review_summary['n_poses']} 个;"
                   f"逐张中文原因在 summary.json)→ {out_dir / 'review_burst_flagged'}")
    else:
        typer.echo("  连拍点名帧复核文件夹:无连拍断言点名帧,未生成 review_burst_flagged/")
    typer.echo(f"  连拍一致性断言:每颗检测器检查 {groups_checked.get('classic', 0)} 组,"
               f"失败 {len(burst_failures)} 组"
               + ("(存疑帧已列入质量摘要,供人工审阅)" if burst_failures else ""))
    typer.echo(f"  全量标记副本:{copy_summary['classic']['total']} 张 × 2 套"
               f"(classic/sb 各一套,人工并排对比哪颗检测器贴得更准,D-06/D-07)")
    for name in ("classic", "sb"):
        typer.echo(f"    {name}:标记 {copy_summary[name]['marked']} 张 / "
                   f"MISS 红框 {copy_summary[name]['miss']} 张(未检出也留痕)")
    typer.echo(f"  标记副本目录:{out_dir / 'marked_copies'}")
    typer.echo(f"  汇总拼图(mosaic/,只是索引不是替代,D-06):"
               f"总览 {mosaic_info['overview_cells']} 格"
               f"(常态抽样 + 最差 {cfg.visualization.mosaic.worst_n} + 全部存疑)"
               f" / 存疑 {mosaic_info['suspect_cells']} 格(未标记原貌)")
    typer.echo(f"  拼图与逐格索引:{out_dir / 'mosaic'}"
               f"(格号对应关系在 mosaic_index.json,细节回单张副本放大看)")
    typer.echo(f"  角点产物:{out_dir}")
    typer.echo(f"  质量摘要:{summary_path}")


@app.command(help="③ 测噪声底:同一拍摄位置的连拍相减,实测这批数据天生的噪声水平")
def noise() -> None:
    _not_implemented(3, "测噪声底")


def _manifest_path_for(corners_run: Path) -> Path:
    """定位与角点产物同一轮的数据清单 manifest.json。

    优先用角点 meta 里记录的上游清单路径(血缘链闭合:标定对齐的正是
    角点当时用的那份清单,即便之后又跑过新的 ingest 也不串轮);
    记录缺失或文件不在了,才退回最新一轮 ingest 产物。
    """
    meta_path = corners_run / "corners_train_classic.meta.json"
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        recorded = (meta.get("lineage", {}).get("upstream", {})
                    .get("manifest.json", {}).get("path"))
        if recorded and Path(recorded).is_file():
            return Path(recorded)
    return latest_run_dir(OUTPUT_ROOT, "ingest") / "manifest.json"


@app.command(help="④ 求解相机内参:两颗检测器的角点 × 针孔/鱼眼两类模型 = 4 个基线并排出结果")
def calibrate() -> None:
    """流水线第④步:在 train 角点上跑 2×2 = 4 个基线标定(D-10)。

    步骤:读配置 → 找最新一轮 corners 产物,读两份 train 角点
    (test 65 张留给后续阶段的独立评分,本步不碰)→ 按人工剔除清单
    逻辑剔除(D-03)→ 四组合循环求解 → 每组合写 calibration.json 与
    residuals.npz → 汇总写 comparison.json → 打印中文对比表。
    """
    started = datetime.now(UTC)
    board = load_board(BOARD_PATH)
    cfg = load_calibration(CALIBRATION_PATH)

    corners_run = latest_run_dir(OUTPUT_ROOT, "corners")
    with open(_manifest_path_for(corners_run), encoding="utf-8") as f:
        manifest = json.load(f)
    width, height = manifest["summary"]["resolution"]

    corners_train = {
        detector: load_corners(corners_run / f"corners_train_{detector}.npz")
        for detector in DETECTORS
    }
    typer.echo(f"开始基线标定:2×2 = 4 个组合(corners run:{corners_run.name},"
               f"图像 {width}x{height})……")
    outcome = run_baselines(corners_train, (int(width), int(height)), cfg, board)

    round_id = f"run_{started.strftime('%Y%m%d_%H%M%S')}"
    run_dir = OUTPUT_ROOT / "calibration" / round_id

    # 每组合各写产物;血缘里记下它消费的那份角点文件路径与指纹(T-03-01),
    # 以及本轮实际被人工剔除的照片名单(T-03-02,谁剔了什么可审计)
    comparison = []
    for detector in DETECTORS:
        npz_path = corners_run / f"corners_train_{detector}.npz"
        for model in ("pinhole", "fisheye"):
            name = f"baseline_{detector}_{model}"
            result = outcome["baselines"][name]
            lineage = make_lineage(
                config_files={"board.yaml": BOARD_PATH,
                              "calibration.yaml": CALIBRATION_PATH},
                upstream={npz_path.name: npz_path},
                round_id=round_id,
            )
            lineage["corners_file"] = str(npz_path)
            lineage["corners_sha256"] = sha256_file(npz_path)
            lineage["excluded_images"] = outcome["excluded_images"]
            save_calibration(run_dir, name, result, lineage)
            comparison.append({
                "name": name,
                "n_views": len(result["image_ids"]),
                "rms_overall": result["rms_overall"],
                "converged": result["converged"],
                "failure_note": result["failure_note"],
            })

    comparison_path = run_dir / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 中文对比表:四组合并排,一眼看出哪组最优、哪组失败(D-10 的直接答案)
    typer.echo("基线标定完成,2×2 = 4 个组合并排已写盘。")
    typer.echo(f"  {'组合':<28}{'视图数':<8}{'整体RMS(像素)':<16}{'收敛':<6}")
    for row in comparison:
        rms_text = f"{row['rms_overall']:.4f}" if row["rms_overall"] is not None else "—"
        flag = "是" if row["converged"] else f"否({row['failure_note'][:24]}……)"
        typer.echo(f"  {row['name']:<28}{row['n_views']:<8}{rms_text:<16}{flag}")
    typer.echo("  白话:鱼眼(fisheye)模型假设镜头按'等距投影'成像——入射角越大,"
               "成像点离中心按角度均匀拉开;普通广角镜头往往不合这个假设,"
               "求解可能不收敛或误差很大,这属预期内的正常发现,记录下来不中断。")
    if outcome["excluded_images"]:
        typer.echo(f"  人工剔除清单生效:本轮剔除 {len(outcome['excluded_images'])} 张"
                   f"(名单已记进每个产物的血缘)")
    typer.echo(f"  标定产物:{run_dir}")
    typer.echo(f"  对比摘要:{comparison_path}")


@app.command(help="⑤ 算理论精度极限:标定前预测这批数据最好能做到什么程度,并检查摆拍位置是否有缺陷(Fisher 信息/CRLB)")
def fisher() -> None:
    _not_implemented(5, "算理论精度极限")


@app.command(help="⑥ 防作弊考试:用没见过的新题考试、换着分组复测、逐一抽查,用数据挑出最合适的相机模型")
def evaluate() -> None:
    _not_implemented(6, "防作弊考试")


@app.command(help="⑦ 误差归因:实际误差 vs 理论极限,差在哪个方向、哪一环、还有没有优化空间")
def analyze() -> None:
    _not_implemented(7, "误差归因")


@app.command(help="⑧ 汇总证据出报告:每一步的结论、图表、数据全部留痕可追溯")
def report() -> None:
    _not_implemented(8, "汇总证据出报告")


@app.command(help="run-all:一条命令按流水线顺序跑完已实现的全流程(未实现的阶段会说明)")
def run_all() -> None:
    """按顺序执行已实现的阶段;未实现的阶段打印占位说明,不中断。"""
    typer.echo("run-all:按流水线顺序执行已实现的阶段……")
    typer.echo("")
    ingest()
    typer.echo("")
    corners()
    typer.echo("")
    calibrate()
    typer.echo("")
    typer.echo("后续阶段(噪声/理论极限/考试/归因/报告)由后续版本实现,"
               "run-all 到此为止。")


def main() -> None:
    """[project.scripts] 的入口函数。"""
    app()


if __name__ == "__main__":
    main()
