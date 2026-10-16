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

from calib.corners.artifacts import CANONICAL_RULE_VERSION, save_corners
from calib.corners.canonical import burst_consistency
from calib.corners.detect import detect_classic, detect_sb
from calib.io.config import load_board, load_calibration
from calib.io.dataset import imread_checked, scan_dataset_dirs, verify_consistent_resolution
from calib.io.manifest import latest_run_dir, write_manifest
from calib.util.lineage import make_lineage

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


@app.command(help="① 照片入库体检:扫描 input_data,核对数据契约,产出带血缘指纹的数据清单 manifest.json(张数/姿态/分辨率全留痕)")
def ingest() -> None:
    """流水线第①步:把 705 张(张数数据驱动)照片登记造册。

    步骤:读配置 → 扫描 train/test 目录 → 逐张核对分辨率一致
    → 写 output/ingest/run_<时间戳>/manifest.json → 打印中文汇总。
    """
    board = load_board(BOARD_PATH)          # 棋盘规格(当前步骤只登记,角点检测用它)
    cfg = load_calibration(CALIBRATION_PATH)

    # 数据目录来自配置(相对当前工作目录),换数据集只改 calibration.yaml
    scan = scan_dataset_dirs(train_dir=Path(cfg.data.train), test_dir=Path(cfg.data.test))
    resolution = verify_consistent_resolution(scan.records)

    run_dir = write_manifest(
        records=scan.records,
        resolution=resolution,
        warnings=scan.warnings,
        out_root=OUTPUT_ROOT / "ingest",
        config_files={"board.yaml": BOARD_PATH, "calibration.yaml": CALIBRATION_PATH},
    )

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

    typer.echo("入库体检完成,数据清单已写盘。")
    typer.echo(f"  训练照片:{train} 张(姿态 {len(poses)} 个,每个姿态连拍张数分布 {dist})")
    typer.echo(f"  测试照片:{test} 张")
    typer.echo(f"  统一分辨率:{resolution[0]}x{resolution[1]}(宽x高)")
    typer.echo(f"  棋盘规格:{board.cols}x{board.rows} 内角点,方格 {board.square_size_mm}{board.unit}")
    typer.echo(f"  警告:{len(scan.warnings)} 条(全部已写进 manifest 留痕)")
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


@app.command(help="② 找棋盘角点:经典与新款两个检测器并跑、各存一份,亚像素窗口按实测方格边长比例推导")
def corners() -> None:
    """流水线第②步:在数据清单登记的全部照片上找角点。

    步骤:读配置 → 找最新一轮 ingest 产物读照片清单 → 逐张读图,
    经典与新款(SB)两颗检测器各跑一次(D-07)→ 各自规范化、按姿态分组
    做连拍一致性断言(D-09)→ 写 4 份角点 npz(train/test × classic/sb)
    与检测质量摘要 → 打印中文汇总。
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

    typer.echo(f"开始找角点:共 {n} 张照片 × 2 颗检测器(经典较慢,请稍等)……")
    for i, rec in enumerate(images):
        gray = imread_checked(Path(rec["path"]))
        for name, detect in detectors.items():
            result = detect(gray, board.cols, board.rows, cfg.corners)
            if result is None:
                continue  # 本帧该检测器未检出:照常占一行(NaN/False),不消失
            slot = arrays[name]
            slot["uv"][i] = result["uv"]
            slot["sq_px"][i] = result["sq_px"]
            slot["win_size"][i] = result["win_size"]
            slot["detected"][i] = True
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
            save_corners(
                out_dir, split, name,
                image_ids=[images[i]["image_id"] for i in rows],
                uv=slot["uv"][rows], sq_px=slot["sq_px"][rows],
                win_size=slot["win_size"][rows], detected=slot["detected"][rows],
                lineage=lineage, meta_extra=meta_extra,
            )

    # 检测质量摘要:分 划分 × 检测器 的成功率与方格/窗口统计 + 存疑帧清单
    detection: dict[str, dict] = {}
    for split in ("train", "test"):
        rows = np.array([i for i, rec in enumerate(images) if rec["split"] == split])
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
    typer.echo(f"  连拍一致性断言:每颗检测器检查 {groups_checked.get('classic', 0)} 组,"
               f"失败 {len(burst_failures)} 组"
               + ("(存疑帧已列入质量摘要,供人工审阅)" if burst_failures else ""))
    typer.echo(f"  角点产物:{out_dir}")
    typer.echo(f"  质量摘要:{summary_path}")


@app.command(help="③ 测噪声底:同一拍摄位置的连拍相减,实测这批数据天生的噪声水平")
def noise() -> None:
    _not_implemented(3, "测噪声底")


@app.command(help="④ 求解相机内参:多类相机模型并排求解,以 OpenCV 结果为对照基准")
def calibrate() -> None:
    _not_implemented(4, "求解相机内参")


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
    typer.echo("后续阶段(噪声/内参/理论极限/考试/归因/报告)由后续版本实现,"
               "run-all 到此为止。")


def main() -> None:
    """[project.scripts] 的入口函数。"""
    app()


if __name__ == "__main__":
    main()
