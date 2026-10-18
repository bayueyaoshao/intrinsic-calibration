"""受控注入实验:往真实照片上加已知强度的高斯像素噪声后重跑定版检测器,
与该照片的干净检测(基线)逐角点比对,量出"像素噪声 σ 被放大成多大的
角点定位误差"(传递曲线的原始数据)。

分工(检测逻辑唯一来源):本模块只负责加噪、调度、留痕落盘;检测一律走
calib.corners.detect 的 detect_sb / detect_classic(阶段一定版配置原样传入),
绝不复制第二份检测实现(研究职责表 / T-03-01-05)。

可复现承诺(不落盘合成图像的正当性来源):每个加噪副本的随机种子由
(图像, σ 档, 重复) 三元组从 base_seed 确定性推导——同种子完整重跑
逐位一致。保真规则(D-13)约束的是测量结果:逐张逐档逐重复的角点位移
数组全量落盘(injection_displacements.npz,纯数值数组无 pickle)。

部分成功约定(研究 Pitfall 7):个别照片读图失败、检测失败都只是"数据"
不是"事故"——该副本位移记 NaN、失败计数加一、来源留痕,整体照常跑完。
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from calib.corners.detect import detect_classic, detect_sb
from calib.io.config import CalibrationConfig, NoiseConfig
from calib.io.dataset import imread_checked
from calib.noise.immerkaer import immerkaer_sigma

# Immerkaer 主线口径名 → immerkaer_sigma 返回字典里的键
_VARIANT_KEYS = {"std": "sigma_std", "robust": "sigma_robust"}


def _seed_for(image_id: str, level_value: float, rep: int, base_seed: int) -> int:
    """从 (图像, σ 档, 重复) 三元组推导该副本的随机种子。

    公式:base_seed + sha1(f"{image_id}|{level}|{rep}") mod 2**32——
    确定性散列,任何一元不同种子就不同;同三元组永远同种子。
    这就是"合成图像不落盘也能精确复现"的正当性来源:复现只需
    原图 + 配置里的 base_seed,不需要任何落盘的加噪副本。
    """
    digest = hashlib.sha1(
        f"{image_id}|{level_value!r}|{rep}".encode("utf-8")).hexdigest()
    return (base_seed + int(digest, 16)) % 2**32


def add_gaussian_noise(img_u8: np.ndarray, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """给灰度图加高斯噪声,返回 uint8 加噪副本。

    加噪步骤刻意模仿真实采集链路:float 域加噪 → 截到 [0, 255] →
    四舍五入回 uint8。**最后一步量化不能省**——真实相机输出的就是
    量化后的整数灰阶,跳过量化直接把 float 图喂检测器,量出来的
    传递曲线与真实链路对不上(研究 Pitfall 4)。

    参数:
        img_u8: 二维灰度图(uint8);sigma: 噪声标准差(灰阶);
        rng:    numpy 随机数生成器(种子由调用方按三元组推导)。
    """
    arr = np.asarray(img_u8)
    if arr.ndim != 2:
        raise AssertionError(
            f"add_gaussian_noise 的输入必须是二维灰度图;实际是 {arr.ndim} 维,"
            f"形状 {arr.shape}。请用 cv2.imread 的 IMREAD_GRAYSCALE 读图。")
    noisy = arr.astype(np.float64) + rng.normal(0.0, sigma, size=arr.shape)
    np.clip(noisy, 0.0, 255.0, out=noisy)
    return np.rint(noisy).astype(np.uint8)


def _anchor_grid(sigma_values: list[float], cfg: NoiseConfig) -> list[float]:
    """按配置组装 σ 网格:σ=0 参照档 + 对数网格,返回完整档位列表。

    配置写了 injection_sigma_min / injection_sigma_max(> 0 的数)就用写死值;
    留空(None)则按实测中位数 σ 自动锚定:下限 max(0.25, σ_med/4)、
    上限 min(128, σ_med×32)——覆盖实测噪声邻域,并向上延伸到检测器失效区。
    返回的网格原样进产物 meta(曲线怎么取的档,证据链要能对账)。
    """
    sigma_med = float(np.median(np.asarray(sigma_values, dtype=np.float64)))
    if not np.isfinite(sigma_med):
        raise AssertionError(
            f"锚定 σ 网格需要至少一张图的有效 Immerkaer 估计;实际中位数算出 "
            f"{sigma_med!r}。请检查输入图像是否全部读取失败。")
    if cfg.injection_sigma_min is not None and cfg.injection_sigma_max is not None:
        lo, hi = float(cfg.injection_sigma_min), float(cfg.injection_sigma_max)
        auto = False
    else:
        lo = max(0.25, sigma_med / 4.0)
        hi = min(128.0, sigma_med * 32.0)
        auto = True
    if hi < lo:
        raise AssertionError(
            f"σ 网格上限({hi!r})小于下限({lo!r}),对数网格没有跨度。"
            f"请检查 noise.injection_sigma_min / injection_sigma_max 配置。")
    levels = np.geomspace(lo, hi, cfg.injection_n_levels)
    return [0.0] + [float(x) for x in levels]


def _rec_get(rec, key: str):
    """从照片记录取字段:清单字典(CLI 传)与 ImageRecord(测试传)都认。"""
    if isinstance(rec, dict):
        return rec[key]
    return getattr(rec, key)


def _inject_one_image(task: tuple) -> dict:
    """单张照片的全部注入副本:读图一次,内部跑完全部 σ 档 × 重复。

    子进程入口:参数打包成一个元组(全部可 pickle),返回字典
    (numpy 数组,可 pickle)。检测直接调定版的 detect_sb / detect_classic,
    配置对象原样传入——本函数绝不自己写检测逻辑。
    """
    (image_id, path_str, uv_clean, grid, n_repeats, base_seed,
     corners_cfg, allow_fallback, cols, rows) = task
    gray = imread_checked(Path(path_str))  # 读图失败抛中文错,由上层按副本失败记
    n_points = uv_clean.shape[0]
    disp = np.full((len(grid), n_repeats, n_points, 2), np.nan, dtype=np.float64)
    src = np.full((len(grid), n_repeats), "", dtype="<U16")
    n_fail = 0
    n_fallback = 0
    for li, sigma in enumerate(grid):
        for rep in range(n_repeats):
            if sigma > 0.0:
                rng = np.random.default_rng(_seed_for(image_id, sigma, rep, base_seed))
                noisy = add_gaussian_noise(gray, float(sigma), rng)
            else:
                # σ=0 参照档:不加噪直接重检干净原图,用于校验基线对齐
                noisy = gray
            result = detect_sb(noisy, cols, rows, corners_cfg)
            if result is not None:
                src[li, rep] = "sb"
            elif allow_fallback:
                result = detect_classic(noisy, cols, rows, corners_cfg)
                if result is not None:
                    src[li, rep] = "classic_fallback"
                    n_fallback += 1
            if result is None:
                n_fail += 1  # 两法都没检出:该副本留 NaN,是数据不是事故
                continue
            disp[li, rep] = result["uv"] - uv_clean
    return {"image_id": image_id, "disp": disp, "src": src,
            "n_fail": n_fail, "n_fallback": n_fallback}


def _per_image_level_rms(disp_level: np.ndarray) -> float | None:
    """一张图一个 σ 档的方均根欧氏误差(只算检出的副本);全失败返回 None。"""
    d = disp_level[np.isfinite(disp_level[:, :, 0])]
    if d.shape[0] == 0:
        return None
    return float(np.sqrt((d ** 2).sum(axis=-1).mean()))


def run_injection_experiment(
    corners_npz_path: Path | str,
    image_records,
    cfg: CalibrationConfig,
    cols: int,
    rows: int,
    out_dir: Path | str,
    progress: Callable[[dict], None] | None = None,
    lineage: dict | None = None,
) -> dict:
    """受控注入实验主入口:加噪 → 重检 → 逐角点位移 → 全保真落盘。

    参数:
        corners_npz_path: 基线角点 npz(阶段一 SB 路线产物,allow_pickle=False 读回);
        image_records:    照片记录清单(manifest["images"] 字典或 ImageRecord 均可),
                          按配置 noise.noise_splits 过滤后进实验;
        cfg:              完整标定配置——注入参数在 cfg.noise,检测配置在
                          cfg.corners(计划签名写 cfg: NoiseConfig,但检测复用
                          必须拿到 CornersConfig,整包传入才是 D-12 的单一来源);
        cols / rows:      棋盘内角点列数 / 行数(检测器要按这个形状找角点);
        out_dir:          产物目录(由调用方组装,本函数不越出它写任何文件);
        progress:         进度回调(库层不依赖 typer,播报由 CLI 注入);
        lineage:          可选血缘指纹(由 CLI 组装后传入,写进产物 meta)。

    返回:
        {"grid", "displacements_path", "per_image_path", "meta"}。

    产物:
        injection_displacements.npz:位移数组 (N, L, R, P, 2) float64 + 网格/
        编号/检出掩码/来源/逐张 Immerkaer σ,全部纯数值或定长文字,无 pickle;
        injection_per_image.json:逐张逐档统计 + 失败/兜底记录(D-13 全保真,
        npz 是原始数据本体,json 是逐张索引)。
    """
    noise_cfg = cfg.noise
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 基线角点读回:allow_pickle=False 是项目既定防线(npz 的 pickle 模式
    # 可执行任意代码,T-03-01-02)
    with np.load(Path(corners_npz_path), allow_pickle=False) as data:
        npz_ids = [str(x) for x in np.asarray(data["image_ids"]).ravel()]
        uv_rows = np.asarray(data["uv"], dtype=np.float64)
        detected_rows = np.asarray(data["detected"], dtype=bool)
    baseline = {image_id: (uv_rows[i], bool(detected_rows[i]))
                for i, image_id in enumerate(npz_ids)}

    selected = [rec for rec in image_records
                if _rec_get(rec, "split") in noise_cfg.noise_splits]
    if not selected:
        raise ValueError(
            f"注入实验没有可跑的照片:记录清单共 {len(list(image_records))} 条,"
            f"按 noise.noise_splits = {list(noise_cfg.noise_splits)} 过滤后为空。"
            f"请检查配置的划分名单与照片清单的 split 字段是否对得上。")

    # 第一步:主进程先对全部照片算 Immerkaer σ(网格锚定要用全量中位数)
    immerkaer_std = np.full(len(selected), np.nan)
    immerkaer_robust = np.full(len(selected), np.nan)
    for i, rec in enumerate(selected):
        gray = imread_checked(Path(_rec_get(rec, "path")))
        imm = immerkaer_sigma(gray)
        immerkaer_std[i] = imm["sigma_std"]
        immerkaer_robust[i] = imm["sigma_robust"]
    variant_key = _VARIANT_KEYS.get(noise_cfg.immerkaer_variant)
    if variant_key is None:
        raise AssertionError(
            f"immerkaer_variant 配置不认识:{noise_cfg.immerkaer_variant!r};"
            f"可选:{sorted(_VARIANT_KEYS)}。请检查 calibration.yaml 的 noise 段。")
    main_values = immerkaer_std if variant_key == "sigma_std" else immerkaer_robust
    grid = _anchor_grid([float(x) for x in main_values], noise_cfg)

    # 第二步:按图像切任务,并行跑全部 (σ 档 × 重复) 副本
    n, n_levels = len(selected), len(grid)
    n_repeats = noise_cfg.injection_n_repeats
    n_points = cols * rows
    tasks = []
    for rec in selected:
        image_id = _rec_get(rec, "image_id")
        entry = baseline.get(image_id)
        if entry is not None and entry[1]:
            tasks.append((image_id, str(_rec_get(rec, "path")), entry[0],
                          tuple(grid), n_repeats, noise_cfg.injection_seed,
                          cfg.corners, cfg.corners.sb_fallback_classic, cols, rows))
        else:
            tasks.append(None)  # 基线缺失/未检出:占位,该张整行记 NaN

    results: dict[str, dict] = {}
    worker_errors: dict[str, str] = {}
    dispatch = [t for t in tasks if t is not None]
    if noise_cfg.injection_max_workers <= 1:
        # 串行:不开子进程(小样与调试用),结果与并行完全一致(种子只看三元组)
        results = {}
        done = 0
        for task in dispatch:
            image_id = task[0]
            try:
                results[image_id] = _inject_one_image(task)
            except Exception as e:  # noqa: BLE001  # 故意宽接:单张失败不拖垮整批
                worker_errors[image_id] = f"{type(e).__name__}:{e}"
            done += 1
            if progress:
                progress({"done": done, "total": len(dispatch), "image_id": image_id})
    else:
        with ProcessPoolExecutor(max_workers=noise_cfg.injection_max_workers) as pool:
            futures = {pool.submit(_inject_one_image, t): t[0] for t in dispatch}
            done = 0
            for fut, image_id in futures.items():
                try:
                    results[image_id] = fut.result()
                except Exception as e:  # noqa: BLE001  # 同上:单张失败转留痕
                    worker_errors[image_id] = f"{type(e).__name__}:{e}"
                done += 1
                if progress:
                    progress({"done": done, "total": len(dispatch),
                              "image_id": image_id})

    # 第三步:聚合成 (N, L, R, P, 2) 位移数组,失败副本整行 NaN + 来源留痕
    displacements = np.full((n, n_levels, n_repeats, n_points, 2), np.nan)
    sources = np.full((n, n_levels, n_repeats), "", dtype="<U16")
    per_image_entries = []
    for i, rec in enumerate(selected):
        image_id = _rec_get(rec, "image_id")
        worker_out = results.get(image_id)
        n_fail_img = n_levels * n_repeats  # 基线缺失/读图失败:全部副本计失败
        n_fb_img = 0
        if worker_out is not None:
            displacements[i] = worker_out["disp"]
            sources[i] = worker_out["src"]
            n_fail_img = worker_out["n_fail"]      # 工作单元已逐副本清点失败数
            n_fb_img = worker_out["n_fallback"]
        if image_id in worker_errors:
            n_fail_img = n_levels * n_repeats      # 子进程抛错的整张照失败计
        entry_levels = []
        for li, sigma in enumerate(grid):
            rms = _per_image_level_rms(displacements[i, li])
            det_i = np.isfinite(displacements[i, li, :, 0])
            entry_levels.append({
                "sigma": float(sigma),
                "n_repeats": n_repeats,
                "n_detected": int(det_i.sum()),
                "n_fail": int(n_repeats - det_i.sum()),
                "n_fallback": int((sources[i, li] == "classic_fallback").sum()),
                "rms_euclidean": rms,
            })
        per_image_entries.append({
            "image_id": image_id,
            "split": _rec_get(rec, "split"),
            "pose_id": _rec_get(rec, "pose_id"),
            "sigma_immerkaer_std": float(immerkaer_std[i]),
            "sigma_immerkaer_robust": float(immerkaer_robust[i]),
            "baseline_detected": bool(baseline.get(image_id, (None, False))[1]),
            "error": worker_errors.get(image_id),
            "n_copies": n_levels * n_repeats,
            "n_fail": n_fail_img,
            "n_fallback": n_fb_img,
            "levels": entry_levels,
        })

    detected = np.isfinite(displacements[..., 0]) & np.isfinite(displacements[..., 1])

    # 落盘:npz(纯数组,无 pickle)+ 逐张 json(全保真索引)
    npz_path = out_dir / "injection_displacements.npz"
    np.savez(npz_path,
             displacements=displacements,
             grid=np.asarray(grid, dtype=np.float64),
             image_ids=np.asarray([_rec_get(r, "image_id") for r in selected]),
             detected=detected,
             source=sources,
             sigma_immerkaer_std=immerkaer_std,
             sigma_immerkaer_robust=immerkaer_robust)
    json_path = out_dir / "injection_per_image.json"
    meta = {
        "n_images": n,
        "grid": grid,
        "n_levels": n_levels - 1,
        "n_repeats": n_repeats,
        "base_seed": noise_cfg.injection_seed,
        "immerkaer_variant": noise_cfg.immerkaer_variant,
        "sigma_median": float(np.median(main_values)),
        "grid_auto_anchored": (noise_cfg.injection_sigma_min is None
                               or noise_cfg.injection_sigma_max is None),
        "max_workers": noise_cfg.injection_max_workers,
        "splits": list(noise_cfg.noise_splits),
        "n_detected_copies": int(detected.sum()),
        "n_failed_copies": int((~detected).sum()),
        "n_fallback_copies": int((sources == "classic_fallback").sum()),
        "lineage": lineage,
        "note": ("合成加噪图像本身不落盘(全量会有几十 GB);每个副本由 "
                 "(图像, σ 档, 重复) 三元组从 base_seed 确定性推导种子,同种子"
                 "完整重跑逐位一致。保真规则约束的是测量结果:逐张逐档逐重复的"
                 "角点位移已全量落盘于 injection_displacements.npz。"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "images": per_image_entries}, f,
                  ensure_ascii=False, indent=2)
    return {"grid": grid, "displacements_path": npz_path,
            "per_image_path": json_path, "meta": meta}
