"""角点产物落盘与读取:corners_<划分>_<检测器>.npz + 同名 meta.json 旁车文件。

布局(带掩码的稠密格式,与数据清单逐行对齐):
- image_ids: (N,)   照片编号(文件名去扩展名),与 manifest 的 images 顺序一致;
                    未检出的照片照样占一行,绝不从数据里消失(D-01)
- uv:       (N, 88, 2) float64,角点像素坐标;未检出整行为 NaN
- sq_px:    (N,)    每帧实测的方格像素边长(下游窗口复核/体检归一复用)
- win_size: (N,)    每帧实际使用的亚像素半窗口(供审查窗口断言 2*win+1 < sq_px)
- detected: (N,)    bool,是否检出

文件名按 划分(train/test)× 检测器(classic/sb)共 4 份(D-07 两颗各存一份,
呼应 ROADMAP 成功标准 2 的 train/test 划分)。

安全防线:读取一律 np.load(..., allow_pickle=False)——npz 的 pickle 模式
可以执行任意代码,关死它,角点文件只可能是纯数字数组(Pitfall 10 / T-02-01)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 规范化规则的版本字符串:规则一变,下游读到的产物含义就变,必须留痕
CANONICAL_RULE_VERSION = "canonical-v1:行方向朝画面右、列方向朝画面下"


def save_corners(out_dir: Path, split: str, detector: str,
                 image_ids: list[str] | np.ndarray,
                 uv: np.ndarray, sq_px: np.ndarray,
                 win_size: np.ndarray, detected: np.ndarray,
                 lineage: dict, meta_extra: dict | None = None) -> Path:
    """写一份角点 npz 与同名 meta.json,返回 npz 路径。

    参数:
        out_dir:    本轮运行目录(如 output/corners/run_<时间戳>)
        split:      "train" 或 "test"
        detector:   "classic" 或 "sb"
        image_ids:  (N,) 与 manifest 对齐的照片编号
        uv:         (N, 点数, 2) float64,未检出为 NaN
        sq_px:      (N,) 方格像素边长,未检出为 NaN
        win_size:   (N,) 半窗口,未检出为 NaN
        detected:   (N,) bool
        lineage:    make_lineage 组装的血缘指纹
        meta_extra: 检测器选项/精化判据/窗口粒度等说明,进 meta.json 留痕
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"corners_{split}_{detector}"
    npz_path = out_dir / f"{name}.npz"
    np.savez(
        npz_path,
        image_ids=np.asarray(image_ids),
        uv=np.asarray(uv, dtype=np.float64),
        sq_px=np.asarray(sq_px, dtype=np.float64),
        win_size=np.asarray(win_size, dtype=np.float64),
        detected=np.asarray(detected, dtype=bool),
    )

    meta = {
        "file": npz_path.name,
        "split": split,
        "detector": detector,
        "n_images": len(image_ids),
        "n_detected": int(np.asarray(detected).sum()),
        "canonical_rule": CANONICAL_RULE_VERSION,
        "lineage": lineage,
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(out_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return npz_path


def load_corners(path: Path | str) -> dict[str, np.ndarray]:
    """读取角点 npz,返回 {键: 数组} 字典。

    强制 allow_pickle=False:遇到含 pickle 对象的文件直接抛错,
    绝不执行其中的任意代码(反序列化安全防线,测试有锁定)。
    """
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
