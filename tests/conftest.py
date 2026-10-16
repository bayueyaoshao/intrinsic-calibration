"""pytest 公共夹具。

make_tiny_jpg:生成一张小的随机灰度 JPEG,供数据模型/分辨率检查的测试用;
附带属性 input_root 指向夹具数据根(input_data 目录)。
(完整的合成棋盘投影夹具——带已知真值角点——由下一个计划建立。)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy
import pytest


@pytest.fixture
def make_tiny_jpg(tmp_path: Path):
    """造图夹具:make_tiny_jpg("input_data/train/<姿态>/<文件>.jpg", 宽, 高) → 路径。"""

    def _make(rel_path: str, width: int, height: int) -> Path:
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        rng = numpy.random.default_rng(42)
        gray = rng.integers(0, 256, size=(height, width), dtype=numpy.uint8)
        if not cv2.imwrite(str(target), gray):
            raise RuntimeError(f"测试夹具写图失败:{target}")
        return target

    # 附带数据根路径,测试直接取用,不用再猜目录
    _make.input_root = tmp_path / "input_data"  # type: ignore[attr-defined]
    return _make
