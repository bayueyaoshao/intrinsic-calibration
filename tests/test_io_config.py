"""配置解析与校验测试(IO-03)+ 血缘指纹测试。

覆盖四类要求:合法加载的值断言、缺字段的中文报错、非法值的中文报错、
源码安全断言(只许 safe_load,禁 yaml.load)。
"""

import hashlib
from pathlib import Path

import pytest

from calib.io.config import load_board, load_calibration
from calib.util.lineage import make_lineage, sha256_file

# 仓库根目录(测试文件在 <根>/tests/ 下)
REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_BOARD = REPO_ROOT / "config" / "board.yaml"
REAL_CALIB = REPO_ROOT / "config" / "calibration.yaml"


# ---- 合法加载:真实配置的值必须与随仓库文件一致 ----

def test_load_real_board_values():
    """真实 board.yaml 解析结果:11 列 × 8 行内角点,方格 25.0mm。"""
    spec = load_board(REAL_BOARD)
    assert (spec.cols, spec.rows, spec.square_size_mm, spec.unit) == (11, 8, 25.0, "mm")


def test_load_real_calibration_values():
    """真实 calibration.yaml:窗口粒度 frame、亚像素参数、连拍容差等关键默认值。"""
    cfg = load_calibration(REAL_CALIB)
    assert cfg.data.train == "input_data/train"
    assert cfg.data.test == "input_data/test"
    assert cfg.seed == 42
    assert cfg.corners.subpix.window_granularity == "frame"
    assert cfg.corners.subpix.min_win == 3
    assert cfg.corners.burst_check.median_tol == 0.05
    assert cfg.corners.burst_check.max_tol == 0.3
    assert cfg.calibrate.fisheye.max_iter == 100
    assert cfg.visualization.mosaic.cols == 8


def test_missing_exclude_images_defaults_to_empty(tmp_path):
    """calibration.yaml 缺 exclude_images 时:默认空清单(人工审阅后才填)。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("data:\n  train: a\n  test: b\n", encoding="utf-8")
    cfg = load_calibration(p)
    assert cfg.exclude_images == ()


# ---- 非法配置:必须抛 ValueError,消息含字段名(中文)----

def _write_board(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "board.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_missing_cols_field_chinese_error(tmp_path):
    """缺 cols 字段:报错必须点名 board.cols,而不是裸 KeyError。"""
    p = _write_board(
        tmp_path,
        "board:\n  board_type: chessboard\n  rows: 8\n"
        "  square_size_mm: 25.0\n  unit: mm\n",
    )
    with pytest.raises(ValueError) as exc_info:
        load_board(p)
    assert "board.cols" in str(exc_info.value)
    assert "实际读到的值" in str(exc_info.value)  # 错误里带了实际值,方便定位


def test_square_size_zero_chinese_error(tmp_path):
    """square_size_mm = 0:物理边长必须是正数。"""
    p = _write_board(
        tmp_path,
        "board:\n  board_type: chessboard\n  cols: 11\n  rows: 8\n"
        "  square_size_mm: 0\n  unit: mm\n",
    )
    with pytest.raises(ValueError, match="square_size_mm"):
        load_board(p)


def test_wrong_board_type_chinese_error(tmp_path):
    """board_type 不是 chessboard:当前只支持棋盘格,报错点名该字段。"""
    p = _write_board(
        tmp_path,
        "board:\n  board_type: charuco\n  cols: 11\n  rows: 8\n"
        "  square_size_mm: 25.0\n  unit: mm\n",
    )
    with pytest.raises(ValueError, match="board_type"):
        load_board(p)


def test_quantile_order_violation(tmp_path):
    """分位线上下颠倒:global_quantile_low 必须小于 high。"""
    p = tmp_path / "calibration.yaml"
    p.write_text(
        "ingest:\n  precheck:\n    global_quantile_low: 0.9\n"
        "    global_quantile_high: 0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="global_quantile_low"):
        load_calibration(p)


# ---- 源码安全断言(T-01-01:配置解析只许 safe_load)----

def test_config_source_uses_safe_load_only():
    """config.py 源码必须含 safe_load、绝不含 yaml.load((防反序列化漏洞)。"""
    source = (REPO_ROOT / "src" / "calib" / "io" / "config.py").read_text(encoding="utf-8")
    assert "safe_load" in source
    assert "yaml.load(" not in source


# ---- 血缘指纹(ITER-01 地基)----

def test_sha256_file_matches_hashlib(tmp_path):
    """sha256_file 与 hashlib 直接计算一致(分块实现没有改变结果)。"""
    p = tmp_path / "data.bin"
    p.write_bytes(b"intrinsic-calibration lineage test")
    expected = hashlib.sha256(b"intrinsic-calibration lineage test").hexdigest()
    assert sha256_file(p) == expected


def test_make_lineage_fields(tmp_path):
    """make_lineage 返回的血缘字典字段齐全,指纹与文件内容对应。"""
    cfg_file = tmp_path / "board.yaml"
    cfg_file.write_text("board: {}", encoding="utf-8")
    upstream_file = tmp_path / "upstream.json"
    upstream_file.write_text("{}", encoding="utf-8")

    lineage = make_lineage(
        config_files={"board.yaml": cfg_file},
        upstream={"manifest.json": upstream_file},
        round_id="run_20260717_172523",
    )

    assert set(lineage.keys()) >= {"created", "calib_version", "config_sha256", "upstream", "round_id"}
    assert lineage["round_id"] == "run_20260717_172523"
    assert lineage["config_sha256"]["board.yaml"] == sha256_file(cfg_file)
    assert lineage["upstream"]["manifest.json"]["sha256"] == sha256_file(upstream_file)
    assert lineage["upstream"]["manifest.json"]["path"].endswith("upstream.json")
    # 内容一变,指纹必须变(血缘指纹的意义所在)
    cfg_file.write_text("board: {changed}", encoding="utf-8")
    assert make_lineage({"board.yaml": cfg_file}, {}, "r")["config_sha256"]["board.yaml"] != \
        lineage["config_sha256"]["board.yaml"]
