"""配置解析与校验:board.yaml(棋盘规格)与 calibration.yaml(算法参数)的唯一入口。

硬性规则:
- 只用 yaml.safe_load(绝不 yaml.load——后者能执行配置里的任意代码,是安全漏洞);
- 解析失败必须给出带字段名与实际值的通俗中文错误,让人一看就知道改哪里;
- 下游永远拿到不可变数据类(dataclass),不接触原始字典——
  字段拼错在开发阶段就暴露,而不是算到一半才炸。

数据类字段名与 yaml 键一一对应,嵌套关系也一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 经典检测器允许的选项名(写在 calibration.yaml 里的名字,代码内翻译成 cv2 常量)
_ALLOWED_CLASSIC_FLAGS = {"adaptive_thresh", "normalize_image", "fast_check"}

# 亚像素窗口粒度的合法取值:frame = 每帧一个窗口;corner = 逐角点局部自适应
_ALLOWED_GRANULARITY = {"frame", "corner"}


@dataclass(frozen=True)
class BoardSpec:
    """棋盘规格(来自 config/board.yaml)。

    square_size_mm 只影响外参平移的物理尺度,不影响内参——
    它绝不进入任何像素级的逻辑(角点窗口等用实测方格像素边长,不用它)。
    """

    cols: int = 11          # 内部角点列数
    rows: int = 8           # 内部角点行数
    square_size_mm: float = 25.0
    unit: str = "mm"


@dataclass(frozen=True)
class DataDirs:
    """数据目录(相对当前工作目录)。"""

    train: str = "input_data/train"
    test: str = "input_data/test"


@dataclass(frozen=True)
class PrecheckConfig:
    """入库体检的判定阈值。

    blur_z_threshold:同姿态连拍内模糊度稳健 z 分数的报警线;
    global_quantile_low/high:全局兜底分位线(单张姿态没有连拍可比时用)。
    """

    blur_z_threshold: float = 3.5
    global_quantile_low: float = 0.01
    global_quantile_high: float = 0.99


@dataclass(frozen=True)
class IngestConfig:
    """第①步入库体检的参数。"""

    precheck: PrecheckConfig = field(default_factory=PrecheckConfig)


@dataclass(frozen=True)
class SubpixConfig:
    """亚像素精化参数(把角点位置从整数像素细化到小数像素)。

    窗口从每帧实测的方格像素边长按比例推导,这里只控制粒度与下限;
    半窗口绝不允许大到跨过方格对边,否则会产生重投影误差里完全看不见的
    系统性定位偏差(本项目头号陷阱)。
    """

    window_granularity: str = "frame"
    min_win: int = 3
    max_iter: int = 30
    epsilon: float = 0.01


@dataclass(frozen=True)
class BurstCheckConfig:
    """同姿态连拍一致性断言的容差(以方格像素边长的倍数表达,不写死像素)。"""

    median_tol: float = 0.05
    max_tol: float = 0.3


@dataclass(frozen=True)
class CornersConfig:
    """第②步找角点参数:经典与新款(SB)双检测器并跑,各存一份。"""

    classic_flags: tuple[str, ...] = ("adaptive_thresh", "normalize_image")
    sb_flags: int = 0
    # SB 未检出的帧是否用经典检测器的结果兜底(填进 SB 路线的槽位)。
    # 三轮实验结论(见 corners 产物 review_sb_failed/DIAGNOSIS.md):SB 在大倾斜/
    # 远距视角下会系统性拒绝共 41 帧,而"缩小档救援 + 全分辨率精化"的产物与
    # classic 完全重合(偏差 0.01px 级)——兜底比多尺度重试更简单且多覆盖
    # 17 张穷举也救不回的帧,故默认开。false = 关闭,行为与只有 SB 时完全一致。
    sb_fallback_classic: bool = True
    subpix: SubpixConfig = field(default_factory=SubpixConfig)
    burst_check: BurstCheckConfig = field(default_factory=BurstCheckConfig)


@dataclass(frozen=True)
class FisheyeConfig:
    """鱼眼 KB4 模型的求解参数。

    flags 默认 2 = CALIB_RECOMPUTE_EXTRINSIC(每轮迭代都重算棋盘位姿):
    实测不加它时 OpenCV 只按初始内参算一次位姿,会收敛到 RMS 上百像素的
    假结果(哪怕数据完美符合 KB4);skew 倾斜因子仍被估计(未加 FIX_SKEW)。
    """

    flags: int = 2
    max_iter: int = 100
    epsilon: float = 1e-6


@dataclass(frozen=True)
class CalibrateConfig:
    """第④步求解内参参数。"""

    fisheye: FisheyeConfig = field(default_factory=FisheyeConfig)


@dataclass(frozen=True)
class MosaicConfig:
    """汇总拼图的布局参数(拼图只是索引,不是逐帧副本的替代)。"""

    per_pose_sample: int = 1
    worst_n: int = 20
    thumb_width: int = 320
    cols: int = 8


@dataclass(frozen=True)
class VisualizationConfig:
    """可视化与副本留痕参数。"""

    jpeg_quality: int = 90
    mosaic: MosaicConfig = field(default_factory=MosaicConfig)


@dataclass(frozen=True)
class CalibrationConfig:
    """calibration.yaml 解析结果:全流程算法参数的唯一载体(D-12)。"""

    data: DataDirs = field(default_factory=DataDirs)
    seed: int = 42
    cv2_threads: int = 0
    exclude_images: tuple[str, ...] = ()
    ingest: IngestConfig = field(default_factory=IngestConfig)
    corners: CornersConfig = field(default_factory=CornersConfig)
    calibrate: CalibrateConfig = field(default_factory=CalibrateConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def _fail(field_name: str, actual, requirement: str) -> None:
    """抛出带字段名与实际值的中文配置错误(统一格式,便于定位)。"""
    raise ValueError(
        f"配置字段 {field_name} 不合法:{requirement};实际读到的值是 {actual!r}。"
        f"请检查对应的 yaml 文件。"
    )


def _check_is_dict(raw, field_name: str) -> dict:
    """确认 yaml 里某一段是"键: 值"结构,否则给中文报错。"""
    if not isinstance(raw, dict):
        _fail(field_name, raw, "应当是一组缩进的配置项(字典)")
    return raw


def _read_yaml(path: Path | str) -> object:
    """读 yaml 文件,safe_load(禁 yaml.load,防配置执行任意代码)。"""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"配置文件不存在:{path}。请确认路径(相对当前工作目录)是否正确。")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_board(path: Path | str) -> BoardSpec:
    """解析并校验棋盘规格 board.yaml,返回不可变的 BoardSpec。

    校验规则:board_type 必须是 chessboard(当前只支持棋盘格);
    cols / rows >= 3(再少无法构成棋盘);square_size_mm > 0。
    任何不合:抛 ValueError,消息含字段名与实际值。
    """
    raw = _read_yaml(path)
    if not isinstance(raw, dict) or "board" not in raw:
        _fail("board", raw, "文件顶层必须有 board: 这一段")
    board = _check_is_dict(raw["board"], "board")

    board_type = board.get("board_type")
    if board_type != "chessboard":
        _fail("board.board_type", board_type, "当前只支持 chessboard(棋盘格)")

    cols = board.get("cols")
    if not isinstance(cols, int) or isinstance(cols, bool) or cols < 3:
        _fail("board.cols", cols, "必须是 >= 3 的整数(内部角点列数)")

    rows = board.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 3:
        _fail("board.rows", rows, "必须是 >= 3 的整数(内部角点行数)")

    square = board.get("square_size_mm")
    if not isinstance(square, int | float) or isinstance(square, bool) or square <= 0:
        _fail("board.square_size_mm", square, "必须是 > 0 的数(方格物理边长,毫米)")

    unit = board.get("unit", "mm")
    if not isinstance(unit, str) or not unit:
        _fail("board.unit", unit, "必须是长度单位的文字(如 mm)")

    return BoardSpec(cols=cols, rows=rows, square_size_mm=float(square), unit=unit)


def _parse_precheck(raw: dict) -> PrecheckConfig:
    section = _check_is_dict(raw.get("precheck", {}), "ingest.precheck")
    z = section.get("blur_z_threshold", PrecheckConfig.blur_z_threshold)
    if not isinstance(z, int | float) or isinstance(z, bool) or z <= 0:
        _fail("ingest.precheck.blur_z_threshold", z, "必须是 > 0 的数(稳健 z 分数报警线)")
    low = section.get("global_quantile_low", PrecheckConfig.global_quantile_low)
    high = section.get("global_quantile_high", PrecheckConfig.global_quantile_high)
    if not isinstance(low, int | float) or isinstance(low, bool) or not 0 < low < 1:
        _fail("ingest.precheck.global_quantile_low", low, "必须是 0 到 1 之间的小数(分位线)")
    if not isinstance(high, int | float) or isinstance(high, bool) or not 0 < high < 1:
        _fail("ingest.precheck.global_quantile_high", high, "必须是 0 到 1 之间的小数(分位线)")
    if low >= high:
        _fail("ingest.precheck.global_quantile_low", (low, high),
              f"下分位线必须小于上分位线(global_quantile_high = {high!r})")
    return PrecheckConfig(blur_z_threshold=float(z),
                          global_quantile_low=float(low),
                          global_quantile_high=float(high))


def _parse_subpix(raw: dict) -> SubpixConfig:
    cfg = SubpixConfig()
    section = _check_is_dict(raw.get("subpix", {}), "corners.subpix")
    granularity = section.get("window_granularity", cfg.window_granularity)
    if granularity not in _ALLOWED_GRANULARITY:
        _fail("corners.subpix.window_granularity", granularity,
              f"只能是 {' 或 '.join(sorted(_ALLOWED_GRANULARITY))}")
    min_win = section.get("min_win", cfg.min_win)
    if not isinstance(min_win, int) or isinstance(min_win, bool) or min_win < 1:
        _fail("corners.subpix.min_win", min_win, "必须是 >= 1 的整数(半窗口下限,像素)")
    max_iter = section.get("max_iter", cfg.max_iter)
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        _fail("corners.subpix.max_iter", max_iter, "必须是 >= 1 的整数(迭代次数上限)")
    epsilon = section.get("epsilon", cfg.epsilon)
    if not isinstance(epsilon, int | float) or isinstance(epsilon, bool) or epsilon <= 0:
        _fail("corners.subpix.epsilon", epsilon, "必须是 > 0 的数(收敛容差,像素)")
    return SubpixConfig(window_granularity=granularity, min_win=min_win,
                        max_iter=max_iter, epsilon=float(epsilon))


def _parse_burst_check(raw: dict) -> BurstCheckConfig:
    cfg = BurstCheckConfig()
    section = _check_is_dict(raw.get("burst_check", {}), "corners.burst_check")
    median_tol = section.get("median_tol", cfg.median_tol)
    max_tol = section.get("max_tol", cfg.max_tol)
    for name, value in (("median_tol", median_tol), ("max_tol", max_tol)):
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            _fail(f"corners.burst_check.{name}", value, "必须是 > 0 的数(方格边长的倍数)")
    if median_tol > max_tol:
        _fail("corners.burst_check.median_tol", (median_tol, max_tol),
              f"中位数容差不应大于最大值容差(max_tol = {max_tol!r})")
    return BurstCheckConfig(median_tol=float(median_tol), max_tol=float(max_tol))


def _parse_corners(raw: dict) -> CornersConfig:
    section = _check_is_dict(raw.get("corners", {}), "corners")
    flags = section.get("classic_flags", CornersConfig.classic_flags)
    if not isinstance(flags, (list, tuple)) or not all(isinstance(f, str) for f in flags):
        _fail("corners.classic_flags", flags, "必须是一列选项名(如 [adaptive_thresh])")
    bad = [f for f in flags if f not in _ALLOWED_CLASSIC_FLAGS]
    if bad:
        _fail("corners.classic_flags", bad,
              f"含有不认识的选项;可选:{sorted(_ALLOWED_CLASSIC_FLAGS)}")
    sb_flags = section.get("sb_flags", CornersConfig.sb_flags)
    if not isinstance(sb_flags, int) or isinstance(sb_flags, bool) or sb_flags < 0:
        _fail("corners.sb_flags", sb_flags, "必须是 >= 0 的整数(选项位掩码)")
    fallback = section.get("sb_fallback_classic", CornersConfig.sb_fallback_classic)
    # 注意顺序:Python 里 True 同时也是 int(issubclass(bool, int)),但 1 不是 bool,
    # 所以先判 bool 再拒其余一切——整数 1/0 是 yaml 手写时最常见的笔误,必须拦下
    if not isinstance(fallback, bool):
        _fail("corners.sb_fallback_classic", fallback,
              "必须是 true 或 false(SB 未检出的帧是否用经典检测器结果兜底)")
    return CornersConfig(classic_flags=tuple(flags), sb_flags=sb_flags,
                         sb_fallback_classic=fallback,
                         subpix=_parse_subpix(section),
                         burst_check=_parse_burst_check(section))


def _parse_fisheye(raw: dict) -> FisheyeConfig:
    cfg = FisheyeConfig()
    section = _check_is_dict(raw.get("fisheye", {}), "calibrate.fisheye")
    flags = section.get("flags", cfg.flags)
    if not isinstance(flags, int) or isinstance(flags, bool) or flags < 0:
        _fail("calibrate.fisheye.flags", flags, "必须是 >= 0 的整数(选项位掩码)")
    max_iter = section.get("max_iter", cfg.max_iter)
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        _fail("calibrate.fisheye.max_iter", max_iter, "必须是 >= 1 的整数(迭代次数上限)")
    epsilon = section.get("epsilon", cfg.epsilon)
    if not isinstance(epsilon, int | float) or isinstance(epsilon, bool) or epsilon <= 0:
        _fail("calibrate.fisheye.epsilon", epsilon, "必须是 > 0 的数(收敛容差)")
    return FisheyeConfig(flags=flags, max_iter=max_iter, epsilon=float(epsilon))


def _parse_visualization(raw: dict) -> VisualizationConfig:
    section = _check_is_dict(raw.get("visualization", {}), "visualization")
    quality = section.get("jpeg_quality", VisualizationConfig.jpeg_quality)
    if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 100:
        _fail("visualization.jpeg_quality", quality, "必须是 1 到 100 的整数(JPEG 质量)")
    mosaic = _check_is_dict(section.get("mosaic", {}), "visualization.mosaic")
    cfg = MosaicConfig()
    per_pose = mosaic.get("per_pose_sample", cfg.per_pose_sample)
    if not isinstance(per_pose, int) or isinstance(per_pose, bool) or per_pose < 1:
        _fail("visualization.mosaic.per_pose_sample", per_pose, "必须是 >= 1 的整数(每姿态抽样张数)")
    worst_n = mosaic.get("worst_n", cfg.worst_n)
    if not isinstance(worst_n, int) or isinstance(worst_n, bool) or worst_n < 0:
        _fail("visualization.mosaic.worst_n", worst_n, "必须是 >= 0 的整数(最差张数)")
    thumb = mosaic.get("thumb_width", cfg.thumb_width)
    if not isinstance(thumb, int) or isinstance(thumb, bool) or thumb < 1:
        _fail("visualization.mosaic.thumb_width", thumb, "必须是 >= 1 的整数(缩略图宽,像素)")
    cols = mosaic.get("cols", cfg.cols)
    if not isinstance(cols, int) or isinstance(cols, bool) or cols < 1:
        _fail("visualization.mosaic.cols", cols, "必须是 >= 1 的整数(拼图每行格数)")
    return VisualizationConfig(jpeg_quality=quality,
                               mosaic=MosaicConfig(per_pose_sample=per_pose,
                                                   worst_n=worst_n,
                                                   thumb_width=thumb, cols=cols))


def load_calibration(path: Path | str) -> CalibrationConfig:
    """解析并校验 calibration.yaml,返回不可变的 CalibrationConfig。

    缺省键用内置默认值(与随仓库的 calibration.yaml 一致);
    类型或范围不合给带字段名与实际值的中文错误;
    exclude_images 缺省时为空元组(人工审阅后填 image_id,程序只做逻辑剔除)。
    """
    raw = _read_yaml(path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        _fail("calibration.yaml 顶层", raw, "应当是一组配置项(字典)")
    top = raw

    data_section = _check_is_dict(top.get("data", {}), "data")
    train = data_section.get("train", DataDirs.train)
    test = data_section.get("test", DataDirs.test)
    for name, value in (("data.train", train), ("data.test", test)):
        if not isinstance(value, str) or not value.strip():
            _fail(name, value, "必须是数据目录路径文字(相对当前工作目录)")

    seed = top.get("seed", CalibrationConfig.seed)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        _fail("seed", seed, "必须是 >= 0 的整数(随机种子)")

    cv2_threads = top.get("cv2_threads", CalibrationConfig.cv2_threads)
    if not isinstance(cv2_threads, int) or isinstance(cv2_threads, bool) or cv2_threads < 0:
        _fail("cv2_threads", cv2_threads, "必须是 >= 0 的整数(OpenCV 线程数)")

    exclude = top.get("exclude_images", [])
    if exclude is None:
        exclude = []
    if not isinstance(exclude, list) or not all(isinstance(x, str) for x in exclude):
        _fail("exclude_images", exclude, "必须是一列 image_id 文字(人工剔除清单)")

    ingest_section = _check_is_dict(top.get("ingest", {}), "ingest")

    return CalibrationConfig(
        data=DataDirs(train=train, test=test),
        seed=seed,
        cv2_threads=cv2_threads,
        exclude_images=tuple(exclude),
        ingest=IngestConfig(precheck=_parse_precheck(ingest_section)),
        corners=_parse_corners(top),
        calibrate=CalibrateConfig(fisheye=_parse_fisheye(top.get("calibrate", {}))),
        visualization=_parse_visualization(top),
    )
