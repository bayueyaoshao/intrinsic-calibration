"""姿态数据模型与 input_data 只读扫描(流水线第①步的数据地基)。

数据布局契约(全部数据驱动,绝不写死张数):
- train/:每个子目录是一个"拍摄姿态",目录名即姿态 ID(pose_id);
  目录内的 .jpg 按文件名字典序排序(文件名含时间戳,字典序 = 拍摄时间序),
  序号即 repeat_idx(第几张连拍),目录内实际张数即 n_repeats。
- test/:平铺的 .jpg,每张独立(不与其他照片构成连拍),pose_id 就用 image_id。

韧性规则(换数据集不崩不静默):
- 文件名不符合时间戳模式:中文警告 + 按自然序继续;
- 空目录:中文警告跳过,计入警告清单;
- 单张姿态(某目录只有 1 张):合法存续,n_repeats=1;
- 分辨率不一致:硬报错(单相机数据契约被破坏,这是前提不是偏好)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2

# 时间戳文件名模式:如 20260717_172523 或 20260717_172523_559573
_TIMESTAMP_NAME = re.compile(r"^\d{8}_\d{6}(_\d+)?$")

# 安全的 image_id 模式:只许字母/数字/下划线/点/横线。
# image_id 会进入输出文件名,这里的限制防止特殊字符(路径分隔符等)
# 把产物写到 output/ 之外(T-01-02 路径安全)。
_SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ImageRecord:
    """一张标定照片的档案:它是谁、在哪、属于哪个姿态的第几张连拍。"""

    image_id: str        # 文件名去扩展名,全局唯一
    path: Path           # 只读路径(相对当前工作目录)
    split: str           # "train" 或 "test"
    pose_id: str         # train:目录名;test:即 image_id(每张独立)
    repeat_idx: int      # 同姿态内按文件名排序的序号(从 0 起)
    n_repeats: int       # 该姿态实际张数(数据驱动,不写死 8)


@dataclass(frozen=True)
class ScanResult:
    """一次目录扫描的完整结果:记录 + 全部警告(警告要写进 manifest 留痕)。"""

    records: tuple[ImageRecord, ...]
    warnings: tuple[str, ...]


def _safe_stem(stem: str, warnings: list[str]) -> bool:
    """检查文件名(去扩展名)能否安全进入输出文件名。

    不合法的名字:记中文警告并跳过该文件(而不是崩溃或带病放行)。
    """
    if _SAFE_IMAGE_ID.match(stem) and stem not in {".", ".."}:
        return True
    warnings.append(
        f"文件名 {stem!r} 含不安全的字符(只允许字母、数字、下划线、点、横线),"
        f"已跳过——防止它破坏产物路径。"
    )
    return False


def _scan_split(split_dir: Path, split: str, warnings: list[str]) -> list[ImageRecord]:
    """扫描一个数据划分(train 或 test)下的全部照片。

    划分目录整个不存在:记中文警告并按 0 张处理(比如只有训练集、
    暂无独立测试集的数据是合法输入),两个划分都为空才在上一层报错。
    """
    if not split_dir.is_dir():
        warnings.append(
            f"数据目录 {split_dir} 不存在,data.{split} 按 0 张照片处理——"
            f"如非有意为之,请检查 config 里 data.{split} 的路径(相对当前工作目录)。"
        )
        return []
    records: list[ImageRecord] = []
    entries = sorted(p for p in split_dir.iterdir())
    subdirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file() and p.suffix.lower() == ".jpg"]

    if split == "train":
        for pose_dir in subdirs:
            jpgs = sorted(p for p in pose_dir.iterdir() if p.is_file()
                          and p.suffix.lower() == ".jpg")
            if not jpgs:
                warnings.append(
                    f"姿态目录 {pose_dir.name} 里没有 .jpg 照片(空目录或只有别的类型),已跳过。"
                )
                continue
            for idx, jpg in enumerate(jpgs):
                if not _TIMESTAMP_NAME.match(jpg.stem):
                    warnings.append(
                        f"文件名 {jpg.name!r} 不符合时间戳模式(期望形如 20260717_172523_559573),"
                        f"已按文件名自然顺序继续编号——连拍时序可能不准,请人工确认。"
                    )
                if not _safe_stem(jpg.stem, warnings):
                    continue
                records.append(ImageRecord(
                    image_id=jpg.stem, path=jpg, split="train",
                    pose_id=pose_dir.name, repeat_idx=idx, n_repeats=len(jpgs),
                ))
    else:  # test:平铺,每张独立
        for jpg in files:
            if not _TIMESTAMP_NAME.match(jpg.stem):
                warnings.append(
                    f"文件名 {jpg.name!r} 不符合时间戳模式,已按文件名自然顺序继续——请人工确认。"
                )
            if not _safe_stem(jpg.stem, warnings):
                continue
            records.append(ImageRecord(
                image_id=jpg.stem, path=jpg, split="test",
                pose_id=jpg.stem, repeat_idx=0, n_repeats=1,
            ))
    return records


def scan_dirs(train_dir: Path, test_dir: Path) -> ScanResult:
    """按配置给的两个目录扫描(train_dir 每姿态一个子目录,test_dir 平铺),
    返回记录 + 警告清单。目录路径来自 calibration.yaml 的 data.train / data.test。"""
    warnings: list[str] = []
    train = _scan_split(Path(train_dir), "train", warnings)
    test = _scan_split(Path(test_dir), "test", warnings)
    if not train and not test:
        raise ValueError(
            f"目录 {train_dir} 与 {test_dir} 下既没有扫到训练照片也没有测试照片,"
            f"请检查 input_data/train 与 input_data/test 是否就位。"
        )
    return ScanResult(records=tuple(train + test), warnings=tuple(warnings))


def scan_dataset_detailed(root: Path) -> ScanResult:
    """扫描 input_data 根目录(下含 train/ 与 test/),返回记录 + 警告清单。"""
    root = Path(root)
    return scan_dirs(root / "train", root / "test")


def scan_dataset(root: Path) -> list[ImageRecord]:
    """扫描 input_data,只返回照片记录(警告版见 scan_dataset_detailed)。"""
    return list(scan_dataset_detailed(root).records)


def imread_checked(path: Path):
    """带检查的读图:cv2.imread 读坏文件不报错而是返回 None,
    这里立刻抛带文件名的中文错误,把报错位置钉在病因上(Pitfall 7)。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            f"照片读取失败:{path}。文件可能损坏、路径不对或没有读权限,"
            f"请人工检查这张图。"
        )
    return img


def verify_consistent_resolution(records) -> tuple[int, int]:
    """逐张读图,断言全部照片分辨率完全一致(单相机、相同拍摄参数的数据契约)。

    返回统一分辨率 (宽, 高),供 manifest 记录;
    不一致:抛中文 ValueError,消息里带两张尺寸不同的照片,直接定位问题。
    """
    reference = None  # (image_id, 宽, 高)
    for record in records:
        gray = imread_checked(record.path)
        height, width = gray.shape[:2]
        if reference is None:
            reference = (record.image_id, width, height)
        elif (width, height) != (reference[1], reference[2]):
            raise ValueError(
                f"分辨率不一致,违反单相机数据契约:{reference[0]} 是 "
                f"{reference[1]}x{reference[2]}(宽x高),而 {record.image_id} 是 "
                f"{width}x{height}。input_data 必须全部来自同一相机、相同拍摄参数,"
                f"请先剔除混入的照片再重跑。"
            )
    if reference is None:
        raise ValueError("没有可检查的照片记录(记录清单为空)。")
    return reference[1], reference[2]
