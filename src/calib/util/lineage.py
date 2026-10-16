"""血缘指纹:让落盘产物的任何数字都可追溯。

每个写盘产物(数据清单、角点文件、标定结果……)都嵌一份血缘字典,记录:
- created:生成时间(UTC,精确到秒)
- calib_version:生成它的 calib 包版本
- config_sha256:用了哪些配置文件、各自的 sha256 指纹(内容变了指纹就变)
- upstream:从哪些上游文件算出来的(相对路径 + sha256)
- round_id:本轮运行标识(如 run_20260717_172523),跨轮对比时用

sha256:一种单向散列指纹——文件内容哪怕只差一个字节,指纹就完全不同,
所以"指纹相同"可以当作"内容相同"的铁证。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path | str) -> str:
    """计算单个文件的 sha256 指纹(十六进制,64 个字符)。

    分块读取(每次 1MB),几百 MB 的大文件也不会把内存吃满。
    文件不存在时抛 FileNotFoundError——血缘是产物的完整性证据,
    静默跳过等于伪造证据,必须让它响。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    """取本包(intrinsic-calibration)的安装版本号。

    未安装为包(比如直接拿源码跑脚本)时返回提示字符串而不是崩溃——
    血缘里留个明确记号,比一个炸掉的写盘流程更有用。
    """
    try:
        return importlib.metadata.version("intrinsic-calibration")
    except importlib.metadata.PackageNotFoundError:
        return "未安装(源码直跑,无版本号)"


def make_lineage(
    config_files: dict[str, Path | str],
    upstream: dict[str, Path | str],
    round_id: str,
) -> dict:
    """组装血缘指纹字典(每个落盘产物写盘前必须调用)。

    参数:
        config_files: 配置文件名 → 路径。键用文件名(如 "board.yaml"),
                      值是路径(相对当前工作目录)。
        upstream:     上游产物名 → 路径。本产物是从这些文件算出来的,
                      例如角点文件的 upstream 含上一轮的 manifest.json。
        round_id:     本轮运行标识,形如 "run_20260717_172523"。
    """
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "calib_version": _package_version(),
        "config_sha256": {
            name: sha256_file(path) for name, path in config_files.items()
        },
        "upstream": {
            name: {"path": str(Path(path)), "sha256": sha256_file(path)}
            for name, path in upstream.items()
        },
        "round_id": round_id,
    }
