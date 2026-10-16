"""存疑照片的原图副本(D-06):拼图缩略图看不清细节,原图另存供人工放大细看。

职责分配(照研究的职责分配表):io/precheck 只产出存疑名单(判定写在
manifest 里),把原图复制一份出来的活归可视化层——副本是给人看的可视化
产物,不是数据判定的一部分。源文件只读不碰:input_data 是只读输入,
这里用 shutil.copy2 复制(连修改时间一起带上),绝不动原文件,
更没有任何删除动作——体检整条链路"只标记不剔除"(D-01)。
"""

from __future__ import annotations

import shutil
from pathlib import Path


def write_suspect_originals(records_with_precheck: list[dict], run_dir: Path) -> list[Path]:
    """把 verdict == "suspect" 的照片原图复制到 run_dir/suspect_originals/<image_id>.jpg。

    参数 records_with_precheck:每条含 image_id、path(只读源路径)与
    precheck.verdict;正是 manifest images 数组的形状(判定来自体检)。

    返回副本路径清单(与存疑名单一一对应,供控制台汇总报数)。
    """
    copy_dir = Path(run_dir) / "suspect_originals"
    copied: list[Path] = []
    for item in records_with_precheck:
        if item.get("precheck", {}).get("verdict") != "suspect":
            continue  # 只复制存疑照片,其余一张都不碰
        source = Path(item["path"])
        target = copy_dir / f"{item['image_id']}.jpg"
        if not target.parent.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)  # copy2 连同修改时间一起复制;源只读不碰
        copied.append(target)
    return copied
