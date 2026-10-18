"""noise.json 产物契约与全保真明细落盘:噪声阶段的输出侧。

两层结构(D-13 产物保真:汇总只是索引不是替代):
- noise.json:顶层契约键 + 全局双口径标量 σ + 来源计数 + 各明细产物的路径
  引用——它是下游(第⑤步 calib fisher)和报告读数的"目录";
- 明细 npz(同目录旁车):逐姿态逐角点 σ 表(每姿态 88×2)与逐张
  Immerkaer σ 全量落盘,纯数值数组 + 定长 unicode 编号列,读写一律
  allow_pickle=False(npz 的 pickle 模式可执行任意代码,项目既定防线)。

失败契约(照 solve/result.py 的 make_failure 体例):命令中途失败时
noise.json 照样落盘——结构键一个不少、数值字段一律 null(落盘成 JSON
null)、failure_note 非空中文写清原因;明细 npz 只在成功时写(没有数据
可写时硬写只会撒谎)。上游对成败一视同仁:先验键、再取数。

血缘(T-03-03-03):noise.json 与每个明细 npz 的 *.meta.json 旁车都带
同一份血缘指纹(调用方经 util/lineage.py 的 make_lineage 组装后传入,
上游 = corners npz + manifest),产物来源可考。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# noise.json 顶层契约键(下游与集成测试按这套键取数;失败时也一个不少)
REPORT_KEYS = ("schema_version", "created", "lineage", "sigma_detector_corner",
               "sigma_temporal_corner", "n_poses_temporal", "n_poses_fallback",
               "per_source_counts", "outlier_poses", "splits", "transfer_curve",
               "injection", "meta", "failure_note")

# 产物文件名(与 noise.json 同目录)
TEMPORAL_NPZ_NAME = "temporal_sigma_per_corner.npz"
IMMERKAER_NPZ_NAME = "immerkaer_per_image.npz"

SCHEMA_VERSION = "noise-v1"


def make_failure_report(lineage: dict, note: str) -> dict:
    """失败契约的工厂:结构键全在、数值字段一律 None、中文说明非空。

    上游读取逻辑对成败一视同仁(键一个不少)——失败只是"数值还没算出来",
    不是换一种结构。结构性的字典/列表给空占位(照 result.py 的 params={}),
    数值标量一律 None。
    """
    text = str(note).strip()
    if not text:
        raise ValueError(
            "make_failure_report 需要非空的中文失败说明(failure_note),"
            "空说明等于伪造证据。请写清哪个环节、什么原因。")
    return {
        "schema_version": SCHEMA_VERSION,
        "created": (lineage or {}).get("created"),
        "lineage": dict(lineage) if lineage else {},
        "sigma_detector_corner": None,
        "sigma_temporal_corner": None,
        "n_poses_temporal": None,
        "n_poses_fallback": None,
        "per_source_counts": {},
        "outlier_poses": [],
        "splits": {},
        "transfer_curve": None,
        "injection": None,
        "meta": {},
        "failure_note": text,
    }


def _check_contract(report: dict) -> None:
    """写盘前先验契约:顶层结构键一个不少,缺键当场中文报错。"""
    missing = [key for key in REPORT_KEYS if key not in report]
    if missing:
        raise AssertionError(
            f"noise.json 契约键缺失:{missing}。顶层必须有这些键({len(REPORT_KEYS)}"
            f" 个,失败时也一样),请用 make_failure_report / 命令体组装完整的报告。")


def _write_detail_npz(out_dir: Path, name: str, arrays: dict,
                      lineage: dict, n_label: str, n_value: int) -> Path:
    """写一份明细 npz 与 *.meta.json 旁车(血缘必带,中文 meta 原样可读)。"""
    npz_path = out_dir / name
    np.savez(npz_path, **arrays)
    meta = {"file": name, n_label: n_value, "lineage": lineage}
    with open(out_dir / f"{Path(name).stem}.meta.json",
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return npz_path


def save_noise_report(out_dir: Path | str, report: dict, lineage: dict) -> dict:
    """把噪声报告落盘:noise.json(契约键全)+ 明细 npz 与 meta 旁车。

    参数:
        out_dir: 本轮运行目录(如 output/noise/run_<时间戳>),不存在则创建;
        report:  命令体组装的完整报告。可带两个私有明细键(写盘时取出转 npz,
                 绝不进 json):"_temporal_detail"(逐姿态逐角点 σ 表)与
                 "_immerkaer_detail"(逐张 Immerkaer σ);
        lineage: make_lineage 组装的血缘指纹(进 noise.json 与每个 npz 旁车)。

    返回 {"noise_json", "temporal_npz", "immerkaer_npz"} 路径字典
    (失败报告没有明细,npz 路径为 None)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_contract(report)

    temporal_npz_path = None
    immerkaer_npz_path = None

    temporal_detail = report.pop("_temporal_detail", None)
    if temporal_detail is not None:
        n_poses = len(temporal_detail["pose_ids"])
        temporal_npz_path = _write_detail_npz(
            out_dir, TEMPORAL_NPZ_NAME, {
                "pose_ids": np.asarray(temporal_detail["pose_ids"]),
                "sigma_per_corner_all": np.asarray(
                    temporal_detail["sigma_per_corner_all"], dtype=np.float64),
                "sources": np.asarray(temporal_detail["sources"]),
                "sigma_pose": np.asarray(temporal_detail["sigma_pose"],
                                         dtype=np.float64),
                "sigma_detector": np.asarray(temporal_detail["sigma_detector"],
                                             dtype=np.float64),
                "sigma_jitter": np.asarray(temporal_detail["sigma_jitter"],
                                           dtype=np.float64),
                "clamped": np.asarray(temporal_detail["clamped"], dtype=bool),
                "n_frames": np.asarray(temporal_detail["n_frames"], dtype=int),
                "n_detected": np.asarray(temporal_detail["n_detected"], dtype=int),
            }, lineage, "n_poses", n_poses)
        # json 里只留引用(汇总只是索引不是替代,D-13)
        report.setdefault("splits", {}).setdefault("train", {})[
            "detail_npz"] = TEMPORAL_NPZ_NAME

    immerkaer_detail = report.pop("_immerkaer_detail", None)
    if immerkaer_detail is not None:
        n_images = len(immerkaer_detail["image_ids"])
        immerkaer_npz_path = _write_detail_npz(
            out_dir, IMMERKAER_NPZ_NAME, {
                "image_ids": np.asarray(immerkaer_detail["image_ids"]),
                "sigma_std": np.asarray(immerkaer_detail["sigma_std"],
                                        dtype=np.float64),
                "sigma_robust": np.asarray(immerkaer_detail["sigma_robust"],
                                           dtype=np.float64),
                "split": np.asarray(immerkaer_detail["split"]),
            }, lineage, "n_images", n_images)
        report.setdefault("meta", {})["immerkaer_detail_npz"] = IMMERKAER_NPZ_NAME

    report["lineage"] = lineage
    json_path = out_dir / "noise.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return {"noise_json": json_path, "temporal_npz": temporal_npz_path,
            "immerkaer_npz": immerkaer_npz_path}


def load_noise_report(out_dir: Path | str) -> dict:
    """读回 noise.json 与全部明细 npz(allow_pickle=False),组装成一个字典。

    返回 {"report": noise.json 字典, "temporal": 明细数组字典或 None,
          "immerkaer": 明细数组字典或 None}。npz 缺文件时对应项为 None
    (失败轮的常态),noise.json 本身缺失直接抛错——下游要读的真相源不在,
    静默吞掉等于撒谎。
    """
    out_dir = Path(out_dir)
    with open(out_dir / "noise.json", encoding="utf-8") as f:
        report = json.load(f)

    def _load_npz(name: str) -> dict | None:
        npz_path = out_dir / name
        if not npz_path.is_file():
            return None
        with np.load(npz_path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    return {"report": report,
            "temporal": _load_npz(TEMPORAL_NPZ_NAME),
            "immerkaer": _load_npz(IMMERKAER_NPZ_NAME)}


# ---------------------------------------------------------------------------
# 第⑤步(CRLB)的产物契约:crlb.json + 逐模型 Fisher 明细 npz 旁车
# ---------------------------------------------------------------------------

# crlb.json 顶层契约键(下游与集成测试按这套键取数;失败时也一个不少)
CRLB_KEYS = ("schema_version", "created", "lineage", "models", "meta",
             "failure_note")

# 明细 npz 文件名模板(逐模型一份,与 crlb.json 同目录)
CRLB_DETAIL_NPZ_TEMPLATE = "fisher_{model_id}.npz"

CRLB_SCHEMA_VERSION = "crlb-v1"


def make_failure_crlb(lineage: dict, note: str) -> dict:
    """CRLB 失败契约的工厂:结构键全在、数值字段一律 None、中文说明非空。

    与 make_failure_report 同一体例(照 solve/result.py 的 make_failure):
    失败只是"数值还没算出来",不是换一种结构——models 仍给空占位,
    下游按键取数、再按 failure_note 判断该不该信数值。
    """
    text = str(note).strip()
    if not text:
        raise ValueError(
            "make_failure_crlb 需要非空的中文失败说明(failure_note),"
            "空说明等于伪造证据。请写清哪个环节、什么原因。")
    return {
        "schema_version": CRLB_SCHEMA_VERSION,
        "created": (lineage or {}).get("created"),
        "lineage": dict(lineage) if lineage else {},
        "models": {},
        "meta": {},
        "failure_note": text,
    }


def _to_jsonable(value):
    """递归把 numpy 类型转成 json 可写的 Python 原生类型。

    ndarray → 嵌套列表;numpy 标量 → Python 标量;inf/-inf 原样通过
    (零信息参数的 CRLB 记无穷大是有内容的结论,不是错误;Python json
    会写成 Infinity,本项目的产物由本项目读回,契约内自洽)。
    """
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_crlb_report(out_dir: Path | str, report: dict, lineage: dict) -> dict:
    """把 CRLB 报告落盘:crlb.json(契约键全)+ 逐模型 Fisher 明细 npz 旁车。

    参数:
        out_dir: 本轮运行目录(如 output/fisher/run_<时间戳>),不存在则创建;
        report:  run_fisher_matrix 组装的完整报告。各模型可带私有明细键
                 "_fisher_detail"(边缘化 Fisher 信息矩阵等大数组,写盘时取出
                 转 npz,绝不进 json——json 只留路径引用,D-13 汇总只是索引);
        lineage: make_lineage 组装的血缘指纹(进 crlb.json 与每个 npz 旁车)。

    返回 {"crlb_json", "detail_npz": {模型ID: npz 路径}} 路径字典
    (失败报告没有明细,detail_npz 为空字典)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 血缘先行落位再验契约:lineage 由参数传入必齐整,失败报告同样带全血缘
    report["lineage"] = lineage
    missing = [key for key in CRLB_KEYS if key not in report]
    if missing:
        raise AssertionError(
            f"crlb.json 契约键缺失:{missing}。顶层必须有这些键({len(CRLB_KEYS)}"
            f" 个,失败时也一样),请用 make_failure_crlb / run_fisher_matrix"
            f" 组装完整的报告。")

    detail_npz: dict[str, Path] = {}
    models = report.get("models") or {}
    for model_id, model_block in models.items():
        detail = model_block.pop("_fisher_detail", None)
        if detail is None:
            continue
        npz_name = CRLB_DETAIL_NPZ_TEMPLATE.format(model_id=model_id)
        npz_path = out_dir / npz_name
        np.savez(npz_path, **{
            key: np.asarray(arr) for key, arr in detail.items()})
        # 旁车 meta:血缘必带,中文 meta 原样可读
        meta = {
            "file": npz_name,
            "model_id": model_id,
            "n_arrays": len(detail),
            "lineage": lineage,
        }
        with open(out_dir / f"{Path(npz_name).stem}.meta.json",
                  "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        detail_npz[model_id] = npz_path
        # json 里只留引用(汇总只是索引不是替代,D-13)
        model_block["detail_npz"] = npz_name

    json_path = out_dir / "crlb.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report), f, ensure_ascii=False, indent=2)
    return {"crlb_json": json_path, "detail_npz": detail_npz}


def load_crlb_report(out_dir: Path | str) -> dict:
    """读回 crlb.json 与全部逐模型明细 npz(allow_pickle=False)。

    返回 {"report": crlb.json 字典, "detail": {模型ID: 数组字典}}。
    某模型 npz 缺文件时对应项为 None(失败轮的常态),crlb.json 本身缺失
    直接抛错——下游要读的真相源不在,静默吞掉等于撒谎。
    """
    out_dir = Path(out_dir)
    with open(out_dir / "crlb.json", encoding="utf-8") as f:
        report = json.load(f)

    def _load_npz(name: str) -> dict | None:
        npz_path = out_dir / name
        if not npz_path.is_file():
            return None
        with np.load(npz_path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    detail: dict[str, dict | None] = {}
    for model_id, model_block in (report.get("models") or {}).items():
        npz_name = model_block.get("detail_npz")
        detail[model_id] = _load_npz(npz_name) if npz_name else None

    return {"report": report, "detail": detail}
