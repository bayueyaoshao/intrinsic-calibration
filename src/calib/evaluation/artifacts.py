"""optimizer.json / evaluate.json 产物契约与落盘:优化器对比(D-13)与
评估交付(选型 + 实验 + 交付)两层的输出侧。

两层结构(D-13 产物保真:汇总只是索引不是替代):
- optimizer.json:顶层契约键 + 每模型推荐优化配置(recommended_solver)+
  16 组合对比表(combos)+ 多起点一致性结果(multistart)——它是 04-04
  起的全部下游实验与报告读数的"目录";
- evaluate.json:选型交付轮的主契约——优化器推荐、四实验(cv/bootstrap/
  slices/lopo)、来源分离、判行表与选型结论(selection)、全量重训交付
  (delivery)全部落一段;逐折分配表、逐次重采样参数、逐姿态误差、test
  逐张评分等明细大数组转 experiments/<段名>.npz 旁车,json 只留引用
  ——文件是唯一真相源,汇总表只是索引不是替代;
- 明细 npz 旁车:纯数值数组 + 定长 unicode 编号列,读写一律
  allow_pickle=False(npz 的 pickle 模式可执行任意代码,项目既定防线),
  每份 npz 配同名 .meta.json 且血缘必带。

失败契约(照 solve/result.py 的 make_failure 体例):命令中途失败时
主 json 照样落盘——结构键一个不少、数值字段一律 null(落盘成 JSON
null)、failure_note 非空中文写清原因。上游对成败一视同仁:
先验键、再取数。

血缘(T-04-02-04 / T-04-05-04):主 json 与明细 npz 的 *.meta.json 旁车
都带同一份血缘指纹(调用方经 util/lineage.py 的 make_lineage 组装后传
入,上游 = corners npz + manifest + optimizer.json 等上游产物),产物
来源可考。

本文件体例整文件照 src/calib/noise/artifacts.py,不跨模块 import 其
私有函数(各阶段产物契约各自成文,改一边不至于震塌另一边)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# optimizer.json 顶层契约键(下游与集成测试按这套键取数;失败时也一个不少)
OPTIMIZER_KEYS = (
    "schema_version",
    "created",
    "lineage",
    "recommended_solver",
    "combos",
    "multistart",
    "meta",
    "failure_note",
)

# 明细 npz 文件名(与 optimizer.json 同目录;当前编排没有大数组,留作扩展位)
OPTIMIZER_DETAIL_NPZ_NAME = "optimizer_detail.npz"

SCHEMA_VERSION = "optimizer-v1"


def make_failure_optimizer(lineage: dict, note: str) -> dict:
    """失败契约的工厂:结构键全在、数值字段一律 None、中文说明非空。

    与 noise 侧 make_failure_report 同一体例:失败只是"数值还没算出来",
    不是换一种结构——recommended_solver / combos 仍给空占位,下游按键
    取数、再按 failure_note 判断该不该信数值;multistart 没跑就 None。
    """
    text = str(note).strip()
    if not text:
        raise ValueError(
            "make_failure_optimizer 需要非空的中文失败说明(failure_note),"
            "空说明等于伪造证据。请写清哪个环节、什么原因。"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created": (lineage or {}).get("created"),
        "lineage": dict(lineage) if lineage else {},
        "recommended_solver": {},
        "combos": [],
        "multistart": None,
        "meta": {},
        "failure_note": text,
    }


def _check_optimizer_contract(report: dict) -> None:
    """写盘前先验契约:顶层结构键一个不少,缺键当场中文报错。"""
    missing = [key for key in OPTIMIZER_KEYS if key not in report]
    if missing:
        raise AssertionError(
            f"optimizer.json 契约键缺失:{missing}。顶层必须有这些键"
            f"({len(OPTIMIZER_KEYS)} 个,失败时也一样),请用 "
            f"make_failure_optimizer / run_optimizer_grid 组装完整的报告。"
        )


def _to_jsonable(value):
    """递归把 numpy 类型转成 json 可写的 Python 原生类型。

    ndarray → 嵌套列表;numpy 标量 → Python 标量;inf/-inf 原样通过
    (零信息参数记无穷大是有内容的结论,不是错误;Python json 会写成
    Infinity,本项目的产物由本项目读回,契约内自洽)。体例照 noise 侧
    同名函数,逐字一致——两处各存一份,不跨模块借私有件。
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


def save_optimizer_report(out_dir: Path | str, report: dict, lineage: dict) -> dict:
    """把优化器对比报告落盘:optimizer.json(契约键全)+ 明细 npz 旁车。

    参数:
        out_dir: 本轮运行目录(如 output/evaluate/run_<时间戳>),不存在则创建;
        report:  run_optimizer_grid / run_multistart 组装的完整报告。可带一个
                 私有明细键 "_optimizer_detail"(大数组字典,写盘时取出转
                 npz,绝不进 json——json 只留路径引用,D-13 汇总只是索引);
        lineage: make_lineage 组装的血缘指纹(进 optimizer.json 与 npz 旁车)。

    返回 {"optimizer_json", "detail_npz"} 路径字典
    (没有明细时 detail_npz 为 None)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 血缘先行落位再验契约:lineage 由参数传入必齐整,失败报告同样带全血缘
    report["lineage"] = lineage
    _check_optimizer_contract(report)

    detail_npz_path = None
    detail = report.pop("_optimizer_detail", None)
    if detail is not None:
        np.savez(
            out_dir / OPTIMIZER_DETAIL_NPZ_NAME,
            **{key: np.asarray(arr) for key, arr in detail.items()},
        )
        # 旁车 meta:血缘必带,中文 meta 原样可读
        meta = {
            "file": OPTIMIZER_DETAIL_NPZ_NAME,
            "n_arrays": len(detail),
            "lineage": lineage,
        }
        with open(
            out_dir / f"{Path(OPTIMIZER_DETAIL_NPZ_NAME).stem}.meta.json", "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        detail_npz_path = out_dir / OPTIMIZER_DETAIL_NPZ_NAME
        # json 里只留引用(汇总只是索引不是替代,D-13)
        report.setdefault("meta", {})["detail_npz"] = OPTIMIZER_DETAIL_NPZ_NAME

    json_path = out_dir / "optimizer.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report), f, ensure_ascii=False, indent=2)
    return {"optimizer_json": json_path, "detail_npz": detail_npz_path}


def load_optimizer_report(out_dir: Path | str) -> dict:
    """读回 optimizer.json 与明细 npz(allow_pickle=False)。

    返回 {"report": optimizer.json 字典, "detail": 明细数组字典或 None}。
    明细 npz 缺文件时 detail 为 None(失败轮的常态),optimizer.json 本身
    缺失直接抛中文错——下游要读的真相源不在,静默吞掉等于撒谎。
    """
    out_dir = Path(out_dir)
    json_path = out_dir / "optimizer.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"optimizer.json 不在:{json_path}。该目录下没有优化器对比产物,"
            f"请先跑优化器对比(或核对 run_<时间戳> 目录名是否传对)。"
        )
    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)

    detail = None
    npz_name = (report.get("meta") or {}).get("detail_npz")
    if npz_name:
        npz_path = out_dir / npz_name
        if npz_path.is_file():
            # npz 读取强制 allow_pickle=False:npz 的 pickle 模式可以执行
            # 任意代码,关死它(项目全部读回路径的既定防线)
            with np.load(npz_path, allow_pickle=False) as data:
                detail = {key: data[key] for key in data.files}

    return {"report": report, "detail": detail}


# ---------------------------------------------------------------------------
# 第⑥步(评估交付)的产物契约:evaluate.json 主契约 + experiments/*.npz 明细旁车
# ---------------------------------------------------------------------------

# evaluate.json 顶层契约键(下游与集成测试按这套键取数;失败时也一个不少)
EVALUATE_KEYS = (
    "schema_version",
    "created",
    "lineage",
    "optimizer",
    "cv",
    "bootstrap",
    "slices",
    "lopo",
    "source_separation",
    "selection",
    "delivery",
    "meta",
    "failure_note",
)

SCHEMA_VERSION_EVALUATE = "evaluate-v1"

# 明细 npz 旁车的子目录名(与 evaluate.json 同轮目录下)
EVALUATE_DETAIL_DIR = "experiments"

# 各实验段的私有明细键 → 段名:写盘时取出转 experiments/<段名>.npz 旁车,
# json 只留 detail_npz 引用(汇总只是索引不是替代,D-13 产物保真)
_EVALUATE_DETAIL_SECTIONS = (
    ("_cv_detail", "cv"),
    ("_bootstrap_detail", "bootstrap"),
    ("_slice_detail", "slices"),
    ("_lopo_detail", "lopo"),
    ("_delivery_detail", "delivery"),
)


def make_failure_evaluate(lineage: dict, note: str) -> dict:
    """失败契约的工厂:结构键全在、各实验段一律 None、中文说明非空。

    与 optimizer 侧 make_failure_optimizer 同一体例:失败只是"数值还没算
    出来",不是换一种结构——下游按键取数、再按 failure_note 判断该不该
    信数值。各实验段置 None(不是空字典):让"没跑"与"跑了但为空"可区分。
    """
    text = str(note).strip()
    if not text:
        raise ValueError(
            "make_failure_evaluate 需要非空的中文失败说明(failure_note),"
            "空说明等于伪造证据。请写清哪个环节、什么原因。"
        )
    return {
        "schema_version": SCHEMA_VERSION_EVALUATE,
        "created": (lineage or {}).get("created"),
        "lineage": dict(lineage) if lineage else {},
        "optimizer": None,
        "cv": None,
        "bootstrap": None,
        "slices": None,
        "lopo": None,
        "source_separation": None,
        "selection": None,
        "delivery": None,
        "meta": {},
        "failure_note": text,
    }


def _check_evaluate_contract(report: dict) -> None:
    """写盘前先验契约:顶层结构键一个不少,缺键当场中文报错。"""
    missing = [key for key in EVALUATE_KEYS if key not in report]
    if missing:
        raise AssertionError(
            f"evaluate.json 契约键缺失:{missing}。顶层必须有这些键"
            f"({len(EVALUATE_KEYS)} 个,失败时也一样),请用 "
            f"make_failure_evaluate / 命令体组装完整的报告。"
        )


def _flatten_detail(value, prefix: str = "") -> dict:
    """明细字典拍平成 {键: 数组} 一层(np.savez 只收平面键值)。

    嵌套字典(如 _slice_detail / _bootstrap_detail 的 {模型: {数组}})
    用双下划线连接父子键(m1__group_index);叶子值交 np.asarray 转
    数组(标量成 0 维数组,读回原样)。
    """
    flat: dict = {}
    if isinstance(value, dict):
        for key, val in value.items():
            child = f"{prefix}__{key}" if prefix else str(key)
            flat.update(_flatten_detail(val, child))
    else:
        flat[prefix] = value
    return flat


def save_evaluate_report(out_dir: Path | str, report: dict, lineage: dict) -> dict:
    """把评估交付报告落盘:evaluate.json(契约键全)+ experiments/*.npz 旁车。

    参数:
        out_dir: 本轮运行目录(如 output/evaluate/run_<时间戳>),不存在则创建;
        report:  命令体组装的完整报告。可带五个私有明细键(写盘时取出转
                 npz 旁车,绝不进 json——json 各段只留 detail_npz 引用,
                 D-13 汇总只是索引):_cv_detail / _bootstrap_detail /
                 _slice_detail / _lopo_detail / _delivery_detail;
        lineage: make_lineage 组装的血缘指纹(进 evaluate.json 与每个
                 npz 旁车的 .meta.json,T-04-05-04)。

    返回 {"evaluate_json", "detail_npz"} 路径字典(detail_npz 是
    {段名: npz 路径} 字典,没有明细时为空字典)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 血缘先行落位再验契约:lineage 由参数传入必齐整,失败报告同样带全血缘
    report["lineage"] = lineage
    _check_evaluate_contract(report)

    detail_dir = out_dir / EVALUATE_DETAIL_DIR
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_paths: dict[str, Path] = {}
    for private_key, section in _EVALUATE_DETAIL_SECTIONS:
        detail = report.pop(private_key, None)
        if detail is None:
            continue
        section_val = report.get(section)
        if not isinstance(section_val, dict):
            raise AssertionError(  # noqa: TRY004  项目统一中文契约报错(体例照 _check_optimizer_contract),不按类型换 TypeError
                f"明细键 {private_key} 带了明细数组,但主契约的 {section} 段"
                f"不是字典(拿到 {type(section_val).__name__}):明细旁车必须"
                f"挂在对应实验段下(json 只留 detail_npz 引用),"
                f"请检查组装顺序。"
            )
        flat = _flatten_detail(detail)
        npz_name = f"{section}.npz"
        np.savez(detail_dir / npz_name, **{key: np.asarray(arr) for key, arr in flat.items()})
        # 旁车 meta:血缘必带,中文 meta 原样可读(T-04-05-04)
        meta = {
            "file": f"{EVALUATE_DETAIL_DIR}/{npz_name}",
            "section": section,
            "n_arrays": len(flat),
            "lineage": lineage,
        }
        with open(detail_dir / f"{Path(npz_name).stem}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        # json 里只留引用(汇总只是索引不是替代,D-13)
        section_val["detail_npz"] = f"{EVALUATE_DETAIL_DIR}/{npz_name}"
        detail_paths[section] = detail_dir / npz_name

    json_path = out_dir / "evaluate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report), f, ensure_ascii=False, indent=2)
    return {"evaluate_json": json_path, "detail_npz": detail_paths}


def load_evaluate_report(out_dir: Path | str) -> dict:
    """读回 evaluate.json 与全部明细 npz(allow_pickle=False)。

    返回 {"report": evaluate.json 字典, "details": {段名: 明细数组字典}}。
    各段 detail_npz 引用的 npz 缺文件直接中文报错——引用写在主契约里而
    文件不在,说明产物不完整,静默吞掉等于撒谎;evaluate.json 本身缺失
    同样直接抛中文错。
    """
    out_dir = Path(out_dir)
    json_path = out_dir / "evaluate.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"evaluate.json 不在:{json_path}。该目录下没有评估交付产物,"
            f"请先跑 calib evaluate(或核对 run_<时间戳> 目录名是否传对)。"
        )
    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)

    details: dict[str, dict] = {}
    for _private_key, section in _EVALUATE_DETAIL_SECTIONS:
        section_val = report.get(section)
        if not isinstance(section_val, dict):
            continue
        npz_name = section_val.get("detail_npz")
        if not npz_name:
            continue
        npz_path = out_dir / npz_name
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"evaluate.json 的 {section} 段引用了明细旁车 {npz_name},"
                f"但文件不在:{npz_path}。明细是汇总的原始证据,缺失说明"
                f"产物不完整,请重跑 calib evaluate。"
            )
        # npz 读取强制 allow_pickle=False:npz 的 pickle 模式可以执行
        # 任意代码,关死它(项目全部读回路径的既定防线)
        with np.load(npz_path, allow_pickle=False) as data:
            details[section] = {key: data[key] for key in data.files}

    return {"report": report, "details": details}
