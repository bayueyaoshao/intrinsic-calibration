"""analyze.json 产物契约与落盘:误差归因六块的输出侧。

六块结构(产物保真,汇总只是索引不是替代):
- budget:误差预算(实际误差 vs CRLB,ANA-01);
- residual_structure:残差方向分解(按视图/画面区域/径向分带,ANA-02);
- test_xyz:test 集误差的相机坐标系三分量分解(ANA-03);
- sensitivity:参数敏感度排序与应对建议(ANA-04);
- corner_opt:角点合成退化寻优实验(COR-02);
- anisotropy:同姿态连拍角点 x/y 方向重复性各向异性(COR-03)。

六块的明细大数组各挂一个私有明细键,写盘时转 analysis/<段名>.npz 旁车,
json 各段只留 detail_npz 引用——文件是唯一真相源,汇总表只是索引;
每份 npz 配同名 .meta.json 且血缘必带。

失败契约(照 evaluation 侧 make_failure_evaluate 体例):命令中途失败时
主 json 照样落盘——结构键一个不少、六块数值段一律 None(不是空字典,
让"没跑"与"跑了但为空"可区分)、failure_note 非空中文写清原因。
上游对成败一视同仁:先验键、再取数。

npz 读写一律 allow_pickle=False(npz 的 pickle 模式可执行任意代码,
项目全部读回路径的既定防线);血缘由调用方经 util/lineage.py 的
make_lineage 组装后传入,主 json 与每份 npz 的 .meta.json 带同一份。

本文件体例整文件照 src/calib/evaluation/artifacts.py,不跨模块 import 其
私有函数(各阶段产物契约各自成文,改一边不至于震塌另一边)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# analyze.json 顶层契约键(下游与集成测试按这套键取数;失败时也一个不少):
# 结构三键 + 误差归因六块 + meta + failure_note = 11 键
ANALYZE_KEYS = (
    "schema_version",
    "created",
    "lineage",
    "budget",
    "residual_structure",
    "test_xyz",
    "sensitivity",
    "corner_opt",
    "anisotropy",
    "meta",
    "failure_note",
)

SCHEMA_VERSION_ANALYZE = "analyze-v1"

# 明细 npz 旁车的子目录名(与 analyze.json 同轮目录下)
ANALYZE_DETAIL_DIR = "analysis"

# 各需求块的私有明细键 → 段名:写盘时取出转 analysis/<段名>.npz 旁车,
# json 只留 detail_npz 引用(汇总只是索引不是替代,D-13 产物保真)
_ANALYZE_DETAIL_SECTIONS = (
    ("_budget_detail", "budget"),
    ("_residual_detail", "residual_structure"),
    ("_test_xyz_detail", "test_xyz"),
    ("_sensitivity_detail", "sensitivity"),
    ("_corner_opt_detail", "corner_opt"),
    ("_anisotropy_detail", "anisotropy"),
)


def make_failure_analyze(lineage: dict, note: str) -> dict:
    """失败契约的工厂:结构键全在、六块数值段一律 None、中文说明非空。

    与 evaluation 侧 make_failure_evaluate 同一体例:失败只是"数值还没
    算出来",不是换一种结构——下游按键取数、再按 failure_note 判断该不
    该信数值。六块置 None(不是空字典):让"没跑"与"跑了但为空"可区分。
    """
    text = str(note).strip()
    if not text:
        raise ValueError(
            "make_failure_analyze 需要非空的中文失败说明(failure_note),"
            "空说明等于伪造证据。请写清哪个环节、什么原因。"
        )
    return {
        "schema_version": SCHEMA_VERSION_ANALYZE,
        "created": (lineage or {}).get("created"),
        "lineage": dict(lineage) if lineage else {},
        "budget": None,
        "residual_structure": None,
        "test_xyz": None,
        "sensitivity": None,
        "corner_opt": None,
        "anisotropy": None,
        "meta": {},
        "failure_note": text,
    }


def _check_analyze_contract(report: dict) -> None:
    """写盘前先验契约:顶层结构键一个不少,缺键当场中文报错。"""
    missing = [key for key in ANALYZE_KEYS if key not in report]
    if missing:
        raise AssertionError(
            f"analyze.json 契约键缺失:{missing}。顶层必须有这些键"
            f"({len(ANALYZE_KEYS)} 个,失败时也一样),请用 "
            f"make_failure_analyze / 命令体组装完整的报告。"
        )


def _to_jsonable(value):
    """递归把 numpy 类型转成 json 可写的 Python 原生类型。

    ndarray → 嵌套列表;numpy 标量 → Python 标量;inf/-inf 原样通过
    (零信息参数记无穷大是有内容的结论,不是错误;体例照 evaluation 侧
    同名函数,逐字一致——两处各存一份,不跨模块借私有件)。
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


def _flatten_detail(value, prefix: str = "") -> dict:
    """明细字典拍平成 {键: 数组} 一层(np.savez 只收平面键值)。

    嵌套字典用双下划线连接父子键(如 budget 块的 meas_rows 直接一层,
    多模型明细则成 m2__group_params 形态);叶子值交 np.asarray 转
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


def save_analyze_report(out_dir: Path | str, report: dict, lineage: dict) -> dict:
    """把误差归因报告落盘:analyze.json(契约键全)+ analysis/*.npz 旁车。

    参数:
        out_dir: 本轮运行目录(如 output/analyze/run_<时间戳>),不存在则创建;
        report:  命令体组装的完整报告。可带六个私有明细键(写盘时取出转
                 npz 旁车,绝不进 json——json 各段只留 detail_npz 引用,
                 D-13 汇总只是索引):_budget_detail / _residual_detail /
                 _test_xyz_detail / _sensitivity_detail / _corner_opt_detail /
                 _anisotropy_detail;
        lineage: make_lineage 组装的血缘指纹(进 analyze.json 与每个
                 npz 旁车的 .meta.json)。

    返回 {"analyze_json", "detail_npz"} 路径字典(detail_npz 是
    {段名: npz 路径} 字典,没有明细时为空字典)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 血缘先行落位再验契约:lineage 由参数传入必齐整,失败报告同样带全血缘
    report["lineage"] = lineage
    _check_analyze_contract(report)

    detail_dir = out_dir / ANALYZE_DETAIL_DIR
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_paths: dict[str, Path] = {}
    for private_key, section in _ANALYZE_DETAIL_SECTIONS:
        detail = report.pop(private_key, None)
        if detail is None:
            continue
        section_val = report.get(section)
        if not isinstance(section_val, dict):
            raise AssertionError(  # noqa: TRY004  项目统一中文契约报错(体例照 evaluation 侧),不按类型换 TypeError
                f"明细键 {private_key} 带了明细数组,但主契约的 {section} 段"
                f"不是字典(拿到 {type(section_val).__name__}):明细旁车必须"
                f"挂在对应需求块段下(json 只留 detail_npz 引用),"
                f"请检查组装顺序。"
            )
        flat = _flatten_detail(detail)
        npz_name = f"{section}.npz"
        np.savez(detail_dir / npz_name, **{key: np.asarray(arr) for key, arr in flat.items()})
        # 旁车 meta:血缘必带,中文 meta 原样可读
        meta = {
            "file": f"{ANALYZE_DETAIL_DIR}/{npz_name}",
            "section": section,
            "n_arrays": len(flat),
            "lineage": lineage,
        }
        with open(detail_dir / f"{Path(npz_name).stem}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        # json 里只留引用(汇总只是索引不是替代,D-13)
        section_val["detail_npz"] = f"{ANALYZE_DETAIL_DIR}/{npz_name}"
        detail_paths[section] = detail_dir / npz_name

    json_path = out_dir / "analyze.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report), f, ensure_ascii=False, indent=2)
    return {"analyze_json": json_path, "detail_npz": detail_paths}


def load_analyze_report(out_dir: Path | str) -> dict:
    """读回 analyze.json 与全部明细 npz(allow_pickle=False)。

    返回 {"report": analyze.json 字典, "details": {段名: 明细数组字典}}。
    各段 detail_npz 引用的 npz 缺文件直接中文报错——引用写在主契约里而
    文件不在,说明产物不完整,静默吞掉等于撒谎;analyze.json 本身缺失
    同样直接抛中文错。
    """
    out_dir = Path(out_dir)
    json_path = out_dir / "analyze.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"analyze.json 不在:{json_path}。该目录下没有误差归因产物,"
            f"请先跑 calib analyze(或核对 run_<时间戳> 目录名是否传对)。"
        )
    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)

    details: dict[str, dict] = {}
    for _private_key, section in _ANALYZE_DETAIL_SECTIONS:
        section_val = report.get(section)
        if not isinstance(section_val, dict):
            continue
        npz_name = section_val.get("detail_npz")
        if not npz_name:
            continue
        npz_path = out_dir / npz_name
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"analyze.json 的 {section} 段引用了明细旁车 {npz_name},"
                f"但文件不在:{npz_path}。明细是汇总的原始证据,缺失说明"
                f"产物不完整,请重跑 calib analyze。"
            )
        # npz 读取强制 allow_pickle=False:npz 的 pickle 模式可以执行
        # 任意代码,关死它(项目全部读回路径的既定防线)
        with np.load(npz_path, allow_pickle=False) as data:
            details[section] = {key: data[key] for key in data.files}

    return {"report": report, "details": details}
