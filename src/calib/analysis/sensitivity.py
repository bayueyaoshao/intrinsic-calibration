"""ANA-04 参数敏感度排序与应对建议:哪个方向最刹不住车、下一步补什么。

白话背景:"参数敏感度"回答的是——往哪个参数组合方向挪,解最刹不住车。
本模块把两条**互相独立**的证据对账后落排序:
1. 理论谱:Fisher 信息(即"数据对每个参数方向各有多少约束"的信息量
   矩阵)的协方差特征分解,fisher npz 里已存好,只消费不重算;
2. 经验谱:bootstrap 重采样参数(50 次重标定的实际散布)与 8 组分组
   独立标定参数的样本协方差特征分解。
两条证据的最大特征方向比夹角(方向一致 = 结论稳),逐参数载荷并排表
随段落盘。

天平纪律(findings 结论 1 反面教材):一切比较用**像素当量**公共货币——
沿特征方向走一步(σ 级步长)参数,在同一参考位姿下投影全部角点取最大
位移(直接复用 evaluation.experiments._px_dispersion,与 budget 段同一
换算)。相对百分比对近零参数(p1/p2)无意义:百分比晃 3–7% 画面上挪
不到 0.02px,不上像素当量天平的排序没有判读价值。

建议纪律(裁定③):建议规则表只生成文案,绝不自动执行任何补拍/重标定
动作(回环归下一阶段);每条建议末尾带固定声明。

解不唯一披露(结论 3):bootstrap 最大位移当量、多起点位移当量与判定
如实透传,不给"参数精确到 X"式承诺。

what-if 补拍预演(裁定②):仅 full 档执行——合成"棋盘够到画面四角"的
虚拟位姿,并入 Fisher 信息预算"补几个姿态能再压多少"。合成证据必须
显式声明"假设非实测"(证据链性质),quick 档干脆不做。

FIM 组装红线:谱只消费 fisher npz 不重算;重算只发生在 what-if 的合成
姿态上,且只调 noise.fisher.marginalized_fisher,本模块绝不重写 FIM
组装(D-10 单一数学源)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calib.evaluation.experiments import _px_dispersion
from calib.models.base import transform_to_camera
from calib.noise.fisher import marginalized_fisher

# 段模式版本:敏感度段的结构约定(加键不改键算兼容,删键必须升版本)
SCHEMA_SENSITIVITY = "analyze-sensitivity-v1"

# 理论谱取前几大特征方向算像素当量(最大方向定建议,再取两个对照)
_N_TOP_DIRECTIONS = 3

# 近零参数的像素当量线:低于它判"画面影响可忽略,不治"
# (findings 结论 1:p1/p2 百分比晃 3–7% 但像素影响 < 0.02px)
_PX_NEGLIGIBLE = 0.05

# "载荷集中在"的判据线:组内归一化 |载荷| 占比之和
_DOMINANT_SHARE = 0.5
# 分支 2 里单个像素参数算"参与耦合"的最低占比
_PAIR_MEMBER_SHARE = 0.15
# 分支 3 里 p1/p2 在载荷表算"可见"的最低占比
_NEARZERO_VISIBLE_SHARE = 0.05

# 像素参数组(分支 2 的耦合候选)
_PIXEL_PARAMS = ("fx", "fy", "cx", "cy")

# 固定文案(原文落盘,判读纪律与声明不许每轮重写)
_ADVICE_NO_AUTO = "本建议只提示方向,不自动执行任何补拍/重标定(回环归下一阶段)"
_SYNTHETIC_NOTE = (
    "补拍预演为合成假设(虚拟位姿非实拍),证据链性质是"
    "'如果补拍能压多少'的预期,不是实测结论"
)
_QUICK_WHAT_IF_NOTE = "quick 档不做 what-if(裁定②):补拍预演属合成假设,仅 full 档执行"
_UNIQUE_NOTE = (
    "参数沿软方向滑动约 0.1–2 px 当量是本批数据的固有性质(同一 RMS 谷底上"
    "解不唯一),不存在「参数精确到 X」的承诺;任何对账容差不得小于该滑动量级"
)


def _require(mapping, key: str, what: str):
    """从上游报告取键,缺失当场中文报错(体例照 budget.py 的取数防线)。"""
    if not isinstance(mapping, dict) or key not in mapping:
        raise AssertionError(
            f"上游报告缺键 {key}({what})。敏感度排序按契约消费 evaluate.json "
            f"与 fisher npz 的落盘段,缺键多半是产物与当前代码不是同一套契约,"
            f"或串了轮次,请先重跑 calib evaluate / calib fisher 再来。"
        )
    return mapping[key]


def _raw_direction(v_dimless: np.ndarray, scale_dimless: np.ndarray) -> np.ndarray:
    """无量纲坐标方向 → 原始参数坐标方向:v_raw = v_dimless × scale。

    Fisher 谱存的是无量纲坐标(像素内参 ÷ 焦距)下的特征向量;落到原始
    参数空间(像素、畸变无量纲小数)要逐分量乘回 scale。往返自检恒等式:
    raw ÷ scale == v_dimless(测试锁定逐位还原)。
    """
    return np.asarray(v_dimless, dtype=np.float64) * np.asarray(scale_dimless, dtype=np.float64)


def _loads_share(vec: np.ndarray, names: list[str]) -> dict:
    """方向向量的逐参数载荷表:|分量| 占比(和为 1),按占比降序排布。"""
    arr = np.asarray(vec, dtype=np.float64)
    denom = float(np.sum(np.abs(arr)))
    share = np.abs(arr) / denom if denom > 0 else np.zeros_like(arr)
    order = np.argsort(-share, kind="stable")
    return {names[i]: float(share[i]) for i in order}


def _loads_signed(vec: np.ndarray, names: list[str]) -> dict:
    """方向向量的逐参数带符号载荷(单位化),按 |分量| 降序排布。"""
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    unit = arr / norm if norm > 0 else arr
    order = np.argsort(-np.abs(unit), kind="stable")
    return {names[i]: float(unit[i]) for i in order}


def _read_spectrum_npz(fisher_npz_path: Path | str, model_id: str,
                       need_f640: bool) -> dict:
    """读回 fisher npz 的特征谱键(allow_pickle=False 写死;谱只消费不重算)。

    键契约(2026-10-21 实查 fisher_m2.npz):eigvals_cov (9,) /
    eigvecs_cov (9,9) / scale_dimless (9,) / param_names (9,) /
    F_marg_views640 (9,9)(仅 full 档 what-if 需要)。
    """
    path = Path(fisher_npz_path)
    if not path.is_file():
        raise AssertionError(
            f"上游 fisher npz 不在:{path}。理论特征谱是敏感度排序的取数源,"
            f"请先跑 calib fisher(或核对 run_<时间戳> 目录名是否传对)。"
        )
    want = ["eigvals_cov", "eigvecs_cov", "scale_dimless", "param_names"]
    if need_f640:
        want.append("F_marg_views640")
    out: dict = {}
    with np.load(path, allow_pickle=False) as data:
        for key in want:
            if key not in data.files:
                raise AssertionError(
                    f"上游 fisher npz({path})缺键 {key}(给模型 {model_id} "
                    f"对账用)。敏感度排序按契约消费特征谱,缺键请先重跑 "
                    f"calib fisher 再来。"
                )
            out[key] = data[key]
    return out


def _read_params_matrix(npz_path: Path | str, key: str, what: str) -> np.ndarray:
    """读回 bootstrap/slices npz 里的参数样本矩阵 (n, P),过滤非有限行。"""
    path = Path(npz_path)
    if not path.is_file():
        raise AssertionError(
            f"上游 npz 不在:{path}({what})。经验谱的取数源缺失,"
            f"请先重跑 calib evaluate 再来。"
        )
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise AssertionError(
                f"上游 npz({path})缺键 {key}({what})。quick 档产物只含"
                f"当选模型,请核对 model_id 与 evaluate.json 的 "
                f"selection.selected 是否一致。"
            )
        arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim != 2:
        raise AssertionError(
            f"{what}({path} 的 {key})形状不合:期望 (样本数, 参数数),"
            f"实际 {arr.shape}。请重跑 calib evaluate。"
        )
    finite = arr[np.isfinite(arr).all(axis=1)]
    if finite.shape[0] < 2:
        raise AssertionError(
            f"{what}({path} 的 {key})有限样本不足(剩 {finite.shape[0]} 行,"
            f"至少要 2 行才能算散布)。样本协方差无从谈起,请重跑 calib evaluate。"
        )
    return finite


def _principal_direction(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """样本协方差的最大特征方向(原始参数空间)与最大特征值。"""
    cov = np.cov(samples, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = int(np.argmax(eigvals))
    return eigvecs[:, idx], float(eigvals[idx])


def _param_px_equiv(name: str, sigma: float, params: np.ndarray, model,
                    object_points: np.ndarray, ref_rvec: np.ndarray,
                    ref_tvec: np.ndarray) -> float:
    """单参数 σ 级晃动的像素当量(与 budget 段同一把尺 _px_dispersion)。"""
    unit = np.zeros_like(params)
    unit[list(model.param_names).index(name)] = 1.0
    return float(_px_dispersion(
        [params, params + float(sigma) * unit],
        model, object_points, ref_rvec, ref_tvec))


def build_advice(directions: list[dict], px_equivs: dict,
                 coverage_section: dict | None, cfg) -> list[dict]:
    """应对建议规则表(findings 结论 5 判据表 + pipeline.md 回头路的产品化)。

    参数:
        directions:       top 方向列表,每项 {"rank", "loads"(占比字典),
                          "px_equiv"(该方向一步长的像素当量)};
        px_equivs:        {"p1": float|None, "p2": float|None}(近零参数的
                          σ 级像素当量,供分支 3 判"治不治");
        coverage_section: residual_structure 段(缺席传 None,覆盖判据退化
                          为"覆盖未知"分支并中文注明);
        cfg:              全量配置(补拍预演的姿态数与目标 r 读 analyze 段)。

    返回 advice 列表,每条 {"direction", "evidence", "advice"};每条 advice
    末尾附固定声明"本建议只提示方向,不自动执行任何补拍/重标定"(裁定③)。
    """
    advice: list[dict] = []

    # ---- 分支一:最大方向载荷集中在 k2/k3 → 按覆盖足迹判"补不补边缘" ----
    top = directions[0]
    k23_share = top["loads"].get("k2", 0.0) + top["loads"].get("k3", 0.0)
    if k23_share >= _DOMINANT_SHARE:
        direction = (f"最大敏感方向集中在 k2/k3(高阶径向畸变,"
                     f"载荷占比 {k23_share:.0%})")
        evidence = (f"该方向 σ 级步长的像素当量 {top['px_equiv']:.3f}px")
        interval = None
        if isinstance(coverage_section, dict):
            train = coverage_section.get("train") or {}
            coverage = train.get("coverage") or {}
            interval = coverage.get("extrapolation_interval")
        if coverage_section is None:
            advice.append({
                "direction": direction,
                "evidence": evidence + ";覆盖足迹段缺席,覆盖判据未知",
                "advice": ("覆盖判据未知:请先补跑残差方向分解拿到覆盖足迹,"
                           "确认最远观测 r 到画面四角之间是否零观测(纯外推),"
                           "再决定补边缘姿态的规模;k2/k3 软方向本身已成立。"
                           + _ADVICE_NO_AUTO),
            })
        elif interval:
            n_poses = int(cfg.analyze.what_if_n_poses)
            target_r = float(cfg.analyze.what_if_target_r)
            advice.append({
                "direction": direction,
                "evidence": (evidence + f";当前最远观测 r={float(interval[0]):.3f},"
                             f"画面四角 r={float(interval[1]):.3f},之间零观测(纯外推)"),
                "advice": (f"补 {n_poses} 个够到画面四角(目标 r ≥ {target_r})的"
                           f"新姿态,不必重拍全部,每姿态 1–2 张即可"
                           f"(findings 结论 5 手段 1)。" + _ADVICE_NO_AUTO),
            })
        else:
            advice.append({
                "direction": direction,
                "evidence": evidence + ";覆盖足迹显示已全域覆盖(外推区间为空)",
                "advice": ("覆盖判据:已全域覆盖,k2/k3 软方向当前覆盖已撑住,"
                           "无需补边缘姿态。" + _ADVICE_NO_AUTO),
            })

    # ---- 分支二:像素参数间强耦合 → 补转角多样性 ----
    for d in directions:
        pix = {k: v for k, v in d["loads"].items() if k in _PIXEL_PARAMS}
        pix_share = float(sum(pix.values()))
        strong = [k for k, v in pix.items() if v >= _PAIR_MEMBER_SHARE]
        if pix_share >= _DOMINANT_SHARE and len(strong) >= 2:
            advice.append({
                "direction": (f"方向 {d['rank']} 集中在像素参数 "
                              f"{'、'.join(strong)}(占比 {pix_share:.0%})"),
                "evidence": (f"像素参数间强耦合,σ 级步长像素当量 "
                             f"{d['px_equiv']:.3f}px"),
                "advice": ("补转角(旋转)多样性姿态,改善像素内参与外参的"
                           "解耦。" + _ADVICE_NO_AUTO),
            })
            break

    # ---- 分支三:近零参数可见但像素当量可忽略 → 不治(结论 1 反面教材);
    # 若只有部分参数低于影响线,只对低于线的判「不治」,线上的如实标不可忽略 ----
    p1_share = max((d["loads"].get("p1", 0.0) for d in directions), default=0.0)
    p2_share = max((d["loads"].get("p2", 0.0) for d in directions), default=0.0)
    px_p1, px_p2 = px_equivs.get("p1"), px_equivs.get("p2")
    px_known = [px for px in (px_p1, px_p2) if px is not None]
    visible = p1_share >= _NEARZERO_VISIBLE_SHARE or p2_share >= _NEARZERO_VISIBLE_SHARE
    if visible and px_known and min(px_known) < _PX_NEGLIGIBLE:
        parts = []
        below = []
        above = []
        for name, px in (("p1", px_p1), ("p2", px_p2)):
            if px is None:
                continue
            parts.append(f"{name} {px:.4f}px")
            if px < _PX_NEGLIGIBLE:
                below.append(name)
            else:
                above.append(name)
        if above:
            # 判据与文案必须一致:有参数在线上时,不许整体断言「均低于/不治」
            # ——只对低于线的参数判「不治」,线上的参数按普通参数正常判读。
            advice.append({
                "direction": "近零参数 p1/p2 在载荷表可见",
                "evidence": (f"σ 级像素当量 {' / '.join(parts)},"
                             f"{'、'.join(below)}低于 {_PX_NEGLIGIBLE}px 的影响线,"
                             f"{'、'.join(above)}不低于影响线(画面影响不可忽略)"),
                "advice": (f"{'、'.join(below)} 不治:画面影响可忽略;"
                           f"{'、'.join(above)} 不适用「近零不治」,按像素当量"
                           f"天平的排序正常判读。" + _ADVICE_NO_AUTO),
            })
        else:
            advice.append({
                "direction": "近零参数 p1/p2 在载荷表可见",
                "evidence": (f"σ 级像素当量 {' / '.join(parts)},均低于 "
                             f"{_PX_NEGLIGIBLE}px 的影响线"),
                "advice": ("不治:画面影响可忽略,相对百分比无意义"
                           "(findings 结论 1 反面教材)。" + _ADVICE_NO_AUTO),
            })
    return advice


def _synthesize_what_if_poses(model, object_points: np.ndarray,
                              params: np.ndarray,
                              n_poses: int, target_r: float):
    """合成"棋盘够到画面四角"的候选位姿:近距 + 斜摆 + 平面内平移的小网格枚举。

    逐候选投影全部角点算最大归一化半径 r_max(角点到画面中心的距离,以
    焦距为单位),筛 r_max ≥ target_r 的候选,按 r_max 从大到小取前
    n_poses 个(不足则全取,由调用方记录)。投影不合法(Z≤0 等)的候选
    直接跳过,不报错——网格枚举的常规过滤。
    """
    names = list(model.param_names)
    idx = {n: i for i, n in enumerate(names)}
    fx = float(params[idx["fx"]])
    fy = float(params[idx["fy"]])
    cx = float(params[idx["cx"]])
    cy = float(params[idx["cy"]])

    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    considered = 0
    for dist in (400.0, 550.0, 700.0):
        for rx_deg in (-40.0, -20.0, 0.0, 20.0, 40.0):
            for ry_deg in (-40.0, -20.0, 0.0, 20.0, 40.0):
                for txf, tyf in ((0.0, 0.0), (0.35, 0.0), (-0.35, 0.0),
                                 (0.0, 0.35), (0.0, -0.35)):
                    considered += 1
                    rvec = np.array([np.deg2rad(rx_deg), np.deg2rad(ry_deg), 0.0])
                    tvec = np.array([txf * dist, tyf * dist, dist])
                    try:
                        pts_cam, _ = transform_to_camera(object_points, rvec, tvec)
                        uv = model.project(pts_cam, params)
                    except (ValueError, AssertionError):
                        continue  # Z≤0 等不合法候选:网格枚举的常规过滤
                    r = np.sqrt(((uv[:, 0] - cx) / fx) ** 2
                                + ((uv[:, 1] - cy) / fy) ** 2)
                    r_max = float(np.max(r))
                    if r_max >= target_r:
                        candidates.append((rvec, tvec, r_max))
    candidates.sort(key=lambda c: -c[2])
    return candidates[:n_poses], considered


def _what_if_block(model, object_points: np.ndarray, params: np.ndarray,
                   f_views640: np.ndarray, cfg,
                   poses=None) -> tuple[dict, dict]:
    """Fisher 补拍预演(仅 full 档):合成位姿并入信息矩阵,算逐参数压缩比。

    F_new = F_marg_views640(σ=1 基准,读 fisher npz)+ Σ 合成单姿态 σ=1
    信息矩阵(逐姿态调 marginalized_fisher,合成姿态天然独立);压缩比 =
    sqrt(diag(inv(F_new))) / sqrt(diag(inv(F_views640)))。σ=1 基准使压缩比
    无量纲,与检测器级 σ 的口径无关(不涉 CRLB 对账铁律)。
    """
    from calib.solve.observation import ObservationSet

    n_poses = int(cfg.analyze.what_if_n_poses)
    target_r = float(cfg.analyze.what_if_target_r)
    if poses is None:
        poses, considered = _synthesize_what_if_poses(
            model, object_points, params, n_poses, target_r)
    else:
        considered = len(poses)

    names = list(model.param_names)
    f_new = np.array(f_views640, dtype=np.float64, copy=True)
    rvecs_out: list[np.ndarray] = []
    tvecs_out: list[np.ndarray] = []
    rmax_out: list[float] = []
    for k, (rvec, tvec, r_max) in enumerate(poses):
        pts_cam, _pose_block = transform_to_camera(object_points, rvec, tvec)
        uv_ideal = model.project(pts_cam, params)  # 理想角点:基准参数的投影
        obs = ObservationSet(
            image_ids=(f"what_if_{k:03d}",),
            pose_ids=(f"what_if_pose_{k:03d}",),
            object_points=object_points,
            image_points=np.asarray(uv_ideal, dtype=np.float64)[None, :, :],
            image_size=(0, 0),  # marginalized_fisher 不读画面尺寸,占位
            corner_source=("synthetic",),
        )
        f_one, _f_full, _diag = marginalized_fisher(
            model, obs, params, [rvec], [tvec])
        f_new = f_new + f_one
        rvecs_out.append(np.asarray(rvec, dtype=np.float64))
        tvecs_out.append(np.asarray(tvec, dtype=np.float64))
        rmax_out.append(float(r_max))

    se_before = np.sqrt(np.diag(np.linalg.inv(np.asarray(f_views640, dtype=np.float64))))
    se_after = np.sqrt(np.diag(np.linalg.inv(f_new)))
    ratio = se_after / se_before
    k_names = [n for n in ("k1", "k2", "k3") if n in names]
    section = {
        "synthetic": _SYNTHETIC_NOTE,
        "baseline": ("F_marg_views640(fisher npz 的 σ=1 边际化信息矩阵)"
                     "+ Σ 合成单姿态 σ=1 信息矩阵"),
        "n_poses_requested": n_poses,
        "target_r": target_r,
        "n_candidates_considered": int(considered),
        "n_poses_selected": len(rvecs_out),
        "selected_r_max": rmax_out,
        "compression_ratio": {n: float(r) for n, r in zip(names, ratio)},
        "compression_ratio_k_reference": {n: float(ratio[names.index(n)])
                                          for n in k_names},
        "compression_reference_note": (
            "压缩比 = σ=1 基准下补拍后/补拍前的逐参数理论标准差之比"
            "(无量纲,与检测器级 σ 口径无关);k1–k3 的单独压缩比只作参考列"
            "——强相关参数的单独下界被参数间补偿空间撑大,正式判读看 "
            "fx/fy/cx/cy,口径同 budget 段 crlb_caliber_note。"),
    }
    detail = {
        "synthetic_rvecs": np.asarray(rvecs_out, dtype=np.float64).reshape(-1, 3),
        "synthetic_tvecs": np.asarray(tvecs_out, dtype=np.float64).reshape(-1, 3),
        "synthetic_r_max": np.asarray(rmax_out, dtype=np.float64).reshape(-1),
    }
    return section, detail


def run_sensitivity(model_id: str, evaluate_report: dict, fisher_npz_path,
                    bootstrap_npz_path, slices_npz_path, model,
                    object_points: np.ndarray, ref_rvec: np.ndarray,
                    ref_tvec: np.ndarray, coverage_section: dict | None,
                    cfg) -> tuple[dict, dict]:
    """编排 ANA-04:双谱对账 + 像素当量 + 建议规则表 + uniqueness + what-if。

    返回 (section, detail):

    - section 进 analyze.json 的 sensitivity 段
      (schema analyze-sensitivity-v1):theory_spectrum / empirical_spectrum /
      px_equivalents / advice / uniqueness / what_if(quick 档为 None 并附
      what_if_note)/ sources;
    - detail 是明细大数组(eigvals_cov / eigvecs_cov / loads_theory /
      loads_bootstrap / px_equiv_per_direction / 合成位姿数组),交给
      _sensitivity_detail 走 npz 旁车。

    参数:
        model_id / model: 目标模型短名与实例(evaluate.json 的
                          selection.selected);
        evaluate_report:  evaluate.json 读回的完整报告(params 取 delivery,
                          uniqueness 取 bootstrap/optimizer/source_separation,
                          p1/p2 的 σ 取 slices.empirical_std);
        fisher_npz_path:  fisher npz 路径(特征谱只消费不重算);
        bootstrap_npz_path / slices_npz_path: evaluate 实验底账 npz;
        object_points / ref_rvec / ref_tvec: 像素当量换算的棋盘物点与
                          参考位姿(第 0 视图,findings 结论 1 惯例);
        coverage_section: residual_structure 段(缺席传 None,建议规则表的
                          覆盖判据退化为"覆盖未知"分支);
        cfg:              全量配置(步长倍数、what-if 档位与姿态数、病态线)。
    """
    names = [str(n) for n in model.param_names]
    delivery = _require(evaluate_report.get("delivery", {}) or {},
                        "models", "evaluate.json 的 delivery 段")
    final_params = _require(delivery, model_id,
                            f"delivery.models[{model_id}]")
    params_by_name = _require(final_params, "final_params",
                              f"delivery.models[{model_id}] 的交付参数")
    params = np.array([float(params_by_name[n]) for n in names], dtype=np.float64)
    step_sigma = float(cfg.analyze.sensitivity_step_sigma)
    evaluate_round = str(
        (evaluate_report.get("lineage") or {}).get("round_id") or "未知轮次")

    # ---- 1) 理论谱:fisher npz 只消费不重算(T-05-12)----
    spectrum = _read_spectrum_npz(
        fisher_npz_path, model_id,
        need_f640=(cfg.analyze.profile == "full"))
    if tuple(str(n) for n in spectrum["param_names"]) != tuple(names):
        raise AssertionError(
            f"fisher npz 的 param_names {list(spectrum['param_names'])} 与模型 "
            f"{model_id} 的参数布局 {names} 对不上:谱的行列次序必须与参数"
            f"布局逐位一致,串了模型或串了轮次都会触发本断言,"
            f"请重跑 calib fisher。"
        )
    eigvals_cov = np.asarray(spectrum["eigvals_cov"], dtype=np.float64)
    eigvecs_cov = np.asarray(spectrum["eigvecs_cov"], dtype=np.float64)
    scale_dimless = np.asarray(spectrum["scale_dimless"], dtype=np.float64)

    order = np.argsort(-eigvals_cov, kind="stable")
    n_top = min(_N_TOP_DIRECTIONS, len(names))
    directions: list[dict] = []
    loads_theory_rows: list[np.ndarray] = []
    px_per_direction: list[float] = []
    for rank, idx_e in enumerate(order[:n_top], start=1):
        v_dimless = eigvecs_cov[:, idx_e]
        v_raw = _raw_direction(v_dimless, scale_dimless)
        step = step_sigma * float(np.sqrt(eigvals_cov[idx_e])) * v_raw
        px = float(_px_dispersion([params, params + step], model,
                                  object_points, ref_rvec, ref_tvec))
        directions.append({
            "rank": rank,
            "eigval": float(eigvals_cov[idx_e]),
            # 载荷表取无量纲坐标分量(plan 口径:谱存在无量纲坐标下,像素
            # 参数被 ÷焦距 压平,原始坐标载荷会被焦距刻度反抢主导——
            # findings 结论 1 的「k2/k3 软方向」只在无量纲载荷下可见)
            "loads": _loads_share(v_dimless, names),
            "px_equiv": px,
        })
        loads_theory_rows.append(v_raw)
        px_per_direction.append(px)

    # ---- 2) 经验谱:bootstrap 为主证,8 组分组为旁证;方向夹角对账 ----
    boot_key = f"{model_id}__params"
    boot_samples = _read_params_matrix(
        bootstrap_npz_path, boot_key,
        "bootstrap 重采样参数")
    v_boot, boot_eigval = _principal_direction(boot_samples)
    slice_samples = _read_params_matrix(
        slices_npz_path, f"{model_id}__group_params", "分组独立标定参数")
    v_slice, slice_eigval = _principal_direction(slice_samples)
    v_theory_raw = loads_theory_rows[0] / np.linalg.norm(loads_theory_rows[0])

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(abs(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))

    empirical = {
        "bootstrap": {
            "n_samples": int(boot_samples.shape[0]),
            "max_eigval": boot_eigval,
            "loads": _loads_signed(v_boot, names),
        },
        "slices": {
            "n_groups": int(slice_samples.shape[0]),
            "max_eigval": slice_eigval,
            "loads": _loads_signed(v_slice, names),
        },
        "cosine_theory_vs_bootstrap": _cos(v_theory_raw, v_boot),
        "cosine_theory_vs_slices": _cos(v_theory_raw, v_slice),
        "caliber_note": ("对账口径:top_directions 的载荷表是无量纲坐标下的"
                         "占比(像素参数被 ÷焦距压平,软方向判读以它为准);"
                         "对账则在原始参数空间——理论方向经无量纲还原"
                         "(×scale_dimless)后,与经验协方差的最大特征方向比 "
                         "|cos| 夹角;bootstrap 为主证,8 组分组独立标定为旁证"
                         "(样本少)。两侧原始坐标载荷并排表见 loads_table。"),
        "loads_table": [
            {"param": names[i],
             "theory_raw": float(v_theory_raw[i]),
             "bootstrap": float(v_boot[i] / np.linalg.norm(v_boot)),
             "slices": float(v_slice[i] / np.linalg.norm(v_slice))}
            for i in range(len(names))
        ],
    }

    # ---- 3) 近零参数的像素当量(分支 3 的判据;σ 取 slices 经验 σ)----
    slices_models = _require(evaluate_report.get("slices", {}) or {},
                             "models", "evaluate.json 的 slices 段")
    slices_entry = _require(slices_models, model_id,
                            f"slices.models[{model_id}]")
    empirical_std = slices_entry.get("empirical_std") or {}
    px_p1 = (_param_px_equiv("p1", float(empirical_std["p1"]), params, model,
                             object_points, ref_rvec, ref_tvec)
             if "p1" in names and "p1" in empirical_std else None)
    px_p2 = (_param_px_equiv("p2", float(empirical_std["p2"]), params, model,
                             object_points, ref_rvec, ref_tvec)
             if "p2" in names and "p2" in empirical_std else None)

    # ---- 4) 建议规则表(只出文案,不执行,裁定③)----
    advice = build_advice(
        directions, {"p1": px_p1, "p2": px_p2}, coverage_section, cfg)

    # ---- 5) uniqueness:解不唯一证据如实透传(findings 结论 3)----
    bootstrap_models = _require(
        evaluate_report.get("bootstrap", {}) or {}, "models",
        "evaluate.json 的 bootstrap 段")
    boot_entry = _require(bootstrap_models, model_id,
                          f"bootstrap.models[{model_id}]")
    bootstrap_max_disp = float(_require(
        boot_entry, "max_disp_px",
        f"bootstrap.models[{model_id}] 的最大位移当量"))
    multistart = _require(evaluate_report.get("optimizer", {}) or {},
                          "multistart", "evaluate.json 的 optimizer 段")
    ms_models = _require(multistart, "models",
                         "optimizer.multistart.models")
    ms_entry = _require(ms_models, model_id,
                        f"optimizer.multistart.models[{model_id}]")
    multistart_disp = float(_require(ms_entry, "param_disp_px",
                                     "多起点参数位移当量"))
    multistart_verdict = str(_require(ms_entry, "verdict", "多起点判定"))
    source_separation = (evaluate_report.get("source_separation") or {})
    pose_share_max = None
    ss_entry = source_separation.get(model_id)
    if isinstance(ss_entry, dict):
        shares = [float(v.get("pose_share"))
                  for v in ss_entry.values()
                  if isinstance(v, dict) and v.get("pose_share") is not None]
        pose_share_max = max(shares) if shares else None
    uniqueness = {
        "bootstrap_max_disp_px": bootstrap_max_disp,
        "multistart_disp_px": multistart_disp,
        "multistart_verdict": multistart_verdict,
        "pose_share_max": pose_share_max,
        "note": _UNIQUE_NOTE,
    }

    # ---- 6) what-if 补拍预演(仅 full 档,裁定②;quick 档干脆不做)----
    synthetic_detail: dict = {
        "synthetic_rvecs": np.zeros((0, 3), dtype=np.float64),
        "synthetic_tvecs": np.zeros((0, 3), dtype=np.float64),
        "synthetic_r_max": np.zeros((0,), dtype=np.float64),
    }
    what_if: dict | None = None
    what_if_note = _QUICK_WHAT_IF_NOTE
    if cfg.analyze.profile == "full":
        what_if, synthetic_detail = _what_if_block(
            model, object_points, params,
            np.asarray(spectrum["F_marg_views640"], dtype=np.float64), cfg)
        what_if_note = None

    section = {
        "schema": SCHEMA_SENSITIVITY,
        "model_id": model_id,
        "step_sigma": step_sigma,
        "theory_spectrum": {
            "param_names": names,
            "eigvals": [float(v) for v in eigvals_cov],
            "top_directions": [
                {"rank": d["rank"], "eigval": d["eigval"],
                 "loads": d["loads"], "px_equiv": d["px_equiv"]}
                for d in directions
            ],
        },
        "empirical_spectrum": empirical,
        "px_equivalents": {
            "per_direction": [
                {"rank": d["rank"], "step_sigma": step_sigma,
                 "px_equiv": d["px_equiv"]} for d in directions
            ],
            "p1_px_equiv": px_p1,
            "p2_px_equiv": px_p2,
            "ref_pose": "第 0 视图(调用方传入的参考位姿,"
                        "findings 结论 1 惯例)",
            "note": ("一切比较用像素当量公共货币:沿方向走 step_sigma×σ 的"
                     "参数步长,在同一参考位姿投影全部角点取最大位移;相对"
                     "百分比对近零参数无意义(findings 结论 1 反面教材)。"),
        },
        "advice": advice,
        "uniqueness": uniqueness,
        "what_if": what_if,
        "what_if_note": what_if_note,
        "sources": {
            "evaluate_round": evaluate_round,
            "fisher_round": Path(fisher_npz_path).parent.name,
            "bootstrap": f"evaluate experiments/bootstrap.npz 的 {boot_key}",
            "slices": (f"evaluate experiments/slices.npz 的 "
                       f"{model_id}__group_params(旁证)"),
            "px_currency": ("evaluation.experiments._px_dispersion"
                            "(与 budget 段同一公共货币,不重写换算)"),
            "fim": ("谱只消费 fisher npz 不重算;重算仅发生在 full 档 "
                    "what-if 的合成姿态,只调 noise.fisher.marginalized_fisher"),
            "findings": ("docs/findings.md 结论 1(像素当量纪律)/"
                         "结论 3(解不唯一)/结论 5(手段清单与判据表)"),
        },
    }
    detail = {
        "eigvals_cov": eigvals_cov,
        "eigvecs_cov": eigvecs_cov,
        "loads_theory": np.asarray(loads_theory_rows, dtype=np.float64),
        "loads_bootstrap": v_boot,
        "px_equiv_per_direction": np.asarray(px_per_direction, dtype=np.float64),
        **synthetic_detail,
    }
    return section, detail
