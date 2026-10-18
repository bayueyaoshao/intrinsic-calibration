"""Fisher 信息矩阵与 CRLB 理论精度界(σ=1 基准)——本模块只做矩阵代数。

红线(D-10):投影与导数全部来自 models/ 与 solve/residuals.py,本文件
没有任何第二份数学——雅可比 J 只从 make_jacobian_fn 来,参数装箱只走
pack_theta;数值差分基准(测试专用那套差分工具)只准出现在 tests/,
绝不进本模块。

术语白话(首现就地解释):
- 雅可比 J:残差对参数的偏导数矩阵——"参数动一点,每个角点的像素坐标
  跟着动多少"。形状 (2·视图数·角点数, 内参个数 + 6·视图数),每行只有
  全部内参列和本视图的 6 个外参列非零(天然稀疏)。
- Fisher 信息矩阵:数据里"关于参数的信息量"的矩阵,即 JᵀJ/σ²(σ 是
  角点噪声的像素标准差)。信息越多参数定得越准,精度下界越好。
- CRLB(Cramér–Rao 下界):任何无偏估计方法的方差都不会低于它——白话
  就是"这批数据天生最准能到多少"。逐参数 CRLB 标准差 =
  σ·sqrt(边际化协方差矩阵的对角元)。
- Schur 消元(边际化):把不关心的外参从误差计算里合并掉,只留内参的
  精度界。公式 F_marg = F_θθ − F_θβ·F_ββ⁻¹·F_βθ;分块求逆代数恒等式
  保证 inv(F_marg) = inv(JᵀJ) 的内参块(测试逐位锁定)。
- 条件数:矩阵"最软方向与最硬方向的比例",越大说明有些参数组合越难
  分辨。某视图外参 6×6 信息块条件数接近奇异 = 该姿态在这批角点下
  不可观测——这是要上报的退化发现,绝不静默绕过。

σ=1 基准:J 不随 σ 变,F = JᵀJ 随 1/σ² 缩放,所以一切 CRLB 在 σ=1 上
算一次,任意噪声口径(检测器口径/时域口径)的 CRLB 都是一次乘法
(crlb_se);Σ 敏感性带是纯乘法(sensitivity_band),双口径 × 敏感性带
全部免费。

规模红线:真实规模(640 视图)下 (P+6V)² 的 JᵀJ 约 118 MB,整体稠密化
被禁止——本模块全程保持稀疏,只有 P×P 与 P×6V 的小块落成稠密数组;
批量 6×6 求逆走一次 LAPACK 批调用,不做逐视图 Python 循环求逆。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from calib.io.config import BoardSpec
from calib.models.base import CameraModel
from calib.solve.baseline import build_object_points
from calib.solve.observation import ObservationSet
from calib.solve.residuals import make_jacobian_fn, pack_theta

# 位姿 6×6 信息块的奇异警戒线:条件数达到它,说明该视图的位姿在这批
# 角点下基本定不住(常来自棋盘近似共线的极端视角),必须报错点名
_POSE_COND_SINGULAR = 1e12

# Σ 敏感性带的相对半宽:8 样本标准差的相对标准误 1/√(2×(8−1)) = 1/√14
# ≈ 0.267,即 ±26.7%(03-RESEARCH §Pattern 3 第 4 点)。σ 由同姿态 8 张
# 连拍估出时,标准差本身带这么大的相对不确定度——报告因此输出
# "下界 ± 带"而不输出裸下界
SIGMA_BAND_REL_8: float = 1.0 / np.sqrt(14.0)


def _check_sigma(sigma) -> float:
    """σ 入口校验:必须是正的有限数(角点噪声的像素标准差),否则中文报错。"""
    sig = float(sigma)
    if not np.isfinite(sig) or sig <= 0.0:
        raise AssertionError(
            f"σ(角点噪声的像素标准差)必须是正的有限数,实际是 {sigma!r}。"
            f"CRLB 与加权 FIM 都按 σ 的乘除折算,0、负数或非有限值没有物理意义。"
        )
    return sig


def _fisher_marginal_from_jacobian(J: csr_matrix, n_int: int,
                                   n_views: int) -> tuple[np.ndarray, csr_matrix, float]:
    """共用分块路线:J(稀疏)→ (F_marg (P,P), F_ββ 稀疏块对角, 最大位姿块条件数)。

    分块只用列切片(J[:, :n_int] 与 J[:, n_int:]),绝不重排行列——行序
    契约由 residuals.row_index 唯一承载,这里自创下标重排一次,就是
    "不报错只算错"的结构性事故(Pitfall 3 的结构性防线)。

    F_ββ 保持稀疏(结构保证块对角:本视图残差行只碰本视图外参列);
    奇异防护在这里做:任何视图 6×6 块条件数 ≥ 1e12 → 中文报错点名该
    视图序号,绝不静默地用伪逆求个"差不多"的解糊过去;幸存块用
    np.linalg.inv 一次批量求逆(LAPACK 批调用,禁循环逐个 inv)。
    """
    J_int = J[:, :n_int]    # 内参列:前 n_int 列
    J_pose = J[:, n_int:]   # 外参列:之后按视图每 6 列一组
    F_tt = (J_int.T @ J_int).toarray()   # (P, P)
    F_tb = (J_int.T @ J_pose).toarray()  # (P, 6V),小矩阵稠密无妨
    F_bb = J_pose.T @ J_pose             # 稀疏,块对角(结构保证)

    blocks = np.stack([F_bb[6 * i:6 * i + 6, 6 * i:6 * i + 6].toarray()
                       for i in range(n_views)])               # (V, 6, 6)
    conds = np.linalg.cond(blocks)
    if conds.max() >= _POSE_COND_SINGULAR:
        raise AssertionError(
            f"第 {int(conds.argmax())} 个视图的外参信息块条件数 {conds.max():.1e} "
            f"接近奇异:该姿态的位姿在这批角点下不可观测(这本身是要上报的退化发现),"
            f"请检查该视图的棋盘可见性与摆放。")
    blocks_inv = np.linalg.inv(blocks)                         # 批量 6×6 求逆

    # Schur 消元:F_marg = F_θθ − Σᵥ F_θβ[:, v 块]·F_ββ[v 块]⁻¹·F_βθ[v 块, :]
    # F_tb 按每 6 列一组还原成 (P, V, 6),einsum 一次算完全部视图的贡献
    # (下标 p/q 是输出的行/列,i/j 是单个 6×6 块内的行列——恒等式测试锁定)
    ftb3 = F_tb.reshape(n_int, n_views, 6)
    F_marg = F_tt - np.einsum("pvi,vij,qvj->pq", ftb3, blocks_inv, ftb3)
    return F_marg, F_bb, float(conds.max())


def marginalized_fisher(model: CameraModel, obs: ObservationSet, params,
                        rvecs, tvecs) -> tuple[np.ndarray, csr_matrix, dict]:
    """σ=1 基准的边际化 Fisher 信息矩阵(外参经 Schur 消元合并掉)。

    参数:
        model:  模型实例(create_model 按名取);
        obs:    观测集——行序契约的唯一承载,F 的行/列与残差严格对齐;
        params: 内参,(P,) 数组或按参数名的字典(缺名中文报错由
                pack_theta 承担,不猜位置);
        rvecs / tvecs: 各 (V,3),逐视图外参(线性化点)。

    返回:(F_marg (P,P) 稠密, F_full (P+6V, P+6V) 稀疏 JᵀJ, 诊断字典)。
    诊断字典含 max_pose_block_cond / n_views / n_params / param_names。

    用法白话:np.linalg.inv(F_marg) 的对角元开方,就是 σ=1 时每个内参
    标准差的理论下界;任意噪声口径用 crlb_se 一次乘法折算。
    """
    n_points = obs.object_points.shape[0]
    n_views = len(obs.image_ids)
    if obs.image_points.ndim != 3 or obs.image_points.shape[1:] != (n_points, 2):
        raise AssertionError(
            f"观测角点数组形状不合:期望 (视图数,{n_points},2)(逐视图逐角点的"
            f"像素坐标),实际是 {obs.image_points.shape}。"
            f"请检查观测集是否由 load_observation_set 按同一棋盘规格装出。"
        )

    theta = pack_theta(model, params, rvecs, tvecs)   # 按参数名装箱,缺名中文报错
    J = make_jacobian_fn(model, obs)(theta)           # D-10:J 只从工厂来
    n_int = len(model.param_names)

    F_marg, _F_bb, max_cond = _fisher_marginal_from_jacobian(J, n_int, n_views)
    F_full = (J.T @ J).tocsr()    # 只保持稀疏,绝不整体 toarray(规模红线)

    diagnostics = {
        "max_pose_block_cond": max_cond,
        "n_views": n_views,
        "n_params": n_int,
        "param_names": tuple(model.param_names),
    }
    return F_marg, F_full, diagnostics


def weighted_fisher(model: CameraModel, obs: ObservationSet, params,
                    rvecs, tvecs, sigma) -> np.ndarray:
    """按观测噪声 σ 加权的 Fisher 信息矩阵:J/σ 后走与 marginalized_fisher
    完全相同的分块与 Schur 路线。

    返回的 F_marg 已含 1/σ² 权重,对它按 σ=1 取 crlb_se 开方,得到的
    就是该 σ 下的逐内参 CRLB 标准差——这是"σ 线性缩放恒等式"测试的
    重算侧:与"σ=1 结果 × σ"的乘法路线互相独立、互相印证(任一噪声
    口径都只是一次乘法,不需要重装 FIM;本函数存在是为了让这条性质
    被独立路线锁死,不进正式主流程)。
    """
    sig = _check_sigma(sigma)
    n_views = len(obs.image_ids)
    theta = pack_theta(model, params, rvecs, tvecs)
    J = make_jacobian_fn(model, obs)(theta)           # D-10:J 只从工厂来
    n_int = len(model.param_names)
    F_marg, _F_bb, _cond = _fisher_marginal_from_jacobian(J * (1.0 / sig),
                                                          n_int, n_views)
    return F_marg


def crlb_se(F_marg: np.ndarray, sigma: float) -> np.ndarray:
    """逐内参 CRLB 标准差向量:σ·sqrt(diag(inv(F_marg)))。

    术语白话:CRLB(Cramér–Rao 下界)= 任何无偏估计方法的方差都不会
    低于它——"这批数据天生最准能到多少"。

    σ 只作乘法出现(F 与 F_marg 都随 1/σ² 缩放),所以一切 CRLB 在
    σ=1 基准上算一次,任意噪声口径(检测器口径/时域口径)只是乘数;
    这条线性缩放恒等式由测试逐位锁定。"逐位"能成立正因为本函数对
    σ 只做一次乘法、不做任何别的运算。
    """
    sig = _check_sigma(sigma)
    F = np.asarray(F_marg, dtype=np.float64)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise AssertionError(
            f"F_marg 形状不合:期望 (P,P) 方阵(边际化 Fisher 信息矩阵),"
            f"实际是 {F.shape}。请确认传入的是 marginalized_fisher 的第一个返回值。"
        )
    var = np.diag(np.linalg.inv(F))
    # 方差非正时的语义(数据极限,不是组装错误):F 可逆但某参数方向在数值
    # 上不携带信息(典型:12 参数有理模型在少量视图下的高阶畸变方向),协方差
    # 对角线被舍入推到 0 以下。CRLB 对无信息方向本就无界,记 inf;NaN 只会
    # 来自 F 里的 NaN(上游 bug),让它原样穿透。
    positive = var > 0
    se = np.full(var.shape, np.inf)
    se[positive] = sig * np.sqrt(var[positive])
    se[np.isnan(var)] = np.nan
    return se


def sensitivity_band(se: np.ndarray, rel: float = SIGMA_BAND_REL_8) -> dict:
    """Σ 不确定性敏感性带:每个 CRLB 标准差乘 (1±rel),纯乘法、不重算 FIM。

    默认半宽 SIGMA_BAND_REL_8(来历见常量注释):σ 由 8 张连拍估出时
    自带约 ±26.7% 的相对不确定度。返回 {"low", "high", "rel"},
    low/high 与输入同形状。
    """
    arr = np.asarray(se, dtype=np.float64)
    r = float(rel)
    if not np.isfinite(r) or not (0.0 < r < 1.0):
        raise AssertionError(
            f"敏感性带半宽 rel 必须是 0 到 1 之间的有限数(相对比例),"
            f"实际是 {rel!r}。"
        )
    return {"low": arr * (1.0 - r), "high": arr * (1.0 + r), "rel": r}


def spectrum_report(F_marg: np.ndarray, param_names: tuple[str, ...],
                    cond_warning: float = 1e4,
                    cond_degenerate: float = 1e6) -> dict:
    """边际化协方差的特征谱报告:特征值、条件数、"软方向"载荷表、退化旗标。

    术语白话:特征分解 = 把矩阵拆成"一个个互不相关的方向 + 每个方向上
    的方差大小"。协方差矩阵(即 inv(F_marg))的最大特征值方向,是参数
    空间的"软方向"——这个参数组合方向上精度最差,典型如 fx–cy 类摆位
    退化。措辞红线(研究 Pattern 3 第 5 点):报告里说"该组合方向上
    精度最差",不要写成"强耦合"——耦合是原因,软方向才是可测量的后果。

    参数:cond_warning / cond_degenerate 是条件数分档阈值(默认值放函数
    参数,03-05 的 FisherConfig 会按 config 覆盖)。

    返回:{"eigvals" 协方差特征值(升序),
          "condition_number" = λmax(F_marg)/λmin(F_marg),
          "top_directions" 软方向的逐项说明(按 |载荷| 降序取前 2–3 个
            参数名与载荷数值),
          "flag" "" / "warning"(cond ≥ cond_warning)/ "degenerate"
            (cond ≥ cond_degenerate)}。
    """
    F = np.asarray(F_marg, dtype=np.float64)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise AssertionError(
            f"F_marg 形状不合:期望 (P,P) 方阵,实际是 {F.shape}。"
            f"请确认传入的是 marginalized_fisher 的第一个返回值。"
        )
    if len(param_names) != F.shape[0]:
        raise AssertionError(
            f"param_names 个数({len(param_names)})与 F_marg 维度({F.shape[0]})不一致,"
            f"载荷表没法按参数名标注。请传入与 F_marg 同长、同序的参数名元组"
            f"(即 model.param_names)。"
        )

    cov = np.linalg.inv(F_marg)          # 边际化协方差(σ=1 基准)
    eigvals, eigvecs = np.linalg.eigh(cov)   # 特征值升序(白话见 docstring 开头)
    f_eigvals = np.linalg.eigvalsh(F)
    condition_number = float(f_eigvals[-1] / f_eigvals[0])

    # 软方向:协方差最大特征值对应的特征向量,按 |载荷| 降序取前 2–3 个参数
    top_vec = eigvecs[:, -1]
    order = np.argsort(-np.abs(top_vec))
    k = min(3, len(order))
    top_directions = [{
        "params": [param_names[int(i)] for i in order[:k]],
        "loadings": [float(top_vec[int(i)]) for i in order[:k]],
        "eigval": float(eigvals[-1]),
    }]

    if condition_number >= cond_degenerate:
        flag = "degenerate"
    elif condition_number >= cond_warning:
        flag = "warning"
    else:
        flag = ""

    return {
        "eigvals": eigvals,
        "condition_number": condition_number,
        "top_directions": top_directions,
        "flag": flag,
    }


# ==== 80 姿态聚合观测集与聚合位姿(03-05,D-01/D-02)====
#
# 为什么要有"80 姿态对照版"(D-01):正式 CRLB 用 640 全帧,与实际标定
# 同口径;同姿态 8 连拍的角点是相关的(同一摆位连拍),把它们当独立观测
# 会高估信息量。80 姿态版把每姿态 8 帧聚成 1 个"均值姿态"再算一遍 CRLB,
# 两版差值 = "连拍相关性吃掉多少信息"——这是有价值的发现,不是缺陷。

def aggregate_observations(corners_npz_path: Path | str, records: list[dict],
                           rvecs_all: np.ndarray, tvecs_all: np.ndarray,
                           image_ids_all, image_size=None) -> dict:
    """同姿态连拍聚合:逐点均值角点 + 旋转均值/平移均值的聚合位姿(D-02)。

    参数:
        corners_npz_path: sb 路线角点文件(image_ids/uv/detected/source);
        records:          数据清单里该切分的逐张记录(image_id/pose_id),
                          顺序必须与角点文件逐行一致(行序契约,错位即报错);
        rvecs_all/tvecs_all: 标定解的逐帧外参 (V,3);
        image_ids_all:    与外参逐行对齐的照片编号(解的 image_ids);
        image_size:       (宽, 高),原样带进返回值供观测集构造用。

    返回 {"poses": {姿态编号: 聚合条目}, "image_size": (宽, 高)},聚合条目含:
        uv_agg        该姿态检出帧的角点逐点均值 (P,2)——均值只吃检出帧;
        detected_agg  该姿态是否全部帧检出(有一帧没检出就记 False);
        rot_mean      聚合旋转 (3,) rodrigues,见下方红线;
        t_mean        聚合平移 (3,) 算术均值;
        n_repeats     进均值的帧数(检出帧数);
        sigma_scale   1/√n_repeats——σ 折减系数。**iid 假设**:连拍各帧
                      噪声互不相关时,均值的噪声才按 √n 缩小;连拍若相关,
                      折减不成立。这正是 640 全帧版与 80 聚合版的 CRLB 差值
                      要量化的"相关性损耗"(D-01 设计意图,不是缺陷);
        sources       进均值的帧的角点来源列表(逐姿态留痕,可审计)。

    红线(研究 Pitfall 6):聚合旋转用 scipy 的 Rotation.mean()(弦长 L2
    均值)——**rodrigues 旋转向量逐分量平均不是旋转平均**:rodrigues 向量
    不是线性空间,转角稍大时逐分量平均会得到错误的转轴与转角,错得离谱。
    绝不这么干;红线测试(与 scipy 对照 + 与逐分量平均显著不同)锁死。
    """
    with np.load(Path(corners_npz_path), allow_pickle=False) as data:
        npz_ids = [str(x) for x in np.asarray(data["image_ids"]).ravel()]
        uv = np.asarray(data["uv"], dtype=np.float64)
        detected = np.asarray(data["detected"], dtype=bool)
        source = [str(x) for x in np.asarray(data["source"]).ravel()]

    # 行序锁:清单记录与角点文件必须逐行一致(研究 Pitfall 10 的同款防线)
    rec_ids = [str(r["image_id"]) for r in records]
    if rec_ids != npz_ids:
        raise AssertionError(
            f"数据清单记录与角点文件的照片编号对不上:清单 {len(rec_ids)} 行、"
            f"角点文件 {len(npz_ids)} 行,或顺序不一致。聚合要求两者逐行一致"
            f"(行序契约),请确认 records 与角点文件来自同一轮 calib corners/ingest。")

    # 解侧外参按照片编号查表(检不出的帧没有解,进不了位姿均值)
    ext_map = {str(iid): k for k, iid in enumerate(image_ids_all)}

    # 按姿态分组(同姿态连拍;保持清单里首次出现的顺序,读产物的人好对照)
    groups: dict[str, list[int]] = {}
    for k, rec in enumerate(records):
        groups.setdefault(str(rec["pose_id"]), []).append(k)

    poses: dict[str, dict] = {}
    for pose_id, idxs in groups.items():
        det = [k for k in idxs if detected[k]]
        if not det:
            raise AssertionError(
                f"姿态 {pose_id} 的连拍没有任何检出帧,均值聚合无从谈起"
                f"(该姿态 {len(idxs)} 帧全部未检出)。请检查角点检出率。")
        uv_agg = uv[det].mean(axis=0)                    # 逐点均值(D-02)
        detected_agg = bool(detected[idxs].all())        # 全部帧检出才算整
        # 聚合位姿:只对解里有外参的帧求均值(与线性化点同源,口径一致)
        sol_rows = [ext_map[npz_ids[k]] for k in det if npz_ids[k] in ext_map]
        if not sol_rows:
            raise AssertionError(
                f"姿态 {pose_id} 的检出帧在标定解里一份外参都对不上"
                f"(image_ids_all 里没有它的任何照片)。聚合位姿必须与线性化点"
                f"同源,请确认传入的解与角点文件是同一批数据算出来的。")
        rs = np.asarray([np.asarray(rvecs_all[k], dtype=np.float64).ravel()
                         for k in sol_rows])
        ts = np.asarray([np.asarray(tvecs_all[k], dtype=np.float64).ravel()
                         for k in sol_rows])
        rot_mean = Rotation.from_rotvec(rs).mean().as_rotvec()   # 弦长均值,见红线
        t_mean = ts.mean(axis=0)                                 # 平移算术均值
        n_repeats = len(det)             # 进均值的帧数(σ 折减按它算)
        poses[pose_id] = {
            "uv_agg": uv_agg,
            "detected_agg": detected_agg,
            "rot_mean": rot_mean,
            "t_mean": t_mean,
            "n_repeats": n_repeats,
            "sigma_scale": 1.0 / np.sqrt(n_repeats),
            "sources": [source[k] for k in det],
        }
    return {"poses": poses, "image_size": image_size}


def build_aggregated_observation_set(board: BoardSpec, aggregate: dict) -> ObservationSet:
    """把聚合结果装成观测集(与 load_observation_set 同一套校验体例)。

    一姿态一视图:image_ids 记姿态编号(pose_ids 同值,阶段四按姿态切分
    的连接键自然就位);object_points 复用 build_object_points;image_points
    逐姿态堆叠 (V, P, 2);corner_source 统一记 "pose_mean"(聚合口径留痕)。

    校验(照 load_observation_set 的中文报错体例):每个姿态必须全部帧检出
    (带未检出帧的聚合不进 Fisher,缺一角点的"均值姿态"是假观测);
    角点数组形状与棋盘规格一致且全为有限值;image_size 必须在聚合结果里。
    """
    poses: dict[str, dict] = aggregate.get("poses", {})
    if not poses:
        raise AssertionError(
            "聚合结果里没有任何姿态(poses 为空),装不出观测集。"
            "请先跑 aggregate_observations 并确认输入有检出帧。")
    image_size = aggregate.get("image_size")
    if image_size is None:
        raise AssertionError(
            "聚合结果里没有 image_size(宽, 高),观测集无法确定图像尺寸。"
            "请在 aggregate_observations 调用时传 image_size,或手工补上。")
    n_points = board.cols * board.rows
    for pose_id, entry in poses.items():
        if not entry["detected_agg"]:
            raise AssertionError(
                f"姿态 {pose_id} 的连拍有未检出帧(detected_agg=False),"
                f"均值聚合观测不完整——带缺口的均值姿态是假观测,不进 Fisher。"
                f"请改用全检出的姿态,或回到 calib corners 排查该姿态的检出问题。")
        if entry["uv_agg"].shape != (n_points, 2):
            raise AssertionError(
                f"姿态 {pose_id} 的聚合角点形状不合:期望 ({n_points},2)"
                f"(棋盘 {board.cols}×{board.rows} 个内角点),"
                f"实际是 {entry['uv_agg'].shape}。多半是棋盘规格与角点文件"
                f"不是同一套配置,请核对 board.yaml。")
        if not np.isfinite(entry["uv_agg"]).all():
            raise AssertionError(
                f"姿态 {pose_id} 的聚合角点里混入了非数值(NaN/无穷),"
                f"聚合输入可疑,请检查角点文件。")
    return ObservationSet(
        image_ids=tuple(poses.keys()),
        pose_ids=tuple(poses.keys()),
        object_points=build_object_points(board),
        image_points=np.stack([entry["uv_agg"] for entry in poses.values()]),
        image_size=(int(image_size[0]), int(image_size[1])),
        corner_source=("pose_mean",) * len(poses),
    )
